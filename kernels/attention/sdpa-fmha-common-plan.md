# Executive plan: offloading hardware detail to `fmha_common_gfx1201.py`

Supersedes the P5.5 row of `sdpa-readability-plan.md`, which named the module
`fmha_common.py`. The arch suffix is deliberate: gfx950 and gfx1250 FMHA use
different *algorithms*, not just different intrinsics, so there is no shared
module in this direction.

## 0. Scope

**Goal.** Make `flash_attn_func_gfx1201_aiw.py` read as the *algorithm*, with
the hardware detail behind named helpers in `fmha_common_gfx1201.py`.

**Non-goals, stated because each is a plausible wrong turn:**

- **Do not touch `flash_attn_utils.py`.** It is the gfx950 module. Its
  structure exists to serve MFMA/VALU co-execution scheduling, which RDNA4 has
  no equivalent of. Sharing would mean one abstraction serving two schedules
  that agree on almost nothing.
- **Do not port its shape.** It is 5110 lines and hard to read. Take the one
  idea that works -- separating roles -- and none of the machinery.
- **Do not chase line count.** A 3246-line kernel is not the problem; 669
  unbroken lines in `_kv_body` is.

## 1. What the file actually looks like

Measured, not estimated. Nested definitions of 25+ lines, and the gaps between
them:

| region                                | lines | named? |
| ------------------------------------- | ----- | ------ |
| `_kv_body` (the main loop)            |   669 | yes    |
| gSWA window + region decomposition    |  ~300 | **no** |
| `_addr_pair` + address setup          |  ~140 | partly |
| cross-shard reduction setup           |  ~145 | **no** |
| `coop_store_v_lds`                    |    87 | yes    |
| `_decode_side` (varlen)               |    77 | yes    |

The two ~300 and ~145 line unnamed stretches are the readability problem. They
are straight-line code in the kernel body with no function boundary, so there
is nothing to name them by and nothing to test in isolation.

## 2. The one lesson worth taking from `flash_attn_utils.py`

**Separate by role.** It has a Q loader, a KV gmem->lds loader, an lds->vgpr
loader, a softmax helper, a store helper. That decomposition is right and is
why the gfx950 kernel body is readable despite the module behind it not being.

**Reject the mechanism.** It expresses roles as classes over a
`DualwaveKernelContext` plus a traits object, which is why finding what any one
line does means three hops. Two symptoms worth not reproducing: the bwd kernels
import `_lse_offset` and `remap_xcd` *from the fwd module*, and the context
object accumulates every field any role might want.

**Use the idiom this file already has instead.** `_addr_pair` is a closure
factory: it takes the stride triple and returns `(tbase, toff, kv_off)` closed
over it. Call sites read `k_addr(start_k, row, col)` with no context object and
no class. It is already the house pattern here; generalise that rather than
importing an inheritance hierarchy.

    def make_kv_stager(knobs, lds, elem_dtype) -> KvStager   # NamedTuple of closures

## 3. What moves, in tiers

Tier by *how much traced state it closes over*, because that is what decides
whether the signature stays honest.

**Tier A -- closes over nothing (pure builders).** Move as-is; each is a
bitwise-identical commit, proven five times in P5.4.

- `_global_load_tr_v8`, `bf16_trunc_pack_v8`, `_pack_bf16_pair`, `_bitcast_i32`
- `_lds_load_v8`, `_lds_store_vx` (take the LDS pointer as an argument)
- `_fadd`/`_fsub`/`_fmul`/`_fmax` -- take `fm_fast` as a parameter, which is
  the one thing `flash_attn_utils` gets right about these

**Tier B -- closes over compile-time knobs only.** Move as closure factories
taking the knobs they need. No runtime values cross the boundary.

- the column-validity family: `_col_safe`, `_col_mask`, `_apply_col_mask`
  (needs `PADDED_HEAD`, `hdim`)
- the cross-shard reduction: `_red_addr`/`_red_store`/`_red_load` plus its
  ~145 lines of setup (needs the LDS layout and `QK_SHARDS`)
- the varlen prologue: `_seqinfo_at`, `_decode_side` (needs `VarlenBits`)

**Tier C -- the two unnamed regions. The actual prize.**

- **gSWA regions (~300 lines).** Pure integer algebra over i32: window
  sentinels in, six block bounds out. It closes over `BLOCK_N` and the sequence
  lengths and nothing else, and it is the single most testable thing in the
  file -- a host-side reference is a dozen lines of Python. Extract as
  `resolve_window(...)` + `decompose_regions(...)` returning a NamedTuple, and
  give it unit tests that do not need a GPU.
- **Q preload + address setup (~140 lines).** Extract `make_addr_pair` (already
  a factory) and `preload_q(...)`.

**Tier D -- inside the inner loop.** An earlier draft of this plan said
`_kv_body` does not move, on the grounds that splitting the main loop and
offloading hardware detail at once would make a regression unattributable.
That was too cautious. These are trace-time Python moves gated bitwise, and
P5.4 landed five consecutive extractions with byte-identical ISA; when the gate
holds there is nothing to attribute. It only becomes true if an extraction
*fails* the gate, and then the rule is to stop and measure, not to have never
tried.

What is shareable is the middle of the three stages -- `gemm(q, k)`, **process
the QK block**, `gemm(p, v)`. The backward pass recomputes P, so it needs the
same mask, the same bias add, the same base-2 scaling and the same dropout;
it does *not* share the GEMMs or the LDS staging, whose operand shapes differ.
That matches what AOTriton shares between its own fwd and bwd -- `parse_window`,
`calculate_intervals`, `closed_interval_isect`, the masked load/store family and
`dropout.py` -- and notably not the inner loop itself, which it splits into
`attn_fwd_inner`, `bwd_inner_dk_dv` and `bwd_inner_dq`.

So the extraction targets inside the loop are the S-block post-processing
steps, each of which the bwd kernel will want:

| block                        | shared with bwd | note |
| ---------------------------- | --------------- | ---- |
| causal / SWA mask on S       | yes             | same predicate, same sentinels |
| KV tail mask                 | yes             | |
| bias add                     | yes             | after the scale, before the mask |
| base-2 scale + row max       | yes             | bwd recomputes P |
| dropout mask                 | yes             | already shared via `philox.py` |
| accumulator element -> KV column map | **no**  | WMMA operand layout; bwd's GEMM shapes differ |
| GEMM1 / GEMM2, LDS staging   | **no**          | different operands |

### 3.1 Rename: `_kv_body` -> `attn_fwd_inner`

Adopted, but not for the reason first offered. The argument was that K and V
are never used at the same time, so "kv" is a misnomer. That part does not
hold: both prefetch distances are 1, so `coop_load_k_global(next_kv_start)`
issues while the current tile's V is still feeding GEMM2 -- K[n+1] and V[n] are
in flight together. "KV" is also the standard name for the axis both are
indexed by, which is what the loop walks.

The rename is still right, for a different reason: **parity with AOTriton**,
which is the integration target. `attn_fwd_inner` is the name of exactly this
function there, and it sits in a family with `bwd_inner_dk_dv` and
`bwd_inner_dq` that this kernel will grow equivalents of. A reviewer diffing
the two implementations should not have to translate names. `_kv_body`
describes what the loop iterates over; `attn_fwd_inner` describes what the
function is, and the latter is the one shared with the reference.

## 4. Sequencing and gates

One helper (or one factory) per commit, as in P3.3/P5.4. The gate is the
tier-1 bitwise ISA check at BLOCK_DMODEL 128, causal and non-causal: these are
trace-time Python moves, so identical output is the expected result and any
difference means state crossed the boundary that should not have.

Order: A, then B, then C. A proves the module and the import path; B introduces
the factory pattern on small subjects; C is the payoff and by then the pattern
is established.

Two things to re-measure rather than assume, both burned this session:

- **Re-dump the baseline from the actual parent commit.** A stale reference
  inverted a conclusion twice.
- **The bitwise gate is not optional for Tier C.** The gSWA code feeds
  `_kv_body`'s loop bounds; if extraction changes an SSA order, the register
  allocator can respond out of proportion -- 160 bytes of scratch from one
  narrowed predicate, measured in P3.2.

## 5. What this is worth

The kernel file loses roughly 700 of 3246 lines and, more to the point, gains
names for two regions that currently have none. The gSWA algebra becomes
testable without a GPU, which is the largest single correctness surface in the
kernel and today has only end-to-end coverage.

It does not make `_kv_body` shorter, and should not pretend to.
