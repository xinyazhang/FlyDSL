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

**Does not move.** `_kv_body` stays in the kernel file. It is the algorithm,
its 669 lines are the pipeline structure the tuning knobs shape, and the
scheduling comments in it are load-bearing. Breaking it up is a separate
question from offloading hardware detail, and answering both at once would make
any regression unattributable.

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
