# Plan 1: Closing the Feature Gap with AOTriton `attn_fwd`

Supersedes `sdpa-close-gap-plan0.md`. Written after building the unified
kernel, which resolved **D1** and changed several assumptions the first draft
rested on. All decisions are now resolved; §1 is done and measured, §2 is what
building it taught us that the first plan got wrong.

---

## 0. Where this kernel is going

**This is not a standalone kernel.** It will be wired into the AOTriton API as
a gfx1201 backend, and once functional parity is reached it **fully replaces
the Triton `attn_fwd` kernel on gfx1201**, on performance grounds. That target
fixes several things that would otherwise look like free choices:

- **The ABI is `aotriton::v3::flash::attn_fwd_params`**, not something of our
  own design. Argument names, semantics and derivations must match
  `modules/flash/csrc/attn_fwd.cc`, which is the layer that turns PyTorch's
  tensors into kernel arguments (`BLOCK_DMODEL` rounding, `PADDED_HEAD`,
  `Num_seqlens` sign convention, `Batch`, `Num_CU`). Read it as the spec.
- **The caller is `mha_fwd_aot` / `mha_varlen_fwd_aot`** in PyTorch
  (`~/mha_all_aot.hip`), at the end of a chain of pure *view* changes:

      any last-dim-contiguous tensor
        -> BSHD shape          (shim layer, shared by CUDA and ROCm)
        -> BSHD shape          (flash API, shared by CUDA and ROCm)
        -> q.permute({0,2,1,3})  (mha_fwd_aot, here)
        -> BHSD shape          (AOTriton API)

  Not one of those steps copies, so:

  - **The shape is BHSD. The memory layout is any `xxxD` permutation** — `D`
    contiguous and innermost, with `B`, `H`, `S` in any order above it. A
    natively-BHSD tensor comes out the far end as BHSD shape over BHSD memory
    (the two transposes cancel); a BSHD one as BHSD shape over BSHD memory; and
    nothing stops a caller handing us something as exotic as SHBD, where batch
    is not even the outermost axis. None of it is detectable from the shape.
  - Therefore **strides must be read, never derived from the shape.** Our
    `STRIDE_TOKEN = num_heads * head_dim` is not merely inflexible — it silently
    assumes one of the layouts and is wrong for the others.
  - **Correctness and performance have different scopes.** The kernel must be
    *correct* for any layout the strides can express, SHBD included. It only
    needs to be *fast* on **BHSD and BSHD**, which are the de-facto standards
    and the only ones the tuning tables and cooperative-load geometry target.
    An exotic layout should produce the right answer slowly, not the wrong
    answer quickly.
  - varlen arrives as `q.unsqueeze(0).transpose(1, 2)`, i.e. `(1, H, total, D)`.
- **Selection is per-functional**, via `context.lookup_optimal(gpu)` with a
  `force_backend_index` override. So partial coverage is deployable: a
  functional we do not yet support falls back to Triton rather than failing,
  which makes the P1..P6 order a shipping order and not just a build order.

---

## 1. Status: D1 is done

`flash_attn_func_gfx1201_aiw.py` (1622 L) unifies the three kernels. The
originals are untouched on disk and now serve as correctness oracles, reachable
through `variant="legacy" | "legacy_bp" | "legacy_m32"`.

**Correctness.** 124 tests pass (61 pre-existing, unchanged, now running
through aiw; 63 new in `test_flash_attn_func_gfx1201_aiw.py`).

| oracle | aiw knobs | result |
|---|---|---|
| baseline | `k_dist=0, v_dist=1, v_layout=row` | **bitwise**, head_dim 16/32/64/128 × causal |
| bp | `k_dist=1, v_dist=1, v_layout=transposed` | **bitwise**, head_dim 64/128 × causal × f16/bf16 |
| m32 | `+ q_row_tiles=2` | **bitwise**, head_dim 64 × causal |
| sharded / chunked | policy defaults | tolerance vs fp32 SDPA, full ladder 16…512 |
| V column window | `head_dim_v` / `d_offset` | tolerance, (512,128) (384,128) (256,128) (256,64) |

**Performance** (B=1 H=8 N=4096 f16, interleaved 3-rep A/B, `bench_aiw_ab.py`):

Worst case **0.984** (head_dim 32 non-causal); head_dim 16 non-causal 0.985;
**every other config in the 13-point ladder × causal is within ±0.5%**.

**Registers**: identical to legacy at every head_dim on the ladder (one config
one VGPR better). Zero new spills; head_dim 512 still spills 3, as before.

This lands inside the "minor regressions accepted" policy. The residual ~1.5%
at head_dim 16/32 is the 64/32-bit address split — which is the deliberate
upgrade, and which the legacy baseline never had.

---

## 2. What building it changed

### 2.1 The knob space has 7 axes, not 5 — and the missing one cost 9.6%

Plan 0's table was wrong. **K and V prefetch distances are independent.**

    kernel      K dist   V dist
    baseline      0        1      <-- asymmetric
    bp            1        1
    m32           1        1

The baseline kernel's "Opt4: pre-issue first V global load before loop" carries
V in registers exactly as bp does; only *K* is staged at distance 0. Folding
the two into one `K_PREFETCH_DIST` knob produced a **(K=0, V=0) schedule that
exists in none of the three originals**, and it cost 9.6% at head_dim 32
non-causal.

Worth stating plainly because the same reasoning applies to the feature work: a
knob that looks like one axis because two variants happen to move it together
may be two.

### 2.2 Bitwise parity has two blind spots

This is the most important methodology finding, and it changes §5 of plan 0.
Bitwise output equality against an oracle is a strong check, but it is blind
to:

1. **Tiling geometry.** Each Q row's arithmetic is identical however rows are
   grouped into blocks. aiw's `q_row_tiles=2` config was building BLOCK_M=512
   with 16 waves where m32 uses 256 with 8 — doubling per-wave register
   pressure on top of the knob's own cost — and passed the bitwise test.
   Caught only by reading the geometry.
2. **Prefetch distance.** Dropping a prefetch never changes the output. §2.1
   passed every correctness test.

Both are *performance* properties invisible to *correctness* tests, on a kernel
where the benchmark noise floor (~5%) is comparable to the effects. Neither
would have been caught by ISA diffing either, since ISA divergence is expected.

**Consequence for P1-P6:** every phase needs an explicit structural assertion
alongside its correctness test — BLOCK_M and wave count, prefetch distances,
barrier count per iteration. `test_block_m_is_invariant_to_q_row_tiles` is the
pattern.

### 2.3 A tuning table needs one key, and `BLOCK_DMODEL` is it

`qk_shards()` keys off `head_dim` alone. Introduce a V/O window narrower than
head_dim and it immediately produces invalid geometry: head_dim 384 prefers 3
shards, which splits a 128-column window into 42-column slices — not a multiple
of `WMMA_N`. Fixed with `resolve_shards()`, which walks down from the policy
preference to the first count satisfying both constraints.

Plan 0 read this as a warning that every table would have to become 2-D over
`(Hdim_qk, Hdim_vo)`. **That was wrong, and AOTriton's build rules say so
directly** (`modules/flash/aot/attn_fwd.py`):

```python
@ati.scalar('BLOCK_DMODEL', options=block_dmodel_values())   # constexpr axis
@ati.scalar(['Hdim_qk', 'Hdim_vo'], 'i32')                   # runtime scalars
@ati.scalar('PADDED_HEAD', options=[False, True])
@ati.derives('Hdim_qk', to='BLOCK_DMODEL', when=ati.eq('PADDED_HEAD', False))
@ati.derives('Hdim_vo', to='BLOCK_DMODEL', when=ati.eq('PADDED_HEAD', False))
```

with `block_dmodel_values()` (`aot/_common.py`) defaulting to
`16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512` — **exactly our
existing head_dim ladder**.

So there is a *single* constexpr tile width. The host picks the smallest ladder
value covering both real head dims; `Hdim_qk` / `Hdim_vo` are runtime arguments
that only control masking and zero-fill, and when `PADDED_HEAD` is False they
are both equal to `BLOCK_DMODEL`. Tuning keys on that one scalar.

Our tables are therefore **already correctly keyed** — `head_dim` is
`BLOCK_DMODEL` under a different name. No 2-D sweep, no register-budget model,
no fallback policy. See **[N3 — RESOLVED]**.

### 2.4 Grid axis order is load-bearing for causal — and AOTriton's is a trap

Found while landing the 3D grid in P1. Under causal masking a workgroup's cost
grows with its `q_tile`: tile 0 walks one KV block, tile N-1 walks N. The x axis
dispatches fastest, so the order decides whether each scheduling group gets a
uniform duration or a 1..N spread.

Measured at B=1 H=8 N=4096 f16 **causal**, `(q_tile, head, batch)` against
`(head, q_tile, batch)`:

| head_dim | ratio |
|---|---|
| 16 | 0.587 |
| 32 | 0.612 |
| 64 | 0.715 |
| 128 | 0.769 |

Non-causal is indifferent (within 1% either way), which is what identifies the
cause as scheduling rather than locality.

**AOTriton uses `dim3{S, H, B}` — q_tile fastest — for `NUM_XCDS == 1`.** That
is not a contradiction, and porting it verbatim would be a mistake: it also
derives `PERSISTENT_TYPE = 2` for *every* causal functional
(`@ati.derives('PERSISTENT_TYPE', to=2, when=ati.ne('CAUSAL_TYPE', 0))`), which
replaces the grid with a work-stealing loop and makes the axis order irrelevant.
Its grid is effectively the non-causal grid. Until persistent-dynamic lands we
need head-fastest, and the ordering should be revisited when it does.

This is the second time a piece of AOTriton has only made sense together with a
feature we have not built yet; the first was `CAUSAL_TYPE` shipping only as
`{0, 3}` because causal is expressed as a window.

### 2.5 V column slicing was dead code in production

`_use_bp()` returns True for every `head_dim >= 48`, and the interface computed
`slice_w = head_dim if use_bp else _v_slice_width(head_dim)` — so the slicing
loop never ran with more than one iteration on the default path. The baseline's
V-window machinery has been unreachable in production since bp took over.

It is now alive and tested in aiw, deliberately: P1 needs an independent
`Hdim_vo` regardless, so the capability had to be preserved rather than
retired. But it means the code path had no production coverage before this
work, and its measured cost is unknown.

### 2.6 The merge added a fourth copy of the scaffolding

Plan 0 step 5 (extract the ~440 lines of shared preamble/tail into a common
module) did **not** happen, because you asked to keep the originals. aiw
therefore carries its own copy of `dtype_to_elem_type`, `_run_compiled`, the
`_llvm_*` / `_pointer_*` glue, `_ptr_arg`, `_wrap_qkvo`, `_launch`, `_compile`
and the whole launch wrapper. There are now **four** copies, not one.

That is the correct trade for now — the oracles are worth more than the
duplication — but it is a debt with a due date. See **[DECISION N1]**.

---

## 3. Decisions

### [N1 — RESOLVED] Scaffolding moves to a common module *after* the oracles retire

Ordering as recommended, so the oracles stay byte-identical to the production
kernels they were. Additional scope: the common module is expected to be shared
with the **backward** kernels too, so design it for that rather than as a
forward-only dedup. (Speculative — the backward kernels do not exist here yet;
do not over-fit the interface to forward's needs in the meantime.)

### [N2 — RESOLVED] Oracles retire at the end of P2, with their numbers recorded

Accepted. **Record the legacy performance numbers before deleting them** —
once the files are gone the A/B in `bench_aiw_ab.py` has no comparison arm, and
the pre-unification ladder becomes unreproducible. Capture the full
head_dim × causal ladder with VGPR/spill counts into `sdpa_lore_gfx1201.md` as
a frozen reference table at the point of retirement.

### [N3 — RESOLVED] Key on `BLOCK_DMODEL`; seqlen binning later; asymmetric hdim is not a goal

Per §2.3, the tables already key correctly — `head_dim` is `BLOCK_DMODEL`.
Three consequences:

1. **One constexpr tile width, two runtime head dims.** Drop plan 0 §2.5's
   `BLOCK_DMODEL_QK` / `BLOCK_DMODEL_VO` split — it does not match the AOT
   model and buys nothing. The host picks the smallest ladder value covering
   `max(Hdim_qk, Hdim_vo)`; the kernel masks and zero-fills down to the real
   dims. `(Hdim_qk, Hdim_vo)` cannot be a tuning key in an AOT kernel because
   they are *arguments*, not build axes.
2. **Optimising asymmetric head dims is explicitly not a goal.** A
   `(7, 511)` call runs a 512-wide QK GEMM and wastes most of it. Correct, and
   accepted. This also retires the last remnant of the dynamic-VGPR idea
   (plan 0 §2.5.1), whose only motivation was asymmetric hdim.
3. **Tuning should eventually key on `(seqlen_q, seqlen_k)`.** AOTriton bins on
   exactly that — `@ati.tune.binning(Max_seqlen_q=le, Max_seqlen_k=le)`. Out of
   scope now: keep the current fixed `seqlen_q == seqlen_k` tables and revisit
   once P2 makes `seqlen_q != seqlen_k` legal. Noted so the tables are not
   mistaken for final.

### [D2 — ANSWERED by P0] Keep only the general path; AOT costs ~0.2%

**P0 is done.** Strides (`stride_0/1/2`, numeric naming) and `sm_scale` are
runtime arguments in `flash_attn_func_gfx1201_aiw.py`, with
`strides_constexpr=True` retained purely as an A/B arm.

Measured with **all four independent stride triples** (`stride_q*`, `stride_k*`,
`stride_v*`, `stride_o*`), i.e. the real AOT argument set, not a proxy:

| measurement | 1 shared triple | **4 per-tensor triples** |
|---|---|---|
| perf, median over ladder × causal | 0.998 | **0.996** |
| worst | 0.970 (hd 16 causal) | 0.967 (hd 16 causal) |
| best | 1.045 (hd 192 causal) | 1.041 (hd 192 causal) |
| VGPRs | +0 to +4 | **+0 to +4** (unchanged) |
| SGPRs | +4 | +22 |
| spills | none new | **none new** |
| output | bitwise identical | bitwise identical |

The spread is symmetric about 1.0 and several configs come out faster, so this
is the board's noise floor, not a cost. Well under the ~3% threshold this
decision was gated on, so: **ship the general path only.** The folded arm stays
in the source as a diagnostic for later phases — if addressing ever becomes
expensive we want to A/B against it — but is not a shipping configuration.

Three things worth carrying forward:

- **The kernel body needed no branch at all.** Only the binding site differs;
  FlyDSL's arithmetic accepts a Python `int` and an `fx` value interchangeably.
  That is what makes one source serve both paths, and it is the pattern every
  later runtime-value promotion should follow.
- **Per-tensor strides are SGPR-only.** Going from one triple to four cost +18
  SGPRs and *zero* additional VGPRs, because the strides are uniform scalars
  that only change which value each address multiplies. SGPRs are not the
  occupancy constraint on RDNA4; VGPRs are.
- **One shared triple was not just imprecise, it was wrong.** `mha_fwd_aot`
  passes K and V through untouched, so their layout is whatever the caller
  allocated and is independent of Q's; under MQA/GQA they carry `Num_head_k`
  and differ by construction. Doing this in P0 rather than P1 also structurally
  separates Q/O from K/V addressing, which is the prerequisite MQA/GQA needs
  anyway — `head_idx` is now a parameter of the address builder, ready to split
  into `off_h_q` / `off_h_k`.

### [D2 — background] FlyDSL does support Triton-style specialization

Confirmed in the source. `Constexpr` (`python/flydsl/expr/typing.py:450`) is a
real parameter annotation on `@flyc.jit`:

- constexpr-annotated params are **excluded from the runtime argument slots**
  (`compiler/jit_function.py:1121`), and
- their values are folded into the **JIT cache key** via
  `Constexpr.value_signature` (`jit_function.py:1321`).

So your assumption holds: one kernel source, specialised at compile time. For
our builders it is simpler still — a stride is either a Python `int` from the
builder closure or an `fx.Int32` kernel argument, and FlyDSL's arithmetic
overloads accept both, so the *body* needs no branch at all. Only the binding
site does.

The maintenance half of D2 therefore dissolves: keeping both paths is one
`const_expr` at the binding site, not two code paths. **P0 still measures the
delta**, but now only to know the price of AOT, not to decide whether we can
afford two paths.

### [D3 — RESOLVED, then SUPERSEDED] `seqinfo_q0/q1/k0/k1` and `VarlenBits`

Originally: `seq_info_q/k` — "bounds" wrongly implies interval endpoints when
the packed case holds cumulative lengths — plus `varlen_mode`.

**Superseded by `sdpa-varlen-plan.md` §1–3**, on two counts. One array per side
is not enough: PyTorch's `seqused_k` supplies *lengths* individually while
`cu_seq_k` supplies *positions* cumulatively, so B and C read different
tensors. Hence `seqinfo_?0` (length) and `seqinfo_?1` (position), named by
role. And `varlen_mode` as an enum samples the space rather than describing it
— varlen is a product of three orthogonal choices, so it becomes
`VarlenBits : u32`, one identically-decoded byte per side, `0` meaning the
conventional dense case. A third byte carries the **LSE layout** in two bits:
the offset *inputs* are derived from Q's addressing, but the arrangement in
memory is a genuine choice, because Transformer Engine requires `(T_q, H_q)`
and plan1 §0's "shape does not imply layout" applies to the rank-2 tensor
exactly as it does to the rank-4 ones.

### [D4 — RESOLVED] ALIBI out of scope

Confirmed independently by the build rules: `@ati.scalar('USE_ALIBI',
options=[False])` — alibi is not in the shipped AOT set at all. Same for
`INT8` / `INT8_KV` / `USE_P_SCALE` and `RETURN_ENCODED_SOFTMAX`.

### [LSE — RESOLVED] Rank-2, fp32, one unified offset formula

`@ati.tensor('L', '*fp32:16', rank=2)` — always rank 2, always fp32. The layout
is a function of `varlen_mode`, and `(B*H, S)` vs `(B, H, S)` is cosmetic (same
buffer, same offsets). The two real layouts are:

| mode | layout | `_lse_offset(b, h, s, H, S)` inputs |
|---|---|---|
| non-varlen | `(B*H, Max_seqlen_q)` | `b=batch_index, s=0, S=Max_seqlen_q` |
| varlen | `(H, TotalS)` | `b=0, s=cu_seqlens_q_start, S=total` |

**Write it as one code path, not an `if varlen_mode` branch.** AOTriton already
does: `_lse_offset` is `((b*H + h) * S) + s` with no branching — the mode
selects the *inputs* (`batch_index`, `cu_seqlens_q_start`, `lse_stride`), which
are computed once in the prologue. Adopt that shape; it is the same trick that
makes the rest of the varlen path branch-free.

### [Output dtype — RESOLVED] Same as Q, by construction

`@ati.type_var('T_io', dtype=MAIN_DTYPES, signature_name='Q')` and Q, K, V,
`Out`, B, A all declared as `'T_io'` — a single type variable. Out cannot
differ from Q in the shipped kernels. Our existing constraint is correct; keep
it and state it as an invariant rather than an incidental restriction.

### [D5 — RESOLVED] `F.pad` and the pad-ratio guard go away, once masking works

Both are host-side workarounds for the kernel's inability to handle a
`seq_len` that is not a multiple of `BLOCK_M`. `F.pad`
(`torch.nn.functional.pad`) allocates padded copies of Q/K/V and the guard
rejects the non-causal cases where the resulting zero K/V rows would perturb
the softmax denominator. In-kernel `Max_seqlen_q/k` masking makes both
obsolete, and keeping the guard would then be actively wrong — it would reject
calls the kernel can answer correctly.

Sequencing: delete them **in the same commit as the masking**, behind the
ragged-shape tests below, never ahead of them. The guard currently converts a
silent wrong answer into a loud failure; removing it early would reverse that.
Deleting `F.pad` also removes a three-tensor copy on every unaligned call
(~25 MB at B=1 H=8 N=4096 d=128 f16), so it is a speedup as well as a
correctness fix.

### Terminology: three things called "hdim"

Earlier drafts of this plan conflated these and drew wrong conclusions twice.
Use the specific term, never the bare word:

| term | meaning | value | who sets it |
|---|---|---|---|
| **`Hdim_qk` / `Hdim_vo`** | logical extent — the count of *real* elements | any integer 1..512, **no alignment guarantee** | caller, as `Q.size(3)` / `V.size(3)` |
| **`pitch`** | storage stride of the `D` axis | multiple of 8 elements (16 B), `>= ceil8(Hdim)` | allocator |
| **`BLOCK_DMODEL`** | compile-time tile width | from the ladder, `round_value(max(Hdim_qk, Hdim_vo))` | build axis |

`attn_fwd.cc` makes the relationship explicit:

```c++
int hdim_qk = in.Q.size(3);                                  // logical, unaligned
int16_t hdim_rounded = round_value(hdim_max, compiled_head_dims);
.BLOCK_DMODEL = hdim_rounded,
.Hdim_qk = hdim_qk, .Hdim_vo = hdim_vo,
.PADDED_HEAD = (hdim_rounded != hdim_qk || hdim_rounded != hdim_vo),
```

So **`Hdim_qk` can be 7.** PyTorch's `pad_last_dim` grows the *allocation* to a
multiple of 8, but the AOTriton API contract requires the caller to slice back
to the real extent (`[:, :, :, :hdim]`), so the kernel receives the logical
size. The padding shows up as `pitch`, not as `Hdim`.

**`pitch` is an analysis-only term and must never appear in the kernel
source.** It is already carried by the strides, and with a flexible memory
layout there is no way to say *which* stride it is. The rule is: **whichever of
`B`, `H`, `S` is innermost in memory supplies the pitch** — and under the fixed
BHSD *shape*, that can be any of the three. Examples, not an enumeration:

| memory layout | innermost of B/H/S | the "pitch" is |
|---|---|---|
| BHSD | `S` | `stride_q2` |
| BSHD | `H` | `stride_q1` |
| SHBD | `B` | `stride_q0` |

Every permutation of the leading three axes is legal (§0), so the question
"which stride is the pitch" has no answer the kernel could compute — it is a
function of the runtime stride *values*, never of the shape. The kernel has no
business asking.

Its only job is in this document — it is what licenses the claim that a whole
8-element chunk starting on a multiple of 8 cannot fault. That is a property of
the §0 contract, not a quantity to compute. Everything the kernel actually
does is expressed in terms of the two things it is given: **strides** for
addressing, **`Hdim_qk`/`Hdim_vo`** for bounds. If `pitch` ever shows up as a
variable, something has been derived that should have been passed.

### The alignment contract the kernel may assume

**Where the guarantee comes from matters:** PyTorch itself promises nothing
about the memory layout of an SDPA input. The contract below is upheld by the
**shim layer shared by the CUDA and ROCm backends** (§0), which normalises
whatever the user allocated before it reaches the flash API. Treat it as a
property of that layer, not of PyTorch — if a future caller bypasses the shim,
the contract does not hold and this kernel is not safe on its inputs.

1. **Base tensors are always 16-byte aligned.**
2. **The `D` axis is contiguous with a 16-byte-multiple stride**, i.e. `pitch`
   is a multiple of 8 elements at f16/bf16. Every other axis is free.
3. **Nothing about `Hdim` itself**, and **nothing about the content of
   `[Hdim, pitch)`** — it may have been zero-filled by `at::pad_symint`, but a
   tensor that arrived by another route need not have been. Do not rely on it;
   AOTriton does not (`composed_load(..., other=0.0)` regardless).

Consequence: a whole 8-element chunk starting at a multiple of 8 and lying
inside `pitch` is always **safe to load** — never a fault — but its tail may be
**garbage** whenever `Hdim % 8 != 0`.

`ENABLE_LDS_VEC16` (32-byte loads, currently on by default) is **not** covered
by the 16-byte guarantee and must be fixed or gated in P1.
### Out-of-range loads: two different problems, one of them free

There are **three** regions, not two, and they need three different treatments:

| region | where | risk | fix |
|---|---|---|---|
| `[Hdim, pitch)` | inside the allocation, straddles one 8-chunk | garbage values | **per-element mask, `other=0`** |
| `[pitch, BLOCK_DMODEL)` | next head's columns (BSHD interleaves H and D) | wrong values | **skip whole chunks** |
| past `seqlen` | outside the allocation | **memory fault** | **buffer bounds** (hardware) |

Accessing beyond `[-1, -1, -1, :]` is a real OOB access, so the seqlen case is
the critical one — it cannot be repaired after the fact, and it is the one
buffer addressing solves for free.

Neither `D` region is a fault hazard, and both are easy to get backwards:

- `[Hdim, pitch)` is **in-allocation but not guaranteed zero**, and because
  `Hdim` carries no alignment guarantee this region can start mid-chunk. So
  **per-element masking on the straddling chunk is genuinely required** when
  `Hdim % 8 != 0`. (An earlier draft claimed it never was, on the mistaken
  belief that the kernel receives a pre-rounded `Hdim`. It does not — see the
  terminology table.)
- `[pitch, BLOCK_DMODEL)` is a different animal: in BSHD the `H` and `D` axes
  are adjacent, so those columns belong to head `h+1` — in-bounds memory
  holding the wrong data. Only at the final head of the final token is it past
  the allocation. Hardware bounds checking does not help; skipping the chunk
  does.

Both `D` regions collapse when `PADDED_HEAD=False`, where
`Hdim_qk == Hdim_vo == BLOCK_DMODEL` by construction (`attn_fwd.cc` derives
exactly this) and no masking is emitted at all. That is the common case and the
one the perf ladder measures; `PADDED_HEAD=True` is the correctness fallback.
Only `Q` and `K` need the treatment — `V`'s `D` tail lands in output columns
the masked `O` store discards, and WMMA accumulates each output element
independently, so it cannot reach a live column.

**Buffer addressing solves the seqlen case in hardware, for free.** RDNA4 ISA
§9.4 (buffer out-of-range rules):

> Address Out-of-Range if: `offset >= ((stride==0 ? 1 : stride) * num_records)`.
> 1. Loads that go out-of-range **return zero**. Stores that are out of range
>    do not store anything.
> 3. Load/store-DWORD-x{2,3,4} perform range-check **per component**.

Per-component checking is the important clause: a 16-byte `dwordx4` (our
`v8f16`) straddling the end returns zeros for exactly the components past it
and real data for the rest. No branch, no clamp, no VALU cost, no fault — the
addressing mode does it.

FlyDSL exposes this: `buffer_ops.create_buffer_resource(...,
num_records_bytes=...)` plus `buffer_ops.buffer_load(rsrc, offset, vec_width,
...)`, whose offset is **i32 in elements**. That is our existing 64-bit-base +
32-bit-offset split, except enforced by the hardware addressing mode instead of
by hand — so adopting it replaces `_split_ptr` rather than adding to it.

**The exception: `GLOBAL_LOAD_TR_B128` has no buffer form.** ISA §11.6.2 lists
only `GLOBAL_LOAD_TR_B128` / `_B64`, and says "all fields of these instructions
are identical to GLOBAL_LOAD_B64 and _B128" — global addressing only. Our
transposed-V path (the default for head_dim >= 48) therefore cannot get
hardware bounds checking and must keep an explicit clamp. That is
correctness-safe: a masked-out KV row has `P = 0` after softmax, so whatever V
holds contributes nothing; the clamp only has to prevent the *fault*, not
produce the right value.

**Measured, and deferred past P2** -- but note the measurement below is of the
*wrong workload*, which is itself the reason to defer.

Buffer bounds catch **out-of-allocation** accesses. On the `D` axis nothing is
out of allocation (see the three-region table above), so they cannot help
irregular head dims. On the **seqlen** axis they can -- that is where a ragged
tail genuinely runs past the tensor. So the feature they serve is **irregular
seqlen**, and that path does not exist yet: today the interface pads seq_len
host-side with `F.pad` (a three-tensor copy) or rejects the call outright, so a
ragged tail never reaches the kernel. Everything measured below is therefore
the *residual clamp overhead on regular shapes*, not the case buffer loads are
for.

First, the detail on why the `D` axis is out of reach. `num_records` is a single
linear bound over the whole tensor, so it cannot express "stop at column hdim
of every row"; a per-row limit would need a per-lane descriptor and descriptors
are uniform SGPR quads. `[Hdim, pitch)` is inside the allocation and
`[pitch, BLOCK_DMODEL)` belongs to the next head, so only the final row's tail
would ever trip the bound. **PADDED_HEAD masking is unaffected either way.**

Second, `GLOBAL_LOAD_TR_B128` has no buffer form, and transposed V is the
*default* path from head_dim 48 up -- so the hot V load cannot use buffer
addressing regardless.

That leaves the seqlen clamp on Q/K/O. Its cost, measured by removing it
(`FMHA_UNSAFE_NO_KV_CLAMP=1`, valid only because the benchmark's seq_len is an
exact multiple of BLOCK_M) -- this is the *ceiling* for what buffer loads could
recover:

| head_dim | causal | with clamp | without | delta |
|---|---|---|---|---|
| 16 | 0 | 45.6 | 41.9 | **-8.1%** |
| 16 | 1 | 36.1 | 36.5 | +1.1% |
| 64 | 0 | 89.3 | 94.7 | +6.0% |
| 64 | 1 | 78.4 | 81.7 | +4.2% |
| 128 | 0 | 98.1 | 100.4 | +2.3% |
| 128 | 1 | 87.3 | 91.2 | +4.5% |
| 256 | 0 | 94.1 | 93.4 | -0.7% |
| 256 | 1 | 76.4 | 80.1 | +4.8% |
| 512 | 0 | 52.7 | 43.8 | **-16.9%** |
| 512 | 1 | 45.1 | 45.2 | +0.2% |

Removing work makes two configs *slower*, which says the clamp is not costing
VALU so much as perturbing register allocation -- head_dim 512 already spills,
and 16 is the narrowest tile. So the honest ceiling is "up to 5% on the middle
of the ladder, negative at both ends".

**`MASK_STEPS` (below) gets the same benefit structurally**, by not emitting the
guard on full blocks at all, and it costs nothing extra because P2's interval
decomposition already produces the full/masked split. Doing buffer loads first
would be paying for a partial version of it.

**Decision: defer buffer loads until P2**, and evaluate them there against the
workload they actually serve -- a ragged `seqlen` reaching the kernel with
`F.pad` removed (D5). Three things should be measured then, none of which are
measurable now:

1. The copy `F.pad` currently performs, which buffer loads plus in-kernel
   masking delete outright. At B=1 H=8 N=4096 d=128 f16 that is ~25 MB of
   copy per call on any unaligned shape -- almost certainly the largest single
   number in this whole comparison, and invisible to the table above because
   the benchmark uses aligned shapes.
2. Whether `MASK_STEPS` has already removed the clamp from the hot path, in
   which case the residual above is gone anyway.
3. Whether the tail blocks, which are the only ones left holding a guard,
   prefer hardware bounds to the clamp.

Independently of performance, buffer loads would turn the OOB argument in this
section from an invariant we maintain into one the hardware enforces. That is
worth something on its own, and does not depend on any of the above.

The original P1 shape, for reference:

- **Q, K, O, row-major V → buffer loads.** Fault-free by construction, and the
  seqlen tail needs no guard at all.
- **Transposed V → keep the clamp**, documented as fault-avoidance only.
- **`D` tail on Q and K → per-element mask on the straddling chunk, whole-chunk
  skip beyond `pitch`.** Emitted only in `PADDED_HEAD=True` builds.
- The one-time LDS-prologue zero-fill from plan 0 §2.5 **does not survive
  contact with a straddling chunk**: the cooperative K store writes whole
  8-element vectors, so it would overwrite pre-zeroed columns in that one
  chunk. Either mask register-side between the global load and the LDS store,
  or restrict the prologue trick to `Hdim % 8 == 0`. The trick is still free in
  the `PADDED_HEAD=False` case, which is where it matters for perf.

### No out-of-bounds access is *argued*, not tested

Worth being blunt about the limits of the poison tests: they show the pad does
not influence the output. They cannot show it was never read, because a kernel
may read and discard. Only an unmapped guard page immediately after the
allocation turns a stray read into a fault -- and a guard page acts at page
granularity, so a test would have to place the tensor so its last valid byte
lands exactly on a page boundary, for every shape under test. Deferred.

So the property is maintained by construction, on one invariant:

> **Every index the kernel forms is inside its axis's extent** -- `batch < B`,
> `head_q < Num_head_q`, `head_k < Num_head_k`, `row < seqlen`, `col < pitch`.
> An address is `base + b*s0 + h*s2 + row*s1 + col`, so if each index is in
> range the address is inside the tensor, whatever the layout.

Where each falls out:

| index | bounded by |
|---|---|
| `batch`, `head_q` | grid extents |
| `head_k` | `head_q // (Num_head_q/Num_head_k)`, so `< Num_head_k` |
| `row` (KV) | `kv_addr` clamps `tile_start` to `<= seq_last`, then the in-tile row so `ts + row < seqlen` |
| `row` (Q, O) | `q_in_bounds` selects row 0 for the load and gates the store |
| `col` | see below |

The column bound is the only non-obvious one. Loads are 8 wide at 8-aligned
columns, and `_col_safe` redirects any chunk with `col >= hdim` to column 0. So
the largest column actually addressed is `8*floor((hdim-1)/8) + 7`, i.e. the
chunk end is `ceil8(hdim)` -- and `ceil8(hdim) <= pitch` is exactly the
alignment contract. **That is what the 16-byte pitch guarantee buys**, and why
violating it (a tight `(B,S,H,100)` tensor) produced a real out-of-row write.

Anything that changes the load width, the redirect, or the pitch contract
invalidates this argument and needs it redone -- which is the point of writing
it down rather than relying on a test that cannot see the difference.

### `MASK_STEPS`: don't pay for the guard on blocks that don't need it

Follow AOTriton's inner-kernel pattern (`_attn_fwd_inner(...,
MASK_STEPS: tl.constexpr)`): put a **`constexpr[bool]` on the inner loop body**
and use the clamped/guarded load form *only* when it is true. The full-block
region is then emitted with plain unguarded loads, and the guarded form exists
only in the masked region.

This composes exactly with the interval decomposition (P2), which already
splits the KV range into full and masked regions for other reasons — the full
region calls the body with `MASK_STEPS=False`, the left/right masked regions
with `MASK_STEPS=True`. No new structure, just a second use of the split.

Three consequences, in descending order of importance:

1. **It resolves the `GLOBAL_LOAD_TR_B128` problem.** The TR load cannot get
   hardware bounds checking, but with this pattern it does not need it on the
   hot path: for a `seqlen_k` divisible by `BLOCK_N` the clamped variant is
   never instantiated, and for a ragged one it is confined to the handful of
   tail blocks where the extra VALU is amortised over almost nothing. The
   asymmetry between buffer-addressable and global-only loads stops mattering.
2. **It is a win for the buffer-loaded tensors too**, not just TR. Even free
   hardware bounds checking costs a wider addressing setup; skipping the guard
   entirely on full blocks is strictly better.
3. **It removes a cost we pay today unconditionally.** The current `kv_addr`
   clamps on *every* iteration at *every* head_dim, because the schedule has no
   notion of which blocks are interior.

**Dynamic VGPR across the `MASK_STEPS` boundary: tried, not usable.**

Attempted at the start of P2. Setting `amdgpu-dynamic-vgpr-block-size=32` in the
function passthrough is *accepted* -- the kernel compiles and runs -- but the
toolchain only half-wires it for an amdhsa compute kernel:

- **No descriptor bit.** The emitted `.amdhsa_kernel` block has no dynamic-VGPR
  field; LLVM writes `.dynamic_vgpr_en` into *PAL* metadata only
  (`AMDGPUAsmPrinter.cpp:1659`), which this target does not use.
- **No prologue allocation.** The only `s_alloc_vgpr` emitted is
  `s_alloc_vgpr 0` immediately before `s_endpgm` -- the epilogue dealloc. A
  wave in dynamic-VGPR mode starts with one block and must allocate up before
  touching anything else; nothing does that.
- **But the register allocator believes it.** VGPR count moved 241 -> 226,
  i.e. LLVM allocated against a budget the hardware will not be in a position
  to provide.

Per ISA 3.3.3, `S_ALLOC_VGPR` is *ignored* when the wave was not launched in
dynamic-VGPR mode, so the kernel ran correctly only because the mode was never
actually enabled. The attribute is therefore not merely inert, it is
misleading: it perturbs allocation without delivering the mechanism.

Neither FlyDSL nor MLIR's ROCDL dialect exposes an `s_alloc_vgpr` op, so using
it at all would mean raw intrinsic emission -- and that would still leave the
dispatch-mode problem, which is a runtime and firmware matter rather than a
codegen one.

Independently of reachability, the expected win was small: the full and masked
regions differ by a handful of mask temporaries, not by an occupancy step, and
the large state (`o_accs`, `q_b_packs`) is loop-carried across both. **Closed.**

<details>
<summary>Original contingency reasoning</summary>

**Contingency: dynamic VGPR allocation across the `MASK_STEPS` boundary.**
Only if register pressure turns out to be the binding constraint — otherwise
disregard.

Plan 0 §2.5.1 rejected dynamic VGPR, but on reasons that do not all carry over
to this particular boundary, so it is worth re-examining rather than treating
as settled:

- The killer there was that `S_ALLOC_VGPR` drains the pipeline **twice per KV
  tile** if placed at the GEMM1/GEMM2 boundary. Here the boundary is between
  *whole regions* — masked, full, masked — which is crossed **O(1) times per
  kernel invocation**, not once per tile. The drain amortises to nothing.
- The other killer was that the register peak is loop-carried (`o_accs`,
  `q_b_packs`) and so cannot be shrunk. That still bounds the *minimum*
  allocation, but the masked region's extra state — mask predicates, guarded
  address arithmetic, and under gSWA the window bounds and piecewise `start_n`
  — is genuinely region-local. So unlike the GEMM-boundary case, there is
  something real to release.

What does **not** change: the segment size is a chip-wide config we cannot set,
dynamic-VGPR workgroups take over a whole WGP, and hardware reserves forward
progress for only one wave per SIMD while our workgroups run 8-16 waves that
would all want the larger allocation at once. That last one is the ISA's
explicitly un-mitigated deadlock case and is the reason to treat this as a
contingency rather than a plan.

Expected size of the win: **small at P2** (a mask is ~16 compares and 16
selects at `BLOCK_N=32`, needing few temporaries beyond the full path), and
**larger at P6**, where gSWA adds window bounds and two-region index logic to
the masked path only. So if it is ever tried, P6 is the point, and only after a
measurement shows occupancy — not latency — is what is limiting us.

</details>

**Residual to handle, not a blocker.** Binding prefetch runs one tile ahead, so
the last iteration of the full-block region addresses block `fb_hi + 1` — which
is the masked region's first block when one exists, and past the end when it
does not. So `MASK_STEPS=False` cannot mean "no bounds logic at all"; either
the *prefetch address* keeps a clamp (cheap: one clamp on `tile_start`, on data
that is discarded anyway, versus today's per-row clamp) or the loop peels its
final iteration. Prefer the clamp-the-prefetch form first and measure peeling
only if it shows up.

**Coupling with D5 — the current clamps become safety-critical.** Today the
kernel cannot fault on the seqlen axis, but not because of anything it does:
`F.pad` hands it tensors already rounded up to `BLOCK_M`, so every address the
clamps admit is inside a real allocation. The moment `F.pad` is deleted the
kernel reads unpadded tensors directly, and `kv_addr`'s clamp stops being a
tidiness measure and becomes the only thing between us and a memory violation.
That is a second reason to land D5 and the masking in one commit, and a reason
to prefer buffer addressing over the clamp wherever the instruction allows it:
a hardware bound cannot be accidentally removed by a later refactor, and a
hand-written clamp can.

### Fast-math: `ninf` is not affordable once bias exists

`-inf` is not an edge case in this kernel, it is the masking mechanism -- and
with **bias** it becomes a *user-supplied* value, since a boolean attention
mask cast to float is a matrix of `-inf`. That makes the `ninf` fast-math flag
(and the function-level `unsafe-fp-math` / `no-nans-fp-math` attributes)
untenable: they license the compiler to assume no operand is infinite, so an
`-inf` flowing through a fast-math op can simply be folded away.

This is not speculative. It silently deleted the KV tail mask in P1e: the mask
was emitted and demonstrably live, `_fmul(-inf, sm_scale)` erased it, and the
tail columns went on contributing to the softmax denominator. Three wrong
theories and an LSE probe were needed to find it.

**Measured cost of giving `ninf` up**, with `denormal-fp-math-f32` held
constant (B=1 H=8 N=4096 f16):

| head_dim | causal | `fast` | `noninf` |
|---|---|---|---|
| 16 | 0 | 37.2 | 37.2 |
| 64 | 0 | 79.8 | 79.3 |
| 128 | 0 | 91.5 | 91.9 |
| 256 | 0 | 88.5 | 88.4 |
| 512 | 0 | 45.5 | 45.4 |

Within noise everywhere. **The permission bought nothing and cost a silent
miscompile**, so the default is now `noninf`: `fast` minus `ninf`, with the two
function attributes dropped. DAZ is kept in every mode -- it is about
denormals, not infinities, and it is where the actual win was all along.

`nnan` is retained: NaN can only arise here from `-inf - -inf`, which the
`m_i` floor rules out, and the API contract excludes NaN inputs. Dropping it
too costs ~0.6% and is available as `FMHA_FP_MODE=safe`.

**Consequence for P4 (bias).** The `-inf`-through-fast-math trap is now closed
at the source, but the *placement* rule still stands and applies to every mask
yet to come: an `-inf` must not pass through an arithmetic op that could fold
it. Bias is added to `qk` before the row max, which is the same position the
KV mask had to move to.

### Note arising: "causal" is not a build axis in the shipped set

`@ati.scalar('CAUSAL_TYPE', options=[0, 3])` — only *two* values ship, and
neither is the top-left (1) or bottom-right (2) variant. Causal is expressed as
a **window**: type 3 with `Window_left`/`Window_right`, including the
`0x80000001` / `0x80000002` sentinels for varlen.

This does not change the phase order you set (gSWA last), but it does mean P6
is not an optional extra — for AOT parity it is the *only* causal path, and the
constexpr `CAUSAL_TYPE` introduced in P2 is strictly an intermediate. Worth
knowing before P2's masking work is designed, so it is built to have its window
values promoted to arguments rather than reworked.

---

## 4. Phases

Unchanged in substance from plan 0, with one **ordering change**: gSWA (P6)
landed early, ahead of P3/P4/P5, at your direction. The original constraint was
dropout before `Window_left/right`, on the grounds that "SWA is incremental
over causal but still needs lots of work, and dropout is a bigger gap".

Taking it early turned out to be the cheaper order, for a reason that was not
visible when the constraint was set: P6's real content is *deleting*
`CAUSAL_TYPE` 1 and 2, not adding windows. Had it stayed last, varlen, bias
and dropout would each have been built against **two** masking paths and then
had one of them removed underneath. They now build against one. The remaining
constraints hold: persistent-dynamic last, no INT8.

Deltas only:

**P0 — price the de-constexpr-ing.** Unchanged, and now cheaper to run: one
builder, one set of knobs, and per D2 the kernel *body* needs no branch —
FlyDSL's arithmetic accepts a Python `int` or an `fx.Int32` interchangeably, so
only the binding site differs. Promote `sm_scale` and the strides
(`stride_q0/q1/q2`, per your numeric-naming instruction). Output is the price
of AOT, not a go/no-go.

**P1 status: DONE.** Numerics (`safe_softmax`), MQA/GQA on a 3D grid,
`BLOCK_DMODEL` / runtime `Hdim_qk` / `Hdim_vo` / `PADDED_HEAD`, logsumexp,
per-tensor strides, in-kernel KV masking with host padding removed (D5), and
`VEC_WIDTH` fixed at 8. Buffer loads are deferred to P2 with reasons recorded
above.

Against the three kernels aiw replaced, B=1 H=8 N=4096 f16, full ladder x
causal: **median 0.995, min 0.843, max 1.170**, ahead on every causal config
from head_dim 48 up. Ragged seqlen, which those kernels could only serve by
copying through `F.pad`: **1.40x at seqlen 4000, 2.04x at 1033**.

The two configs still below 0.9 -- head_dim 16 and 32 non-causal -- are the
`safe_softmax` and KV-mask costs logged in sections 6.1 and 6.3, both of which
P2 should recover.

*Numerics result.* Both corrections landed behind one knob so the pre-unification
oracles stay usable:

- `m_i` initialises to `-3.40282e+38`; the mask fill stays `-inf`. This is
  **preventative** — with the current causal implementation the first KV tile is
  never fully masked (tile 0 always contains `kv = 0 <= q_row`), so `-inf - -inf`
  is unreachable today.

  **It becomes verifiable at P4, with bias.** A bias tensor may hold any
  non-NaN value, `-inf` included, and callers routinely use a large negative
  bias as an attention mask. A bias that covers the whole first KV tile drives
  its row max to `-inf`, and an `-inf`-initialised `m_i` then computes
  `exp2(-inf - -inf) = NaN`. That is the first configuration in which the bug
  is reachable at all, so the regression test belongs in P4 — not P2 or P6,
  which merely widen the exposure (`seqlen_q > seqlen_k` leaving whole rows
  masked, and windows excluding entire tiles).
- The QK scale moves *before* the row max, replacing
  `exp2(fma(s, qk_scale, -qk_scale*m))`. **Demonstrated**, not just conformed
  to: with `causal=True` the first query row attends to exactly one key, so its
  softmax is 1.0 and its output is `V[0]` exactly. The corrected form gives
  `exp2(0) == 1` exactly; the old form computed the rounding error of
  `qk_scale*m` instead of zero, an error of ~1 ulp of `qk_scale*m` that **grows
  with input magnitude**. Measured max relative error against an fp64 reference,
  head_dim 128 causal:

  | input magnitude | old form | corrected |
  |---|---|---|
  | 30 | 3.0e-4 | 2.5e-4 |
  | 100 | 6.2e-4 | 1.5e-4 |
  | 300 | 4.1e-4 | **0.0** |
  | 1000 | 4.9e-4 | **0.0** |
  | 3000 | 6.6e-4 | **0.0** |

  Below magnitude ~100 the difference is under f16 output precision and
  invisible, which is why the first probe found nothing. Non-causal shows no
  difference at any magnitude — the max-element error is diluted across 256
  terms.

*Cost.* A correct form needs `2N` ops per tile where the FMA form needed `N`
(scale-then-subtract or subtract-then-scale, both `2N`; there is no `N` form
that is exact). Measured median **0.989** over the ladder, but with two real
outliers: **head_dim 32 non-causal 0.895** and **head_dim 16 non-causal 0.963**
(7-rep medians, spread 0.887-0.903 — not noise). Both are the wide-`BLOCK_N`,
softmax-bound configs the tuning table already flags. It is *not* register
pressure: at head_dim 32 the corrected form uses **fewer** VGPRs (93 vs 98) and
gets *higher* occupancy (16 vs 14 waves/SIMD). It is straightforward VALU in a
loop where softmax already dominates.

Accepted: correctness over 1%. Logged in §6 for the post-gap optimisation pass.

**P1 — layout generality, MQA/GQA, LSE, numerics.** Simplified by N3:
- **One constexpr tile width.** Rename the `head_dim` build parameter to
  `BLOCK_DMODEL` to match AOTriton, add runtime `Hdim_qk` / `Hdim_vo` and
  constexpr `PADDED_HEAD`. No two-width split, no table rework, no asymmetric
  tuning. This is a much smaller item than plan 0 scoped.
- **Strides become arguments, and shape stops implying layout.** Today
  `STRIDE_TOKEN = num_heads * head_dim` derives the layout from the shape. The
  caller hands us BHSD shape over BSHD memory (§0), so that derivation is wrong
  for the actual production call — not merely inflexible. Name them
  `stride_q0/q1/q2` numerically per your porting instruction; `stride_?3` is
  the contiguous `D` axis and stays 1.
- **Move Q/K/O and row-major V to buffer loads** (§3, "Out-of-range loads"),
  which makes the seqlen axis fault-free in hardware and replaces the
  hand-rolled 64/32 address split. Transposed V keeps its clamp — no buffer
  form of `GLOBAL_LOAD_TR_B128` exists — which the `MASK_STEPS` knob then
  keeps off the hot path.
- `D` tail: on `Q`/`K`, skip the trailing `(BLOCK_DMODEL - Hdim)/8` chunks and
  zero their registers — whole chunks only, never per-element. `V` needs
  nothing beyond masking the `O` store. `PADDED_HEAD=False` builds have
  `Hdim == BLOCK_DMODEL` and skip this entirely.
- **Fix or gate `ENABLE_LDS_VEC16`** — 32-byte loads over a 16-byte-guaranteed
  pitch is an out-of-allocation read once arbitrary `Hdim` is legal.
- MQA/GQA on a 3D grid (also a P-later prerequisite for persistent-dynamic).
- LSE as a single branch-free offset formula (see LSE above), fp32, rank 2,
  behind an `L != nullptr` gate.
- **structural assertions** per §2.2 accompany each change.

Still carries the three numerics items, of which the FMA one is a real defect
we have (`p = exp2(fma(s_raw, sm_scale_log2e, -sm_scale_log2e * m_new))` is
structurally the pattern AOTriton flags in ROCm/aotriton#54), and it lands
together with LSE because both need `m_i` kept in the scaled domain.

**P2 — interval decomposition, in-kernel ragged masking.** Now also carries the
`MASK_STEPS` constexpr (§3): the full-block region emits unguarded loads, the
masked regions emit guarded ones. That is the same full/masked split the
interval decomposition already produces, so it costs no extra structure — and
it is what keeps the un-buffer-able TR load off the hot path. It is also the
point at which the legacy oracles retire (N2 — record their numbers first).
Deletes the 16-named-scalar causal unroll, which is duplicated in all four
kernels today and is the reason `causal` requires `BLOCK_N == 32`. Design the
window values as *promotable*: per the note above, the shipped AOT set has no
CAUSAL_TYPE 1/2, so P2's constexpr causal is an intermediate on the way to P6's
runtime window, not a parallel path.

**P6 gSWA — DONE except step 4.** Planned and executed in
`sdpa-gswa-plan.md`; steps 1-3 are in. Both objectives are met: sliding windows
generalized to negative bounds on either side, and `CAUSAL_TYPE` 1 and 2
**deleted from the kernel**, which now ships `{0, 3}` exactly as AOTriton does.
The plan's grep criterion returns 0 and the diagonal exists only as
`window_right`.

| | |
|---|---|
| 256-wide window vs causal, B=1 H=8 N=4096 | **2.2-3.3x** across the ladder |
| causal routed through the window path | ladder median +0.77%, worst -0.55% |
| cost | +25 VGPRs at head_dim 128 (no spill); head_dim 192/256 spill slightly deeper |
| structure | two loop bodies, not three (identical WMMA counts) |

Two findings worth carrying forward:

- **Bitwise equivalence between the window path and the deleted causal path is
  achievable, and is the sharpest oracle in this codebase.** The masked and
  unmasked loop bodies are separately-optimised copies of the same computation
  and do *not* agree to the last bit, so a single tile routed to the wrong one
  shows up immediately -- even where the two are mathematically identical and
  every tolerance test passes. It caught a prefetch bug and settled an
  off-by-one in the interval boundaries. It holds only because the full region
  is walked *first*; ordering the masked runs first is equally correct and
  would have thrown it away.
- **It does not prove work is skipped.** A fully dead tile is a bitwise no-op
  (`corr = exp2(0) = 1.0`, `p = 0`), so the kernel could walk leading dead
  tiles and still agree to the last bit. That needs a measurement, and got one.

**Step 4 (varlen sentinels) is deferred into P3**, not outstanding here.
`Window_left`/`Window_right` must accept `0x80000001` / `0x80000002` resolved
per-sequence, which is meaningless without per-sequence lengths. The host-side
`_CAUSAL_WINDOW` table is where they land.

**P3 varlen.** Detailed separately in `sdpa-varlen-plan.md`. Varlen is not an
enum but a product of **three orthogonal choices** — is the token axis stacked,
how is length given, where does a sequence start — encoded as `VarlenBits :
u32` with one identically-decoded byte per side. That subsumes AOTriton's four
`VarlenType` values, collapses two of them (compact and strided differ only in
a pointer, not in code), and covers a case the enum cannot express at all:
PyTorch's `seqused_k`, which takes lengths from an individual array and
positions from a cumulative one. All of it still reduces to six scalars
computed once in the prologue, and the kernel's addressing and LSE offset are
already written in the shape that needs — the LSE layout turns out not to be an
independent choice at all, but Q's addressing applied to a rank-2 tensor.

**P4 bias · P5 dropout.** Unchanged in content, and each is now simpler than
plan 0 scoped it: there is one masking path to extend, not two. P4 still
carries the `m_i` floor regression test, which is only *reachable* with bias.
Both gain a varlen dimension once P3 lands — bias is indexed per sequence, and
dropout's Philox offset must be per-sequence to stay reproducible.

**Still owed before the phase set is clean:** the tuning re-sweep (§5 of the
gSWA plan -- the tables have now gone stale a fourth time, and gSWA moved
register pressure again), and the legacy oracles' retirement (N2), which P2
was supposed to carry and did not.

**Deferred:** persistent-dynamic (own task, wants P1's 3D grid), `NUM_XCDS`,
INT8, fused `RETURN_ENCODED_SOFTMAX`, `PRE_LOAD_V`, mxfp8. Dynamic VGPR
allocation is **rejected**, not deferred — see plan 0 §2.5.1; the register peak
is loop-carried, so there is nothing to shrink at a phase boundary.

---

## 5. Verification standard (revised)

Supersedes plan 0 §5. Three gates, not one:

1. **Correctness** — bitwise against a preserved oracle where the FP reduction
   order is unchanged; tolerance against fp32 SDPA where it is not
   (`QK_SHARDS > 1`).
2. **Structure** — explicit assertions on BLOCK_M, wave count, prefetch
   distances and barrier count. Added because §2.2 found two performance bugs
   that every correctness gate passed.
3. **Performance** — `bench_aiw_ab.py`: interleaved 3-rep A/B, median of
   per-rep *ratios*, full head_dim ladder × causal, reported with VGPR and
   spill counts from `21_final_isa.s`. Never two sweeps compared after the
   fact; the board drifts ~5%.

Cumulative-risk table (each phase adds to a loop already latency-bound and
spilling at head_dim 512):

| phase | adds | watch |
|---|---|---|
| P0/P1 | ~12 stride SGPRs, address `v_mad`s | VGPR at hdim 192/256 |
| P2 | dynamic trip count, piecewise `start_n` | loss of the unrolled inner loop |
| P4 | a second global stream in the hot loop | LDS budget, `s_waitcnt` bubbles |
| P5 | Philox VALU + state | VGPR at hdim >= 192; spills |

Standing baseline: `B=1 H=8 N=4096 f16`, causal and non-causal, head_dim ∈
{16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 384, 512}. Captured at every
phase boundary.

### 5.1 Shape matrix — AOTriton's `test_fast`, plus two adversarial cases

Our current tests use round numbers throughout (`seq_len` 256/384/512,
`num_heads` 2, `batch` 1-2). That is exactly the matrix that hides ragged-tail
bugs, and it is why the cooperative-load batch-count defect reached NaN before
being caught.

The full AOTriton matrix (`REGULAR_SEQLEN` × `PRIME_SEQLEN_*` × all head dims ×
…) is ~300K cases — not affordable. Use **`test_backward.py::test_fast`'s
subset** instead, which is AOTriton's own answer to the same problem:

| axis | values | note |
|---|---|---|
| `BATCH` | 3 | non-round: at 1 a batch-stride bug is invisible |
| `N_HEADS` | 5, **(10, 2)** | the tuple is GQA — 10 Q heads, 2 KV heads |
| `D_HEAD` | 8, 64, 184, **(24, 152)**, **(120, 8)** | tuples are `(Hdim_qk, Hdim_vo)` |
| `seqlen_q` | 11, 523, 2048 | prime / prime / regular |
| `seqlen_k` | 31, 337, 1063 | **different list from q** |
| `causal` | False, True | |
| `dropout_p` | 0.0 (, 0.5 from P5) | |
| `dtype` | f16, bf16 | `DTYPES` also has f32; we do not support it |
| `sm_scale` | `'l1'` | |
| `storage_flip` | True | transposes axes (1,2) of the **allocation** |

Roughly **360 forward cases** before dropout, 720 after — tractable per phase.

Four things this encodes that our current tests do not:

- **`seqlen_q` and `seqlen_k` come from different lists.** Any surviving
  `Lq == Lk` assumption — which our kernel currently enforces outright — fails
  immediately rather than on some later shape.
- **`storage_flip=True` is the layout test.** It permutes the allocation dims
  while keeping the logical shape, asserting only `x != 3 and y != 3` ("last
  dimension must be continuous"). That is precisely the `xxxD`-permutation
  contract from §0, and the thing a shape-derived `STRIDE_TOKEN` cannot
  survive.
- **Tuple `D_HEAD` values are asymmetric `Hdim_qk` / `Hdim_vo`.** `(24, 152)`
  and `(120, 8)` are the cases N3 declined to *optimise* — they still have to
  be *correct*.
- **`D_HEAD = 184`** is a multiple of 8 that is not on the ladder, so it forces
  `BLOCK_DMODEL = 192` and `PADDED_HEAD = True`; `D_HEAD = 8` forces
  `BLOCK_DMODEL = 16`.

`test_padded_head_is_independent_of_pad_contents` asserts bitwise-identical
output across pad poisons, which also fails for correct but non-deterministic
algorithms (split-K over head_dim). None are planned; if that changes, relax it
to per-poison tolerance and keep the NaN case.

#### Two extra cases

1. **`D_HEAD = 512`.** The top of the ladder, and the only point that spills
   (3 registers) today. Not in `test_fast`.

2. **`Hdim = 113` in a 128-wide allocation, padding pre-filled with NaN.**
   Allocate `D = 128`, fill the whole tensor with `NaN`, then write real data
   into `[0, 113)` and pass `Hdim_qk = Hdim_vo = 113`. Any unmasked read of
   `[113, 128)` propagates `NaN` straight to the output, so the assertion is
   simply `torch.isfinite(out).all()`.

   This is the sharpest test in the set because it covers **both** `D`-tail
   regions at once: `[113, 120)` is the straddling chunk that needs per-element
   masking, and `[120, 128)` is whole chunks past `ceil8(Hdim)`. It would have
   caught the claim — made and retracted twice in this document — that
   per-element `D` masking is never required. Add it as a standing regression,
   not just a P1 check.

   (`_common_test.py` already has the `fillnan` helper for this idiom; AOTriton
   uses it on output and backward tensors to prove unwritten regions are never
   read. Same trick, applied to input padding.)

---

## 6. Outstanding costs — for the optimisation pass after the gap closes

Every feature phase buys correctness with some performance. Rather than
stopping to reclaim each one as it appears — which would interleave badly with
the feature work and risk tuning against a moving target — they are recorded
here and revisited **once functional parity is reached**.

The discipline: a cost only belongs here if it has been *measured* (interleaved
A/B, 3+ reps, per §5) and *diagnosed* far enough to say what would be tried.
"Probably slower" is not an entry.

### 6.2 ~~logsumexp at head_dim 256~~ — RESOLVED by re-tuning

The 8% LSE regression at head_dim 256 non-causal is gone, and the config is
now 1.15x *faster* than the kernel it replaced. Kept below because the cause is
worth remembering.

Re-sweeping `(shards, q_tiles)` found the old entry was simply mistuned:

    (shards, q_tiles)   BLOCK_M  waves | non-causal  causal
    (2, 4)  <- old         64       8  |    74.1      71.8
    (1, 16) <- new        256      16  |    92.6      74.9

1.25x non-causal, 1.05x causal. Two things this teaches:

- **The old note said "16 waves rejected: reduction buffer over LDS".** True
  only *with* sharding -- that buffer exists solely when `QK_SHARDS > 1`. The
  sweep had varied waves while holding shards at 2, so unsharded 16 waves was
  never tried. A rejection reason recorded against one axis silently pruned a
  point on another.
- **The new config spills 3 registers where the old spilled none** (241 -> 256
  VGPRs) and is still 25% faster. BLOCK_M 64 -> 256 quarters the workgroup
  count and the K/V traffic with it. Spill count is not a proxy for speed.

384 and 512 were swept the same way and are already optimal (384: 61.4/60.9 at
(3,4), best alternative 54.2; 512: 52.2/44.6 at (4,2), best alternative 36.7).
Only 256 was wrong.

**Still open:** the rest of the ladder has not been re-swept since
`safe_softmax` and LSE moved the register budget. head_dim 256 is unlikely to
be the only entry whose measurement has gone stale -- see 6.1, which is the
same class of problem.

<details>
<summary>Original entry (P1d)</summary>

### 6.2b logsumexp at head_dim 256 — P1d

| config | ratio |
|---|---|
| head_dim 256, non-causal | **0.916** (was 0.987 pre-LSE) |
| every other ladder point | within 1% |

5-rep medians, spread 0.916-0.918.

**Cause.** `m_final` was dead at loop exit and is now live into the epilogue,
because LSE is `(m + log2(l)) * ln2`. VGPRs 238 -> 241. head_dim 256 is the one
config sitting on a register cliff -- `QK_SHARDS=2` and already the joint
highest VGPR count on the ladder -- so three registers land differently there
and nowhere else. Occupancy is unchanged at either count, so the mechanism is
scheduling rather than wave count.

Not the unconditional arithmetic: hoisting the `log2`, the scale and the
address computation inside the store guard changed nothing (still 0.916). The
live range is the cost, not the work.

**Cost is paid whether or not LSE is requested**, because the gate is on the
`L` pointer at runtime. Requesting it is then free (1.000-1.001 everywhere).

**To try, in order:**

1. Retune head_dim 256 -- it is the only ladder point at this cliff, and
   `_Q_TILES_BY_HEAD_DIM[256] = 4` was measured before three registers moved.
2. Make LSE a build axis instead of a runtime gate, so inference builds do not
   carry it. Doubles the functional count for this kernel, which is exactly
   what the runtime gate exists to avoid -- only worth it if 1 fails and the
   8% matters.
3. Accept. It is one config, and LSE is not optional for training.

</details>

### 6.3 ~~KV tail mask on aligned shapes~~ — RESOLVED by the P2 region split

Gone. Splitting the KV range into a full region (no masks emitted) and a tail
region (KV + causal, fused into one select) removed the per-tile mask cost and
took the ladder from median 0.995 to **1.063**. head_dim 16 non-causal, the
worst case at 0.783, is now 0.865; every causal config improved, several
dramatically (192: 1.170 -> 1.373).

Two things learned doing it:

- **The body has to be emitted twice**, and that is not free: duplicating it
  pushed head_dim 128 from 149 to 212 VGPRs and head_dim 192 from 219 to 256
  with 3 spills. Re-sweeping recovered head_dim 128 (q_tiles 16 -> 8, 84.4 ->
  92.5 non-causal, 81.2 -> 92.0 causal); 192 was already optimal and keeps a
  0.938. **Any change that alters register pressure invalidates the tuning
  tables** -- third time now (256 after LSE, 128 after this).
- **The AST rewriter will not transform `range()` inside a nested Python
  loop** -- it falls through to the builtin and fails on the `init=` kwarg. The
  two regions therefore call a extracted `_kv_body()` from two explicit loops
  rather than looping over a list of regions.

### 6.4 (retired) KV tail mask on aligned shapes — P1e (D5)

| config | ratio (was, pre-mask) |
|---|---|
| head_dim 16 non-causal | **0.783** (0.945) |
| head_dim 32 non-causal | 0.891 (0.969) |
| head_dim 512 non-causal | 0.860 (0.988) |
| median over the ladder | 0.991 |

**Cause.** The mask is `NUM_S_VALS` compare+selects per KV tile, applied
unconditionally. It is worst exactly where `BLOCK_N` is widest and the loop is
most softmax-bound: head_dim 16 uses `BLOCK_N = 128`, so `NUM_S_VALS = 64` and
the mask adds 128 VALU ops per tile.

**Things already tried and rejected:**

- *Guarding it* with "is this the tail tile" -- much worse (head_dim 192
  non-causal 78.0 against 98.3 TFLOPS). The `scf.if` region boundary blocks
  scheduling across it in a latency-bound loop, costing more than the selects
  it skips.
- *Vectorising it* onto the eight-wide accumulators, which requires vectorising
  the scale too -- helped head_dim 512 (0.860 -> 0.967) but hurt the small dims
  more (16: 0.783 -> 0.692). Net worse.

**The real fix is P2.** The interval decomposition peels the tail into its own
region, so full blocks are emitted with no mask at all -- `MASK_STEPS` proper,
where the split is structural rather than a per-tile branch. This entry should
disappear then; if it does not, revisit the two rejected options with the new
loop shape.

**Not a reason to keep host padding.** The mask is what allows a ragged seqlen
to reach the kernel at all, and doing so is worth 1.42x at seqlen 4000 and
2.08x at 1033, against a legacy path that pays for three tensor copies.

### 6.1 `safe_softmax` at small head_dim — P1a

| config | ratio |
|---|---|
| head_dim 32, non-causal | **0.895** |
| head_dim 16, non-causal | 0.963 |
| median over the full ladder | 0.989 |

7-rep medians; the head_dim 32 spread is 0.887–0.903, so this is a real effect
and not board drift.

**Cause.** An exact softmax needs `2N` VALU ops per KV tile where the (wrong)
FMA form needed `N` — both scale-then-subtract and subtract-then-scale are
`2N`, and no exact `N` form exists. Both affected configs are the wide-`BLOCK_N`
cases the tuning table already annotates as softmax-bound rather than
saturation-bound, so extra VALU lands directly on the critical path.

**Not** register pressure: at head_dim 32 the corrected form uses *fewer* VGPRs
(93 vs 98) and reaches *higher* occupancy (16 vs 14 waves/SIMD).

**To try, in order:**

1. Re-sweep `_DIST0_BLOCK_N_BY_HEAD_DIM_NONCAUSAL` for head_dim 16/32. Those
   entries (128 and 64) were measured under the cheap softmax, and the balance
   they encode has moved. This is the first concrete instance of the
   table-revision work N3 flagged, and the cheapest thing to try.
2. Check whether the scale can ride the `q_b_packs` load instead. Rejected once
   already — Q is f16, so scaling it costs a rounding of ~2^-11, which is worse
   than the FMA error it would replace — but worth re-examining if a f32 staging
   path exists that does not cost registers.
3. Accept it. These are the two smallest head dims, and correctness is not
   negotiable against 10% on one config.
