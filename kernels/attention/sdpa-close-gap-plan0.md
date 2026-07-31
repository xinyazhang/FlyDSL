# Plan 0: Closing the Feature Gap with AOTriton `attn_fwd`

Draft for review. Responds to `sdpa-feature-gap.md` and
"Simulate Causal Masks with Sliding Window Attention (SWA).pptx".

Nothing here is implemented yet. Sections marked **[DECISION]** need your
sign-off before the phase that depends on them starts.

---

## 1. Where we are today

The three gfx1201 builders take **five** runtime arguments between them:

```python
flash_attn_func_kernel(Q, K, V, O, seq_len: i32)   # grid: (batch * num_q_tiles * num_heads,)
```

Everything else in `sdpa-feature-gap.md`'s "must become kernel arguments" list is
a Python-level build-time constant, folded into the binary at trace time:

| AOTriton argument | Today in FlyDSL |
|---|---|
| `stride_q*/k*/v*/o*` | Not present. `STRIDE_TOKEN = num_heads * head_dim`, BSHD contiguous, hard-coded |
| `Sm_scale` | `sm_scale=None` → `1/sqrt(head_dim)` folded into `c_sm_scale_log2e`; interface does not even expose it |
| `Num_head_q/k` | `NUM_HEADS` constexpr; `head_idx = block_id % NUM_HEADS` folds to a shift/mask. No MQA/GQA |
| `Num_seqlens`, `cu_seqlens_*` | Absent. No varlen |
| `Max_seqlen_q/k` | Single `seq_len` arg, and `Lq == Lk` is enforced by the interface. Ragged tails are handled **host-side** by `F.pad` + a 0.5% pad-ratio `ValueError` |
| `Hdim_qk/vo`, `PADDED_HEAD` | `head_dim` constexpr, must be a multiple of 16. `head_dim_v`/`d_offset` exist but mean "compute a **column slice** of a head_dim-wide V", not "V is narrower than QK" |
| `Window_left/right`, `CAUSAL_TYPE` | Boolean `causal` only; top-left == bottom-right because `Lq == Lk` is forced |
| `B`, `stride_b*` | Absent |
| `dropout_p`, `philox_*` | Absent |
| `L` (logsumexp) | **Absent — not computed, not stored, no output tensor** |
| `PERSISTENT_TYPE`, `NUM_XCDS`, `INT8` | Absent (deferred per your doc) |

Two further structural facts that shape the plan:

**(a) There are 2.5 kernels, not one.** `flash_attn_func_gfx1201.py` (991 L),
`..._bp.py` (1267 L) and `..._m32.py` (979 L) are near-parallel bodies. The
interface routes `head_dim >= 48` to `bp`, `16/32` and V-sliced `> 256` to
baseline, and `m32` only on explicit opt-in. Every feature below would
otherwise have to be written two or three times.

**(b) The causal mask is unrolled into 16 explicitly named scalars**
(`s_v0 … s_v15`, `flash_attn_func_gfx1201.py:660-728`) because FlyDSL's `if`
rewriter needs one MLIR `Value` per conditional state variable, not a list.
That is why `causal` hard-requires `BLOCK_N == 32`. Any generic mask —
`seqlen_k` tail, `seqlen_q` tail, gSWA — hits this wall immediately.

---

## 2. The one structural idea that pays for itself

`calculate_intervals` is not just the gSWA feature. It is the mechanism that
**deletes** the masking problem from the hot loop, and it subsumes four
separate line items in your gap doc:

- irregular `Max_seqlen_k` (trailing partial block),
- irregular `Max_seqlen_q` (`mask_on_seq_q`),
- causal top-left and bottom-right,
- generalized SWA.

The payoff for us specifically: once the KV range is split into
`[lb_lo, lb_hi]` / `[fb_lo, fb_hi]` / `[rb_lo, rb_hi]`, the **full-block loop
carries no mask at all** and the masked loop **always** masks. The
`if tile_needs_mask:` branch — and with it the 16-named-scalar unroll and the
`BLOCK_N == 32` restriction — goes away. The mask becomes an unconditional
`select` on the eight-wide `Vec` accumulators, which is `NUM_S_ACCS` values
rather than `NUM_S_ACCS * 8`, and can be emitted in a loop.

So my recommendation is to build the **interval scaffolding early** (Phase 2),
with `window_left`/`window_right` computed from a constexpr `CAUSAL_TYPE`, and
then make gSWA (Phase 6) the near-trivial change of promoting those two values
to kernel arguments plus adding `parse_window`. Doing the tail-masking work
first in some ad-hoc way and then redoing it for gSWA is the outcome I want to
avoid.

The cost is that the current single `scf.for` over `[0, kv_upper)` becomes the
two-region loop from slide 18 — a dynamic trip count with a piecewise
`start_n`. Phase 2's perf gate exists to price that.

---

## 2.5 Resolved: arbitrary `Hdim_qk` / `Hdim_vo`

Settled in review; recorded here because it is a design input to P1, not an
open question.

**Contract.**

- `Hdim_qk` and `Hdim_vo` are independent, and each may be **any** integer
  `1 <= h <= 512`. `(hdim_qk, hdim_vo) = (7, 511)` and `(257, 13)` are both
  legal. Our current `head_dim % 16 == 0` assert must go.
- The **row pitch** of the `D` axis is guaranteed to be a multiple of **16
  bytes**, and each `D` row is 16-byte aligned. PyTorch enforces this before
  dispatch: a compact-but-irregular `(2, 4, 9, 7)` is padded and copied to
  `(2, 4, 9, 8)`. For f16/bf16 that means the pitch is a multiple of **8
  elements**.
- Elements in `[hdim, pitch)` exist and are addressable, but their **contents
  are undefined** — possibly `NaN`/`Inf` bit patterns.

**What this buys us.** Three separate simplifications, all from the 16-byte
pitch:

1. **No per-element D masking, ever.** Our vector unit is the 8-element
   `v8f16` (16 B), which is exactly the pad granularity. A `v8f16` load whose
   start offset is a multiple of 8 is always wholly inside the allocation.
   Masking is per-8-chunk, which is `BLOCK_DMODEL/8` predicates rather than
   `BLOCK_DMODEL`.
2. **Only `Q` and `K` need zero-fill; `V` does not.** Garbage in `Q[d]`/`K[d]`
   for `d >= hdim_qk` corrupts the `QK^T` dot product and must be zeroed
   (both sides — zeroing only one is unsafe, since `0 * NaN = NaN`). Garbage in
   `V[:, d]` for `d >= hdim_vo` lands only in output column `d`, which the
   masked `O` store discards; WMMA accumulates each output element
   independently, so it cannot contaminate a live column.
3. **`K`'s zero-fill can be hoisted out of the loop entirely.** Because we
   stage `K` through LDS, we can pre-zero LDS columns `[hdim_qk,
   BLOCK_DMODEL_QK)` **once** before the main loop and have the cooperative
   load write only `[0, hdim_qk)`. Those LDS bytes are never overwritten, so
   they stay zero for every KV tile. That turns AOTriton's per-iteration
   `PADDED_HEAD` masked load into an O(1) prologue. AOTriton cannot do this —
   it loads `K` straight to registers. `Q` is loaded once anyway, so it is
   free there too.

   This also weakens the case for building separate `PADDED_HEAD` on/off
   variants, which your gap doc was already sceptical of. Plan: measure, and
   expect a single variant to suffice for us.

**What it costs us.**

- **`ENABLE_LDS_VEC16` becomes unsafe as written.** It is on by default
  (`VEC_WIDTH = 16`, i.e. 32-byte loads) and a 16-byte-padded row does not
  cover it, so the tail rows of a tensor could read out of allocation. Either
  gate it on `pitch % 16 elements == 0` or guard the final chunk. Must be
  handled in P1, not deferred.
- **`BLOCK_DMODEL` is coarser than the pad.** WMMA wants K/N in 16-element
  units, PyTorch guarantees only 8, so `hdim_qk = 7` gives a 16-element tile
  over an 8-element allocation. The second 8-chunk of that tile must be
  *skipped*, not merely zeroed after loading.
- **Two independent tile widths.** `BLOCK_DMODEL_QK` and `BLOCK_DMODEL_VO`
  become separate build parameters. This finally disentangles the overloaded
  `head_dim_v`, which today means "column slice of a `head_dim`-wide V" rather
  than "V is narrower than QK". Both concepts are now needed, and they must be
  separate names.

**Interaction with V column slicing (was open question 8).** Benign at the
extremes. `(7, 511)` slices V into 4×128 and re-runs a 1-K-step GEMM1 per
slice — cheap. `(257, 13)` needs no slicing at all. The expensive case is
both dims large, which is the case we already handle today.

### 2.5.1 Dynamic VGPR allocation — investigated, not scheduled

Checked against RDNA4 ISA §3.3.3 / the `S_ALLOC_VGPR` instruction page and the
LLVM AMDGPU backend. **Available on gfx1201, but not usable for this.**

**It is available.** All three hard gates pass: compute shader only ✓,
wave32 only ✓, gfx12 ✓. The plumbing exists end to end — the amdhsa kernel
descriptor has the enable bit (`COMPUTE_PGM_RSRC2_GFX120.ENABLE_DYNAMIC_VGPR`,
bit 6, so this is not PAL-only), LLVM exposes
`llvm.amdgcn.s.alloc.vgpr` returning an `i1` success flag, and the mode is
turned on by the `amdgpu-dynamic-vgpr-block-size` function attribute — which
FlyDSL can already set through the same `passthrough` path we use for
`amdgpu-sched-strategy`. So "can we reach it" is a solved problem.

**Three reasons not to use it, in increasing order of severity.**

1. *Mode is global and coarse.* **Terminology warning:** ISA §3.3.3's "block"
   is a **VGPR allocation segment**, and has nothing to do with
   `BLOCK_M`/`BLOCK_N`. The ISA notes blocks "are also called segments in some
   contexts"; this document uses **segment** throughout to avoid the
   collision.

   A wave may hold at most **8 segments**. Segment size is either 16 or 32
   VGPRs, so the ceiling is 128 or 256 VGPRs per wave respectively — and the
   size is a **chip-wide config register**: "it cannot be modified per draw or
   dispatch." At 16, "waves must not access VGPRs above 127 — results are
   unpredictable." So the kernel does not choose this; the driver does.
   LLVM's `amdgpu-dynamic-vgpr-block-size` function attribute (legal values 16
   or 32) is a *declaration to the compiler* of what the chip is configured
   for — it drives occupancy math and the `> 8 segments` diagnostic
   (`AMDGPUAsmPrinter.cpp:1146`) — not a control that programs the hardware.
   Declaring 32 on a chip configured for 16 is silently undefined behaviour.

   Dynamic-VGPR workgroups also **take over a whole WGP** — no mixing with
   normal workgroups. And allocation can *fail* (`SCC=0`, software retry
   loop); hardware reserves forward progress for only **one wave per SIMD**,
   and the ISA says outright that this "does not prevent deadlock when
   multiple waves require the maximum allocation to progress." Our workgroups
   run 8-16 waves that all want the same large allocation, which is exactly
   the un-mitigated case.

2. *`S_ALLOC_VGPR` drains the pipeline — twice.* The instruction executes
   `WaitIdleExceptStoreCnt()` **both before and after** changing the
   allocation, and "no following instruction can issue until the allocation
   operation is complete." Only `STORECNT` may be outstanding. Placing one at
   the GEMM1→GEMM2 boundary therefore costs two full drains per KV tile.
   `sdpa_lore_gfx1201.md` records that the single largest win in this kernel
   was *removing* forced drains (33 of 35 LDS full drains) from a loop we
   characterised as LDS-latency-bound. This would reintroduce them by
   construction.

3. *Our register peak is loop-carried, not transient — so there is nothing to
   shrink.* This is the one that would kill the idea even if (1) and (2) were
   free. Dynamic VGPR reclaims registers that are dead across a phase
   boundary. In both asymmetric extremes the large allocation is live across
   the **entire** KV loop:
   - `(7, 511)`: `o_accs` is `hdim_vo/2` = 256 VGPRs, carried from the first
     tile to the epilogue. It cannot be released after GEMM2 — the next
     iteration accumulates into it.
   - `(257, 13)`: `q_b_packs` is `hdim_qk/4` ≈ 68 VGPRs, loaded once in the
     prologue and read by GEMM1 on every iteration. It cannot be released
     after GEMM1 either.

   The transient working set that *could* be freed at the boundary — `s_accs`,
   `p_packs`, K/V staging vectors — is small and already reused.

   Note also that the V column-slicing path already gets this effect for free:
   the interface issues one **kernel launch per 128-column slice**, so the
   register file is fully released between slices by the dispatcher.

**What to do instead.** Your underlying instinct is right — when
`Hdim_qk != Hdim_vo` there is real slack in the register budget — but the
lever is tile shape, not time-sharing the register file. Today
`default_block_m()` / `default_block_n()` key off a **single** `head_dim`.
Once QK and VO widths are independent, they should key off **both**: at
`(7, 511)` GEMM1's register cost collapses and that budget should be spent on
a wider `BLOCK_M`/`BLOCK_N`; at `(257, 13)` the reverse. That is cheap,
schedulable, needs no new hardware feature, and captures most of the same
win. Folded into P1 as a tuning item.

**Recorded as a negative result** for `sdpa_lore_gfx1201.md` when P1 lands, so
we do not re-derive it later.

---

## 3. Decisions needed before we start

### [DECISION D1] ~~Consolidate the three builders?~~ **RESOLVED: unify.**

**Decision.** Unify into a single kernel in a **new file**,
`flash_attn_func_gfx1201_aiw.py`. Do **not** overwrite or mutate the three
existing files.

*Naming.* `aiw` = "all-in-one", with `w[onder]` standing in for the `o` because
`aio` reads as "async IO" and `ai1` reads as "Artificial Intelligence One".
This rationale belongs in the module docstring so nobody later "fixes" it.

**Variant selection is `const_expr` branching**, not separate code paths:

```python
if const_expr(K_PREFETCH_DIST == 0):
    ...baseline's fused load->store...
else:
    ...bp's split load / store at distance 1...
```

The variants smash together cleanly this way because every axis in §D1.1 is a
whole-block substitution, not an inline predicate.

**Addressing:** the unified kernel uses 64-bit base + 32-bit offset
everywhere (bp's scheme). This is a deliberate upgrade for baseline-equivalent
configs, not a regression.

**ISA divergence from the existing kernels is expected and legitimate** — the
addressing change alone guarantees it. My earlier proposal to gate the refactor
on ISA equality is therefore withdrawn; see the revised §D1.4.

**Regression policy:** minor perf regressions are acceptable. The feature work
in P1-P6 will raise register demand regardless, so holding aiw to the current
kernels' exact numbers would be optimising for a baseline we are about to
leave behind.

#### D1.1 What the divergence actually is

Measured, not estimated:

| | baseline | bp | m32 |
|---|---|---|---|
| total lines | 992 | 1268 | 980 |
| preamble + builder constants | 305 | 403 | 243 |
| **kernel body** | **554** | **732** | **618** |
| launch/compile tail | 133 | 133 | 119 |
| `def`s | 43 | 52 | 41 |

**1336 of 3240 lines (41%) are scaffolding duplicated three times**, and the
launch/compile tail is near-identical (133/133/119). About 25 of the ~43
helpers — `_fadd`…`_fmax`, `_lds_load_v8`, `_lds_store_vx`, `wmma_acc`,
`reduction_peer`, `bf16_trunc_pack_v8`, all the `_llvm_*`/`_pointer_*` glue,
`_ptr_arg`/`_wrap_qkvo`/`_launch`/`_compile` — are common to all three.
Line-similarity is 64% (base↔bp), 59% (base↔m32), 54% (bp↔m32), so even the
"divergent" bodies are mostly common.

**The three files are not three designs. They are one design at three points
in a 5-axis knob space**, plus drift:

| axis | baseline | bp | m32 | knob |
|---|---|---|---|---|
| K staging distance | 0 (fused load→store) | 1 (split) | 1 | `K_PREFETCH_DIST` ∈ {0,1} |
| V LDS layout | row-major | transposed via `global_load_tr_b128` | transposed | `V_LDS_LAYOUT` |
| Q rows per wave | 16 | 16 | 32 | `Q_ROW_TILES` ∈ {1,2} |
| head-dim sharding | 1 | `bp_qk_shards()` 1-4 + cross-shard reduce | 1 | `QK_SHARDS` |
| V/O column slicing | yes (`head_dim_v`/`d_offset`) | no | no | `VO_SLICE` |
| V staging passes | 1 | `vo_chunks()` | 1 | `VO_CHUNKS` |

**Plus one axis that is pure drift, not a design choice.** bp's 64-bit-base +
32-bit-offset addressing (`_split_ptr` / `tile_base` / `tile_off`) never
propagated to baseline, which still does full-width `fx.Index` arithmetic
everywhere. Baseline is *correct* — `fx.Index` is 64-bit — just slower. This is
a measured optimization that exists in one file because that is where it
happened to be written. It is the clearest evidence that three files is already
costing us.

#### D1.2 The reframe

bp is already the most general of the three (52 helpers, and it is the only one
with a *coupled* knob policy — `bp_qk_shards` / `bp_q_tiles` / `vo_chunks` are
a validated triple backed by measured tables). So the job is not "merge three
files". It is:

> **Generalize bp's wave decomposition from 1-D to 2-D, then absorb baseline's
> two unique capabilities as knobs.**

bp already splits waves along head_dim (`QK_SHARDS`). m32 splits along Q rows
(`Q_ROW_TILES`). The union is a wave owning `Q_ROW_TILES × 16` rows and
`1/QK_SHARDS` of head_dim — one clean 2-D parameter, of which bp has half.

Note the knobs are **not independent**: `QK_SHARDS > 1` needs a cross-shard
reduction buffer that competes with V for LDS, forcing `VO_CHUNKS > 1`; and
`Q_ROW_TILES = 2` doubles `o_accs` + `q_b_packs` + `s_accs`, which collides
with `QK_SHARDS`. A merged builder therefore needs a **validity predicate**
over the knob space, not a set of free flags. That is the main genuine cost —
but bp already implements exactly this pattern, so we are extending machinery
rather than inventing it.

#### D1.3 Build order for `flash_attn_func_gfx1201_aiw.py`

Each step brings aiw to parity with one more existing variant. Because the old
files stay, "parity" is directly checkable rather than asserted.

0. **Skeleton.** New file; scaffolding (the 1336-line common set) lifted in
   once, 64/32 addressing throughout, knob dataclass + validity predicate.
   Knobs: `K_PREFETCH_DIST`, `V_LDS_LAYOUT`, `Q_ROW_TILES`, `QK_SHARDS`,
   `VO_SLICE`, `VO_CHUNKS`.
1. **bp parity** — `K_PREFETCH_DIST=1`, `V_LDS_LAYOUT=transposed`,
   `Q_ROW_TILES=1`, sharding per `bp_qk_shards`. Widest existing coverage, so
   it exercises the most machinery first.
2. **`VO_SLICE`** → aiw covers `head_dim > 256`.
3. **`K_PREFETCH_DIST=0` + `V_LDS_LAYOUT=row`** → baseline parity;
   `head_dim` 16/32 covered.
4. **`Q_ROW_TILES=2`** → m32 parity. Last because it is the invasive one: it
   changes the wave→data mapping, so every index expression moves.
5. **Interface switchover.** `flash_attn_func_gfx1201_interface.py` routes to
   aiw; the three old builders stay importable as oracles and reference.

Degrades gracefully: stopping after step 3 still yields a single kernel
covering the whole default dispatch path, with m32 left as an opt-in variant.
Rough cost: steps 0-1 ≈ 3 days, 2-3 ≈ 2 days, 4 ≈ 2 days, 5 ≈ 1 day.

#### D1.4 Verification (revised — ISA gate withdrawn)

ISA divergence is expected and legitimate (D1 decision), so the gate is
**numerical equality against the preserved originals**, which is a stronger
check anyway and is exactly what keeping the old files buys us.

- **Bitwise gate where the FP reduction structure is unchanged.** The K
  prefetch distance, V LDS layout and `Q_ROW_TILES` knobs do not alter
  floating-point operation order, so aiw must match the corresponding original
  **bit for bit**. `test_binding_prefetch_matches_baseline_bitwise` shows
  baseline and bp already agree bitwise today, so this bar is known to be
  achievable.
- **Tolerance gate where it does change.** `QK_SHARDS > 1` splits the QK
  dot-product across waves and sums the partials, so rounding differs by
  construction. Those configs (`head_dim` 224, and the `head_dim//128` default
  above 128) compare against the reference at tolerance, not bitwise.
- **Perf:** interleaved 3-rep head_dim ladder plus spill counts, per the lore
  doc's discipline. Minor regressions accepted per the D1 decision; the number
  to *record* rather than gate on. Large or non-monotonic regressions still
  warrant investigation — "minor" means a few percent, not a cliff.

### [DECISION D2] Do we keep constexpr specializations for the JIT path?

Your doc frames the changes as "required to use as an **AOT** kernel". Making
strides, `num_heads` and `sm_scale` runtime is unambiguously right for AOT, but
it is not free on gfx1201:

- Address arithmetic today folds `STRIDE_TOKEN` into immediate offsets. Runtime
  strides mean a `v_mad` per address. `sdpa_lore_gfx1201.md` already records
  addressing at ~51 VGPRs; this pushes it up, and we are register-limited at
  `head_dim >= 192`.
- `head_idx = block_id % NUM_HEADS` currently folds to a shift/mask. A runtime
  `num_heads` makes it an integer division unless we move to a 3D grid
  (see P1).

FlyDSL is a JIT DSL, so we *can* keep both: one `STRIDES_CONSTEXPR` build flag,
AOT builds set it false. The question is whether the maintenance cost of two
paths is worth whatever the measured delta turns out to be.

**Recommendation:** do not decide yet. Phase 0 measures the delta first
(that is its entire purpose). If it is under ~3% we keep only the general path
and the question dissolves.

*Cheapened by D1.* Now that aiw is knob-driven, `STRIDES_CONSTEXPR` is just
one more `const_expr` branch rather than a second code path, so keeping both
costs far less than it would have across three files. That lowers the bar for
answering D2 "keep both", but does not pre-empt the measurement.

### [DECISION D3] Name for the dual-role `cu_seqlens_q` argument

You asked me to propose one. The argument is a `[batch+1]` cumulative-offset
array in AOTriton, but upstream hands us `[batch]` real lengths.

- **`seq_bounds_q` / `seq_bounds_k` (recommended).** Neutral about whether the
  contents are prefix sums or lengths; "bounds" reads correctly for both. The
  `Num_seqlens` sign already selects the interpretation, so the name should
  not contradict either reading.
- `seqlens_q` — matches upstream, but actively misleading in the packed-varlen
  case where the values are offsets, not lengths.
- `q_seq_index` — accurate for the cumulative case, wrong for the lengths case.

I would also rename `Num_seqlens` itself to **`varlen_mode`**, since its
magnitude is never used — only `> 0`, `== 0`, `< 0`. Calling it a count when it
is a three-way enum is the thing that made it hard to read in the first place.

### [DECISION D4] Is ALIBI in scope?

`sdpa-feature-gap.md` does not list it. AOTriton's `attn_fwd` has `USE_ALIBI`
and `stride_az/ah`, but the body reads an `alibi_slopes` symbol that is **not a
kernel parameter** — as written that branch cannot compile, so I read it as
dead/unfinished. Assuming **out of scope** unless you say otherwise.

### [DECISION D5] Where does the padded-`seq_len` host workaround go?

Once P2 lands in-kernel `Max_seqlen_q/k` masking, the interface's `F.pad` +
`_MAX_NONCAUSAL_PAD_RATIO` `ValueError` becomes both unnecessary and wrong
(it rejects calls the kernel could then handle correctly). Plan assumes we
delete it in P2. Flagging because it is a user-visible API behaviour change.

---

## 4. Phases

Ordering follows your constraints: dropout before gSWA; gSWA and persistent-
dynamic last; INT8 never.

### P0 — Price the de-constexpr-ing (gate, not a feature)

**Goal.** Learn what AOT-ability costs before building on it.

**Do.** On one builder only, behind a build flag, promote `sm_scale` and the
strides to kernel arguments. Named `stride_q0, stride_q1, stride_q2` per your
porting instruction (`stride_q3` is the contiguous `D` dim, stays 1 and is not
passed). Keep BSHD semantics identical so the existing tests pass unchanged.

**Verify.** `test_flash_attn_func_gfx1201.py` green. Then the A/B that
matters: interleaved 3-rep benchmark (per the lore doc's measurement
discipline — this board drifts ~5%) across `head_dim` 64/128/192/256, causal
and non-causal, reporting TFLOPS and **VGPR/spill counts**.

**Output.** A number that resolves D2. Also the first real read on whether
`head_dim >= 192` starts spilling.

**Risk.** This is the phase most likely to produce an unpleasant surprise.
Sizing it first is deliberate.

---

### P1 — Layout generality, MQA/GQA, LSE, numerics

**Goal.** Everything that does not touch the masking structure.

**Do.**

1. **Strides + arbitrary layout.** Land P0 unconditionally. Switch the kernel's
   *shape* convention to BHSD to match AOTriton (the interface transposes; no
   `.contiguous()` needed since we read strides). Confirm the 64-bit base /
   32-bit offset split from `sdpa_lore_gfx1201.md` still holds — the `i32`
   offset must remain confined to the `D` dimension, since B/H/S can exceed 4G.
2. **`Num_head_q/k` + MQA/GQA.** `off_h_k = off_h_q // (Num_head_q // Num_head_k)`.
   Move to a **3D grid** `(start_m, off_h_q, off_z)` so no runtime integer
   division is needed for the head/batch decomposition. This is a
   prerequisite for persistent-dynamic later anyway.
3. **`Hdim_qk` / `Hdim_vo` + `PADDED_HEAD`.** Per §2.5 this is the largest
   item in P1. Concretely:
   - Drop the `head_dim % 16 == 0` assert; keep `<= 512`.
   - Split `BLOCK_DMODEL` into independent `BLOCK_DMODEL_QK` /
     `BLOCK_DMODEL_VO` (each `hdim` rounded up to 16 elements), and separate
     that concept from the existing `head_dim_v`/`d_offset` V column slicing,
     which stays but must be renamed so the two are not confused.
   - `Hdim_qk`/`Hdim_vo` become runtime arguments; the tile widths stay
     constexpr.
   - Per-8-chunk load guards on `Q`/`K` (never per-element). Skip chunks
     wholly beyond the pitch; zero-fill `[hdim_qk, BLOCK_DMODEL_QK)`.
   - Hoist `K`'s zero-fill to a one-time LDS prologue (§2.5 point 3).
   - Leave `V` unzeroed; mask the `O` store along `D` instead.
   - Fix or gate `ENABLE_LDS_VEC16` — 32-byte loads over a 16-byte-guaranteed
     pitch is an OOB read at the tensor tail.
   - Measure whether a separate `PADDED_HEAD=False` variant is worth building
     at all, given the prologue trick removes the per-iteration cost.
   - **Retune `default_block_m()` / `default_block_n()` on the pair
     `(Hdim_qk, Hdim_vo)`, not a single `head_dim`** (§2.5.1). This is the
     schedulable substitute for dynamic VGPR: when one GEMM's register cost
     collapses, spend the slack on a wider tile for the other. The existing
     tables are one-dimensional and will simply be wrong for asymmetric pairs.
4. **LSE output (`L`), optional.** Currently not computed at all. `L_not_null`
   gate so `L == nullptr` skips it.
5. **Numerics, from your "Improvements from Stock FAv2" list.** Three items,
   all small, all worth doing here because item (c) is entangled with LSE:
   - **`-inf` → `-3.40282e+38`.** We use `float("-inf")` for both `m_i` init
     and the mask fill (`c_neg_inf`). Straight substitution.
   - **`1/l_i` via reciprocal.** Already `arith.divf(1.0, l_final)` — believed
     conformant, will confirm in the ISA.
   - **Avoid FMA — this is a real bug we have.** Our loop keeps `m_i`
     *unscaled*, then computes
     `p = exp2(fma(s_raw, sm_scale_log2e, -sm_scale_log2e * m_new))`
     (`flash_attn_func_gfx1201.py:739-749`). That is structurally the pattern
     AOTriton flags in ROCm/aotriton#54 and warns against by name. The fix is
     AOTriton's: scale `qk` **before** the row max, keep `m_i` in the scaled
     domain, and make the exponent a plain subtract. Conveniently this is also
     the convention LSE needs (`m_i + log2(l_i)` must be scaled), so both land
     together.

**Verify.** Existing suite; new GQA cases (`Hq/Hk` = 8/1, 8/2, 8/8); LSE
against `torch.logsumexp` on the reference scores; non-multiple-of-16 head
dims. For the FMA fix specifically, a large-magnitude-input accuracy case that
fails before and passes after — otherwise we are asserting a fix we cannot
demonstrate.

---

### P2 — Interval decomposition, in-kernel ragged masking

**Goal.** The structural change from §2. No new user-facing feature except
that `seq_len` no longer has to be padded and `Lq != Lk` becomes legal.

**Do.**

1. Port `closed_interval_isect` / `is_closed_interval_empty` /
   `closed_interval_size` / `div_rd` / `calculate_intervals`.
2. Add `Max_seqlen_q` / `Max_seqlen_k` as separate arguments; drop the
   interface's `Lq == Lk` restriction.
3. Split the inner loop into the two-region form from slide 18 (`nblocks_1 +
   nblocks_2` with the piecewise `start_n`). Full-block region unmasked,
   masked region always masked.
4. **Delete the named-scalar causal unroll.** Mask becomes an unconditional
   `select` over the `Vec` accumulators. `BLOCK_N == 32` restriction lifts,
   which also unblocks the `head_dim` 16/32 wide-`BLOCK_N` tuning already in
   `default_block_n`.
5. Add `CAUSAL_TYPE` (0/1/2) as a constexpr; derive `window_left`/`window_right`
   internally from it. **Do not** expose them as arguments yet.
6. Early-exit path: fully-masked tiles write zeros to `O` and `+inf` to `L`.
7. Per D5, remove the host-side `F.pad` and the pad-ratio `ValueError`.

**Verify.** Non-multiple-of-`BLOCK_M/N` seqlens across a sweep; `Lq != Lk`
both causal alignments against `_reference_scaled_dot_product_attention`;
bottom-right vs top-left divergence explicitly asserted (they differ exactly
when `Lq != Lk`, so this is the first phase where that is testable).

**Perf gate.** Interleaved A/B vs P1. Expect a regression on short sequences
where the masked region is a large fraction of the row; the two-region loop
also costs us the compile-time trip count the current unroll depends on. If
the regression is bad, the fallback is a constexpr "regular" specialization
selected when `seqlen_k % BLOCK_N == 0 && !causal` — which is exactly the
regular/irregular split your `Max_seqlen_q/k` section calls for.

---

### P3 — Varlen

**Goal.** `varlen_mode` (D3) three-way: `0` regular BHSD, `> 0` packed 1HTD,
`< 0` BHSD padded to `Max_seqlen_q`.

**Do.** `seq_bounds_q/k` loads and the per-batch `seqlen_q/k` derivation; the
`seq_strides_q/k` THD+padding path; compact LSE layout `(H, TotalS)` matching
FA (`_lse_offset`); `start_M >= seqlen_q` early-out.

**Depends on** P2 — every varlen tile is a ragged tile, so this is nearly free
once intervals exist, and near-impossible before.

**Verify.** Port AOTriton's `test_varlen.py` cases; ragged batches including
zero-length and single-token sequences.

**Note.** You flagged `04cdead5c837f21754de39a175a30c479da74625` for detail — I
have not read it yet and will before starting this phase.

---

### P4 — Bias

**Goal.** `B` + `stride_b0/b1/b2` (`stride_b3 == 1`), `BIAS_TYPE` 0/1.

**Do.** Load the `[BLOCK_M, BLOCK_N]` bias tile, add `bias * log2(e)` to `qk`
before the row max. Masked-region loads need the same `mask_on_seq_q/k`
treatment as K/V.

**Cost we should size early:** the bias tile is `BLOCK_M × BLOCK_N` elements
per KV step. At `BLOCK_M=128, BLOCK_N=32` that is a second full-rate global
stream in a loop the lore doc already characterises as latency-bound. This may
need its own LDS staging rather than direct-to-register loads. Sized here
rather than discovered in P4.

**Verify.** Against reference with a random bias; bias + causal; bias + varlen.

---

### P5 — Dropout (Philox)

**Goal.** `dropout_p`, `philox_seed_ptr`, `philox_offset1/2`, seed/offset
outputs. Ordered before gSWA per your instruction.

**Do.** Implement Philox 4×32-10 directly — full 64-bit seed, full 64-bit
offset, all 10 rounds, per your explicit "do not sacrifice PRNG quality"
constraint. Adopt AOTriton's `PHILOX_RN_PER_OFFSET = 4` amortization (one
offset yields 4 `u32`, so the offset grid is `BLOCK_N/4` wide). Not required to
bit-match AOTriton's stream.

**Design note.** Philox is ~10 rounds of 32×32→64 multiplies per 4 outputs.
On a loop that is LDS-latency-bound rather than VALU-bound, there may be issue
slots to hide this in — that is the optimistic read. The pessimistic read is
that it is pure added VALU *and* added VGPR pressure at exactly the head dims
where we already spill. Prototype the RNG standalone in `kernels/microbench/`
and price it before wiring it into the kernel.

**Verify.** Dropout demands a deterministic oracle: return the dropout mask
(AOTriton's `RETURN_ENCODED_SOFTMAX` kernel, or a debug path) and feed it to
`sdpa_math(..., dropout_mask=...)` in `_common_test.py`. Plus a statistical
test that the keep rate matches `1 - p` and that streams differ across
seed/offset.

---

### P6 — Generalized SWA

**Goal.** Runtime `Window_left` / `Window_right`, negative values allowed.

**Do.** Promote the two values from P2's constexpr derivation to kernel
arguments; add `parse_window` with the two sentinels (`0x80000001` causal-left,
`0x80000002` causal-right) for varlen, where there is no uniform
`seqlen_q/k` (slide 17). If P2 was built as §2 describes, this phase is small.

**Verify.** Symmetric windows; one-sided windows; **negative** windows in both
directions; window + varlen via the sentinels; window + causal-equivalence
(`window_left = seqlen_q, window_right = 0` must reproduce top-left causal
bit-for-bit against P2's output — a strong self-consistency check).

**Perf.** Your doc is right that the two-region loop needs revisiting here.
I would defer that to the persistent-dynamic task rather than tuning twice.

---

### Deferred (explicitly not in this plan)

`PERSISTENT_TYPE` / persistent-dynamic (own task, implement + tune together,
and it wants P1's 3D grid); `NUM_XCDS` (always 1 on gfx1201, revisit for
gfx1250); `INT8`; `RETURN_ENCODED_SOFTMAX` as a fused option (separate kernel,
as AOTriton ships it); `mxfp8` (design not settled); `PRE_LOAD_V` (already
optimized); `BLOCK_M/N` retuning (do it once, after P2 changes the loop shape).

---

## 5. Perf watch list

Every phase adds either runtime values (SGPR + address VALU) or live state
(VGPR) to a loop that `sdpa_lore_gfx1201.md` characterises as **LDS-latency-
bound** and that already spills at `head_dim >= 192`. Cumulative risk, not
per-phase risk. Concretely:

| Phase | Adds | Watch |
|---|---|---|
| P0/P1 | ~12 stride SGPRs, address `v_mad`s | VGPR count at hdim 192/256 |
| P2 | dynamic trip count, piecewise `start_n` | loss of the unrolled inner loop |
| P4 | a second global stream in the hot loop | LDS budget, `s_waitcnt` bubbles |
| P5 | Philox VALU + state | VGPR at hdim ≥ 192; spills |

**Discipline (from the lore doc, non-negotiable):** interleaved A/B, ≥3 reps,
never trust a single-run delta — this board drifts ~5%, which is larger than
most of the effects we will be chasing. Every phase reports TFLOPS **and**
spill counts across the `head_dim` ladder, not just the shape being worked on.

Suggested standing baseline: `B=1 H=8 N=4096 f16`, causal and non-causal,
`head_dim` ∈ {16, 64, 128, 192, 256, 512}, captured once at P0 and re-run at
every phase boundary.

---

## 6. Open questions

1. **D1** — consolidate the three builders, or freeze two?
2. **D2** — keep constexpr specializations alongside the general path? (P0
   answers the perf half; the maintenance half is yours.)
3. **D3** — `seq_bounds_q/k` + `varlen_mode` naming: acceptable?
4. **D4** — ALIBI in or out?
5. **D5** — removing the interface's `F.pad` + pad-ratio `ValueError` is a
   user-visible behaviour change. Confirm.
6. **LSE layout.** AOTriton has three (`(B*H, S)`, `(B, H, S)` compact,
   `(H, TotalS)` varlen). Which do we owe upstream?
7. **Output dtype.** AOTriton stores `O` in `Out.type.element_ty`, which need
   not equal `Q`'s. We hard-code `dtype(O) == dtype(Q)`. Real requirement or
   incidental?
8. ~~`Hdim_qk != Hdim_vo` interaction with V column slicing.~~ **Resolved —
   see §2.5.** ~~Dynamic VGPR allocation.~~ **Resolved — see §2.5.1:
   available on gfx1201 but rejected, primarily because our register peak is
   loop-carried rather than transient.** Replaced with a
   `(Hdim_qk, Hdim_vo)`-aware `BLOCK_M`/`BLOCK_N` tuning item in P1.

   Minor footnote, not a blocker: the VGPR **segment** size (§2.5.1 point 1)
   is a chip-wide driver config we cannot set from a kernel. If the R9700 is
   configured for 16, any dynamic-VGPR wave is capped at 128 VGPRs. Only
   matters if some future work revives the feature — it is not load-bearing
   for the rejection, which rests on the loop-carried argument.
