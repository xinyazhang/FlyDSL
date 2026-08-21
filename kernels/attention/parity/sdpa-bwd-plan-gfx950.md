# Executive plan: flash-attention **backward** for gfx950 — dK/dV and dQ(/dB)

Companion to `sdpa-close-gap-gfx950.md`, which took the forward kernel from a
head_dim-64/128 fast path to full AOTriton parity (P0–P7). This plans the
backward pass on the same stack: a dual-wave body for the widths that fit and a
staged/sharded *wide* body for the ones that do not, reaching the feature
surface the forward now has.

The gfx1201 backward already exists and is thoroughly documented
(`fmha_bwd_dkdv_gfx1201_kernel.py`, `fmha_bwd_dq_gfx1201_kernel.py`,
`fmha_bwd_fuse_gfx1201_kernel.py`, ~4000 lines with three tuning modules and
2100 lines of tests). **It is the specification, not the implementation** —
and only the first two are being ported; §4 says why the fused one is not.

The forward work established that features port across architectures and
schedules do not; the same split applies here, and the sections below say which
is which.

---

## 0. The shape of the problem, in one page

Backward needs three outputs from five inputs plus one reduction:

```
delta[i]     = rowsum(dO[i,:] * O[i,:])                 preprocess
P[i,j]       = exp2(qk_scale * S[i,j] - lse2[i])        S = Q.K^T
dP[i,j]      = dO[i,:] . V[j,:]
dS[i,j]      = P[i,j] * (dP[i,j] - delta[i])
dV[j,:]     += sum_i P[i,j]  * dO[i,:]
dK[j,:]     += sum_i dS[i,j] * Q[i,:]   * sm_scale
dQ[i,:]      = sum_j dS[i,j] * K[j,:]   * sm_scale
dB[i,j]      = dS[i,j]                                  (bias builds only)
```

`dK`/`dV` reduce over **query** rows; `dQ`/`dB` reduce over **key** rows. Those
are different loop nests over the same data, which is why AOTriton — and
gfx1201 after it — ship them as separate kernels.

**dK/dV is the forward loop transposed.** K/V stay resident and Q/dO stream
past; the forward keeps Q resident and streams K/V. Every helper in the stack
was written against "a tensor, its bounds and where it lands" rather than
against K or V by name, so the roles swap without rewriting them.

**dQ is shaped like the forward.** Q resident, KV streaming, three GEMMs per
tile instead of two.

---

## 1. The architectural bet — **spiked, and it pays**

gfx1201's dK/dV stages **four LDS tiles** — Q, Qᵀ, dO, dOᵀ — because its four
GEMMs contract Q and dO over two different axes:

| GEMM | contracts | A operand wants |
|---|---|---|
| `S  = Q·Kᵀ` | d | Q row-major |
| `dP = dO·Vᵀ` | d | dO row-major |
| `dVᵀ = dOᵀ·P` | q | dO **transposed** |
| `dKᵀ = Qᵀ·dS` | q | Q **transposed** |

Those four tiles bound `block_m` there and forced a `transposed_source` knob
whose "derived" arm pays eight strided scalar reads per operand.

**gfx950 needs two tiles, not four.** Answered from two independent sources
before writing any code.

### 1.1 The ISA says the instruction is for exactly this

CDNA4 ISA §11.4, *MFMA Transpose Load from LDS*:

> **DS_READ_B64_TR_B16** — Used for either **column major matrix A** or row
> major matrix B data load to 2 VGPRs. Element size is 16b. **Two instructions
> load a complete matrix.** The first loads K=0..3 and K=8..11 into two VGPRs,
> and the next loads K=4..7 and 12..15. Each lane (one VGPR) holds 4
> consecutive M or N values.

"Column major matrix A" *is* the transposed A operand the two q-contracted
GEMMs want. The instruction was designed for this case; the forward already
uses it for V, which is the same shape of problem one tensor at a time.

Two instructions per operand also matches the arithmetic: a `32x32x16` bf16 A
operand is 8 bf16 per lane = 4 VGPRs, and each read returns 2.

### 1.2 AITER's tuned kernel confirms it in practice

`bwd_hd128_bf16_a32_psskddv.co` disassembled: **512 `ds_read_b64_tr_b16`**, the
single most frequent instruction in the kernel, against 512 MFMAs — 1:1. They
issue in pairs at +32-byte offsets and land directly in MFMA operand registers,
sometimes straight into AGPRs:

```
ds_read_b64_tr_b16 v[36:37], v11 offset:36224
ds_read_b64_tr_b16 v[38:39], v11 offset:36256      <- +32B, completes the pair
...
ds_read_b64_tr_b16 a[36:37], v16 offset:32768      <- directly into an AGPR
v_mfma_f32_32x32x16_bf16 a[192:207], a[112:115], v[60:63], a[192:207]
```

**And AITER has two code generations, which is the sharper evidence.** Its
hd64 and hd192 kernels use `v_mfma_f32_16x16x16_bf16` with **zero** transpose
reads — the gfx1201-shaped approach of staging both orientations. Its hd128 and
hd192_128 kernels use the wider MFMA shapes *with* heavy transpose use:

| kernel | `tr_b16` | `ds_read_b128` | MFMA shapes |
|---|---|---|---|
| hd64 | 0 | 120 | `16x16x16` ×480 |
| hd192 | 0 | 128 | `16x16x16` ×240 |
| hd128 | **512** | 220 | `16x16x32` ×384, `32x32x16` ×128 |
| hd192_128 | **408** | 198 | `16x16x32` ×288, `16x16x16` ×120, `32x32x16` ×60 |

So the transpose path is the newer, tuned one, adopted where it mattered and
not retrofitted elsewhere. That is a recommendation *and* a warning: it is
worth doing and it was not free enough to backport.

### 1.3 Two constraints the ISA attaches, and one is a hazard

- **"Prior to executing these instructions the EXEC mask must be set to all
  1's."** A transpose read inside a divergent region is undefined, not merely
  slow. The forward's masking is `select`-shaped rather than branch-shaped so
  it never hit this, but the backward has genuine `scf.if` regions around
  edge tiles. **Treat this as a hazard class of its own**: it belongs in
  `sdpa_lore_gfx950.md` next to the two inline-asm hazards, because the failure
  mode is the same — a wrong-but-finite answer with no diagnostic.
- LDS addresses must be aligned to the data size, and 64-bit DS ops need an
  even-aligned VGPR pair.

### 1.4 What is left of B0

Not "does it work" but "what is the exact map". One probe: stage a Q tile,
read it back through the transpose path, and check the lane→(m, k) mapping
against a host reference for the specific MFMA shapes chosen in §2. Half a day,
and it produces the table every LDS layout below is written against.

---

## 2. What transfers unchanged, and it is most of the feature surface

P0–P7 built machinery that is *about the problem*, not about the direction of
the loop. All of it is reusable as-is:

| what | where | note |
|---|---|---|
| stride ABI, `(batch, head, seq)` named per axis | forward `_args` | the bwd tensors get the same treatment |
| `decode_addressing`, `lse_token_pitch` | `fmha_common_gfx1201` | already arch-shared; the forward proved it |
| `lse_row_addressing` (`_HT`/`_TH`) | same | bwd *reads* LSE where the forward writes it |
| `resolve_window`, the sentinels | same | |
| `philox.py`, `Philox.grid_plane/grid_offset` | `philox.py` | **the mask must regenerate bit-identically**; this is the contract P6 pinned |
| `MaskedAxis`, the padded-head/8xD contract | forward | including the `HDIM_QK_FLOOR` trick |
| the geometry guard and the rows-per-wave ceiling | `fmha_traits_gfx950` | P7 found the second one the hard way |
| `SharedAllocator`, DMA helpers, alias scopes | `flash_attn_utils` | imported, never edited |

**AOTriton is the algorithm reference for all of this**, and it is worth
reading directly rather than only through the gfx1201 port:
`bwd_kernel_dk_dv.py` / `bwd_inner_dk_dv.py`, `bwd_kernel_dq.py` /
`bwd_inner_dq.py`, `bwd_kernel_fuse.py`, plus `bwd_preprocess.py` and
`bwd_postprocess.py`. The gfx1201 files are a port *of* those with our ABI
substituted, so where the two disagree the Triton source says what the
algorithm requires and the gfx1201 source says what a previous port chose.

`decompose_causal_regions` deserves its own line: gfx1201 discovered that a KV
block's Q range is **the same function with the axes swapped**, called as
`decompose_causal_regions(start_k, k_len, q_len, w_right, w_left, BLOCK_N, BLOCK_M, alive)`.
That identity holds here too and is the whole of causal/window support for
dK/dV.

**What does not transfer is the schedule and the operand algebra.** gfx1201 is
WMMA 16x16x16 on wave32 with no global→LDS path; gfx950 is MFMA 32x32x16 on
wave64 with DMA and MFMA/VALU co-execution. Concretely:

- **The MFMA shape is a tuning axis here, which it was not in the forward.**
  The forward uses `32x32x16` throughout. AITER's backward uses all three of
  `16x16x16`, `16x16x32` and `32x32x16`, mixing two shapes within one kernel
  (§1.2), and exposes `ts_qo ∈ {16, 32}` as a knob. So "a wave owns 32 KV rows"
  is a *choice* — 16 is equally available and halves the accumulator (§3) at
  the cost of halving `BLOCK_N` for a given wave count. Pick it per width by
  measurement, and do not bake 32 into the helpers.
- The lane→(m, k) maps differ, so every LDS layout is re-derived. The forward's
  `_score_column_runs` / threshold-table technique is the right tool: derive
  the map once, use it everywhere, never transcribe it twice. With two MFMA
  shapes in play that discipline stops being tidiness.
- The dual-wave 8-cluster pipeline applies directly, because dK/dV streams
  **two** tensors (Q, dO) exactly as the forward streams two (K, V).

---

## 3. Register pressure, and why the tiling choice decides the wide threshold

This is the number that decides how much of the plan is "wide".

A dK/dV wave holds **two** accumulators, `dKᵀ` and `dVᵀ`. **At 32 KV rows per
wave** they are `[d][32]` f32, which over 64 lanes is `d/2` VGPRs each:

| head_dim | dKᵀ | dVᵀ | total | against 256 AGPRs |
|---|---|---|---|---|
| 64 | 32 | 32 | 64 | comfortable |
| 128 | 64 | 64 | 128 | fits |
| 192 | 96 | 96 | 192 | tight |
| 256 | 128 | 128 | **256** | the entire AGPR file |
| 384+ | 192 | 192 | 384 | impossible unsharded |

The forward carries **one** O accumulator of the same `d/2`, which is why its
wide path starts at 384. So dK/dV looks like it needs sharding from 256, and
192 looks marginal.

**B1 measured against that and it is premature.** At head_dim 128 the first
build spilled 246 times at 0 AGPRs and did 200 TF; the fix was not sharding but
the wave-count / `waves_per_eu` *pair* — 4 waves alone gave 403 TF, the same
build with `waves_per_eu=1` gave 721, and 2 waves reached 788 with 108 AGPRs
and zero spills. A build at 0 AGPRs with a nonzero spill count is not short of
registers, it is forbidden from using half of them, and no amount of sharding
fixes that. So treat the table as an upper bound on *demand*, and settle the
threshold in B3 by sweeping the pair — the arithmetic below is what a wave
needs, not what the allocator will give it.

**This table is a function of the tiling choice, not a constant.** At 16 KV
rows per wave the accumulators are `[d][16]` and cost `d/4` each, halving every
number above and pushing the wide threshold *later* than the forward's rather
than earlier — paid for by halving `BLOCK_N` at a given wave count, so the same
work needs more workgroups or more waves. That trade is precisely what AITER's
`ts_qo` knob selects, and it is the first thing B3 should sweep. Read the split
below as the 32-row case:

```
head_dim <= 128     dual-wave, unsharded
head_dim 160..224   dual-wave, watch AGPRs (P7's lesson: measure, do not infer from spills)
head_dim >= 256     wide: D_STAGES for LDS, shard the d axis of dK/dV
```

dQ is better placed: one accumulator `[32][d]`, same as the forward's O, so its
wide threshold should land near the forward's. gfx1201 already found that
`QK_SHARDS` fits dQ *more* neatly than the forward, because GEMM1's reduction
axis and GEMM3's output axis are the same axis — one offset serves all three.
That reasoning is arch-independent and should be reused.

---

## 4. No fused kernel, and this is a scope decision rather than a deferral

**Two kernels: dK/dV and dQ. There is no gfx950 fused variant in this plan,
and none is planned after it.** `fmha_bwd_fuse_gfx1201_kernel.py` is not being
ported.

The reason is what fusion is *for*. Its whole return is saving one dispatch,
which only registers when the dispatch is a visible fraction of the work —
small batches, short sequences, decode-shaped problems. That regime is already
served: **AOTriton's Triton `bwd_kernel_fuse` covers it acceptably**, and at
those sizes the gap to hand-tuned assembly is not what decides anything. Where
this stack earns its keep is the large workloads, and there the dispatch is
noise and the split kernels win on their own terms.

The cost side supports it. gfx1201 measured the fused kernel: two program roles
selected by `block_idx.x`, the dK/dV role needing 52480 B of LDS and the dQ
role 13568 B, and **a workgroup's LDS allocation is static, so the binary
reserves the larger for both** — the dQ programs run at a quarter of the
occupancy a split kernel gives them. On gfx950 that trade is worse, because the
dual-wave body already spends most of its 160 KB on two KV tiles in flight.

So this is not "fuse later once dispatch overhead shows up in a profile". The
profile would have to show it on a workload that has a better answer already.
If that changes, the thing to build is a *persistent* backward, not a
two-role one — the occupancy cliff above is a property of the role split, not
of doing more work per launch.

---

## 5. `delta` — take it as a tensor first, kernel it later

`delta = rowsum(dO * O)` is a preprocess. AOTriton has a `bwd_preprocess`
kernel and AITER ships it as 12 separate `odo` binaries, so both tuned
implementations keep it out of the main kernel. gfx1201 takes it as a
host-computed tensor and says so.

Do the same: one argument, `(dO.float() * O.float()).sum(-1)` on the host, then
a gfx950 preprocess kernel as a later, separately-measured step. It changes one
argument and nothing else, and the two references agreeing on the split is
reason enough to keep it out of the main kernel. (This is the `delta`
preprocess, not §4's dK/dV+dQ fusion -- unrelated decisions that happen to
share a word.)

---

## 6. Phases

Each phase is a shippable increment with its own gate. The forward's phase
structure earned its keep — every one of P1–P7 found at least one bug that the
previous phase's tests could not see — so this mirrors it.

### B0 — the transpose spike *(mostly done; ~half a day left)*

§1 answers the question this phase existed to ask: **two LDS tiles, not four**,
on the ISA's own description of the instruction and on AITER's tuned kernel
using 512 of them 1:1 with its MFMAs.

What remains is the lane map, not the decision. One probe: stage a Q tile, read
it back through `ds_read_b64_tr_b16`, and check lane→(m, k) against a host
reference for the MFMA shapes §2 selects. It produces the table every LDS
layout downstream is written against.

Fold in two ISA constraints while writing it (§1.3): the **EXEC mask must be
all 1's** across these reads, and 64-bit DS ops need an even-aligned VGPR pair.
The first is a hazard, so give it a test that runs the read under a divergent
region and shows the guard holds.

*Gate:* the lane-map table, written down, plus that hazard test.

### B1 — dK/dV, dense, non-causal, head_dim 64/128

The dual-wave body with Q/dO streaming, four GEMMs, `dKᵀ`/`dVᵀ` accumulators
and the `v8` epilogue. No masking, no features.

*Gate:* against `torch.autograd.grad` on the forward kernel's own output, so
the fwd/bwd pair is self-consistent; plus against a pure-torch reference
implementing the equations of §0 in fp64.

### B2 — dQ, dense, non-causal, head_dim 64/128

Three GEMMs, `kt_lds_layout` for GEMM3, delta as a tensor.

*Gate:* same two oracles. Additionally, **dQ and dK/dV must agree with a single
autograd call** — they are separate kernels computing parts of one gradient,
and testing them apart hides a scale or transpose error that cancels.

### B3 — the head_dim ladder and the wide body *(before the features, deliberately)*

Extend to the forward's `LADDER` (32…512), adding `D_STAGES` and d-axis
sharding for dK/dV per §3, and sweeping the rows-per-wave choice that section
identifies. Expect the wide body to be a separate file, as
`fmha_wide_gfx950.py` is, and expect the 8-minute build rule to bite.

**This phase runs before masking, and the ordering is the lesson rather than a
preference.** On the forward, head_dim 512 was not a wider version of head_dim
128 — it was a second kernel body, arrived at over P8.0–P8.2 with the LDS
blocker, the wave-count effect, cross-phase prefetch and three separate
addressing bugs found one at a time. It needed frequent review and it needed
the *simplest possible* kernel underneath it, because every one of those
investigations was a bisection and each extra live feature is another variable
in it.

The backward is worse on both counts: two accumulators instead of one (§3),
four GEMMs instead of two, and a tiling axis the forward did not have (§2). So
get 512 working against a dense, non-causal, no-feature kernel — then add the
features, where each one is a `const_expr` arm over a structure that already
holds.

Doing it the other way costs twice. Every wide-path bisection would carry a
causal mask and a window guard through it, and every masking bug found at 512
would be ambiguous between the feature and the tiling.

*Gate:* the 8xD input contract, the padded-head path, and — from P7 — the
rows-per-wave ceiling enforced rather than commented.

### B4 — causal and windows

Now cheap, because the structure is settled. `decompose_causal_regions` with
the axes swapped for dK/dV; the forward's two-sided guard for dQ. The forward's
P3 lesson applies directly: the **bit-identical sentinel oracle** (a window
build fed `WINDOW_BOTRIGHT` must reproduce plain causal exactly) is the
sharpest test available, and the tile cut must be verified by *timing*, because
a dead tile is a no-op.

Watch for the literal-zero tile base: P3 found four instances of it in the
forward, and the backward has the same shape of prologue. **Both bodies**, and
that is a direct P3 finding — the wide body's KV loop started at a literal tile
0, so its window was correct and its tile cut inert, measuring 0.92x where the
dual-wave body got 6.7x. Running B3 first is also what makes that check
possible here at all.

### B5 — varlen

`VarlenBits`, all five modes, reusing the forward's decode. **Each side needs
its own batch index** (P4's `0x040B` bug) and a causal varlen build wants
`cross_seqlen`; both are already-solved problems, and the fix is to reuse the
forward's resolver rather than re-derive it.

### B6 — dropout, and then dB

Dropout first, because it has a hard contract: the backward must regenerate the
forward's mask **bit-identically**, so it must call the same
`grid_plane`/`grid_offset` with the same `(seed, offset)` the forward reported
through `philox_seed_output`/`philox_offset_output`. P6's tiling-independence
test becomes a *cross-kernel* test here: forward and backward at different tile
geometries must agree.

Then `dB`. **AOTriton has it and our gfx1201 port dropped it** --
`bwd_kernel_dq` takes `B` and emits `DQ, DB`, with `store_db` guarding a
`db_ptr` store inside `bwd_inner_dq`, while `fmha_bwd_dq_gfx1201_kernel.py`
mentions neither. `dB = dS`, so it is an extra store on a value the kernel
already has, not new arithmetic. It belongs to the **dQ** kernel, not dK/dV,
because that is where `dS` is materialised per (q, k) element with the query
rows resident.

### B7 — tuning, with P7's method

A knob sweep, and then **interleaved single-GPU A/B for anything under 10%**.
P7's headline is that a four-GPU concurrent sweep produced 2–5% phantom deltas
at every rung and none survived re-measurement. Use the sweep to rank and to
find breakage; use A/B to decide. Reuse `tooling/sweep_knobs_gfx950.py` and
`tooling/ab_knobs_gfx950.py`, and give the runner the per-build timeout P7
lacked.

---

## 7. Verification: three sources, and what each is good for

The forward had a *bitwise* oracle — a production kernel computing the same
thing in the same order. **The backward has no such oracle.** It has three
other sources, and confusing what each is for is the main way this work could
go wrong.

### 7.1 PyTorch's math backend — the correctness gate

The reference is `torch.nn.attention.SDPBackend.MATH` autograd, used the way
`test/test_transformers.py::test_flash_attention_vs_math_ref_grads` uses it,
and **that methodology matters more than the reference does**. A fixed
tolerance on a bf16 backward is either so loose it accepts real bugs or so
tight it fails on arithmetic order. The test instead computes the same problem
three ways:

```
ours      = our kernel, bf16/f16
ref_low   = math backend, same dtype           <- an equally imprecise honest answer
ref_high  = math backend, fp64                 <- ground truth
```

and asserts `err(ours, ref_high) <= fudge * err(ref_low, ref_high)`, per output
tensor. The bound is then *the precision the problem inherently has*, measured
rather than guessed, and it scales automatically with sequence length, head dim
and dtype. Per-tensor fudge factors, because dQ accumulates over the key axis
and is legitimately looser than dK/dV.

This subsumes the fp64 reference an earlier draft of this plan proposed, and it
is better: writing the equations of §0 out by hand is a second implementation
that can be wrong in the same way ours is.

**Additionally, self-consistency with our own forward.** Take `O` and `LSE`
from the gfx950 forward, feed them to the backward, and compare against
`torch.autograd.grad` through that same forward. This is not redundant with the
math gate — it catches convention mismatches (scale folding, `log2e`, LSE
layout, causal alignment) that a standalone comparison against math would
attribute to the backward when they are really a disagreement between our two
halves.

### 7.2 AOTriton — the algorithm reference, not the oracle

`../aotriton/modules/flash/kernel/` is the specification for *what to compute*:
`bwd_kernel_dk_dv` + `bwd_inner_dk_dv`, `bwd_kernel_dq` + `bwd_inner_dq`,
`bwd_kernel_fuse`, `bwd_preprocess`, `bwd_postprocess`. Read it directly, not
only through the gfx1201 port — `bwd_kernel_fuse` included, since it is the
fallback for the small-workload regime §4 declines to build for — where the two disagree, Triton says what the
algorithm requires and gfx1201 says what one previous port chose.

**It is not a differential oracle for everything**, and the boundary is sharp:

- Its varlen is `cu_seqlens_q/k` + `num_seqlens` + `seq_strides`, i.e.
  AOTriton's four-`VarlenType` enum. **`VarlenBits` is not in it.**
  `sdpa-varlen-plan.md` §0.1 already says this phase overshoots AOTriton and
  that the oracle there is *N separate dense calls*, an oracle built from our
  own dense path. The same applies to B5: for `0x150B` and `0x040B` there is
  nothing in AOTriton to differentially test against.
- It *does* have `dB` (§B6) and `BIAS_TYPE` on both kernels, so bias is a place
  AOTriton **is** available as a reference and our gfx1201 port is not.

### 7.3 AITER — ASM clues, and a design AITER chose that we must not

`../aiter/hsa/gfx950/fmha_v3_bwd/` is hand-written gfx950 assembly, and it is
the only source here that has actually been tuned on this architecture. Worth
reading for scheduling and LDS technique. It is **not** a functional reference,
for two reasons that are visible in its own file list.

**It is one fused `dqdkdv` kernel with dQ accumulated by atomic add to VRAM.**
Floating-point addition is not associative and atomic ordering is not
reproducible, so its dQ is **non-deterministic run to run**. We want
deterministic kernels; that alone rules the design out, and it is why the split
dK/dV + dQ shape of §4 is a requirement here rather than a preference.

**Its feature surface is a small subset of ours.** From `fmha_bwd_dqdkdv.csv`,
91 variants over: dtype {fp16, bf16}, hdim pairs {64/64, 128/128, 192/128,
192/192}, mask {0, 1, 2}, `mode` {0, 1} (batch/group), plus `pssk`, `pddv`,
`atomic32` and `bf16_cvt`. No bias, no dropout, no arbitrary head dim, no
`VarlenBits`. Measuring against it is measuring a different problem.

Three things it does teach, and they are actionable:

- **The pipeline is four kernels, not one**: `odo` (12 variants) is exactly the
  `delta = rowsum(dO * O)` preprocess of §5, shipped separately — which
  supports taking delta as a tensor first and fusing later.
- **`dq_convert` and `dq_shuffle` (12 + 4 variants) exist only to clean up
  after the atomics** — an fp32 accumulation buffer converted and reshuffled
  into the output layout. A deterministic dQ kernel needs neither, so their
  absence from our plan is a consequence of the §4 decision, not an omission.
- **`bf16_cvt` takes four values**, i.e. the bf16 rounding mode is a tuned
  parameter. Our forward already carries `daz`/`fp_mode` knobs and P5 found
  that `ninf` is not affordable once `-inf` reaches arithmetic. Expect rounding
  to matter more in the backward, where dQ sums over the whole key axis.

### 7.4 Bit-identity across configurations

Available without any reference kernel, and in the forward it caught the most:
a window build with causal sentinels against a causal build; `lpt_tile_order`
on against off; two wave geometries against each other; `p = 0` dropout against
none; varlen against *N* dense calls. Every one of these transfers.

### 7.5 Two traps that will recur

- **`torch`'s `is_causal` is top-left aligned and these kernels are
  bottom-right.** They agree only when `Sq == Sk`. P4 lost time to this, and
  the backward has more places for it to hide because dK/dV walks the transpose
  of the same region.
- **Always run a known-good control.** Three wrong conclusions in the forward
  work came from a probe with no control.

---

## 8. Risks

| risk | why it is real here | mitigation |
|---|---|---|
| ~~the transpose spike fails~~ | **resolved in §1** -- ISA and AITER both say two tiles | residual risk is the lane map only |
| a transpose read reaches a divergent region | the ISA requires EXEC all-1s and the failure is a wrong-but-finite answer | §1.3; a dedicated test in B0, and an entry in the lore beside the inline-asm hazards |
| baking one MFMA shape into the helpers | AITER mixes two shapes in one kernel and exposes the tile size as a knob | keep the shape a parameter from B1; the threshold-table technique makes that cheap |
| dK/dV register pressure at 192–256 | two accumulators, not one (§3) | shard from 256; measure at 192 rather than inferring from spills (P7) |
| inline-asm hazards resurface | the forward hit two, one still unexplained | `sdpa_lore_gfx950.md` §"Recognising one" first, before theorising; never emit a memory op as inline asm |
| the dropout mask disagrees across fwd/bwd | silent wrong gradients, no shape check notices | cross-kernel tiling-independence test in B6 |
| wide-path builds that never terminate | P7 wedged four sweep shards on exactly this | per-build timeout in every harness; treat a slow build as a result |
| split-K + backward | the forward's combine kernel is not stride-general and is guarded, not fixed | out of scope; keep the guard |
| reaching for AITER's design under schedule pressure | it is the only gfx950-tuned source, and its dQ atomics are genuinely faster | its dQ is non-deterministic by construction; read it for scheduling, never for structure |
| bf16 rounding in the dQ reduction | AITER makes `bf16_cvt` a tuned 4-valued parameter, so it is not noise | the §7.1 error-ratio gate prices it automatically; do not paper over a ratio regression with a bigger fudge factor |

---

## 9. Out of scope

- **A fused dK/dV + dQ kernel** (§4). Not deferred — not wanted. Fusion buys
  one dispatch, which only matters on small workloads, and AOTriton's Triton
  fused kernel already serves those adequately.
- **Split-K** on any backward kernel.
- **fp8** backward.
- Fixing the forward's split-K combine addressing, which remains guarded.
- The forward's unexplained head_dim 96 wait-state hazard, whose workaround is
  a register perturbation (`sdpa_lore_gfx950.md`, Hazard 2). If the backward
  reproduces something of the same shape at some width, that is evidence worth
  recording — but chasing it is not part of this plan.

---

## 10. Sequencing and what "done" means

B0 no longer gates the *design* — §1 settled that — only the lane map B1 and B2
are written against, which is half a day. B1 and B2 are independent after it
and can proceed in parallel if there are two people.

**Then B3 — the full ladder including 512 — before any feature work.** That is
the one reordering against the forward's own phase sequence, and it is
deliberate: the forward reached 384/512 at P8, *after* windows, and paid for it
by debugging a second kernel body with the masking already in place. §B3 gives
the argument. Get the hard structural width working against the simplest
possible kernel; the features are then `const_expr` arms over a structure that
already holds.

B4–B6 are ordered after it, because each phase's tests are what make the next
one's failures legible. B7 is last, once the register budget has settled —
which is the same reason the forward's sweep was P7.

**Done** is: dK/dV and dQ on the full `LADDER`, with causal, windows, varlen,
dropout, bias and padded heads, passing the §7.1 error-ratio gate against the
math backend, at a throughput recorded honestly against the MFMA ceiling.

Two of those go past our gfx1201 backward rather than merely matching it:
**`dB`**, which AOTriton has and that port dropped, and **`VarlenBits`**, which
AOTriton does not have at all.

---

## Outcome: B2 — dQ and dB *(dense, non-causal, head_dim 64/128, bf16)*

Files: `fmha_bwd_dq_gfx950.py`, `fmha_tuning_bwd_dq_gfx950.py`,
`test_fmha_bwd_dq_gfx950.py`. 35 tests, all passing; the forward's own suite
still passes untouched (329/329 — no file outside these three was edited).

### What was built, and the one observation the design rests on

**All three GEMMs are GEMMs the forward already emits.** Not approximately:
the same `ParityGemmHelper.qk` and `.pv`, the same `ParityKvLdsToVgprLoader`
read paths, the same `cast_p` packing, the same `ParityStoreHelper`. What
changes is only *which tensor is staged into which LDS slot*:

| | forward equivalent | A operand | staged as |
|---|---|---|---|
| `S = Q·Kᵀ` | `qk` | K | K layout, `load_k(0)` |
| `dP = dO·Vᵀ` | `qk` | V | K layout, `load_k(1)` |
| `dQ = dS·K` | `pv` | Kᵀ | **V layout**, `load_v(0)` |

§1 predicted the transpose read would be the hard part. It was not, and the
reason is worth stating: **`dS` reaches GEMM3 through `cast_p`, so it carries
the same K permutation `P` does**, and the forward's V transpose read is built
against exactly that permutation. The two line up with no shuffle and no new
lane map. The B0 lane-map probe §1.4 asks for was not needed for dQ — the map
that was already validated is the map this kernel wants. (B1 and B3 may still
need it; this is not a claim that B0 is unnecessary in general.)

Consequence: **the kernel was correct on its first run.** That is not normal
here, and it is the strongest available evidence for the contract's "subclass,
do not port" rule.

### Three LDS tiles in four slots

K is staged **twice** — once in the K layout for GEMM1 and once in the V layout
for GEMM3 — because the two stagings differ in line padding (`SMEM_K_PAD` 8
elements, `SMEM_V_PAD` 32). The forward's allocation already has four slots for
two tiles in flight; this body keeps one tile in flight and spends three,
leaving `(V, buf1)` unused (17 KB at head_dim 128, inside the cap).

That buys **zero new addressing code**: the `m0` tables, the buffer bases and
the alias scopes are the forward's, already validated by
`tooling/probe_kv_staging.py`. It costs 50% more KV DMA traffic than the
algorithm needs, and it is the first thing to attack if a profile asks. Two
routes exist and both are B7 work: read one V-padded staging with a
K-parameterised reader (the two layouts differ only in the line stride), or
double-buffer into the fourth slot and pipeline.

### Measured

`B=4 H=8 S=4096` bf16 non-causal, GPU 6 idle at 39 °C, same session for both
rows so the comparison is same-machine-state:

| head_dim | dQ | forward, same session | ratio | VGPR | AGPR | spills |
|---|---|---|---|---|---|---|
| 64 | 751 TF | 752 TF | 1.00x | 170 | 0 | 0 |
| 128 | 973 TF | 1018 TF | 0.96x | 240 | 0 | 0 |

TFLOP/s counts `6·B·H·Sq·Sk·d` for dQ (three GEMMs) against `4·…` for the
forward (two), i.e. both are MFMA-issue rates and are directly comparable.
**An unpipelined body at the forward's MFMA rate was the surprise.** The
dual-wave schedule buys the forward its rate; dQ reaches the same rate with a
plain `barrier / DMA / wait / barrier / compute` loop, which says the three
GEMMs per tile already cover the DMA latency the forward needed eight clusters
to hide. Do not read that as "pipelining is worthless here" — read it as "the
arithmetic intensity per KV tile is 1.5x the forward's, so the same latency is
easier to hide". Small shapes are launch-bound as expected (115 TF at
`B=1 H=8 S=1024 d=64`, 32 workgroups over 256 CUs).

### The one real finding: **do not pre-scale Q in the backward**

The forward folds `sm_scale * log2e` into Q and rounds the product back to
bf16, which saves a multiply per score. Doing the same here was the first
implementation and it passed at `sm_scale = 1/sqrt(d)`. It fails a scale sweep:

| `sm_scale` | Q pre-scaled | scaled on the f32 scores |
|---|---|---|
| 0.05 | ratio 1.29 | 1.29 |
| 0.25 | 3.02 | 1.64 |
| 1.00 | **10.90** | 1.68 |
| 2.00 | **20.20** | 1.85 |

(ratio = `err(ours, fp64) / err(math-in-bf16, fp64)`, the §7.1 gate.)

The mechanism: rounding `qk_scale·Q` to bf16 puts a `|S|·2⁻⁸` error into the
*exponent*, and `P = exp2(S − lse2)` inherits it as a relative error. The
forward tolerates it because `O` is a normalised average and the error largely
cancels; `dS = P·(dP − delta)` does not normalise and `dQ` sums it over the
whole key axis. A host model of both variants reproduces the kernel's own
numbers to two digits, which is what identifies the Q rounding rather than
anything else — the control the lore keeps asking for.

The fix is free: `fma(S, qk_scale, −lse2)` replaces the `lse2` subtract that
was needed anyway. **AOTriton scales after the dot in both directions**
(`qk += Qk_scale * tl.dot(q, k)`, `p = exp2(qk_scale*qk - l_i)`), so this is
its arithmetic and the forward's fold is the gfx950 schedule's local choice.

Two things follow for the rest of the plan. **B1 should check the same thing**
— `dK = dSᵀ·Q` has the same shape of exposure. And a fudge factor is not a
tuning knob: 10.9 was a real defect and `DQ_FUDGE = 4.0` would have hidden it
at one scale and caught it at another. Sweep the scale.

### dB

`dB = dS`, stored **per element** — 32 `buffer_store`s per lane per KV tile.
Vectorising is available in principle (a lane's 16 scores are four runs of four
contiguous columns) and is not correct in general: a run straddling `seqlen_k`
would have to be partially written and a multi-dword store is all-or-nothing,
so a vectorised version needs a second runtime arm for the tail tile *and* a
`stride_db_seq_q` divisible by 4 (an 8-byte store off a row pitch of 201 elements
is 2-byte aligned).

It costs **4-5x** when enabled, at `B=2 H=8 S=2048`: 55 → 271 us at head_dim 64,
89 → 373 us at 128. Not bandwidth — the tensor is 134 MB, ~27 us at this
board's rate. Off by default, so nobody pays for it unasked, and the two-arm
version is a clean B6/B7 item.

### Verification, and what it caught

- **Error-ratio gate** (§7.1) against the torch math backend at bf16 and fp64,
  four shapes × two head dims, plus GQA and a scale sweep. Observed dQ ratios
  1.0–1.9; the gate is 4.0.
- **Self-consistency with our forward**: the gfx950 forward's own `O` and
  `LSE`, `delta` computed from that `O`, against `torch.autograd.grad`. This is
  the test that would have caught a `log2e` or LSE-layout disagreement, and it
  is why `fmha.lse_row_addressing` is called rather than the layout re-derived.
- **The joint dQ + dK/dV check §6 asks for is live and passing.** B1's kernel
  landed while this was being written; one fp64 autograd call, all three
  gradients inside the gate. The test skips rather than fails if B1's module or
  front end moves, so this suite cannot be turned red by a sibling in flight —
  which also means a green run here is not by itself evidence that the joint
  check ran. Check for the skip.
- **dB** against autograd through an explicit zero bias, including a
  `seqlen_k = 201` case that exercises every tail state of the 4-column runs.
- Structural: BSHD-viewed-as-BHSD strides bit-identical to contiguous, nothing
  written past `seqlen_q`, no NaN in a Q block's padded tail, run-to-run
  bit-identical (the determinism claim §7.3 makes against AITER's atomics).

### Interfaces, and one that had to be reconciled

`delta` and `LSE` are **rank-2 `(batch·heads, tokens)` f32**, checked with the
shared `abi.row_tensor_arg` and read through `fmha.lse_row_addressing`. This
file originally took rank 3 `(B, H, Sq)` — the shape the forward writes — and
was changed to match B1, which had already adopted rank 2 via the shared
helper. Same memory either way; a caller holding `(B, H, S)` passes
`lse.view(-1, S)`. **One ABI mattered more than the nicer shape**, and the
shared checker is the tie-breaker the contract §7 asks for.

Argument order follows the forward: tensors, varlen block, `max_seqlen_*`, the
window pair, philox, head counts, `hdim_qk`/`hdim_vo`, `sm_scale`, then three
strides per tensor. Slots for varlen, windows, philox and bias are present and
ignored.

### Not done, and why

- **Causal, windows, varlen, dropout, bias input, head_dim off {64, 128}.**
  Refused by name in `BwdDqKnobs._with_traits` rather than ignored — each would
  otherwise build, run and return a correctly-shaped wrong answer. B3–B6.
- **Padded heads and asymmetric `hdim_qk`/`hdim_vo`**, also refused, and this
  one is a finding rather than a scope line. **The two head extents are used
  the other way round here than in the forward.** GEMM2 reads V through the
  *K* register path, so `ParityKvLdsToVgprLoader.load_k`'s padded-head mask —
  written against `hdim_qk` — is applied to V's columns; and dQ is written
  through the O store, whose `_final_o_global` suppression is written against
  `hdim_vo` while dQ is `hdim_qk` wide. Both are one line, both are only
  testable once the ladder exists, and B3 should fix them together.
- **No software pipelining and no double buffering.** One tile in flight. At
  0.96–1.00x the forward's MFMA rate the case for the dual-wave schedule here
  is not obvious, and B7 should measure before building it.
- **K staged twice** (above). 50% surplus KV DMA.
- **`dB` per-element store** (above). 4-5x when enabled.
- **`QK_SHARDS` is not implemented**, but nothing forecloses it: §3's
  observation that GEMM1's reduction axis and GEMM3's output axis are the same
  axis holds here, and the kernel indexes the D axis through `D_CHUNKS` /
  `K_STEPS_QK` everywhere rather than through literals.

### One hazard did *not* fire, and the reason is worth keeping

§1.3 and the contract flag the ISA requirement that **EXEC be all 1s across
`ds_read_b64_tr_b16`**. This kernel has exactly one transpose read site and it
is not inside any `scf.if`: the only branchy region in the loop is
`seq_pad_mask_if_needed`, which touches no LDS, and the `active` guard is
`None` for a dense non-causal build. B4 and B5 turn `active` on, and that guard
is workgroup-uniform (so EXEC is all-1s or the block does not run) — but a
*causal* build that puts a masked region around an edge tile would be the first
thing here to violate it. Check it there rather than assuming it inherits.

---

## Outcome: B1 — dK/dV *(dense, non-causal, head_dim 64/128, bf16, MHA)*

Files: `fmha_bwd_dkdv_gfx950.py`, `fmha_tuning_bwd_dkdv_gfx950.py`,
`test_fmha_bwd_dkdv_gfx950.py`. 46 tests, all passing. Nothing outside those
three files was edited; the forward's own suite still passes, 329/329 in
12m31s. (The brief said 383 — the file collects 329 on this checkout. Zero
failures either way.)

### The design, and the two things that made it small

**dK/dV is the forward with the roles swapped, and the swap is four
descriptors.** K and V stay resident in registers, Q and dO stream through the
LDS slots the staging machinery calls K and V. The traits' `BLOCK_M` becomes
the KV block and `BLOCK_N` the streamed Q tile; nothing derived from either
changes meaning, because the staging is about *64 rows through LDS* and does
not care whose rows they are.

**The MFMA's A and B operands take the same per-lane layout** — 32 outer rows
on `lane % 32`, 16 contraction elements from `lane // 32` and the element index
— so the forward's K reader and its Q loader produce interchangeable packs, and
`S = Q·Kᵀ` / `dP = dO·Vᵀ` are `DualwaveGemmHelper.qk` with the two operands
swapped. Output is `[A's row][B's row]` with B's row on `lane % 32`.

**§1's bet paid, and B0's lane map was again not needed.** The two q-contracted
GEMMs read the *same LDS bytes* the row-major GEMMs read, through the forward's
V transpose path, unmodified. That works because the transposed read hands back
element order `[0,1,2,3,8,9,10,11]` on the q axis and `_pack_p_v8_slices` slices
the accumulator in exactly that order — the same coincidence the forward relies
on one axis over. So: **two LDS tiles, and both are read two ways.** The one
requirement is that both tiles are staged in the **V** line stride
(`SMEM_V_PAD`, 32 elements), since the transpose path is only validated against
that; the row-major read does not care which stride it is handed, so it is
`_k_read_base` with `STREAM_LINE_STRIDE` substituted.

Like B2, the kernel was **correct on its first run**. Two independent phases
saying that about "subclass, do not port" is worth more than either saying it
once.

### Measured

`B=4 H=8 S=4096` bf16 non-causal, GPU 5 idle at 34 °C. TFLOP/s counts
`8·B·H·Sq·Sk·d` (four GEMMs), so it is an MFMA-issue rate directly comparable
to the forward's `4·…` and B2's `6·…`.

| head_dim | waves | BLOCK_KV | dK/dV | VGPR | AGPR | spills | LDS |
|---|---|---|---|---|---|---|---|
| 64 | 4 | 128 | **692 TF** | 240 | 0 | 0 | 34 KB |
| 128 | 2 | 64 | **789 TF** | 364 | 108 | 0 | 68 KB |

`ds_read_b64_tr_b16` count equals the MFMA count exactly (64:64 and 128:128 per
tile pair), which is AITER's tuned `bwd_hd128` ratio — evidence the operand path
is the intended one rather than merely a working one.

### The AGPR cliff arrives one width earlier than the forward's

This is the finding worth carrying into B3. A dK/dV wave holds **two**
accumulators, so at head_dim 128 they are 128 VGPRs before anything else is
live. At 8 waves (2 per SIMD) a wave may address 256 registers *in total*, so
the allocator cannot reach the AGPR file at all:

| head_dim | waves | `waves_per_eu` | AGPR | spills | TFLOP/s |
|---|---|---|---|---|---|
| 128 | 8 | 2 | 0 | 118 | 444 |
| 128 | 4 | 2 | 0 | 126 | 403 |
| 128 | 4 | **1** | — | — | 721 |
| 128 | **2** | 2 | **108** | **0** | **788** |
| 64 | 8 | 2 | 0 | 0 | 657 |
| 64 | **4** | 2 | 0 | 0 | **698** |

1.8x between the same code at 8 waves and at 2, entirely on whether the AGPR
file is reachable. The lore's "check `agpr_count` alongside spills" is exact
here, and note the third row: at 4 waves the *only* thing standing between 403
and 721 TF is `waves_per_eu`, i.e. a scheduling hint deciding a register
budget. §3's table said dK/dV needs sharding from 256; on this evidence it
needs a *wave-count* answer from 128, which is cheaper and should be tried
first.

head_dim 64 prefers 4 waves over 8 by 5%, which is inside the band §B7 says a
sweep cannot decide — confirmed by interleaved single-GPU A/B, nine reps,
360.5 us median against 379.3 with non-overlapping min/max.

One structural change was worth 2.2x on its own before any of that: reading the
two 32-row halves of a staged tile **separately** rather than as a pair (each
half is `K_STEPS_QK * 4` VGPRs = 64 at head_dim 128). 200 → 444 TF.

### `dS` in bf16 is the whole of the error, and it is flat

Every shape measures the same ratio: `err(ours, fp64) / err(bf16-math, fp64)` =
**1.35–1.46**, 2.3e-3 against 1.7e-3, unmoved by batch, heads, sequence length
or head dim. That flatness identifies it as one systematic extra rounding — `P`
and `dS` truncated to bf16 before the q-contracted GEMMs, where torch keeps an
fp32 intermediate — rather than anything that accumulates. The gate is 2.0.

### B2's Q-prescaling finding reproduces here, and this kernel does not do it

B2's outcome asks B1 to check the same exposure. It is real and this kernel
avoids it: Q is **not** pre-scaled, and `P = exp2(fma(S, qk_scale, −lse2))`
folds the scale into the subtraction that had to happen anyway. Swept
`sm_scale` over 200x, ratios flat at 1.36–1.55 on both rungs. A host model of
the two variants, everything fp32 except the one rounding, is the control:

| `sm_scale` | Q pre-scaled, dK | scaled on the f32 scores, dK |
|---|---|---|
| 0.125 | 2.41e-3 | 1.64e-3 |
| 1.0 | 1.42e-2 | 1.66e-3 |
| 4.0 | **5.58e-2** | 1.64e-3 |

So the forward's Q fold is a forward-only optimisation twice over, and
AOTriton's `p = exp2(qk_scale*qk - l_i)` is the arithmetic both backward
kernels should keep.

### Nothing is masked, and that is a property rather than an omission

Dense and non-causal leaves only the ragged tail, and the buffer descriptors
answer it: Q and dO are bounded at `seqlen_q`, so a staged row past the end
reads zero; K, V, dK and dV at `seqlen_kv`. A padding q row gets `S = 0` and
`LSE = 0` (also out of its resource), hence `P = exp2(0) = 1` — nonzero, and it
still contributes exactly nothing, because `dV += P·dOᵀ` and `dK += dSᵀ·Q` both
multiply by a zero staged tensor. The tile count is rounded **up** to an even
number for the two-buffer loop for the same reason: the extra tile is inert.

The consequence for §1.3's EXEC hazard: there is no `scf.if` anywhere in this
body, so the transpose reads cannot reach a divergent region. B4 is where that
stops being free — it is the first phase that wants a branch around an edge
tile — and the guard should be tested there rather than assumed to inherit.

**The LSE and delta reads are four contiguous rows starting at a multiple of
four**, because the accumulator's row map is `8·(r//4) + 4·(lane//32) + (r%4)`.
Four `buffer_load_dwordx4` per accumulator half, and a group can never straddle
`seqlen_q` unless `seqlen_q % 4`, which is tested (65, 101, 103, 67, 2, 1).
`_score_column_runs` is where that grouping is stated; it now has two consumers
(the forward's bias reads and this), which is §4's rule earning its keep.

### Verification

- **Error-ratio gate** (§7.1) against torch's math backend at bf16 and fp64,
  four shapes × two rungs, ten ragged/asymmetric sequence pairs, and a
  `sm_scale` sweep.
- **Self-consistency with our forward**: the gfx950 forward's own `O` and
  `LSE`, `delta` from that `O`, against `torch.autograd.grad`.
- **The joint dQ + dK/dV check §6 asks for is live and passing** against B2's
  kernel — one fp64 autograd call, all three gradients inside the gate. It
  `importorskip`s B2's module, so a green run here is not by itself evidence
  that it ran; check for the skip.
- Structural: BSHD-viewed-as-BHSD bit-identical to contiguous (all six stride
  triples at once), nothing written past `seqlen_kv` into an over-allocated
  output, run-to-run bit-identical, and every unimplemented mode refused rather
  than approximated.

### A degenerate case the ratio gate cannot express

`seqlen_k == 1` makes `dK` **analytically zero** — a one-key softmax gives
`p = 1`, so `dp == delta` and `dS = p·(dp − delta)` cancels exactly. The fp64
reference norm is `0.0`, the bf16 backend's error is `0.0`, and no multiple of
zero admits a finite answer. Ours is 9.4e-6 in norm against a `dV` norm of 285,
i.e. 3.3e-8 of the problem. Handled the way gfx1201's suite handles the same
shape of thing (`window=(0,0)` reaches it there): a floor under the denominator
taken from the *other* gradient, plus a 1e-5 additive term that only ever
decides the degenerate case. It is not a kernel finding, but it is the second
place this cancellation has produced a meaningless ratio, so it is worth
expecting a third.

### Not done, and why

- **GQA.** `num_heads_q != num_heads_k` needs dK/dV **summed** over the q heads
  sharing a kv head, and one workgroup owns one `(q head, kv block)` and would
  write rather than accumulate. Refused host-side; the natural fix is a loop
  over the group inside the kernel, with K/V resident throughout, and it is a
  B3-sized change rather than a B1 one.
- **The ladder, padded heads, asymmetric `hdim_qk`/`hdim_vo`.** B3. `_args`
  refuses anything but an exact rung.
- **Causal, windows, varlen, dropout, bias.** B4–B6. The ABI carries every
  argument slot so the wire format does not move, and `resolve` raises on each.
- **No software pipeline.** The body is `wait / barrier / read / compute /
  barrier / prefetch`, one tile of prefetch distance, two barriers per tile.
  The dual-wave schedule's eight clusters are not ported. B2 observed that its
  three GEMMs per tile already cover the DMA latency; with four this is more
  true, and the 789 TF at head_dim 128 is against a forward that gets ~1117 in
  its own suite — so there is a gap, and pipelining is the obvious place to
  look for it, but it is B7 work and it should be measured before it is built.
- **`BLOCK_Q` is pinned at 64** by the transpose read covering four 16-row
  k-substeps. A 128-row streamed tile would need eight, which is describable
  but is a second geometry to validate; not attempted.

---

## Outcome: B3/dQ — the ladder to 512 *(dense, non-causal, bf16)*

**Every rung of `LADDER` (32 … 512) is built, correct and tested**, with the
8xD input contract, padded heads and asymmetric `hdim_qk`/`hdim_vo`. 169 tests
pass with no skips, which means the joint dQ + dK/dV autograd check ran at all
ten rungs. Nothing outside the three dQ files was edited.

### The LDS blocker: one region for K, not two

B2 staged K twice — K pitch for GEMM1, V pitch for GEMM3 — because the two
readers disagree only about `SMEM_K_PAD` (8 elements) vs `SMEM_V_PAD` (32).
Three regions is 199 KB at head_dim 512 against a 163840 B cap, so this was a
blocker rather than the B7 optimisation B2 called it.

The fix is to stage K **once, in the V layout**, and re-point the *K path* at
the V pitch:

    (V region) <- K   -> load_v(0)      GEMM3, the stock transpose read
                      -> the K path on V-pitch lines   GEMM1
    (K region) <- V   -> the K path, stock             GEMM2

`BwdDqKvLdsToVgprLoader._read_k_packs` does that by `replace`-ing two trait
fields (`SMEM_K_LINE_STRIDE` and the k-step outer stride) around the shared
formula, so nothing is transcribed. **Which reader moves is deliberate**: the K
path is a plain `llvm.LoadOp` with two address expressions, while the transpose
path is `ds_read_b64_tr_b16` with alias scopes, an even-VGPR-pair constraint
and two open hazards against it. Leave the fragile one stock.

`LDS_KV_TOTAL_SIZE` then drops to one buffer — one trait field in
`make_bwd_dq_traits` — and the fourth region is no longer allocated:

| head_dim | 256 | 384 | 512 |
|---|---|---|---|
| B2, three regions | 99 KB | 149 KB | **199 KB, over cap** |
| B3, two regions | 66.5 KB | 99.8 KB | **133.0 KB** |

No `D_STAGES` anywhere on the ladder, which is the substantive divergence from
the forward's family W.

### Measured, per rung

`B=2 H=8 S=2048` bf16 non-causal, GPU 6 idle, `6·B·H·S²·d` FLOPs:

| head_dim | waves | BLOCK_M | granule | LDS | TFLOP/s | build |
|---|---|---|---|---|---|---|
| 32 | 4 | 128 | 32 | 8.3 KB | 420 | 1.2 s |
| 64 | 4 | 128 | 64 | 16.6 KB | 580 | 1.2 s |
| 96 | 4 | 128 | 32 | 24.9 KB | 667 | 1.3 s |
| 128 | 2 | 64 | 64 | 33.2 KB | **732** | 1.4 s |
| 160 | 2 | 64 | 32 | 41.6 KB | 678 | 1.5 s |
| 192 | 2 | 64 | 64 | 49.9 KB | 663 | 1.5 s |
| 224 | 2 | 64 | 32 | 58.2 KB | 676 | 1.6 s |
| 256 | 2 | 64 | 64 | 66.5 KB | 698 | 1.7 s |
| 384 | 4 | 128 | 64 | 99.8 KB | **340** | 1.9 s |
| 512 | 4 | 128 | 64 | 133.0 KB | **113** | 2.3 s |

Every build is under 2.5 s, so the 8-minute cap never bound. head_dim 128 at
732 TF is up from B2's 496 at the same shape, entirely from the geometry.

### `waves_per_eu = 1`, and the discriminator that says why

The full `(num_waves, waves_per_eu)` grid is in
`BwdDqKnobs._with_wave_geometry`'s docstring. The headline reproduces B1's:
**`waves_per_eu = 1` is never worse and is worth up to 4x.** head_dim 256 at
four waves is 675 TF at 1 and 169 at 2, and the ISA dumps separate the two
exactly as the addendum predicts:

| build | VGPR | AGPR | spills | scratch | TF |
|---|---|---|---|---|---|
| 256, w4, wpe 2 | 256 | **0** | **191** | 768 B | 169 |
| 256, w2, wpe 1 | 460 | 204 | 0 | 0 | 712 |

Zero AGPRs with a nonzero spill count is not a register shortage, it is a
prohibition. The forward's default of 2 is wrong for this kernel at every rung
measured, so `_BWD_DQ_FALLBACK` deliberately does not carry it.

Wave count: 4 below head_dim 128, 2 from 128 to 256, 4 above. Several
neighbouring points differ by under 10%, which the lore says a sweep cannot
settle; those are left where they fall rather than tuned.

### 384 and 512 are register-bound, and 512 is *structurally* so

**Flat across the entire (waves, wpe) grid** — 335–339 TF at 384 and 113–117
at 512, at every one of six points. Not occupancy. The ISA says why:

| head_dim | VGPR (total) | AGPR | spills | scratch |
|---|---|---|---|---|
| 384 | 512 | 256 | 112 | 452 B |
| 512 | 512 | 256 | **546** | 1804 B |

Both saturate the 512-register unified file and still spill. The accounting is
arithmetic, at 32 rows per wave:

    Q packs     head_dim / 4      128 at d=512, loop-invariant
    dO packs    head_dim / 4      128
    dQ acc      head_dim / 2      256
    ------------------------------------------------------
                                  512 = the entire file, before a single operand

So **head_dim 512 cannot be made to fit by scheduling**: the loop-invariant
operands plus the accumulator are the whole register file on their own, and the
three streams of K/V/Kᵀ packs (256 each if fully materialised) have nowhere to
go. At 384 the same three terms are 384, leaving 128 — enough for a *streamed*
operand (one pack pair at a time) plus the score accumulators, which is why 384
plausibly is fixable and 512 is not.

Two levers, in the order they should be tried:

1. **Stream the A operands** — fuse the LDS read into the MFMA loop so one pack
   is live instead of `K_STEPS_QK` of them. Local, and it should recover 384.
   Not attempted here: it has to work under the padded-head mask, which is
   per-k-step, and B3's remaining budget went to the correctness gates.
2. **`QK_SHARDS`** for 512, which is what plan §3 already predicts fits dQ
   neatly — GEMM1's reduction axis and GEMM3's output axis are the same axis,
   so one shard offset serves Q, dO *and* dQ and halves all three terms above
   (64 + 64 + 128 = 256, leaving 256 for operands). It needs the cross-shard S
   reduction through LDS that the forward declined to build. This is the real
   answer at 512 and it is a phase of its own.

Neither is a *correctness* gap: both rungs pass the error-ratio gate.

### The near miss, and the guard it produced

`Gfx950Knobs._with_d_axis_splits` turns `d_stages = 2` on at block_dmodel > 256
— exactly the rungs B3 added — and under `D_STAGES > 1` the inherited
`ParityGemmHelper.qk` silently becomes `qk_stage(..., stage=0)` while `pv`
writes only the first stage's chunks. This loop never advances the stage, so
384 and 512 would have reduced over **half the head dim**, written half the
accumulator, and returned a finite wrong answer.

It was caught by an LDS figure that was half what it should have been, and by
nothing else — the error-ratio sweep that produced the table above ran against
an earlier override that happened to pin `d_stages = 1`. `BwdDqKnobs` now
overrides `_with_d_axis_splits` *and* refuses a pinned `d_stages`/`qk_shards`/
`vo_shards` by name, because relying on a default to hold is what nearly went
wrong. `test_d_axis_splits_are_refused`.

### Padded heads: the two crossed extents, fixed and tested

B2's outcome named them; B3 spends them.

- **GEMM2 reads V through the K register path**, whose padded-head mask reads
  `self.hdim_qk`. There are now two `BwdDqKvLdsToVgprLoader` instances, one per
  tile, differing in the LDS pitch *and* the extent — plus `HDIM_VO_FLOOR`, the
  vo counterpart of `HDIM_QK_FLOOR`, which drops to 0 when `head_dim_v` sits
  below the rung's floor (`head_dim 128, head_dim_v 40`).
- **dQ is stored through the O path**, whose suppression reads `hdim_vo` while
  dQ is `hdim_qk` wide. `BwdDqStoreHelper` rebinds the attribute.

Both coincide in every symmetric build, so nothing before `test_asymmetric_hdim`
could tell the fix from its absence.

**And a third one, which was a genuine silent wrong answer.** A build resolved
without `head_dim_v` gets `padded_head=False` and emits *no* D-axis mask, so a
V narrower than the tile is reduced over the caller's slack — finite, right
shape, 0.70 relative error. This file's own test helper made exactly that
mistake. The guard is now in `_args`: a non-padded build requires
`hdim_qk == hdim_vo == BLOCK_DMODEL`. A real caller can make the same mistake,
so it belongs in the kernel and not in the test that found it.

### Also fixed: head_dim 32 crashed

`D_CHUNKS == 1` is the one rung where the loop carries a single value, and an
`scf.for` with one result hands it back **unwrapped**. `loop_results[0]` then
indexes a `vector<16xf32>` and returns an f32, which surfaces two frames away
inside `_scale_o_accs` as "Cannot cast type to VectorType". `_carried()`
normalises it. The forward never sees this because it always carries `m_row`
and `l_row` alongside.

### Gates

- Full `LADDER`, square and ragged, error-ratio gate per rung
  (`test_ladder_error_ratio`). Observed ratios 1.0–1.9 against a 4.0 gate.
- **8xD contract**: every multiple of 8 from 8 to 512, plainly contiguous, with
  an overrun canary row (`test_grid8_contiguous_is_exact_and_writes_nothing_past_dq`).
- Padded heads with a **poisoned** pad, asymmetric hdim including the
  floor-fallback case, and a store-suppression canary.
- Rows-per-wave ceiling asserted at both layers — `make_bwd_dq_traits` raises,
  and the knob-level geometry list rejects the same configuration earlier.
- Wave geometries: every entry `BwdDqKnobs._SUPPORTED_GEOMETRIES` adds over the
  forward's is *run*, not asserted.
- Joint dQ + dK/dV against one fp64 autograd call, at all ten rungs.
- The forward's suite unchanged; no file outside the three dQ files was edited.

### dB does not move the ceiling, but it costs at every rung

The rung table above is the `store_db=False` arm. Measured as a matrix, same
shape, with the store enabled at every rung:

| head_dim | dB off, VGPR/AGPR/spill | dB on, VGPR/AGPR/spill | cost |
|---|---|---|---|
| 32 | 400 TF, 140/0/0 | 73 TF, 174/0/0 | 5.5x |
| 64 | 555, 190/0/0 | 136, 224/0/0 | 4.1x |
| 96 | 659, 236/16/0 | 193, 272/16/0 | 3.4x |
| 128 | 713, 260/4/0 | 256, 330/74/0 | 2.8x |
| 160 | 666, 312/56/0 | 291, 383/127/0 | 2.3x |
| 192 | 695, 360/104/0 | 339, 435/179/0 | 2.0x |
| 224 | 694, 412/156/0 | 358, 483/227/0 | 1.9x |
| 256 | 698, 460/204/**0** | 319, 512/256/**14** | 2.2x |
| 384 | 334, 512/256/112 | 137, 512/256/288 | 2.4x |
| 512 | 113, 512/256/546 | 83, 512/256/802 | 1.4x |

**Every rung builds and is correct with dB on, 512 included**, and the dQ error
is bit-for-bit the same on both arms. So dB is a cost, not a ceiling, and no
rung has to be withdrawn for it. Three things the matrix says that the single
line did not:

- **dB adds a flat ~70 registers at every rung**, not a proportional amount: it
  is the `dS` tile plus the store's address arithmetic, and neither scales with
  `d`. That is enough to tip head_dim **256 from zero spills into 14**, which
  is the rung where the ladder's register headroom actually runs out.
- **The relative cost is worst at the narrow end** (5.5x at 32, 1.4x at 512),
  because the store count is `32 per lane per KV tile` regardless of `d` while
  the GEMM work is linear in it. A vectorised store would help the small rungs
  most, which is the opposite of where the register pressure is.
- The earlier "4-5x" figure came from head_dim 64/128 only and was not
  representative of the ladder.

**The lever that helped was the live range, not the accumulator.** `dS` exists
twice -- 32 f32 scores, and after `cast_p` 16 packed bf16 -- and the store now
reads the packs, so the f32 form dies at `cast_p` instead of living across the
32-store sequence. Three placements measured on the rungs that spill:

| variant | 256 | 384 | 512 |
|---|---|---|---|
| f32 lists, before `cast_p` | 280 | **149** | 56 |
| packs, after `cast_p` *(kept)* | **319** | 137 | **83** |
| packs, after `pv` | 305 | 146 | 73 |

Below head_dim 256 the three are register-identical, so the allocator was
already sinking the f32 form and there was nothing to win. At 512 the pack
source is **1.49x**. 384 prefers the f32 form by 9%, which the lore says a
sweep cannot settle and which is an allocator outcome rather than a mechanism
-- the spill counts are not monotone with the rate in any column. Kept the
variant whose one decisive measurement agrees with the mechanism.

Not tried: interleaving the store into GEMM3's `pv` loop so each pack dies
right after its MFMAs. That is the sharper version of the same lever and it is
the next thing to measure if dB's cost at 384/512 matters.

### Still not done

- **384 at half rate and 512 at a sixth** with dB off, above; both worse again
  with dB on. The two levers are streaming the A operands (384) and
  `QK_SHARDS` (512), in that order.
- **`dB`'s store is still per element** -- 32 per lane per KV tile. Vectorising
  needs a runtime tail arm and a 4-divisible `stride_db_seq_q`; it would help the
  narrow rungs most, where the cost ratio is worst.
- **No pipelining, one tile in flight.** At 128 the body reaches 732 TF against
  the forward's ~1018 on the same board for two GEMMs; whether a dual-wave
  schedule closes that is a B7 measurement.
- Causal, windows, varlen, dropout and bias input remain refused by name.

---

## Outcome: B3 — dK/dV on the full ladder *(dense, non-causal, MHA, bf16)*

`LADDER` is the forward's entire `(32 … 512)`, with the 8xD input contract and
padded heads. 134 tests, all passing, no skips -- so the joint dQ + dK/dV
autograd check ran. The forward's own suite still passes untouched (329/329 in
12m31s on this checkout).

**Every rung was correct on its first run**, including 384 and 512. Three
phases in a row have now said that about "subclass, do not port"; at some point
it stops being luck.

### Measured, per rung

`B=4 H=8 S=4096` bf16 non-causal, GPU 5 idle. TFLOP/s counts **nominal**
`8·B·H·Sq·Sk·d` -- the duplicated S and dP a shard recomputes are *not*
credited, so the sharded rungs' MFMA-issue rate is higher by the factor in the
last column.

| head_dim | gran | waves | wpe | shards | buf | BLOCK_KV | tight | VGPR | AGPR | spills | build | TFLOP/s | ×dup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 32 | 4 | 2 | 1 | 2 | 128 | yes | 150 | 0 | 0 | 0.7 s | **508** | 1.0 |
| 64 | 64 | 4 | 1 | 1 | 2 | 128 | no | 232 | 0 | 0 | 0.8 s | **713** | 1.0 |
| 96 | 32 | 4 | 2 | 1 | 2 | 128 | no | 256 | 0 | 4 | 1.0 s | **805** | 1.0 |
| 128 | 64 | 2 | 2 | 1 | 2 | 64 | no | 330 | 74 | 0 | 1.2 s | **744** | 1.0 |
| 160 | 32 | 4 | 2 | 1 | 2 | 128 | no | 356 | 100 | 0 | 1.3 s | **751** | 1.0 |
| 192 | 64 | 4 | 1 | 1 | 2 | 128 | no | 488 | 232 | 0 | 1.4 s | **636** | 1.0 |
| 224 | 32 | 4 | 1 | 1 | 2 | 128 | no | 486 | 230 | 0 | 1.6 s | **737** | 1.0 |
| 256 | 64 | 4 | 1 | 2 | 2 | 64 | no | 424 | 168 | 0 | 1.4 s | **451** | 1.5 |
| 384 | 64 | 4 | 1 | 2 | 1 | 64 | yes | 512 | 256 | 31 | 1.7 s | **283** | 1.5 |
| 512 | 64 | 4 | 1 | 4 | 1 | 32 | yes | 512 | 256 | 38 | 1.8 s | **260** | 2.5 |

The error ratio against the math backend is **1.41–1.48 at every rung**, flat in
width exactly as it was flat in shape at B1. Build times peak at 1.8 s; nothing
in the whole sweep came near the eight-minute cap, so that gate never fired.

### LDS never binds, and single-buffering is what buys 384 and 512

The addendum's arithmetic held: a staged slot is `68 · head_dim` elements, so
two tensors single-buffered are `272 · head_dim` bytes and head_dim 512 fits in
139264 of the 163840 cap **with a whole tile of Q and of dO resident**. The
second stream buffer is the only thing LDS ever costs, and dropping it at 384
and 512 is the entire LDS story. **`D_STAGES` was never needed and is not
implemented here.**

Dropping the buffer costs the prefetch distance, not a code path:
`NUM_STREAM_BUFFERS` is a number the tile loop reads, and at 1 the DMA for the
next tile is issued at the end of this one and waited on immediately.

### The lever the addendum did not name, and it dominates above 128

`(num_waves, waves_per_eu)` was first and it worked as B1 said. The **second**
lever turned out not to be sharding but `BLOCK_KV`, and it is bigger:

| head_dim | waves | BLOCK_KV | TFLOP/s |
|---|---|---|---|
| 160 | 2 | 64 | 408 |
| 160 | **4** | **128** | **690** |
| 224 | 2 | 64 | 486 |
| 224 | **4** | **128** | **723** |

Same registers per wave -- 4 waves is still one per SIMD -- and 1.7x. The
reason is that **every workgroup streams the whole of Q and dO for its head**,
so the total read traffic is `seqlen / BLOCK_KV` copies of that slab. Raising
the wave count at fixed shards raises `BLOCK_KV` and halves the traffic for
free. That is why every rung but 128 lands on 4 waves, and why 8 is worse
everywhere it was tried (256: 198 TF against 451; 384: 91 against 283; 512: 81
against 260) -- 8 waves crosses back over the AGPR cliff.

It also reframes `DKV_SHARDS`: sharding *divides* `BLOCK_KV` at a fixed wave
count, so it pays the traffic lever back at the same time as it buys
accumulator space. That is most of why 256 (451 TF at 2 shards) sits below its
unsharded neighbours 224 (737) and 192 (636) rather than between them.

### `DKV_SHARDS` is `VO_SHARDS`, and that was the cheap part

D is an *output* axis for both accumulators, so shards write disjoint columns
and never have to agree on anything -- no cross-wave reduction, no extra
barrier, no summation order. Exactly the forward's `VO_SHARDS` bargain, and the
implementation reuses the field itself rather than adding one, so
`make_traits`' shard derivation and its even-chunk validation came for free.
Two sites know about it: the transposed read folds the shard's first chunk into
`stream_col_read_base` (the forward's `load_v_shard` trick, and it needs the
even-chunk rule to be legal), and the store adds the shard's column origin.

### `TIGHT_REGISTERS`: one knob for a real crossover

Two choices in the tile body hold the same thing live and cost the same thing:
whether a staged half is read pack-by-pack into its MFMA or all at once, and
whether the tile's two 32-row halves go through the softmax together. They are
one knob because they trade the same currency -- live f32 against the
scheduler's freedom to overlap an MFMA burst.

| head_dim | loose | tight |
|---|---|---|
| 32 | 472 TF, 0 spills | **511**, 0 |
| 128 | **744**, 0 | 606, 0 |
| 224 | **723**, 0 | 645, 0 |
| 256 | 148, 246 spills | **375**, 166 |
| 384 | 197, 368 spills | **283**, 31 |
| 512 | 213, 294 spills | **261**, 38 |

The crossover is at 384, where the accumulators stop leaving room. head_dim 32
is the odd one at the narrow end and is not a register story: with
`D_CHUNKS == 1` there is barely any independent work for the loose arm to
overlap, so all it does is lengthen live ranges.

### A negative result worth recording: KV-block-fast grid order

Since every workgroup streams the whole of Q and dO, putting the **KV block**
on the fast grid axis so that concurrently-issued workgroups share that slab is
the obvious move. It is **12–15% slower at every rung tried** -- 512: 230 TF
against 260; 384: 260 against 283; 256: 390 against 433. The eight XCDs have
separate L2s, so sharing one slab duplicates it across all of them instead of
spreading distinct work over them. Same conclusion as the forward's head-fastest
choice, for the opposite reason. The knob was measured and then **deleted**; the
grid order is a literal again, with the measurement in the comment.

### Padded heads: the cheap mask is the other one

The forward masks Q once in its prologue and K on the hot path, and measured
27–54% for the second. Here the roles are swapped, so the cheap side is **K and
V** -- resident, masked once, every k-step -- and Q and dO are the hot ones,
masked only on the steps `HDIM_QK_FLOOR` cannot rule out (two of thirty-two at
head_dim 512).

Only the two *d-contracted* GEMMs need masked operands at all. In `dV` and `dK`
the head dim is the **output** axis, so a pad column of dO can only reach a pad
column of dV and the store suppresses it by address. Asymmetric
`hdim_qk`/`hdim_vo` is supported and the two extents are used the way round
they read: K and dK against `hdim_qk`, V, dO and dV against `hdim_vo`.

All 64 multiples of 8 from 8 to 512 pass with plainly contiguous tensors, each
with a sentinel row contiguous with the last real one on **both** outputs --
the suppression is written twice and a canary on one would not see the other.
A tight odd head_dim is refused rather than corrupted, including the BSHD case
where the pitch check alone would wave it through.

### head_dim 96 does not need the forward's Hazard 2 workaround

`ParityGemmHelper.qk` carries an `s_nop 1` because head_dim 96 computes a wrong
answer without it, mechanism still unknown. This kernel has no such nop and
head_dim 96 is correct at ratio 1.437 across three shapes. That is a data
point, not an explanation: the hazard is a register-allocation coincidence, and
a different body places registers differently. It does say the defect is not
inherent to granule-32 staging or to `ds_read_b64_tr_b16` counts, both of which
this kernel has at 96.

### The rows-per-wave ceiling is enforced, and the lever behind it is *not* built

`_with_traits` raises unless `ROWS_PER_WAVE == 32`, and `BLOCK_KV` is *derived*
from `32 · waves / shards` rather than pinned beside the wave count, so the two
cannot disagree. That is P7's twelve silently-wrong configurations answered.

**But the addendum's lever 2 -- rows per wave 32 → 16 -- is not implemented**,
and this is the honest gap in B3. It needs `v_mfma_f32_16x16x16`, whose v4f32
accumulator is a different register layout through every helper: the score
accumulator, the `8·(r//4) + 4·(lane//32) + (r%4)` row map the LSE and delta
reads are grouped by, `_pack_p_v8_slices`, the `permlane32_swap` store, and a
**re-derived transpose-read lane map** -- the existing one delivers `m = lane %
32` and 16-row A wants `m = lane % 16` with the second 16 lanes carrying
different *tokens* rather than different columns. That is B0's probe, for real
this time. It is a second family, not a parameter, and `DKV_SHARDS` reached the
same accumulator relief without new operand algebra.

It is also the *better* answer and should be built before anyone tunes 512
further. At 16 rows per wave the per-wave loop invariant is `0.75·d` -- resident
`d/4`, accumulators `d/2` -- which is 384 at head_dim 512, the same as four-way
sharding gives. But it gets there with **no duplicated S/dP** (against 2.5x) and
with `BLOCK_KV = 16 · waves = 64` (against 32), i.e. half the Q/dO traffic. Both
of the two things that make 512 slow.

### Not done

- **Causal, windows, varlen, dropout, bias input, GQA.** Refused by name; the
  ABI carries every argument slot.
- **16 rows per wave**, above.
- **No software pipeline**, one tile in flight, two barriers per tile.
- **The wide rungs are duplicated-work- and traffic-bound, not spill-bound.**
  384 and 512 still spill 31 and 38, which is small and did not respond to any
  remaining knob; the 2.5x MFMA duplication and the 32-row `BLOCK_KV` are the
  cost, and both have the same fix (previous section).
- **`BLOCK_Q` is pinned at 64** by the transpose read covering four 16-row
  k-substeps. 128 would need eight -- describable, a second geometry to
  validate, not attempted.
