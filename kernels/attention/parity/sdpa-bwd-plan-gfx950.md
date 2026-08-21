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
wide path starts at 384. **dK/dV needs sharding from 256**, and 192 will be
marginal.

**This table is a function of the tiling choice, not a constant.** At 16 KV
rows per wave the accumulators are `[d][16]` and cost `d/4` each, halving every
number above and pushing the wide threshold *later* than the forward's rather
than earlier — paid for by halving `BLOCK_N` at a given wave count, so the same
work needs more workgroups or more waves. That trade is precisely what AITER's
`ts_qo` knob selects, and it is the first thing B4 should sweep. Read the split
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

### B3 — causal and windows

`decompose_causal_regions` with the axes swapped for dK/dV; the forward's
two-sided guard for dQ. The forward's P3 lesson applies directly: the
**bit-identical sentinel oracle** (a window build fed `WINDOW_BOTRIGHT` must
reproduce plain causal exactly) is the sharpest test available, and the tile
cut must be verified by *timing*, because a dead tile is a no-op.

Watch for the literal-zero tile base: P3 found four instances of it in the
forward, and the backward has the same shape of prologue.

### B4 — the head_dim ladder and the wide body

Extend to the forward's `LADDER` (32…512), adding `D_STAGES` and d-axis
sharding for dK/dV per §3. This is where the 8-minute build rule bites; expect
the wide body to be a separate file, as `fmha_wide_gfx950.py` is.

*Gate:* the 8xD input contract, the padded-head path, and — from P7 — the
rows-per-wave ceiling enforced rather than commented.

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
and can proceed in parallel if there are two people; B3–B6 are ordered,
because each phase's tests are what make the next one's failures legible. B7 is last, after the register
budget has settled — which is the same reason the forward's sweep was P7.

**Done** is: dK/dV and dQ on the full `LADDER`, with causal, windows, varlen,
dropout, bias and padded heads, passing the §7.1 error-ratio gate against the
math backend, at a throughput recorded honestly against the MFMA ceiling.

Two of those go past our gfx1201 backward rather than merely matching it:
**`dB`**, which AOTriton has and that port dropped, and **`VarlenBits`**, which
AOTriton does not have at all.
