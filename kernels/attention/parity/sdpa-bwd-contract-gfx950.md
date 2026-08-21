# gfx950 backward: the shared contract

Two kernels are being written in parallel — **dK/dV** (B1) and **dQ+dB** (B2).
They must look like one codebase and like the forward, not like two ports that
met in the middle. This is the agreement; where it and personal taste disagree,
this wins.

Read `sdpa-bwd-plan-gfx950.md` first for *what* and *why*. This is *how*.

---

## 1. Scope split

| | dK/dV (B1) | dQ+dB (B2) |
|---|---|---|
| file | `fmha_bwd_dkdv_gfx950.py` | `fmha_bwd_dq_gfx950.py` |
| tuning | `fmha_tuning_bwd_dkdv_gfx950.py` | `fmha_tuning_bwd_dq_gfx950.py` |
| test | `test_fmha_bwd_dkdv_gfx950.py` | `test_fmha_bwd_dq_gfx950.py` |
| resident | K, V | Q |
| streams | Q, dO | K, V |
| GEMMs | 4 | 3 |
| outputs | dK, dV | dQ, **dB** |

**dQ owns dB even though the name does not say so.** `dB = dS`, and `dS` is
materialised per (q, k) element only in that kernel. AOTriton's
`bwd_kernel_dq` emits `DQ, DB` for the same reason; our gfx1201 port dropped it
and a separate session is restoring it there.

Neither agent writes a fused kernel. See plan §4.

**B1 and B2 are this phase. Not causal, not windows, not varlen, not dropout.**
Dense, non-causal, head_dim 64 and 128 only. The ladder is B3 and the features
follow it; a `const_expr` hook left where a feature will go is welcome, a
half-implemented feature is not.

---

## 2. What to subclass, and what never to touch

`kernels/attention/flash_attn_utils.py` is **imported, never edited.** Four
production kernels import it. Everything that differs lives in your own file as
a subclass, exactly as `fmha_dualwave_gfx950.py` does for the forward.

Start from these, in this order:

1. **`fmha_dualwave_gfx950.py`** — the forward's parity subclasses. Your
   context should subclass `ParityKernelContext`: it already carries per-tensor
   strides named for their axis, the varlen row offsets (pinned at 0 until B5),
   the padded-head machinery, the bias descriptor and the philox prologue. You
   inherit the whole P0–P7 feature surface by subclassing rather than porting.
2. **`flash_attn_func_gfx950.py`** — the dual-wave body, for the pipeline
   shape: 8 clusters, two tiles in flight, where the barriers and `waitcnt` go.
3. **`fmha_wide_gfx950.py`** — the simpler staged body. **Read this one
   first if the dual-wave body is hard to follow**; it is the same algorithm
   without the software pipeline, and B1/B2 may legitimately start there and
   pipeline later.

Do **not** copy `fmha_bwd_*_gfx1201_kernel.py` structurally. Read it for the
algorithm and for the traps it documents; its WMMA 16x16x16 wave32 operand
algebra does not transfer.

---

## 3. The transposed operands — B0's answer

The two q-contracted GEMMs (`dVᵀ = dOᵀ·P`, `dKᵀ = Qᵀ·dS`) need a `[row][d]`
LDS tile read column-major. **That is exactly what the forward already does to
V**, and it is validated end to end.

So: **reuse the forward's V read path.** Do not derive a new lane map.

- `_ds_read_tr16_b64_imm` (`flash_attn_utils`) — the ROCDL op, with alias
  scopes. Never emit this as inline asm; see the lore, Hazard 1.
- `_v_lds_read_base_per_lane`, `_swizzled_v_imm_lo`, and the
  `V_LDS_TO_REG_*` strides that `fmha_traits_gfx950.make_traits` derives from
  the unified `tok_off(t)` formula.
- `tooling/probe_kv_staging.py` validates it: `0 / 4096 wrong` on both the D
  map and the token map. **Extend that probe for your tile before trusting
  your addressing** — it is an hour, and it is the difference between a bug
  you find now and one you bisect at head_dim 512 in B3.

Two ISA constraints (CDNA4 §11.4), both hard:

- **EXEC must be all 1s across these reads.** The forward never hit this
  because its masking is `select`-shaped. The backward has real `scf.if`
  regions around edge tiles. A transpose read inside one is undefined, and the
  failure mode is the house speciality: wrong, finite, no diagnostic.
- 64-bit DS ops need an even-aligned VGPR pair.

---

## 4. The MFMA shape is a parameter, not a constant

The forward uses `32x32x16` everywhere. **The backward must not bake that in.**
AITER's tuned gfx950 backward uses `16x16x16`, `16x16x32` and `32x32x16`, mixes
two shapes inside one kernel, and exposes the tile size as a knob.

Consequences you must respect from the first commit:

- Rows-per-wave (16 or 32) is a trait, not a literal. It sets `BLOCK_N` for
  dK/dV and the accumulator size — at 16 rows the dK/dV accumulators cost
  `d/4` per lane instead of `d/2`, which moves the whole wide threshold.
- Derive every lane→(m, k) map **once**, from a table, and index it. The
  forward's `_score_column_runs` is the pattern: one function returns the runs,
  and bias, dropout and the mask all consume it. With two MFMA shapes live,
  transcribing a map twice is how the two copies drift.

---

## 5. Coding practice

From `CLAUDE.md`, and non-negotiable:

- `range_constexpr` for compile-time loops; `range(..., init=[...])` for
  `scf.for` with carried values.
- **No branch-local `return`/`yield`** in traced functions; one exit path.
- Do not define a value only inside an `if`/`else` and use it after.
- Nested helpers may read captured values, never mutate them.
- `const_expr` for anything that selects *which code is emitted*.
- black line length 120; ruff `E,W,F,I` with the project `pyproject.toml`
  (`E731` is project-ignored — do not `--select` past the config).
- Name strides for their axis: `stride_q_batch/_head/_seq`. Never numeric
  slots. The forward renamed 92 of these for a reason.

House style for comments, and it is enforced in review: **say why, not what.**
Record the measurement or the bug that produced a decision. A comment that
restates the code is noise; a comment that says "this is 2 wait states because
X, measured" is the reason the forward's hazards were findable.

---

## 6. Verification, from the first commit

`FLYDSL_RUNTIME_ENABLE_CACHE=0` **always** — the helpers are not in the JIT
cache key, and stale artifacts have produced phantom passes here before.

Your gate is plan §7.1, the error-ratio method, not a fixed tolerance:

```
ours     = your kernel, bf16
ref_low  = torch math backend, bf16
ref_high = torch math backend, fp64
assert err(ours, ref_high) <= fudge * err(ref_low, ref_high)     # per tensor
```

Plus, and this one catches convention bugs nothing else will: **run the gfx950
forward, take its `O` and `LSE`, and compare against `torch.autograd.grad`
through that same forward.** If your backward disagrees with our forward about
`sm_scale` folding, `log2e`, the LSE layout or causal alignment, this is what
says so.

**dQ and dK/dV must also be checked against one autograd call together.** They
are separate kernels computing parts of one gradient; a scale or transpose
error that cancels between them passes both suites separately.

Traps that have already cost time on the forward:

- `torch`'s `is_causal` is **top-left**; these kernels are **bottom-right**
  (`delta = seqlen_kv - seqlen_q`). They agree only at `Sq == Sk`.
- **Always run a known-good control.** Three wrong conclusions on the forward
  came from a probe with no control.
- A perf claim under 10% needs interleaved single-GPU A/B, not a sweep. Use
  `tooling/ab_knobs_gfx950.py`.

---

## 7. Interfaces the two of you share

Agree these before writing, and if one of you needs to change one, say so
rather than forking it:

- **`delta`** is a host-computed tensor argument, `(dO.float()*O.float()).sum(-1)`,
  shaped like LSE. Both kernels take it; neither computes it. (Plan §5.)
- **LSE** is read through `fmha.lse_row_addressing`, the same function the
  forward writes it with. Do not re-derive the layout.
- **Argument order is the ABI.** Mirror the forward's grouping: tensors, then
  the varlen block, then `max_seqlen_*`, then the window pair, then head
  counts, `hdim_qk`/`hdim_vo`, `sm_scale`, then strides per tensor, then the
  trailing scalars. Leave the slots for features you are not implementing.
- **Traits** subclass `ParityDualwaveTraits`, one per kernel, with the backward
  fields added. Anything both need goes in a shared module rather than being
  defined twice.

---

## 8. What "done" means for B1 and B2

- Dense, non-causal, head_dim 64 and 128, bf16.
- The §6 gates pass, including the joint autograd check.
- `bash scripts/check_python_style.sh`-equivalent clean (black + ruff, project
  config).
- The forward still passes. `test_flash_attn_func_gfx950.py` alone collects
  **329**; the 383 figure is the combined run with `test_philox.py` and
  `tests/kernels/test_flash_attn_fwd.py`, which is the one to use — you are
  subclassing the forward's helpers, and a change that reaches them is a change
  to four production kernels.
- A short outcome section appended to `sdpa-bwd-plan-gfx950.md`, in the style
  of the forward's: what was measured, what surprised you, what is not done.
  If you hit a hazard of the kind `sdpa_lore_gfx950.md` describes, add it there.

---

# Addendum: B3 — the ladder to 512

B1 and B2 shipped dense non-causal at head_dim 64/128, both **unpipelined, one
tile in flight** — which is already the *wide* shape. B3 is therefore not "port
to a second body"; it is "make the body you have reach the whole ladder".

Extend `LADDER` to the forward's `(32, 64, 96, 128, 160, 192, 224, 256, 384,
512)`, with the 8xD input contract and padded heads. Still no causal, windows,
varlen, dropout or bias input — those are B4–B6 and land on top.

## The two kernels hit different walls, and this is the whole brief

Per staging slot at 64 rows: `64 * head_dim * ~2.07 B`. Accumulators are
`d/2` VGPRs per lane at 32 rows per wave, `d/4` at 16.

| head_dim | dK/dV LDS (2 slots) | dQ LDS (3 slots) | dK/dV acc @32 | dQ acc @32 |
|---|---|---|---|---|
| 256 | 66 KB | 99 KB | **256** | 128 |
| 384 | 99 KB | 149 KB | **384** | 192 |
| 512 | **133 KB** | **199 KB — over** | **512** | **256** |

Cap is 163840 B; 256 arch + 256 acc VGPRs, with operands and P/dS on top.

**dK/dV is register-bound and its LDS is fine all the way to 512.** Two tiles
read two ways is a cheap layout and it keeps paying. The accumulators are the
problem: two of them, `d/2` each at 32 rows per wave.

**dQ is LDS-bound before it is register-bound *with `store_db` off*.** With dB
on it is both, and the ladder has to be established as a **rung x dB matrix**
rather than a single line — establishing it on the cheap arm leaves dB's
ceiling undiscovered, which is the same shape of mistake as testing one
`sm_scale`.

Costed at head_dim 512, 32 rows per wave: the dQ accumulator is 256 VGPRs (the
whole AGPR file), Q resident is 128 arch VGPRs, the f32 `dS` tile ~32 and its
packed bf16 form ~16, before K/V operands and addressing. The dB store then
holds `dS` live across a 32-element per-lane store sequence, in the one place
there is no room. **If dB is what tips it, the lever is the live range, not the
accumulator**: store from the packed bf16 rather than the f32 (dB is a bf16
tensor, so that rounding happens anyway), or interleave the store into the
GEMM3 loop so only the consumed slice is live.

The LDS wall is separate and comes first, and the cause is already known: it stages **K twice** (K layout for GEMM1, V layout for GEMM3), which is
the third slot. B2 deferred that to B7. **It is now a B3 blocker** — at two
slots dQ fits 512 in 133 KB, at three it does not fit at all. Fix the double
staging before reaching for `D_STAGES`; B2's own notes name two ways (a K
reader parameterised on the V line stride, or double-buffering into the unused
fourth slot). Also stop allocating the unused fourth slot, which is over the
cap at 384 by itself.

## Order of attack — cheapest lever first

B1's measurement is the reason this is written as an order rather than a list.
At head_dim 128 it found 4 waves alone gave 403 TF and *the same build* with
`waves_per_eu=1` gave 721; 2 waves reached 788 with zero spills. A build at 0
AGPRs with a nonzero spill count is not short of registers, it is forbidden
from using half of them.

1. **`(num_waves, waves_per_eu)` as a pair.** Free, and it answered head_dim
   128. Read `agpr_count` alongside spills; that pair is the discriminator.
2. **Rows per wave 32 → 16.** Halves the dK/dV accumulators outright. Costs
   `BLOCK_N` at a fixed wave count. Contract §4 already required this to be a
   trait rather than a literal — this is the phase that spends it.
3. **Fix dQ's double-K staging.** Structural, and it is what makes 512 fit.
4. **`D_STAGES`** — split the D axis in time, one LDS residency per pass.
   The forward's mechanism; see `fmha_wide_gfx950.py`.
5. **d-axis sharding** (`VO_SHARDS`'s analogue) — last, because it costs wave
   count and cross-shard reduction. dK/dV will likely still need it at 384/512
   even after 1–3; dQ should not.

**Do not skip to 4 or 5 because the table says the accumulator is too big.**
The table is demand, not what the allocator will grant, and B1 already showed
those two differ by 1.8x.

## Gates

- The full `LADDER`, both bodies, error-ratio gate per rung.
- The **8xD contract**: every multiple of 8 from 8 to 512 correct with plainly
  contiguous tensors, and nothing written past the last row. The forward's
  `test_grid8_contiguous_is_exact_and_writes_nothing_past_o` is the model, and
  it found a real class of bug there.
- Padded heads, and note B2's finding: the two head extents are used the *other
  way round* in dQ — GEMM2 reads V through the K register path whose mask is
  written against `hdim_qk`, while dQ stores through the O path whose
  suppression is against `hdim_vo`. Two one-line sites, but they are crossed.
- **From P7: enforce the rows-per-wave ceiling** rather than commenting it. The
  forward shipped twelve silently-wrong configurations because `make_traits`
  stated that invariant in prose and checked nothing.
- **8-minute build cap.** A configuration whose register allocator runs away is
  telling you it does not fit. The forward's P7 sweep wedged four shards on
  exactly this; treat a slow build as a result.
- The joint dQ + dK/dV autograd check must still pass, at every new rung.
