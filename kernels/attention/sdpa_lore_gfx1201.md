# SDPA on gfx1201: what worked, what didn't, and why

Accumulated findings from optimising the RDNA4 flash-attention kernels
(`flash_attn_func_gfx1201*.py`). Companion to
[`gfx1201_fmha.md`](gfx1201_fmha.md) (running notes and hardware capabilities),
[`plan_gfx1201_large_hdim.md`](plan_gfx1201_large_hdim.md) (the wide-head_dim
design) and [`sdpa_efficiency_model.md`](sdpa_efficiency_model.md).

The two ceiling documents answer **different questions and should not be
conflated**. `sdpa_efficiency_model.md` is a *theoretical* upper bound: it uses
architectural constants (16 cyc/WMMA from the Matrix Instruction Calculator,
4 cyc transcendental) and deliberately assumes away memory and LDS stalls,
`s_waitcnt` bubbles and occupancy limits, to answer "how much does SDPA lose
purely because softmax cannot co-execute with WMMA on RDNA4?" -- 78% at
head_dim 128. It should stay free of measured data, or it stops being an upper
bound and becomes board-specific. The numbers below are the opposite kind: they
plug *this board's* measured constants into the same serial model to ask "how
much is left on the table in the current kernel?" Use the first to judge whether
SDPA is worth pursuing on this architecture at all, and the second to judge
whether more scheduling work will pay.

**Read the "What did not work" section before trying anything.** Most of the
ideas in it are good ideas that a reasonable person would try again.

## Where the kernels stand

Measured B=1 H=8 N=4096 f16 non-causal, through
`flydsl_flash_attn_func_gfx1201`. Board peak is ~205 TFLOPS measured
(`kernels/microbench/wmma_peak.py`); AMD's published figure is 191.

| head_dim | 16 | 32 | 48 | 64 | 80 | 96 | 128 | 160 | 192 | 224 | 256 | 384 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TFLOPS | 48 | 73 | 79 | 91 | 96 | 101 | 103 | 98 | 104 | 80 | 83 | 64 | 53 |

At head_dim 128 the loop body is 32 WMMA against 17 `v_exp_f32`, 16
`v_cvt_f16_f32` and 165 other VALU ops. Costed with LLVM's resource cycles
(below) and this board's measured ~11.3 cyc/WMMA, that is 362 WMMA cycles
against 297 VALU -- a **55% WMMA share**, so the strictly-serial ceiling is
~113 TFLOPS here (~105 against the 191 spec). At 103 we are at ~91% of it.
**There is not much left in scheduling.**

(The 55% is lower than the theoretical model's 78% mainly because a measured
11.3 cyc/WMMA is cheaper than the 16 the model assumes, which makes the fixed
softmax term a larger fraction. Different constants, different question -- see
above.)

## The single most important fact

> **Register spills dominate everything else on this kernel.** Check the spill
> table before forming any other hypothesis.

Every large win in the head_dim ladder came from removing spills, and every
regression that looked like something else turned out to be spills:

- head_dim 192/224 slower than 256 -> they were on the unsharded path, spilling
  24 and 64 registers.
- head_dim 160 dip -> 13 spills at 4 waves, 5 at 8.
- head_dim 224 regressing at 8 waves -> 76 spills against 11 at 4.
- head_dim 512 at 22 TFLOPS -> unpadded LDS *and* spills.

Spills go to scratch, which is global memory. A handful of `scratch_load` /
`scratch_store` in the loop body costs more than almost any instruction-count
optimisation can win back.

```bash
grep -oE "vgpr_spill_count: *[0-9]+" 21_final_isa.s
grep -cE "scratch_(load|store)" 21_final_isa.s
```

## Structural facts that constrain everything

### The Q operand is `rows_per_wave x d_per_wave`

Trading one against the other is a wash:

| rows/wave | d/wave | `q_b_packs` |
|---|---|---|
| 16 | 512 | 128 VGPR |
| 16 | 128 | **32** |
| 32 | 256 | 128 |
| 64 | 128 | 128 |

Consequences:
- **`BLOCK_M` is not a lever on per-wave registers.** Rows per wave is pinned at
  `WMMA_M = 16`; `BLOCK_M` only sets how many waves are in the workgroup.
- The only way to shrink Q is to shard the head dimension while holding rows per
  wave at 16 -- and then the shard-waves must be given *different* work
  afterwards or their softmax and GEMM2 are pure duplication.
- Sharding **V/O** rather than rows is what avoids that duplication: each wave
  computes a different slice of O for the *same* 16 rows.

### Per-wave register budget, head_dim 512, measured from the ISA

| what | registers | share |
|---|---|---|
| Q operand packs | 128 | 53% |
| O accumulators | 64 | 26% |
| S accumulators | 16 | 7% |
| K/V staging, addressing, m/l | 35 | 14% |

`s_accs` was never the problem. Q is.

### Small head_dim is softmax-bound, not saturation-bound

Softmax cost is per (Q row, KV tile) and does **not** scale with head_dim:

| head_dim | WMMA per wave per KV tile | fixed cost |
|---|---|---|
| 16 | 4 | 17 `v_exp_f32` + 2 barriers |
| 128 | 32 | same |

At head_dim 16 that is a hard ~37 TFLOPS ceiling for `BLOCK_N=32`, and the
kernel measured 37.4 -- already there. A wider `BLOCK_N` is the lever, because
it amortises the *per-tile* part (correction exp, m/l update, O rescale,
barriers) over more KV columns. `BLOCK_M` does not help: it changes workgroup
count, not the per-wave ratio.

### Hardware facts worth not re-deriving

- **No F32xF32 WMMA on RDNA4** (ISA Table 41). `v_wmma_f32_16x16x4_f32` exists
  in LLVM but is real-ized under `VOP3P_Real_WMMA_gfx1250` only. Keeping P in
  f32 through GEMM2 is therefore impossible without abandoning the matrix cores.
- **WGP mode does not raise the per-workgroup LDS limit.**
  `getLocalMemorySize()` doubles in WGP mode but feeds only occupancy
  (`AMDGPUSubtarget.cpp:49,59`); the check that rejects an oversized kernel is
  `getAddressableLocalMemorySize()` = 65536 (`AMDGPUAsmPrinter.cpp:1380`), which
  never doubles. HIP reports `shared_memory_per_block = 65536`.
- **LLVM resource cycles** (trust these over guesses; they come from internal
  hardware docs): `Write32Bit = 1`, but **`WriteFloatCvt = 4`** and
  **`WriteTrans32 = 4`**. `v_cvt_f16_f32` is explicitly in a
  `let SchedRW = [WriteFloatCvt]` block, so a naive 1-cycle assumption
  undercounts conversions badly.
- **LDS is 32 banks x 4 B.** A power-of-two row stride is the *worst* case: a
  512-element f16 row is 256 dwords and `256 mod 32 == 0`, so all 16 lanes
  reading different rows start in the same bank (16-way conflict). The `+4`
  element padding makes the stride `== 2 (mod 32)`. It is not monotonic -- `+8`
  gives `== 4` and is worse than `+4`. The condition is
  `stride_dwords == 2*odd (mod 32)`.

## What worked

| change | effect |
|---|---|
| `amdgpu-sched-strategy=max-memory-clause` | +13.5% baseline causal, +3.4% bp causal |
| 64-bit base + 32-bit offset addressing | head_dim 512 spills 16 -> 9; wins across the ladder |
| Head-dim sharding across waves (`QK_SHARDS`) | enables 256/384/512 |
| Chunked V staging (`VO_CHUNKS`) | head_dim 512: 22.4 -> 41.4 |
| Per-head_dim `(shards, q_tiles)` table | 224 +17%, 384 +19% |
| V TR tail guard (allow a remainder) | head_dim 160: 70.0 -> 88.9 |
| Wider `BLOCK_N` at small head_dim | 16 +29%, 32 +18% (non-causal only) |

### The GCN scheduler serialises LDS loads by default

The default scheduler sinks each `ds_load` next to its consuming `v_wmma` and
lets the allocator funnel them all through one VGPR quad. The WAR dependency
then forces a full `s_wait_dscnt 0x0` between every load and use -- 33 of 35
waits in the head_dim 128 loop were full drains.

Setting LLVM's `amdgpu-sched-strategy` function attribute (through the existing
`passthrough` block) to `max-memory-clause` drops full drains to 14 and spreads
loads across distinct quads. **Gated, not universal**: it regresses m32 by 5.1%.

### The 64/32 address split

Forming the global element index in i64 puts every address in a VGPR *pair* and
every address computation in an `add`+`addc` chain. Splitting into a uniform
64-bit `(batch, head, tile)` base plus a divergent 32-bit intra-tile offset
halves both, and is also exactly what LLVM's `SelectGlobalSAddr` folds into an
SGPR base plus a VGPR offset.

**The split is mandatory, not just faster.** See the safety note below.

## What did NOT work, and why

### Cutting VALU work (`v_cvt_pk_rtz_f16_f32`)

Replacing the P conversion's 16 `v_cvt_f16_f32` (4 cycles each) + 8
`v_pack_b32_f16` with 8 `v_cvt_pk_rtz_f16_f32` removed an **ISA-verified 64 VALU
cycles per iteration** and moved throughput **0.4%**.

The loop is latency-bound, not VALU-throughput-bound. The additive
"WMMA + VALU" ceiling model overstates what VALU cuts buy. Do not spend effort
shaving VALU ops.

### Source-level scheduling

A hand-batched GEMM1 (loads hoisted ahead of the WMMAs, sweeping group depth
1/2/4/8) produced a **byte-identical schedule**. The backend rescheduler ignores
source order; the function attribute is the only lever.

### `max-ilp` scheduling

Consistently -4.5%. Only `max-memory-clause` helps.

### LDS swizzle replacing padding

Fully implemented (rotate at 4- and 8-element granularity, and XOR), correct at
every head_dim, and LDS landed at exactly the predicted sizes. **Net loss and
reverted:**

| head_dim | padded | rotate-swizzled | spills |
|---|---|---|---|
| 128 | 97.6 | 87.5 | 0 -> 0 |
| 384 | 45.1 | 26.4 | 0 -> **32** |
| 512 | 22.4 | 24.1 | 50 -> 50 |

**What the bank-conflict analysis missed: the cost of computing the swizzled
address.** Padding gives every access a `base + immediate` form because the row
stride is a compile-time constant. A swizzle makes the block index a *runtime*
function of the row, so each access needs its own live address register -- on a
kernel pinned at the 256-VGPR ceiling that turns a spill-free kernel into 32
spills.

Also, counter-intuitively, **XOR was worse than a rotate** (232 VGPR against
186 at head_dim 128) despite being one instruction instead of three: it blocks
base+immediate folding more than an add does.

The bank tables were not wrong, they were **incomplete**. Conflict count is one
term, and on a register-starved kernel it is not the dominant one.

### `ds_add_f32` for the cross-wave S reduction

Atomic float add to LDS exists on gfx1201 and is genuinely selected (16 of them,
no compare-exchange loop). It is **20x worse** than an explicit write/read
reduction:

| variant | ns | WMMA-equivalents |
|---|---|---|
| explicit partials | 2.49 | **54** |
| `ds_add_f32` | 41.55 | **1055** |

Three waves accumulating onto the *same* addresses serialise at the bank, and
the read-modify-write compounds it. The traffic model that predicted atomics
would halve the cost counted bytes moved and ignored contention. It would make
the kernel 3.7x *slower* than not sharding at all.

`kernels/microbench/lds_reduce.py` reproduces this.

### Parking output accumulators in LDS

Implemented behind `o_accs_in_lds` and measured at head_dim 512: **51.1 against
53.5**. Two reasons:

1. Reading the accumulator at the top of the loop body and writing at the bottom
   leaves it live across the whole body -- exactly like a loop-carried value --
   so no register is freed. Narrowing it needs the softmax rescale folded into
   the load *and* GEMM2's loop order swapped from `for pks: for dc:` to
   `for dc: for pks:`, which perturbs the V prefetch pipeline.
2. By the time it was tried there was nothing left to reclaim: the address split
   and wave tuning had taken head_dim 512 from 16 spills to 3, and every other
   head_dim to zero.

**Re-measure the spill table before reopening this.**

### Row-partitioning after head-dim sharding

Sharding GEMM1's reduction across waves and then partitioning *rows* for
softmax/GEMM2 fixes the duplication but **not the registers**: a wave covering
64 rows x 128 d holds the same 128 VGPRs of Q as 16 rows x 512 d, and `s_accs`
grows 4x. It came to 355 VGPR, worse than the 243 it replaced. Shard V/O, not
rows.

### Sharding GEMM1 alone

`QK_SHARDS` distributes the same GEMM1 work; it does not reduce it. The
redundancy that costs 320 WMMA at head_dim 512 comes from *separate launches*
each recomputing all of GEMM1. Sharding only buys work reduction indirectly, by
freeing registers to afford a wider V slice and hence fewer launches. An early
revision that sharded GEMM1 without sharding V/O issued **384** WMMA against the
unsharded 320 -- a regression, because its four shard-waves then duplicated
softmax and GEMM2.

### bf16 round-to-nearest-even

`bf16_trunc_pack_v8` truncates (round-toward-zero), which measures **1.6x the
output RMS error of torch SDPA** at bf16 (f16 is already at exact parity). RTNE
closes the gap exactly but costs 2-3% on bp and 2.7-5.4% on m32. Kept as
truncation **by decision, not oversight**.

Note there is no output *bias* despite the one-sided rounding: O sums `P*V` with
zero-mean V, so it cancels into variance.

## Correctness traps

These all produced **silently wrong results rather than a crash**, which is the
failure mode to fear here.

### Floor division in the cooperative load

`NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD` drops rows whenever RPB
neither reaches `BLOCK_N` nor divides it. head_dim 160/192/224 give RPB
25/21/18, so the count came to 1 and only 25/21/18 of the 32 KV rows reached
LDS; the rest was **stale LDS and the output was NaN**. Fixed with `ceil()` in
both kernels -- and note it had to be fixed *twice*, because bp was gated to
head_dim 64/128 when the baseline was fixed and the bug became reachable later
when that guard was lifted.

**When you widen a kernel's supported range, re-audit every `//` in its tiling
arithmetic.**

### A shard slice that is not a multiple of `WMMA_K`

head_dim 224 with 4 shards gives a 56-wide slice; `K_STEPS_QK = 56 // 16` is 3,
so only 48 of 56 columns were reduced. Measured relative error **0.97** -- it
ran happily and returned garbage. Now rejected at build time.

### 32-bit indices spanning more than the head dimension

The global element index is the **full BHSD linear position**:

```
((batch * seq_len) + token) * nheads * head_dim + head * head_dim + d
```

so it spans B, S and H. Truncating it to i32 caps the tensor at 2G elements, and
the TR path truncated the *byte* offset, halving that to ~1 GB -- B=8 S=32768
nheads=32 head_dim=128 is already 2.1 GB. Only the **intra-tile** part is safely
32-bit, bounded by `max(BLOCK_M, BLOCK_N) * nheads * head_dim + head_dim`.

This was shipped briefly in the unsafe form because the *speedup* was verified
carefully and the *range* was not, on a bench shape where it happened to fit.

## Measurement discipline

### This board drifts ~5% between whole-script runs

A non-interleaved A/B first measured the scheduler change at **+4.8%** when the
truth was **+0.4%**; re-running the same binary gave 96.3 then 91.2 TFLOPS.

**Always interleave arms (`for rep: for arm`) and repeat 3x.** Real effects come
out flat to +/-0.1 TFLOPS across reps; anything that moves more than that
between reps of the same arm is drift. Use `bench_one.py` (one config per
process), not a multi-config sweep.

### Prove a no-op with an ISA byte-diff

When a refactor is supposed to change nothing, dump the ISA both ways and
`diff`. That is stronger and cheaper than benchmarking, and it caught a 4-VGPR
regression (190 -> 194) that crossed the 8-waves/SIMD boundary at 1536/192 and
cost 4% while the *instruction mix was identical*.

### Watch for allocation-granularity cliffs

Occupancy is `1536 / vgpr_count` waves per SIMD. Small register changes matter
enormously near a boundary: 190 -> 194 VGPRs is 8 -> 7 waves.

### DCE and strength reduction will silently gut a microbenchmark

A register probe with zero-initialised state was constant-folded; a co-issue
probe's `x = x*1.0 + 1.0` was strength-reduced out of the loop; a `nogemm2`
ablation dead-coded the entire loop and measured "no compute". Make state
data-dependent and runtime-derived, consume every result, and **verify against
the ISA**.

## Things that are per-head_dim and measured, not derived

There is no clean formula for the tiling. More waves helps while registers and
LDS allow and hurts the moment it pushes either over, and neither is visible
before compiling. So these are lookup tables with the sweep recorded in the
comment:

- `_BP_SHARDS_BY_HEAD_DIM`, `_BP_Q_TILES_BY_HEAD_DIM` (bp)
- `_BLOCK_M_BY_HEAD_DIM`, `_BLOCK_N_BY_HEAD_DIM_NONCAUSAL` (baseline)
- `_BP_MIN_HEAD_DIM = 48` -- the threshold above which bp beats the baseline

**A new head_dim gets the default and may be leaving 5-15% on the table.** If
that becomes a maintenance problem, the fix is an autotune pass that compiles a
few configs and picks by spill count -- spills predicted the winner in every
case measured here.

## Known remaining gaps

| gap | size | what it needs |
|---|---|---|
| causal stuck at `BLOCK_N=32` | 16/32 causal at 37/61 vs 48/73 non-causal | the 16-scalar mask unroll rewritten |
| head_dim 512 at 53 | LDS-capped at 51456 B | a different decomposition |
| head_dim 224 at 80 | `THREADS_PER_ROW_LOAD = 14` | awkward geometry, not scheduling |
| arbitrary strides | `STRIDE_TOKEN` is computed, not passed | stride arguments for non-contiguous BHSD views |

## Why further profiling is probably not the next move

At head_dim 128 we are at ~91% of the strictly-serial WMMA+VALU ceiling
computed with this board's constants. Every large win this session came from
**static analysis** -- reading ISA, counting
registers, checking LLVM's scheduling model -- not from tracing. ATT has a poor
track record on this kernel specifically: the profiling shape changed the
occupancy regime and inverted stall attribution, and per-wave stall is not
recoverable time at high occupancy (both recorded in `gfx1201_fmha.md`).

The remaining gaps in the table above are structural, and none of them is a
micro-scheduling problem.

## head_dim 32 non-causal is bimodal; the ladder A/B cannot resolve it

`bench_aiw_ab.py` at head_dim 32 non-causal returns either ~58.3 or ~63
TFLOPS, and **which one is not a function of the code**. Alternating the
pre-gSWA kernel against the gSWA one, three runs each:

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| pre-gSWA | 63.6 | 58.2 | 58.3 |
| gSWA | 58.3 | 62.9 | 58.3 |

The legacy reference measured in the same reps is stable throughout (70.9 to
71.6), so this is not board drift and the interleaved-ratio discipline does
not remove it — the ratio column swings 0.82 to 0.89 with it.

This cost a round of bisecting: the same -8% appeared in two consecutive
before/after comparisons, looked reproducible, and was neither a regression
nor noise in the usual sense. A single A/B at this point can produce a
confident-looking -8% in either direction.

**Treat any head_dim 32 non-causal delta below ~10% as unresolved unless at
least three alternating runs agree.** The point is plausibly bistable in a
clock or allocation state rather than in the kernel; nobody has chased it
down. head_dim 16 non-causal, measured alongside, is stable to 0.5%.
