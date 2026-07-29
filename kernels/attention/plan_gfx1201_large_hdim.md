# Plan: wide head dimensions on gfx1201 FMHA

Status: **revision 3, in progress.** Step 1 (measurement gate) is the current
work; steps 2-3 are not started.

Goal: at `head_dim` 512, cut per-wave registers from 243 to ~147 and issued
WMMA from 320 to 128 per 16 output rows, by sharding the head dimension across
waves in GEMM1 and the V/O columns across the same waves in GEMM2.

## Where the register pressure actually is

Measured from the d=512 ISA (`vgpr_count: 243`, `vgpr_spill_count: 0`),
classifying registers by loop lifetime:

| what | registers | count | share |
|---|---|---|---|
| **Q operand packs** (`q_b_packs`) | `v[65:192]` | **128** | **53%** |
| O accumulators (`o_accs`) | `v[1:64]` | 64 | 26% |
| S accumulators (`s_accs`) | `v[227:242]` | 16 | 7% |
| K/V staging, addressing, m/l | scattered | 35 | 14% |

Identified from the ISA, not inferred: `v[1:64]` is eight 8-wide accumulators
each hit exactly 2x (`PV_K_STEPS`), `v[227:242]` two 8-wide accumulators each
hit 32x (`K_STEPS_QK`), and `v[65:192]` is read by 64 of the loop's 80 WMMAs
and **never written inside the loop**. 140 registers are loop-invariant; 128 of
them are Q.

A wave holds 16 Q rows x 512 head dim = 8192 halves over 32 lanes = 128 VGPRs.
It is loaded once and held for the whole KV loop because Q is reused by every
KV tile. There is no waste to reclaim -- Q at head_dim 512 simply costs half
the register file.

The 64:16 GEMM1:GEMM2 split in that same loop is the other half of the problem:
**each launch issues 64 GEMM1 WMMAs to produce 16 GEMM2 WMMAs of useful
output**, and repeats that four times over the four V slices.

## The design rule that decides the structure

> A wave's Q operand is `rows_per_wave x d_per_wave`. Trading one against the
> other is a wash.

| rows/wave | d/wave | `q_b_packs` |
|---|---|---|
| 16 | 512 | 128 (today) |
| **16** | **128** | **32** |
| 32 | 256 | 128 |
| 64 | 128 | 128 |

Two consequences:

- **`BLOCK_M` is not a lever.** Rows per wave is already pinned at
  `WMMA_M = 16`, the tile floor; `BLOCK_M` only sets how many waves are in the
  workgroup. It cannot shrink Q.
- **`s_accs` was never the problem.** It is 16 VGPRs, 7% of the file. Q is 53%.

So the only way to shrink Q is to shard the head dimension while keeping rows
per wave at 16 -- and then the four shard-waves must be given *different* work
afterwards, or their softmax and GEMM2 are pure duplication.

## Structures considered

| | q_b | o_acc | s | VGPR | launches | WMMA/16row | LDS | |
|---|---|---|---|---|---|---|---|---|
| today, no shard | 128 | 64 | 16 | 243 | 4 | 320 | 41472 | baseline |
| shard d only (rev 2) | 32 | 128 | 16 | 211 | 2 | **384** | 49664 | *regression* |
| shard d, partition rows | 128 | 128 | 64 | **355** | 2 | 192 | 49664 | over VGPR |
| **shard d + shard VO (rev 1/3)** | **32** | **64** | **16** | **147** | **1** | **128** | 66048 -> 33024 | **chosen** |

Why the two rejected ones fail:

- **Shard d only**: after the reduction all four waves hold the same S and do
  the same softmax and the same GEMM2. Issued WMMA is 384, worse than today's
  320. The earlier "192" figure counted useful work, not issued.
- **Shard d, partition rows afterwards**: fixes the duplication but not the
  registers. A wave covering 64 rows x 128 d holds the same 128 VGPRs of Q as
  16 rows x 512 d, and `s_accs` grows 4x to hold partials for 64 rows.

**Sharding V/O rather than rows is what avoids the duplication**: each wave
computes a different slice of O for the *same* 16 rows, so no GEMM2 repeats.
Partitioning by rows instead would force every wave to hold Q for all those
rows, which is exactly the 128-VGPR problem. Only softmax duplicates 4x, and
that is the cheap part -- 17 `v_exp_f32` against 32 WMMA per wave.

## Chosen design

Waves indexed `(q_tile, shard)`, `QK_SHARDS = 4`, `NUM_WAVES = 8` => 2 Q-tiles,
`BLOCK_M = 32`, `BLOCK_N = 32`. `head_dim` 512 runs in **one launch**, no V
column slicing.

Per wave, per KV tile:

1. **GEMM1 partial.** Wave *s* reduces over head-dim slice *s* only:
   `S_partial = K[:, slice_s] @ Q[:, slice_s]^T`. 8 K-steps x 2 N-tiles = 16
   WMMA. Q operand is 16 rows x 128 d = **32 VGPR**.
2. **Reduce S across the 4 shard-waves** through LDS. Variant chosen by Step 1.
3. **Softmax** on the full S. Duplicated 4x; accepted.
4. **GEMM2 on its own V/O slice.** Wave *s* accumulates
   `O[:, vo_slice_s] += P @ V[:, vo_slice_s]`, `vo_slice = 512/4 = 128`.
   8 D-chunks x 2 PV-steps = 16 WMMA. `o_accs` = **64 VGPR**.

Totals: **147 VGPR**, **128 WMMA per 16 output rows per KV tile** (32 per wave
x 4 waves, none duplicated), against today's 243 and 320.

### LDS

K stays in LDS at full width: `32 * 516 * 2` = **33024 B**. V does **not** go
through LDS -- each wave consumes a private 128-wide slice, so it is loaded
straight to registers in WMMA-operand layout via `global_load_tr_b128`, the
mechanism `flash_attn_func_gfx1201_bp.py` already uses to stage V transposed.
Since the four waves touch disjoint slices of both K and V, each is still read
exactly once collectively -- no amplification.

Keeping V in LDS instead would need `32 * 516 * 2 * 2` = 66048 B, over the
64 KiB limit. WGP mode does not help: `getLocalMemorySize()` doubles in WGP
mode (`AMDGPUBaseInfo.cpp:1179`) but feeds only occupancy
(`AMDGPUSubtarget.cpp:49,59`), while the check that rejects an oversized kernel
is `getAddressableLocalMemorySize()` = 65536
(`AMDGPUAsmPrinter.cpp:1380`), which never doubles. HIP reports
`shared_memory_per_block = 65536`. The 128 KiB buys two 64 KiB workgroups per
WGP, i.e. occupancy, already ours since gfx1201 carries no `FeatureCuMode`.

## Step 1 -- measure the cross-wave S reduction. DECISION GATE.

The whole design rests on a reduction that runs once per Q-tile per KV tile, in
the LDS-latency regime that already dominates this kernel (see "The GCN
scheduler serializes LDS loads" in `gfx1201_fmha.md`). Measure before building.

S per Q-tile is 16 rows x 32 KV x 4 B = 2048 B (64 B per lane). Three variants:

| | mechanism | traffic | softmax |
|---|---|---|---|
| **A** | each wave writes its partial, each reads the other 3 and sums | 8 KB write + 24 KB read = **32 KB** | 4x |
| **B** | `ds_add_f32` atomic accumulate, barrier, all read result | 8 KB atomic + 8 KB read = **16 KB** | 4x |
| **C** | B, but one wave softmaxes and republishes P as f16 | 8 + 2 + 1 + 4 = **~15 KB** | 1x |

A is O(N^2) in shard count, B and C are O(N). `llvm.atomicrmw fadd` on an
addrspace(3) pointer selects `ds_add_f32`, which is real-ized for gfx12
(`DS_Real_gfx10_gfx11_gfx12_gfx13<0x015>`); `AtomicBinOp.fadd` is exposed in
the FlyDSL LLVM bindings.

Two caveats to carry into the measurement:

- **An atomic add is a read-modify-write at the bank**, so B's 8 KB of atomics
  may cost like 16 KB of bandwidth. B's advantage over A could be much smaller
  than the traffic table suggests.
- **`ds_add_f32` is non-deterministic**: summation order varies run to run and
  float add is not associative, so results differ in the last bits between
  runs. Only 4 terms, so the magnitude is tiny, but it forecloses bit-exact
  reproducibility and would break a bitwise-equality test of the kind the bp
  variant already has. A keeps a fixed order and stays deterministic.

**Gate:** the reduction must cost materially less than the 192 WMMA per 16 rows
it saves (~2170 cycles at 11.3 cyc/WMMA spread over 4 waves). If the cheapest
variant lands near or above that, the design does not pay and we stop.

Deliverable: `kernels/microbench/lds_reduce.py`, reporting ns and estimated
cycles per reduction for A/B/C plus a no-reduction baseline.

## Step 2 -- implement, only if Step 1 clears

- wave index split into `(q_tile, shard)`;
- `q_b_packs` preload restricted to the wave's own head-dim slice;
- S reduction (variant from Step 1) + barrier;
- V via `global_load_tr_b128`, no V LDS;
- GEMM2 restricted to the wave's V/O slice; output store likewise;
- drop the launch-level V slicing for `head_dim > 256`.

## Step 3 -- measure

Against 31.6 TFLOPS at head_dim 512 and 36.9 at 384. Interleaved A/B, 3 reps --
this board drifts ~5% between runs and a naive before/after already produced
one wrong conclusion (+4.8% that was really +0.4%). Keep the current
slice-per-launch path as correctness reference and fallback.

## Risks

1. **Reduction cost** -- Step 1 gates it.
2. **Softmax 4x redundant.** Throughput, not latency (the waves are on
   different SIMDs), and small next to 32 WMMA per wave. Variant C removes it
   at the cost of an extra barrier.
3. **`BLOCK_M` drops to 32**: more workgroups, more K re-reads across them.
   Partly offset by 147 VGPR against 243.
4. **One or two extra barriers per KV iteration.** Barriers phase-lock all 8
   waves; a previous one-barrier restructure measured -2.7% for that reason.
5. **V from global rather than LDS** loses the LDS staging that the current
   kernel relies on; the bp kernel's TR path is the reference but it still
   lands in LDS there, so this is not a like-for-like reuse.

## Deferred

`hdim_qk != hdim_vo` (MLA). Not needed for `head_dim > 256`. The builder's
current `head_dim_v` / `d_offset` are a **column slice of a V tensor that is
`hdim_qk` wide** -- V's stride is still `num_heads * head_dim` and `Out` is
allocated `hdim_qk` wide -- so they are not `hdim_vo` and must not be exposed
under that name. A true `hdim_vo` needs a second token stride, `Out` allocated
at `hdim_vo`, `o_accs` sized from it, and matching interface validation.
Rename them to `VO_SLICE` / `vo_slice_offset` to keep the name free.
