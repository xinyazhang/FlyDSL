# Plan: wide head dimensions on gfx1201 FMHA

Status: **revision 3, in progress.** Step 1 (measurement gate) is **done and
passed**; Step 2 (implementation) is next.

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

### LDS: V stays in LDS, swizzle replaces padding

**Corrected.** An earlier draft said V would be taken register-direct via
`global_load_tr_b128`, "the mechanism bp already uses". Only half of that is
true. bp does use TR loads, but it stages V *through* LDS:

```
coop_load_v_global  ->  TR loads from global into registers
coop_store_v_lds    ->  _v_store_transposed writes them to LDS
_load_v_rowmajor    ->  _lds_load_v8 reads V back for GEMM2
```

The TR load's purpose there is to land V in LDS already in WMMA-operand layout,
because gfx1201 has no `ds_load_tr_b64` for an LDS-side transpose. Register-
direct V is **not** an existing mechanism, and "LDS bypass for V" was previously
measured at **-15%** at head_dim 128. With the gate margin at 1.76x rather than
the 2.5x the WMMA count suggested, that is not a risk worth stacking on.

So V stays in LDS, and the 64 KiB budget is met by replacing the `+4` element
padding with an XOR swizzle, which costs no extra LDS:

| | padded | swizzled |
|---|---|---|
| K: `BLOCK_N x (head_dim+pad)` | 32 x 516 x 2 = 33024 | 32 x 512 x 2 = **32768** |
| V transposed: `head_dim x (BLOCK_N+pad)` | 512 x 36 x 2 = 36864 | 512 x 32 x 2 = **32768** |
| total | 69888 (over) | **65536** (exactly at the cap) |

Zero margin, so any future addition to LDS forces a rethink. Register-direct V
stays on the shelf as a separate experiment if the swizzle proves awkward.

WGP mode does not help: `getLocalMemorySize()` doubles in WGP mode
(`AMDGPUBaseInfo.cpp:1179`) but feeds only occupancy
(`AMDGPUSubtarget.cpp:49,59`), while the check that rejects an oversized kernel
is `getAddressableLocalMemorySize()` = 65536 (`AMDGPUAsmPrinter.cpp:1380`),
which never doubles. HIP reports `shared_memory_per_block = 65536`.

## Step 1 RESULT: gate PASSES with explicit partials; atomics REJECTED

Measured with `kernels/microbench/lds_reduce.py` (grid 512, iters 2000, best of
5), normalised per Q-tile per KV tile. The WMMA yardstick is linear over
64/128/256 WMMA per unit, giving 0.0390 ns per WMMA:

| variant | ns | WMMA-equivalents |
|---|---|---|
| baseline (no reduction) | 0.39 | - |
| **explicit partials** | 2.49 | **54** |
| `ds_add_f32` atomic | 41.55 | **1055** |

**`ds_add_f32` is 20x worse than an explicit write/read reduction and is
rejected.** The ISA confirms `ds_add_f32` was genuinely selected (16 of them, no
compare-exchange loop), so this is real hardware behaviour, not a lowering
artefact: three waves atomically accumulating onto the *same* 512 addresses
serialises at the bank, and the read-modify-write cost compounds it. The traffic
model that predicted atomics would halve the cost was wrong -- it counted bytes
moved and ignored contention.

With explicit partials, per Q-tile per KV tile at head_dim 512:

| | WMMA-equivalents |
|---|---|
| today | 320 |
| sharded (128) + reduction (54) | **182** |

**1.76x**, extrapolating 31.6 -> ~56 TFLOPS. Lower than the ~65 estimated from
WMMA count alone, because the reduction eats 54 of the 192 saved, but still a
clear win. Gate passes.

Caveats to carry into Step 3: the microbench's LDS has no competing traffic,
whereas in the real kernel the reduction contends with K tile reads; softmax
stays 4x duplicated; and the extra barriers phase-lock all 8 waves.

Variant C (single-wave softmax, republish P) is not worth pursuing while
atomics are rejected -- its whole point was to build on cheap atomic
accumulation.

## Step 1 method (superseded by the result above)

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

## Step 2 -- fuse into the perf kernel (gate cleared)

Implement inside **`flash_attn_func_gfx1201_bp.py`**, the 91.9 TFLOPS kernel at
head_dim 128 -- not as a new file. `QK_SHARDS` becomes a parameter and
**`QK_SHARDS == 1` is today's kernel exactly**: no reduction, no extra barrier,
no duplicated softmax, no V/O sharding. Every sharded construct sits behind
`const_expr(QK_SHARDS > 1)` so the unsharded path traces to identical IR.

`QK_SHARDS = max(1, head_dim // SHARD_WIDTH)` with `SHARD_WIDTH = 128`, so
head_dim <= 128 keeps the current single-wave-per-Q-tile structure and only
head_dim 256/384/512 shard.

**Step 2a -- restructure, `QK_SHARDS = 1` only.** Split the wave index into
`(q_tile, shard)` and thread it through, with `QK_SHARDS` pinned to 1. Success
criterion: **bit-identical output and no measurable perf change at head_dim
64/128**. This isolates the refactor from the new behaviour.

**Step 2b -- add the sharded path.** Behind `const_expr(QK_SHARDS > 1)`:
- Q preload restricted to the wave's own head-dim slice (`q_b_packs` 128 -> 32);
- partial GEMM1 over that slice;
- explicit-partial S reduction through LDS + barrier (variant from Step 1;
  atomics rejected);
- softmax on the reduced S, duplicated across the shard-waves;
- GEMM2 and the output store restricted to the wave's V/O slice.

**Step 2c -- lift bp's head_dim guard.** It currently accepts only 64/128;
extend to 256/384/512 and route those away from the baseline kernel's
launch-level V slicing.

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

## LDS swizzle: verification results (2026-07-29)

Padding exists for **bank conflicts, not alignment** -- and head_dim being a
power of two is what causes them. LDS is 32 banks x 4 B; a 512-element row is
256 dwords and `256 mod 32 == 0`, so all 16 lanes reading 16 different rows
start in the same bank (16-way). `+4` elements = +2 dwords makes the stride
`== 2 (mod 32)`, spreading lane *r* onto banks {2r, 2r+1} -- all 32, once.
It is not monotonic: `+8` gives `== 4` and falls back to 2-way. The condition
is `stride_dwords == 2*odd (mod 32)`.

**Rotate beats XOR as a padding-free replacement.** `swz_block = (block + row)
mod NB` over 4-element (8-byte) blocks is a permutation for *any* `NB`, whereas
XOR needs its mask to divide `NB`, which only holds when `head_dim % 64 == 0`.
For the GEMM1 K read (16 lanes, 16 rows, fixed column):

| head_dim | no pad | XOR no pad | rotate no pad | +4 pad |
|---|---|---|---|---|
| 64..512 (incl. 80/96/160/224) | 4-16 way | 1-way only if hd%64==0 | **1-way** | 1-way |
| 16 / 32 / 48 | 4/8/4-way | 4/2/4-way | 4/2/2-way | 1-way |

16/32/48 have `NB = 4/8/12 < 16` blocks per row, so 16 lanes cannot reach 32
banks under *any* permutation -- a floor, not a scheme failure. Rotate already
attains it.

Extending to the other two access patterns, **rotate matches `+4` padding
exactly** on all of them at head_dim 64/128/256/384/512:

| pattern | +4 pad | rotate, no pad |
|---|---|---|
| K read (GEMM1) | 32/32 banks | 32/32 |
| K store (coop) | 32/32 | 32/32 |
| V^T store + GEMM2 read | 32/32 | 32/32 |

So rotate is a safe drop-in for padding and frees its storage: at head_dim 512
K 32768 + V 32768 = **65536**, exactly at the cap and conflict-free, removing
the 16-way K / 8-way V conflicts that currently cost that width -29%.

Cost is amortised to the prologue: in the GEMM1 read `row` is `lane16`
(loop-invariant) and `block` is constexpr per unrolled step, so
`(block + lane16) mod NB` precomputes outside the KV loop -- the conditional
subtract (`t = blk + r; if t >= NB: t -= NB`, valid since `blk + r < 2*NB`)
never enters it.

**Caveat on the metric.** The tables count banks covered and worst lanes-per-
bank. With 32 lanes x 2 dwords = 64 accesses over 32 banks, 2 per bank is the
floor, so "4-way" on a 32/32 row is not necessarily pathological. The
comparison between schemes is sound; the absolute conflict factors should be
confirmed against ISA or a profile before being quoted as speedups.

### Next

1. Implement rotate uniformly at the four sites (K store, GEMM1 K read, V^T
   store, GEMM2 V read), dropping `_LDS_PAD` entirely.
2. Add 512 to `_BP_HEAD_DIMS` and re-measure.
3. Re-verify the full head_dim ladder for regressions -- especially 64/128,
   which must stay byte-identical or better.

## Rotate swizzle: IMPLEMENTED, MEASURED, REVERTED (2026-07-29)

The bank-conflict analysis above is correct, and the implementation worked --
LDS came out at exactly the predicted 16384 / 32768 / 49152 / 65536 B for
head_dim 128 / 256 / 384 / 512, all unpadded, with correct results at every
head_dim in both masking modes. **But it is a net loss and has been reverted.**

Paired A/B, B=1 H=8 N=4096 f16 non-causal, bp kernel:

| head_dim | padded | rotate-swizzled | delta | spills |
|---|---|---|---|---|
| 128 | 97.6 | 87.5 | **-10%** | 0 -> 0 |
| 256 | 75.3 | 73.9 | -2% | 0 -> 0 |
| 384 | 45.1 | 26.4 | **-41%** | 0 -> **32** |
| 512 | 22.4 | 24.1 | +7.6% | 50 -> 50 |

**What the bank model missed: the cost of computing the swizzled address.**
Padding gives every access a `base + immediate` form, because the row stride is
a compile-time constant and the column steps are constexpr. A swizzle makes the
block index a *runtime* function of the row, so each access needs its own live
address register -- and these kernels already sit at the 256-VGPR ceiling. At
head_dim 384 that pushed a spill-free kernel to 32 spills and cost 41%.

Three variants were tried, in this order:

1. `(blk + row) mod nb` with a true remainder -- 58 spills at 512. A runtime
   remainder by a non-power-of-two lowers to a magic-multiply sequence.
2. Conditional subtract instead (`r = row & (R-1)` with R a power of two, so
   `blk + r < 2*nb` and one compare/cndmask suffices) -- 58 -> 39 spills.
3. XOR where the block count allows it -- **worse** (232 VGPR at 128 against
   the rotate's 186): XOR blocks base+immediate folding more than an add does.
4. 8-element blocks for K (one address per GEMM1 read instead of two) with
   4-element blocks for V -- best of the four, 32/50 spills, and the numbers
   above.

Even at its best the swizzle only helps head_dim 512, and 24.1 is still below
the baseline's 31.6, so it does not unlock that width either.

**Conclusion: padding stays.** head_dim 512 remains on the baseline's
launch-level V slicing, and `_BP_HEAD_DIMS` stays {256, 384}. Reopening this
needs a way to make head_dim 512 fit 64 KiB *without* per-access address
arithmetic -- the constraint is not bank conflicts alone but bank conflicts at
zero register cost.

Worth noting for anyone re-reading the analysis above: the bank tables are not
wrong, they are incomplete. Conflict count is only one term, and on a kernel
pinned at the register ceiling it is not the dominant one.

## Chunked V staging: head_dim 512 now 41.3 TFLOPS (2026-07-29)

The swizzle attempt above assumed the LDS pressure was irreducible -- that
sharding V/O across waves *forces* the full head_dim of V^T resident, so the
only way to fit 64 KiB was to drop padding. That was wrong. Staging V in
`VO_CHUNKS` passes decouples "how many waves work concurrently" from "how much
V is resident", and restores the padding without any swizzle.

Partition **(B)**: chunk *c* covers a contiguous window of `VO_CHUNK_COLS`
output columns, and within it wave *s* owns `VO_CHUNK_COLS / QK_SHARDS`. The
LDS window is contiguous, so no global->local `d` remap is needed; only the
output store takes a per-chunk column offset. Each wave still owns
`head_dim / QK_SHARDS` columns in total, split across chunks, so `o_accs` is
unchanged at 64 VGPRs.

`vo_chunks()` picks the fewest passes that let the *padded* tile fit, so only
head_dim 512 chunks (2 passes); 64-384 stay at one pass and are untouched.

| head_dim | chunks | LDS | padded | VGPR | spills | TFLOPS |
|---|---|---|---|---|---|---|
| 128 | 1 | 17664 | yes | 194 | 0 | - |
| 256 | 1 | 35072 | yes | 246 | 0 | 75.2 |
| 384 | 1 | 52480 | yes | 256 | 10 | 45.0 |
| **512** | **2** | **51456** | **yes** | 256 | 16 | **41.3** |

head_dim 512: **22.4 -> 41.4 (+85%)**, and past the baseline's 31.6 by 31%, so
it joins `_BP_HEAD_DIMS`. Spills fell 28 -> 16 because only one chunk of V is
prefetched at a time. Cost is one extra barrier pair per KV tile.

The prefetch runs one step ahead of the flattened (iteration, chunk) sequence,
so exactly one chunk of V rides the loop in registers.

**Known follow-up:** the restructure costs 4 VGPRs at head_dim 128 on the bp
path (190 -> 194), which crosses the 8-waves/SIMD boundary (1536/192) and
measures -4% there. The instruction mix is byte-for-byte identical -- same
WMMA, LDS, barrier and wait counts, no spills -- so it is purely allocation.
Wrapping GEMM2 in a function was ruled out as the cause (that alone still gives
190). Not chased further because head_dim 128 routes to the baseline by
default, so only explicit `use_binding_prefetch=True` sees it.

## o_accs-to-LDS offload: IMPLEMENTED, MEASURED, REVERTED (2026-07-29)

Idea: park some of the output accumulators in LDS instead of registers. Each is
8 VGPRs per lane, so every one moved frees 8 registers from the loop-carried
set -- the term that decides spilling -- at a cost of
`NUM_WAVES * 8 * WARP_SIZE` f32 of LDS and one read/write per KV iteration.

Implemented behind an `o_accs_in_lds` builder parameter and measured at head_dim
512, the only config with LDS headroom and spills left. **It does not pay.**

| o_accs in LDS | VGPR | spills | scratch | ds ops | LDS | TFLOPS |
|---|---|---|---|---|---|---|
| 0 | 256 | 3 | 6 | 85 | 51456 | 53.5 |
| 1 | 256 | 4 | 8 | 105 | 59648 | 51.1 |

Two reasons, and the second is the decisive one.

**It did not shorten the live range.** Reading the accumulator at the top of the
loop body and writing it at the bottom leaves it live across the whole body --
exactly like a loop-carried value -- so nothing is freed. Narrowing it would
mean folding the softmax rescale into the load and moving the read/write into
GEMM2, and even then the `for pks: for dc:` order touches each accumulator at
both `pks` values, spanning the whole of GEMM2. It would need the loop order
swapped to `for dc: for pks:`, which perturbs the V prefetch pipeline.

**There is almost nothing left to reclaim.** When this idea was parked, head_dim
512 spilled 16 registers. The 64/32 address split took it to 9 and the wave-count
tuning to 3. Across the whole ladder:

| hdim | 96 | 128 | 160 | 192 | 224 | 256 | 384 | 512 |
|---|---|---|---|---|---|---|---|---|
| spills | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |

Six scratch ops per iteration against the 20 LDS ops needed to replace them is a
losing trade regardless of how the live range is arranged.

**Before reopening this, re-measure the spill table.** The idea is only worth
anything if some config is spilling meaningfully again -- a new head_dim
defaulting into an untuned `(shards, q_tiles)` would be the likely candidate.
