# Plan: wide head dimensions on gfx1201 FMHA

Status: **draft for review, not started.** Revision 2.

Goal: make `head_dim` 384/512 faster on `flash_attn_func_gfx1201.py` by cutting
per-wave register pressure, so that the existing V column slicing can use
fewer, wider slices and stop paying for so many redundant GEMM1 passes.

## Revision 2: what changed

- **`hdim_qk != hdim_vo` (MLA) is deferred.** Not needed for `head_dim > 256`,
  and dropping it removes the whole second-token-stride / `Out`-width /
  interface-validation workstream. Self-attention has `hdim_qk == hdim_vo`, and
  that is the only case in scope.
- **Only the Q·K^T (GEMM1) head dimension is sharded across waves.** Revision 1
  sharded both GEMM1 and the V/O output within a workgroup; this shards GEMM1
  only and keeps the existing per-launch V column slicing untouched.

Terminology kept from revision 1: `QK_SHARD` is the sub-division of the head
dimension that one wave reduces over in GEMM1. `VO_SLICE` is the existing
column slice of V/Out that one *launch* computes.

## What sharding GEMM1 does and does not buy

`S[i,j] = sum_d Q[i,d] * K[j,d]`, so the reduction can be split: wave *s*
computes a partial sum over its own `QK_SHARD`, and the partials are reduced
through LDS. Each wave then only needs its own slice of Q in registers, so
`q_b_packs` drops from `head_dim/4` to `head_dim/(4 * QK_SHARDS)`.

**It reduces registers, not total GEMM1 work.** Splitting the reduction across
four waves distributes the same 64 WMMA; it does not remove any. The redundancy
that costs 320 WMMA per Q-tile at head_dim 512 comes from having **four
separate launches**, each recomputing the whole of GEMM1 because every V slice
needs the full P.

So the work saving is **indirect**: freed registers are spent on a wider
`VO_SLICE`, which means fewer launches, which means fewer redundant GEMM1
passes.

## Budgets (head_dim 512, BLOCK_N 32)

Overhead calibrated against the measured d=512 point (243 VGPR, 0 spills, at
`q_b_packs`=128 and `o_accs`=64), giving 51 VGPR of non-scaling
`s_accs`/`p_vals`/addressing.

| QK_SHARDS | VO_SLICE | launches | q_b | o_acc | VGPR | GEMM1 | GEMM2 | WMMA | LDS | |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 128 | 4 | 128 | 64 | 243 | 256 | 64 | 320 | 41472 | today |
| 4 | 128 | 4 | 32 | 64 | 147 | 256 | 64 | 320 | 41472 | registers freed, no work saved |
| **4** | **256** | **2** | **32** | **128** | **211** | **128** | **64** | **192** | **49664** | **target** |
| 2 | 256 | 2 | 64 | 128 | 243 | 128 | 64 | 192 | 49664 | no register headroom |
| 4 | 512 | 1 | 32 | 256 | 339 | 64 | 64 | 128 | 66048 | VGPR + LDS |
| 8 | 512 | 1 | 16 | 256 | 323 | 64 | 64 | 128 | 66048 | VGPR + LDS |

**Target: `QK_SHARDS=4`, `VO_SLICE=256`, 2 launches.** 211 VGPR (32 below
today's 243, which already spills nothing) and **192 WMMA against 320, a 40%
cut**. LDS 49664 B, comfortably inside the 64 KiB per-workgroup limit.

Note the last two rows: `o_accs` alone is 256 VGPRs at `VO_SLICE=512`, so the
ideal 128 WMMA is **unreachable by QK sharding at any shard count**. Getting
there requires splitting V/O *within* a workgroup, which is revision 1's
design. This plan deliberately stops at 192.

### Wave organisation

`NUM_WAVES = Q_TILES * QK_SHARDS`. With 8 waves and `QK_SHARDS=4` that is 2
Q-tiles, so `BLOCK_M = 32`. All four shard-waves of a Q-tile cover the same 16
Q rows and differ only in which head-dimension slice they reduce.

After the reduction every shard-wave holds the full S for those rows. They then
run softmax and GEMM2 **redundantly** — same P, same `VO_SLICE`, same output.
That is 4x wasted work on those stages, and is the price of not sharding V/O.
It costs throughput, not latency, because the waves sit on different SIMDs; it
does mean only one of them should perform the output store.

*Open question below: whether that redundancy is bad enough to justify going
straight to revision 1.*

## LDS

Unchanged from today apart from the wider V slice: K at full head_dim (33024 B)
plus a 256-wide V slice (16640 B) = 49664 B. No swizzle or TR-load change
needed at this slice width, so the `global_load_tr_b128` question from revision
1 is deferred with it.

For reference, the earlier premise that WGP mode would provide 128 KiB does not
hold: `getLocalMemorySize()` does double in WGP mode
(`AMDGPUBaseInfo.cpp:1179`) but feeds only occupancy
(`AMDGPUSubtarget.cpp:49,59`); the check that rejects an oversized kernel is
`getAddressableLocalMemorySize()` = 65536 (`AMDGPUAsmPrinter.cpp:1380`), which
never doubles, and HIP reports `shared_memory_per_block = 65536`. WGP's 128 KiB
means two 64 KiB workgroups per WGP — occupancy, already ours since gfx1201
carries no `FeatureCuMode`.

## Steps

**Step 1 — measure the cross-wave S reduction in isolation. DECISION GATE.**
Unchanged from revision 1, and still the thing that decides whether any of this
is worth building. Four partial S tiles must be summed across waves through
LDS: lower bound ~16 KB per Q-tile per KV-tile (each wave contributes 2 KB and
receives 2 KB), ~32 KB for a naive all-to-all. At 128 B/clk/CU that is 128–256
cycles, against 128 WMMA saved per Q-tile. This lands squarely in the
LDS-latency regime that already dominates this kernel (see "The GCN scheduler
serializes LDS loads" in `gfx1201_fmha.md`), so it needs a measurement, not an
estimate.

**Step 2 — implement**, only if Step 1 clears:
- wave index split into `(q_tile, shard)`;
- `q_b_packs` preload restricted to the wave's own `QK_SHARD`;
- partial-S reduce through LDS + barrier;
- store guarded to one shard-wave;
- `VO_SLICE` default raised to 256 for `head_dim > 256`.

**Step 3 — measure** against 31.6 TFLOPS at head_dim 512 and 36.9 at 384, with
interleaved A/B and 3 reps. Keep the current path as correctness reference and
fallback.

## Risks

1. **S-reduction cost** — the main one; Step 1 gates it.
2. **Softmax and GEMM2 go 4x redundant.** Throughput, not latency, but it is
   pure waste and it is what revision 1 avoided. If Step 1 shows the reduction
   is cheap, revision 1 becomes more attractive than this plan.
3. **`BLOCK_M` drops to 32**, so a workgroup covers 32 Q rows: more workgroups
   and more K re-reads across them. Partly offset by 211 VGPR against 243.
4. **One extra barrier per KV iteration** (the reduce). Barriers phase-lock all
   8 waves, and a previous one-barrier restructure measured -2.7% for that
   reason.
5. Measurement discipline: this board drifts ~5% between runs and a naive
   before/after already produced one wrong conclusion (+4.8% that was really
   +0.4%). Interleave arms, 3 reps.

## Open questions for review

- **Is 192 WMMA the right stopping point?** This plan reaches it with a
  contained change but pays 4x redundant softmax + GEMM2. Revision 1 reaches
  128 WMMA with no redundancy, for a bigger change. Both need the same S
  reduction, so Step 1 gates both — worth deciding after that measurement
  rather than before.
- Approve the Step 1 gate, or go straight to implementation?
- `QK_SHARDS=4` with `BLOCK_M=32`, or `NUM_WAVES=16` (4 Q-tiles x 4 shards,
  `BLOCK_M=64`) to keep the Q tile wider?
- Confirm MLA / `hdim_qk != hdim_vo` is deferred, and that the misleading
  `head_dim_v` / `d_offset` builder parameters should be renamed to
  `VO_SLICE` / `vo_slice_offset` in the meantime, so the name stays free.
