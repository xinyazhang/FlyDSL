# Plan: wide head dimensions on gfx1201 FMHA

Status: **draft for review, not started.**

Goal: make large head dimensions fast on `flash_attn_func_gfx1201.py`, and put
`hdim_qk != hdim_vo` on a correct footing rather than the slicing hack that is
in the tree today.

## Terminology

Four distinct quantities that earlier notes (and the current code) conflate.
Use these names from here on.

| name | meaning |
|---|---|
| `hdim_qk` | Head dimension of the **Q and K** tensors. The GEMM1 reduction length. |
| `hdim_vo` | Head dimension of the **V and Out** tensors. The GEMM2 output width. |
| `QK_SHARD` | Sub-division of `hdim_qk` that one wave reduces over in GEMM1. |
| `VO_SHARD` | Sub-division of `hdim_vo` that one wave accumulates in GEMM2. |

`hdim_qk` and `hdim_vo` are **tensor** properties: `V` and `Out` both have
last dimension `hdim_vo`, and the accumulator's shape is implied by `Out`, so
`o_accs` is sized from `hdim_vo` (or from `VO_SHARD` once sharded), never from
`hdim_qk`. `QK_SHARD` and `VO_SHARD` are **kernel tiling** choices and are
independent of each other and of the tensor shapes.

MLA-style shapes have `hdim_qk != hdim_vo` (e.g. 576 / 512). Self-attention has
them equal. Both must work.

### What is in the tree today is *not* `hdim_vo`

The builder currently takes `head_dim_v` + `d_offset`. Those are a **column
slice of a V tensor that is `hdim_qk` wide** -- V's row stride is still
`num_heads * head_dim`, and `Out` is still allocated `hdim_qk` wide. That is
correct for the job it does (splitting a wide self-attention head across
launches) but it is *not* `hdim_vo`, and exposing it publicly under that name
would silently misread a real MLA V tensor, whose stride is
`num_heads * hdim_vo`.

Action: rename `head_dim_v` -> `VO_SLICE` and `d_offset` -> `vo_slice_offset`
to free the name, before introducing a true `hdim_vo`.

## Correction: WGP mode does not raise the per-workgroup LDS ceiling

A WGP is 2 CUs with 128 KiB of LDS, and LLVM models that:

```cpp
// getLocalMemorySize(), AMDGPUBaseInfo.cpp:1179
if (isGFX10Plus(STI) && !STI.getFeatureBits().test(FeatureCuMode))
    BytesPerCU *= 2;                       // 128 KiB in WGP mode
```

But that doubled value is read **only** at `AMDGPUSubtarget.cpp:49` and `:59`,
both occupancy calculations. The limit that rejects an oversized kernel is a
different one, and it never doubles:

```cpp
// AMDGPUAsmPrinter.cpp:1380
if (MFI->getLDSSize() > STM.getAddressableLocalMemorySize()) {   // 65536
```

HIP agrees: `shared_memory_per_block = 65536` on this device. Nabu confirms a
hard per-workgroup cap rooted in the DS instruction's 16-bit offset field,
though it could not cite an ISA section and was asked a leading question, so
weight it below the other two.

So WGP's 128 KiB buys **two 64 KiB workgroups resident per WGP** -- occupancy,
not a bigger single workgroup. gfx1201 already runs in WGP mode (no
`FeatureCuMode` in the gfx12 feature list), so that benefit is already ours.

**This does not block the design.** K+V at full 512 width is 66048 B, over by
exactly 512 -- which is precisely the `+4` element padding. See LDS below.

## Current state

`hdim_qk == hdim_vo`, 16..512, all verified against SDPA. Above 256 the
interface loops over 128-wide V column slices, one launch each, because
attention is column-separable in V (`O[:, s] = P @ V[:, s]`, and P does not
depend on V).

| hdim | slices | VGPR | spills | TFLOPS |
|---|---|---|---|---|
| 256 | 1 | 256 | 36 | 67.5 |
| 384 | 3 | 243 | 0 | 36.9 |
| 512 | 4 | 243 | 0 | 31.6 |

The problem: each launch redoes the **whole** GEMM1, because every slice needs
the full P.

| per Q-tile per KV-tile, hdim 512 | GEMM1 | GEMM2 | total WMMA |
|---|---|---|---|
| today (4 launches) | 4 x 64 = 256 | 64 | **320** |
| ideal | 64 | 64 | **128** |

## Proposed design: shard both dimensions across waves

Waves are indexed `(q_tile, shard)`. With `NUM_WAVES = 8` and 4 shards, that is
2 Q-tiles x 4 shards, so `BLOCK_M = 32`.

**GEMM1 shards along the reduction.** `S[i,j] = sum_d Q[i,d] * K[j,d]`, so wave
*s* computes a partial sum over its own `QK_SHARD` and the shards are reduced
through LDS. Each wave ends up holding the full S and runs softmax on it.

**GEMM2 shards along the output.** Wave *s* accumulates
`O[:, VO_SHARD_s] += P @ V[:, VO_SHARD_s]`.

The two shard widths are independent, which is what makes `hdim_qk != hdim_vo`
fall out of the same mechanism instead of needing a special case.

### Budgets (hdim_qk = hdim_vo = 512, BLOCK_N = 32, 4 shards)

| per wave | today | sharded |
|---|---|---|
| `q_b_packs` | 128 | **32** (own `QK_SHARD` only) |
| `o_accs` | 64 | 64 (`VO_SHARD`/2) |
| `s_accs` + `p_vals` + misc | ~51 | ~56 |
| **total VGPR** | 243 | **~152** |

| WMMA per Q-tile per KV-tile | GEMM1 | GEMM2 | total |
|---|---|---|---|
| today | 256 | 64 | 320 |
| sharded | 64 | 64 | **128** |

### LDS

Each wave touches a **disjoint** slice of both K and V, so collectively each is
read exactly once -- no amplification from sharding. Two ways to fit 64 KiB:

- **(a) V straight from global via `global_load_tr_b128`, K in LDS** ->
  **33024 B**. Large headroom; the TR machinery already exists in the bp
  kernel, which stages V transposed that way. **Preferred.**
- (b) Both in LDS, XOR swizzle replacing the `+4` padding -> 32768 x 2 =
  65536 exactly. Fits with zero margin.

## Steps

**Step 0 -- rename, then introduce a true `hdim_vo`.** *Not independent of the
rest, contrary to an earlier note.* `V` and `Out` carry `hdim_vo` as their real
last dimension, so this touches:

- a second token stride: `STRIDE_TOKEN_VO = num_heads * hdim_vo`, distinct from
  `STRIDE_TOKEN_QK`, used for every V and Out address;
- `Out` allocation in the interface at `hdim_vo`, not `hdim_qk`;
- `o_accs` sized from `hdim_vo` (implied by `Out`), not from `hdim_qk`;
- interface validation: `q.shape[-1] == k.shape[-1] == hdim_qk`,
  `v.shape[-1] == hdim_vo`, output last dim `hdim_vo`;
- the existing `VO_SLICE`/`vo_slice_offset` become a sub-division *of
  `hdim_vo`*, which is the correct relationship.

Worth landing on its own -- it makes MLA shapes correct and single-launch --
but it is a real change, not a rename.

**Step 1 -- measure the cross-wave S reduction in isolation. DECISION GATE.**
Lower bound is ~16 KB per Q-tile per KV-tile (each wave contributes 2 KB and
receives 2 KB); a naive all-to-all is ~32 KB. At 128 B/clk/CU that is 128-256
cycles against ~192 WMMA saved. **This could consume the entire gain**, and it
lands squarely in the LDS-latency regime that already dominates this kernel
(see "The GCN scheduler serializes LDS loads" in `gfx1201_fmha.md`). Measure
before building.

**Step 2 -- implement**, only if Step 1 clears: wave indexing, partial-S reduce,
V via TR loads, shard-aware epilogue.

**Step 3 -- measure** against 31.6 TFLOPS at hdim 512, keeping the slice-per-
launch path as correctness reference and fallback.

## Risks

1. **S-reduction cost** -- the main one; Step 1 gates it.
2. **`BLOCK_M` drops to 32**, so a workgroup covers 32 Q rows. More workgroups,
   more K re-reads across them. Partly offset by better occupancy (152 VGPR
   against 243).
3. **Softmax goes 4x redundant** -- each wave softmaxes the same reduced S.
   Costs throughput, not latency, since the waves sit on different SIMDs.
4. **Two extra barriers per KV iteration** (reduce, broadcast). Barriers
   phase-lock all 8 waves, and a previous one-barrier restructure measured
   -2.7% for exactly this reason.
5. Everything here is measured with **interleaved A/B, 3 reps** -- this board
   drifts ~5% between runs and a naive before/after already produced one wrong
   conclusion (+4.8% that was really +0.4%).

## Open questions for review

- Approve the **Step 1 gate**, or go straight to implementation?
- **(a) V-via-TR** or **(b) swizzle** for the LDS budget?
- Should Step 0 (`hdim_vo`) land before Step 1, given it is a prerequisite for
  MLA shapes but not for the sharding measurement?
- Is `BLOCK_M = 32` acceptable, or should `NUM_WAVES` rise to 16 (4 Q-tiles x 4
  shards, `BLOCK_M = 64`) to keep the Q tile wider?
