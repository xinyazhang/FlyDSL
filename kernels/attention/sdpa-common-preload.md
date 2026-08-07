# Design: a shared tile preload for `fmha_common_gfx1201`

Tier C's second half. Deferred out of the Tier C commits because, unlike the
region algebra, this one needs a design before it needs an edit.

## 1. Why it is worth doing, from AOTriton

The forward Q preload is not a forward-only routine. AOTriton's equivalent
pair, `composed_ptrs` + `composed_load`, is imported by **every one of its
eleven kernel files**:

| file                    | preloads                        |
| ----------------------- | ------------------------------- |
| `fwd_kernel.py`         | Q                               |
| `fwd_kernel_inner.py`   | K, V per tile                   |
| `bwd_kernel_dq.py`      | Q, dO                           |
| `bwd_kernel_dk_dv.py`   | **K^T, V^T**                    |
| `bwd_kernel_fuse.py`    | Q, dO, O, and K^T/V^T           |
| `bwd_inner_*.py` (x3)   | the per-tile operands           |
| `bwd_preprocess.py`     | O, dO                           |
| `bwd_postprocess.py`    | accumulators                    |

Two things follow. The abstraction is **"preload a tile of *some* tensor into
WMMA operand registers"**, not "preload Q" -- `bwd_kernel_dk_dv` wants exactly
this shape with K and V substituted and the transpose flipped. And it is the
single most-reused routine in that codebase, which is the strongest available
evidence that it belongs in a shared module rather than in one kernel.

## 2. What AOTriton's version parameterises, and what it maps to here

    composed_ptrs(T, stride_z, stride_h, stride_n, stride_k,
                  batch, head, row_offsets, D0, D1, D2, TRANSPOSED)
    composed_load(p0, p1, p2, row_offsets, D0, D1, D2,
                  seqlen, hdim, other=0.0,
                  PADDED_ROW, PADDED_COL, TRANSPOSED)

| their knob            | our equivalent                                        |
| --------------------- | ----------------------------------------------------- |
| `stride_*` + batch/head | `fmha.make_addr_pair(...)` -- already extracted     |
| `D0, D1, D2`          | **not needed.** They split BLOCK_DMODEL into three chunks because Triton wants power-of-two blocks; we have a BLOCK_DMODEL ladder and V column slicing instead |
| `PADDED_ROW`, `seqlen`| `MaskedAxis(seqlen_q)` -- already extracted           |
| `PADDED_COL`, `hdim`  | `MaskedAxis(hdim, active=PADDED_HEAD)` -- ditto       |
| `other=0.0`           | the zero vector the discard selects                    |
| `TRANSPOSED`          | our `fmha.global_load_tr_v8` vs `load_global_f16xN`   |

**Most of the parameterisation is already extracted.** What is missing is only
the loop that walks a lane's rows and column steps and packs the result --
which is why this is a smaller job than its 40 lines suggest, and why it is
worth doing as a shared routine rather than a Q-shaped one.

## 3. The interface

The forward Q preload today is 40 lines producing four values, two of which
are consumed 1000 lines away (`q_row_i32s` in the causal mask,
`q_in_bounds_all` in the epilogue store). A naive extraction would take ten
parameters and return four lists -- a wide interface in both directions, which
is why this needed a design rather than a `git mv`.

The fix is to return **one object** and to stop returning what the caller can
recompute:

    @dataclass(frozen=True, slots=True)
    class PreloadedTile:
        packs: list          # [row_tile][k_step] -> WMMA operand, masked
        rows: list           # absolute row per row-tile, fx.Index
        rows_i32: list       # the i32 copy, for masks that want signed
        in_bounds: list      # per row-tile i1

    def preload_tile(
        ptr, tile_base, toff, *,        # from make_addr_pair
        rows_axis, cols_axis,           # two MaskedAxis
        rows, rows_in_tile,             # per row-tile, caller-computed
        col_base, k_steps, k_step,      # column walk
        elem_dtype, width=8,
        transposed=False,
    ) -> PreloadedTile

`rows` and `rows_in_tile` stay caller-computed: they encode the wave-to-row
mapping, which is a *schedule* decision (`Q_ROW_TILES`, `wave_q_offset`,
`WMMA_M`) and differs between fwd and bwd. Passing them keeps the schedule in
the kernel and the memory shape in the module -- the same split the rest of
this module already draws.

`rows_i32` is returned rather than recomputed by the caller, because there is a
measured reason it must be built at a particular point: computing it *before*
the bounds test starts its live range early and cost 16 bytes of scratch and
6% at BLOCK_DMODEL 384 causal. That constraint has to travel with the code, so
the routine owns both and the docstring says why.

## 4. Constraints this must respect

Every one of these is a rule already paid for elsewhere in the session:

- **No dynamic `if`.** The routine may branch only on `const_expr` values.
  The Q preload qualifies: its only conditionals are over `PADDED_HEAD` and
  the tile counts.
- **No Python object live across an `scf.if`.** `PreloadedTile` is returned
  and unpacked immediately, like `CausalRegions`. The `MaskedAxis` arguments
  arrive as *factory calls* at the call site, not as captured variables.
- **`width` is per-access, not per-axis.** The Q preload is 8 wide because
  `load_global_v8f16` matches the WMMA operand; the cooperative loads are
  `VEC_WIDTH`. They are both 8 today from unrelated definitions -- do not
  bind them together again.
- **The i32 rule for anything mask-facing.** `rows_i32` feeds the causal mask,
  where `fx.Index` being unsigned would make a negative bound enormous.

## 5. Gating

The Q preload runs in every build, so the usual configs cover it -- but the
*arms* need choosing deliberately, which is where three near-misses came from
this session:

| config             | exercises                                  |
| ------------------ | ------------------------------------------ |
| hd128 non-causal   | the base path                              |
| hd128 causal       | `rows_i32` reaching the mask               |
| hd100 padded       | `cols_axis.active`                         |
| hd80               | `Q_ROW_TILES == 2`, the only width that takes it |
| hd384 causal       | the spilling build where the `rows_i32` live-range effect was measured |

Bitwise identical at all five, then the full suite. If hd384 causal moves,
the live-range constraint in §3 has been violated and the fix is to move the
`rows_i32` construction, not to accept the number.

## 6. Sequencing

Three commits, because the first two are mechanical and the third is where the
risk is:

1. `PreloadedTile` + `preload_tile` in the module, forward Q switched to it.
2. Fold the K/V cooperative loads onto the same routine *if* they fit -- they
   differ by using LDS staging rather than register packing, so this may turn
   out to be a different routine that shares only `MaskedAxis`. Decide after
   (1), not now.
3. `transposed=True` support, unused by fwd, added only when the bwd kernel
   that needs it exists. **Not before**: an untested parameter with no caller
   is a liability, and `bwd_kernel_dk_dv` is the design's justification, not
   its current consumer.
