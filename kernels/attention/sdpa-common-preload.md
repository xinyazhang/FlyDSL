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

Named `LoadedTile` / `load_tile`, not `Preloaded*`: "pre-" describes *when the
caller runs it* -- before the KV loop -- which is the caller's business and not
true of every use. `bwd_inner_dq` loads its operands inside the loop.

The forward Q preload today is 40 lines producing four values, two of which
are consumed 1000 lines away (`q_row_i32s` in the causal mask,
`q_in_bounds_all` in the epilogue store). A naive extraction would take ten
parameters and return four lists -- a wide interface in both directions, which
is why this needed a design rather than a `git mv`.

The fix is to return **one object** and to stop returning what the caller can
recompute:

    @dataclass(frozen=True, slots=True)
    class LoadedTile:
        packs: list[list[fx.Vector]]  # [row_tile][k_step], masked WMMA operand
        rows: list[fx.Index]          # absolute row per row-tile
        rows_i32: list[fx.Int32]      # the same rows, signed, for the mask
        in_bounds: list[ArithValue]   # i1 per row-tile

    def load_tile(
        ptr, tile_base, toff, *,        # from make_addr_pair
        rows_axis, cols_axis,           # two MaskedAxis
        rows, rows_in_tile,             # list[fx.Index], caller-computed
        col_base, k_steps, k_step,      # column walk
        elem_dtype, width=8,
        transposed=False,
    ) -> LoadedTile

The types were read off a traced build rather than inferred. **`rows` and
`rows_i32` are the same numbers at two widths, and both are needed**:
addressing wants the 64-bit `fx.Index`, because `toff` multiplies it by
`stride_seq` and that product does not fit in 32 bits on a non-compact tensor;
the causal mask wants the signed `fx.Int32`, because `fx.Index` is unsigned and
a negative bound would compare as enormous. Neither can be dropped in favour of
a cast at the point of use -- see the live-range note below.

`in_bounds` is an `ArithValue` and not a raw `ir.Value` because `MaskedAxis.valid`
returns one; `packs` is `fx.Vector` because the masking select does.

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
- **No Python object live across an `scf.if`.** `LoadedTile` is returned
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

1. `LoadedTile` + `load_tile` in the module, forward Q switched to it.
2. Fold the K/V cooperative loads onto the same routine *if* they fit -- they
   differ by using LDS staging rather than register packing, so this may turn
   out to be a different routine that shares only `MaskedAxis`. Decide after
   (1), not now.
3. `transposed=True` support, unused by fwd, added only when the bwd kernel
   that needs it exists. **Not before**: an untested parameter with no caller
   is a liability, and `bwd_kernel_dk_dv` is the design's justification, not
   its current consumer.


# Part II: `Tile` -- on-chip data, residence, and movement

Extends §3. Reviewed against the kernel; the survey findings are in §8-§10 and
one of them changes the shape of the proposal.

## 7. The concern: a `Tile` cannot own its VGPRs

The proposal is `TileInfo` carrying "vram region, lds (optional), vgprs". The
first two are fine. **The third cannot be a field.**

Loaded values are traced MLIR values, so an object holding them is a Python
object holding traced state -- and this module has a hard rule about those: no
variable of any kind may hold a non-MLIR Python object that is live across a
dynamic `if`. The AST rewriter collects such variables as `scf` state, which
must be MLIR-backed. That was established three ways earlier: `UnboundLocalError`
at BLOCK_DMODEL 16, `state variable 'fastmath' is FastMath, not an MLIR Value`,
and -- the one that matters here -- passing the object as a *parameter* fails
too, at 16 causal.

This is not a corner case for this design. It is the exact site the design
targets: `coop_load_store_k` reads its operands inside `if row_valid:` whenever
`KV_NEEDS_GUARD`, which is true for every BLOCK_DMODEL whose load geometry does
not tile BLOCK_N exactly -- 16 among them.

**So split description from data:**

    @dataclass(frozen=True, slots=True)
    class TileInfo:
        """*Where* a tile lives and what shape it is. No traced values."""
        rows: int; cols: int          # tile extent, const_expr
        num_row_subtiles: int         # see §9 on the name
        lds_base: int | None          # element offset, None if never staged
        lds_stride: int | None
        vec_width: int                # elements per access

`TileInfo` is all `const_expr` ints, so it is safe anywhere -- it can be built
host-side, like `FastMath`. The **values** stay where they are today: plain
lists returned from and passed to the movement functions. A `Tile` that owns
both would be unusable at the one call site that most needs it.

## 8. Residence and movement

Movement functions take a `TileInfo` and explicit values; they do not mutate.

| direction    | gfx1201 today                                    | status |
| ------------ | ------------------------------------------------ | ------ |
| vram -> vgpr | `load_global_f16xN` / `global_load_tr_v8`        | exists |
| vram -> lds  | **load to vgpr, then store** -- no direct path    | worth naming anyway |
| lds -> vgpr  | `lds_load_v8`                                    | exists |
| vgpr -> lds  | `lds_store_vx`                                   | exists |
| vgpr -> vram | `_store_global_half` (the O epilogue)            | exists |

`vram -> lds` earns a name **because** gfx1201 lacks the instruction. The
kernel already spells the fused form by hand --
`coop_store_v_lds(coop_load_v_global(kv_block_start, 0), 0)` -- and gfx950 and
gfx1250 have a direct path. Naming it `stage_vram_to_lds(info, ...)` lets the
gfx1201 implementation stay load-then-store while the other two substitute one
instruction, without the *caller* changing. That is the whole value: the seam
is placed where the ISA difference is.

`lds -> vram` is not implemented. No caller.

## 9. `NUM_SUB_TILES` collides -- both names need an axis

`N_SUB_TILES` already exists and means something else:

    K_SUB_N     = 2 * WMMA_N          # KV columns per sub-tile
    N_SUB_TILES = BLOCK_N // K_SUB_N  # sub-tiles per KV block, along columns
    Q_ROW_TILES                       # row sub-tiles per wave, along rows

They are sub-tiles of different blocks along different axes, so
`TileInfo.NUM_SUB_TILES` for the Q one would make the confusion worse rather
than better. Answering "sub against what?" for both:

| now           | proposed              | reads as                              |
| ------------- | --------------------- | ------------------------------------- |
| `Q_ROW_TILES` | `ROW_SUBTILES`        | row sub-tiles of the Q block, per wave |
| `N_SUB_TILES` | `COL_SUBTILES`        | column sub-tiles of the KV block       |
| `K_SUB_N`     | `COLS_PER_SUBTILE`    | its unit, which is what it already is  |

`TileInfo.num_row_subtiles` then has an unambiguous referent. Renaming
`Q_ROW_TILES` also touches the knob (`FmhaKnobs.q_row_tiles`) and its tuning
table, so it is its own commit, ahead of any `Tile` work.

## 10. Folding the `for qt` loops: only two of three kinds

Fifteen sites, and they are not one pattern:

**(a) Map over row-tiles -- foldable.** `s_accs_all[qt] = reduce_...(s_accs_all[qt], ...)`
and the list comprehensions building per-row-tile lists. A
`TileInfo.map_subtiles(fn, *lists)` collapses these to one line each.

**(b) Large per-row-tile bodies -- foldable, low value.** Softmax update, P
pack, the LSE and O epilogues. Folding buys the loop header and nothing else;
the body is the algorithm.

**(c) Schedule-bound interleaving -- must NOT fold.** In GEMM1 and GEMM2 the
`qt` loop is the *innermost* one, inside the `ks` / V-chunk unroll:

        for st_idx ...:
            for qt in range_constexpr(Q_ROW_TILES):
                s_accs_all[qt][..] = wmma_acc(k_pack_a, q_packs[qt][ks], ...)

That nesting order is the optimisation -- it is what `q_row_tiles=2` buys, each
K operand feeding every row-tile while it is live in registers. A visitor that
owns the iteration would be free to reorder it, and reordering it discards the
reuse. These two sites keep their explicit loops, and the reason belongs in a
comment there.

**Verdict on the visitor question:** a `map_subtiles` helper for (a) is worth
having; (b) is cosmetic; (c) is off limits. That is 5 of 15 sites, which is a
smaller win than the count suggests -- worth doing after `load_tile`, not
instead of it.

## 11. Revised sequencing

1. Rename `Q_ROW_TILES` -> `ROW_SUBTILES`, `N_SUB_TILES` -> `COL_SUBTILES`,
   `K_SUB_N` -> `COLS_PER_SUBTILE` (§9). Mechanical, bitwise, knob included.
2. `load_tile` per §3, unchanged by this part.
3. `TileInfo` as the const_expr descriptor (§7), threaded through `load_tile`
   and the LDS helpers.
4. `stage_vram_to_lds` (§8), replacing the hand-fused
   `coop_store_v_lds(coop_load_v_global(...))`.
5. `map_subtiles` for the (a) sites only (§10).

Steps 3-5 are the `Tile` design proper and each is independently gateable.
Step 1 first because every later step reads better in the new names.
