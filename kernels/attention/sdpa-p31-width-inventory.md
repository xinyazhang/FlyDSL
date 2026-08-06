# P3.1 -- integer width inventory

Reviewable artifact required before P3.2 touches any type. The plan's §5.2 rule:
sequence-space quantities are `i32` because they fit; addressing casts to 64
bits, and **the cast goes before the multiply, not after**.

## 1. Quantities that genuinely need 64 bits

Everything not on this list is `i32`. Each entry names the product that
overflows, because "it's an address" is not a reason on its own -- most of the
addressing arithmetic here is bounded by a tile.

| quantity                                   | overflows because                                                     | measured / bounded            |
| ------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------- |
| Philox plane base `off_zh * Sq * stride`   | reaches 2^32 at B*H = 256 with 8K sequences                            | already `fx.Int64`, commented |
| `max_seqlen_? * STRIDE_TOKEN`              | grows with the tensor; 8K x (H*D) exceeds 2^31 at H=8 D=128 f16        | line 1189-1190                |
| `tbase` / `bh + seq_start * s_seq`         | element offset into the whole tensor -- the thing 64-bit addressing is for | line 1258            |
| the six `stride_*` launch arguments        | element strides of a multi-GB tensor                                    | already `fx.Int64` in the ABI |
| `_q_batch` / `_k_batch` / `_?_row_off`     | batch-major element offsets, same product as `tbase`                    | line 1226-1228                |

## 2. Quantities that are bounded and must become `i32`

| group                                            | bound                                   | sites |
| -------------------------------------------------- | ----------------------------------------- | ------- |
| `seqlen_q_v`, `seqlen_k_v`, `seq_last`           | a sequence length; i32 by ABI already     | 3     |
| `start_q`, `start_k`, `q_tile_idx`, `_ntiles`    | < max_seqlen, itself an i32 argument      | ~12   |
| `shard_qk_off`, `shard_vo_off`                   | **measured max 384** (§11.2)              | 2     |
| head_dim column indices (`q_col`, `k_col`, `d_*`)| < BLOCK_DMODEL <= 512                     | ~20   |
| KV region bounds (`_v_lo/_v_hi/_fb_*/_lb_*/_rb_*`)| block indices, < ceil(8K / 32) = 256      | ~15   |
| in-tile row/lane offsets                         | < BLOCK_N or < 32                         | ~20    |

## 3. The 100 `fx.Index(` sites, classified

| class          | count | disposition                                                        |
| ---------------- | ------- | -------------------------------------------------------------------- |
| grid / launch  | 5     | **stay `fx.Index`** -- `gpu.block_idx` and `.launch()` want index   |
| addressing     | 22    | **stay 64-bit**, per §1                                             |
| sequence/other | 73    | **-> `fx.Int32`**, per §2                                           |

## 4. What P3.2 must not do

The hazard is not narrowing something too far -- §11.2 shows the bounds are
comfortable. It is widening in the wrong order:

```python
fx.Int64(a * b)              # WRONG: multiplies in 32 bits, then widens
fx.Int64(a) * fx.Int64(b)    # right: widens, then multiplies
```

Both compile. The tensor big enough to need the 64-bit product is exactly the
one where the first form has already wrapped, and **no existing test reaches
it** -- the suite's largest shape is far below B*H = 256 at 8K, so both orders
pass. Today the file has 7 `fx.Int64(` sites and exactly one contains a
multiply, bounded by 24. P3.2's gate is that this count stays at zero-that-
matter, checked by grepping every new `fx.Int64(` for a multiply inside it,
plus a host-side check of the resolved offsets at a synthetic large shape.


## 5. Correction: `stride_seq` is not bounded by `num_heads * head_dim`

Added after the fact, because §1 originally justified leaving the per-lane
offset at 32 bits with "no physical layout reaches 2**31 there". That is
**false**, and the reasoning behind it was: input tensors are not required to
be compact, and a view keeps the strides of its source.

Slicing eight heads out of a `(1, 64, 16384, 512)` f16 tensor -- 1 GiB, nothing
exotic -- gives a view whose `stride(1)` is 8388608. At `BLOCK_M = 256` the
per-lane row offset is exactly 2**31.

So `toff`'s 32-bit product is a genuine restriction on legitimate input, and
`_strides_of` currently converts it into a rejection. That is the right
behaviour versus silent corruption and the wrong end state. **P3.2 owns
removing it**, and the two candidates are:

- widen the per-lane offset to 64 bits, which costs VALU on every address in
  the inner loop -- measure before assuming it is affordable; or
- fold `row_in_tile * s_seq` into the 64-bit base. The obstacle is that the
  base is uniform per workgroup and the row is per-lane, so this needs the row
  term hoisted to wherever the lane's row is already known.

**The first was done and measured.** Two truncations were throwing the high
half away -- `fx.Int32(off32)` in `_split_ptr`, and an
`arith.index_cast(T.i32, ...)` in `_global_load_tr_v8`. `s_seq` is already
`fx.Index`, so the products were always computed in 64 bits; only these casts
narrowed them. Removing both makes all three axes of
`test_element_offsets_past_2gi` pass.

It costs, and the cost is why the second option still matters:

| head_dim | 64    | 80    | 128   | 192       | 256   |
| -------- | ----- | ----- | ----- | --------- | ----- |
| ratio    | 0.983 | 0.965 | 0.965 | **0.911** | 0.974 |

Against a 0.993 self-test floor, so every one of those is real. The instruction
counts barely move (2020 -> 2044 at head_dim 128) -- this is not code size but
addressing mode: a 64-bit divergent offset stops LLVM's SelectGlobalSAddr
keeping the base in SGPRs, so the address becomes a VGPR pair.

**So option two is now the task, not an alternative.** Keep the divergent
offset 32-bit and carrying only `col` -- bounded by BLOCK_DMODEL <= 512 -- and
move `row_in_tile * s_seq` into the 64-bit term. The obstacle remains that the
base is uniform per workgroup while the row is per-lane, so it needs a third
address component: uniform-64 + divergent-64-row + divergent-32-col, or the row
folded per-lane once outside the loop rather than per address inside it.


## 6. Recovering the cost: hoist the row term, do not merely split it

Analysed, not yet implemented.

The proposal was to pull `row * s_seq` out of `toff` so the address becomes
`u64 base + u64 row + i32 col`, on the theory that the row term is a scalar.
**It is not.** `q_rows_in_tile = wave_q_offset + qt * WMMA_M + lane16`, and
`lane16 = lane % 16`, so the row is per-lane and the product is a *vector*
u64. Splitting alone therefore still leaves a 64-bit divergent operand, which
is precisely what stops LLVM's `SelectGlobalSAddr` keeping the base in SGPRs --
so it would not recover the 9%.

The half that does work is that the term is **loop-invariant**: it depends only
on `lane16`, `wave_q_offset` and `qt`, all fixed for the kernel's lifetime.

    per-lane u64 base = tile_base(uniform) + rows_in_tile * s_seq   # once
    each access       = base + col                                  # i32

so the 64-bit multiply and add are paid once rather than per address, and a
small constant `col` can fold into the load's immediate-offset field and cost
no VALU at all.

The correctness half of the proposal is right and is what makes the split
legal: **only `row * s_seq` can exceed i32.** `col` is a head_dim column index,
bounded by `BLOCK_DMODEL <= 512`, so it stays 32-bit by proof rather than by
assumption -- which is the property the original code lacked.

Two things to check when implementing. The K/V side re-bases every KV
iteration, so its uniform term changes in the loop while its row term does not;
the hoist still applies but to a different split point. And the win is a
prediction, not a measurement -- the tier-1 fingerprint will say whether the
addressing mode came back, and only tier 2.9 says whether the 9% did.
