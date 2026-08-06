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
