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
| `seqlen_q_v` (only -- see §7)                    | a sequence length; i32 by ABI already     | 1     |
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

Implemented, as the `kv_addr_hoist` knob. Section 6.1 records what the ISA
said once it was measured rather than predicted, and where the prediction in
the rest of this section was wrong.

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


### 6.1 What the ISA said

The prediction above was half right, and the wrong half was the important one.

**Right:** the row product is per-lane, so splitting it out of the divergent
offset does not restore `SelectGlobalSAddr`. Hoisting is the operative move,
not splitting.

**Wrong:** that the cost was the addressing mode. Counting inside the KV loop
body at BLOCK_DMODEL 192, pre-64-bit against 64-bit:

|                   | pre | recomputed |
| ----------------- | --- | ---------- |
| loop instructions | 575 |        649 |
| `v_mul_lo_u32`    |   3 |         14 |
| `v_add_co_u32`    |  11 |         21 |
| SGPR-base ops     |   3 |          0 |

The eleven extra multiplies are the story. `_kv_addr` clamped the *row* --
`row = in_range ? row_in_tile : seq_last - ts` -- and `ts` moves every KV
iteration, so `row * s_seq` was loop-carried and a full 64x64 multiply was
re-emitted per load per iteration. Widening it from 32 to 64 bits is what made
that expensive. The lost addressing mode came along with it but was not the
main term; saddr was already only 3 of 7 accesses before.

So the fix is to select between two whole *offsets* instead of between two
rows, which leaves `row_in_tile * s_seq` loop-invariant for LICM to hoist.

**It is not free, and not uniformly a win.** The hoisted form keeps one 64-bit
value live per cooperative load for the whole loop. Scratch per lane,
recomputed then hoisted: 192 `164 -> 44`, 256 `104 -> 160`, 384 `164 -> 328`,
512 `140 -> 124`. The sign of that change predicts the sign of the speedup at
every width on the ladder, so the form is a tuning knob and not a replacement
-- see `_KV_ADDR_HOIST_HEAD_DIMS` for the policy and its two exceptions.

Full ladder against the recomputed kernel, worst and best over
N in {1024, 4096} x {causal, not}:

| BLOCK_DMODEL |   16 |   32 |   48 |   64 |   80 |   96 |  128 |  160 |  192 |  224 |  256 |  384 |  512 |
| ------------ | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| worst        | 0.99 | 1.02 | 1.00 | 1.00 | 1.03 | 1.01 | 1.03 | 1.00 | 1.01 | 1.00 | 1.01 | 1.00 | 1.14 |
| best         | 1.03 | 1.04 | 1.00 | 1.05 | 1.05 | 1.04 | 1.05 | 1.05 | 1.41 | 1.01 | 1.02 | 1.00 | 1.24 |

The widths reading 1.00 are the ones the policy leaves recomputed; their
builds are bitwise identical to the kernel before the knob, which is why they
need no benchmark to defend.

Two process notes, both mistakes made on the way here:

- The tier 2.9 screen said the change was uniformly good (worst 0.891 at
  head_dim 256). The full ladder said 0.59 at 384. Five head dims are not a
  ladder when the effect is register pressure, because the widths that spill
  are exactly the ones the screen omits.
- Reading spill counts against the *pre-64-bit* build rather than the
  recomputed one inverted the conclusion for head_dim 512 -- it appeared to
  spill more under the hoist and win anyway, which made the mechanism look
  unexplainable. Against the right baseline it spills less. Diff against the
  build the knob actually replaces.


## 7. P3.2 negative result: the K sequence length cannot follow the Q one

`seqlen_q_v` narrowed cleanly. `seqlen_k_v` did not, and the difference is not
about the values -- both are bounded by an i32 ABI argument -- but about who
consumes them.

`seqlen_q_v` had two consumers, both pure comparisons, so narrowing it removed
two hand-written `arith.cmpi(slt, ...)` workarounds along with the width and
came out slightly *smaller*: scratch 56 -> 52 at BLOCK_DMODEL 192.

`seqlen_k_v` has five, and none of them are pure comparisons:

| consumer                       | why it stays 64-bit                             |
| ------------------------------ | ----------------------------------------------- |
| `_kv_addr`'s `ts` select       | `ts` feeds `tbase`, which multiplies by `s_seq` |
| `kv_off`'s bounds predicate    | compares against `ts` and `row_in_tile`, both addressing |
| `kv_off`'s clamp arm           | the selected row feeds `toff`'s multiply        |
| `_full_end`, `kv_upper`        | `scf.for` bounds, which are index-typed         |
| the mask column predicate      | already i32 -- this one was a round trip, now removed |

Narrowing the first three costs **160 bytes of scratch at BLOCK_DMODEL 192**,
52 -> 212, which is most of the head_dim 192 win from `kv_addr_hoist`. Two
attempts, both measured:

| variant                                       | scratch @ 192 | insts |
| --------------------------------------------- | ------------- | ----- |
| baseline (Q narrowed only)                    |            52 |  2388 |
| + narrow `ts`, `seq_last` and the predicates  |           212 |  2468 |
| + narrow the predicates only, `ts` left alone |           212 |  2465 |

The second row was the hypothesis -- that `ts` was the problem because
`tbase(fx.Index(ts))` re-widens it -- and the third row refutes it. The cost is
the *predicates*: truncating a value that is also live in 64 bits makes both
forms live at once, and at a width that already spills, that is paid in
scratch.

**Rule for the rest of P3.2:** narrow a quantity only when *every* consumer is
sequence-space. A quantity that is compared in sequence space but multiplied in
address space is an addressing quantity, and section 1 already says so -- what
this adds is that the comparison does not get to be narrowed on its own either.


## 8. Phase 3 outcome: the site count was the wrong unit

P3.1 classified 100 `fx.Index(` sites and concluded 73 should become
`fx.Int32`. Four commits later the file still has 98, and that is the right
answer rather than an unfinished one.

What the classification got wrong is that it sorted sites by *what the quantity
is* -- a sequence position, a column index -- when the property that decides
the width is *what it flows into*. Re-audited on that basis:

| group (P3.1 §2)           | sites | verdict                                              |
| ------------------------- | ----- | ---------------------------------------------------- |
| `seqlen_q_v`              |     1 | narrowed, but only the uniform compare (§8.1)        |
| `seqlen_k_v`, `seq_last`  |     2 | refuted -- §7                                        |
| tile / row origins        |   ~12 | feed `tbase`/`o_tbase`; addressing by §1             |
| shard + column indices    |   ~22 | summed into a 64-bit offset in `toff`                |
| KV region bounds          |   ~15 | **already i32** -- the gSWA work converted them      |
| in-tile row / lane offsets|   ~20 | feed `row * s_seq`; addressing by §1                 |

So the 73 was really about 16, of which 15 were already done before P3.2
started and the last one is half-done by choice.

### 8.1 The rule, restated

Truncating a value that stays live in 64 bits does not remove a value, it adds
one. Both forms are then live, and at any width that already spills, that is
paid in scratch. Three independent measurements of the same effect:

| change                                     | cost                                    |
| ------------------------------------------ | --------------------------------------- |
| narrow `ts` / `seq_last` / the KV predicate | +160 B scratch @ BLOCK_DMODEL 192       |
| narrow the KV predicate only                | +160 B scratch @ BLOCK_DMODEL 192       |
| narrow the per-lane Q row compare           | +16 B scratch and -6% @ 384 causal      |

The one narrowing that stuck -- `_alive` -- is the one whose operand is uniform
and has no 64-bit consumer. That is the test: **narrow a quantity only when
every consumer is sequence-space**, not when the value merely fits.

### 8.2 What Phase 3 did deliver

Two bitwise-identical cleanups and one small one:

- `_scmp_i32` deleted. Its docstring named its own exit condition and the
  condition was met; 14 call sites became plain `<`/`>`/`==`/`!=`.
- `_ssel_i32`'s operand coercion dropped. The helper stays, against the plan:
  `ArithValue.select` returns a raw MLIR value, so the `fx.Int32` around the
  *result* is load-bearing and deleting it would lengthen 14 call sites.
  `_smin_i32`/`_smax_i32` became one-liners as a side effect.
- the K sequence length's widen-then-truncate round trip removed.

P3.3 is therefore complete and P3.2 is closed as mostly-not-applicable. The
`fx.Int64(a * b)` hazard the phase was most worried about (§4) never appeared:
the file still has zero `fx.Int64(` sites containing a multiply.
