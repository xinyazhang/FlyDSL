# Can the KV loop, the causal split and the preload move to a library?

Survey only. Measurements, then a recommendation per candidate.

## 0. Summary

| candidate | shareable? | needs `@flyc.jit`? | recoverable |
|---|---|---|---|
| KV loop body, fwd <-> bwd | **no** -- 0.06--0.24 structural similarity | n/a | 0 |
| KV loop body, fuse's two roles | **yes** -- 0.85, same file | no | ~40 |
| causal-region setup | **yes** -- ~20 lines x3 | **yes** | ~40 |
| masked-column map | **yes** -- already known, x4 | **yes** | ~8 |
| Q/K/V preload | **already done** | -- | ~0 |

The headline is the first row, and it contradicts the guess that motivated the
survey.

## 1. The KV loop does not generalise

Nested loop bodies, anonymised (identifiers to `V`, strings to `S`) and
compared by AST shape:

| pair | lines | ratio |
|---|---|---|
| `fwd.kv_loop_body` <-> `dq.kv_loop_body` | 375 vs 179 | **0.18** |
| `fwd.kv_loop_body` <-> `fuse.kv_body` | 375 vs 50 | 0.08 |
| `fwd.kv_loop_body` <-> `fuse.q_body` | 375 vs 93 | 0.06 |
| `dq.kv_loop_body` <-> `fuse.kv_body` | 179 vs 50 | 0.19 |
| `dq.kv_loop_body` <-> `fuse.q_body` | 179 vs 93 | 0.24 |
| `fuse.kv_body` <-> `fuse.q_body` | 50 vs 93 | **0.85** |

For reference, the host helpers that *did* factor cleanly measured 0.82--1.00.
0.18 is not a near-miss; it is a different function.

That is not an accident of style. The three loops carry different mathematics:

- **forward** -- QK, then online softmax (running max, rescale of the
  accumulator and of `l`), then PV. Its loop-carried state is `(m, l, O)`.
- **dq** -- QK, then *recompute* P from the stored LSE with no running max at
  all, then dS = P*(dP - delta), then dQ. Loop-carried state is just the dQ
  accumulator.
- **dkdv** -- runs transposed: one wave owns 16 KV rows, K/V stay register-
  resident, and Q/dO stream past. Its loop is over *Q* tiles, not KV.

A shared "KV loop" would need the union of three carried-state shapes and a
policy argument selecting between three unrelated bodies. That is a `switch`
wearing a function's clothes, and it would make the forward's schedule -- the
thing the whole tuning effort is about -- harder to read and riskier to touch.

**Recommendation: do not.** What *is* shared already is shared: the GEMM atom
(`fmha.wmma_acc`), staging (`stage`, `publish`, `read_batches`, `reader`),
addressing (`make_addr_pair`, `split_ptr`), masking geometry
(`decompose_causal_regions`), and softmax fastmath (`FastMath`). The loop is
the part that composes them differently per kernel, which is the part that
should stay visible.

**The one real duplicate is inside `fuse`**: `kv_body` and `q_body` at 0.85, in
the same file, being the dK/dV and dQ halves of the fused kernel. ~40 lines.
Worth doing, and it needs no library -- one local helper with a role flag.

## 2. The causal-region setup does generalise

`fmha.decompose_causal_regions` already exists; what is copied is the ~20 lines
that *drive* it. fwd L1296--1314 and dq L668--688 are the same block modulo one
unused field, and fuse has a variant at L763 and L919:

```python
if const_expr(CAUSAL):
    _regions = fmha.decompose_causal_regions(start_q, seqlen_q_i32, seqlen_k_i32,
                                             _wl_i32, _wr_i32, BLOCK_M, BLOCK_N, _alive)
    _BN_I32 = fx.Int32(BLOCK_N)
    _n_l, _n_f, _n_r = _regions.n_left, _regions.n_full, _regions.n_right
    _l_col0, _f_col0 = _regions.left_col0, _regions.full_col0
    _r_col0, _m_col0 = _regions.right_col0, _regions.masked_col0
    _n_masked = fx.Index(_n_l + _n_r)
else:
    _full_end = (seqlen_k_v // fx.Index(BLOCK_N)) * fx.Index(BLOCK_N)
    kv_upper  = fx.Index(((seqlen_k_v + fx.Index(BLOCK_N - 1)) // fx.Index(BLOCK_N)) * fx.Index(BLOCK_N))
    _full_end = fx.Index(_full_end if _alive else fx.Index(0))
    kv_upper  = fx.Index(kv_upper  if _alive else fx.Index(0))
```

**This is the case that needs `@flyc.jit`, and it is a clean example of it.**
The `const_expr(CAUSAL)` arm is Python-level and would move as-is, but the two
`_alive` clamps in the `else` arm are ternaries, and a ternary in an
undecorated module function is evaluated by Python at trace time -- `bool()` on
an `fx.Boolean`, which either raises or bakes in one arm. So the helper must be
decorated, or those two lines must go back to `.select`.

Shape: `fmha.kv_tile_plan(causal, start_q, seqlen_q, seqlen_k, wl, wr,
BLOCK_M, BLOCK_N, alive)` returning one object with `n_left/n_full/n_right`,
the four column origins, `n_masked` and `kv_tiles`. The non-causal arm fills
the same fields, which is what fuse's local `_plain_regions` (L178) already
does for itself -- so that helper folds in too.

~40 lines, and it removes fuse's private counterpart.

## 3. The masked-column map -- already on the list

Four copies (fwd L1836, dq L904, fuse L780 and L935), now one line each since
the ternary conversion:

```python
return _l_col0 + _i * _BN_I32 if _i < _n_l else _r_col0 + (_i - _n_l) * _BN_I32
```

Needs `@flyc.jit` for the same reason. ~8 lines, but the value is that the seam
between the left and right runs is the part that was actually wrong during
bring-up, and four copies is four chances to get it wrong again. Do it with
section 2 -- it takes the same arguments.

## 4. Q/K/V preload is already a library

`Aperture` + `stage`/`publish`/`read_batches`/`reader` is the preload
abstraction, and all four kernels use it: 4/6/8/7 `Aperture` constructions and
3/3/4/7 readers respectively. What remains in the kernels is the *configuration*
-- eight keyword arguments per aperture naming the LDS base, stride, vector
width and cooperative-load geometry -- and that is per-tensor-per-kernel by
nature.

The forward's `coop_load_v_global` / `coop_store_v_lds` (13 + 12 lines) are the
only preload code outside that surface, and they are forward-only: they exist
for the prefetched transposed-V path, which no backward kernel has.

**Recommendation: nothing to do.** This one is already where the survey wanted
it to be.

## 5. Order

1. `fmha.kv_tile_plan` + `fmha.masked_col`, together, decorated -- ~48 lines,
   and it retires fuse's `_plain_regions`.
2. fuse's `kv_body`/`q_body` merge -- ~40 lines, local, no library.

Both gate on the full suite plus bitwise ISA at head_dim 128 and 384 in both
masking modes. Section 1 is the note to keep: the loop is not a candidate, and
re-proposing it later should cost a re-measurement first.
