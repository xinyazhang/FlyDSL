# P6 in detail: Generalized Sliding Window Attention

Companion to `sdpa-close-gap-plan1.md`, which sketches gSWA in a paragraph.
This is the implementation plan.

---

## 0. The goal, including the one that is easy to leave unstated

Two objectives, and the second is not a tidy-up:

1. **Add sliding-window attention**, generalized so `Window_left` and
   `Window_right` may be negative.
2. **Delete `CAUSAL_TYPE` 1 and 2 from the kernel.** gSWA subsumes them
   exactly; keeping them would leave two ways to express one thing, drifting
   apart under maintenance. The kernel should ship `CAUSAL_TYPE ∈ {0, 3}` and
   nothing else.

Objective 2 is what AOTriton already does — `@ati.scalar('CAUSAL_TYPE',
options=[0, 3])` in `modules/flash/aot/attn_fwd.py`. Types 1 and 2 exist only
as *host-side* conveniences that resolve to window values before dispatch. Our
P2b work added 1 and 2 to the kernel as a stepping stone; this phase removes
them again, and the phase is not finished until it has.

**Success criterion for objective 2:** `grep -c "CAUSAL_TYPE == 1\|CAUSAL_TYPE
== 2"` in the kernel returns 0, and the diagonal is expressed only through
`window_right`.

---

## 1. What P2b already built

More than half of this phase is already in place, which is worth stating
because it changes what the remaining work is.

| gSWA needs | already exists |
|---|---|
| a shifted diagonal | `_diag_i32`, which **is** `window_right` |
| region decomposition | the two-region full/masked split |
| a masked-only code path | `_MASK_STEPS` |
| somewhere to put the window predicate | the fused `_dead` mask |
| rows that see no keys | handled — `l` clamp and LSE `+inf` |
| `-inf` surviving the compiler | `ninf` dropped from fast-math |

Concretely, today's mask is

```python
_dead = (_col >= seqlen_k) | (_col > q_row + _diag)
```

and gSWA's is

```python
_dead = (_col >= seqlen_k) | (_col > q_row + w_right) | (_col < q_row - w_left)
```

One extra term. **The kernel-side arithmetic is nearly done; the work is in the
interval calculation and the loop shape.**

The mapping that makes objective 2 free:

| CAUSAL_TYPE | window_left | window_right |
|---|---|---|
| 1 (top-left) | `seqlen_q` | `0` |
| 2 (bottom-right) | `seqlen_q` | `seqlen_k - seqlen_q` |

`window_left = seqlen_q` means "unbounded on the left" — no row can reach
further back than that. So **our current kernel is already gSWA with
`window_left` pinned to unbounded**, and `_diag_i32` is already computing
exactly the `window_right` column of that table.

---

## 2. What actually changes: three intervals, not two

With only a right window, masked blocks are a suffix and the split is
`[full][right-masked]`. A left window makes them a *prefix as well*:

```
    kv blocks:   [ left-masked ][ full ][ right-masked ]
```

and the full region can be empty, or the two masked regions can overlap and
merge. That is what `calculate_intervals` computes, and it is the one genuinely
new piece of logic.

### 2.1 The intervals

Following `masked_load_store.py`, in closed intervals of block indices. `div_rd`
is round-*down* division (Python's `//` on negatives already does this; C's
does not, which is why AOTriton spells it out).

```
lsec = [ div_rd(start_M - w_left,               BLOCK_N),
         div_rd(min(start_M + BLOCK_M, seqlen_q) - w_left - 1, BLOCK_N) ]
rsec = [ div_rd(start_M + w_right,              BLOCK_N),
         div_rd(min(start_M + BLOCK_M, seqlen_q) + w_right - 1, BLOCK_N) ]
vb   = [ 0, (seqlen_k - 1) // BLOCK_N ]        # blocks that exist at all

lb = lsec ∩ vb          # left-masked
rb = rsec ∩ vb          # right-masked
fb = [lsec.hi + 1, rsec.lo - 1] ∩ vb   # full, only if lsec and rsec are disjoint
```

### 2.2 The three special cases

Each is a real shape, not a corner case to hand-wave:

1. **Irregular `seqlen_q`** (`Y_low < seqlen_q < Y_high`): the Q block is
   partly past the end, so the `min(..., seqlen_q)` above no longer bounds
   every row the same way. **No full blocks**; union `lb` and `rb` into one
   masked run.
2. **`lb` and `rb` intersect**: the window is narrower than a block, so no
   block is fully live. Merge to `lb = [lsec.lo, rsec.hi] ∩ vb`, `fb` and `rb`
   empty.
3. **Irregular `seqlen_k`**: the last block is partly past the end. Take the
   trailing block *out* of `fb` and give it to `rb`. This is the case our KV
   tail mask handles today, and it must be folded into the interval logic
   rather than left as a separate unconditional mask.

### 2.3 The loop shape: two bodies, not three

The obvious reading of §2 is three loops. **Do not do that.** The body is
already emitted twice and it cost 63 VGPRs at head_dim 128 and pushed head_dim
192 into spilling (plan1 §6.2, §2.6). A third copy is the single largest risk
in this phase.

AOTriton's inner loop instead walks the two masked runs as *one* loop over a
piecewise index — which is exactly what slide 18 is about:

```python
for i in range(n_left + n_right):
    start_n = (lb_lo + i) if i < n_left else (rb_lo + i - n_left)
```

So the emitted structure stays at **two bodies**:

```
    masked loop   over lb ++ rb   (piecewise start_n, MASK_STEPS=True)
    full loop     over fb         (MASK_STEPS=False)
```

Cost is one select per masked iteration to pick `start_n`. Cheap, and it is
paid only in the masked region.

**Consequence for the prefetch.** The distance-1 K/V prefetch currently runs
one tile past the region end, which is harmless because the next tile is
either the next region's first or clamped. With a piecewise index the "next"
tile is discontinuous at the seam, so the prefetch must use the same piecewise
mapping — computing `start_n(i+1)`, not `start_n(i) + BLOCK_N`. Getting this
wrong prefetches the wrong tile and is invisible to a correctness test whenever
the value is overwritten before use.

---

### 2.4 Everything derived from a window is signed. Use i32 and say so.

`Window_left` and `Window_right` are **signed and routinely negative** — that
is the entire content of the word "generalized". Bottom-right causal at
`seqlen_q > seqlen_k` already gives `window_right < 0` today, and a caller may
pass negative values on either side deliberately to shift a band off the
diagonal.

This is not a theoretical hazard. It cost a debugging round in P2b: clamping
the KV bound with `_lim > 0` **kept** `-128` instead of replacing it, because
`fx.Int32` comparisons default to *unsigned* and `-128` as `u32` is enormous. A
huge bound then reached the loop. The Q-row bound check in the kernel carries a
comment about the same trap, from before that.

**Rules, applied without exception to anything downstream of a window:**

1. **Compute in `fx.Int32`, never `fx.Index`.** `fx.Index` is unsigned; the
   moment a negative intermediate touches it the value is garbage rather than
   negative.
2. **Use explicit signed predicates** — `arith.CmpIPredicate.slt` / `sgt` /
   `sle` / `sge`. Do not rely on the `<` and `>` operator overloads, which pick
   unsigned.
3. **Convert to `fx.Index` only after clamping to a known-non-negative range,**
   and clamp with a signed compare. Loop bounds and block counts are the only
   things that should ever become `Index`.
4. **`div_rd` must round toward negative infinity**, not toward zero.
   `start_M - w_left` is negative whenever the window reaches past the start of
   the sequence, and C-style truncation gives the wrong block index there.
   Python's `//` already floors; a hand-written `x // y` in the DSL may not.

**Where this bites specifically:**

| quantity | can be negative |
|---|---|
| `w_left`, `w_right` | by definition |
| `start_M - w_left`, `start_M + w_right` | the `lsec` / `rsec` endpoints |
| `lsec.lo`, `rsec.lo` after `div_rd` | block indices below 0 |
| `fb = [lsec.hi + 1, rsec.lo - 1]` | empty *and* inverted |
| `q_row + w_right`, `q_row - w_left` | the mask comparison operands |

The interval helpers make this manageable if ported faithfully:
`closed_interval_isect` returns a deliberately inverted sentinel `(-114, -514)`
for the empty case, and `is_closed_interval_empty` is `lo > hi` — so emptiness
is representable without a separate flag, but only if the comparison is signed.

**Suggested guard:** keep window-derived values in a distinct naming
convention (`_w*` / `*_i32`) so a review can spot an `fx.Index(...)` wrapped
around one. A silent unsigned compare produces plausible-looking output on
square shapes and fails only where the window goes negative, which is exactly
the region the tests in §4 target.

## 3. Steps

Each lands independently and leaves the tree green.

### Step 1 — windows as arguments, causal still a build axis

Add `window_left` / `window_right` as `i32` kernel arguments. Derive them on
the host from `CAUSAL_TYPE` using the table in §1. Leave the kernel's
`CAUSAL_TYPE` 1/2 branches in place, but make them *compute the same windows*
rather than a diagonal.

- The mask gains its third term.
- The interval logic is unchanged: `window_left = seqlen_q` keeps `lb` empty,
  so the existing two-region split still applies.

**Gate:** a window must reproduce the dedicated causal path. **Not bitwise** --
see the note below; the bar is a tight tolerance with a negative control.

### Step 2 — the left interval

Implement `calculate_intervals` with all three special cases, and the piecewise
masked loop.

- Port `div_rd`, `closed_interval_isect`, `is_closed_interval_empty`,
  `closed_interval_size`.
- Fold the KV tail mask (§2.2 case 3) into the interval logic. It stops being
  unconditional, which is worth measuring on its own — that mask is what
  plan1 §6.3 charged 5-15% for before the P2a split.

**Gate:** correctness across the window matrix in §4, *and* a benchmark
showing the ladder has not regressed. Also assert the emitted structure — two
loop bodies, not three — since §2.3's whole point is invisible to a
correctness test.

### Step 3 — delete `CAUSAL_TYPE` 1 and 2

The objective. Kernel keeps `0` and `3`. The host maps 1/2 to windows, exactly
as `mha_fwd_aot` does via `calculate_swa`. `_diag_i32` disappears into
`window_right`.

**Gate:** the grep in §0 returns 0, and the causal tests — unchanged — still
pass, now running through the window path.

### Step 4 — varlen sentinels

`Window_left`/`Window_right` accept `0x80000001` (causal top-left) and
`0x80000002` (causal bottom-right), resolved per-sequence against that
sequence's `seqlen_q`/`seqlen_k`. Needed because varlen has no single
`seqlen_q` to compute a uniform window from (slide 17).

**Depends on P3.** Do not attempt before varlen exists; the sentinel is
meaningless without per-sequence lengths.

---

## 4. Test matrix

Beyond `test_fast` (plan1 §5.1), gSWA needs its own axes:

| axis | values | why |
|---|---|---|
| `window_left` | 0, 1, 16, 64, `seqlen_q`, **-16**, **-64** | negative is the "generalized" part |
| `window_right` | 0, 1, 16, 64, `seqlen_k - seqlen_q`, **-16**, **-64** | ditto |
| `(Lq, Lk)` | square, `Lq < Lk`, `Lq > Lk` | the diagonal shifts differently |
| ragged | `Lq`, `Lk` from the prime lists | all three special cases in §2.2 |

Four properties worth asserting beyond "matches the reference":

1. **Causal equivalence.** `window_left = seqlen_q, window_right = 0` must
   reproduce `CAUSAL_TYPE=1`, and `window_right = seqlen_k - seqlen_q` must
   reproduce `CAUSAL_TYPE=2`. This is the test that justifies deleting them,
   and it must be written in step 1 while both paths still exist.

   **Bitwise was the wrong bar, and step 1 measured why.** The two are separate
   builds, and the window path is a structurally different kernel -- in step 1
   it masks every tile where the causal path masks only the diagonal ones.
   Under `reassoc`/`contract` fast-math LLVM may fuse and reorder them
   differently, so bit-equality is not something the toolchain owes us. About
   40% of the shape matrix does come out bit-identical, which is exactly the
   trap: bitwise would have looked plausible and then failed on the rest.

   What the bar must catch is a wrong *set of columns*, so the test proves it
   can rather than asserting it: the same window shifted one column right is
   run too and must differ by a wide margin. Measured over the §4 matrix at
   head_dim 64 and 128:

   | | worst | weakest |
   |---|---|---|
   | matched window vs causal | 2.5e-4 | — |
   | one-column shift vs causal | — | 2.2e-1 |

   ~850x apart, so the bar sits at 1e-3: 4x above the noise, 200x below the
   smallest real error. A negative control is what makes a tolerance an
   assertion about the kernel rather than about the number chosen.
2. **Empty windows.** A window admitting no keys for some rows is reachable
   here far more easily than under causal — `window_left + window_right < 0`
   does it. Those rows must give `O = 0` and `LSE = +inf`. Already handled, but
   gSWA is the first feature that makes it *easy* to hit, so test it directly.
3. **A window narrower than `BLOCK_N`**, which triggers §2.2 case 2 (lb and rb
   merge) and leaves no full blocks at all.
4. **A window wider than `seqlen_k`**, which should degenerate to no masking.

---

## 5. Risks

| risk | why it matters | mitigation |
|---|---|---|
| **A third loop body** | 2 bodies already cost 63 VGPRs at hd 128 and spilled hd 192 | piecewise `start_n`, §2.3; assert the body count |
| Prefetch across the seam | discontinuous `start_n`; wrong tile is invisible to correctness tests | prefetch via the same piecewise map; test with a window that forces a seam mid-loop |
| Interval arithmetic on negatives | `fx.Index` is unsigned and `fx.Int32` compares default to unsigned; this already cost a debugging round in P2b | the rules in §2.4, applied without exception |
| `-inf` through fast-math | already bit us once | mask stays after the scale; `ninf` stays off |
| Tuning tables go stale again | third time now (256 after LSE, 128 after P2a) | re-sweep after step 2, before declaring the phase done |

---

## 6. What this phase does *not* do

- **Persistent-dynamic.** plan1 §2.4 records that AOTriton's grid only makes
  sense together with it, and that our head-fastest axis order should be
  revisited when it lands. That is a separate task.
- **`PERSISTENT_TYPE`, `NUM_XCDS`, INT8** — unchanged from plan1.
- **Backward.** Windows must eventually reach the dk/dv kernel with the mask
  *transposed* (slide 6: "should not assume the causal masks always be the
  lower-triangular shape"). Out of scope, but the window parameters should be
  named and stored so that a transposed consumer is straightforward.
