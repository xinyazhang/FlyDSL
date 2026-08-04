# P4 in detail: Bias Tensor

Companion to `sdpa-close-gap-plan1.md`, whose P4 entry is one line, and to
`sdpa-feature-gap.md`, whose bias section is three words (`B` and `stride_b*`).
This is the implementation plan.

---

## 0. The goal, including the one that is easy to leave unstated

1. **Add a bias tensor.** `B` of shape `(B, H, Sq, Sk)` with four strides,
   added to the attention scores before the softmax. `BIAS_TYPE ∈ {0, 1}`,
   matching AOTriton.
2. **Make the `m_i` floor verifiable.** P1 changed `m_i` to initialise at
   `-3.40282e+38` rather than `-inf` and recorded that the fix was
   *preventative*: with causal masking alone the first KV tile always contains
   a live column, so `-inf - -inf` is unreachable. **Bias is the first
   configuration in which it is reachable at all**, so the regression test
   plan1 promised belongs here and the phase is not done without it.
3. **Do not pay for bias when it is off.** `BIAS_TYPE` is a build axis, so a
   `BIAS_TYPE == 0` build must be bit-identical to today and emit no bias
   instruction at all.

Objective 2 is the one that would otherwise be quietly skipped: it is a test
for a fix that landed three phases ago, and nothing fails today without it.

---

## 1. Where bias enters, and why the order is forced

AOTriton adds bias inside the inner loop:

```python
qk += (bias * 1.44269504089)
```

after `qk` has been multiplied by `qk_scale`, and the constant is `log2(e)`.
That is not an implementation detail to copy blindly — it follows from the
same base-2 softmax domain this kernel uses. `m_i` and the exponent live in
the scaled domain (P1's softmax correction), so a bias expressed in natural
units has to be converted before it can be added to a score that already is.

The order is therefore fixed at three points, not two:

```
    S = QK^T
    S *= sm_scale * log2e          <- P1: scale before the row max
    S += bias * log2e              <- here
    S  = mask ? -inf : S           <- gSWA: after the scale, and after this
```

**Bias goes before the mask**, so that a masked column stays `-inf` rather
than becoming `-inf + bias`. That matters because bias may itself be `-inf`:
callers routinely pass a large negative or `-inf` bias as an attention mask,
which is exactly how objective 2 becomes reachable.

### 1.1 `ninf` stays off, and now it is load-bearing

plan1 §3 already argues this ("`ninf` is not affordable once bias exists") and
the kernel already builds without it. The difference is that until now the
argument was hypothetical — the only `-inf` in the kernel was one it wrote
itself, and a compiler that deleted it would have been caught by the causal
tests. With bias, `-inf` arrives from *user data*, so `ninf` would license
deleting a value the caller supplied. Re-assert this with a test rather than
leaving it to the build flags.

---

## 2. The load: a vector, not a gather

The obvious worry is that bias needs one element per `(q_row, kv_col)` pair
and this kernel's scores live in WMMA accumulators with a scrambled column
order. Element `i` of the flattened accumulators is KV column

```
    (i // 16) * 32 + ((i // 8) % 2) * 16 + klane * 8 + i % 8
```

which looks like a gather. It is not. Within each group of eight consecutive
`i`, only `i % 8` varies, so those eight elements are **eight contiguous
columns** starting at `klane * 8`. One `v8` load covers them — the same shape
and width as the existing K and V loads.

So bias costs `NUM_S_ACCS` vector loads per KV tile per Q row-tile, on a row
index that is uniform per lane (`lane16`) and a column base that is uniform
per `klane`. No gather, no scalarisation.

### 2.1 What it does cost

A new global load stream in a loop that is **already latency-bound and already
spilling at head_dim 192 and above** (plan1 §6.2, §2.6). This is the phase's
main risk and it is a performance risk, not a correctness one. Two mitigations
are available and should be measured in this order:

1. **Nothing.** The loop has spare memory-level parallelism precisely because
   it is latency- rather than bandwidth-bound; the bias load may hide behind
   the existing K/V waits.
2. **Prefetch it with K and V**, at the same distance, so it joins the
   existing pipeline rather than adding a serial dependency.

Do not reach for (2) before measuring (1): it adds `NUM_S_ACCS` more
loop-carried vectors, which is the resource this kernel is shortest of.

---

## 3. Bias and varlen are orthogonal, and that is a property to preserve

Checked against the source rather than assumed, because an earlier draft of
this section got it wrong and proposed rejecting the combination.

**Orthogonal in the build.** `BIAS_TYPE` is a build axis (`options=[0, 1]`)
while varlen is entirely runtime, so one compiled kernel already serves both.
There is no `@ati.disable` pairing them — the only bias exclusion is against
*causal*, §3.2.

**Orthogonal at runtime too, provided bias is indexed like Q.** Varlen does
not add a mode to the body; it re-purposes `batch_index` and a row offset, and
every tensor that consumes those two gets varlen for free. Compare:

| tensor | batch selector | row index                     |
| ------ | -------------- | ----------------------------- |
| Q      | `batch_index`  | `cu_seqlens_q_start + offs_m` |
| bias   | `batch_index`  | `offs_m`                      |

AOTriton's bias omits the row offset. That is invisible for the `BATCHED`
modes — dense and padded varlen both set `cu_seqlens_q_start = 0` and select
the plane with `batch_index`, so bias behaves exactly as it does dense. It
only bites for the `STACKED` modes, where `batch_index` is pinned to 0 and the
row offset carries the sequence.

**So the rule for this kernel is one line: index bias with the same
`(batch_index, q_row_off)` the varlen decode already produced, and the same
`k_row_off` on the column.** The six scalars exist precisely so that a tensor
does not need to know which layout it is in, and bias is another consumer of
them. That makes every varlen mode work uniformly, and it costs two extra
terms in an address that is computed once per tile.

This diverges from AOTriton for the stacked modes, deliberately: a bias laid
out to match packed Q/K is the only thing a caller could sensibly pass there,
and `sdpa-varlen-plan.md` §0.1 already records that AOTriton is not the oracle
for configurations past its enum.

### 3.1 Not tested, and that is a decision

Bias-under-varlen gets **no test**. The combination has no caller today, and
inventing a reference for it would cost more than the logic is worth. What is
required instead is that the *logic* be right by construction — bias reads the
same two scalars as Q and V, so if varlen is correct for them it is correct
for bias, and there is no third path that could drift.

Stated explicitly because "untested" and "unsupported" are different claims
and the code should not be read as making the second one.

### 3.2 AOTriton ships no causal + bias at all

Worth flagging because it removes an oracle. `_common.py`:

```python
if causal != 0 and bias_type != 0:
    return True          # functional disabled
```

with the comment "causal+matrix-bias unsupported". So AOTriton has **no**
compiled kernel for that combination on any architecture.

This kernel has one masking path and one bias add, and they compose without a
special case, so there is no reason to inherit the restriction — but it means
the causal rows of §5's matrix have no AOTriton counterpart and must be
checked against a reference implementation instead. It also means §1's
ordering claim (bias before the mask) cannot be validated against AOTriton
behaviour; it has to stand on the argument.

## 4. Steps

### Step 1 — `BIAS_TYPE` and the tensor

`B` pointer and `stride_b0..b3` as arguments, `BIAS_TYPE` as a build axis.
Load and add per §1, `BIAS_TYPE == 0` emitting nothing.

**Gate:** `BIAS_TYPE == 0` is bit-identical to today across the ladder, and a
zero bias at `BIAS_TYPE == 1` matches it to tolerance. The first is the real
check — it is what proves objective 3.

### Step 2 — the `m_i` floor regression test

A bias that covers the whole first KV tile with `-inf` drives that tile's row
max to `-inf`. With an `-inf`-initialised `m_i` the correction term is
`exp2(-inf - -inf) = NaN`; with the floor it is `exp2(-inf + 3.4e38) = 0`.

**Gate:** the test fails against a deliberately reverted floor and passes with
it. A regression test that has never been seen to fail is not yet a regression
test.

### Step 3 — measure, then decide about prefetching

§2.1. Ladder A/B with bias on and off, and the on/off gap reported per
head_dim rather than as a median — the cost should scale with `NUM_S_ACCS`,
so a flat profile means something is wrong.

---

## 5. Test matrix

| axis          | values                                           | why                                            |
| ------------- | ------------------------------------------------ | ---------------------------------------------- |
| `BIAS_TYPE`   | 0, 1                                             | 0 must be bit-identical to today               |
| bias values   | zeros, random, large negative, `-inf`            | `-inf` is objective 2 and the `ninf` check     |
| `-inf` extent | one column, one whole KV tile, a whole row       | a whole row is "this query attends to nothing" |
| shape         | `(B, H, Sq, Sk)`, and broadcast-ish strides of 0 | stride 0 is how callers share one plane        |
| with causal   | off, top-left, bottom-right, window              | bias and mask must compose, not fight          |
| ragged seqlen | primes                                           | the tail tile loads bias out of range too      |

Two properties beyond "matches the reference":

1. **Zero bias is a no-op.** `BIAS_TYPE == 1` with an all-zero tensor must
   match `BIAS_TYPE == 0` — bitwise if the add is the only difference, which
   it should be at zero.
2. **A whole-row `-inf` bias gives `O = 0` and `LSE = +inf`**, the same
   contract dead rows already have under gSWA. Bias reaches that state through
   a different path, so it is worth asserting separately.

---

## 6. Risks

| risk                                 | why it matters                                                | mitigation                                                         |
| ------------------------------------ | ------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Register pressure**                | a new load stream in a loop already spilling at head_dim 192+ | measure before prefetching; §2.1                                   |
| `-inf` deleted by fast-math          | now arrives from user data, not just from the kernel          | `ninf` stays off; assert with an `-inf` bias                       |
| Bias masked *after* the mask         | `-inf + bias` un-masks a column that gSWA killed              | order fixed in §1; test bias together with causal                  |
| Tuning tables go stale               | fifth time; bias changes the live set                         | re-sweep only if the A/B shows a shift                             |
| Bias drifts from Q's varlen indexing | bias would silently read the wrong rows under stacked varlen  | index it from the same six scalars (§3); no separate path to drift |
| No AOTriton oracle for causal + bias | that pair is disabled there, so parity cannot be checked      | reference implementation for those rows (§3.2)                     |

---

## 7. What this phase does *not* do

- **ALIBI** — D4, out of scope, and it is a *computed* bias rather than a
  loaded one, so it shares the add but not the load.
- **Bias gradients.** The backward pass needs `db`, which is a reduction over
  the same tile. Out of scope, but the load geometry in §2 is what it will
  reuse, so keep it addressable rather than inlined.
- **P5 dropout**, which is next and independent.
