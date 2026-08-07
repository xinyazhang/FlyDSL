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

### 2.1 What it does cost — measured

A bias tensor is `(B, H, Sq, Sk)`, so reading it is **O(N²) traffic against
O(N·D) for everything else**. At B=1 H=8 N=4096 head_dim 64 that is 268 MB of
bias against 16.8 MB for Q, K, V and O combined — sixteen times the rest of
the kernel's memory.

Measured, non-causal, interleaved:

| head_dim | `NUM_S_ACCS` | no bias | bias | ratio |
| -------- | ------------ | ------- | ---- | ----- |
| 16       | 16           | 40.6    | 10.6 | 3.82x |
| 32       | 8            | 58.9    | 16.2 | 3.64x |
| 64       | 4            | 88.2    | 28.4 | 3.11x |
| 128      | 4            | 93.6    | 56.1 | 1.68x |
| 192      | 4            | 75.8    | 60.5 | 1.26x |
| 256      | 4            | 95.4    | 72.4 | 1.32x |

**This is the cost of the feature, not a regression to fix.** `BIAS_TYPE` is a
build axis, and `BIAS_TYPE == 0` is bit-identical to the pre-bias kernel — a
caller who does not want bias pays nothing, not even an argument read.

The shape of the cost falls out of the ratio above:

```
bias bytes      = 2·B·H·N²
attention flops = 4·B·H·N²·D      ->   bytes/flop = 1 / (2·head_dim)
```

so **the relative cost depends on head_dim alone and not on sequence length**
— the same 268 MB serves sixteen times the arithmetic at head_dim 256 as at
16. It does not get worse as models grow their context.

### 2.2 Prefetching does not help, and the measurement says why

The plan originally proposed prefetching bias with K and V. It was tested and
it does not work, for a reason worth recording rather than re-deriving.

The bias stream runs at **327 GB/s**, and the R9700 peaks near 645. That looks
like half the machine left on the table until the access pattern is counted:
lanes 0-15 and lanes 16-31 read the *same* sixteen Q rows, contributing 16
bytes each, so a wave touches sixteen rows at **32 contiguous bytes** apiece.
Each of those lands in a 64-byte cache line, giving a hard ceiling of 50% of
peak. 327/645 is 50.7%.

**We are already at the ceiling the layout allows**, so there is no latency
left to hide. Confirmed directly as well: hoisting the loads from after GEMM1
(where they were issued and immediately waited on) to before it gives a whole
GEMM1 and cross-shard reduction of free overlap, and moved nothing — 3.11x
stayed 3.11x at head_dim 64. A distance-1 prefetch would buy more overlap and
the same bandwidth, at the cost of `NUM_S_ACCS` loop-carried vectors.

The only real lever would be cache-line utilisation, which the WMMA
accumulator layout dictates: a lane's eight elements are eight columns of one
row, and two lanes cover sixteen. Changing that means changing how S is held.
Out of scope, and recorded here so the next person does not re-run the
prefetch experiment.

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

### 3.2 Causal + bias is rejected, because it is undefined

AOTriton disables the functional outright — `_common.py`:

```python
if causal != 0 and bias_type != 0:
    return True          # functional disabled
```

An earlier draft read that as a capability gap and proposed *not* inheriting
it, since this kernel has one masking path and one bias add and composes them
without a special case. That was the wrong reading. **The restriction is
semantic, not technical.**

Causal is an attention mask with a fixed pattern. Bias *is* an attention mask,
supplied directly — a large negative or `-inf` entry is how callers spell "do
not attend here". Passing both asks which one wins where they disagree, and
there is no defined answer: the caller has specified the same thing twice, in
two vocabularies, with no rule for reconciling them.

PyTorch takes the same position, and more explicitly than AOTriton:

| backend   | `attn_mask` together with `is_causal=True`                               |
| --------- | ------------------------------------------------------------------------ |
| math      | `RuntimeError: Explicit attn_mask should not be set when is_causal=True` |
| flash     | `No available kernel` — none is compiled for the pair                    |
| efficient | accepted, silently                                                       |

Two of three refuse, one of them by name. The third is the outlier, not the
precedent.

**So this kernel rejects it too**, on the host, with a message that says why
rather than "unsupported". `BIAS_TYPE == 1` together with `CAUSAL_TYPE != 0`
is a caller error.

### 3.3 The KV tail mask is not an attention mask, and still orders after bias

The rejection removes the *semantic* mask from the picture but not the bounds
one. Columns past `seqlen_k` are not keys the caller chose to hide; they do not
exist, and their bias entries do not exist either. So the tail mask must still
win, which it does because §1 puts the bias add before it.

This is why §1's ordering claim survives §3.2: the mask it orders against is
the one masking *non-existent* keys, which composes with bias in the only way
it can.

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

**Outcome — met, and emphatically.** With the floor, 0 NaNs. With `c_m_init`
reverted to `-inf`, **32768 of 32768** output elements are NaN. The P1 fix has
been unverifiable for three phases and is now demonstrably load-bearing.

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
| with causal   | **rejected** -- assert the error                 | undefined semantics, §3.2                      |
| ragged seqlen | primes                                           | the tail tile loads bias out of range too      |

Two properties beyond "matches the reference":

1. **Zero bias is a no-op**, to tolerance. `BIAS_TYPE == 1` with an all-zero
   tensor must match `BIAS_TYPE == 0`. An earlier draft said "bitwise, since
   the add is the only difference" — wrong, and wrong the same way gSWA step 1
   was: adding a floating-point operation changes the emitted code even when
   the operation is a mathematical no-op, and `reassoc`/`contract` may then
   contract differently. Measured: 33 of 32768 elements differ, by at most
   ~1 f16 ULP.
2. **A whole-row `-inf` bias gives `O = 0` and `LSE = +inf`**, the same
   contract dead rows already have under gSWA. Bias reaches that state through
   a different path, so it is worth asserting separately.

---

## 6. Risks

| risk                                 | why it matters                                                              | mitigation                                                             |
| ------------------------------------ | --------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| ~~Register pressure~~                | not the binding cost -- the stream is bandwidth-bound at the layout ceiling | measured, §2.2; prefetching rejected                                   |
| `-inf` deleted by fast-math          | now arrives from user data, not just from the kernel                        | `ninf` stays off; assert with an `-inf` bias                           |
| Bias added *after* the tail mask     | `-inf + bias` revives a column past seqlen_k                                | order fixed in §1; the ragged-seqlen row of §5                         |
| Tuning tables go stale               | bias changes the live set                                                   | `BIAS_TYPE == 0` is bit-identical, so the shipped tables are untouched |
| Bias drifts from Q's varlen indexing | bias would silently read the wrong rows under stacked varlen                | index it from the same six scalars (§3); no separate path to drift     |
| Causal + bias accepted by accident   | it is undefined, and a silent answer is worse than an error                 | host-side rejection with a reason (§3.2); assert it                    |

---

## 7. What this phase does *not* do

- **Causal + bias**, §3.2 — rejected rather than deferred.
- **ALIBI** — D4, out of scope, and it is a *computed* bias rather than a
  loaded one, so it shares the add but not the load. It is *not* subject to
  §3.2: ALiBi is a positional prior rather than a mask, so composing it with
  causal is well defined.
- **Bias gradients.** The backward pass needs `db`, which is a reduction over
  the same tile. Out of scope, but the load geometry in §2 is what it will
  reuse, so keep it addressable rather than inlined.
- **P5 dropout**, which is next and independent.
