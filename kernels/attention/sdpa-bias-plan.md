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

## 3. The interaction that needs a decision: bias under varlen

AOTriton forms the bias base as

```python
B_ptrs = B + batch_index * stride_bz + off_h_q * stride_bh + offs_m[:, None] * stride_bm
```

Note what is *absent*: `cu_seqlens_q_start`. The row index is `offs_m`, which
is the position **within the sequence**, and the batch selector is
`batch_index` — which compact varlen sets to **0**.

So under compact varlen every sequence would read the same bias plane, indexed
by its own local row. That is either a deliberate restriction (bias is not
supported with varlen) or a gap; either way we should not copy it without
saying which. Three options:

| option                      | meaning                                                  | cost                               |
| --------------------------- | -------------------------------------------------------- | ---------------------------------- |
| **A. reject** bias + varlen | host raises; matches what AOTriton effectively does      | none, and no semantics invented    |
| **B. per-sequence plane**   | `stride_bz` indexes the sequence, `z` selects it         | one more term, `(N, H, Sq, Sk)`    |
| **C. packed bias**          | bias rows follow `q_row_off`, columns follow `k_row_off` | matches packed Q/K; largest change |

**Recommend A for this phase**, with B or C deferred until a caller asks:
the varlen decomposition already gives `q_row_off` and `k_row_off`, so C is a
two-term change later, and inventing a layout now that no caller uses risks
getting it wrong in a way that is expensive to unpick. This is a decision for
you, not a default I should pick silently.

---

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

| risk                               | why it matters                                                   | mitigation                                        |
| ---------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------- |
| **Register pressure**              | a new load stream in a loop already spilling at head_dim 192+    | measure before prefetching; §2.1                  |
| `-inf` deleted by fast-math        | now arrives from user data, not just from the kernel             | `ninf` stays off; assert with an `-inf` bias      |
| Bias masked *after* the mask       | `-inf + bias` un-masks a column that gSWA killed                 | order fixed in §1; test bias together with causal |
| Tuning tables go stale             | fifth time; bias changes the live set                            | re-sweep only if the A/B shows a shift            |
| Varlen semantics invented silently | §3 has no obviously right answer and AOTriton does not answer it | decide explicitly; recommend rejecting for now    |

---

## 7. What this phase does *not* do

- **ALIBI** — D4, out of scope, and it is a *computed* bias rather than a
  loaded one, so it shares the add but not the load.
- **Bias gradients.** The backward pass needs `db`, which is a reduction over
  the same tile. Out of scope, but the load geometry in §2 is what it will
  reuse, so keep it addressable rather than inlined.
- **P5 dropout**, which is next and independent.
