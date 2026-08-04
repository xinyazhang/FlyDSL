# P5 in detail: Dropout

Companion to `sdpa-close-gap-plan1.md`, whose P5 entry is one line. This is
the implementation plan, and the phase adds more code than any before it —
a PRNG, a second kernel, and a reproducibility contract that is easy to
satisfy incorrectly.

---

## 0. The goal, including the one that is easy to leave unstated

1. **Philox in its own module, `philox.py`, arch-agnostic.** The PRNG is not
   attention: it has its own tests, its own reference, and a second consumer
   (§7). It is also **destined to move into FlyDSL as a library**, so it is
   written generic from the start — no gfx1201 in the name, no target
   intrinsics, nothing that assumes wave32 or a particular ALU. What is
   target-specific is the *default* it is called with, not the code.
2. **64-bit seed and 64-bit offset.** Both are `u64` in AOTriton
   (`philox_seed_ptr`, `philox_offset1` are `*u64`, `philox_offset2` is
   `uint64`), and PyTorch's RNG state is 64-bit. §2 shows this does *not*
   force 64-bit arithmetic.
3. **Make the arithmetic width a build-time choice, not a decision.** The
   32-bit and 64-bit variants trade ALU cost against calls per element, and
   which wins is a property of the target's integer ALU — so `PHILOX_WIDTH` is
   a constexpr with a per-arch default, exactly as `BLOCK_M` is. §4 measures
   gfx1201's default; it does not settle the question for anyone else.
4. **`ENABLE_DROPOUT` as a build axis.** Off must be bit-identical to today,
   as `BIAS_TYPE == 0` is.
5. **A second kernel that returns the mask**, mirroring AOTriton's
   `debug_fill_dropout_rng`.

**The unstated one: the mask must not depend on the tiling.** Element
`(z, h, m, n)` must get the same random number whatever `BLOCK_M`, `BLOCK_N`,
the region split, or the number of workgroups happen to be. That is what makes
dropout reproducible across a re-tune, and it is the property most easily lost
— every other phase in this project was free to change tile sizes, and this
one silently is not. §3 is the whole of it, and §8 tests it directly.

---

## 1. Philox, and why it is a separate file

Philox-4x32-10 is a counter-based PRNG: no state, no sequence, just a pure
function of `(seed, counter)`. That is exactly what a GPU needs — every
element derives its own random from its own coordinates, with no ordering
between them.

Triton's implementation (`python/triton/language/random.py`) is the reference
this must match bit for bit, because callers will compare against
`torch.rand`-seeded runs:

```python
for _ in range(n_rounds):                       # n_rounds = 10
    _c0, _c2 = c0, c2
    c0 = umulhi(B, _c2) ^ c1 ^ k0
    c2 = umulhi(A, _c0) ^ c3 ^ k1
    c1 = B * _c2
    c3 = A * _c0
    k0 += PHILOX_KEY_A
    k1 += PHILOX_KEY_B
```

with, for the 32-bit variant, `A = 0xD2511F53`, `B = 0xCD9E8D57`,
`KEY_A = 0x9E3779B9`, `KEY_B = 0xBB67AE85`.

**Separate module** (`philox.py`), for four reasons: it is testable in
isolation against a CPU reference with no attention in the picture; §7's mask
kernel uses it too and must not import the attention builder to get it; the
backward pass needs the identical stream, so a shared module is what keeps
forward and backward from drifting; and it is **going into FlyDSL as a
library**, which sets the bar for what may appear in it.

Written arch-agnostic from the start, which is cheaper than retrofitting:

- **No target intrinsics.** `umulhi` and the low product are plain `fx`
  arithmetic; the backend picks `v_mul_hi_u32` on RDNA4 and whatever the
  equivalent is elsewhere. §4 shows the codegen is already good.
- **No wave-size or layout assumptions.** The module maps
  `(seed, offset) -> N randoms`. How a caller assigns offsets to elements is
  the caller's business — that is §3, and it lives in the attention kernel.
- **Two entry points over one core**, §4.2: a *block* form that returns an
  `M x N` tile and a *streamed* form that returns one row, for callers that
  cannot afford the tile in registers. Both call the same round function, and
  §4.2 requires them to agree bit for bit.
- **Width is a parameter, not a constant.** `PHILOX_WIDTH ∈ {32, 64}` selects
  the variant; the round constants and lane type follow from it. A caller that
  does not care takes the per-arch default.
- **`n_rounds` is a parameter too**, defaulting to 10. Philox-4x32-**7** is
  also standard and cheaper, and someone will eventually want it.

---

## 2. 64-bit seed and offset without 64-bit arithmetic

The requirement is that seed and offset are 64-bit *quantities*. It is not
that the PRNG do 64-bit *arithmetic*. Triton's 32-bit path already takes both:

| Philox input | fed from                              |
| ------------ | ------------------------------------- |
| `k0`, `k1`   | `seed & 0xffffffff`, `seed >> 32`     |
| `c0`, `c1`   | `offset & 0xffffffff`, `offset >> 32` |
| `c2`, `c3`   | 0, 0                                  |

So a 64-bit seed occupies the whole 64-bit key and a 64-bit offset the low
half of the counter, with 32-bit lanes throughout.

**And the split is free, not two shifts.** A 64-bit value lives in two
consecutive 32-bit registers, so its halves are addressable directly — the
compiler lowers `trunc` and `lshr 32; trunc` to register naming. Confirmed on
gfx1201: the whole hi/lo extraction of a `u64` kernarg emits **zero**
instructions between the `s_load_b128` and the store. There is no cost to
carrying 64-bit seed and offset through a 32-bit PRNG, not even a cheap one.

The 64-bit variant instead puts the whole seed in `k0` and the whole offset in
`c0`, with `u64` lanes and the wider constants. It yields 4 x u64 = 8 usable
u32 per call against 4.

---

## 3. The offset scheme *is* the reproducibility contract

Adopt AOTriton's, because it is already the shape callers expect:

```
philox_offset_stride = cdiv(Max_seqlen_k, RN_PER_OFFSET)
batch_philox_offset  = philox_offset_base + off_zh * Max_seqlen_q * philox_offset_stride
offset(m, n)         = batch_philox_offset + m * philox_offset_stride + n // RN_PER_OFFSET
random(m, n)         = philox(seed, offset(m, n))[n % RN_PER_OFFSET]
```

with `off_zh = z * Num_head_q + h`.

**Read what is absent: `BLOCK_M` and `BLOCK_N`.** `m` and `n` are global
element coordinates and the stride comes from `Max_seqlen_k`, so the mask is a
function of `(seed, base, z, h, m, n)` alone. Re-tuning the kernel cannot
change a single random number. That is the contract, and it is worth stating
as one because it is invisible in any test that uses a single tile size.

Two consequences, and a third that follows from making the width
configurable.

- **`Max_seqlen_k` is part of the contract.** The same `(seed, base)` gives
  different masks at different `Max_seqlen_k`. That matches AOTriton and is
  what callers already expect, but it means the mask kernel of §7 must be
  handed the same `Max_seqlen_k`, not the sequence's own length.
- **`RN_PER_OFFSET` is part of it too**, because it appears in the offset
  arithmetic: `n // RN` and `cdiv(Max_seqlen_k, RN)`.

### 3.1 `PHILOX_WIDTH` is in the contract, and that is unremarkable

The two variants are different functions — different round constants,
different lane width, 4 randoms per offset against 8 — so they produce
different streams. That is not a defect to weigh: no two PRNGs agree, and
philox32 and philox64 are simply two PRNGs. The contract is

> the mask is a function of `(seed, base, z, h, m, n)` **and `PHILOX_WIDTH`**

and a mask is reproducible within a build, not across builds that chose
different widths. Cross-architecture bit-exact runs are not achievable anyway.

**The invariant that actually matters is forward/backward, and it is the whole
reason this is a shared module.** Since FAv1 the design has been that the two
passes run a *bitwise identical* PRNG, so the mask never has to be stored or
passed — only `(seed, offset)` crosses the boundary, and the backward pass
regenerates the same numbers from the same coordinates. A dropout mask for a
4K x 4K attention is 268 MB per head; regenerating it is free by comparison.

That places three requirements, all on us rather than on the caller:

- **One module, both passes.** `philox.py` is what keeps them from drifting.
  A copy-paste would drift on the first constant anyone re-derived.
- **`PHILOX_WIDTH` is a parameter of the op, not of a kernel.** Forward,
  backward and the mask kernel take the same value. A mismatch is silent: the
  backward pass regenerates a *different* valid mask and produces plausible
  gradients.
- **The width must be recorded, not inferred**, so a caller can tell whether
  two runs are comparable — and so the backward pass cannot guess wrong.

### 3.2 The offset is 64-bit; the arithmetic that builds it must be too

Two different things could wrap, and only one of them is safe by
construction.

**The counter does not alias.** A 64-bit offset is split across `c0` (low) and
`c1` (high), and the seed across `k0`/`k1`, so Philox sees a 128-bit counter of
which we use 64 bits. Offsets `0` and `2^32` land on genuinely different
counter states, and so do seeds `1` and `1 + 2^32`. Verified against a CPU
reference rather than assumed — putting the offset in `c0` alone *would* alias,
which is presumably why Triton splits it.

**The arithmetic that computes the offset can overflow, and it is int32.**

```
offset(m, n) = base + off_zh * Max_seqlen_q * stride + m * stride + n // RN
                      \_______ this product _______/
```

`off_zh`, `Max_seqlen_q` and `stride` are all `int32` in AOTriton, and the
peak offset is about `B * H * Sq * Sk / RN`:

| B   | H   | Sq   | Sk   | RN  | peak offset    | fits `i32` |
| --- | --- | ---- | ---- | --- | -------------- | ---------- |
| 1   | 8   | 4096 | 4096 | 4   | 33,554,432     | yes        |
| 8   | 32  | 4096 | 4096 | 4   | 1,073,741,824  | yes        |
| 8   | 32  | 8192 | 8192 | 4   | 4,294,967,296  | **no**     |
| 32  | 64  | 8192 | 8192 | 4   | 34,359,738,368 | **no**     |

So it overflows at large-but-real shapes — a 64-batch 8K-context model is not
exotic. And the failure mode is the bad one: the offset wraps into a range
already used by another `(z, h)` pair, so two heads silently share a dropout
stream. Every statistical test still passes, because a shared stream is just
as random as an unshared one.

**Rule: build the offset in 64-bit from the start.** `base` is already `u64`;
the product must be widened *before* multiplying, not after — widening the
result of an `int32` multiply preserves the wrap. This is the same width
hazard as `sdpa-varlen-plan.md` §5, in a place where nothing faults to
announce it.

The rule is close to free on this target, which removes the usual reason to
skip it. A `32 x 32 -> 64` widening product is **one instruction** in both
paths: `s_mul_u64` when the operands are uniform, `v_mad_co_u64_u32` when they
are per-lane. Measured from the emitted ISA, not assumed.

Choosing `RN = 8` (§4) halves the offset space and buys one more doubling of
head count or context before the wrap — a small point in its favour, and not
a substitute for 64-bit arithmetic.

---

## 4. `PHILOX_WIDTH`: a configurable knob with a measured default

Not a decision to make once. Which variant wins depends on the target's
integer ALU, and this module is meant to outlive gfx1201 — so the width is a
constexpr with a per-arch default table, the same shape as `BLOCK_M`.

**A constexpr, not two kernels.** Builds are JIT'd per configuration already,
so a `const_expr` branch costs nothing at runtime and keeps one source, one
set of tests, and one place where the round constants live. Two kernels would
double the surface of a module that is about to become a library.

**The ISA counts, measured on gfx1201.** Philox's inner loop needs `umulhi`
and a low product on the *counter*, which is per-lane and therefore VALU:

| operation                      | emitted                                                         | VALU ops |
| ------------------------------ | --------------------------------------------------------------- | -------- |
| `32 x 32 -> 64` widening       | `v_mad_co_u64_u32`                                              | **1**    |
| `64 x 64 -> 64`, low half only | `2x v_mul_lo_u32` + `v_mad_co_u64_u32` + `v_add3_u32` + carries | **~6**   |

`v_mul_hi_u32` gives the 32-bit `umulhi` in one instruction; the 64-bit
variant needs the high 64 bits of a 64x64 product, worse than the low half
above. Roughly 10 VALU ops for 4 x u32 against ~42 for 8 x u32 — about **2x
more work per random bit** on this target.

**The 64-bit case is about call count, not arithmetic.** Element `i` of the
flattened accumulators is KV column
`(i//16)*32 + ((i//8)%2)*16 + klane*8 + i%8`, so a group of eight consecutive
elements is **eight contiguous columns** — one 64-bit call, or two 32-bit
calls. That halves the offset arithmetic and the registers holding results,
and doubles the headroom before §3.2's overflow.

### 4.1 Register pressure is the second axis, and the benchmark must report it

Throughput is not the only difference. Philox's state is `c0..c3` plus
`k0, k1` — **six registers at 32-bit lanes, twelve at 64-bit**, since each
`u64` occupies a register pair. Outputs differ the same way: 4 x u32 against
8 x u32, held until consumed. So the 64-bit variant costs roughly ten more
VGPRs in a kernel that already spills at head_dim 192 and above (plan1 §6.2).

Two things bound that, and they are worth stating because they say where to
look:

- **Dropout extends nothing's lifetime.** It sits between the row sum and the
  `exp2(m_i - m_new)` correction — in Triton's shape, between `l_ij` and
  `alpha`. Everything live across that window (`p`, the O accumulators, `m`,
  `l`) is live with or without dropout. The added pressure is Philox's own
  state and the randoms, both of which die inside the window.
- **Randoms are consumed per group.** Applying eight columns at a time keeps
  at most one call's output live rather than a whole tile's — and §4.2 takes
  that further.

Bounded is not zero at head_dim 192+, where ten registers is the difference
between spilling and not. So the microbenchmark reports **VGPR count alongside
randoms/second**, and step 3 re-checks spills after integration: a PRNG
measured in isolation has the whole register file to itself, which is not the
situation it will be in.

### 4.2 Experimental: streamed generation, one row at a time

> **Measured, and the answer is that this section's premise does not hold in
> this stack.** The two shapes compile to *byte-identical ISA* at head_dim 192
> and at head_dim 256, including the 140 bytes of scratch at 256 — same VGPR
> count, same spill instructions, throughput within 0.1%. The reasoning below
> is imported from Triton, where `tl.rand` over a tile is a single tile-shaped
> op and the tile really is materialised. Here `range_constexpr` unrolls the
> group loop, so every random is an individual SSA value in one basic block
> with no control flow between the calls. The order they appear in the source
> is not information the backend has to respect: how many are live at once is
> the scheduler's decision, not the author's. No knob shipped — see step 4.

`PHILOX_WIDTH` trades ALU against registers. This trades *shape* against
registers, and is the lever to reach for when the width choice is not enough.

**The block form** — AOTriton's `fast_philox` — generates an `M x N` tile of
randoms and then applies dropout to it. The tile is live all at once, so the
register cost is `M*N / threads` per thread on top of everything the softmax
already holds.

**The streamed form** generates and applies one row at a time:

```
for m in rows_this_lane_owns:
    r = philox_row(seed, offset_of(m), N)     # N randoms, this row only
    apply(p[m], r)                            # consumed immediately
```

so at most one row of randoms is ever live. For the attention kernel that is
`N` values instead of `M*N`, and `N` is `BLOCK_N` — at BLOCK_N 32 with the
32-bit variant, eight offsets' worth rather than a whole tile's.

**Why this is worth having rather than always doing the obvious thing.** The
block form is not wasteful in general: it amortises the offset arithmetic
across the tile and gives the scheduler a wide window of independent loads.
Under low register pressure it should win. The streamed form pays a little
more address arithmetic per row to keep the live set flat. Which is better is
therefore a function of the *caller*, not of the PRNG:

| caller                      | pressure                      | form     |
| --------------------------- | ----------------------------- | -------- |
| §7's mask kernel            | none — it does nothing else   | block    |
| attention at small head_dim | comfortable                   | block    |
| attention at head_dim 192+  | already spilling (plan1 §6.2) | streamed |

So both ship, and the attention kernel picks by head_dim exactly as it picks
`Q_TILES_PER_BLOCK`.

**The two forms must agree bitwise, and that is a test rather than a
deduction.** Philox is a pure function of `(seed, offset)`, so *if* both forms
derive the same offset for element `(m, n)` they cannot disagree. The risk is
entirely in that "if": the streamed form is the one that invites an
incremental derivation — `offset += stride` as the row loop advances, instead
of recomputing `base + m*stride + n//RN` from coordinates. Incremental is
cheaper and agrees with the block form right up until it does not, because the
two wrap differently at §3.2's overflow. A row-major walk that never revisits
a row also hides the disagreement from any test that checks one row at a time.

So: **`philox_row(seed, offset_of(m), N)` takes an absolute offset**, computed
from coordinates by the caller, and the module never advances a counter across
calls. The gate is a direct comparison of the whole tile, both forms, both
widths.

**So gfx1201's default is measured, not argued.** A microbenchmark under
`kernels/microbench/`, timing both at the shapes the attention loop uses and
reporting randoms/second *and* VGPR count, because the register cost is what
the op counts miss. The result sets one entry in a table:

```python
_PHILOX_WIDTH_BY_ARCH = {
    "gfx1201": 32,      # measured; see kernels/microbench/philox.py
}                       # default 32 elsewhere until someone measures
```

**Changing an entry changes every mask** (§3.1), so this table is versioned
and documented rather than re-swept like the tuning tables. That asymmetry is
the one thing about `PHILOX_WIDTH` that is not like `BLOCK_M`.

## 5. The threshold trick: no float conversion

AOTriton never converts a random to a float. It compares the raw integer:

```python
idropout_p = ((dropout_p - 0.5) * 0xFFFFFFFF).to(tl.int32)
keep       = rng_output > idropout_p          # rng_output bitcast to int32
```

A uniform `u32` read as `i32` is uniform on `[-2^31, 2^31)`, so a threshold at
`(p - 0.5) * 2^32` keeps a `1 - p` fraction. `p = 0` gives `-2^31` (keep all),
`p = 1` gives `+2^31` (keep none).

This is worth copying rather than doing the obvious `uint_to_uniform_float(x)
< p`: it removes a convert and a float compare per element, on a path that
runs once per score. The host computes `idropout_p` once.

The surviving elements are then scaled by `1 / (1 - p)`, which folds into the
existing softmax normalisation rather than becoming its own multiply — see §6.

---

## 6. Where dropout applies, and what it must not disturb

Dropout is applied to `P` (the post-softmax probabilities), after the
exponential and before `P @ V`:

```
    P = exp2(S - m)
    P = keep ? P * (1/(1-p)) : 0        <- here
    O += P @ V
    l += sum(P)                          <- **before** dropout, not after
```

**`l` accumulates the undropped sum.** Dropout scales the output but must not
change the softmax denominator, or the result is no longer an expectation of
the undropped attention. AOTriton keeps `l_i` on the pre-dropout `p`, and the
LSE it writes is the undropped one — which is also what the backward pass
needs. Getting this backwards produces plausible output that is wrong by a
factor that varies per row, and no shape check catches it.

The `1/(1-p)` scale can be folded into the existing `1/l` epilogue multiply
rather than applied per element, since it is uniform — one scalar instead of
`NUM_S_VALS` multiplies per tile.

---

## 7. The mask kernel

A second kernel that fills a `(B, H, Sq, Sk)` tensor with the same mask,
mirroring AOTriton's `debug_fill_dropout_rng`. It shares the Philox module and
the §3 offset scheme, and takes the same `(seed, offset_base, Max_seqlen_q,
Max_seqlen_k)`.

Its value is not debugging output; it is that **it makes the reproducibility
contract testable without the attention kernel**. §8's tiling-invariance test
compares this kernel's output against itself at different `BLOCK_M`/`BLOCK_N`,
and against the attention kernel's actual behaviour, which is a far sharper
check than comparing two attention outputs statistically.

Emit both encodings AOTriton does: `float32` in `[0, 1)` for inspection, and
the raw keep/drop for comparison.

---

## 8. Test matrix

The statistical tests are the weak ones. Put the weight on the exact ones.

| test                                | what it catches                                                                             |
| ----------------------------------- | ------------------------------------------------------------------------------------------- |
| **Philox vs a CPU reference**       | the PRNG itself, bit for bit, before attention is involved                                  |
| **Philox vs `torch`/Triton stream** | that we match the stream callers expect                                                     |
| **tiling invariance**               | the §3 contract: same mask at BLOCK_M/N 64/32, 128/32, 256/64                               |
| **mask kernel vs attention**        | that the kernel actually applies the mask it claims                                         |
| `ENABLE_DROPOUT=0` bit-identical    | objective 4                                                                                 |
| `p = 0`                             | must be bit-identical to dropout off, not merely close                                      |
| `p = 1`                             | every element dropped; `O = 0`                                                              |
| mean/variance at p = 0.1, 0.5, 0.9  | the weak test, kept because it catches a wrong threshold                                    |
| seed/offset are 64-bit              | pass values above 2^32 and check the mask changes                                           |
| **both `PHILOX_WIDTH` values**      | every row above, run twice -- the module is a library, not a gfx1201 detail                 |
| the two widths *differ*             | that `PHILOX_WIDTH` reaches the stream rather than being ignored                            |
| **block vs streamed, whole tile**   | §4.2 — compared as a tile, not row by row: a row-major walk hides an incremental-offset bug |
| large `B*H*Sq*Sk`                   | §3.2: the offset product overflowing `int32` and aliasing heads                             |

The 64-bit row matters: a 32-bit truncation of the seed passes every other
test in this table, because any consistent stream looks random.

---

## 9. Steps

### Step 1 — `philox.py`, standalone and arch-agnostic
Both widths behind `PHILOX_WIDTH`, plus `n_rounds`. Tested against a CPU
reference and against Triton's stream. No attention, no target intrinsics.

**Gate:** bit-exact agreement with the reference **at both widths**, for a
spread of seeds and offsets including values above 2^32.

**Outcome — met.** `philox.py` and `test_philox.py`; 9 pass, 3 skip. The
reference turned out to be three rather than two: **Random123's published
known-answer vectors** for Philox-4x32-10 pin the round function against the
algorithm's own specification, which is the only check here that does not
depend on someone's reading of someone's source. The numpy model then extends
that to the 64-bit-seed/offset packing and to width 64.

The Triton cross-check is written but **skips in this environment** — Triton's
HIP backend compiles a utility module at import and the Python dev headers are
absent. What that costs is narrow: the KAT vectors cover the algorithm and the
numpy model covers the packing, so only a *misreading of Triton specifically*
would slip through, and the constants and word order are shared with the
vectors.

### Step 2 — the microbenchmark, and gfx1201's default
§4. Measure randoms/second and VGPR cost at both widths, and set the
`_PHILOX_WIDTH_BY_ARCH` entry for gfx1201.

**Gate:** a number and a recorded rationale, in the table rather than in a
commit message — §3.1 makes this entry mask-visible, so the next person to
touch it needs to see why it is what it is.

**Outcome — 32-bit, by a wider margin than predicted.**

| width | u32/call | G randoms/s | VGPRs (one call) |
| ----- | -------- | ----------- | ---------------- |
| 32    | 4        | **284.5**   | **8**            |
| 64    | 8        | 65.7        | 26               |

**4.3x** the throughput per random and 18 fewer registers. Both margins beat
§4's instruction-count estimate of ~2x, and the reason is that the estimate
counted the *low* half of a 64x64 product: Philox needs the **high** half,
whose expansion is worse. So the 64-bit variant's one advantage — eight
randoms per call, matching an eight-column accumulator group exactly — does
not come close to paying for itself on this target.

Recorded in `philox.py`'s `_WIDTH_BY_ARCH` with the numbers and the reasoning,
not in this file, since that is where someone changing it will be.

A note on the benchmark itself: VGPRs are measured at a **one-call** unroll.
The throughput run unrolls 64 calls to saturate the pipeline, which also
saturates the register file and made both widths report the 256 cap — an
artifact that would have hidden the entire register difference.

### Step 3 — `ENABLE_DROPOUT` in the attention kernel
The offset scheme, the threshold compare, the `l`-before-dropout ordering.

**Gate:** `ENABLE_DROPOUT=0` bit-identical; `p=0` bit-identical to off; and
**a spill check at head_dim 192 and 256**, at both widths and both generation
forms (§4.1, §4.2) — the microbenchmark measures Philox with the register file
to itself, which is not the situation it will be in here.

**Done.** `ENABLE_DROPOUT=0` is bit-identical to HEAD and `p=0` is
bit-identical to dropout off; `p -> 1` gives exactly zero; LSE matches the
undropped reference; the mask depends only on `(seed, offset)`. The spill
check ran at both widths and is recorded in `sdpa_lore_gfx1201.md` — the cost
is +3 VGPRs at head_dim 128 and +24 bytes of scratch at 256, and width 64 is
worse on registers as well as throughput.

Only the block form exists so far, so "both generation forms" carries into
step 4. What the attention kernel does today is already finer-grained than
§4.1's whole-tile form: it generates and consumes 8 randoms per accumulator
group, so at most one group is live. Step 4's comparison is therefore that
per-group form against the whole-tile one, not against a not-yet-written
streamed one.

### Step 4 — streamed generation, and whether it is worth it
§4.2. Add `philox_row` beside the block form, gate on the whole-tile bitwise
comparison, then measure spills and throughput at head_dim 192 and 256 against
the block form.

**Gate:** bitwise identical to the block form at both widths, *and* a number
saying whether it helps. It is an experiment — if it does not reduce spills it
still ships in the module, since §7's mask kernel and low-pressure callers
keep using the block form, but the attention kernel keeps whichever won.

**Done — the experiment ran and nothing shipped, which is the correct
outcome.** Two findings, in the order they were reached:

1. **`philox_row` was already there under another name.** §4.2 specifies
   "takes an absolute offset, and the module never advances a counter across
   calls". That is exactly the contract of `Philox.span_u32` / `keep_span`
   from step 3, which take a first offset and a count. Adding `philox_row`
   would have been a second name for one helper.

2. **The block/streamed distinction does not survive the compiler.** Both
   shapes were built behind a temporary knob and compared:

   | head_dim | form      | VGPR | scratch | spill insts | TFLOPS |
   | -------- | --------- | ---- | ------- | ----------- | ------ |
   | 128      | per-group | 226  | 0       | 0           | 58.86  |
   | 128      | block     | 226  | 0       | 0           | 58.82  |
   | 192      | per-group | 250  | 0       | 0           | 63.98  |
   | 192      | block     | 250  | 0       | 0           | 64.03  |
   | 256      | per-group | 256  | 140     | 66          | 71.15  |
   | 256      | block     | 256  | 140     | 66          | 71.20  |

   Not merely equal — the emitted ISA is byte-identical at 192 (3346 lines)
   and at 256 (3629 lines). The bitwise mask gate passed at both widths and
   both head_dims before the knob was removed, but it was testing two spellings
   of one program.

The knob was reverted rather than shipped: a build parameter that provably
cannot change the output or the code is configurability with a maintenance
cost and no lever attached. What the attention kernel does — generate and
consume 8 columns of one row per accumulator group — is what it already did in
step 3, and is finer-grained than §4.2's row anyway, since a row is `BLOCK_N`
columns.

The register-pressure lever that *does* exist is `PHILOX_WIDTH`, measured in
step 2 and re-measured under integration in step 3.

### Step 5 — the mask kernel, and the invariance test
§7 and the tiling-invariance row of §8, which is the phase's real gate.

---

## 10. Risks

| risk                                    | why it matters                                                                         | mitigation                                                                   |
| --------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Mask depends on tiling**              | silently breaks reproducibility on any future re-tune                                  | §3; the invariance test is the phase's gate                                  |
| `PHILOX_WIDTH` changed after shipping   | every mask changes (§3.1); unlike `BLOCK_M` this is observable                         | versioned, documented table; not re-swept with tuning                        |
| Producer and consumer disagree on width | backward regenerates a different mask; gradients stay plausible                        | width is a parameter of the *op*, not of each kernel                         |
| `l` accumulated after dropout           | plausible output, wrong by a per-row factor, no shape check                            | §6; assert LSE against the undropped reference                               |
| 64-bit seed silently truncated          | every statistical test still passes                                                    | the >2^32 row of §8                                                          |
| Offset product overflows `int32`        | two heads share a stream; every statistical test still passes                          | §3.2 -- widen before multiplying; assert the peak offset                     |
| Register pressure from Philox state     | ~6 vs ~12 registers by width, in a loop already spilling at head_dim 192+              | §4.1; VGPR count in the microbenchmark *and* a spill check after integration |
| Streamed form drifts from block form    | an incremental offset agrees until it wraps (§3.2), and a row-major walk never notices | §4.2 — `philox_row` takes an *absolute* offset; compare whole tiles          |
| Diverging from Triton's stream          | callers compare against seeded `torch` runs                                            | step 1 gates on bit-exactness, not distribution                              |

---

## 11. What this phase does *not* do

- **`RETURN_ENCODED_SOFTMAX`** as a fused output of the attention kernel.
  AOTriton ships it `options=[False]`; §7's separate kernel covers the need.
- **Backward.** dk/dv must regenerate the identical mask, which is exactly why
  Philox lives in its own module (§1) and why `PHILOX_WIDTH` is a parameter of
  the op rather than of a kernel.
- **The move into FlyDSL proper.** `philox.py` is written to the standard a
  library needs — arch-agnostic, parameterised, independently tested — but it
  stays under `kernels/attention/` until there is a second caller outside this
  directory. Per `CLAUDE.md`, DSL-level numeric helpers belong in
  `python/flydsl/expr/`, and that is the eventual home.
- **Varlen + dropout.** The offset base uses `off_zh = z * Num_head_q + h`, and
  under varlen `z` is the sequence index, so the scheme is already
  per-sequence — unlike bias, which needed the row offset. Worth confirming
  when it is wired, but no design decision is pending.
