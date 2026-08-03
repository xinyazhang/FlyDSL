# P3 in detail: Variable-Length Sequences

Companion to `sdpa-close-gap-plan1.md`, which resolves the LSE layout but whose
D3 naming (`seq_info_q/k`) this plan supersedes — see §3.

Written after P6, and shaped by it: the gSWA phase established that a *bitwise*
equivalence to an existing path is both achievable and the sharpest oracle in
this codebase. Varlen admits the same gate, for the same reason, and §7 makes
it the headline requirement rather than an afterthought.

---

## 0. The goal, including the one that is easy to leave unstated

1. **Support variable-length sequences** — which is not one feature but a
   product of three independent choices, §1.
2. **Absorb P6 step 4.** `Window_left`/`Window_right` must accept the sentinels
   `0x80000001` / `0x80000002`, resolved *per sequence*. gSWA deferred this
   here because a sentinel is meaningless without per-sequence lengths.
3. **Do not add an `if varlen_mode` to the body.** The bits may be decoded
   exactly once, in the prologue, into six scalars. Everything downstream reads
   those scalars and cannot tell what layout it is in.

**Success criterion for objective 3:** outside the prologue, the kernel
contains no reference to `varlen_bits`, `seqinfo_*`, or `Max_seqlen_*`.

---

## 1. Varlen is three orthogonal choices, not an enum

AOTriton models this as `VarlenType ∈ {None, CompactVarlen, PaddedVarlen,
StridedVarlen}`. That enum is a *sample* of the space, not a description of it,
and PyTorch already ships a case outside it — see §1.4. The three axes:

### A. Is the token axis stacked?

| | shape | how sequence `z` is selected |
|---|---|---|
| `BATCHED` | BHSD | the batch index: `batch_index = z` |
| `STACKED` | 1THD (rank 4, `B` fixed at 1) | a row offset along `T` |

### B. How is the *length* of sequence `z` given?

| | array shape | `seqlen(z)` |
|---|---|---|
| `MAX` | — | `Max_seqlen` (every sequence the same) |
| `CUMULATIVE` | `(N+1,)` | `a[z+1] - a[z]` |
| `INDIVIDUAL` | `(N,)` | `a[z]` |

### C. Where does sequence `z` *start* along the token axis?

| | `row_off(z)` |
|---|---|
| `IMPLIED` | `0` if `BATCHED`, else `z * Max_seqlen` |
| `ARRAY` | `b[z]`, from a cumulative position array |

**All three are per-side.** Q and K may differ, and the case that forces this
is ordinary: packed queries against a rectangular KV cache.

### 1.1 Why A cannot be folded into C

It is tempting to drop `STACKED` and say `BATCHED` is just `row_off = z *
Max_seqlen`. That is true only when `stride_batch == Max_seqlen * stride_seq`,
i.e. for a contiguous BHSD tensor — and plan1 §0 is explicit that **shape does
not imply layout**. `storage_flip` alone breaks it. The batch stride is an
independent number the kernel reads off the tensor, so selecting a batch slice
and offsetting within one are two different operations.

### 1.2 Why C has two states, not the three you might expect

The natural reading is three: *reuse `cu_seqlens`*, *use `seq_strides`*, or
*regular*. But the first two are **the same kernel code with a different
pointer** — both compute `row_off = b[z]` from a cumulative array; `cu_seqlens`
and TE's `cu_seqlens_padded` differ only in whether the sequences have gaps
between them, which the kernel never needs to know.

So `CompactVarlen` and `StridedVarlen` are not two modes. They are one mode
called with two different arrays. That collapse is the main return on
decomposing at all, and it is worth stating as a result rather than hiding as
an implementation detail.

(A third state `REUSE`, meaning "position array == length array", would save
one live pointer. It is representable in the reserved bits if a measurement
ever justifies it. Not now.)

### 1.3 AOTriton's four types, decomposed

| `VarlenType` | Q side | K side |
|---|---|---|
| `None` | BATCHED, MAX, IMPLIED | BATCHED, MAX, IMPLIED |
| `CompactVarlen` | STACKED, CUMULATIVE, ARRAY(`cu_q`) | STACKED, CUMULATIVE, ARRAY(`cu_k`) |
| `PaddedVarlen` | BATCHED, CUMULATIVE, IMPLIED | BATCHED, CUMULATIVE, IMPLIED |
| `StridedVarlen` | STACKED, CUMULATIVE, ARRAY(`sst_q`) | STACKED, CUMULATIVE, ARRAY(`sst_k`) |

### 1.4 The case the enum cannot express

`torch.nn.attention.varlen.varlen_attn` takes `cu_seq_k` **and** `seqused_k`
together:

> `seqused_k` — Number of valid KV tokens per batch element; shape `(N,)`.
> When set, only the first `seqused_k[i]` tokens in the key/value sequence for
> batch element `i` participate in attention. Useful for KV-cache decoding
> where the cache slot is larger than the actual sequence.

That is **length from an individual array, position from a cumulative one** —
axis B and axis C taking their values from *different tensors*. No
`VarlenType` covers it, because the enum assumes one array serves both roles.

Decomposed, it is just: K side = `STACKED, INDIVIDUAL, ARRAY`.

And the rectangular-cache variant — a BHSD cache with `seqused_k` and no
`cu_seq_k` at all — is `BATCHED, INDIVIDUAL, IMPLIED`. Also uncovered by the
enum, also free here.

---

## 2. `VarlenBits : u32`

One byte per side, identically decoded, so the kernel has **one** decoder
called twice.

```
  bits  0      STACKED     0 = BHSD              1 = 1THD
  bits  2:1    LENGTH      0 = MAX               1 = CUMULATIVE   2 = INDIVIDUAL
  bits  3      POSITION    0 = IMPLIED           1 = ARRAY
  bits  7:4    reserved

  VarlenBits = q_byte | (k_byte << 8)
  bits 31:16   reserved   (paged KV, §8)
```

`VarlenBits == 0` is BHSD / MAX / IMPLIED on both sides — the conventional
dense case, and the default, exactly as required.

| configuration | bits |
|---|---|
| dense | `0x0000` |
| compact varlen | `0x0B0B` |
| padded varlen | `0x0202` |
| strided varlen | `0x0B0B` *(same code; `seqinfo_?1` differs)* |
| packed Q, `seqused_k` on packed KV | `0x0D0B` |
| packed Q, `seqused_k` on a BHSD cache | `0x040B` |

That two AOTriton types share `0x0B0B` is the §1.2 collapse showing up in the
encoding.

---

## 3. `seqinfo_q0 / q1 / k0 / k1`

Supersedes D3's `seq_info_q/k`, which assumed one array per side. Two are
needed because B and C can read different tensors (§1.4). They are named by
**role**, so the bits say only how to *interpret* them, never which slot to
look in:

| | role | read when | indexed |
|---|---|---|---|
| `seqinfo_?0` | **length** source | `LENGTH != MAX` | `[z]`, `[z+1]` |
| `seqinfo_?1` | **position** source | `POSITION == ARRAY` | `[z]`, and `[N]` for the total |

Compact varlen passes the same pointer as both `?0` and `?1`. That redundancy
is deliberate: fixing the roles keeps the decoder branch-free on *which*
pointer, at the cost of one duplicated argument.

### 3.1 The decoder

```python
def decode_side(bits8, z, N, max_seqlen, s0, s1):
    stacked = bits8 & 1
    lenmode = (bits8 >> 1) & 3
    posmode = (bits8 >> 3) & 1

    seqlen = (max_seqlen        if lenmode == 0 else
              s0[z + 1] - s0[z] if lenmode == 1 else
              s0[z])

    row_off = s1[z] if posmode else (z * max_seqlen if stacked else 0)
    batch_index = 0 if stacked else z
    return seqlen, row_off, batch_index
```

Called twice. The Q call additionally yields the LSE stride:

```python
lse_stride = s1_q[N] if q_posmode else (N * max_seqlen_q if q_stacked
                                        else max_seqlen_q)
```

### 3.2 LSE is not a fourth choice

Plan1 resolved LSE as one branch-free offset formula, `(b*H + h)*S + s`. Under
this decomposition that is **exactly Q's addressing applied to a rank-2
tensor**: `b = batch_index_q`, `s = row_off_q`, `S = lse_stride`. So the
`(B*H, Max_seqlen_q)` and `(H, TotalS)` layouts are not modes to select
between — they are what the formula produces for `BATCHED` and `STACKED`. No
LSE bits, and nothing to keep in sync.

This is the strongest evidence that the decomposition is the right one: a
choice that looked independent turns out to be derived.

---

## 4. What changes in the kernel: six scalars

After the prologue, everything is one of:

| | `seqlen_q` | `seqlen_k` | `q_row_off` | `k_row_off` | `batch_index` | `lse_stride` |
|---|---|---|---|---|---|---|

and the two places that consume them are already written in the right shape.

**Addressing.** `_addr_pair` computes `bh = batch_idx * s_batch + head *
s_head`. Adding `+ q_row_off * s_seq` covers every configuration, for all four
tensors. Note `batch_index` is *also* now decoded rather than
`gpu.block_idx.z` — the one existing line that changes.

**LSE.** `_lse_base = (batch_idx * num_head_q + head_q) * lse_stride` becomes
the same expression `+ q_row_off`, per §3.2.

**Cost.** `z` is uniform per workgroup, so the `seqinfo` reads are **scalar**
loads — at most six, once, into SGPRs. They do not touch the VGPR budget gSWA
just raised.

---

## 5. Row offsets are 64-bit; the hazard is width, not sign

gSWA's arithmetic hazard was signedness. This one points the other way.

`q_row_off * s_seq` on a packed tensor is a *whole-batch* quantity: at
`T = 128K`, `H = 32`, `D = 128` the byte offset passes 2^32. The kernel's
addressing is deliberately a **64-bit base plus a 32-bit offset** (plan1). So:

1. **`q_row_off` / `k_row_off` go into the 64-bit base**, never the 32-bit
   per-lane offset. §4's `bh` is the insertion point; `toff` must not see them.
2. `seqinfo` values are `int32` on the wire — they are, in both PyTorch and TE
   — and must be widened *before* multiplying by a stride.
3. Lengths stay signed `int32`, and everything in gSWA plan §2.4 still applies,
   because `window_right = seqlen_k - seqlen_q` is computed from them.

An overflowed offset produces a wrong address inside the same allocation, so it
reads plausible data. Like the gSWA prefetch bug, it is invisible at test
scale — hence the one deliberately large case in §7.

---

## 6. The grid grows, and a workgroup must be able to do nothing

The Q grid axis becomes `ceil(Max_seqlen_q / BLOCK_M)`, so **every sequence
gets the longest sequence's worth of workgroups**.

### 6.1 An early exit without an early `return`

`CLAUDE.md` forbids branch-local `return` in traced functions, and this kernel
is one long single-exit trace. The structural equivalent is to make the work
empty rather than skip it:

- Force the KV loop counts to zero. Under gSWA that is already the natural
  representation (`n_l = n_f = n_r = 0`), so the causal path needs nothing new;
  the non-causal path needs `kv_upper` clamped the same way.
- Rely on the existing row-bound masking for the stores. **Verify this covers
  the LSE store**, which is a separate store with its own guard, not just O.

### 6.2 Measure the waste

The idle fraction is `1 - mean(seqlen)/max(seqlen)`, which is large for real
batches. AOTriton excludes varlen from persistent
(`unsupported_by_persistent = Num_seqlens != 0`) and eats the same cost. Fixing
it is persistent-dynamic's job, not this phase's — but **produce the number
here** so that task starts with a measurement instead of an intuition.

---

## 7. Test matrix, and the oracle that makes it cheap

**Headline gate: varlen with `N` sequences must equal `N` separate dense calls,
bitwise.**

This holds for a reason, not by luck. A varlen workgroup and its dense
counterpart cover the same tiles, in the same order, with the same values; only
the base address differs. Every floating-point operation is identical.

It is the right gate here specifically because **an addressing bug that lands
inside the right allocation reads plausible data** — a tolerance comparison
against a reference would accept it. Two caveats to write into the test so a
later reader does not weaken it: the grid differs (and under `_REVERSE_Q_TILES`
so does the tile-to-workgroup mapping), which changes nothing any surviving
workgroup computes; and LSE lives in a different layout, so compare
per-sequence slices rather than buffers.

### Axes

| axis | values | why |
|---|---|---|
| Q byte × K byte | the six rows of §2, plus mixed Q/K | the point of the decomposition |
| length distribution | uniform; one-long-many-short; all-equal | all-equal hides position bugs |
| `N` | 1, 2, 7, 22 | `N = 1` must reduce to dense exactly |
| `seqlen = 0` | leading, middle, trailing | must write nothing at all |
| `seqlen_k = 0`, `seqlen_q > 0` | | every row dead: `O = 0`, `LSE = +inf` |
| ragged lengths | primes | tiles ending mid-sequence |
| causal | off, top-left, bottom-right, window | `window_right` is per-sequence now |
| sentinels | `0x80000001`, `0x80000002` | objective 2; must match the host table |

### Properties beyond "matches the reference"

1. **Single-sequence reduction.** `N = 1` compact must be bitwise identical to
   the dense call. Cheapest instance of the headline gate; make it pass first.
2. **Padded vs compact agreement.** The same logical batch both ways. Catches a
   `batch_index` / `row_off` mix-up, because those two configurations differ in
   *precisely* those fields.
3. **Position array actually read.** Build a case whose gaps are non-zero and
   assert it differs from the gapless interpretation of the same buffer —
   otherwise an implementation that ignores `seqinfo_?1` passes everything.
4. **`seqused_k` shortens, and only K.** With `seqused_k[z] < cu_k[z+1]-cu_k[z]`
   the result must equal a dense call on the truncated K, and must *differ*
   from one on the full K. This is the §1.4 case; nothing else covers it.
5. **One large case**, `T` big enough that `q_row_off * stride` exceeds 2^32,
   marked `large_shape`. §5 is unreachable at normal test scale.

---

## 8. Steps

### Step 1 — bits, decoder, and the stacked/cumulative path
`VarlenBits`, the four `seqinfo` pointers, the §3.1 decoder, the six scalars,
the empty-work path, LSE via §3.2. Enable `0x0000` and `0x0B0B` only; reject
other bytes with a clear error.

**Gate:** §7 property 1, then the headline gate for compact. Dense must be
**bit-identical to today** — it will be if the decoder folds at `bits == 0`.

### Step 2 — `IMPLIED` positions and `BATCHED` stacking
Enables `0x0202` (padded varlen) and, for free, THD-with-uniform-stride.
Should be small if step 1 factored the decoder properly; if it is not, that is
the signal to refactor before continuing.

### Step 3 — `INDIVIDUAL` lengths
Enables `0x0D0B` and `0x040B` — the PyTorch `seqused_k` cases. Two lines in the
decoder plus §7 property 4.

Strided varlen needs **no step**: it is `0x0B0B` with a different pointer
(§1.2). Add it to the test matrix, not to the kernel.

### Step 4 — the window sentinels (P6 step 4)
`parse_window` moves into the kernel: `0x80000001 → (seqlen_q, 0)` and
`0x80000002 → (seqlen_q, seqlen_k - seqlen_q)`, per sequence. The host-side
`_CAUSAL_WINDOW` stays for the dense path — it costs nothing there and keeps
the sentinel off the hot path — but the two must agree, which §7 tests.

---

## 9. Coverage: what these bits cannot say

Asked for explicitly, so stated explicitly. The encoding spans
`2 × 3 × 2 = 12` configurations per side, 144 pairs. Every one is either
meaningful or harmless — `BATCHED + ARRAY`, for instance, means "slice `z`,
then skip `b[z]` rows", which is unusual but well-defined and free. What falls
*outside*:

1. **Paged / block-table KV.** `torch.nn.attention.varlen` accepts a
   `block_table` of shape `(N, max_pages_per_seq)` with K/V shaped
   `(total_pages, page_size, H, D)`. A sequence is then a *list* of physical
   pages, so position is per-page, not per-sequence, and no scalar `row_off`
   can express it. This needs an indirection load inside the KV loop — a real
   feature, not an encoding gap. Bits 31:16 are reserved for it.
2. **Per-sequence head counts or head dims.** Not expressible, and not a thing
   any API asks for.
3. **Differing `N` between Q and K.** The sequence count is shared by
   construction; the bits are per-side but `z` is not.
4. **Non-cumulative position arrays.** `POSITION = ARRAY` reads `b[z]`, which
   works for any array, cumulative or not. But `lse_stride` reads `b[N]`, which
   assumes the array is a prefix sum with a total in the last slot. An arbitrary
   scatter of starts would need the total supplied separately. Both TE's
   `cu_seqlens_padded` and `cu_seqlens` satisfy this, so it costs nothing today
   — but it is an assumption, and it should be asserted on the host rather than
   discovered later.
5. **A sequence starting mid-slot with a length that runs past the next
   sequence's start.** Representable, and the kernel would happily read another
   sequence's tokens. Nothing validates non-overlap. Host-side check.

Items 4 and 5 are the two places where the encoding is more permissive than the
semantics, and both are cheap to guard on the host.

---

## 10. What this phase does *not* do

- **Paged KV** (§9.1) and **persistent-dynamic** (§6.2).
- **Backward.** The dk/dv kernel needs the same six scalars and the sentinels
  transposed (gSWA plan §6). Name the decoder so it can be lifted, not copied.
- **P4 bias / P5 dropout**, both of which gain a varlen dimension once this
  lands: bias is indexed per sequence, and Philox offsets must be per-sequence
  to stay reproducible. Noted now, scoped later.
