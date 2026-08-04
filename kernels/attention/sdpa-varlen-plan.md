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

|           | shape                         | how sequence `z` is selected       |
| --------- | ----------------------------- | ---------------------------------- |
| `BATCHED` | BHSD                          | the batch index: `batch_index = z` |
| `STACKED` | 1THD (rank 4, `B` fixed at 1) | a row offset along `T`             |

### B. How is the *length* of sequence `z` given?

|              | array shape | `seqlen(z)`                            |
| ------------ | ----------- | -------------------------------------- |
| `MAX`        | -           | `Max_seqlen` (every sequence the same) |
| `CUMULATIVE` | `(N+1,)`    | `a[z+1] - a[z]`                        |
| `INDIVIDUAL` | `(N,)`      | `a[z]`                                 |

### C. Where does sequence `z` *start* along the token axis?

|           | `row_off(z)`                            | reads            |
| --------- | --------------------------------------- | ---------------- |
| `IMPLIED` | `0` if `BATCHED`, else `z * Max_seqlen` | nothing          |
| `REUSE`   | `seqinfo_?0[z]` — the length array      | *already loaded* |
| `ARRAY`   | `seqinfo_?1[z]`                         | one scalar load  |

`REUSE` requires `LENGTH == CUMULATIVE`: only then is `seqinfo_?0[z]` a
position. It exists so that classical varlen costs nothing extra — §1.2.

**All three are per-side.** Q and K may differ, and the case that forces this
is ordinary: packed queries against a rectangular KV cache.

### 1.1 Why A cannot be folded into C

It is tempting to drop `STACKED` and say `BATCHED` is just `row_off = z *
Max_seqlen`. That is true only when `stride_batch == Max_seqlen * stride_seq`,
i.e. for a contiguous BHSD tensor — and plan1 §0 is explicit that **shape does
not imply layout**. `storage_flip` alone breaks it. The batch stride is an
independent number the kernel reads off the tensor, so selecting a batch slice
and offsetting within one are two different operations.

### 1.2 Why `REUSE` is a state and not a host-side convenience

It is tempting to drop `REUSE` and have compact varlen pass `cu_seqlens` as
*both* `seqinfo_?0` and `seqinfo_?1`. The kernel would have one fewer case, and
compact and strided would become the same code with a different pointer.

**That costs a memory access on the mode everyone uses.** Under
`LENGTH == CUMULATIVE` the prologue has already loaded `seqinfo_?0[z]` and
`seqinfo_?0[z+1]` to compute the length — and for compact varlen the position
*is* `seqinfo_?0[z]`, already sitting in a register. Reading it again through a
second pointer is a scalar load that buys nothing, plus a live pointer pair.

So `REUSE` is not redundancy; it is the *absence* of a redundant load.
Classical varlen — by far the most common configuration — reads **no position
array at all**, and `seqinfo_?1` is not passed.

The price is that compact and strided are genuinely different codes rather than
one code with two pointers: one extra line in the decoder, against a scalar
load saved on every workgroup of the common path. It also removes a hazard —
with the same pointer in both slots the two *roles* are indistinguishable in
the configuration everyone runs (V5).

### 1.3 AOTriton's four types, decomposed

| `VarlenType`    | Q side                              | K side                              |
| --------------- | ----------------------------------- | ----------------------------------- |
| `None`          | BATCHED, MAX, IMPLIED               | BATCHED, MAX, IMPLIED               |
| `CompactVarlen` | STACKED, CUMULATIVE, REUSE          | STACKED, CUMULATIVE, REUSE          |
| `PaddedVarlen`  | BATCHED, CUMULATIVE, IMPLIED        | BATCHED, CUMULATIVE, IMPLIED        |
| `StridedVarlen` | STACKED, CUMULATIVE, ARRAY(`sst_q`) | STACKED, CUMULATIVE, ARRAY(`sst_k`) |

Only `StridedVarlen` passes `seqinfo_?1` at all.

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
  per-side byte:
  bit   0      STACKED     0 = BHSD              1 = 1THD
  bits  2:1    LENGTH      0 = MAX               1 = CUMULATIVE   2 = INDIVIDUAL
  bits  4:3    POSITION    0 = IMPLIED           1 = REUSE        2 = ARRAY
  bits  7:5    reserved

  bits  7:0    Q side
  bits 15:8    K side
  bits 17:16   LSE_LAYOUT  0 = HEAD_MAJOR        1 = TOKEN_MAJOR  2,3 reserved
  bits 23:18   reserved
  bits 31:24   reserved    (paged KV, §9.1)
```

Byte 2 holds the LSE layout (§3.2); two bits rather than one, deliberately, so
that padded and blocked LSE arrangements have somewhere to go without another
ABI change.

`VarlenBits == 0` is BHSD / MAX / IMPLIED on both sides — the conventional
dense case, and the default, exactly as required.

| configuration                         | bits     | passes `seqinfo_?1` |
| ------------------------------------- | -------- | ------------------- |
| dense                                 | `0x0000` | neither side        |
| compact varlen                        | `0x0B0B` | neither side        |
| padded varlen                         | `0x0202` | neither side        |
| strided varlen                        | `0x1313` | both                |
| packed Q, `seqused_k` on packed KV    | `0x150B` | K only              |
| packed Q, `seqused_k` on a BHSD cache | `0x040B` | neither side        |

Only one of the six reads a position array on both sides. `seqused_k` on
packed KV must use `ARRAY` rather than `REUSE`, because its `seqinfo_k0` holds
individual lengths and so is not a position array — §1.4 showing up in the
encoding.

---

## 3. `seqinfo_q0 / q1 / k0 / k1`

Supersedes D3's `seq_info_q/k`, which assumed one array per side. Two are
needed because B and C can read different tensors (§1.4). They are named by
**role**, so the bits say only how to *interpret* them, never which slot to
look in:

|              | role                | read when           | indexed        |
| ------------ | ------------------- | ------------------- | -------------- |
| `seqinfo_?0` | **length** source   | `LENGTH != MAX`     | `[z]`, `[z+1]` |
| `seqinfo_?1` | **position** source | `POSITION == ARRAY` | `[z]`          |

The roles are fixed and never swapped. `POSITION == REUSE` takes a *position*
out of the length array, which is sound only because `CUMULATIVE` makes that
array hold positions as well — and it reuses the value already loaded rather
than issuing a second access (§1.2).

### 3.1 The decoder

```python
def decode_side(bits8, z, max_seqlen, s0, s1):
    stacked = bits8 & 1
    lenmode = (bits8 >> 1) & 3
    posmode = (bits8 >> 3) & 3

    if lenmode == 0:                       # MAX
        s0_z, seqlen = None, max_seqlen
    elif lenmode == 1:                     # CUMULATIVE
        s0_z = s0[z]                       # the load REUSE then reuses
        seqlen = s0[z + 1] - s0_z
    else:                                  # INDIVIDUAL
        s0_z, seqlen = None, s0[z]

    row_off = (s0_z  if posmode == 1 else  # REUSE: already in a register
               s1[z] if posmode == 2 else  # ARRAY
               (z * max_seqlen if stacked else 0))
    batch_index = 0 if stacked else z
    return seqlen, row_off, batch_index
```

Called twice, and it never reads `[N]`: `lse_stride` comes off the LSE tensor
on the host (§4.2), so no total has to be recovered from a `seqinfo` array.

### 3.2 LSE: the *indices* are derived, the *layout* is not

Plan1 resolved LSE as one branch-free offset formula, `(b*H + h)*S + s`. Under
this decomposition the **inputs** are exactly Q's addressing applied to a
rank-2 tensor: `b = batch_index_q`, `s = row_off_q`, `S = lse_stride`. So
`(B*H, Max_seqlen_q)` and `(H, TotalS)` are not modes to select between — they
are what the formula produces for `BATCHED` and `STACKED`, and no bits are
needed to distinguish them.

**But the arrangement of those indices in memory is a separate choice**, and
an earlier draft of this plan claimed otherwise. Transformer Engine — which
uses AOTriton as a backend — requires LSE in the `(T_q, H_q)` **layout**, not
merely that shape. PyTorch's varlen documentation specifies a shape and says
nothing about memory, but TE's requirement is real and no value of
`lse_stride` produces the transpose.

This is plan1 §0 applying to LSE exactly as it applies to Q/K/V/O: *shape does
not imply layout*. Having asserted that principle for the rank-4 tensors and
then quietly assumed the rank-2 one was head-major was inconsistent.

Two formulas, selected by `LSE_LAYOUT`:

| `LSE_LAYOUT`    | offset                                 | contiguous axis | pitch        |
| --------------- | -------------------------------------- | --------------- | ------------ |
| 0 `HEAD_MAJOR`  | `(b * H + h) * lse_stride + s`         | token           | `lse_stride` |
| 1 `TOKEN_MAJOR` | `(b * lse_stride + s) * lse_pitch + h` | head            | `lse_pitch`  |

`lse_stride` keeps its meaning — tokens per row-group — and continues to come
from `lse.stride(0)` on the host (§4.2). `lse_pitch` is one new scalar, the
head-axis pitch, which is `>= num_head_q` so that **padding for alignment is
free in either layout**: head-major pads via `lse_stride`, token-major via
`lse_pitch`. That is why the field is two bits and not one — a padded or
blocked arrangement that neither formula covers has somewhere to go without
another ABI change.

`LSE_LAYOUT == 0` is the default and today's behaviour, so `VarlenBits == 0`
remains the conventional dense case.

---

## 4. What changes in the kernel: six scalars

After the prologue, everything is one of:

|     | `seqlen_q` | `seqlen_k` | `q_row_off` | `k_row_off` | `batch_index` | `lse_stride` |
| --- | ---------- | ---------- | ----------- | ----------- | ------------- | ------------ |

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

### 4.1 Orthogonal with MQA/GQA, and not by accident

Checked rather than assumed, because the two features touch the same address
expression:

```
base = batch_index * s_batch  +  head * s_head  +  row_off * s_seq
         \________ varlen ________/   \_ GQA _/   \___ varlen ___/
```

They act on **three different axes of a rank-4 tensor**. Varlen decides which
batch slice and which row; GQA decides which head, via
`head_k = head_q // (num_head_q // num_head_k)` — pure head arithmetic that
reads no sequence or batch quantity. LSE is indexed by `num_head_q` and the Q
side's decode, so it inherits both without a special case.

**The coupling that would have broken it is already closed.** For a 1THD
tensor, `stride(1)` — the per-token stride — is `H * D`, and under GQA that is
`num_head_q * D` for Q but `num_head_k * D` for K. So `q_row_off` and
`k_row_off` are scaled by *different* multipliers. That works only because
strides are per-tensor, which plan1 §2.1 already established for exactly this
reason ("K and V ... carry Num_head_k rather than Num_head_q, so their head
stride differs from Q's by construction"). A kernel still deriving one shared
`STRIDE_TOKEN` from the shape would be silently wrong for every packed GQA
call — the tokens of K would be strided as if K had Q's head count.

So the orthogonality is real, but it is inherited from the per-tensor stride
decision rather than free. Worth one test rather than none: §7's suite B
carries a GQA row.

### 4.2 `lse_stride` needs no derivation

AOTriton loads it from the device (`tl.load(cu_seqlens_q + Num_seqlens)`)
because its API only receives `Max_seqlen_q`. Ours already reads
`lse.stride(0)` off the LSE tensor on the host, which is the same number by
construction for every layout in §2 — `TotalS` for `(H, TotalS)`,
`Max_seqlen_q` for `(B*H, Max_seqlen_q)`. Keep that: it removes a scalar load
from the prologue and a host/device round trip from the caller.

What it does *not* supply is the head-axis pitch, which `TOKEN_MAJOR` needs:
`lse_pitch` is a second scalar, read from `lse.stride(0)` of a token-major
tensor while `lse_stride` then comes from its logical token count. The host
resolves which is which from `LSE_LAYOUT`; the kernel does not.

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

### Two suites, not one product

The full product — lengths × modes × causal × GQA — is far too large. It
factors cleanly because the two interesting axes are independent (§1): the
*mode* decides how a sequence is located, the *lengths* decide what is in it.
So project onto each axis in turn.

**Suite A — one mode, many length patterns.** `CompactVarlen` (`0x0B0B`) only,
since it is the mode every caller uses, against a spread of length sets:

| axis                           | values                                                     |
| ------------------------------ | ---------------------------------------------------------- |
| `N`                            | 1, 2, 7, 22                                                |
| length distribution            | uniform; one-long-many-short; all-equal; all-ragged primes |
| zero-length                    | leading, middle, trailing                                  |
| `seqlen_k = 0`, `seqlen_q > 0` | every row dead                                             |
| causal                         | off, top-left, bottom-right, explicit window               |

**Suite B — one length pattern, every mode.** A single deliberately awkward
length set — ragged, `N = 7`, including one zero and one much-longer sequence,
`seqlen_q != seqlen_k` — run through every configuration:

| configuration                         | bits                                    |
| ------------------------------------- | --------------------------------------- |
| dense                                 | `0x0000`                                |
| compact                               | `0x0B0B`                                |
| strided (gaps between sequences)      | `0x1313`                                |
| padded                                | `0x0202`                                |
| packed Q, `seqused_k` on packed KV    | `0x150B`                                |
| packed Q, `seqused_k` on a BHSD cache | `0x040B`                                |
| mixed: packed Q, dense K              | `0x000B`                                |
| GQA (§4.1), on compact                | `0x0B0B`, `num_head_q = 4 * num_head_k` |

**What the factorisation gives up, and why that is acceptable.** It cannot see
an interaction that needs *both* an unusual length pattern and an unusual mode
— a zero-length sequence in the middle of a `seqused_k` batch, say. That risk
is bounded by §1: the mode is consumed entirely in the prologue, which turns it
into three scalars, and every length pattern then flows through identical code.
An interaction would have to be a prologue bug, and suite B's length set is
chosen to be awkward enough to expose one. If a mode-specific bug is ever found
in the field, that assumption is what failed, and the fix is to cross those two
rows — not to build the full product now.

### Properties neither suite covers

1. **Single-sequence reduction.** `N = 1` compact must be bitwise identical to
   the dense call. Cheapest instance of the headline gate; make it pass first.
2. **Position array actually read.** A strided case with non-zero gaps must
   differ from the gapless reading of the same buffer — otherwise an
   implementation that ignores `seqinfo_?1` passes everything above.
3. **`seqused_k` shortens, and only K.** With
   `seqused_k[z] < cu_k[z+1] - cu_k[z]`, the result must equal a dense call on
   the truncated K *and* differ from one on the full K. Two assertions, because
   only the second one fails if `seqused_k` is ignored.
4. **One large case**, `T` big enough that `q_row_off * stride` exceeds 2^32,
   marked `large_shape`. §5 is unreachable at normal test scale.

---

## 8. Steps

The steps implement *axis values*, not named modes — the decomposition paying
off again, since each named mode then appears the moment its combination
becomes reachable.

### Step 1 — bits, decoder, and everything that reads no position array

`VarlenBits`, `seqinfo_?0`, `lse_pitch`, the §3.1 decoder, the six scalars, the
empty-work path, and both LSE layouts. Implement `LENGTH ∈ {MAX, CUMULATIVE}`
and `POSITION ∈ {IMPLIED, REUSE}` with both `STACKED` values.

That is three named modes at once — **dense `0x0000`, compact `0x0B0B`, padded
`0x0202`** — because all three are combinations of those axis values.
`seqinfo_?1` is not passed by any of them, so the second pointer pair need not
exist yet.

**Gate:** §7 property 1, then the headline gate for compact and padded. Dense
must be **bit-identical to today** — it will be if the decoder folds at
`bits == 0`.

### Step 2 — `POSITION == ARRAY`

Adds `seqinfo_?1` and one decoder case, which is strided varlen (`0x1313`).
Small, but *not free* as an earlier draft claimed: `REUSE` and `ARRAY` are
different code, and this is the step that writes the second one.

**Gate:** §7 property 2 — a strided case whose gaps differ per sequence. This
step introduces the only path that reads `seqinfo_?1`, so its coverage is
exactly that test plus step 3's.

### Step 3 — `LENGTH == INDIVIDUAL`

One decoder case, giving the PyTorch `seqused_k` pair: `0x150B` on packed KV
and `0x040B` on a rectangular cache. `0x150B` needs `ARRAY` and so depends on
step 2; `0x040B` does not.

**Gate:** §7 property 3.

### Step 4 — the window sentinels (P6 step 4)
`parse_window` moves into the kernel: `0x80000001 → (seqlen_q, 0)` and
`0x80000002 → (seqlen_q, seqlen_k - seqlen_q)`, per sequence. The host-side
`_CAUSAL_WINDOW` stays for the dense path — it costs nothing there and keeps
the sentinel off the hot path — but the two must agree, which §7 tests.

---

## 9. Coverage: what these bits cannot say

Asked for explicitly, so stated explicitly. The encoding spans
`2 × 3 × 3 = 18` configurations per side, of which **`REUSE` with a
non-`CUMULATIVE` length is invalid** — `seqinfo_?0[z]` is not a position there
— leaving 14 valid per side and 196 pairs. That constraint is host-validated
(V4). Every one is either
meaningful or harmless — `BATCHED + ARRAY`, for instance, means "slice `z`,
then skip `b[z]` rows", which is unusual but well-defined and free. What falls
*outside*:

1. **Paged / block-table KV.** `torch.nn.attention.varlen` accepts a
   `block_table` of shape `(N, max_pages_per_seq)` with K/V shaped
   `(total_pages, page_size, H, D)`. A sequence is then a *list* of physical
   pages, so position is per-page, not per-sequence, and no scalar `row_off`
   can express it. This needs an indirection load inside the KV loop — a real
   feature, not an encoding gap. Byte 3 (bits 31:24) is reserved for it.
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

## 10. Resolved

All eight settled; recorded with the reasoning, not just the verdict.

### [V1 — RESOLVED] `VarlenBits` is a runtime argument

Not a build axis: the combinations cannot be afforded. The tuning key stays
`BLOCK_DMODEL` alone (N3), and a build axis would multiply the functional count
by the number of shipped configurations — of which there are already seven in
§7's suite B, before bias and dropout add theirs.

The decode is ~10 scalar ops and at most six scalar loads, once per workgroup,
entirely in SGPRs; D2 priced runtime scalars at ~0.2%. If `bits == 0` ever
measures badly it can be promoted to a *specialisation* later without an ABI
change.

### [V2 — RESOLVED] `LSE_LAYOUT`, two bits, in byte 2

The concern was real and my framing of it was not: I had called the LSE layout
"derived". **PyTorch specifies a shape and says nothing about memory, but
Transformer Engine — an AOTriton backend — requires the `(T_q, H_q)`
layout.** So it is a live requirement, not a hypothetical.

Two bits rather than one, because efficient-attention implementations pad the
LSE layout for alignment and the field should have room for arrangements
neither current formula covers. Padding itself needs no new codes — §3.2's
`lse_stride` and `lse_pitch` absorb it in either layout.

### [V3 — RESOLVED] Ship Q-side `INDIVIDUAL`

PyTorch exposes only `seqused_k`; others may want the Q side. Symmetry is free
— the decoder is one function called twice — whereas rejecting it would cost a
validation branch and an asymmetry to explain.

### [V4 — RESOLVED] Host assertions as *documentation*

The §9.4 and §9.5 assumptions — `seqinfo_?1` is a prefix sum with its total in
slot `[N]`, and sequences do not overlap — are written as documented
preconditions on the launch shim, with the assertion spelled out in the
docstring so a reader can see exactly what is assumed.

Not enforced on every call: checking them means reading the `seqinfo` tensors
back to the host, which is a device sync on the hot path. The document is the
contract; the check is available to anyone debugging.

### [V5 — RESOLVED] `REUSE` dissolves the hazard

The concern was that §1.2's original formulation rested on an invariant nothing
enforced — every *position* read going to `seqinfo_?1`, every *length* read to
`seqinfo_?0` — while compact varlen passed the **same pointer** as both, making
the two roles indistinguishable in the configuration everyone runs. A later
shortcut taking a length from `?1` (say `?1[z+1] - ?1[z]`, which looks natural)
would have passed every compact test and silently broken strided, where that
difference is the *padded* extent rather than the real length.

**`REUSE` removes it.** Compact no longer passes `seqinfo_?1` at all, so the
same shortcut faults on a null pointer instead of quietly returning a wrong
answer — silent-and-common becomes loud-and-immediate.

What remains is narrower and honest: `seqinfo_?1` is read only by strided
(`0x1313`) and by `seqused_k` on packed KV (`0x150B`), so that path's entire
coverage is §7 properties 2 and 3. Both must exist before either configuration
is claimed, and property 2's gaps must vary *per sequence* so a uniform-gap
implementation cannot pass by accident.

Noted alongside: AOTriton's strided path is deployed but incompatible with
`torch.nn.attention.varlen`, which supplies `cu_seq_k` rather than
`cu_seqlens_padded`, so a **shim cumsum kernel** builds the position array.
That cumsum belongs on the host side of our shim too — the kernel takes
whatever array it is handed and cannot tell which it is, which is the point.

### [V6 — RESOLVED] `N` reuses `batch_size`

As AOTriton does (`.Batch = num_seqlens == 0 ? batch : num_seqlens`). The grid's
z extent is the sequence count under both readings, and a second argument could
only ever disagree with the first.

### [V7 — RESOLVED] `Max_seqlen_q/k` stay unconditional

Kept as plain arguments even where a mode never reads them, so the shim may
pass anything in those cases. Precisely, the kernel reads them only when:

| argument       | read when                                                |
| -------------- | -------------------------------------------------------- |
| `Max_seqlen_q` | Q `LENGTH == MAX`, or Q `POSITION == IMPLIED && STACKED` |
| `Max_seqlen_k` | K `LENGTH == MAX`, or K `POSITION == IMPLIED && STACKED` |

The host needs `Max_seqlen_q` unconditionally regardless, to size the grid.

### [V8 — RESOLVED] Collect the grid-waste measurement here

No harm, and persistent-dynamic then starts from a number rather than an
intuition. ~20 lines against a skewed length distribution, reported as the
idle-workgroup fraction `1 - mean(seqlen)/max(seqlen)` alongside measured
throughput.

---

## 11. What this phase does *not* do

- **Paged KV** (§9.1) and **persistent-dynamic** (§6.2).
- **Backward.** The dk/dv kernel needs the same six scalars and the sentinels
  transposed (gSWA plan §6). Name the decoder so it can be lifted, not copied.
- **P4 bias / P5 dropout**, both of which gain a varlen dimension once this
  lands: bias is indexed per sequence, and Philox offsets must be per-sequence
  to stay reproducible. Noted now, scoped later.
