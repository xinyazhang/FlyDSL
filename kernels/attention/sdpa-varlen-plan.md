# P3 in detail: Variable-Length Sequences

Companion to `sdpa-close-gap-plan1.md`, which resolves the naming (D3) and the
LSE layout but not the mechanism. This is the implementation plan.

Written after P6, and shaped by it: the gSWA phase established that a *bitwise*
equivalence to an existing path is both achievable and the sharpest oracle in
this codebase. Varlen admits the same kind of gate, for the same reason, and
§6 makes it the headline requirement rather than an afterthought.

---

## 0. The goal, including the one that is easy to leave unstated

1. **Support variable-length sequences** — plural layouts, see §1.
2. **Absorb P6 step 4.** `Window_left`/`Window_right` must accept the
   sentinels `0x80000001` (top-left) and `0x80000002` (bottom-right), resolved
   *per sequence*. gSWA deferred this here because a sentinel is meaningless
   without per-sequence lengths. The phase is not done until the host-side
   `_CAUSAL_WINDOW` table has an in-kernel counterpart.
3. **Do not add an `if varlen_mode` to the body.** The mode may branch exactly
   once, in the prologue, to produce six scalars. Everything downstream reads
   those scalars and cannot tell which mode it is in.

Objective 3 is a design constraint with teeth, and it is the reason this phase
is small. It is also what AOTriton does: its prologue is a three-way branch and
its body has no varlen conditionals at all.

**Success criterion for objective 3:** outside the prologue, the kernel
contains no reference to `varlen_mode`, `cu_seqlens`, or `seq_strides`.

---

## 1. There are four modes, not two

This is the part most likely to be under-scoped. `varlen` is not one feature;
AOTriton ships four `VarlenType` values, and two of them exist specifically for
Transformer Engine (added in `04cdead5`, "Support Additional Varlen Memory
Layouts"):

| `VarlenType` | tensor shape | lengths from | LSE layout |
|---|---|---|---|
| `None` = 0 | BHSD | `Max_seqlen_q/k` | `(B*H, Max_seqlen_q)` |
| `CompactVarlen` = 1 | THD (packed, no gaps) | `cu_seqlens_q/k` | `(H, TotalS)` |
| `PaddedVarlen` = 2 | BHSD | `cu_seqlens_q/k` | `(B*H, Max_seqlen_q)` |
| `StridedVarlen` = 3 | THD with gaps | `cu_seqlens_q/k` for *length*, `seq_strides_q/k` for *position* | `(H, TotalS_padded)` |

The two TE modes are the interesting ones because they separate two things
that coincide in classical varlen:

- **`PaddedVarlen`** keeps the rank-4 layout and only shortens the sequences.
  Positions are unchanged; just the lengths come from `cu_seqlens`.
- **`StridedVarlen`** packs sequences but leaves padding *between* them, so the
  position of sequence `z` is no longer `cu_seqlens_q[z]`. TE calls the
  position array `cu_seqlens_padded`.

So "where sequence `z` starts" and "how long it is" are independent, and the
prologue must treat them as two separate lookups. Any design that assumes
`start[z+1] - start[z] == length[z]` handles compact varlen and silently
corrupts strided.

---

## 2. What actually changes: six scalars, computed once

Every mode reduces to the same six values, after which the kernel is identical:

| | `seqlen_q` | `seqlen_k` | `q_row_off` | `k_row_off` | `batch_index` | `lse_stride` |
|---|---|---|---|---|---|---|
| None | `Max_seqlen_q` | `Max_seqlen_k` | 0 | 0 | `z` | `Max_seqlen_q` |
| Compact | `cu_q[z+1]-cu_q[z]` | `cu_k[z+1]-cu_k[z]` | `cu_q[z]` | `cu_k[z]` | **0** | `TotalS` |
| Padded | `cu_q[z+1]-cu_q[z]` | `cu_k[z+1]-cu_k[z]` | **0** | **0** | `z` | `Max_seqlen_q` |
| Strided | `cu_q[z+1]-cu_q[z]` | `cu_k[z+1]-cu_k[z]` | `sst_q[z]` | `sst_k[z]` | **0** | `sst_q[N]` |

`batch_index = 0` for the packed modes is the whole trick: a packed tensor is
one batch whose sequence axis is `T`, so the batch stride must not be applied
and the row offset does the work instead.

### 2.1 The kernel is already shaped for this

Both places that would have to change are already written in the right form,
which is worth stating because it bounds the work.

**Addressing.** `_addr_pair` computes

```python
bh = batch_idx * s_batch + head * s_head
tbase = bh + tile_start * s_seq
```

and AOTriton's is

```python
o_base = Out + batch_index * stride_oz + off_h_q * stride_oh
             + cu_seqlens_q_start * stride_om
```

The only difference is the third term. Adding `q_row_off * s_seq` into `bh`
covers all four modes, for all four tensors, with no other change — and it goes
into the **64-bit base**, not the 32-bit offset, which §4 explains is required.

**LSE.** The kernel computes

```python
_lse_base = (batch_idx * num_head_q + head_q) * lse_stride
```

against AOTriton's `_lse_offset(b, h, s, H, S) = (b*H + h)*S + s`. Same
expression with `s = 0`. Adding `+ q_row_off` completes it. Plan1's LSE
decision already called for exactly this ("the mode selects the *inputs*"), so
this is that decision being cashed in.

### 2.2 What the prologue costs

`off_z` is uniform across the workgroup, so `cu_seqlens_q[off_z]` and its three
siblings are **scalar** loads, four to six of them, once per workgroup. They
land in SGPRs and do not touch the VGPR budget that gSWA just pushed up.

The prologue's branch is a genuine branch, but on a uniform condition and
outside every loop.

---

## 3. The grid grows, and a workgroup must be able to do nothing

Today the grid's Q axis is `ceil(seqlen_q / BLOCK_M)`, which is exact. Under
varlen there is no single `seqlen_q`, so it becomes
`ceil(Max_seqlen_q / BLOCK_M)` and **every sequence gets the longest
sequence's worth of workgroups.** Sequences shorter than the maximum therefore
dispatch workgroups with nothing to do.

Two consequences:

### 3.1 An early exit, without an early `return`

`CLAUDE.md` forbids branch-local `return`/`yield` in traced functions, and the
kernel is one long single-exit trace. So `if start_M >= seqlen_q: return` is
not available.

The structural equivalent is to make the workgroup's *work* empty rather than
skip it:

- Force the KV loop counts to zero. Under gSWA this is already the natural
  representation — `n_l = n_f = n_r = 0` — so causal needs nothing new. The
  non-causal path needs its `kv_upper` clamped the same way.
- Rely on the existing row-bound masking for the O and LSE stores, which
  already suppresses rows past `seqlen_q`. **Verify this covers the LSE store
  too**, not just O; it is a separate store with its own guard.

The workgroup still launches, loads Q (wastefully), and runs a no-op epilogue.
That is the price of a one-size-fits-all grid, and it is bounded.

### 3.2 This is the case for persistent-dynamic, and it should be recorded

plan1 §2.4 defers persistent-dynamic and notes AOTriton forces
`PERSISTENT_TYPE = 2` for causal. Varlen makes the argument sharper: with a
skewed length distribution the wasted fraction is
`1 - mean(seqlen) / max(seqlen)`, which is large for realistic batches. AOTriton
excludes varlen from persistent (`unsupported_by_persistent = Num_seqlens != 0`),
so it currently eats the same cost.

Out of scope here. But **measure the waste** as part of this phase so the
persistent-dynamic task has a number to beat, rather than an intuition.

---

## 4. Row offsets are 64-bit, and that is not the same trap as gSWA's

gSWA's arithmetic hazard was *signedness*. Varlen's is *width*, and it points
the other way.

A packed tensor's `T` axis is the sum over the batch, so `q_row_off * s_seq`
is routinely far larger than any single sequence's extent — at `T = 128K`,
`num_heads = 32`, `head_dim = 128`, the byte offset exceeds 2^32 comfortably.
The kernel's addressing is deliberately a **64-bit base plus a 32-bit
offset** (plan1; the split is what made the ISA diverge from the pre-unification
kernels). The rule that falls out:

1. **`q_row_off` / `k_row_off` are added to the 64-bit base**, never to the
   32-bit per-lane offset. §2.1's `bh` is the correct insertion point; the
   `toff` path must not see them.
2. **`cu_seqlens` values are `int32` on the wire** (they are, in both PyTorch
   and TE) but must be widened before multiplying by a stride.
3. Sequence *lengths* stay `int32` and stay signed — everything §2.4 of the
   gSWA plan says about window arithmetic applies unchanged, because
   `window_right = seqlen_k - seqlen_q` is still computed from them.

An offset that overflows produces a wrong address rather than a fault, so it
reads plausible data from the same allocation. Like the gSWA prefetch bug, it
survives any test that only looks at small shapes — see §6.

---

## 5. Steps

Each lands independently and leaves the tree green.

### Step 1 — the prologue, and `CompactVarlen`

Introduce `seq_info_q` / `seq_info_k` (pointers) and `varlen_mode` (runtime
`i32`), and the six-scalar prologue. Implement modes `None` and
`CompactVarlen` only; the other two return from the same table.

- Grid Q axis keys on `Max_seqlen_q`.
- The empty-work path of §3.1.
- LSE `(H, TotalS)`.

**Gate:** §6's bitwise equivalence for compact varlen, plus the whole existing
suite unchanged — mode `None` must be bit-identical to today, which it will be
if the prologue folds correctly.

### Step 2 — `PaddedVarlen`

Two table entries (`q_row_off = 0`, `batch_index = z`). Should be a handful of
lines if step 1's prologue is factored right; if it is not, that is the signal
to refactor before continuing.

### Step 3 — `StridedVarlen`

Adds `seq_strides_q/k` and decouples position from length (§1). The mode most
likely to expose an assumption baked in during step 1.

### Step 4 — the window sentinels (P6 step 4)

`parse_window` moves into the kernel: `Window_left == 0x80000001` resolves to
`(seqlen_q, 0)` and `0x80000002` to `(seqlen_q, seqlen_k - seqlen_q)`, per
sequence. AOTriton's is 20 lines and ours is the same shape.

The host-side `_CAUSAL_WINDOW` table stays for the non-varlen path — it costs
nothing there and keeps the sentinel off the hot path — but the two must
agree, which §6 tests directly.

---

## 6. Test matrix, and the oracle that makes it cheap

**The headline gate: varlen with `B` sequences must equal `B` separate
non-varlen calls, bitwise.**

This holds for a reason, not by luck. A varlen workgroup and its non-varlen
counterpart compute over the same tiles, in the same order, with the same
values; only the base address differs. Every floating-point operation is
identical, so the results must be too. It is the same argument that made the
gSWA gate work, and it has the same payoff: **an addressing bug that lands
inside the right allocation still shows up**, because the data it reads is
different data.

A tolerance-based comparison against a reference would miss exactly the bugs
this phase is most likely to produce.

Two caveats to state in the test, so a future reader does not weaken it:

- The grid differs (varlen dispatches `ceil(Max_S/BLOCK_M)` per sequence), and
  under `_REVERSE_Q_TILES` the tile-to-workgroup mapping differs too. Neither
  changes what any surviving workgroup computes.
- LSE lives in a different layout, so compare per-sequence slices, not buffers.

### Axes

| axis | values | why |
|---|---|---|
| `varlen_type` | compact, padded, strided | §1 — three code paths through one prologue |
| length distribution | uniform; one-long-many-short; all-equal | all-equal is the degenerate case that hides position bugs |
| `n_seqlen` | 1, 2, 7, 22 | 1 must reduce to non-varlen exactly |
| `seqlen = 0` | leading, middle, trailing | an empty sequence must write nothing at all |
| `seqlen_k = 0`, `seqlen_q > 0` | | every row dead: `O = 0`, `LSE = +inf` |
| ragged lengths | prime values | tiles that end mid-sequence |
| causal | off, top-left, bottom-right, window | `window_right` is per-sequence now |
| sentinels | `0x80000001`, `0x80000002` | step 4; must match the host table |

### Three properties beyond "matches the reference"

1. **Single-sequence reduction.** `n_seqlen = 1` compact varlen must be
   bitwise identical to the non-varlen call. The cheapest possible instance of
   the headline gate, and the first thing to make pass.
2. **Padded vs compact agreement.** The same logical batch expressed both ways
   must give the same numbers. This is what catches a `batch_index` /
   `row_off` mix-up, since the two modes differ in *precisely* those two
   fields and nothing else.
3. **Strided is not compact.** Build a strided case whose gaps are non-zero
   and assert it differs from the compact interpretation of the same buffer —
   otherwise a strided implementation that ignores `seq_strides` passes
   everything.

### One shape that must be large

The 64-bit offset of §4 is unreachable at test scale. Include **one** case with
`T` large enough that `q_row_off * stride` exceeds 2^32, even if it must be
`num_heads = 1` and a short run, and mark it `large_shape`. A correctness suite
that tops out at a few thousand tokens cannot see the bug this guards.

---

## 7. Risks

| risk | why it matters | mitigation |
|---|---|---|
| **Scope read as one mode** | `varlen` sounds like one feature; it is four, two of them TE-specific | §1's table up front; steps 2 and 3 are separate landings |
| Position vs length conflated | compact makes `start[z+1]-start[z] == length[z]` true, strided makes it false | §1; test property 3 |
| 32-bit offset overflow | wrong address inside the same allocation, so it reads plausible data | §4 rule 1; the `large_shape` case |
| Early exit needs a `return` | not available in a single-exit trace | §3.1 — empty work, not skipped work |
| LSE store guard | O's row bound is well tested; LSE's is a separate store | assert it explicitly in the empty-sequence case |
| Grid waste on skewed batches | can dominate; invisible in a uniform-length test | measure it in this phase, fix it in persistent-dynamic |
| Tuning tables go stale | fifth time (LSE, P2a, gSWA, and the sweep still owed) | the sweep is already outstanding; do not re-sweep twice |

---

## 8. What this phase does *not* do

- **Persistent-dynamic**, though §3.2 argues varlen strengthens its case and
  asks for a number.
- **Backward.** The dk/dv kernel needs the same six scalars, and the window
  sentinels transposed (gSWA plan §6). Name the prologue so it can be lifted.
- **P4 bias / P5 dropout**, which both gain a varlen dimension once this lands
  — bias is indexed per sequence, and dropout's Philox offset must be
  per-sequence to stay reproducible. Note it now; neither is in scope here.
