# Handoff: bring the gfx950 kernels onto the shared kernarg ABI

For whoever picks this up on `personal/xinyazhang/sdpa-gfx950-feature-bwd`, on
real gfx950 hardware. Written from the gfx1201 side, which has just finished the
same exercise; **nothing here was executed on gfx950** — every claim below is
read off the branch's source at `07b7271`, and every one is checkable in a
minute. Where I could not check something from source, it says so.

## Why

Three consumers now have to agree on one argument order: the gfx950 kernels,
the gfx1201 kernels, and `flyati/modules/flash/aot/flyc_attn_fwd.py`, the ATI
description that dispatches an hsaco directly and therefore hardcodes the
kernarg layout. `sdpa-bwd-contract-gfx950.md` §7 already says argument order
*is* the ABI; this is that rule applied across the arch boundary.

gfx1201 moved to gfx950's convention, not the other way round — gfx950 had the
written contract, so it won. The four changes below are the residue: places
where gfx950 differs from what all three now agree on.

## The agreed block

```
Q, K, V, B, <outputs>, LSE, Delta?, <varlen>, max_seqlen_*, window pair,
philox, num_head_q, num_head_k, hdim_qk, hdim_vo, sm_scale, <strides>
```

`Q, K, V, B` is common to every kernel in both families. `<outputs>` is the
only part that varies:

| kernel | `<outputs>` |
|---|---|
| forward | `O, LSE` (LSE is written, so it sits in the group) |
| dQ | `DQ, DB` |
| dK/dV | `DK, DV` |
| fused (gfx1201 only) | `DK, DV, DQ` |

Strides run in tensor order and the bias pair closes them: `... stride_b_*`,
plus `stride_db_*` for a kernel that writes dB.

## What to change

### 1. `Bias` -> `B` — all three gfx950 kernels

`flash_attn_func_gfx950.py`, `fmha_bwd_dq_gfx950.py`,
`fmha_bwd_dkdv_gfx950.py`. A rename, but check for a bare `B` first: on
gfx1201 every pre-existing `B` was MFMA/WMMA operand notation inside comments,
so it was safe, and it is worth confirming rather than assuming.

This one is a net win beyond consistency: `flyc_attn_fwd.py:166` already
declares `wires_to='B'`, so AOTriton's operand is *already* called `B` and the
rename makes that mapping an identity.

### 2. Forward: `B` moves ahead of the outputs

```
now:     Q, K, V, O, LSE, Bias, Workspace, BlockTable, ...
target:  Q, K, V, B, O, LSE, Workspace, BlockTable, ...
```

The backward kernels already have `Q, K, V, Bias, DO, ...` and need only the
rename from item 1, not a move.

### 3. Forward: drop `batch_size` from the kernarg

It is declared in `flash_attn_func_gfx950.py`'s `@flyc.kernel` signature and
**never read in the kernel body** — I checked the body specifically. Its one
job is sizing the grid's third axis, which happens in the launcher, and the
kernel recovers its plane from `block_idx.z`.

Both gfx950 backward kernels already omit it, so this is the forward catching
up with its own family.

Two notes from doing this on gfx1201:

- `@flyc.kernel` and its `@flyc.jit` launcher do **not** need identical
  argument lists. That belief is what kept the dead kernarg alive on our side
  for months; the aotriton side does not parse the jit signature for order.
  Keep it in the launcher, which still uses it.
- If gfx950 has an equivalent of `test_signature_parity_gfx1201.py`, add
  `batch_size` to that module's launcher-only allowlist in the same commit, or
  the test fails on a change that is correct.

### 4. `fmha_bwd_dq_gfx950.py`: give B its own strides, when B6 lands

Not a rename — a gap that only opens when the bias input is wired.

Today dQ takes `Bias` but passes `bias_strides=(0, 0, 0)` (line 756), with the
slot reserved and unread, exactly as `dq950.py:1123` says: *"Bias: no build
here reads it; the slot is held for B6."* That is fine while nothing reads it.
The moment it does, it needs `stride_b_batch/_head/_seq_q` of its own — it
currently has only `stride_db_*`.

**Do not reuse `stride_db_*` for the B read.** That is precisely the bug we
shipped and then fixed on gfx1201: B and DB are different tensors and nothing
obliges `dbias` to be laid out like the bias it is the gradient of. It is
correct whenever both are freshly allocated and contiguous, so every test
passes, and it writes to wrong addresses the first time a caller hands in a
view. The regression test that catches it is worth copying —
`test_dbias_with_a_different_layout_from_bias` in
`test_fmha_bwd_dq_gfx1201.py`: `dbias` is a column slice of a twice-as-wide
buffer, so its last axis is still contiguous as the ABI requires but its row
pitch differs, and the test asserts both the gradient *and* that nothing landed
in the columns outside the slice.

dK/dV already has `stride_b_*` and no DB, which is the right shape.

## What needs no change

`LSE`, `Delta`, `num_head_q`, `num_head_k`, `sm_scale`, and
`stride_b_batch/_head/_seq_q` are all already what the three consumers agreed
on — in several cases *because* gfx950 spelled them that way first and gfx1201
moved. `sm_scale` already sits ahead of the stride block in all three gfx950
kernels.

One name was contested and gfx950 keeps it: `num_head_q`, not `num_head`.
`flyc_attn_fwd.py:231` wires to `Num_head_q`, so `num_head_q` is the 1:1
mapping and a rename would move away from AOTriton, not toward it. gfx1201
briefly renamed it and reverted.

> **The reasoning in this paragraph is wrong; the conclusion still stands.**
> `wires_to='Num_head_q'` names the attribute ATI reads off its *context
> object*, not the kernel argument it feeds, so it says nothing about what the
> kernarg should be called. See the outcome section: gfx950 keeps `num_head_q`
> as a decision, and gfx1201 currently spells it `num_head`.

## Two traps, both of which cost us a cycle

**Moving a kernarg is two edits, not one.** The signature and the host tuple
that feeds `run_compiled` must move together. An off-by-one-slot ABI does not
fail a build — it faults the GPU on a pointer that used to be a stride, and a
launcher/kernel parity test cannot see it either, because it compares the
kernel to its launcher and both are wrong in the same way. Only a numerical run
catches it. Budget a correctness check per kernel, not per batch of kernels.

**Watch for `# noqa` comments on the lines you are matching.** The gfx1201
forward's `O:` kernarg carries a `# noqa: E741`, which made an exact-text
replace of the pointer block match the forwarding call and *neither* signature,
leaving the file half-renamed. F821 caught it; nothing semantic would have.

## Suggested order

1. Item 1 alone (rename), all three kernels, run the suites. Pure rename, so
   any failure is a missed occurrence.
2. Items 2 and 3 together (forward reorder + dead kernarg), then the forward
   suite — `test_flash_attn_func_gfx950.py` alone collects 329 per the contract
   doc, and it is the one that matters since everything subclasses its helpers.
3. Item 4 whenever B6/B7 wires the bias input; it is not blocking today.

## Verifying agreement afterwards

This prints the four gfx1201 signatures side by side and is the check I used;
point it at the gfx950 files and the heads should match through `Q K V B`, with
`<outputs>` differing per kernel:

```python
import ast, pathlib
for f, k in (("flash_attn_func_gfx950.py", "<kernel fn name>"), ...):
    t = ast.parse(pathlib.Path(f).read_text())
    for n in ast.walk(t):
        if isinstance(n, ast.FunctionDef) and n.name == k:
            a = [x.arg for x in n.args.args]
            print(k, a[:a.index("seqinfo_q0")])
```

gfx1201's, for comparison:

```
flash_attn_func_aiw_kernel   Q K V B O LSE
bwd_dq_kernel                Q K V B DO DQ DB LSE Delta
bwd_dkdv_kernel              Q K V B DO DK DV LSE Delta
bwd_fuse_kernel              Q K V B DO DK DV DQ LSE Delta
```

---

# Outcome, on gfx950 hardware

Done. All three gfx950 kernels now carry the shared kernarg block, verified by
comparing ASTs against the gfx1201 branch rather than against this document —
**which was stale in three places by the time it was picked up.** The branch
had moved on (`personal/xinyazhang/sdpa-gfx1201-feature`, through `1b58fb93`),
so the kernels were taken as the source of truth and the doc as commentary.
That is the right order in general: a handoff note describes a moving target.

Where it was stale, for the next person:

- **Item 3 (drop `batch_size` from the forward kernarg) was already done.** The
  gfx950 forward kernel had no `batch_size` when this was picked up; it lives in
  the launcher only, which is what the item asked for.
- **Item 4 (give dQ its own bias strides) was done by B7.** dQ carries
  `stride_b_batch/_head/_seq_q` *and* `stride_db_batch/_head/_seq_q` as
  separate arguments, which is the shape the item asks for and the bug it warns
  about is not present. The gfx1201 regression test it recommends copying is
  still worth copying and is **not** yet ported.
- **`num_head_q` is the one place gfx950 does *not* follow gfx1201, and the
  argument both sides used was void.** This document argued for `num_head_q`
  from `flyc_attn_fwd.py:231` wiring to `Num_head_q`; `1b58fb93` argued for
  `num_head` from AOTriton's `num_head`/`num_head_k` pair and renamed all four
  gfx1201 kernels. **`wires_to='Num_head_q'` names the attribute ATI reads off
  its *context object* — the input side — not the kernel argument it feeds.**
  So that line constrains neither spelling, and the "1:1 mapping" reasoning in
  the section above is wrong even though its conclusion is the one adopted.
  gfx950 keeps `num_head_q` by decision, not by inference.

  The consequence to state plainly: gfx950 and gfx1201 **disagree** on this one
  kernarg name until gfx1201 reverts `1b58fb93`'s rename. Nothing breaks —
  kernarg names are not part of the binary layout, and the slot is in the same
  position in both families — but the two signatures are no longer textually
  identical, and anything comparing them by name needs to know that.

What changed on gfx950: `Bias` -> `B` in all three kernels, and `B` moved ahead
of `O` in the forward. `num_head_q` is unchanged, and so is the context keyword
`num_head_q=` — that attribute lives in the shared `flash_attn_utils.py`
context and is not part of this ABI in any case.

## The three gfx950-only kernargs are gone, not nulled

`Workspace`, `BlockTable` and `block_table_stride` — split-K and paged, neither
of which exists on gfx1201 — **leave the kernarg entirely in builds that do not
use them.** The forward's kernarg segment goes 636 -> 536 bytes for the same
configuration, so the ATI-dispatched build now matches
`flash_attn_func_aiw_kernel` argument for argument.

Baking them as null pointers would *not* have been enough, and the reason is
worth stating because it is the same reason the whole exercise exists: the two
pointers sat **between `LSE` and `seqinfo_q0`**, so an occupied-but-null slot
still shifts every later argument two positions away from gfx1201's layout. A
consumer that hardcodes the block reads the wrong slots whatever the pointers
contain. Position is the ABI; content is not.

The mechanism is a `Constexpr` annotation.
`compiler/kernel_function.py` sorts constexpr-annotated parameters into
`constexpr_values` and never adds them to `kernel_arg_types`, so the parameter
is folded at trace time and no slot is emitted. Two properties make it a
*per-build* choice: the `def` runs inside the builder on every build, and this
module has no `from __future__ import annotations`, so the annotation
expression is evaluated there rather than kept as a string. Hence
`_WS_ANN = fx.Tensor if _WS_RUNTIME else fx.Constexpr`, with `_WS_RUNTIME =
SPLITK or DUALWAVE_SWP_DEBUG_LAZY_COUNTS` and `_BT_RUNTIME = PAGED`.

Two things to know before copying this:

- **The stand-in value must be an `int`, not `None`.** A constexpr value goes
  into the JIT cache key through `Constexpr.value_signature`, which accepts
  int/bool/float/str/tuple/lambda and *raises* on anything else. `0` is passed
  and never read; every consumer is behind a `const_expr` guard.
- **The launcher takes the same annotations as the kernel.** Doing it that way
  means the folded value arrives as a Python constant and forwards to the
  kernel unchanged, so there is no build-time conditional inside traced code at
  all — the alternative was a ternary in a `@flyc.jit` body, which the AST
  rewriter would have had to be trusted to leave alone.

Coverage, stated honestly: the split-K tests build with `_WS_RUNTIME` true and
pass, so both `Workspace` branches are exercised. **No paged forward kernel is
built anywhere in the parity suite, before or after this change** — the paged
coverage in `test_traits_constructor_matches_production_exactly` is traits-only.
The `_BT_RUNTIME` true branch is therefore unexercised, and it is a no-op by
construction: when `PAGED` is on the annotation is `fx.Tensor`, textually what
it was.

Both traps in this document are real and both were hit in the reading rather
than the writing: the identical `Q, K, V, O, LSE, Bias, ...` block appears in
the forward's `@flyc.kernel` **and** its `@flyc.jit` launcher, so every
exact-text edit matched twice and had to be done by line number with an
assertion on the current text. The `# noqa: E741` on `O:` is still there.

Verification: line-number edits with asserted preconditions, an AST comparison
against the gfx1201 branch, `ruff` for F821, then the numerical suites — the
document is right that nothing else catches an off-by-one slot.
