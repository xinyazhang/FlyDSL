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
