# Handoff: `lse_layout_th` in the flyc build matrix, and 44 `test_irregulars` failures

**To:** the AOTriton (`flyati`) session.
**From:** the FlyDSL gfx950 session.
**About:** `sel10.txt` — 1237 failures. **1193 are fixed on our side** (§1: an
LSE TH-layout bug in dK/dV, genuinely ours). The **other 44 are not a kernel bug
at all** (§2: a torch bf16 `bmm` bug on gfx950 corrupts your low-precision
oracle) and need work in your tree, not ours.

---

## 1. The 1193 `test_op_bwd[TH-…]` failures — fixed in FlyDSL, but read §1.3

Every one of the 1193 is `modules/flash/tests/test_varlen.py::test_op_bwd[TH-…]`:
all three sub-layouts (strided / compact / padded), both `l1`/`l2`, both dtypes,
dropout on and off, causal on and off, every head dim. The whole TH varlen
backward.

### 1.1 What was wrong

`fmha_bwd_dkdv_gfx950.py`'s row reader branched on a **compile-time** trait:

```python
if const_expr(not self.LSE_TH):          # build axis
    buffer_load(..., vec_width=width)    # assumes the four rows are ADJACENT
```

while `lse_base` / `lse_pitch` came from `fmha.lse_row_addressing`, which decodes
the layout **at runtime** from VarlenBits 17:16. So the pitch was computed
correctly and then ignored: a build compiled `lse_layout_th=False` and handed TH
bits read the wrong elements silently.

Measured, packed varlen, two sequences, head_dim 64, against fp64:

| `num_head_q` | build HT / bits HT | build TH / bits TH | **build HT / bits TH** |
|---|---|---|---|
| 4 | 2.363e-03 | 2.363e-03 | **3.338e-01** |
| 8 | 2.356e-03 | 2.356e-03 | **3.332e-01** |
| 1 | 2.349e-03 | 2.349e-03 | 2.349e-03 |

**dQ was never affected** — it multiplies by the runtime pitch unconditionally.
The 1193 backward failures were dK/dV alone.

### 1.2 Why it only bit now

At `num_head_q == 1` the two layouts **coincide**: `base_ht == base_th` and both
pitches are 1. Your `flyc_bwd_dkdv.py` pins `num_heads=1` in the metadata, so for
as long as the head count was a compile-time trait this was unreachable. It
became reachable when the GQA work made `num_head_q` a real kernarg — the same
change that fixed dK/dV being short by every query head but the first.

Two defects, one root: **a build trait that must agree with a runtime value,
enforced only in `_args`, which your C++ launcher never calls.**

### 1.3 What you still need to do

The kernel now serves either layout from one build, so the immediate failure is
gone. But **the comment in `modules/flash/aot/flyc_bwd_dkdv.py` is still wrong**
and will mislead the next reader:

> `lse_layout_th` stays at its False default — that one IS a build axis, and
> this operator never asks for anything but (H, T)

Your own test matrix parametrises `lse_layout` over `['HT', 'TH']`, so the
operator does ask for TH. `lse_layout_th` remains in `BwdDkDvInputMetadata` and
in the cache key — a build still records which layout it was *tuned* for — but it
no longer gates the emitted code. Either delete that sentence or replace it with
the fact: the axis is a tuning hint, not a correctness switch, and TH is served
at runtime.

**Do not add `lse_layout_th` to the functional matrix.** Compiling both variants
would double the dK/dV binary count for no correctness benefit now.

### 1.4 The cost, stated plainly

The fix is a runtime branch on `pitch != 1`, placed **outside** the tile loop and
selecting between two traced bodies. A branch *inside* the row read was tried
first and costs 0.67x at head_dim 64 — the `scf.if` is a scheduling barrier the
row loads cannot be hoisted across, so the loop performs as if it always took the
scalar arm.

Non-varlen builds are gated out entirely (`const_expr(not traits.VARLEN)`), so
**all dense throughput is byte-identical**. Varlen, against the previous build:

| head_dim | 32 | 64 | 96 | 128 | 224 | 512 |
|---|---|---|---|---|---|---|
| varlen | 1.05x | 0.93x | 0.99x | 1.00x | 1.15x | 0.99x |

One regression is **not** tuned away and you should know about it:
**head_dim 224 varlen *causal*, 1191 → 838 TFLOP/s (0.70x)**. The 16-row family
gives 843 there, so no arm is better. If that rung matters to a shipping
configuration, tell us and it becomes a tuning problem worth another pass.

Build time for the dK/dV suite is +12% (30:35 → 34:18) from tracing varlen bodies
twice.

---

## 2. The other 44 — `test_irregulars`: an oracle failure, not a kernel failure

Diagnosed. **Nothing to fix in FlyDSL.** Not the KV tail mask either: the run at
`bf888e12` already carried that fix (`modules/flash/flyc/flash_attn_func_gfx950.py:185`
has `_KvTailCausalMaskMixin`), and it could not apply here anyway — see below.

### 2.1 The ID reads the other way round

pytest emits stacked `parametrize` params bottom-decorator-first, so in

```
test_irregulars[Flyc-BiasOff-True-l1-dtype1-0.0-CausalOff-2081-257-hdim16-5-3]
```

the `2081` is **`seqlen_k`** and the `257` is **`seqlen_q`**
(`_core_test_backward.py:121-122`), confirmed by the harness's own dump
`q.shape=[3,5,257,16] k.shape=[3,5,2081,16]`. So `seqlen_q < seqlen_k`, and the
KV-tail-mask defect — a `seqlen_q > seqlen_k` top-left-causal bug — cannot reach
this case. The rest: `True` is `storage_flip` (BSHD), `dtype1` is **bf16**, `5`
is `N_HEADS`, `3` is `BATCH`.

### 2.2 Read `tfts` correctly — it cost a round-trip

`tfts` is keyed by `TENSOR_NAMES = ('q','k','v','b')` but holds the
**gradients**: `validate_with_reference` zips `grads = (dq,dk,dv,db)` against
those names (`_common_test.py:525`). `tfts['q']` is dQ's target fudge factor, not
a fingerprint of `q`. And a `tft` is `test_error / ref_error`
(`_common_test.py:329`) — the nans are a **nan denominator**, not a nan tensor.
The tell is `'b': 1.0`: with `BiasOff`, `_validate` returns `(True, 0.0, 1.0)` on
its first line without ever touching the reference, so `b` is the one entry that
does not divide by `ref_error` and the one entry that is not nan.

**The inputs are finite.** Measured through `SdpaContext` at the failing point:
`q`, `k`, `v`, `dout` all `nan=0 inf=0`, range `[0.0000, 0.9961]`. Nothing
non-finite ever reaches the kernel.

### 2.3 What is broken is `lp_ref`, the low-precision oracle

`lp_refout_tensors[0]` holds 6064 nans of 61680, while the fp32
`refout_tensors[0]` is clean and agrees with the flyc forward to 2.0e-3 —
textbook bf16. `_validate` then computes `ref_error = lmax(ref − lp_ref) = nan`,
so `threshold = max(1e-5, nan * fudge) = nan` and `test_error <= threshold` is
False for **every** tensor at once. That simultaneity is itself the signature: a
real kernel bug does not fail the forward and all three gradients by exactly the
same margin.

Had the oracle been sound the case would have passed with ~3x headroom: at benign
shapes `ref_error ≈ 2.1e-3` and `OUT_FUDGE_FACTOR = 3.0`, so threshold 6.4e-3
against our 2.0e-3.

### 2.4 The flyc kernels are correct here

Measured against a **CPU-only fp64 reference** — softmax, logsumexp and autograd
all on CPU in float64, only the kernel on GPU. B=3 H=5 D=16 bf16, BSHD-strided,
`scale=1/16`, relative Frobenius error:

| | O | LSE | dQ | dK | dV |
|---|---|---|---|---|---|
| 257/2081 | 1.636e-03 | 1.616e-05 | 1.556e-04 | 6.013e-05 | 1.638e-03 |
| 257/2081 causal | 1.650e-03 | 1.630e-05 | 1.546e-04 | 6.012e-05 | 1.988e-03 |
| 256/2048 (control) | 1.671e-03 | 1.620e-05 | 1.542e-04 | 6.006e-05 | 1.726e-03 |
| 256/2048 causal (control) | 1.618e-03 | 1.634e-05 | 1.563e-04 | 6.011e-05 | 2.011e-03 |

Zero nans anywhere. Target and control are indistinguishable. Same at head_dim
128.

### 2.5 Root cause: a torch bf16 batched GEMM bug on gfx950

Reachable with no AOTriton and no FlyDSL, torch `2.12.0+rocm7.14.0`:

```python
import torch
a = torch.rand(15, 257,   16, device='cuda', dtype=torch.bfloat16)
b = torch.rand(15,  16, 2081, device='cuda', dtype=torch.bfloat16)
out = torch.full((15, 257, 2081), -12345.0, device='cuda', dtype=torch.bfloat16)
torch.bmm(a, b, out=out)
print(int((out == torch.tensor(-12345.0, dtype=torch.bfloat16)).sum()))  # 274733 never written
```

Output columns 128..2080 are wrong; ~275k elements are never written at all.
`batch = BATCH*N_HEADS = 15`, `M = seqlen_q = 257`, `N = seqlen_k = 2081`,
`K = head_dim`. It needs **all** of: bf16 (fp16 and fp32 clean), batch 15 (1 and
8 clean), M 257 (256 and 512 clean), N 2081 (2048 and 4096 clean), K ≤ 64 (128
clean). Each row of that sweep was a fresh interpreter, so allocator state cannot
leak between them; the wrong region is deterministic across repeats.

The leftover bytes are whatever the caching allocator last left there — which is
why only 44 of the ~500 matching parameter points show nan, and why the failing
head-dim set moves between runs. **The rest of the family is silently wrong
without tripping anything.**

Verified against controls in your own suite (target fails, all four pass,
deterministic): `storage_flip=False`, `seqlen_k=1063`, `dtype0` (fp16),
`seqlen_q=523`. The failing set matches the GEMM bug's axes exactly: 44/44 bf16,
44/44 `storage_flip=True`, 44/44 `2081-257`, 0/44 `storage_flip=False`.

### 2.6 Two things to do on your side, in order

1. **Make a broken oracle loud.** `_validate` guards
   `isnan(test_error) and not isnan(ref_error)` but not the converse. Add the
   mirror: if `ref_error` is nan, or `lp_ref` contains nan while `ref` does not,
   the *reference* is unusable — `pytest.fail`/`xfail` with "oracle produced
   NaN" rather than reporting the backend as inaccurate. Right now a torch bug
   and a backend bug are indistinguishable in the log. While you are there,
   consider renaming the `tfts` keys to `dq/dk/dv/db`; `'q': nan` reads as "the
   input q is nan" and did exactly that to two readers.
2. **Report the torch/hipBLASLt bug** with the four-line reproducer. It is a
   silent wrong-answer bug in bf16 `bmm`, so it is not only a test-harness
   problem. Note the precedent already in `create_ref_inputs`: the
   `cunn_SoftMaxForward` workaround that moved the reference to CPU, retired on
   "known softmax issues have been fixed in 2.7". Different kernel, not fixed. A
   shape-triggered CPU fallback for the bf16 reference would unblock the suite —
   and a CPU reference is the right default for an oracle regardless, since a GPU
   one can share the failure it is meant to detect.

**Do not widen a fudge factor for these.** The threshold is nan, not too tight;
no finite fudge factor changes the outcome, and raising one would only hide real
regressions elsewhere.

---

## 3. What to re-run

1. The 1193 `test_op_bwd[TH-…]` against FlyDSL at or after the commit carrying
   this file. Expect them to pass; if any do not, that is a new finding and we
   want the case.
2. **Not** the 44 `test_irregulars` — re-running them changes nothing until the
   oracle is fixed (§2.6 item 1). They will keep failing against a nan threshold
   no matter what our kernel does, and the run that produced `sel10.txt` already
   carried every FlyDSL fix that could conceivably apply.
3. Nothing else — the rest of your matrix was unaffected by this change, and our
   own suite is green at 748 / 314 / 338 / 13 / 21+3.

Provenance, since it came up: the KV tail mask fix is FlyDSL `1c62d7fd`
(2026-09-02 06:06 UTC); `flyc_pass10.out` records `AOTRITON_GIT_SHA1 = bf888e12`
(2026-09-07 04:15) and both it and `sel10.txt` are stamped 2026-09-07 05:47, with
the vendored kernels at `modules/flash/flyc/` synced 2026-09-07 01:55. The run
postdates the fix. The 1193 TH failures do predate the LSE fix `2fd1a264`
(2026-09-07 08:22), which is consistent with §1.
