# Overview

This document describes feature gaps with AOTriton's SDPA Triton kernel (Found
at `~/dockerhome/meff/aotriton/modules/flash/kernel/fwd_kernel.py` as Triton
kernel `attn_fwd`)

# Required Changes to use as an AOT kernel

As a JIT DSL, lots of variables can be baked into the kernel binary. A
concrete example is the number of heads, which is commonly baked into the
Triton kernel as `tl.constexpr` for marginal performance boost.

However, if we want to use it as AOT kernel, we have to read them as kernel
parameters because we need to use a fixed set of kernels to cover all possible
inputs, and cannot generate kernels on the fly.

Here is a maybe-incomplete list of arguments that's `tl.constexpr` but must be
kernel arguments in a production ready SDPA AOT kernel.

* All `stride_*`, Except for the last dimension
  + Concretely: `stride_qk, stride_kk, stride_vn, stride_on, stride_bn`. They
    are all constexpr `1`
  + `D` dimension must be contiguous otherwise the performance is abysmal and
    users should fix the data's memory layout rather than asking for support.
* `Sm_scale`
* `Num_head_q/k`
* `Num_seqlens` (This implies a major feature gap, will discuss later)
* `Max_seqlen_q/k`
* `Hdim_qk/vo` (This is a major feature gap, will discuss later)
* `Window_left/right` (This is a major feature gap, will discuss later)

## `stride_*`

The kernel should handle all memory layouts, not only the standard BHSD/BSHD
ones. The only constraint is the `D` dimension (last dimension) must be
contiguous.

We need to pass them as kernel arguments instead of let kernel calculate them
from tensor shapes. The latter (and current) approach assume contiguous
tensors which is not always true.

Note the kernel assume the tensors are always BHSD **SHAPE** (exception: bias
tensor B is BHSqSk). If a tensor were created as some other shape like BSHD,
it must be transposed to fit the BHSD shape. Since the kernel reads strides
rather than guessing from the shape, `torch.transpose` is sufficient for this
goal and no need to call `.contiguous()`, which is cheap.

### Special Instruction When Porting

Name the stride variable numerically rather than with odd suffix like `z/h/m/k`.

For example you should use `stride_q0, stride_q1, stride_q2`, not `stride_qz, stride_qh, stride_qm`.
(`stride_qk` does not need to be ported and is hard-coded as 1).

Rationale: these odd suffixes are likely inherited from the math equations but
in practice they are hard to read if you don't care about the equations.
Actually coding mistakes by misusing the strides happened more than once
during AOTriton's kernel development.

## `Hdim_qk/vo`

These arguments implies two features

* The kernel must handle Q/K/V/Out's real hdim is different from `BLOCK_DMODEL`
  + As you can find in the kernel, `BLOCK_DMODEL` is the hdim baked into the
    kernel binary. But only `Hdim_qk/vo` elements shall be loaded into
    register and the rest must be padded as zero.
    - It's a Triton's design limitation that it must use power of two shapes,
      but we do not have such limitation since we are working on a much lower
      level.
  + A `constexpr[bool]` `PADDED_HEAD` is added to the kernel to enable or
    disable the padded loading since it has great performance impact.
    - Justify the overhead, we can have `PADDED_HEAD` in our kernel and build two
      variants for best performance as well. (However I'm curious why this is
      a problem since the register pressure from padded loading are temporary.)
* The kernel must handle `Hdim_qk != Hdim_vo` cases
  + The major work should be automatically completed when `PADDED_HEAD` is handled.

## `Num_seqlens`

`Num_seqlens` is used as a sentinel variable to switch between regular SDPA
inputs and varlen inputs

* `Num_seqlens == 0`: standard regular BHSD tensor.
* `Num_seqlens > 0`: Varlen inputs, 1HTD input shape
* `Num_seqlens < 0`: Varlen inputs, BHSD input shape, but S is padded to
  `Max_seqlen_q` so it's regular.
  + Note, the upstream API supplies real `seqlens_q/k` instead, so we should use
    real `seqlens_q/k` rather than AOTriton's `cu_seqlens_q/k`. Please figure
    out a better name for an argument that has dual role: `cu_seqlens_q` vs
    `real_seqlens_q` (`cu_` here is cumulative)

You probably needs to check `04cdead5c837f21754de39a175a30c479da74625` for
more details.

## `Max_seqlen_q/k`

For real inputs, `seqlen_q/k` are not always multiple of `BLOCK_M/N`, the
kernel must address this. Triton kernel's lesson is we have to implement the
inner kernel with regular/irregular variant and only call the irregular
variant at the trailing blocks. (Note "trailing blocks" may not be at the
trail with causal masks and our **generalised** sliding windowed attention)

## `Window_left/right`

AOTriton's kernel implements Causal masks with **Generalized** Sliding Window
Attention (gSWA). Unlike the conventional SWA which uses negative
`window_left/right` for "disable SWA on this direction", our gSWA
use negative values of `Window_left/right` to shift the relevant SWA boundary in the opposite
direction. Hence, the top-left and bottom-right variant of causal masks can
both be implemented in our gSWA. Check "Simulate Causal Masks with
Sliding Window Attention (SWA).pptx" for more details.

This is a big feature and should defer to later phase. Notably: the inner
loop's `for block_index in tl.range(nblocks_1+nblocks_2,
num_stages=NUM_STAGES):`  is required to support gSWA, where the next block
may not be adjacent to current one. Hence its performance needs revise.
(I think we should implement gSWA after persistent dynamic since both are for
causal masks and needs optimization?)

# Missing Feature: Bias Tensor

`B` and `stride_b*`

# Missing Feature: dropout support

`dropout_p` and `philox_*` are the inputs. You can easily parse how its
implemented. This is a big feature and let's defer it to later phase -- but
before `Window_left/right` because SWA is incremental over causal but still
needs lots of work, and dropout is a bigger gap.

Note: you don't need to match the exact implementation (I hacked the triton's
PRNG API a lot), just implement the philox PRNG in the most efficient way (but
do not scarify the quality or PRNG like cutting the seed/offset down to
32-bit or reducing the rounds of the PRNG)

# Improvements from Stock FAv2 Triton kernel

Here I listed features I added to the kernel itself but not visible from its
interface

## `MQA/GQA` support

Need to support `Num_head_q != Num_head_k`

## Safer softmax

`m_i = tl.full([BLOCK_M], -3.40282e+38, dtype=tl.float32)`

Do not use -inf. Nan will be created on certain cases.

## Avoid FMA

See
https://github.com/ROCm/aotriton/commit/14d673f4ea90a5a4e1cea5442d22bc7b1e9146cf

The bug is recorded as https://github.com/ROCm/aotriton/issues/54

## Use recip

`l_recip = 1 / l_i[:, None]`

## Optional logsumexp calculation

`if L_not_null:`

This means we can call the kernel with logsumexp = nullptr

## Compact logsumexp tensor in varlen mode.

Match FA's behavior

# Real Feature gap, but not for now

## `PERSISTENT_TYPE`

We use `2` aka `PERSISTENT_DYNAMIC` to dynamically assign workloads to
work-groups, which provides reasonable performance boost for causal and
presumabley for varlen workloads, but this feature requires performance fine
tuning and we should defer this to a separate, following up task which both
implement and fine-tune the persistent dynamic kernel.

## `NUM_XCDS`

It's always `1` on gfx1201. We can continue this when working on gfx1250.

## `INT8`

No plan to support INT8 ATM. However a similar function `mxfp8` is planned but
the design is not finalized yet (re-use `attn_fwd` or create a separate
kernel?).

## `PRE_LOAD_V`

This is a tuning knob to relocate V tensor's load, but I believe the current
one is already optimized.

# Looks like but not a real feature gap

## `RETURN_ENCODED_SOFTMAX`

The AOT kernel is always built as `RETURN_ENCODED_SOFTMAX=False`, and we ship
a separate kernel and run it after `attn_fwd`

## `BLOCK_M/N`

Tuning knob, Already optimized on gfx1201, for other arches they should be
optimized individually as well.

## `BLOCK_DMODEL0/1/2`

This is to support non-power-of-two hdims since Triton only allows POT
tensors. We don't need them since we already support them.

