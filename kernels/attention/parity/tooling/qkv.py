# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Q/K/V builders for the tooling scripts: BHSD *shape*, either *layout*.

The kernel ABI fixes the shape and leaves the layout free, and those are
different things -- `_strides_of` in the forward kernel spells the distinction
out. Every tooling script had the shape hardcoded and the layout implied by
whichever `torch.randn` argument order someone typed, which is how three of
them kept building BSHD-shaped tensors for months after the ABI moved to BHSD.
Comparisons stayed valid, because both sides of a diff used the same wrong
config, but the labels did not.

So: one builder, shape not negotiable, layout an argument.

    LAYOUT_BHSD   contiguous `(B, H, S, D)`. What AOTriton hands the kernel,
                  and the one where a head's tokens are `D` apart.
    LAYOUT_BSHD   allocated `(B, S, H, D)` and transposed to BHSD shape. The
                  same logical tensor with tokens `H*D` apart -- a genuinely
                  different access pattern, and the reason this is a knob
                  rather than a detail.

Both satisfy the only layout constraint the kernel has, `stride(3) == 1`.
"""

import torch

LAYOUT_BHSD = "bhsd"
LAYOUT_BSHD = "bshd"
LAYOUTS = (LAYOUT_BHSD, LAYOUT_BSHD)


def make_qkv(
    batch, heads, seq, head_dim, *, dtype=torch.float16, device="cuda", layout=LAYOUT_BHSD, seed=None, n=3, pitch=None
):
    """`n` tensors of BHSD shape `(batch, heads, seq, head_dim)`.

    `pitch` pads the D axis of the allocation and slices back, which is what a
    non-multiple-of-8 `head_dim` needs -- the kernel's 8-wide accesses run to
    `ceil8(head_dim)` and that has to land inside the tensor.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    gen = None if seed is None else torch.Generator(device=device).manual_seed(seed)
    alloc_d = head_dim if pitch is None else pitch

    def one():
        if layout == LAYOUT_BHSD:
            t = torch.randn(batch, heads, seq, alloc_d, dtype=dtype, device=device, generator=gen)
        else:
            # Allocate BSHD, view as BHSD: same values, tokens H*D apart.
            t = torch.randn(batch, seq, heads, alloc_d, dtype=dtype, device=device, generator=gen).transpose(1, 2)
        return t if pitch is None else t[..., :head_dim]

    out = tuple(one() for _ in range(n))
    return out[0] if n == 1 else out
