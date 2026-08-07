# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""The mask kernel, and the attention kernel checked against it.

Three layers, weakest last:

1. The mask kernel against a CPU Philox -- the values are right.
2. The mask kernel against itself at other tilings -- `sdpa-dropout-plan.md`
   §3's contract, that the mask is a function of absolute
   `(batch, head, row, column)` and of nothing about how work was divided.
3. The attention kernel against PyTorch's math backend, *fed this kernel's
   mask*. This is the one that says the two kernels agree about which element
   gets which random, which is what the backward pass will depend on.

Layer 3 is only sharp because `_scaled_dot_product_attention_math` takes a
`dropout_mask` and applies it the way the kernel does -- after the softmax,
with the `1/(1-p)` scale, leaving the denominator undropped. Passing the same
mask as `attn_mask` instead would be a *different computation*: `attn_mask` is
additive before the softmax, so the survivors get renormalised over themselves
and the result is not dropout at all.
"""

import numpy as np
import pytest
import torch

from dropout_mask_gfx1201 import dropout_mask
from flash_attn_func_gfx1201_aiw import build_flash_attn_func_aiw_module
from philox import PHILOX_WIDTHS, dropout_threshold, randoms_per_offset
from test_philox import _ref_u32

_ARCH_OK = None


def _require_env():
    global _ARCH_OK
    if _ARCH_OK is None:
        from flydsl.runtime.device import get_rocm_arch

        _ARCH_OK = get_rocm_arch()
    if not _ARCH_OK.startswith("gfx120"):
        pytest.skip(f"gfx120* only, got {_ARCH_OK}")


def _keep_from_raw(raw, p):
    """Host-side keep mask: the same signed compare the kernel makes.

    Redone in torch rather than read out of the kernel on purpose -- the
    threshold is shared code, but nothing else about this compare is, so a
    sign error in `keep_mask` shows up here.
    """
    return raw > dropout_threshold(p)


# ---------------------------------------------------------------------------
# Layer 1: the values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", PHILOX_WIDTHS)
def test_mask_matches_cpu_philox(width):
    """Every element is the random the §3 offset scheme says it should be."""
    _require_env()
    B, H, SQ, SK = 2, 3, 64, 64
    seed, off = 0x1234_5678_9ABC_DEF0, 12345
    r = dropout_mask(B, H, SQ, SK, seed, off, philox_width=width).cpu().numpy()

    rn = randoms_per_offset(width)
    row_stride = -(-SK // rn)
    want = np.empty((B, H, SQ, SK), dtype=np.int64)
    for z in range(B):
        for h in range(H):
            base = off + (z * H + h) * SQ * row_stride
            for m in range(SQ):
                for c0 in range(0, SK, rn):
                    vals = _ref_u32(seed, base + m * row_stride + c0 // rn, width)
                    want[z, h, m, c0:c0 + rn] = vals[: min(rn, SK - c0)]
    # The kernel stores int32; the reference is unsigned.
    assert np.array_equal(r.astype(np.int64) & 0xFFFFFFFF, want)


def test_uniform_encoding_is_the_raw_value_rescaled():
    """`[0, 1)` is for reading; the threshold path never sees a float."""
    _require_env()
    kw = dict(philox_seed=4242, philox_offset=9)
    raw = dropout_mask(1, 2, 64, 128, **kw).double()
    uni = dropout_mask(1, 2, 64, 128, encoding="uniform", **kw).double()
    assert torch.allclose(uni, raw * 2.32830658e-10 + 0.5, atol=1e-6)
    assert uni.min() >= 0.0 and uni.max() <= 1.0


# ---------------------------------------------------------------------------
# Layer 2: tiling invariance -- the phase's real gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", PHILOX_WIDTHS)
@pytest.mark.parametrize("block_m,block_n", [(128, 32), (256, 64), (64, 64)])
def test_mask_is_tiling_invariant(width, block_m, block_n):
    """Same mask at every tiling, or the mask is not reproducible.

    This is the contract that lets the tuning tables be re-swept: `BLOCK_M`
    and `BLOCK_N` are scheduling decisions, and a mask that moved with them
    would silently change every model's dropout pattern on the next re-tune,
    and break a backward pass that was compiled at a different tiling from
    its forward.
    """
    _require_env()
    B, H, SQ, SK = 2, 2, 200, 300     # deliberately not a multiple of any tile
    kw = dict(philox_seed=0xDEAD_BEEF_CAFE, philox_offset=77, philox_width=width)
    ref = dropout_mask(B, H, SQ, SK, block_m=64, block_n=32, **kw)
    got = dropout_mask(B, H, SQ, SK, block_m=block_m, block_n=block_n, **kw)
    assert torch.equal(ref, got)


def test_widths_produce_different_streams():
    """`PHILOX_WIDTH` reaches the stream rather than being quietly ignored.

    The two widths are *not* expected to agree -- 4x32 and 4x64 are different
    generators. What matters is that the choice is observable, since a build
    that ignored it would pass every other test here (any consistent stream
    looks random) and then disagree with a backward pass built the other way.
    """
    _require_env()
    kw = dict(philox_seed=5, philox_offset=0)
    a = dropout_mask(1, 1, 64, 64, philox_width=32, **kw)
    b = dropout_mask(1, 1, 64, 64, philox_width=64, **kw)
    assert not torch.equal(a, b)


@pytest.mark.parametrize("field", ["seed", "offset"])
def test_seed_and_offset_are_64_bit(field):
    """A 32-bit truncation of either passes every other test in this file.

    Both values differ only above bit 32, so a build that truncated them would
    produce identical masks here and remain plausible everywhere else.
    """
    _require_env()
    lo = dict(philox_seed=1, philox_offset=1)
    hi = dict(lo, **{f"philox_{field}": (1 << 40) + 1})
    assert not torch.equal(dropout_mask(1, 1, 64, 64, **lo),
                           dropout_mask(1, 1, 64, 64, **hi))


# ---------------------------------------------------------------------------
# Layer 3: the attention kernel applies the mask this kernel reports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_attention_matches_math_backend_given_the_mask(p, head_dim):
    """The whole dropout contract, end to end, as an exact-ish comparison.

    `_scaled_dot_product_attention_math` with a `dropout_mask` is the same
    computation the kernel performs: softmax first, then zero the dropped
    entries, then scale survivors by `1/(1-p)`, leaving the denominator
    computed over everything. So the only thing that can differ is *which*
    entries were dropped -- which is exactly the agreement being tested.

    The tolerance is f16 accumulation, not statistics. A single misplaced
    random moves one row's output by an `O(1)` amount and fails this.
    """
    _require_env()
    B, H, SQ, SK = 1, 8, 256, 256
    seed, off = 20250805, 3
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(B, H, SQ, head_dim, dtype=torch.float16, device="cuda")
        for _ in range(3)
    )
    o = torch.empty_like(q)
    build_flash_attn_func_aiw_module(
        num_heads=H, head_dim=head_dim, causal=False, dtype_str="f16", dropout=True
    )(q, k, v, o, B, SQ, dropout_p=p, philox_seed=seed, philox_offset=off)
    torch.cuda.synchronize()

    keep = _keep_from_raw(dropout_mask(B, H, SQ, SK, seed, off), p)
    ref, _ = torch.ops.aten._scaled_dot_product_attention_math(
        *(t.float() for t in (q, k, v)),
        dropout_p=p,
        dropout_mask=keep,
    )
    ref = ref.half()

    err = (o.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().max().item()
    assert err <= 4e-2 * max(scale, 1.0), (
        f"attention and the mask kernel disagree about the mask: "
        f"max|diff| = {err:.4f} against |ref| = {scale:.4f}"
    )


def test_mask_kernel_disagreeing_is_detected():
    """The negative control for the test above.

    A comparison against a reference is only worth its tolerance if a wrong
    mask actually breaks it. Feeding the reference a mask from a different
    seed must fail, or the tolerance is doing the work.
    """
    _require_env()
    B, H, SQ, head_dim, p = 1, 8, 256, 64, 0.5
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(B, H, SQ, head_dim, dtype=torch.float16, device="cuda")
        for _ in range(3)
    )
    o = torch.empty_like(q)
    build_flash_attn_func_aiw_module(
        num_heads=H, head_dim=head_dim, causal=False, dtype_str="f16", dropout=True
    )(q, k, v, o, B, SQ, dropout_p=p, philox_seed=1, philox_offset=0)
    torch.cuda.synchronize()

    keep = _keep_from_raw(dropout_mask(B, H, SQ, SQ, 2, 0), p)   # wrong seed
    ref, _ = torch.ops.aten._scaled_dot_product_attention_math(
        *(t.float() for t in (q, k, v)),
        dropout_p=p,
        dropout_mask=keep,
    )
    ref = ref.half()
    err = (o.float() - ref.float()).abs().max().item()
    assert err > 4e-2 * max(ref.float().abs().max().item(), 1.0), (
        "a mask from the wrong seed still passed -- the tolerance is too loose"
    )
