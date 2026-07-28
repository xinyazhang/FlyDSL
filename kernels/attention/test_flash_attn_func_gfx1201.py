# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness tests for the gfx1201 FMHA kernels, against PyTorch SDPA.

Covers both scheduling variants behind
``flydsl_flash_attn_func_gfx1201``: the baseline kernel and the
binding-prefetch variant (``use_binding_prefetch=True``).

This file is **deliberately outside the `tests/` tree** and is not wired into
`scripts/run_tests.sh`. It follows the prototype convention of this directory --
bare module imports, cwd must be `kernels/attention` -- so run it individually::

    export ROCM_PATH=$(rocm-sdk path --root)
    cd kernels/attention && python3 -m pytest test_flash_attn_func_gfx1201.py -v

Note that `scripts/run_tests.sh` passes explicit directories under `tests/`, so
this file is never collected by the project's suite. A bare `pytest` from the
repository root would collect it; that is not how this project is run, but be
aware it would cost several JIT compiles.

Each distinct (num_heads, head_dim, causal, dtype, variant) combination is a
separate kernel build, so the cases below deliberately share `num_heads` to keep
the `lru_cache` in the interface warm and the run time bounded.
"""

import os

import pytest
import torch
import torch.nn.functional as F

from flash_attn_func_gfx1201_bp import build_flash_attn_func_bp_module
from flash_attn_func_gfx1201_interface import flydsl_flash_attn_func_gfx1201

# Relative-error tolerance against an fp32 reference. The kernel accumulates in
# fp32 but rounds Q/K/V and the P matrix to the input dtype, so the floor is set
# by the input dtype's mantissa, not by the accumulator.
_REL_TOL = {torch.float16: 2e-2, torch.bfloat16: 6e-2}
_COS_TOL = 0.999

_NUM_HEADS = 2
_KERNEL_BLOCK_M = 128


def _require_env():
    if not torch.cuda.is_available():
        pytest.skip("no HIP/CUDA device")
    arch = torch.cuda.get_device_properties(0).gcnArchName.lower().split(":")[0]
    if not arch.startswith("gfx1201"):
        pytest.skip(f"kernel is gfx1201-only, got {arch}")
    if not os.environ.get("ROCM_PATH") and not os.path.isdir("/opt/rocm"):
        pytest.skip(
            "ROCM_PATH is unset and /opt/rocm is absent, so the JIT cannot find "
            "ld.lld; set ROCM_PATH=$(rocm-sdk path --root)"
        )


def _qkv(batch, seq, heads, head_dim, dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        torch.randn(batch, seq, heads, head_dim, dtype=dtype, device="cuda", generator=gen) for _ in range(3)
    )


def _reference(q, k, v, causal):
    """fp32 BSHD reference via PyTorch SDPA."""
    qb, kb, vb = (x.transpose(1, 2).float() for x in (q, k, v))
    return F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal).transpose(1, 2)


def _compare(got, ref):
    scale = ref.abs().max().item()
    rel = (got.float() - ref).abs().max().item() / scale
    cos = F.cosine_similarity(got.float().flatten(), ref.flatten(), dim=0).item()
    return rel, cos


# (batch, seq, head_dim, causal, dtype)
_SHAPES = [
    (1, 256, 64, False, torch.float16),
    (1, 256, 128, False, torch.float16),
    (1, 384, 128, True, torch.float16),
    (2, 512, 128, False, torch.float16),
    (1, 256, 64, True, torch.bfloat16),
    (2, 512, 128, False, torch.bfloat16),
]


def _shape_id(shape):
    b, s, d, causal, dtype = shape
    return f"b{b}_s{s}_d{d}_{'causal' if causal else 'full'}_{str(dtype).split('.')[-1]}"


@pytest.mark.parametrize("use_binding_prefetch", [False, True], ids=["baseline", "bindingprefetch"])
@pytest.mark.parametrize("shape", _SHAPES, ids=[_shape_id(s) for s in _SHAPES])
def test_matches_sdpa(shape, use_binding_prefetch):
    """Both variants must match an fp32 SDPA reference."""
    _require_env()
    batch, seq, head_dim, causal, dtype = shape
    q, k, v = _qkv(batch, seq, _NUM_HEADS, head_dim, dtype)

    out = flydsl_flash_attn_func_gfx1201(q, k, v, causal=causal, use_binding_prefetch=use_binding_prefetch)
    torch.cuda.synchronize()

    assert out.shape == q.shape
    assert out.dtype == q.dtype

    rel, cos = _compare(out, _reference(q, k, v, causal))
    assert rel < _REL_TOL[dtype], f"max rel err {rel:.3e} exceeds {_REL_TOL[dtype]:.1e}"
    assert cos > _COS_TOL, f"cosine similarity {cos:.6f} below {_COS_TOL}"


@pytest.mark.parametrize("shape", _SHAPES, ids=[_shape_id(s) for s in _SHAPES])
def test_binding_prefetch_matches_baseline_bitwise(shape):
    """The binding-prefetch variant reschedules only; it must not change results.

    This is stricter than the SDPA comparison on purpose -- it catches a
    reordering that perturbs accumulation even when the result stays within
    tolerance. If a future optimisation deliberately changes the arithmetic
    (e.g. a different accumulation order), relax this to an allclose and say so.
    """
    _require_env()
    batch, seq, head_dim, causal, dtype = shape
    q, k, v = _qkv(batch, seq, _NUM_HEADS, head_dim, dtype)

    base = flydsl_flash_attn_func_gfx1201(q, k, v, causal=causal, use_binding_prefetch=False)
    bp = flydsl_flash_attn_func_gfx1201(q, k, v, causal=causal, use_binding_prefetch=True)
    torch.cuda.synchronize()

    assert torch.equal(base, bp), f"max |base - bp| = {(base.float() - bp.float()).abs().max().item():.3e}"


@pytest.mark.parametrize("use_binding_prefetch", [False, True], ids=["baseline", "bindingprefetch"])
def test_causal_seqlen_not_multiple_of_block(use_binding_prefetch):
    """seq_len is padded up to BLOCK_M internally, then sliced back off.

    Causal only: padded K/V rows sit past every real query, so the mask removes
    them. The non-causal path rejects this instead (see the test below).
    """
    _require_env()
    seq = 200
    assert seq % _KERNEL_BLOCK_M != 0
    q, k, v = _qkv(1, seq, _NUM_HEADS, 128, torch.float16)

    out = flydsl_flash_attn_func_gfx1201(q, k, v, causal=True, use_binding_prefetch=use_binding_prefetch)
    torch.cuda.synchronize()

    assert out.shape == q.shape
    rel, cos = _compare(out, _reference(q, k, v, causal=True))
    assert rel < _REL_TOL[torch.float16], f"max rel err {rel:.3e}"
    assert cos > _COS_TOL, f"cosine similarity {cos:.6f}"


def test_noncausal_padding_ratio_rejected():
    """Non-causal padding beyond 0.5% must raise, not silently scale the output.

    Padded K/V keys give QK^T = 0, but exp(0) = 1 still enters the softmax
    denominator.
    """
    _require_env()
    q, k, v = _qkv(1, 200, _NUM_HEADS, 128, torch.float16)
    with pytest.raises(ValueError, match="padding ratio"):
        flydsl_flash_attn_func_gfx1201(q, k, v, causal=False)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(num_heads=_NUM_HEADS, head_dim=96), "head_dim"),
        (dict(num_heads=_NUM_HEADS, head_dim=128, block_n=64), "BLOCK_N"),
    ],
    ids=["head_dim_96", "block_n_64"],
)
def test_binding_prefetch_rejects_unsupported_config(kwargs, match):
    """Stage 1 of the binding-prefetch variant guards its supported config set."""
    with pytest.raises(ValueError, match=match):
        build_flash_attn_func_bp_module(causal=False, dtype_str="f16", **kwargs)


def test_shape_and_dtype_validation():
    """Interface-level guards that do not require a kernel build."""
    _require_env()
    q, k, v = _qkv(1, 256, _NUM_HEADS, 128, torch.float16)

    with pytest.raises(ValueError, match="must share"):
        flydsl_flash_attn_func_gfx1201(q, k[:, :128], v[:, :128], causal=True)
    with pytest.raises(ValueError, match="dtype must match"):
        flydsl_flash_attn_func_gfx1201(q, k.to(torch.bfloat16), v, causal=True)
    with pytest.raises(ValueError, match="rank"):
        flydsl_flash_attn_func_gfx1201(q[0], k[0], v[0], causal=True)
