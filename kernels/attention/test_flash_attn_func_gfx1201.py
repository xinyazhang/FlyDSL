# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness tests for the gfx1201 FMHA kernels, against PyTorch SDPA.

Covers both scheduling variants behind
``flydsl_flash_attn_func_gfx1201``: the baseline kernel and the
binding-prefetch schedule (``FmhaKnobs(k_prefetch_dist=1)``).

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

from flash_attn_func_gfx1201_aiw import build_flash_attn_func_aiw_module
from fmha_tuning_gfx1201 import FmhaKnobs
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
        torch.randn(batch, heads, seq, head_dim, dtype=dtype, device="cuda", generator=gen) for _ in range(3)
    )


def _reference(q, k, v, causal):
    """fp32 BHSD reference via PyTorch SDPA."""
    qb, kb, vb = (x.float() for x in (q, k, v))
    return F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal)


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

    out = flydsl_flash_attn_func_gfx1201(q, k, v, causal=causal, knobs=FmhaKnobs(k_prefetch_dist=1 if use_binding_prefetch else None))
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

    base = flydsl_flash_attn_func_gfx1201(q, k, v, causal=causal, knobs=None)
    bp = flydsl_flash_attn_func_gfx1201(q, k, v, causal=causal, knobs=FmhaKnobs(k_prefetch_dist=1))
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

    out = flydsl_flash_attn_func_gfx1201(q, k, v, causal=True, knobs=FmhaKnobs(k_prefetch_dist=1 if use_binding_prefetch else None))
    torch.cuda.synchronize()

    assert out.shape == q.shape
    rel, cos = _compare(out, _reference(q, k, v, causal=True))
    assert rel < _REL_TOL[torch.float16], f"max rel err {rel:.3e}"
    assert cos > _COS_TOL, f"cosine similarity {cos:.6f}"


_IRREGULAR_SEQLENS = [11, 17, 37, 67, 157, 200, 257, 523, 1033, 2063]


@pytest.mark.parametrize("seq", _IRREGULAR_SEQLENS)
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_irregular_seqlen(seq, causal):
    """seq_len need not divide BLOCK_M, and is no longer padded host-side.

    This used to raise for non-causal shapes: the interface rounded seq_len up
    with F.pad, and a zero-padded key gives QK^T = 0 whose exp(0) = 1 still
    lands in the softmax denominator, so anything beyond 0.5% padding was
    rejected outright. The kernel now masks the KV tail to -inf instead, so
    the copy and the restriction are both gone.

    Values are AOTriton's PRIME_SEQLEN_Q, plus 200 for the case the old guard
    rejected.
    """
    _require_env()
    q, k, v = _qkv(1, seq, _NUM_HEADS, 128, torch.float16)
    out = flydsl_flash_attn_func_gfx1201(q, k, v, causal=causal)
    torch.cuda.synchronize()
    assert out.shape == q.shape
    assert torch.isfinite(out).all(), f"seq={seq} causal={causal} not finite"
    rel, cos = _compare(out, _reference(q, k, v, causal))
    assert rel < 5e-3, f"seq={seq} causal={causal} rel={rel:.3e}"


def test_irregular_seqlen_does_not_copy():
    """The inputs must reach the kernel untouched -- no F.pad, no contiguous().

    Regression guard: the padding copy was three tensors per call, and its
    removal is only real if nothing reintroduces it.
    """
    _require_env()
    q, k, v = _qkv(1, 200, _NUM_HEADS, 128, torch.float16)
    before = (q.data_ptr(), k.data_ptr(), v.data_ptr())
    flydsl_flash_attn_func_gfx1201(q, k, v, causal=False)
    torch.cuda.synchronize()
    assert (q.data_ptr(), k.data_ptr(), v.data_ptr()) == before


def test_builder_rejects_unsupported_head_dim():
    """head_dim must be a multiple of 16 within 16..512.

    Was a binding-prefetch test; retargeted when that kernel retired. Its
    companion case -- BLOCK_N must be 32 -- retired with it rather than moving,
    because that constraint was specific to the binding-prefetch schedule and
    the unified kernel deliberately runs BLOCK_N 64 and 128 on the distance-0
    path at small head_dim.
    """
    with pytest.raises(AssertionError, match="BLOCK_DMODEL"):
        build_flash_attn_func_aiw_module(
            num_heads=_NUM_HEADS, head_dim=100, causal=False, dtype_str="f16"
        )


# The m32 variant gives each wave two Q row sub-tiles so one K/V operand feeds two
# WMMAs; BLOCK_M is unchanged and the wave count per workgroup halves. Allowed
# up to head_dim 80 -- past that the doubled per-wave state exceeds the
# 256-VGPR cap and spills. Whether it is *faster* is shape-dependent; see the
# table at `_ROW_SUBTILES_2_HEAD_DIMS` in the interface.
_M32_SHAPES = [
    (1, 512, False, torch.float16),
    (2, 1024, False, torch.float16),
    (1, 512, True, torch.float16),
    (2, 768, True, torch.bfloat16),
    (1, 300, True, torch.float16),  # seq_len not a multiple of BLOCK_M=256
]


@pytest.mark.parametrize(
    "shape",
    _M32_SHAPES,
    ids=[f"b{b}_s{s}_{'causal' if c else 'full'}_{str(d).split('.')[-1]}" for b, s, c, d in _M32_SHAPES],
)
def test_m32_matches_sdpa(shape):
    _require_env()
    batch, seq, causal, dtype = shape
    q, k, v = _qkv(batch, seq, _NUM_HEADS, 64, dtype)
    out = flydsl_flash_attn_func_gfx1201(q, k, v, causal=causal, knobs=FmhaKnobs(row_subtiles=2))
    torch.cuda.synchronize()
    assert out.shape == q.shape and out.dtype == q.dtype
    rel, cos = _compare(out, _reference(q, k, v, causal))
    assert rel < _REL_TOL[dtype], f"max rel err {rel:.3e}"
    assert cos > _COS_TOL, f"cosine similarity {cos:.6f}"


def test_m32_rejects_head_dim_128():
    """head_dim 128 would spill; the interface must refuse rather than regress."""
    _require_env()
    q, k, v = _qkv(1, 256, _NUM_HEADS, 128, torch.float16)
    with pytest.raises(ValueError, match="head_dim <= 80"):
        flydsl_flash_attn_func_gfx1201(q, k, v, causal=False, knobs=FmhaKnobs(row_subtiles=2))


@pytest.mark.parametrize("head_dim", [16, 32, 48, 80])
def test_m32_accepted_below_the_spill_bound(head_dim):
    """The bound is the VGPR cap, so everything under it must work.

    It read `head_dim != 64` for a while, which is stricter than the reason
    given for it -- the doubled per-wave state only grows with head_dim, so a
    narrower head cannot be the thing that spills.
    """
    _require_env()
    q, k, v = _qkv(1, 512, _NUM_HEADS, head_dim, torch.float16)
    out = flydsl_flash_attn_func_gfx1201(q, k, v, causal=True, knobs=FmhaKnobs(row_subtiles=2))
    torch.cuda.synchronize()
    rel, cos = _compare(out, _reference(q, k, v, True))
    assert rel < _REL_TOL[torch.float16], f"max rel err {rel:.3e}"
    assert cos > _COS_TOL, f"cosine similarity {cos:.6f}"


def test_shape_and_dtype_validation():
    """Interface-level guards that do not require a kernel build."""
    _require_env()
    q, k, v = _qkv(1, 256, _NUM_HEADS, 128, torch.float16)

    with pytest.raises(ValueError, match="must share"):
        flydsl_flash_attn_func_gfx1201(q, k[:, :, :128], v[:, :, :128], causal=True)
    with pytest.raises(ValueError, match="dtype must match"):
        flydsl_flash_attn_func_gfx1201(q, k.to(torch.bfloat16), v, causal=True)
    with pytest.raises(ValueError, match="rank"):
        flydsl_flash_attn_func_gfx1201(q[0], k[0], v[0], causal=True)


# The full head_dim ladder the kernel is expected to cover. 160/192/224 are the
# regression cases for the cooperative-load batch-count bug: ROWS_PER_BATCH_LOAD
# came to 25/21/18, and BLOCK_N // that == 1, so only 25/21/18 of the 32 KV rows
# were written to LDS and the output came back NaN.
_HEAD_DIMS = [16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 384, 512]


@pytest.mark.parametrize("head_dim", _HEAD_DIMS)
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_head_dim_ladder(head_dim, causal):
    """Every supported head_dim matches SDPA, for both masking modes."""
    _require_env()
    q, k, v = _qkv(1, 256, 2, head_dim, torch.float16)
    got = flydsl_flash_attn_func_gfx1201(q, k, v, causal=causal)
    assert got.shape == q.shape
    assert torch.isfinite(got).all(), f"head_dim={head_dim} causal={causal} produced non-finite output"
    rel, cos = _compare(got, _reference(q, k, v, causal))
    assert rel < 5e-3, f"head_dim={head_dim} causal={causal} rel={rel:.3e}"
    assert cos > 0.9999, f"head_dim={head_dim} causal={causal} cos={cos:.6f}"


def test_head_dim_requires_aligned_pitch():
    """An off-ladder head_dim needs an 8-element-aligned D pitch."""
    _require_env()
    q, k, v = _qkv(1, 256, 2, 100, torch.float16)   # contiguous, pitch 100
    with pytest.raises(ValueError, match="pitch"):
        flydsl_flash_attn_func_gfx1201(q, k, v, causal=False)


@pytest.mark.parametrize("head_dim", [640, 1024])
def test_head_dim_validation(head_dim):
    """Only head_dim beyond the largest compiled tile is rejected.

    Values that are not themselves tile widths are no longer an error: they
    round up to the next compiled BLOCK_DMODEL and the real extent rides along
    as a runtime argument (PADDED_HEAD). See test_head_dim_off_ladder.
    """
    _require_env()
    q, k, v = _qkv(1, 256, 2, head_dim, torch.float16)
    with pytest.raises(ValueError, match="head_dim"):
        flydsl_flash_attn_func_gfx1201(q, k, v, causal=False)


@pytest.mark.parametrize("head_dim", [8, 24, 100, 113, 200, 272])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_head_dim_off_ladder(head_dim, causal):
    """head_dim need not be a compiled tile width, nor a multiple of 16.

    These previously raised. They now round up to the next BLOCK_DMODEL, and
    the kernel masks the difference.
    """
    _require_env()
    # Allocate with the D axis padded to a multiple of 8 elements, then slice
    # back -- the alignment contract, and what PyTorch's SDPA shim delivers.
    pitch = (head_dim + 7) // 8 * 8
    gen = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (
        torch.randn(1, 2, 256, pitch, dtype=torch.float16, device="cuda", generator=gen)[
            ..., :head_dim
        ]
        for _ in range(3)
    )
    out = flydsl_flash_attn_func_gfx1201(q, k, v, causal=causal)
    torch.cuda.synchronize()
    assert out.shape == q.shape
    assert torch.isfinite(out).all(), f"head_dim={head_dim} produced non-finite output"
    rel, cos = _compare(out, _reference(q, k, v, causal))
    assert rel < 5e-3, f"head_dim={head_dim} causal={causal} rel={rel:.3e}"
    assert cos > 0.9999, f"head_dim={head_dim} causal={causal} cos={cos:.6f}"
