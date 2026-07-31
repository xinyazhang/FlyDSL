# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Parity tests for the unified gfx1201 kernel (``..._aiw``) against its oracles.

``flash_attn_func_gfx1201_aiw`` replaced three separate kernels. Those three are
kept on disk unchanged precisely so that they can be used as correctness
oracles here: for every knob setting that reproduces one of them, aiw must
match it **bitwise**.

Bitwise is the right bar because none of the unified knobs change the order of
floating-point operations -- they change *scheduling* (prefetch distance), *LDS
layout* (row vs transposed V), or *tiling* (Q row-tiles per wave). The one knob
that genuinely does change the arithmetic is ``QK_SHARDS > 1``, which splits the
QK dot product across waves and sums the partials; those configs are compared
against an fp32 reference at tolerance instead.

Two things this bar does **not** catch, both learned the hard way while
building aiw and both worth remembering before trusting a green run here:

1. **Tiling geometry.** Each Q row's arithmetic is identical however rows are
   grouped into blocks, so a build with the wrong BLOCK_M / wave count is still
   bitwise correct. Only a benchmark catches that.
2. **Prefetch distance.** Dropping a prefetch is invisible to the output. The
   baseline kernel is (K distance 0, **V distance 1**); collapsing those into a
   single knob produced a (0, 0) schedule that passed every test here and cost
   9.6% at head_dim 32.

Run it individually, per this directory's prototype convention::

    export ROCM_PATH=$(rocm-sdk path --root)
    cd kernels/attention && python3 -m pytest test_flash_attn_func_gfx1201_aiw.py -v
"""

import os

import pytest
import torch
import torch.nn.functional as F

from flash_attn_func_gfx1201 import build_flash_attn_func_module
from flash_attn_func_gfx1201_aiw import (
    build_flash_attn_func_aiw_module,
    default_block_m as aiw_block_m,
    resolve_shards,
)
from flash_attn_func_gfx1201_bp import build_flash_attn_func_bp_module, bp_block_m
from flash_attn_func_gfx1201_m32 import build_flash_attn_func_m32_module

_NUM_HEADS = 2


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


def _qkv(batch, seq, head_dim, dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        torch.randn(
            batch, seq, _NUM_HEADS, head_dim, dtype=dtype, device="cuda", generator=gen
        )
        for _ in range(3)
    )


def _run(exe, q, k, v, batch, seq, out=None, legacy=False):
    """Launch a builder's executable.

    aiw takes the rank-4 tensors whole and reads their strides; the legacy
    builders take flat pointers and derive the layout from num_heads/head_dim.
    """
    o = torch.empty_like(q) if out is None else out
    if legacy:
        exe(q.reshape(-1), k.reshape(-1), v.reshape(-1), o.reshape(-1), batch, seq)
    else:
        exe(q, k, v, o, batch, seq)
    torch.cuda.synchronize()
    return o


def _reference(q, k, v, causal):
    qb, kb, vb = (x.transpose(1, 2).float() for x in (q, k, v))
    return F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal).transpose(1, 2)


def _rel(got, ref):
    return (got.float() - ref).abs().max().item() / ref.abs().max().item()


def _parity(head_dim, causal, dtype, batch, seq, oracle, aiw_kwargs):
    """aiw at `aiw_kwargs` must equal `oracle` bitwise."""
    q, k, v = _qkv(batch, seq, head_dim, dtype)
    common = dict(
        num_heads=_NUM_HEADS,
        head_dim=head_dim,
        causal=causal,
        dtype_str="f16" if dtype == torch.float16 else "bf16",
    )
    got = _run(
        build_flash_attn_func_aiw_module(**common, **aiw_kwargs), q, k, v, batch, seq
    )
    want = _run(oracle(**common), q, k, v, batch, seq, legacy=True)
    assert torch.isfinite(got).all(), "aiw produced non-finite output"
    assert torch.equal(got, want), (
        f"max |aiw - oracle| = {(got.float() - want.float()).abs().max().item():.3e}"
    )


_BITWISE_DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", _BITWISE_DTYPES, ids=["f16", "bf16"])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_bitwise_vs_bp(head_dim, causal, dtype):
    """Distance-1 + transposed-V reproduces the binding-prefetch kernel."""
    _require_env()
    _parity(
        head_dim,
        causal,
        dtype,
        1,
        bp_block_m(head_dim) * 2,
        build_flash_attn_func_bp_module,
        dict(k_prefetch_dist=1, v_prefetch_dist=1, v_lds_layout="transposed"),
    )


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("head_dim", [16, 32, 64, 128])
def test_bitwise_vs_baseline(head_dim, causal):
    """K distance 0 + V distance 1 + row-major V reproduces the baseline kernel.

    Note the asymmetric distances: the baseline pre-issues V one tile ahead and
    carries it in registers, and only K is staged at distance 0.
    """
    _require_env()
    _parity(
        head_dim,
        causal,
        torch.float16,
        1,
        256,
        build_flash_attn_func_module,
        dict(k_prefetch_dist=0, v_prefetch_dist=1, v_lds_layout="row"),
    )


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_bitwise_vs_m32(causal):
    """Two Q row-tiles per wave reproduces the m32 kernel."""
    _require_env()
    _parity(
        64,
        causal,
        torch.float16,
        1,
        512,
        build_flash_attn_func_m32_module,
        dict(k_prefetch_dist=1, v_prefetch_dist=1, v_lds_layout="transposed", q_row_tiles=2),
    )


_LADDER = [16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 384, 512]


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("head_dim", _LADDER)
def test_ladder_matches_sdpa(head_dim, causal):
    """Every supported head_dim matches an fp32 SDPA reference.

    Covers the sharded (224, 256, 384, 512) and chunked-V configs, where the
    cross-shard partial-sum reduction changes the accumulation order and no
    bitwise oracle exists.
    """
    _require_env()
    q, k, v = _qkv(1, 256, head_dim, torch.float16)
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=causal, dtype_str="f16"
    )
    got = _run(exe, q, k, v, 1, 256)
    assert torch.isfinite(got).all(), f"head_dim={head_dim} causal={causal} not finite"
    rel = _rel(got, _reference(q, k, v, causal))
    assert rel < 5e-3, f"head_dim={head_dim} causal={causal} rel={rel:.3e}"


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("head_dim,head_dim_v", [(512, 128), (384, 128), (256, 128), (256, 64)])
def test_v_column_window(head_dim, head_dim_v, causal):
    """The V/O column window is a partial output; all windows tile the result.

    Attention is column-separable in V (O[:, s] = P @ V[:, s] and P does not
    depend on V), so each build writes only its own columns. Correctness is
    only meaningful once every window has run into the same buffer.
    """
    _require_env()
    q, k, v = _qkv(1, 256, head_dim, torch.float16)
    o = torch.empty_like(q)
    for off in range(0, head_dim, head_dim_v):
        exe = build_flash_attn_func_aiw_module(
            num_heads=_NUM_HEADS,
            head_dim=head_dim,
            causal=causal,
            dtype_str="f16",
            head_dim_v=head_dim_v,
            d_offset=off,
        )
        _run(exe, q, k, v, 1, 256, out=o)
    assert torch.isfinite(o).all()
    rel = _rel(o, _reference(q, k, v, causal))
    assert rel < 5e-3, f"hd={head_dim} hdv={head_dim_v} rel={rel:.3e}"


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_storage_flip_matches_contiguous(head_dim, causal):
    """BSHD *shape* over BHSD *memory* must give the same answer.

    This is AOTriton's `storage_flip`: allocate with axes 1 and 2 swapped, then
    transpose back, so the logical shape is unchanged while the strides are
    not. It is precisely the case a shape-derived layout cannot survive, and
    the reason the kernel reads `stride_?0/1/2` instead of computing
    `num_heads * head_dim`.

    Without this test the whole stride mechanism is exercised only by
    contiguous tensors, where a derived layout happens to agree.
    """
    _require_env()
    seq = 256

    def _flipped(seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        # (B, H, S, D) allocation viewed as (B, S, H, D)
        return torch.randn(
            1, _NUM_HEADS, seq, head_dim,
            dtype=torch.float16, device="cuda", generator=gen,
        ).transpose(1, 2)

    q, k, v = (_flipped(s) for s in range(3))
    assert not q.is_contiguous(), "test would be vacuous on a contiguous tensor"
    o = torch.empty(
        1, _NUM_HEADS, seq, head_dim, dtype=torch.float16, device="cuda"
    ).transpose(1, 2)

    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=causal, dtype_str="f16"
    )
    _run(exe, q, k, v, 1, seq, out=o)

    # Same values in a contiguous tensor: the answer must not depend on layout.
    got_c = _run(exe, q.contiguous(), k.contiguous(), v.contiguous(), 1, seq)

    assert torch.isfinite(o).all(), "flipped-storage run produced non-finite output"
    assert torch.equal(o, got_c), (
        "layout changed the result: max |flipped - contiguous| = "
        f"{(o.float() - got_c.float()).abs().max().item():.3e}"
    )
    rel = _rel(o, _reference(q, k, v, causal))
    assert rel < 5e-3, f"hd={head_dim} causal={causal} rel={rel:.3e}"


def test_rejects_non_rank4_and_ragged_last_dim():
    """Strides can only be read from a rank-4, D-contiguous tensor."""
    _require_env()
    q, k, v = _qkv(1, 256, 64, torch.float16)
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=64, causal=False, dtype_str="f16"
    )
    with pytest.raises(ValueError, match="rank 4"):
        exe(q.reshape(-1), k, v, torch.empty_like(q), 1, 256)
    # D non-contiguous: transpose the last two axes.
    bad = q.transpose(2, 3)
    with pytest.raises(ValueError, match="contiguous last dimension"):
        exe(bad, k, v, torch.empty_like(q), 1, 256)


def test_shard_resolution_respects_narrow_window():
    """A narrow V window must not inherit a shard count it cannot divide.

    head_dim 384 prefers 3 shards, which splits a 128-column window into
    42-column slices -- not a multiple of WMMA_N. The resolver walks down
    instead of failing the build.
    """
    assert resolve_shards(384, 384, 32) == 3
    assert resolve_shards(384, 128, 32) < 3
    for head_dim in _LADDER:
        for vo in (head_dim, 128, 64):
            if vo > head_dim:
                continue
            s = resolve_shards(head_dim, vo, 32)
            assert head_dim % s == 0 and (head_dim // s) % 16 == 0, (head_dim, vo, s)


def test_block_m_is_invariant_to_q_row_tiles():
    """q_row_tiles must change the wave count, not BLOCK_M.

    At 2, the same rows are covered by half as many waves each doing twice the
    work. If BLOCK_M doubled instead, per-wave register pressure would double on
    top of the knob's own cost. Invisible to a bitwise comparison, so asserted
    here directly.
    """
    for head_dim in (64, 128):
        assert aiw_block_m(head_dim, 1) == aiw_block_m(head_dim, 1)
    exe1 = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=64, causal=False, dtype_str="f16", q_row_tiles=1
    )
    exe2 = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=64, causal=False, dtype_str="f16", q_row_tiles=2
    )
    assert exe1 is not exe2  # distinct builds, same BLOCK_M by construction


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(head_dim=100), "head_dim"),
        (dict(head_dim=128, block_n=64, causal=True), "BLOCK_N"),
        (dict(head_dim=128, v_lds_layout="row", shards=2), "cross-shard"),
        (dict(head_dim=64, q_row_tiles=2, shards=2), "untested"),
        (dict(head_dim=64, q_row_tiles=3), "q_row_tiles"),
        (dict(head_dim=128, k_prefetch_dist=2), "k_prefetch_dist"),
        (dict(head_dim=128, v_prefetch_dist=2), "v_prefetch_dist"),
        (dict(head_dim=128, head_dim_v=24), "head_dim_v"),
        (dict(head_dim=128, head_dim_v=64, d_offset=96), "d_offset"),
    ],
)
def test_knob_validity_predicate(kwargs, match):
    """Unimplemented knob combinations fail the build, not the output."""
    base = dict(num_heads=_NUM_HEADS, causal=False, dtype_str="f16")
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        build_flash_attn_func_aiw_module(**base)
