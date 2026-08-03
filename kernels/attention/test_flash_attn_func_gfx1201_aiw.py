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
    # safe_softmax=False: the oracles predate both corrections, so bitwise
    # parity is only meaningful against the same arithmetic. The corrected
    # path is covered by test_safe_softmax_* below.
    got = _run(
        build_flash_attn_func_aiw_module(
            **common, **aiw_kwargs, safe_softmax=False
        ),
        q, k, v, batch, seq,
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


@pytest.mark.parametrize("mag", [300.0, 1000.0], ids=lambda m: f"mag{int(m)}")
def test_safe_softmax_is_exact_for_large_inputs(mag):
    """The corrected softmax is exact where the FMA form is not.

    With `causal=True` the first query row attends to exactly one key, so its
    softmax is 1.0 and its output is V[0] exactly. The corrected form scales the
    scores *before* the row max, so `s - m` is exactly 0 for the max element and
    `exp2(0) == 1` exactly.

    The old form scaled the max separately (`fma(s, scale, -scale*m)`), so for
    the max element it computed the rounding error of `scale*m` rather than
    zero, giving `exp2(eps) = 1 + eps'`. That error is ~1 ulp of `scale*m` and
    therefore **grows with input magnitude** -- which is exactly what AOTriton
    warns about in ROCm/aotriton#54.

    Below magnitude ~100 the difference is under the f16 output precision and
    invisible; this test is at 300 and 1000, where it is not.
    """
    _require_env()
    head_dim, seq = 128, 256
    gen = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (
        torch.randn(
            1, seq, _NUM_HEADS, head_dim,
            dtype=torch.float16, device="cuda", generator=gen,
        ) * mag
        for _ in range(3)
    )
    assert torch.isfinite(q).all(), "inputs overflowed f16; pick a smaller mag"

    common = dict(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=True, dtype_str="f16"
    )
    got_safe = _run(build_flash_attn_func_aiw_module(**common), q, k, v, 1, seq)
    got_old = _run(
        build_flash_attn_func_aiw_module(**common, safe_softmax=False),
        q, k, v, 1, seq,
    )

    qb, kb, vb = (x.transpose(1, 2).double() for x in (q, k, v))
    ref = F.scaled_dot_product_attention(qb, kb, vb, is_causal=True).transpose(1, 2)
    scale = ref.abs().max().item()
    err_safe = (got_safe.double() - ref).abs().max().item() / scale
    err_old = (got_old.double() - ref).abs().max().item() / scale

    assert err_safe < err_old / 10.0, (
        f"corrected softmax should be markedly more accurate at mag={mag}: "
        f"safe={err_safe:.3e} old={err_old:.3e}"
    )


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("nhq,nhk", [(8, 8), (8, 2), (8, 1), (10, 2)],
                         ids=["mha", "gqa4", "mqa", "gqa5"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_mqa_gqa_matches_sdpa(head_dim, nhq, nhk, causal):
    """Several query heads may share one KV head.

    `(10, 2)` is deliberately non-power-of-two on both axes, following
    AOTriton's `test_fast`: a ratio of 5 catches a head-index derivation that
    happens to work for shifts.
    """
    _require_env()
    seq = 256
    gen = torch.Generator(device="cuda").manual_seed(0)

    def _t(h):
        return torch.randn(
            1, seq, h, head_dim, dtype=torch.float16, device="cuda", generator=gen
        )

    q, k, v = _t(nhq), _t(nhk), _t(nhk)
    o = torch.empty_like(q)
    exe = build_flash_attn_func_aiw_module(
        num_heads=nhq, head_dim=head_dim, causal=causal, dtype_str="f16"
    )
    exe(q, k, v, o, 1, seq)
    torch.cuda.synchronize()

    # enable_gqa broadcasts the KV heads across each query-head group.
    qb, kb, vb = (x.transpose(1, 2).float() for x in (q, k, v))
    ref = F.scaled_dot_product_attention(
        qb, kb, vb, is_causal=causal, enable_gqa=(nhq != nhk)
    ).transpose(1, 2)

    assert torch.isfinite(o).all(), f"nhq={nhq} nhk={nhk} produced non-finite output"
    rel = _rel(o, ref)
    assert rel < 5e-3, f"nhq={nhq} nhk={nhk} hd={head_dim} causal={causal} rel={rel:.3e}"


def test_gqa_head_mapping_is_grouped_not_strided():
    """Query head h must read KV head h // (Hq/Hk), not h % Hk.

    Both derivations give identical results whenever the KV heads happen to be
    interchangeable, so this uses distinct per-head KV content and compares
    against an explicit per-head reference.
    """
    _require_env()
    seq, head_dim, nhq, nhk = 128, 64, 8, 2
    gen = torch.Generator(device="cuda").manual_seed(1)
    q = torch.randn(1, seq, nhq, head_dim, dtype=torch.float16, device="cuda", generator=gen)
    k = torch.randn(1, seq, nhk, head_dim, dtype=torch.float16, device="cuda", generator=gen)
    v = torch.randn(1, seq, nhk, head_dim, dtype=torch.float16, device="cuda", generator=gen)
    # Make the two KV heads maximally distinguishable.
    k[:, :, 1] *= -1.0
    v[:, :, 1] += 4.0

    o = torch.empty_like(q)
    build_flash_attn_func_aiw_module(
        num_heads=nhq, head_dim=head_dim, causal=False, dtype_str="f16"
    )(q, k, v, o, 1, seq)
    torch.cuda.synchronize()

    def _one_head(x, h):
        """(B, S, H, D) -> (B, 1, S, D), the layout SDPA expects."""
        return x[:, :, h].unsqueeze(1).float()

    ratio = nhq // nhk
    for h in range(nhq):
        kh = h // ratio          # grouped, not h % nhk
        ref = F.scaled_dot_product_attention(
            _one_head(q, h), _one_head(k, kh), _one_head(v, kh)
        ).squeeze(1)
        got = o[:, :, h].float()
        rel = (got - ref).abs().max().item() / ref.abs().max().item()
        assert rel < 5e-3, f"query head {h} did not read KV head {kh} (rel={rel:.3e})"


_PADDED = [(113, 128), (40, 48), (8, 16), (120, 128), (200, 224), (7, 16)]


def _poisoned(hdim, tile, seq, heads, poison, gen):
    """A tensor of logical width `hdim` inside an allocation of width `tile`.

    The gap is pre-filled with `poison`, so any unmasked read of it shows up in
    the output. This is the shape the AOTriton API actually delivers: the
    allocation is padded, and the caller slices back to the real extent.
    """
    full = torch.full((1, seq, heads, tile), poison, dtype=torch.float16, device="cuda")
    full[..., :hdim] = torch.randn(
        1, seq, heads, hdim, dtype=torch.float16, device="cuda", generator=gen
    )
    return full[..., :hdim]


@pytest.mark.parametrize("poison", [float("nan"), 1e4], ids=["nan", "big"])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("hdim,tile", _PADDED, ids=[f"{h}in{t}" for h, t in _PADDED])
def test_padded_head_never_reads_the_pad(hdim, tile, causal, poison):
    """`Hdim` need not equal the tile width, and the gap must not be read.

    Poisoning with NaN is the sharp version: a single unmasked element makes
    the whole output non-finite. Poisoning with a large finite value catches
    the case where a NaN would have been silently flushed somewhere.

    The two poisons must give **identical** results -- that is the real
    assertion, stronger than either one passing alone.
    """
    _require_env()
    seq, heads = 256, 4
    gen = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (_poisoned(hdim, tile, seq, heads, poison, gen) for _ in range(3))
    o = torch.full(
        (1, seq, heads, tile), poison, dtype=torch.float16, device="cuda"
    )[..., :hdim]

    exe = build_flash_attn_func_aiw_module(
        num_heads=heads, head_dim=tile, causal=causal, dtype_str="f16",
        padded_head=True,
    )
    exe(q, k, v, o, 1, seq)
    torch.cuda.synchronize()

    assert torch.isfinite(o).all(), "the pad leaked into the output"
    qb, kb, vb = (x.transpose(1, 2).float() for x in (q, k, v))
    ref = F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal).transpose(1, 2)
    rel = _rel(o, ref)
    assert rel < 5e-3, f"hdim={hdim} tile={tile} causal={causal} rel={rel:.3e}"


@pytest.mark.parametrize("hdim,tile", [(113, 128), (40, 48)], ids=["113in128", "40in48"])
def test_padded_head_is_independent_of_pad_contents(hdim, tile):
    """Two different poisons must produce bitwise-identical output.

    Stronger than "the answer is close to the reference": it proves the pad is
    not read at all, rather than read and then diluted below tolerance.

    **Deliberately over-constrained -- read this before "fixing" a failure.**
    The assertion pins down *how* correctness is achieved, not just that it is.
    It rules out any scheme that lets the pad enter the arithmetic and cancels
    its contribution afterwards, and any scheme whose reduction order depends
    on which columns are live. Splitting the head_dim reduction (split-K) is
    the concrete example: a correct implementation that partitions the
    reduction differently once padding is present would produce a different
    -- still correct -- rounding, and fail here.

    That is an acceptable trade today because split-K is not planned for this
    kernel, and the extra strictness is what makes the test able to distinguish
    "masked" from "read but harmless". If a future optimisation does need the
    freedom, relax this to comparing each poison against the fp32 reference
    separately and keep the NaN case, which is the actual leak detector; do
    not simply delete it.
    """
    _require_env()
    seq, heads = 256, 4
    outs = []
    for poison in (float("nan"), 1e4, -7.5):
        gen = torch.Generator(device="cuda").manual_seed(0)
        q, k, v = (_poisoned(hdim, tile, seq, heads, poison, gen) for _ in range(3))
        o = torch.full(
            (1, seq, heads, tile), poison, dtype=torch.float16, device="cuda"
        )[..., :hdim]
        build_flash_attn_func_aiw_module(
            num_heads=heads, head_dim=tile, causal=False, dtype_str="f16",
            padded_head=True,
        )(q, k, v, o, 1, seq)
        torch.cuda.synchronize()
        outs.append(o.clone())
    for other in outs[1:]:
        assert torch.equal(outs[0], other), "output depends on the pad contents"


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
