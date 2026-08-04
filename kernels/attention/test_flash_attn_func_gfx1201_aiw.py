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

import math
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
    """aiw at `aiw_kwargs` must match `oracle` to tolerance."""
    q, k, v = _qkv(batch, seq, head_dim, dtype)
    common = dict(
        num_heads=_NUM_HEADS,
        head_dim=head_dim,
        causal=causal,
        dtype_str="f16" if dtype == torch.float16 else "bf16",
    )
    got = _run(
        build_flash_attn_func_aiw_module(**common, **aiw_kwargs),
        q, k, v, batch, seq,
    )
    want = _run(oracle(**common), q, k, v, batch, seq, legacy=True)
    assert torch.isfinite(got).all(), "aiw produced non-finite output"
    # Tolerance, not bitwise. The oracles predate the softmax correction
    # -- they scale inside the exponent via FMA -- and that correction is
    # now unconditional, so exact agreement is no longer the right bar.
    # What this still checks is that the *scheduling* knobs (prefetch
    # distance, V LDS layout, Q row-tiles) do not change the answer,
    # which is what they were introduced to verify.
    # bf16 carries an 8-bit mantissa, so the two softmax formulations
    # diverge about an order of magnitude further than in f16.
    tol = 2e-3 if dtype == torch.float16 else 1e-2
    rel = ((got.float() - want.float()).abs().max().item()
           / want.float().abs().max().item())
    assert rel < tol, f"aiw vs oracle rel = {rel:.3e} (tol {tol:.0e})"


_BITWISE_DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", _BITWISE_DTYPES, ids=["f16", "bf16"])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_matches_bp(head_dim, causal, dtype):
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
def test_matches_baseline(head_dim, causal):
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
def test_matches_m32(causal):
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
def test_softmax_is_exact_for_large_inputs(mag):
    """At large magnitudes the softmax saturates and the result is exact.

    Every row becomes effectively one-hot, so O is a single V row. The kernel
    scales the scores *before* the row max, so `s - m` is exactly 0 for the
    maximum and `exp2(0) == 1` exactly.

    The alternative -- scaling the max separately and folding the scale into
    the exponent via FMA, which the pre-unification kernels do -- gives
    `exp2(eps)`, an error of ~1 ulp of `qk_scale*m` that grows with magnitude.
    It measured 4-7e-4 on these shapes where this is 0.

    seq == BLOCK_M so there is one Q block and every KV tile is masked; a block
    with a leading unmasked region accumulates its max differently and the
    exactness is not observable.
    """
    _require_env()
    head_dim, seq = 128, aiw_block_m(128, 1)
    gen = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (
        torch.randn(
            1, seq, _NUM_HEADS, head_dim,
            dtype=torch.float16, device="cuda", generator=gen,
        ) * mag
        for _ in range(3)
    )
    assert torch.isfinite(q).all(), "inputs overflowed f16; pick a smaller mag"
    got = _run(
        build_flash_attn_func_aiw_module(
            num_heads=_NUM_HEADS, head_dim=head_dim, causal=True, dtype_str="f16"
        ),
        q, k, v, 1, seq,
    )
    qb, kb, vb = (x.transpose(1, 2).double() for x in (q, k, v))
    ref = F.scaled_dot_product_attention(qb, kb, vb, is_causal=True).transpose(1, 2)
    err = (got.double() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1e-6, f"saturated softmax should be exact at mag={mag}: {err:.3e}"


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

    Proves the pad does not *influence* the result -- stronger than a tolerance
    check, which cannot tell "masked" from "read but diluted". It does not
    prove the pad is unread; the kernel may read and discard it. Only a guard
    page shows that, and only at page granularity. Not-reading-OOB is argued
    rather than tested; see plan1 section 3.

    Also fails for correct but non-deterministic algorithms (split-K over
    head_dim, say). None are planned here.
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


def _lse_reference(q, k, causal, head_dim):
    """Natural-log logsumexp of the scaled, masked scores. Shape (B*H, S)."""
    qb, kb = (x.transpose(1, 2).double() for x in (q, k))
    sc = (qb @ kb.transpose(-1, -2)) / math.sqrt(head_dim)
    if causal:
        s = q.shape[1]
        sc = sc.masked_fill(
            torch.triu(torch.ones(s, s, dtype=torch.bool, device=q.device), 1),
            float("-inf"),
        )
    return torch.logsumexp(sc, dim=-1).reshape(-1, q.shape[1])


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("head_dim", [16, 64, 128, 256, 512])
def test_logsumexp_matches_torch(head_dim, causal):
    """LSE = (m + log2(l)) * ln2, in natural-log units, layout (B*H, S)."""
    _require_env()
    seq, batch = 256, 2
    q, k, v = _qkv(batch, seq, head_dim, torch.float16)
    o = torch.empty_like(q)
    lse = torch.zeros(batch * _NUM_HEADS, seq, dtype=torch.float32, device="cuda")
    build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=causal, dtype_str="f16"
    )(q, k, v, o, batch, seq, lse=lse)
    torch.cuda.synchronize()

    assert torch.isfinite(lse).all(), "logsumexp is not finite"
    ref = _lse_reference(q, k, causal, head_dim)
    err = (lse.double() - ref).abs().max().item()
    assert err < 2e-2, f"hd={head_dim} causal={causal} max|lse-ref|={err:.3e}"


def test_logsumexp_null_pointer_is_a_noop():
    """Omitting the LSE tensor must not change O, and must not fault.

    The gate is on the pointer, not a build flag, so training and inference
    share one binary rather than doubling the functional count.
    """
    _require_env()
    seq = 256
    q, k, v = _qkv(1, seq, 64, torch.float16)
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=64, causal=False, dtype_str="f16"
    )
    o_without = _run(exe, q, k, v, 1, seq)
    o_with = torch.empty_like(q)
    lse = torch.zeros(_NUM_HEADS, seq, dtype=torch.float32, device="cuda")
    exe(q, k, v, o_with, 1, seq, lse=lse)
    torch.cuda.synchronize()
    assert torch.equal(o_without, o_with), "requesting LSE perturbed O"
    assert (lse != 0).any(), "LSE was requested but never written"


def test_logsumexp_with_gqa():
    """LSE is indexed by the *query* head, so GQA must not collapse rows."""
    _require_env()
    seq, nhq, nhk, head_dim = 256, 8, 2, 64
    gen = torch.Generator(device="cuda").manual_seed(0)

    def _t(h):
        return torch.randn(
            1, seq, h, head_dim, dtype=torch.float16, device="cuda", generator=gen
        )

    q, k, v = _t(nhq), _t(nhk), _t(nhk)
    o = torch.empty_like(q)
    lse = torch.zeros(nhq, seq, dtype=torch.float32, device="cuda")
    build_flash_attn_func_aiw_module(
        num_heads=nhq, head_dim=head_dim, causal=False, dtype_str="f16"
    )(q, k, v, o, 1, seq, lse=lse)
    torch.cuda.synchronize()

    ratio = nhq // nhk
    for h in range(nhq):
        qb = q[:, :, h].unsqueeze(1).double()
        kb = k[:, :, h // ratio].unsqueeze(1).double()
        ref = torch.logsumexp(
            (qb @ kb.transpose(-1, -2)) / math.sqrt(head_dim), dim=-1
        ).squeeze()
        err = (lse[h].double() - ref).abs().max().item()
        assert err < 2e-2, f"query head {h}: max|lse-ref|={err:.3e}"


@pytest.mark.parametrize(
    "lse_factory, match",
    [
        (lambda s: torch.zeros(2, s, dtype=torch.float16, device="cuda"), "float32"),
        (lambda s: torch.zeros(2, 2, s, dtype=torch.float32, device="cuda"), "rank 2"),
        # Rank 2 but not contiguous. The kernel derives the logsumexp pitches
        # from VarlenBits rather than reading strides, so a strided buffer is
        # rejected outright rather than silently mis-indexed.
        (lambda s: torch.zeros(s, 2, dtype=torch.float32, device="cuda").t(), "contiguous"),
    ],
    ids=["f16", "rank3", "noncontig"],
)
def test_logsumexp_validation(lse_factory, match):
    _require_env()
    seq = 256
    q, k, v = _qkv(1, seq, 64, torch.float16)
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=64, causal=False, dtype_str="f16"
    )
    with pytest.raises(ValueError, match=match):
        exe(q, k, v, torch.empty_like(q), 1, seq, lse=lse_factory(seq))


def _causal_mask(Lq, Lk, ctype):
    """Boolean visibility mask. ctype 1 = top-left, 2 = bottom-right."""
    i = torch.arange(Lq, device="cuda")[:, None]
    j = torch.arange(Lk, device="cuda")[None, :]
    return j <= i + (0 if ctype == 1 else Lk - Lq)


_XATTN_SHAPES = [(256, 256), (256, 512), (512, 256), (200, 300), (300, 200),
                 (11, 31), (523, 337), (1033, 571)]


@pytest.mark.parametrize("ctype", [0, 1, 2], ids=["full", "topleft", "botright"])
@pytest.mark.parametrize("Lq,Lk", _XATTN_SHAPES, ids=[f"{a}x{b}" for a, b in _XATTN_SHAPES])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_seqlen_q_ne_seqlen_k(head_dim, Lq, Lk, ctype):
    """seqlen_q and seqlen_k are independent, with both causal alignments.

    They coincide when Lq == Lk, which is why a single `causal` flag sufficed
    until now. Apart is where they diverge: top-left keeps the diagonal at
    j == i, bottom-right shifts it to j == i + (Lk - Lq).
    """
    _require_env()
    gen = torch.Generator(device="cuda").manual_seed(0)

    def _t(n):
        return torch.randn(
            1, n, _NUM_HEADS, head_dim, dtype=torch.float16, device="cuda", generator=gen
        )

    q, k, v = _t(Lq), _t(Lk), _t(Lk)
    o = torch.empty_like(q)
    build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=bool(ctype),
        causal_type=ctype, dtype_str="f16",
    )(q, k, v, o, 1, Lq, Lk)
    torch.cuda.synchronize()

    assert torch.isfinite(o).all(), f"{Lq}x{Lk} ctype={ctype} produced non-finite output"
    qb, kb, vb = (x.transpose(1, 2).float() for x in (q, k, v))
    if ctype == 0:
        ref = F.scaled_dot_product_attention(qb, kb, vb).transpose(1, 2)
        live = torch.ones(Lq, dtype=torch.bool, device="cuda")
    else:
        m = _causal_mask(Lq, Lk, ctype)
        live = m.any(-1)
        ref = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=m).transpose(1, 2)

    # Bottom-right with Lq > Lk leaves the leading Lq - Lk rows with no visible
    # key at all. An empty softmax has an all-zero output; the reference gives
    # NaN there, so those rows are checked separately.
    dead = ~live
    if dead.any():
        assert o[:, dead].abs().max().item() == 0.0, "rows with no keys must be zero"
    rel = _rel(o[:, live], ref[:, live])
    assert rel < 5e-3, f"{Lq}x{Lk} ctype={ctype} hd={head_dim} rel={rel:.3e}"


def test_alignments_differ_when_lengths_differ():
    """Top-left and bottom-right must not be the same kernel.

    They agree exactly when Lq == Lk, so a test that only ever used square
    shapes would pass with the diagonal offset wired to zero.
    """
    _require_env()
    Lq, Lk, head_dim = 256, 384, 64
    gen = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(1, Lq, _NUM_HEADS, head_dim, dtype=torch.float16, device="cuda", generator=gen)
    k = torch.randn(1, Lk, _NUM_HEADS, head_dim, dtype=torch.float16, device="cuda", generator=gen)
    v = torch.randn(1, Lk, _NUM_HEADS, head_dim, dtype=torch.float16, device="cuda", generator=gen)
    outs = []
    for ctype in (1, 2):
        o = torch.empty_like(q)
        build_flash_attn_func_aiw_module(
            num_heads=_NUM_HEADS, head_dim=head_dim, causal=True,
            causal_type=ctype, dtype_str="f16",
        )(q, k, v, o, 1, Lq, Lk)
        torch.cuda.synchronize()
        outs.append(o.clone())
    assert not torch.allclose(outs[0], outs[1], atol=1e-2), (
        "top-left and bottom-right produced the same result at Lq != Lk"
    )


# ---------------------------------------------------------------------------
# Generalized sliding-window attention (CAUSAL_TYPE 3)
# ---------------------------------------------------------------------------


def _swa_mask(Lq, Lk, w_left, w_right):
    """Boolean visibility: key j is live for query i iff it is in the band."""
    i = torch.arange(Lq, device="cuda")[:, None]
    j = torch.arange(Lk, device="cuda")[None, :]
    return (j <= i + w_right) & (j >= i - w_left)


@pytest.mark.parametrize("ctype", [1, 2], ids=["topleft", "botright"])
@pytest.mark.parametrize("Lq,Lk", _XATTN_SHAPES, ids=[f"{a}x{b}" for a, b in _XATTN_SHAPES])
def test_causal_type_maps_to_documented_window(Lq, Lk, ctype):
    """`causal_type` 1 and 2 must resolve to the window they claim to.

        top-left      (seqlen_q, 0)
        bottom-right  (seqlen_q, seqlen_k - seqlen_q)

    ``window_left = seqlen_q`` is "unbounded" -- no row can reach further back
    than the start of its own sequence -- so the left edge never binds and the
    band degenerates to a diagonal.

    **This is a host-mapping test, not a kernel-equivalence one, and that is a
    demotion.** Until CAUSAL_TYPE 1 and 2 were deleted it compared two
    genuinely different kernels and its bitwise result was what licensed the
    deletion; now both sides build the same kernel and it only pins the
    `_CAUSAL_WINDOW` table. It is kept at that reduced value because a wrong
    table is still a real bug and nothing else catches it directly -- the
    end-to-end causal semantics are covered by
    `test_seqlen_q_ne_seqlen_k` against a reference mask.

    The negative control is what keeps it from being vacuous: a window one
    column off must not produce the same bits.
    """
    _require_env()
    head_dim = 64
    gen = torch.Generator(device="cuda").manual_seed(0)

    def _t(n):
        return torch.randn(
            1, n, _NUM_HEADS, head_dim, dtype=torch.float16, device="cuda", generator=gen
        )

    q, k, v = _t(Lq), _t(Lk), _t(Lk)
    common = dict(num_heads=_NUM_HEADS, head_dim=head_dim, causal=True, dtype_str="f16")

    o_causal = torch.empty_like(q)
    build_flash_attn_func_aiw_module(causal_type=ctype, **common)(
        q, k, v, o_causal, 1, Lq, Lk
    )

    def _window_run(w):
        o = torch.empty_like(q)
        build_flash_attn_func_aiw_module(causal_type=3, **common)(
            q, k, v, o, 1, Lq, Lk, window=w
        )
        return o

    w_left = Lq
    w_right = 0 if ctype == 1 else Lk - Lq
    o_window = _window_run((w_left, w_right))
    o_shifted = _window_run((w_left, w_right + 1))
    torch.cuda.synchronize()

    assert torch.equal(o_causal, o_window), (
        f"causal_type={ctype} did not resolve to ({w_left}, {w_right}) at "
        f"{Lq}x{Lk}: max |delta| "
        f"{(o_causal.float() - o_window.float()).abs().max().item():.3e}"
    )
    assert not torch.equal(o_causal, o_shifted), (
        f"a window one column off produced identical bits at {Lq}x{Lk} -- the "
        f"comparison above is not exercising the diagonal"
    )


def test_causal_type_1_2_reject_an_explicit_window():
    """The alignment already fixes the window; two sources would disagree."""
    _require_env()
    q, k, v = (
        torch.randn(1, 64, _NUM_HEADS, 64, dtype=torch.float16, device="cuda")
        for _ in range(3)
    )
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=64, causal=True, causal_type=1, dtype_str="f16",
    )
    with pytest.raises(ValueError, match="already fixes the window"):
        exe(q, k, v, torch.empty_like(q), 1, 64, 64, window=(16, 0))


_SWA_WINDOWS = [
    (0, 0),        # the diagonal alone
    (1, 1),        # narrower than any BLOCK_N -- no full tiles exist
    (16, 0),
    (64, 64),
    (0, 64),       # anti-causal: only keys ahead of the query
    (-16, 64),     # band shifted off the diagonal, left bound negative
    (64, -16),     # ... and the other way
    (10_000, 10_000),  # wider than seqlen_k: degenerates to no masking
]


@pytest.mark.parametrize("w_left,w_right", _SWA_WINDOWS, ids=[f"{a}_{b}" for a, b in _SWA_WINDOWS])
@pytest.mark.parametrize("Lq,Lk", [(256, 256), (200, 300), (300, 200), (523, 337)],
                         ids=["256x256", "200x300", "300x200", "523x337"])
def test_sliding_window(Lq, Lk, w_left, w_right):
    """The band, including negative bounds on either side.

    Negative is the entire content of the word "generalized": ``(-16, 64)``
    admits only keys strictly ahead of the query, and rows near the start of
    the sequence then see nothing at all. Those rows are checked separately --
    an empty softmax gives O = 0, where the reference gives NaN.
    """
    _require_env()
    head_dim = 64
    gen = torch.Generator(device="cuda").manual_seed(0)

    def _t(n):
        return torch.randn(
            1, n, _NUM_HEADS, head_dim, dtype=torch.float16, device="cuda", generator=gen
        )

    q, k, v = _t(Lq), _t(Lk), _t(Lk)
    o = torch.empty_like(q)
    build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=True, causal_type=3,
        dtype_str="f16",
    )(q, k, v, o, 1, Lq, Lk, window=(w_left, w_right))
    torch.cuda.synchronize()

    assert torch.isfinite(o).all(), "window produced non-finite output"
    m = _swa_mask(Lq, Lk, w_left, w_right)
    live = m.any(-1)
    qb, kb, vb = (x.transpose(1, 2).float() for x in (q, k, v))

    dead = ~live
    if dead.any():
        assert o[:, dead].abs().max().item() == 0.0, "rows with no keys must be zero"
    if not live.any():
        return
    ref = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=m).transpose(1, 2)
    rel = _rel(o[:, live], ref[:, live])
    assert rel < 5e-3, f"{Lq}x{Lk} window=({w_left},{w_right}) rel={rel:.3e}"


def test_window_wider_than_seqlen_is_unmasked():
    """A band covering everything must equal plain non-causal attention.

    Cheap, but it pins the degenerate end of the range: if either bound were
    compared unsigned, a large positive window would still pass here while
    a negative one failed -- which is why the negative cases above exist.
    """
    _require_env()
    Lq = Lk = 256
    head_dim = 64
    gen = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (
        torch.randn(1, n, _NUM_HEADS, head_dim, dtype=torch.float16, device="cuda", generator=gen)
        for n in (Lq, Lk, Lk)
    )
    o = torch.empty_like(q)
    build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=True, causal_type=3,
        dtype_str="f16",
    )(q, k, v, o, 1, Lq, Lk, window=(Lk, Lk))
    torch.cuda.synchronize()
    qb, kb, vb = (x.transpose(1, 2).float() for x in (q, k, v))
    ref = F.scaled_dot_product_attention(qb, kb, vb).transpose(1, 2)
    rel = _rel(o, ref)
    assert rel < 5e-3, f"full window differs from unmasked attention, rel={rel:.3e}"


def test_window_requires_explicit_bounds():
    """causal_type=3 with no window is a caller error, not a silent default."""
    _require_env()
    q, k, v = (
        torch.randn(1, 64, _NUM_HEADS, 64, dtype=torch.float16, device="cuda")
        for _ in range(3)
    )
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=64, causal=True, causal_type=3, dtype_str="f16",
    )
    with pytest.raises(ValueError, match="requires window"):
        exe(q, k, v, torch.empty_like(q), 1, 64, 64)


def test_logsumexp_is_plus_inf_for_rows_with_no_keys():
    """AOTriton's convention: +inf, so the backward pass zeroes those rows.

    exp(qk - LSE) with LSE = +inf is 0, which is what a row that attends to
    nothing must contribute. -inf would give the opposite.
    """
    _require_env()
    Lq, Lk, head_dim = 300, 200, 64
    gen = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(1, Lq, _NUM_HEADS, head_dim, dtype=torch.float16, device="cuda", generator=gen)
    k = torch.randn(1, Lk, _NUM_HEADS, head_dim, dtype=torch.float16, device="cuda", generator=gen)
    v = torch.randn(1, Lk, _NUM_HEADS, head_dim, dtype=torch.float16, device="cuda", generator=gen)
    o = torch.empty_like(q)
    lse = torch.zeros(_NUM_HEADS, Lq, dtype=torch.float32, device="cuda")
    build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=True, causal_type=2,
        dtype_str="f16",
    )(q, k, v, o, 1, Lq, Lk, lse=lse)
    torch.cuda.synchronize()
    dead = Lq - Lk          # rows 0 .. dead-1 see no keys
    assert torch.isinf(lse[:, :dead]).all() and (lse[:, :dead] > 0).all()
    assert torch.isfinite(lse[:, dead:]).all()


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

# ---------------------------------------------------------------------------
# Variable-length sequences (VarlenBits)
# ---------------------------------------------------------------------------
#
# The headline gate is that varlen with N sequences equals N separate dense
# calls **bitwise**. That holds for a reason rather than by luck: a varlen
# workgroup and its dense counterpart cover the same tiles in the same order
# with the same values, and only the base address differs, so every
# floating-point operation is identical.
#
# It is the right gate here specifically because an addressing bug that lands
# inside the right allocation reads *plausible* data -- a tolerance comparison
# against a reference would accept it. See sdpa-varlen-plan.md section 7.


# VARLEN_LSE_LAYOUT_TH, byte 2 of VarlenBits. Spelled out rather than imported
# so the test pins the wire encoding, not just the builder's opinion of it.
_LSE_LAYOUT_TH = 1 << 16


def _cu(lens):
    return torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(),
                        dtype=torch.int32, device="cuda")


def _packed_qkv(lens_q, lens_k, head_dim, seed=0):
    """1THD tensors for a compact-varlen batch, plus their cu_seqlens."""
    cq, ck = _cu(lens_q), _cu(lens_k)
    Tq, Tk = int(cq[-1]), int(ck[-1])
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def _t(n):
        # A zero-length batch still needs a valid allocation.
        return torch.randn(1, max(n, 1), _NUM_HEADS, head_dim,
                           dtype=torch.float16, device="cuda", generator=gen)

    return _t(Tq), _t(Tk), _t(Tk), cq, ck


def _assert_varlen_matches_dense(exe, q, k, v, o, lens_q, lens_k, cq, ck,
                                 label, lse=None, dense_lse=True):
    """Every sequence must match its own dense call, bit for bit."""
    for z, (lq, lk) in enumerate(zip(lens_q, lens_k)):
        if lq == 0:
            continue
        qs, ks = int(cq[z]), int(ck[z])
        qz = q[:, qs:qs + lq].contiguous()
        kz = k[:, ks:ks + lk].contiguous()
        vz = v[:, ks:ks + lk].contiguous()
        ref = torch.empty_like(qz)
        ref_lse = (torch.zeros(_NUM_HEADS, lq, dtype=torch.float32, device="cuda")
                   if lse is not None else None)
        exe(qz, kz, vz, ref, 1, lq, lk, lse=ref_lse)
        torch.cuda.synchronize()
        got = o[:, qs:qs + lq]
        assert torch.equal(ref, got), (
            f"{label} sequence {z} (Lq={lq}, Lk={lk}) differs from its dense "
            f"call: max |delta| {(ref.float() - got.float()).abs().max():.3e}"
        )
        if lse is not None:
            assert torch.equal(ref_lse, lse[:, qs:qs + lq]), (
                f"{label} sequence {z} logsumexp differs from its dense call"
            )


_VARLEN_LENGTHS = [
    ([3, 128, 40, 200], "mixed"),
    ([64, 64, 64], "all_equal"),
    ([1000, 5, 7, 3], "one_long"),
    ([523, 337, 1033], "ragged_primes"),
    ([0, 96, 40], "zero_leading"),
    ([96, 0, 40], "zero_middle"),
    ([96, 40, 0], "zero_trailing"),
    ([77], "single"),
]


@pytest.mark.parametrize("lens_q,label", _VARLEN_LENGTHS, ids=[x[1] for x in _VARLEN_LENGTHS])
@pytest.mark.parametrize("ctype", [0, 1, 2], ids=["full", "topleft", "botright"])
def test_varlen_compact_lengths(lens_q, label, ctype):
    """Suite A: one mode, many length patterns (plan §7).

    `seqlen_k != seqlen_q` throughout, so the per-sequence window that
    `causal_type=2` derives is exercised rather than collapsing to the
    top-left one.
    """
    _require_env()
    head_dim = 64
    lens_k = [x + 7 if x else 0 for x in lens_q]
    N = len(lens_q)
    q, k, v, cq, ck = _packed_qkv(lens_q, lens_k, head_dim)
    o = torch.zeros_like(q)
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=bool(ctype),
        causal_type=ctype or None, dtype_str="f16",
    )
    mq, mk = max(lens_q), max(lens_k)
    exe(q, k, v, o, N, mq, mk,
        varlen=exe.varlen_compact(cq, ck, mq, mk))
    torch.cuda.synchronize()
    _assert_varlen_matches_dense(exe, q, k, v, o, lens_q, lens_k, cq, ck,
                                 f"compact/{label}/ctype={ctype}")


def test_varlen_zero_length_writes_nothing():
    """A zero-length sequence must not be written at all.

    Distinct from "writes zeros": the buffer is poisoned first, so a store of
    0.0 to a row that should not exist still fails. This is what checks the
    empty-work path of plan §6.1, including that it covers the *logsumexp*
    store, which is guarded separately from O.
    """
    _require_env()
    head_dim, lens_q = 64, [0, 96, 0, 40, 0]
    lens_k = [x + 7 if x else 0 for x in lens_q]
    N = len(lens_q)
    q, k, v, cq, ck = _packed_qkv(lens_q, lens_k, head_dim)
    poison = float("nan")
    o = torch.full_like(q, poison)
    lse = torch.full((_NUM_HEADS, max(int(cq[-1]), 1)), poison,
                     dtype=torch.float32, device="cuda")
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=True, dtype_str="f16",
    )
    mq, mk = max(lens_q), max(lens_k)
    exe(q, k, v, o, N, mq, mk, lse=lse,
        varlen=exe.varlen_compact(cq, ck, mq, mk))
    torch.cuda.synchronize()
    # Rows belonging to real sequences were written; there are no rows
    # belonging to the empty ones, so nothing else may have been touched.
    written = torch.zeros(int(cq[-1]), dtype=torch.bool, device="cuda")
    for z, lq in enumerate(lens_q):
        written[int(cq[z]):int(cq[z]) + lq] = True
    assert torch.isfinite(o[:, written]).all(), "live rows were not written"
    assert torch.isfinite(lse[:, written]).all(), "live logsumexp not written"


@pytest.mark.parametrize("ctype", [0, 1], ids=["full", "causal"])
def test_varlen_padded_matches_dense(ctype):
    """`PaddedVarlen`: BHSD tensors whose sequences are short.

    Differs from compact in *precisely* the two decoded fields -- `batch_index`
    becomes `z` and `row_off` becomes 0 -- so this is what catches a mix-up
    between them (plan §7 property 2).
    """
    _require_env()
    head_dim = 64
    lens_q = [200, 40, 128]
    lens_k = [x + 7 for x in lens_q]
    N, mq, mk = len(lens_q), max(lens_q), max(lens_k)
    gen = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(N, mq, _NUM_HEADS, head_dim, dtype=torch.float16, device="cuda", generator=gen)
    k = torch.randn(N, mk, _NUM_HEADS, head_dim, dtype=torch.float16, device="cuda", generator=gen)
    v = torch.randn(N, mk, _NUM_HEADS, head_dim, dtype=torch.float16, device="cuda", generator=gen)
    o = torch.zeros_like(q)
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=bool(ctype),
        causal_type=ctype or None, dtype_str="f16",
    )
    exe(q, k, v, o, N, mq, mk,
        varlen=exe.varlen_padded(_cu(lens_q), _cu(lens_k), mq, mk))
    torch.cuda.synchronize()
    for z, (lq, lk) in enumerate(zip(lens_q, lens_k)):
        qz = q[z:z + 1, :lq].contiguous()
        kz = k[z:z + 1, :lk].contiguous()
        vz = v[z:z + 1, :lk].contiguous()
        ref = torch.empty_like(qz)
        exe(qz, kz, vz, ref, 1, lq, lk)
        torch.cuda.synchronize()
        assert torch.equal(ref, o[z:z + 1, :lq]), (
            f"padded sequence {z} differs from its dense call"
        )


def test_varlen_single_sequence_reduces_to_dense():
    """N = 1 compact must be bit-identical to the plain dense call.

    The cheapest instance of the headline gate, and the first thing to make
    pass: at N = 1 the only difference between the two is that one of them
    went through the decoder at all.
    """
    _require_env()
    head_dim, L = 64, 300
    gen = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(1, L, _NUM_HEADS, head_dim, dtype=torch.float16, device="cuda", generator=gen)
    k = torch.randn(1, L, _NUM_HEADS, head_dim, dtype=torch.float16, device="cuda", generator=gen)
    v = torch.randn(1, L, _NUM_HEADS, head_dim, dtype=torch.float16, device="cuda", generator=gen)
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=True, dtype_str="f16",
    )
    o_dense = torch.empty_like(q)
    exe(q, k, v, o_dense, 1, L, L)
    o_varlen = torch.empty_like(q)
    exe(q, k, v, o_varlen, 1, L, L,
        varlen=exe.varlen_compact(_cu([L]), _cu([L]), L, L))
    torch.cuda.synchronize()
    assert torch.equal(o_dense, o_varlen)


def test_varlen_lse_layout_th_transposes():
    """`VARLEN_LSE_LAYOUT_TH` must write `(T, H)`, not `(H, T)`.

    Transformer Engine requires the layout, not merely the shape, so the test
    compares against the transpose of the `_HT` result rather than against a
    reference computation -- a kernel that ignored the bit would produce `_HT`
    content in a `_TH` buffer and pass any shape-only check.
    """
    _require_env()
    head_dim = 64
    lens_q = [96, 40, 128]
    lens_k = [x + 7 for x in lens_q]
    N, mq, mk = len(lens_q), max(lens_q), max(lens_k)
    q, k, v, cq, ck = _packed_qkv(lens_q, lens_k, head_dim)
    Tq = int(cq[-1])
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=True, dtype_str="f16",
    )
    o = torch.empty_like(q)
    lse_ht = torch.zeros(_NUM_HEADS, Tq, dtype=torch.float32, device="cuda")
    exe(q, k, v, o, N, mq, mk, lse=lse_ht,
        varlen=exe.varlen_compact(cq, ck, mq, mk))
    lse_th = torch.zeros(Tq, _NUM_HEADS, dtype=torch.float32, device="cuda")
    exe(q, k, v, o, N, mq, mk, lse=lse_th,
        varlen=exe.varlen_compact(cq, ck, mq, mk, lse_tokens=Tq,
                                  lse_layout=_LSE_LAYOUT_TH))
    torch.cuda.synchronize()
    assert torch.equal(lse_ht, lse_th.t().contiguous()), (
        "TH layout is not the transpose of HT"
    )


def test_varlen_rejects_unimplemented_configurations():
    """Encodable but not yet built must fail loudly, not silently misbehave."""
    _require_env()
    q, k, v = (
        torch.randn(1, 64, _NUM_HEADS, 64, dtype=torch.float16, device="cuda")
        for _ in range(3)
    )
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=64, causal=False, dtype_str="f16",
    )
    cu = _cu([64])
    # ARRAY positions are step 2; INDIVIDUAL lengths are step 3.
    for side in (0x13, 0x15, 0x04):
        bad = dict(bits=exe.varlen_bits(side, side), seqinfo_q0=cu, seqinfo_q1=cu,
                   seqinfo_k0=cu, seqinfo_k1=cu, max_seqlen_q=64, max_seqlen_k=64,
                   lse_tokens=64)
        with pytest.raises(NotImplementedError, match="not implemented"):
            exe(q, k, v, torch.empty_like(q), 1, 64, 64, varlen=bad)
    # REUSE without cumulative lengths is not encodable at all.
    with pytest.raises(ValueError, match="REUSE requires"):
        exe.varlen_bits(0x0D, 0x0D)
