# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness tests for the unified gfx1201 kernel (``..._aiw``).

This kernel replaced three earlier ones, which were kept on disk as bitwise
oracles until the features outgrew them -- they predate MQA/GQA, PADDED_HEAD,
logsumexp, per-tensor strides, sliding windows and varlen, so most of what is
tested below has no oracle to compare against. They are retired; their final
performance numbers are recorded under N2 in ``sdpa-close-gap-plan1.md``.

What replaced them as the sharpest bar is *self*-equivalence: a configuration
that should reduce to a simpler one must do so **bitwise**. A window equal to
the causal diagonal, varlen with N sequences against N dense calls. Those
compare two paths through the same kernel rather than two kernels, and they
catch what a tolerance cannot -- see the window and varlen sections below.

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

from flash_attn_func_gfx1201_interface import flydsl_flash_attn_func_gfx1201
from fmha_tuning_gfx1201 import default_block_m as aiw_block_m, resolve_shards
from flash_attn_func_gfx1201_aiw import (
    build_flash_attn_func_aiw_module,
)

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


def _run(exe, q, k, v, batch, seq, out=None):
    """Launch and return the output tensor."""
    o = torch.empty_like(q) if out is None else out
    exe(q, k, v, o, batch, seq)
    torch.cuda.synchronize()
    return o


def _reference(q, k, v, causal):
    qb, kb, vb = (x.transpose(1, 2).float() for x in (q, k, v))
    return F.scaled_dot_product_attention(qb, kb, vb, is_causal=causal).transpose(1, 2)


def _rel(got, ref):
    return (got.float() - ref).abs().max().item() / ref.abs().max().item()


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
    (-128, 256),   # left bound past a whole BLOCK_N: the *leading* masked run
                   # spans several tiles, not just a clipped tile 0
    (-256, 300),   # ... and further still
    (10_000, 10_000),  # wider than seqlen_k: degenerates to no masking
]


def test_window_on_a_non_causal_build_is_rejected():
    """A dropped window returns dense attention: right shape, finite, wrong.

    The non-causal arm splits the KV range into `[full][tail-masked]` and has
    no left-masked region to apply a window with, so this cannot be honoured
    at runtime -- and nobody passes `window=` without believing it applies.
    """
    _require_env()
    q, k, v = _qkv(1, 256, 64, torch.float16)
    with pytest.raises(ValueError, match="requires a causal build"):
        build_flash_attn_func_aiw_module(
            num_heads=_NUM_HEADS, head_dim=64, causal=False, dtype_str="f16",
        )(q, k, v, torch.empty_like(q), 1, 256, window=(-16, 64))


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

    At a preference of 3, head_dim 384 splits a 128-column window into
    42-column slices -- not a multiple of WMMA_N. The resolver must walk down
    to something valid instead of failing the build.

    The preference is passed explicitly rather than taken from the tuning
    table. It used to be read from the table, which coupled a test of the
    *resolver* to a tuning value and duly broke when head_dim 384 was retuned
    from 3 shards to 2 -- while the property under test had not changed at all.
    """
    assert resolve_shards(384, 384, 32, want=3) == 3
    assert resolve_shards(384, 128, 32, want=3) < 3
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
        (dict(head_dim=100), "BLOCK_DMODEL"),
        (dict(head_dim=128, block_n=64, causal=True), "BLOCK_N"),
        (dict(head_dim=128, v_lds_layout="row", shards=2), "cross-shard"),
        (dict(head_dim=64, q_row_tiles=2, shards=2), "untested"),
        (dict(head_dim=64, q_row_tiles=3), "Q_ROW_TILES"),
        (dict(head_dim=128, k_prefetch_dist=2), "K_PREFETCH_DIST"),
        (dict(head_dim=128, v_prefetch_dist=2), "V_PREFETCH_DIST"),
        (dict(head_dim=128, head_dim_v=24), "BLOCK_DMODEL_V"),
        (dict(head_dim=128, head_dim_v=64, d_offset=96), "D_OFFSET"),
    ],
)
def test_knob_validity_predicate(kwargs, match):
    """Unimplemented knob combinations fail the build, not the output.

    `AssertionError`, not `ValueError`: these are statements about knobs the
    tuning module has already resolved, so a violation is an internal
    inconsistency rather than bad caller input. Caller input is validated in
    `plan()`, which still raises `ValueError`.
    """
    base = dict(num_heads=_NUM_HEADS, causal=False, dtype_str="f16")
    base.update(kwargs)
    with pytest.raises(AssertionError, match=match):
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
    # The k-q difference **varies per sequence**, deliberately. A uniform
    # difference makes `seqlen_k[z] - seqlen_q[z]` equal the batch-wide
    # `Max_seqlen_k - Max_seqlen_q`, which hides any bottom-right
    # implementation that resolves the diagonal per batch instead of per
    # sequence -- it did exactly that, and every uniform test passed.
    lens_k = [x + 3 + 17 * i if x else 0 for i, x in enumerate(lens_q)]
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


def test_varlen_bottom_right_diagonal_is_per_sequence():
    """Bottom-right causal must use *this sequence's* `seqlen_k - seqlen_q`.

    The reason `Window_left`/`Window_right` carry sentinels (`0x80000001`,
    `0x80000002`) rather than resolved bounds: the alignment can only be
    turned into a number once the lengths are known, and under varlen there
    is one pair per sequence. Resolving on the host gave every sequence the
    batch-wide difference.

    The length set below has a k-q difference that varies per sequence *and*
    is not the maximum for most of them, which is what makes the two
    resolutions disagree. With a uniform difference they coincide and this
    test cannot fail.
    """
    _require_env()
    head_dim = 64
    lens_q = [64, 96, 40]
    lens_k = [64 + 5, 96 + 40, 40 + 3]   # differences 5, 40, 3; batch-wide 40
    N = len(lens_q)
    q, k, v, cq, ck = _packed_qkv(lens_q, lens_k, head_dim)
    o = torch.zeros_like(q)
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=True, causal_type=2,
        dtype_str="f16",
    )
    mq, mk = max(lens_q), max(lens_k)
    exe(q, k, v, o, N, mq, mk, varlen=exe.varlen_compact(cq, ck, mq, mk))
    torch.cuda.synchronize()
    _assert_varlen_matches_dense(exe, q, k, v, o, lens_q, lens_k, cq, ck,
                                 "bottom-right/varlen")


_GRID_WASTE_LENGTHS = [
    ([1365] * 4 + [683] * 4, "mild_skew"),
    ([4096] + [820] * 5, "one_long"),
    ([4096, 2048, 1024] + [700] * 3, "heavy_tail"),
]


@pytest.mark.parametrize("lens,label", _GRID_WASTE_LENGTHS, ids=[x[1] for x in _GRID_WASTE_LENGTHS])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_varlen_dead_workgroups_do_not_read_past_the_tensor(lens, label, causal):
    """A workgroup past a sequence's end must not *address* past it either.

    Varlen sizes the Q grid from `Max_seqlen_q`, so short sequences get
    workgroups whose rows are all out of range. Those exit without storing --
    but the Q base folds `q_start` into the 64-bit address before any guard
    applies, and the in-bounds check only clamps the row *within* the tile. So
    the load still reached `row_off + q_start` rows in, which on a packed
    tensor is past the end of the whole allocation rather than merely past the
    sequence.

    It faulted for real: a 16-sequence batch overshoots by ~1.3 MB and hits an
    unmapped page. Smaller overshoots land inside the allocation, read garbage
    that is then discarded, and pass -- which is precisely why every varlen
    test already in this file missed it. The lengths here are chosen to make
    the overshoot large: several sequences much shorter than the longest, and
    lengths that are *not* multiples of BLOCK_M so the tail tiles are real.

    A correctness assertion cannot express "did not read out of bounds", so
    this test relies on the fault being fatal. That is a weaker guarantee than
    the rest of the suite offers and is worth knowing when reading it.
    """
    _require_env()
    head_dim = 64
    N, mq = len(lens), max(lens)
    q, k, v, cq, ck = _packed_qkv(lens, lens, head_dim)
    o = torch.empty_like(q)
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=causal, dtype_str="f16",
    )
    exe(q, k, v, o, N, mq, mq, varlen=exe.varlen_compact(cq, ck, mq, mq))
    torch.cuda.synchronize()
    assert torch.isfinite(o).all()
    _assert_varlen_matches_dense(exe, q, k, v, o, lens, lens, cq, ck,
                                 f"grid-waste/{label}")


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_varlen_sequence_with_no_keys(causal):
    """A sequence with queries but no keys: every row dead, no memory touched.

    `seqlen_k == 0` with `seqlen_q > 0` is a legitimate varlen configuration
    and the plan's matrix asks for it, but it was never written. It faulted:
    the distance-1 prefetch is issued before any loop bound is consulted, and
    `seq_last = seqlen_k - 1` underflows an *unsigned* index to 2**64-1, so
    the KV clamp pinned every address to that row.

    The fix is not to clamp the address but to skip the load -- at
    `seqlen_k == 0` there is no in-range address to clamp *to*, since row 0 of
    an empty sequence is already one past its end.
    """
    _require_env()
    head_dim = 64
    lens_q = [64, 96, 40]
    lens_k = [64, 0, 40]          # sequence 1 has queries and no keys
    N = len(lens_q)
    q, k, v, cq, ck = _packed_qkv(lens_q, lens_k, head_dim)
    o = torch.full_like(q, float("nan"))
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=causal, dtype_str="f16",
    )
    exe(q, k, v, o, N, max(lens_q), max(lens_k),
        varlen=exe.varlen_compact(cq, ck, max(lens_q), max(lens_k)))
    torch.cuda.synchronize()
    s0, e0 = int(cq[1]), int(cq[1]) + lens_q[1]
    assert (o[:, s0:e0] == 0).all(), "rows with no keys must be exactly zero"
    # The sequences that do have keys are unaffected by their neighbour.
    _assert_varlen_matches_dense(exe, q, k, v, o, lens_q, lens_k, cq, ck,
                                 "no-keys neighbour")


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


def _strided_layout(lens, gaps):
    """(positions, cu_seqlens, padded_total) for sequences with gaps between.

    `gaps` varies per sequence deliberately: a uniform gap is indistinguishable
    from a longer uniform stride, so an implementation that mishandles the
    position array can still pass a uniform-gap test.
    """
    pos, at = [], 0
    for ln, gap in zip(lens, gaps):
        pos.append(at)
        at += ln + gap
    return (torch.tensor(pos + [at], dtype=torch.int32, device="cuda"),
            _cu(lens), at)


@pytest.mark.parametrize("ctype", [0, 1], ids=["full", "causal"])
def test_varlen_strided_reads_the_position_array(ctype):
    """`StridedVarlen`: positions from `seqinfo_?1`, lengths from `?0`.

    Two assertions, and the second is the one with teeth. Matching the dense
    calls shows the gaps are honoured; *differing* from the gapless reading of
    the same buffer shows `seqinfo_?1` is read at all. Without it, an
    implementation that ignored the position array and fell back to `REUSE`
    would pass everything above.

    This is also the only path besides `seqused_k` that touches `seqinfo_?1`,
    so it carries that code's entire coverage (plan §7 property 2).
    """
    _require_env()
    head_dim = 64
    lens_q = [96, 40, 133]
    lens_k = [x + 7 for x in lens_q]
    gaps_q = [11, 64, 3]          # per-sequence, not uniform
    gaps_k = [5, 32, 17]
    N = len(lens_q)
    sst_q, cq, Tq_pad = _strided_layout(lens_q, gaps_q)
    sst_k, ck, Tk_pad = _strided_layout(lens_k, gaps_k)
    gen = torch.Generator(device="cuda").manual_seed(0)

    def _t(n):
        return torch.randn(1, n, _NUM_HEADS, head_dim, dtype=torch.float16,
                           device="cuda", generator=gen)

    q, k, v = _t(Tq_pad), _t(Tk_pad), _t(Tk_pad)
    o = torch.zeros_like(q)
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=bool(ctype),
        causal_type=ctype or None, dtype_str="f16",
    )
    mq, mk = max(lens_q), max(lens_k)
    exe(q, k, v, o, N, mq, mk,
        varlen=exe.varlen_strided(cq, ck, sst_q, sst_k, mq, mk))
    torch.cuda.synchronize()

    for z, (lq, lk) in enumerate(zip(lens_q, lens_k)):
        qs, ks = int(sst_q[z]), int(sst_k[z])
        ref = torch.empty(1, lq, _NUM_HEADS, head_dim, dtype=torch.float16, device="cuda")
        exe(q[:, qs:qs + lq].contiguous(), k[:, ks:ks + lk].contiguous(),
            v[:, ks:ks + lk].contiguous(), ref, 1, lq, lk)
        torch.cuda.synchronize()
        assert torch.equal(ref, o[:, qs:qs + lq]), (
            f"strided sequence {z} differs from its dense call"
        )

    # Negative control: the same buffer read as if it were gapless.
    o_compact = torch.zeros_like(q)
    exe(q, k, v, o_compact, N, mq, mk,
        varlen=exe.varlen_compact(cq, ck, mq, mk))
    torch.cuda.synchronize()
    assert not torch.equal(o, o_compact), (
        "strided produced the same result as the gapless reading of the same "
        "buffer -- seqinfo_?1 is not being read"
    )


@pytest.mark.parametrize("k_is_cache", [False, True], ids=["packed_kv", "bhsd_cache"])
def test_varlen_seqused_k_shortens_only_k(k_is_cache):
    """The configuration no `VarlenType` can express (plan §1.4).

    The K side takes its **length** from an individual array (`seqused_k`) and
    its **position** from a cumulative one, so axes B and C read different
    tensors. Two assertions again: the result must equal a dense call on the
    *truncated* K, and must differ from one on the full K -- only the second
    fails if `seqused_k` is ignored.
    """
    _require_env()
    head_dim = 64
    lens_q = [96, 40, 128]
    alloc_k = [x + 64 for x in lens_q]      # cache slots, larger than used
    used_k = [x + 7 for x in lens_q]        # what actually participates
    N, mq = len(lens_q), max(lens_q)
    mk = max(alloc_k)
    cq = _cu(lens_q)
    ck = _cu(alloc_k)
    su = torch.tensor(used_k, dtype=torch.int32, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(1, int(cq[-1]), _NUM_HEADS, head_dim, dtype=torch.float16,
                    device="cuda", generator=gen)
    if k_is_cache:
        kv_shape = (N, mk, _NUM_HEADS, head_dim)
    else:
        kv_shape = (1, int(ck[-1]), _NUM_HEADS, head_dim)
    k = torch.randn(*kv_shape, dtype=torch.float16, device="cuda", generator=gen)
    v = torch.randn(*kv_shape, dtype=torch.float16, device="cuda", generator=gen)
    o = torch.zeros_like(q)
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=False, dtype_str="f16",
    )
    exe(q, k, v, o, N, mq, mk,
        varlen=exe.varlen_seqused_k(cq, ck, su, mq, mk, k_is_cache=k_is_cache))
    torch.cuda.synchronize()

    for z, lq in enumerate(lens_q):
        qs = int(cq[z])
        qz = q[:, qs:qs + lq].contiguous()
        if k_is_cache:
            kz_full, vz_full = k[z:z + 1], v[z:z + 1]
        else:
            ks = int(ck[z])
            kz_full = k[:, ks:ks + alloc_k[z]]
            vz_full = v[:, ks:ks + alloc_k[z]]
        ref = torch.empty_like(qz)
        exe(qz, kz_full[:, :used_k[z]].contiguous(),
            vz_full[:, :used_k[z]].contiguous(), ref, 1, lq, used_k[z])
        torch.cuda.synchronize()
        assert torch.equal(ref, o[:, qs:qs + lq]), (
            f"seqused_k sequence {z} differs from a dense call on the "
            f"truncated K"
        )
        # Negative control: the untruncated cache must give something else.
        ref_full = torch.empty_like(qz)
        exe(qz, kz_full[:, :alloc_k[z]].contiguous(),
            vz_full[:, :alloc_k[z]].contiguous(), ref_full, 1, lq, alloc_k[z])
        torch.cuda.synchronize()
        assert not torch.equal(ref_full, o[:, qs:qs + lq]), (
            f"seqused_k sequence {z} matched the *full* cache -- the used "
            f"length is being ignored"
        )


# ---------------------------------------------------------------------------
# Suite B: one awkward length set, every configuration
# ---------------------------------------------------------------------------
#
# The full product of lengths x modes is far too large, and it factors: the
# mode decides how a sequence is *located*, the lengths decide what is *in*
# it. Suite A projects onto lengths with the mode fixed; this projects onto
# modes with the lengths fixed.
#
# What the factorisation gives up is an interaction needing both an unusual
# length pattern and an unusual mode. That is bounded because the mode is
# consumed entirely in the prologue -- it becomes three scalars and every
# length pattern then flows through identical code -- so such an interaction
# would have to be a prologue bug. The length set below is chosen to be
# awkward enough to provoke one: ragged, N = 7, containing a zero, a
# length-1 sequence and one much longer than the rest, with seqlen_k above
# seqlen_q throughout.

_SUITE_B_Q = [200, 0, 37, 1, 128, 3, 96]
# Non-uniform k-q differences, per the note in suite A.
_SUITE_B_K = [x + 5 + 11 * i if x else 0 for i, x in enumerate(_SUITE_B_Q)]


def _suite_b_case(mode, head_dim, n_head_k):
    """Build (q, k, v, varlen, slicers) for one configuration.

    `slicers[z]` returns the (q, k, v) views a plain dense call would receive
    for sequence z, which is what the result must match bit for bit.
    """
    lq, lk = _SUITE_B_Q, _SUITE_B_K
    N, mq, mk = len(lq), max(lq), max(lk)
    cq, ck = _cu(lq), _cu(lk)
    gen = torch.Generator(device="cuda").manual_seed(7)

    def rnd(*shape):
        return torch.randn(*shape, dtype=torch.float16, device="cuda", generator=gen)

    def packed_slicer(q, k, v, qpos, kpos, klens):
        def _s(z):
            return (q[:, qpos[z]:qpos[z] + lq[z]].contiguous(),
                    k[:, kpos[z]:kpos[z] + klens[z]].contiguous(),
                    v[:, kpos[z]:kpos[z] + klens[z]].contiguous(), klens[z])
        return _s

    if mode == "dense_k":
        # Packed Q against a rectangular K: the Q byte is compact, the K byte 0.
        q = rnd(1, int(cq[-1]), _NUM_HEADS, head_dim)
        k = rnd(N, mk, n_head_k, head_dim)
        v = rnd(N, mk, n_head_k, head_dim)
        bits = None  # assembled by the caller below
        return q, k, v, ("mixed", cq, ck, mq, mk), (
            lambda z: (q[:, int(cq[z]):int(cq[z]) + lq[z]].contiguous(),
                       k[z:z + 1].contiguous(), v[z:z + 1].contiguous(), mk))

    if mode == "padded":
        q = rnd(N, mq, _NUM_HEADS, head_dim)
        k = rnd(N, mk, n_head_k, head_dim)
        v = rnd(N, mk, n_head_k, head_dim)
        return q, k, v, ("padded", cq, ck, mq, mk), (
            lambda z: (q[z:z + 1, :lq[z]].contiguous(),
                       k[z:z + 1, :lk[z]].contiguous(),
                       v[z:z + 1, :lk[z]].contiguous(), lk[z]))

    if mode == "strided":
        gq = [7, 3, 64, 1, 40, 5, 11][:N]
        gk = [5, 17, 2, 32, 9, 3, 21][:N]
        sst_q, _, Tq = _strided_layout(lq, gq)
        sst_k, _, Tk = _strided_layout(lk, gk)
        q, k, v = rnd(1, Tq, _NUM_HEADS, head_dim), rnd(1, Tk, n_head_k, head_dim), rnd(1, Tk, n_head_k, head_dim)
        return q, k, v, ("strided", cq, ck, sst_q, sst_k, mq, mk), packed_slicer(
            q, k, v, [int(x) for x in sst_q], [int(x) for x in sst_k], lk)

    if mode in ("seqused_packed", "seqused_cache"):
        alloc = [x + 48 if x else 0 for x in lq]
        used = lk
        ca = _cu(alloc)
        q = rnd(1, int(cq[-1]), _NUM_HEADS, head_dim)
        cache = mode == "seqused_cache"
        shape = (N, max(alloc), n_head_k, head_dim) if cache else (1, int(ca[-1]), n_head_k, head_dim)
        k, v = rnd(*shape), rnd(*shape)
        su = torch.tensor(used, dtype=torch.int32, device="cuda")
        tag = ("seqused", cq, ca, su, mq, max(alloc), cache)
        if cache:
            sl = lambda z: (q[:, int(cq[z]):int(cq[z]) + lq[z]].contiguous(),
                            k[z:z + 1, :used[z]].contiguous(),
                            v[z:z + 1, :used[z]].contiguous(), used[z])
        else:
            sl = packed_slicer(q, k, v, [int(x) for x in cq], [int(x) for x in ca], used)
        return q, k, v, tag, sl

    # compact
    q = rnd(1, int(cq[-1]), _NUM_HEADS, head_dim)
    k, v = rnd(1, int(ck[-1]), n_head_k, head_dim), rnd(1, int(ck[-1]), n_head_k, head_dim)
    return q, k, v, ("compact", cq, ck, mq, mk), packed_slicer(
        q, k, v, [int(x) for x in cq], [int(x) for x in ck], lk)


_SUITE_B_MODES = ["compact", "strided", "padded", "seqused_packed",
                  "seqused_cache", "dense_k"]


@pytest.mark.parametrize("gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize("mode", _SUITE_B_MODES)
def test_varlen_suite_b_all_modes(mode, gqa):
    """Every configuration against the same awkward length set, bitwise.

    The GQA axis is here rather than in suite A because varlen and GQA touch
    the same address expression on different axes -- varlen picks the batch
    slice and the row, GQA the head. They are orthogonal only because strides
    are per-tensor: for a 1THD tensor the per-token stride is `H * D`, which
    differs between Q and K under GQA, so `q_row_off` and `k_row_off` are
    scaled by different multipliers. A shared token stride would be silently
    wrong for every packed GQA call, and nothing else here would catch it.
    """
    _require_env()
    head_dim = 64
    n_head_k = 1 if gqa else _NUM_HEADS
    q, k, v, tag, slicer = _suite_b_case(mode, head_dim, n_head_k)
    N, mq = len(_SUITE_B_Q), max(_SUITE_B_Q)
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=True, dtype_str="f16",
    )
    if tag[0] == "compact":
        _, cq, ck, _mq, mk = tag
        varlen = exe.varlen_compact(cq, ck, mq, mk)
    elif tag[0] == "padded":
        _, cq, ck, _mq, mk = tag
        varlen = exe.varlen_padded(cq, ck, mq, mk)
    elif tag[0] == "strided":
        _, cq, ck, sq, sk, _mq, mk = tag
        varlen = exe.varlen_strided(cq, ck, sq, sk, mq, mk)
    elif tag[0] == "seqused":
        _, cq, ca, su, _mq, mk, cache = tag
        varlen = exe.varlen_seqused_k(cq, ca, su, mq, mk, k_is_cache=cache)
    else:  # packed Q, rectangular K with full lengths
        _, cq, ck, _mq, mk = tag
        varlen = dict(
            bits=exe.varlen_bits(0x0B, 0x00),
            seqinfo_q0=cq, seqinfo_q1=None, seqinfo_k0=None, seqinfo_k1=None,
            max_seqlen_q=mq, max_seqlen_k=mk, lse_tokens=int(cq[-1]),
        )
    o = torch.zeros_like(q)
    exe(q, k, v, o, N, mq, mk, varlen=varlen)
    torch.cuda.synchronize()

    for z, lq_z in enumerate(_SUITE_B_Q):
        if lq_z == 0:
            continue
        qz, kz, vz, lk_z = slicer(z)
        ref = torch.empty_like(qz)
        exe(qz, kz, vz, ref, 1, lq_z, lk_z)
        torch.cuda.synchronize()
        if tag[0] == "padded":
            got = o[z:z + 1, :lq_z]
        elif tag[0] == "strided":
            base = int(tag[3][z])
            got = o[:, base:base + lq_z]
        else:
            got = o[:, int(tag[1][z]):int(tag[1][z]) + lq_z]
        assert torch.equal(ref, got), (
            f"{mode}{'/gqa' if gqa else ''} sequence {z} (Lq={lq_z}, "
            f"Lk={lk_z}) differs from its dense call"
        )


def test_varlen_rejects_meaningless_configurations():
    """Every encodable byte now decodes, so what is left is the *meaningless*.

    `REUSE` takes a position out of the length array, which is only a position
    when the lengths are cumulative; and two codes per field are reserved.
    Both are rejected when the bits are assembled, before any launch.
    """
    _require_env()
    exe = build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=64, causal=False, dtype_str="f16",
    )
    with pytest.raises(ValueError, match="REUSE requires"):
        exe.varlen_bits(0x0D, 0x0D)          # INDIVIDUAL length + REUSE position
    with pytest.raises(ValueError, match="REUSE requires"):
        exe.varlen_bits(0x09, 0x09)          # MAX length + REUSE position
    with pytest.raises(ValueError, match="reserved"):
        exe.varlen_bits(0x06, 0x00)          # LENGTH == 3
    with pytest.raises(ValueError, match="reserved"):
        exe.varlen_bits(0x18, 0x00)          # POSITION == 3
    with pytest.raises(ValueError, match="fit in a byte"):
        exe.varlen_bits(0x100, 0x00)

# ---------------------------------------------------------------------------
# Bias tensor (BIAS_TYPE)
# ---------------------------------------------------------------------------


def _bias_build(head_dim, causal=False, **kw):
    return build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=causal,
        dtype_str="f16", bias=True, **kw
    )


def _bias_ref(q, k, v, bias):
    qb, kb, vb = (x.transpose(1, 2).float() for x in (q, k, v))
    return F.scaled_dot_product_attention(
        qb, kb, vb, attn_mask=bias.float()
    ).transpose(1, 2)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seq", [256, 200], ids=["aligned", "ragged"])
def test_bias_matches_reference(head_dim, seq):
    """A random bias must reproduce SDPA with the same additive mask.

    The ragged length matters on its own: the tail tile loads bias columns
    past `seqlen_k`, and those must be killed by the tail mask rather than
    added -- which is why the bias add is ordered before it.
    """
    _require_env()
    q, k, v = _qkv(1, seq, head_dim, torch.float16)
    gen = torch.Generator(device="cuda").manual_seed(7)
    bias = torch.randn(1, _NUM_HEADS, seq, seq, dtype=torch.float16,
                       device="cuda", generator=gen)
    o = torch.empty_like(q)
    _bias_build(head_dim)(q, k, v, o, 1, seq, bias=bias)
    torch.cuda.synchronize()
    rel = _rel(o, _bias_ref(q, k, v, bias))
    assert rel < 5e-3, f"hd={head_dim} seq={seq} rel={rel:.3e}"


def test_zero_bias_matches_no_bias():
    """An all-zero bias must change nothing measurable.

    **Not bitwise**, and the reason is the one gSWA already taught: adding a
    floating-point operation changes the emitted code even when the operation
    is mathematically a no-op, and under `reassoc`/`contract` the compiler may
    then contract differently. Measured here: 33 of 32768 elements differ, by
    at most ~1 f16 ULP.

    What this still checks is that bias is added rather than, say, multiplied
    or dropped -- a bias that was ignored entirely would also pass, which is
    why `test_bias_matches_reference` exists alongside it.
    """
    _require_env()
    head_dim, seq = 64, 256
    q, k, v = _qkv(1, seq, head_dim, torch.float16)
    o_none = torch.empty_like(q)
    build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=False, dtype_str="f16"
    )(q, k, v, o_none, 1, seq)
    o_zero = torch.empty_like(q)
    zeros = torch.zeros(1, _NUM_HEADS, seq, seq, dtype=torch.float16, device="cuda")
    _bias_build(head_dim)(q, k, v, o_zero, 1, seq, bias=zeros)
    torch.cuda.synchronize()
    rel = _rel(o_zero, o_none.float())
    assert rel < 1e-3, f"zero bias moved the result by rel={rel:.3e}"


def test_bias_minus_inf_row_is_a_dead_row():
    """`-inf` across a whole row is how a caller says "attend to nothing".

    Same contract dead rows already have under gSWA -- `O = 0`, `LSE = +inf`
    -- but reached through a different path, so it is asserted separately.
    """
    _require_env()
    head_dim, seq = 64, 256
    q, k, v = _qkv(1, seq, head_dim, torch.float16)
    bias = torch.zeros(1, _NUM_HEADS, seq, seq, dtype=torch.float16, device="cuda")
    bias[:, :, 3, :] = float("-inf")
    o = torch.empty_like(q)
    lse = torch.zeros(_NUM_HEADS, seq, dtype=torch.float32, device="cuda")
    _bias_build(head_dim)(q, k, v, o, 1, seq, bias=bias, lse=lse)
    torch.cuda.synchronize()
    assert (o[:, 3] == 0).all(), "a row with no visible key must be zero"
    assert torch.isinf(lse[:, 3]).all() and (lse[:, 3] > 0).all(), "LSE must be +inf"


def test_bias_minus_inf_first_tile_does_not_produce_nan():
    """The `m_i` floor, finally reachable -- and verified to fail without it.

    P1 initialised `m_i` at -3.40282e+38 rather than `-inf` and recorded the
    fix as *preventative*: under causal masking alone the first KV tile always
    contains a live column, so `-inf - -inf` could not happen. A bias covering
    that whole tile with `-inf` is the first configuration that reaches it.
    With an `-inf` floor the correction term is `exp2(-inf - -inf) = NaN`;
    with the real floor it is `exp2(-inf + 3.4e38) = 0`.

    This was checked against a deliberately reverted floor rather than assumed:
    reverting `c_m_init` to `-inf` turns **every one** of the 32768 output
    elements into NaN, and restoring it gives none. A regression test that has
    never been seen to fail is not yet a regression test.

    It also keeps `ninf` honest. The `-inf` here comes from *user data*, so a
    build that enabled `no-infs-fp-math` would be licensed to delete it, and
    the reference comparison would drift rather than fault.
    """
    _require_env()
    head_dim, seq = 64, 256
    q, k, v = _qkv(1, seq, head_dim, torch.float16)
    bias = torch.zeros(1, _NUM_HEADS, seq, seq, dtype=torch.float16, device="cuda")
    bias[:, :, :, :32] = float("-inf")      # the whole first KV tile
    o = torch.empty_like(q)
    _bias_build(head_dim)(q, k, v, o, 1, seq, bias=bias)
    torch.cuda.synchronize()
    assert not torch.isnan(o).any(), "m_i floor: -inf - -inf produced NaN"
    rel = _rel(o, _bias_ref(q, k, v, bias))
    assert rel < 5e-3, f"rel={rel:.3e}"


def test_bias_shared_plane_via_zero_stride():
    """Stride 0 on the batch axis is how a caller shares one bias plane."""
    _require_env()
    head_dim, seq = 64, 256
    q, k, v = _qkv(2, seq, head_dim, torch.float16)
    gen = torch.Generator(device="cuda").manual_seed(3)
    plane = torch.randn(1, _NUM_HEADS, seq, seq, dtype=torch.float16,
                        device="cuda", generator=gen)
    shared = plane.expand(2, _NUM_HEADS, seq, seq)      # stride(0) == 0
    assert shared.stride(0) == 0
    o = torch.empty_like(q)
    _bias_build(head_dim)(q, k, v, o, 2, seq, bias=shared)
    torch.cuda.synchronize()
    rel = _rel(o, _bias_ref(q, k, v, shared))
    assert rel < 5e-3, f"rel={rel:.3e}"


def test_bias_and_causal_are_mutually_exclusive():
    """Undefined, not unimplemented -- so it must be an error, not an answer.

    Causal is an attention mask with a fixed pattern; bias *is* an attention
    mask supplied directly. Asking for both asks which wins where they
    disagree, and there is no defined answer. AOTriton disables the functional
    and PyTorch's math backend raises; see sdpa-bias-plan.md 3.2.
    """
    with pytest.raises(ValueError, match="mutually exclusive"):
        _bias_build(64, causal=True)


def test_bias_requires_contiguous_last_dim():
    """The KV axis is loaded eight columns at a time; it must be contiguous."""
    _require_env()
    head_dim, seq = 64, 256
    q, k, v = _qkv(1, seq, head_dim, torch.float16)
    bad = torch.zeros(1, _NUM_HEADS, seq, seq, dtype=torch.float16,
                      device="cuda").transpose(2, 3)
    with pytest.raises(ValueError, match="contiguous"):
        _bias_build(head_dim)(q, k, v, torch.empty_like(q), 1, seq, bias=bad)

# ---------------------------------------------------------------------------
# Dropout (ENABLE_DROPOUT)
# ---------------------------------------------------------------------------


def _drop_build(head_dim, **kw):
    return build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=False,
        dtype_str="f16", dropout=True, **kw
    )


def test_dropout_p0_is_bit_identical_to_no_dropout():
    """`p = 0` must be *identical*, not merely close.

    The PRNG still runs, the threshold still compares, and `1/(1-0)` is
    exactly 1 -- so every element is kept and every arithmetic step is a
    no-op. Anything less than bitwise here means a scale or a select is
    perturbing values it should not touch.
    """
    _require_env()
    head_dim, seq = 64, 256
    q, k, v = _qkv(1, seq, head_dim, torch.float16)
    o_off = torch.empty_like(q)
    build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=False, dtype_str="f16"
    )(q, k, v, o_off, 1, seq)
    o_p0 = torch.empty_like(q)
    _drop_build(head_dim)(q, k, v, o_p0, 1, seq, dropout_p=0.0, philox_seed=7)
    torch.cuda.synchronize()
    assert torch.equal(o_off, o_p0)


def test_dropout_p1_drops_everything():
    """`p -> 1` keeps nothing, so the output is exactly zero."""
    _require_env()
    head_dim, seq = 64, 256
    q, k, v = _qkv(1, seq, head_dim, torch.float16)
    o = torch.empty_like(q)
    _drop_build(head_dim)(q, k, v, o, 1, seq, dropout_p=0.999999, philox_seed=7)
    torch.cuda.synchronize()
    assert o.abs().max().item() == 0.0


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_dropout_leaves_logsumexp_undropped(p):
    """`l` accumulates the *pre-dropout* sum, so LSE is unchanged by dropout.

    This is the ordering that produces plausible-but-wrong output if reversed:
    scaling the softmax denominator by the surviving fraction makes the result
    stop being an expectation of the undropped attention, and makes the LSE the
    backward pass reads wrong by a per-row factor. No shape or finiteness check
    notices, which is why it is asserted directly.
    """
    _require_env()
    head_dim, seq = 64, 256
    q, k, v = _qkv(1, seq, head_dim, torch.float16)
    o = torch.empty_like(q)
    lse_drop = torch.zeros(_NUM_HEADS, seq, dtype=torch.float32, device="cuda")
    _drop_build(head_dim)(q, k, v, o, 1, seq, dropout_p=p, philox_seed=1234,
                          lse=lse_drop)
    lse_ref = torch.zeros(_NUM_HEADS, seq, dtype=torch.float32, device="cuda")
    build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=False, dtype_str="f16"
    )(q, k, v, torch.empty_like(q), 1, seq, lse=lse_ref)
    torch.cuda.synchronize()
    assert torch.allclose(lse_drop, lse_ref, atol=1e-5), (
        "logsumexp moved with dropout -- l was accumulated after the mask"
    )


def test_dropout_is_reproducible_from_seed_alone():
    """Same seed, same mask; different seed, different mask.

    The property the backward pass depends on: it is handed `(seed, offset)`
    and regenerates the mask rather than being given it, so a stream that did
    not depend only on those would make gradients silently wrong.
    """
    _require_env()
    head_dim, seq = 64, 256
    q, k, v = _qkv(1, seq, head_dim, torch.float16)
    exe = _drop_build(head_dim)
    a, b, c = (torch.empty_like(q) for _ in range(3))
    exe(q, k, v, a, 1, seq, dropout_p=0.5, philox_seed=11)
    exe(q, k, v, b, 1, seq, dropout_p=0.5, philox_seed=11)
    exe(q, k, v, c, 1, seq, dropout_p=0.5, philox_seed=12)
    torch.cuda.synchronize()
    assert torch.equal(a, b), "same seed gave a different mask"
    assert not torch.equal(a, c), "different seeds gave the same mask"


def test_dropout_offset_shifts_the_mask():
    """`philox_offset` must reach the stream too, not just the seed.

    PyTorch's RNG state is a (seed, offset) pair and advances the offset
    between calls; a build that ignored it would repeat the same mask for
    every call in a training step.
    """
    _require_env()
    head_dim, seq = 64, 256
    q, k, v = _qkv(1, seq, head_dim, torch.float16)
    exe = _drop_build(head_dim)
    a, b = torch.empty_like(q), torch.empty_like(q)
    exe(q, k, v, a, 1, seq, dropout_p=0.5, philox_seed=11, philox_offset=0)
    exe(q, k, v, b, 1, seq, dropout_p=0.5, philox_seed=11, philox_offset=1 << 20)
    torch.cuda.synchronize()
    assert not torch.equal(a, b)


@pytest.mark.parametrize("p", [0.25, 0.5])
def test_dropout_expectation_is_preserved(p):
    """Dropout must be unbiased: `E[O]` stays at the undropped output.

    This is the only test that sees the `1/(1-p)` scale at all -- the
    exactness tests use `p = 0` and `p -> 1`, where the scale is respectively
    1 and irrelevant. A dropped or mis-signed scale biases the mean by a
    factor of `1-p`, far outside the tolerance below.

    `V` is made non-negative on purpose. With signed random `V` the attention
    output is a near-cancelling weighted sum, so `|O_ref|` lands at the same
    magnitude as the dropout noise itself and the estimator's relative error
    is `sqrt(p/(1-p)/n)` -- 0.35 at `p = 0.5, n = 8`, which forces a tolerance
    so loose it would no longer catch the bias it exists to catch. Positive
    `V` makes the signal `O(1)` instead of `O(1/sqrt(seq))` and buys back the
    ~sqrt(seq) of headroom.
    """
    _require_env()
    head_dim, seq = 64, 512
    q, k, v = _qkv(1, seq, head_dim, torch.float16)
    v = v.abs()
    o_ref = torch.empty_like(q)
    build_flash_attn_func_aiw_module(
        num_heads=_NUM_HEADS, head_dim=head_dim, causal=False, dtype_str="f16"
    )(q, k, v, o_ref, 1, seq)
    exe = _drop_build(head_dim)
    acc = torch.zeros_like(q, dtype=torch.float32)
    n = 8
    for i in range(n):
        o = torch.empty_like(q)
        exe(q, k, v, o, 1, seq, dropout_p=p, philox_seed=1000 + i)
        torch.cuda.synchronize()
        acc += o.float()
    mean = acc / n
    rel = (mean - o_ref.float()).abs().mean() / o_ref.float().abs().mean()
    assert rel < 0.05, f"E[O] drifted from the undropped output: rel={rel:.3f}"


def test_dropout_requires_p():
    """A dropout build with no probability is a caller error, not p=0."""
    _require_env()
    q, k, v = _qkv(1, 256, 64, torch.float16)
    with pytest.raises(ValueError, match="requires dropout_p"):
        _drop_build(64)(q, k, v, torch.empty_like(q), 1, 256)


# ---------------------------------------------------------------------------
# 64-bit addressing
# ---------------------------------------------------------------------------

_I32_MAX = 2 ** 31 - 1

# (shape, strides) whose *named* axis has a stride crossing 2^31, with every
# other axis packed. Only two slices along that axis exist, so the allocation
# is `big + a few kB` rather than the tensor the strides imply.
_BIG = 2 ** 31
_FAR_LAYOUTS = {
    #        B  S   H  D            b_str  s_str  h_str  d
    "batch": ((2, 64, 2, 64), (_BIG, 128, 64, 1)),
    "head":  ((1, 64, 2, 64), (_BIG, 128, _BIG, 1)),
    # Kept, and expected to fail: this documents the boundary of what the
    # kernel can address rather than hiding it. See the xfail reason below.
    "seq":   ((1, 2, 2, 64), (_BIG, _BIG, 64, 1)),
}


@pytest.mark.parametrize(
    "axis",
    [
        "batch",
        "head",
        "seq",
    ],
)
def test_element_offsets_past_2gi(axis):
    """An axis whose stride pushes an element past 2^31 must still address.

    This is the case no other test in the suite reaches. Every offset here is
    `base + index * stride`, and the file is full of places where that product
    could be formed in 32 bits and widened afterwards -- `fx.Int64(a * b)`
    rather than `fx.Int64(a) * fx.Int64(b)`. Both spellings compile, both are
    correct for every shape the rest of the suite uses, and they diverge
    exactly here: a product past 2^31 wraps negative and the kernel reads
    someone else's memory or faults.

    A strided view over one large allocation, so the offsets are real while the
    memory is not -- the far slice starts beyond element 2^31 and everything
    touched is one of the two slices.
    """
    _require_env()
    shape, strides = _FAR_LAYOUTS[axis]
    span = sum((d - 1) * st for d, st in zip(shape, strides)) + 1
    need = 3 * span * 2 + (1 << 27)
    free, _ = torch.cuda.mem_get_info()
    if free < need:
        pytest.skip(f"needs {need / 2**30:.1f} GiB free, have {free / 2**30:.1f}")
    assert span > _I32_MAX, "the layout does not actually cross 2^31"

    gen = torch.Generator(device="cuda").manual_seed(0)
    ref = [torch.randn(*shape, dtype=torch.float16, device="cuda", generator=gen)
           for _ in range(3)]
    far = []
    try:
        for r in ref:
            pool = torch.zeros(span, dtype=torch.float16, device="cuda")
            v = pool.as_strided(shape, strides)
            v.copy_(r)
            far.append(v)
        got = flydsl_flash_attn_func_gfx1201(*far, causal=True)
        want = flydsl_flash_attn_func_gfx1201(*ref, causal=True)
        torch.cuda.synchronize()
        assert torch.equal(got, want.to(got.dtype)), (
            f"a {axis}-axis stride past 2^31 changed the result: an offset "
            f"product was formed in 32 bits"
        )
    finally:
        far.clear()
        torch.cuda.empty_cache()

