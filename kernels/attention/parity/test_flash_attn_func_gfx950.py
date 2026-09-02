# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness for the gfx950 parity kernel (P0 + P1).

Run from this directory, with `ROCM_PATH` exported:

    export ROCM_PATH=$(rocm-sdk path --root)
    cd kernels/attention/parity
    python3 -m pytest test_flash_attn_func_gfx950.py -q

Three kinds of test, and the distinction is what makes the suite worth having:

- **Oracle** -- `kernels/attention/flash_attn_gfx950.py` computes the same
  thing by the same arithmetic in the same order, so an unpadded parity build
  must match it **bitwise**. This is much sharper than a tolerance against
  SDPA: it fails on a reordering that no tolerance would notice, which is
  exactly what an addressing change is most likely to introduce.
- **Semantics** -- tolerance against `F.scaled_dot_product_attention` for the
  configurations the oracle cannot express (padded heads, GQA, arbitrary
  layouts, asymmetric hdim).
- **Structural** -- assertions that a mistake would otherwise turn into
  plausible-looking numbers: that the stride slots mean what they say, and that
  the pad is not read.
"""

import itertools
import math
import time
from dataclasses import replace

import fmha_common_gfx1201 as fmha
import pytest
import torch
import torch.nn.functional as F
from flash_attn_func_gfx950 import build_flash_attn_func_gfx950_module as build
from fmha_tuning_gfx950 import LADDER, LADDER_PLANNED, FmhaInputMetadata, fmha_knobs, tile_width_for
from gfx950_standalone import dualwave  # noqa: F401  (puts the repo root on sys.path)

from kernels.attention.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module as build_prod

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    reason="requires a gfx950 device",
)

DT = torch.bfloat16
# bf16 through a deep reduction. Set from the production kernel's own measured
# error against fp32 SDPA, not from taste -- the two agree bitwise on every
# shape they share, so anything looser would be hiding a real regression.
TOL = 2.0e-2


def _rand(*shape, dtype=DT):
    return torch.randn(*shape, device="cuda", dtype=dtype)


def _err(got, ref):
    """Max absolute error, normalised by the reference's magnitude when > 1.

    Absolute error alone is the wrong metric once `sm_scale` is a free
    argument: raising the scale sharpens the softmax and grows `|O|` with it
    (0.24 at scale 0.05, 4.32 at scale 0.5 for the same inputs), so a fixed
    absolute bound would tighten as the scale falls and fail as it rises for
    no reason but the output's size. The production kernel shows the same
    growth to within 6% at every scale measured, which is what identifies this
    as bf16 output precision rather than anything this kernel does.

    Normalising only when `|ref| > 1` keeps the bound absolute for the ordinary
    O(1) cases, so no existing test is weakened.
    """
    scale = max(ref.abs().max().item(), 1.0)
    return (got.float() - ref).abs().max().item() / scale


def _ref(q, k, v, causal=False, scale=None, gqa=False):
    return F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), is_causal=causal, scale=scale, enable_gqa=gqa
    )


def _run(q, k, v, *, causal=False, scale=None, lse=None, o=None):
    """Build for this shape and dispatch, returning O.

    `hdim_vo` is taken from V's own last dimension rather than passed, so the
    asymmetric-hdim tests cannot accidentally disagree with the tensor they
    hand in.
    """
    b, hq, s, d = q.shape
    dv = v.shape[3]
    if o is None:
        o = torch.empty(b, hq, s, dv, device="cuda", dtype=q.dtype)
    fn = build(
        num_heads=hq,
        head_dim=d,
        head_dim_v=dv,
        causal=causal,
        dtype_str="bf16" if q.dtype is torch.bfloat16 else "f16",
        num_kv_heads=k.shape[1],
        return_lse=lse is not None,
    )
    fn(q, k, v, o, b, s, seqlen_k=k.shape[2], scale=scale, lse=lse)
    return o


# ---------------------------------------------------------------------------
# Oracle: bitwise against the production dualwave kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
# 192 and 256 are absent deliberately: `flash_attn_gfx950.py` still rejects them
# in its own guard, so there is no oracle to compare against, not a comparison
# that fails. The parity kernel's 192/256 rungs are covered by
# `test_ladder_matches_sdpa` instead.
@pytest.mark.parametrize("head_dim", [64, 128])
def test_bitwise_matches_production_kernel(head_dim, causal):
    """An unpadded parity build must be bit-identical to the production kernel.

    The parity kernel reaches the same data by a different address
    computation -- per-tensor strides with the head folded into the buffer
    descriptor, against a single token pitch with a per-access head offset. The
    arithmetic on the values is untouched, so the results must agree exactly.
    """
    b, h, s = 2, 8, 1024
    q, k, v = (_rand(b, h, s, head_dim) for _ in range(3))
    got = _run(q, k, v, causal=causal)

    # The production kernel takes BSHD-flattened contiguous tensors.
    qp, kp, vp = (t.transpose(1, 2).contiguous() for t in (q, k, v))
    op = torch.empty_like(qp)
    build_prod(h, head_dim, causal=causal, dtype_str="bf16")(qp, kp, vp, op, b, s)

    assert torch.equal(got, op.transpose(1, 2).contiguous())


# ---------------------------------------------------------------------------
# P0: strides, layouts, head counts, scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("layout", ["bhsd", "bshd", "mixed"])
def test_memory_layout_is_free(layout, causal):
    """Shape is the ABI; layout is not. Any permutation with D innermost works.

    `bshd` is the case that matters in production -- PyTorch's SDPA shim hands
    down a BHSD-shaped *view* of BSHD memory, so a kernel that derives its
    token pitch from the shape is not merely inflexible, it is wrong for the
    real call.
    """
    b, h, s, d = 2, 8, 512, 128

    def mk(kind):
        if kind == "bhsd":
            return _rand(b, h, s, d)
        return _rand(b, s, h, d).transpose(1, 2)

    if layout == "mixed":
        q, k, v = mk("bhsd"), mk("bshd"), mk("bhsd")
    else:
        q, k, v = (mk(layout) for _ in range(3))
    if layout != "bhsd":
        assert not q.is_contiguous() or not k.is_contiguous()

    got = _run(q, k, v, causal=causal)
    assert _err(got, _ref(q, k, v, causal)) < TOL


_LAYOUTS = ("bhsd", "bshd")


def _mk_layout(kind, b, h, s, d, *, gap=0):
    """A `(B, H, S, D)`-shaped tensor whose *memory* is laid out as `kind`.

    `gap` over-allocates the sequence axis and slices it back, so the strides
    outside the sliced axis no longer follow from the shape. That is the case
    a kernel deriving strides instead of reading them gets wrong, and it is
    invisible to a contiguous test: with `gap=0` a BHSD head stride is `s*d`,
    which is exactly what a derivation would guess.
    """
    if kind == "bhsd":
        return _rand(b, h, s + gap, d)[:, :, :s, :]
    return _rand(b, s + gap, h, d)[:, :s, :, :].transpose(1, 2)


@pytest.mark.parametrize("ql,kl,vl,ol", list(itertools.product(_LAYOUTS, repeat=4)))
def test_every_qkvo_layout_combination(ql, kl, vl, ol):
    """All 16 layouts of the four tensors, chosen independently.

    Q, K, V and O each carry their own three strides, so nothing forces them to
    agree -- and in practice they do not: PyTorch's SDPA shim hands down BHSD
    *views* of BSHD memory for the inputs while the caller may well have
    allocated O contiguously. `test_memory_layout_is_free` covered three of
    these sixteen.

    One build serves them all, because `strides_constexpr` is off in the parity
    ABI and the strides are runtime arguments -- so this is 16 dispatches over
    two compiles, not sixteen compiles.
    """
    b, h, s, d = 2, 4, 256, 64
    q = _mk_layout(ql, b, h, s, d)
    k = _mk_layout(kl, b, h, s, d)
    v = _mk_layout(vl, b, h, s, d)
    o = _mk_layout(ol, b, h, s, d)
    _run(q, k, v, o=o)
    assert _err(o, _ref(q, k, v)) < TOL


@pytest.mark.parametrize("kind", _LAYOUTS)
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_layout_with_gapped_outer_strides(kind, causal):
    """Strides that do not follow from the shape must still be honoured.

    Every layout test above allocates exactly what it uses, so a head stride is
    always `seq * head_dim` (BHSD) or `head_dim` (BSHD) -- both of which a
    kernel could have *derived* rather than read. Slicing an over-allocated
    sequence axis breaks that coincidence on all four tensors at once.
    """
    b, h, s, d = 2, 4, 256, 64
    q, k, v = (_mk_layout(kind, b, h, s, d, gap=37) for _ in range(3))
    o = _mk_layout(kind, b, h, s, d, gap=37)
    assert q.stride(1) not in (s * d, d) or q.stride(0) != h * s * d, "the gap must actually perturb a stride"
    _run(q, k, v, causal=causal, o=o)
    assert _err(o, _ref(q, k, v, causal)) < TOL


@pytest.mark.parametrize("ql,kl", list(itertools.product(_LAYOUTS, repeat=2)))
def test_layout_combinations_under_gqa(ql, kl):
    """K/V carry `num_kv_heads`, so their head stride differs from Q's by shape.

    Worth its own case because the two families of stride are derived from
    different head counts: a kernel that reused Q's head stride for K would
    pass every MHA layout test here and fail only under GQA.
    """
    b, hq, hk, s, d = 2, 8, 2, 256, 64
    q = _mk_layout(ql, b, hq, s, d)
    k = _mk_layout(kl, b, hk, s, d)
    v = _mk_layout(kl, b, hk, s, d)
    got = _run(q, k, v, causal=True)
    assert _err(got, _ref(q, k, v, causal=True, gqa=True)) < TOL


def test_stride_slots_are_batch_head_seq():
    """The three stride slots must mean (batch, head, seq), in that order.

    Swapping head with seq produces finite garbage and never faults, which is
    why this is asserted rather than left to the tolerance tests: a kernel that
    read the slots in the wrong order would still pass `test_memory_layout` for
    a *square* case where the two happen to coincide. Here `h != s`, so a swap
    cannot survive.
    """
    b, h, s, d = 2, 4, 256, 64
    assert h != s, "the point of this test is that the two axes differ"
    q, k, v = (_rand(b, h, s, d) for _ in range(3))
    got = _run(q, k, v)
    assert _err(got, _ref(q, k, v)) < TOL


@pytest.mark.parametrize("nhq,nhk", [(8, 8), (8, 4), (8, 2), (8, 1)], ids=["mha", "gqa2", "gqa4", "mqa"])
def test_gqa_mqa(nhq, nhk):
    b, s, d = 2, 512, 128
    q = _rand(b, nhq, s, d)
    k, v = (_rand(b, nhk, s, d) for _ in range(2))
    got = _run(q, k, v)
    assert _err(got, _ref(q, k, v, gqa=nhq != nhk)) < TOL


@pytest.mark.parametrize("scale", [0.05, 0.125, 1.0 / math.sqrt(128), 0.5])
def test_runtime_sm_scale(scale):
    """`sm_scale` is an argument, not `1/sqrt(head_dim)` folded at build time.

    It has to be: under a padded head the compiled tile is not the real extent,
    so a builder-derived scale is simply the wrong number.
    """
    b, h, s, d = 2, 8, 512, 128
    q, k, v = (_rand(b, h, s, d) for _ in range(3))
    got = _run(q, k, v, scale=scale)
    assert _err(got, _ref(q, k, v, scale=scale)) < TOL


def test_logsumexp_matches_torch():
    b, h, s, d = 2, 4, 512, 128
    q, k, v = (_rand(b, h, s, d) for _ in range(3))
    lse = torch.empty(b, h, s, device="cuda", dtype=torch.float32)
    _run(q, k, v, lse=lse)
    want = torch.logsumexp(
        (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(d),
        dim=-1,
    )
    assert (lse - want).abs().max().item() < 5e-2


# ---------------------------------------------------------------------------
# P1: runtime hdim, padded head, asymmetric hdim
# ---------------------------------------------------------------------------

_LADDER_CASES = [16, 17, 32, 33, 40, 48, 64, 80, 96, 100, 113, 128, 129, 160, 192, 200, 224, 256, 300, 384, 448, 512]


def _padded_qkv(b, h, s, hdim, hdim_v, poison=None):
    """Q/K/V whose D pitch is 8-aligned, optionally with a poisoned pad.

    The 8-element pitch is the alignment contract: loads are 8 wide, so the
    chunk containing `hdim` must land inside the allocation. What the pad
    *contains* is deliberately not part of the contract, which is what the
    poison tests here exist to hold the kernel to.
    """
    pitch = (max(hdim, hdim_v) + 7) // 8 * 8

    def mk(real):
        t = _rand(b, h, s, pitch)
        if poison is not None:
            t[..., real:] = poison
        return t[..., :real]

    return mk(hdim), mk(hdim), mk(hdim_v), pitch


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("hdim", _LADDER_CASES)
def test_ladder_matches_sdpa(hdim, causal):
    """Every head_dim in 16..128, served by the tile above it."""
    b, h, s = 2, 8, 512
    q, k, v, pitch = _padded_qkv(b, h, s, hdim, hdim)
    o = torch.empty(b, h, s, pitch, device="cuda", dtype=DT)[..., :hdim]
    _run(q, k, v, causal=causal, o=o)
    assert _err(o, _ref(q, k, v, causal)) < TOL


@pytest.mark.parametrize(
    "hdim,hdim_v", [(128, 64), (96, 48), (64, 32), (128, 96), (192, 128), (256, 128), (384, 256), (512, 256)]
)
def test_asymmetric_hdim(hdim, hdim_v):
    """`hdim_qk != hdim_vo`. The reference asm ships this shape as hd192/hd128."""
    b, h, s = 2, 8, 512
    q, k, v, pitch = _padded_qkv(b, h, s, hdim, hdim_v)
    o = torch.empty(b, h, s, pitch, device="cuda", dtype=DT)[..., :hdim_v]
    _run(q, k, v, o=o)
    assert _err(o, _ref(q, k, v)) < TOL


@pytest.mark.parametrize("poison", [0.0, 1e4, float("inf"), float("nan")], ids=["zero", "big", "inf", "nan"])
@pytest.mark.parametrize("hdim", [17, 33, 80, 100, 113, 129, 193, 225, 241])
def test_padded_head_ignores_pad_contents(hdim, poison):
    """The answer must not depend on what sits in the D-axis padding.

    **This is the test that found the one real bug in P1.** Masking Q alone
    makes the padded columns contribute `0 * pad`, which is 0 for any finite
    pad -- so `big` and `zero` passed while `nan` and `inf` produced NaN, since
    `0 * NaN` is NaN. K carries its own mask now (`ParityKvLdsToVgprLoader`);
    without it these two ids fail and the other two do not, which is why the
    poison values are parametrized rather than folded into one case.

    The wide `hdim` values are here for `HDIM_QK_FLOOR`, which lets K skip the
    steps it knows are real. Each is one past a rung -- 129, 193, 225, 241 sit
    at `floor + 1` for the 192, 224, 256 and 256 tiles -- so all but the two
    surviving steps are skipped and any off-by-one in the floor lets poison
    straight through.
    """
    b, h, s = 2, 8, 256
    q, k, v, pitch = _padded_qkv(b, h, s, hdim, hdim, poison=poison)
    o = torch.empty(b, h, s, pitch, device="cuda", dtype=DT)[..., :hdim]
    _run(q, k, v, o=o)
    err = _err(o, _ref(q, k, v))
    assert math.isfinite(err) and err < TOL


# The input contract is 8xD even though the compiled tiles are 32xD: loads and
# stores are 8 columns wide, so a head_dim that is a multiple of 8 is a whole
# number of chunks and the kernel never touches a column it was not given.
# Every multiple of 8 the ladder can reach, so a rung that mishandles its
# sub-8-grid widths cannot hide behind the ones the suite happens to name.
_GRID8 = list(range(8, 513, 8))


@pytest.mark.parametrize("hdim", _GRID8)
def test_grid8_contiguous_is_exact_and_writes_nothing_past_o(hdim):
    """A plainly contiguous 8xD tensor -- no padded view -- must just work.

    Separate from `test_ladder_matches_sdpa` because that one allocates through
    `_padded_qkv`, so every tensor it builds is a view into a wider allocation
    and the D pitch is 8-aligned *by construction*. That is the easy case and
    it cannot fail the way a tight `(B, H, S, 24)` can. What a caller actually
    passes is `torch.randn(b, h, s, hdim)`, whose pitch is `hdim` itself.

    The extra O row is a canary: it is contiguous with the last real row, so a
    tail chunk overrunning the final row lands in it and nowhere else.
    """
    b, h, s = 1, 4, 256
    q, k, v = (_rand(b, h, s, hdim) for _ in range(3))
    assert q.stride(2) == hdim, "the point of this test is a tight pitch"
    sentinel = -12345.0
    obuf = torch.full((b, h, s + 1, hdim), sentinel, device="cuda", dtype=DT)
    o = obuf[:, :, :s, :]
    _run(q, k, v, o=o)
    assert torch.all(obuf[:, :, s, :] == sentinel), "a store ran past the last O row"
    assert _err(o, _ref(q, k, v)) < TOL


def test_tight_odd_hdim_is_refused_not_corrupted():
    """An odd head_dim in a tight allocation has nowhere to put the tail chunk.

    `ceil8(100)` is 104, so the kernel touches four columns that belong to the
    next row. Refusing is the contract; the alternative is a wrong answer in a
    tensor the caller never suspected.
    """
    b, h, s, hdim = 1, 4, 256, 100
    q, k, v = (_rand(b, h, s, hdim) for _ in range(3))
    o = torch.empty(b, h, s, hdim, device="cuda", dtype=DT)
    with pytest.raises(ValueError, match="not a multiple of 8"):
        _run(q, k, v, o=o)


def test_odd_hdim_bshd_without_slack_is_refused():
    """BSHD hides the overrun from a pitch check, so the check is not a pitch.

    Heads of one token are adjacent in BSHD, so the gap after a D row is `hdim`
    itself and there is no slack -- while `stride(2)` is `num_heads * hdim`,
    which is a tidy multiple of 8 whenever `num_heads` is even. Checking the
    pitch alone (what the gfx1201 interface does) would accept this and let the
    tail chunk write into the next head.
    """
    b, h, s, hdim = 1, 4, 256, 100
    assert (h * hdim) % 8 == 0, "the case is only interesting when the pitch looks fine"
    q, k, v = (torch.randn(b, s, h, hdim, device="cuda", dtype=DT).transpose(1, 2) for _ in range(3))
    o = torch.empty(b, s, h, hdim, device="cuda", dtype=DT).transpose(1, 2)
    assert q.stride(2) % 8 == 0, "the pitch check would pass here"
    with pytest.raises(ValueError, match="unused element"):
        _run(q, k, v, o=o)


def test_padded_head_never_writes_past_hdim_vo():
    """O's D-tail chunk may spill into the caller's pad, but not past it.

    The store is 128 bits and cannot be split, so a chunk straddling `hdim_vo`
    writes into the allocation's own padding -- permitted, and the reason the
    pitch contract exists. A chunk starting at or past `hdim_vo` must be
    dropped entirely; this pins that by checking the *next row* is untouched.
    """
    b, h, s, hdim = 1, 4, 256, 100
    pitch = 104
    q, k, v, _ = _padded_qkv(b, h, s, hdim, hdim)
    full = torch.full((b, h, s, pitch), -7.0, device="cuda", dtype=DT)
    _run(q, k, v, o=full[..., :hdim])
    # Columns from ceil8(hdim) on were never in any store chunk.
    untouched = full[..., (hdim + 7) // 8 * 8 :]
    assert torch.all(untouched == -7.0), "store ran past the 8-aligned chunk containing hdim_vo"


# ---------------------------------------------------------------------------
# Policy / knob resolution (host-side, no GPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hdim,want",
    [
        (1, 32),
        (16, 32),
        (32, 32),
        (33, 64),
        (64, 64),
        (65, 96),
        (96, 96),
        (97, 128),
        (128, 128),
        (129, 160),
        (160, 160),
        (161, 192),
        (192, 192),
        (193, 224),
        (224, 224),
        (225, 256),
        (256, 256),
        (257, 384),
        (384, 384),
        (385, 512),
        (512, 512),
    ],
)
def test_tile_width_rounds_up(hdim, want):
    assert tile_width_for(hdim) == want


def test_tile_width_rejects_past_the_widest_tile():
    """Every planned rung is built now, so only "too wide" is left to report.

    `LADDER_PLANNED` is empty, which makes the `NotImplementedError` branch of
    `tile_width_for` unreachable rather than wrong -- it stays for the next
    rung that gets designed before it is built.
    """
    assert not LADDER_PLANNED
    with pytest.raises(ValueError, match="exceeds the widest tile"):
        tile_width_for(1024)


def test_hdim_qk_floor_is_the_ladder_gap():
    """The floor is the rung below, and only for a tile the ladder chose.

    Pinning `block_dmodel` is allowed to be arbitrarily wide for the head_dim
    -- 256 for a head_dim of 64 is legal -- so no floor can be claimed there,
    and claiming one would silently unmask the pad.
    """
    k = fmha_knobs("gfx950")
    for hdim, want in [(64, 32), (65, 64), (129, 128), (240, 224), (500, 384)]:
        got = k.resolve(FmhaInputMetadata(num_heads=8, head_dim=hdim)).hdim_qk_floor
        assert got == want, f"head_dim {hdim}: floor {got}, want {want}"
    assert k.resolve(FmhaInputMetadata(num_heads=8, head_dim=17)).hdim_qk_floor == 0  # narrowest rung
    pinned = replace(k, block_dmodel=256).resolve(FmhaInputMetadata(num_heads=8, head_dim=64))
    assert pinned.hdim_qk_floor == 0


def test_hdim_below_floor_is_refused_not_computed():
    """A call narrower than the build's floor must raise, not return numbers.

    The kernel leaves the low D columns unmasked on the strength of the floor,
    so this call would reduce over the caller's padding. It is the one way to
    reach that state, and it is a host-side check away from being an error.
    """
    b, h, s, hdim = 1, 4, 256, 240
    q, k, v, pitch = _padded_qkv(b, h, s, hdim, hdim)
    fn = build(num_heads=h, head_dim=hdim, head_dim_v=hdim, causal=False, dtype_str="bf16", num_kv_heads=h)
    # 104, not an odd width: a tight odd tensor would trip the D-pitch guard
    # first and this test would pass without ever reaching the floor check.
    narrow = [t[..., :104].contiguous() for t in (q, k, v)]
    o = torch.empty(b, h, s, 104, device="cuda", dtype=DT)
    with pytest.raises(ValueError, match=r"serves hdim_qk in \(224, 256\]"):
        fn(*narrow, o, b, s, seqlen_k=s, scale=None, lse=None)


def test_padded_head_is_derived_not_guessed():
    for hdim in LADDER:
        assert fmha_knobs("gfx950").resolve(FmhaInputMetadata(num_heads=8, head_dim=hdim)).padded_head is False
    assert fmha_knobs("gfx950").resolve(FmhaInputMetadata(num_heads=8, head_dim=40)).padded_head is True
    assert fmha_knobs("gfx950").resolve(FmhaInputMetadata(num_heads=8, head_dim=64, head_dim_v=32)).padded_head is True


# ---------------------------------------------------------------------------
# P7: what the knob re-sweep found
# ---------------------------------------------------------------------------


def test_rows_per_wave_cannot_exceed_the_mfma_m_extent():
    """`BLOCK_M` is derived, and pinning a larger one is a wrong answer.

    A wave holds at most the MFMA's 32 rows, so `BLOCK_M` is really
    `q_tiles * 32`; a bigger one builds a kernel whose helpers address rows its
    accumulator does not have, and it **does not fail**. The sweep found twelve
    such points -- all head_dim 512 at 8 waves with `vo_shards > 1`, so 64 or
    128 rows per wave -- each returning finite garbage at 0.15-0.28 relative
    error.

    Sharding is where it bites: `vo_shards` divides `q_tiles` while `block_m`
    stays whatever was pinned, so a geometry that was consistent unsharded
    silently stops being so. That is why the check is on the derived value and
    not on the pinned tuple, which `_check_helpers_support_geometry` already
    covers and which looked entirely legal here.
    """
    meta = FmhaInputMetadata(num_heads=8, head_dim=512, causal=False)
    bad = fmha_knobs("gfx950", num_waves=8, block_m=256, block_n=64, head_dim_granule=64, d_stages=2, vo_shards=2)
    with pytest.raises(ValueError, match="rows per wave"):
        bad.resolve(meta)
    # Fewer rows per wave stays legal: the wave runs a full 32-row MFMA and
    # discards what it does not own, which is wasteful and not wrong.
    fmha_knobs("gfx950", num_waves=8, block_m=128, block_n=64, head_dim_granule=64).resolve(
        FmhaInputMetadata(num_heads=8, head_dim=128, causal=False)
    )


@pytest.mark.parametrize("lazy_rescale", [True, False])
def test_one_accumulator_builds_on_both_rescale_paths(lazy_rescale):
    """head_dim 32 is the only width with `D_CHUNKS == 1`, and it has two paths.

    `_anchor_v_o` asks an inline asm for a struct of `D_CHUNKS` outputs, and at
    one output LLVM does not diagnose it -- it aborts with an `UNREACHABLE`,
    killing the process rather than raising. The lazy path was fixed when
    head_dim 32 was built; `rescale_o`, which only `lazy_rescale=False` takes,
    reaches the production anchor and still aborted. The sweep found it by
    being the first thing to build that combination.
    """
    b, h, s = 1, 4, 512
    q, k, v = (_rand(b, h, s, 32) for _ in range(3))
    o = torch.empty_like(q)
    meta = FmhaInputMetadata(num_heads=h, head_dim=32, causal=False)
    from flash_attn_func_gfx950 import build_flash_attn_func_gfx950_module_primary as build_primary

    fn = build_primary(meta, fmha_knobs("gfx950", lazy_rescale=lazy_rescale).resolve(meta))
    fn(q, k, v, o, b, s, seqlen_k=s, scale=None, lse=None)
    assert _err(o, _ref(q, k, v)) < TOL


@pytest.mark.parametrize("hdim", [64, 256])
def test_lpt_tile_order_is_bit_identical(hdim):
    """Reversing the causal Q-block order must change nothing but the schedule.

    It is a bijection over the same index set, which is what makes it safe to
    leave as a tuning knob. It is also worth **0.0% on gfx950** across every
    width and shape measured, so it ships off -- the same answer the gfx1201
    dK/dV port reached, and for a related reason: with 8 XCDs and this many
    workgroups the tail imbalance the reordering targets is already absorbed.
    """
    b, h, s = 1, 4, 1024
    q, k, v = (_rand(b, h, s, hdim) for _ in range(3))
    meta = FmhaInputMetadata(num_heads=h, head_dim=hdim, causal=True)
    from flash_attn_func_gfx950 import build_flash_attn_func_gfx950_module_primary as build_primary

    outs = []
    for lpt in (False, True):
        o = torch.empty_like(q)
        build_primary(meta, fmha_knobs("gfx950", lpt_tile_order=lpt).resolve(meta))(
            q, k, v, o, b, s, seqlen_k=s, scale=None, lse=None
        )
        outs.append(o)
    assert torch.equal(outs[0], outs[1])
    assert fmha_knobs("gfx950").resolve(meta).build_traits(meta).LPT_TILE_ORDER is False


# ---------------------------------------------------------------------------
# P6: dropout (philox)
# ---------------------------------------------------------------------------


def _run_dropout(q, k, v, p, seed=1234, causal=False, **geom):
    b, h, s, d = q.shape
    o = torch.empty_like(q)
    fn = build(
        num_heads=h,
        head_dim=d,
        causal=causal,
        dropout=p is not None,
        dtype_str="bf16",
        num_kv_heads=k.shape[1],
        **geom,
    )
    fn(q, k, v, o, b, s, seqlen_k=k.shape[2], scale=None, lse=None, dropout_p=p, philox_seed=seed, philox_offset2=0)
    return o


@pytest.mark.parametrize("hdim", [64, 384], ids=["dualwave", "wide"])
def test_dropout_p_zero_is_bit_identical(hdim):
    """`p = 0` must leave the answer alone, exactly.

    The threshold for `p = 0` keeps every random but the single value
    `-2**31`, and the survivor scale is 1, so the only difference from a build
    without dropout is arithmetic that must cancel. Bitwise, not tolerance:
    this is the cheapest way to catch the scale or the mask being applied in
    the wrong place.
    """
    b, h, s = 1, 4, 512
    q, k, v = (_rand(b, h, s, hdim) for _ in range(3))
    assert torch.equal(_run_dropout(q, k, v, 0.0), _run_dropout(q, k, v, None))


@pytest.mark.parametrize("hdim", [64, 384], ids=["dualwave", "wide"])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_dropout_is_a_function_of_the_seed(hdim, causal):
    """Same seed, same mask; different seed, different mask -- and never NaN.

    Dropout composes with causal masking, unlike bias: a positional mask says
    which keys exist and dropout thins the ones that do, so there is nothing to
    reconcile.
    """
    b, h, s = 1, 4, 512
    q, k, v = (_rand(b, h, s, hdim) for _ in range(3))
    a = _run_dropout(q, k, v, 0.5, seed=1234, causal=causal)
    again = _run_dropout(q, k, v, 0.5, seed=1234, causal=causal)
    other = _run_dropout(q, k, v, 0.5, seed=999, causal=causal)
    assert torch.equal(a, again)
    assert not torch.equal(a, other)
    assert not torch.isnan(a.float()).any()


def test_dropout_mask_does_not_depend_on_the_tiling():
    """**The reproducibility contract**, and it is invisible at one tile size.

    A mask generated here is regenerated by the backward pass and by the debug
    mask kernel, so it must be a function of element coordinates alone --
    `grid_plane` is handed `max_seqlen_q`/`max_seqlen_k`, never `BLOCK_M` or
    `BLOCK_N`. From this phase onward that is a constraint on the tuner, and
    this is the test that enforces it: the same problem built with two
    *different, both-supported* wave geometries must give bit-identical output.

    The no-dropout control matters. It shows the two geometries agree anyway,
    so a difference under dropout could only come from the mask.
    """
    b, h, s, d = 1, 4, 512, 64
    q, k, v = (_rand(b, h, s, d) for _ in range(3))
    fam_a = dict(num_waves=8, block_m=256, block_n=64, head_dim_granule=64)
    fam_b = dict(num_waves=4, block_m=128, block_n=64, head_dim_granule=64)
    assert torch.equal(_run_dropout(q, k, v, None, **fam_a), _run_dropout(q, k, v, None, **fam_b)), "control"
    assert torch.equal(_run_dropout(q, k, v, 0.5, **fam_a), _run_dropout(q, k, v, 0.5, **fam_b))


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_dropout_keep_rate_matches_p(p):
    """The measured keep rate must be `1 - p`, read straight out of O.

    Constructed so the answer is the rate itself: `Q = K = 0` makes every score
    equal, so the softmax is uniform at `1/S` and the undropped denominator is
    1; with `V[..., 0] = 1` the first output column is
    `(kept / S) / (1 - p)`, whose expectation is exactly 1. Multiplying back by
    `1 - p` recovers the keep rate directly, with no reference kernel.
    """
    b, h, s, d = 1, 4, 2048, 64
    q = torch.zeros(b, h, s, d, device="cuda", dtype=DT)
    k = torch.zeros(b, h, s, d, device="cuda", dtype=DT)
    v = torch.zeros(b, h, s, d, device="cuda", dtype=DT)
    v[..., 0] = 1.0
    got = _run_dropout(q, k, v, p, seed=7)
    keep = (got[..., 0].float() * (1.0 - p)).mean().item()
    assert abs(keep - (1.0 - p)) < 0.02, f"keep rate {keep:.4f}, want {1 - p:.4f}"


def test_dropout_expectation_is_unbiased():
    """Averaging over seeds must converge to the undropped answer.

    This is what the `1/(1-p)` survivor scale is for, and the only test here
    that would catch it being dropped or applied twice -- every other dropout
    test passes with any constant scale. The bound is the analytic standard
    error: per-element deviation at `p = 0.5` is about the size of the output,
    so over `n` seeds the mean should sit within a few `1/sqrt(n)`.
    """
    b, h, s, d = 1, 4, 512, 64
    q, k, v = (_rand(b, h, s, d) for _ in range(3))
    n = 32
    acc = torch.zeros(b, h, s, d, device="cuda", dtype=torch.float32)
    for seed in range(n):
        acc += _run_dropout(q, k, v, 0.5, seed=1000 + seed).float()
    acc /= n
    want = _ref(q, k, v)
    rel = ((acc - want).abs().mean() / want.abs().mean()).item()
    assert rel < 3.0 / n**0.5, f"mean over {n} seeds deviates {rel:.4f}"


def test_dropout_tensor_requirements_are_checked():
    b, h, s, d = 1, 4, 256, 64
    q, k, v = (_rand(b, h, s, d) for _ in range(3))
    o = torch.empty_like(q)
    with pytest.raises(ValueError, match="requires dropout_p"):
        build(num_heads=h, head_dim=d, causal=False, dropout=True, dtype_str="bf16", num_kv_heads=h)(
            q, k, v, o, b, s, seqlen_k=s, scale=None, lse=None
        )
    with pytest.raises(ValueError, match="not compiled for dropout"):
        build(num_heads=h, head_dim=d, causal=False, dtype_str="bf16", num_kv_heads=h)(
            q, k, v, o, b, s, seqlen_k=s, scale=None, lse=None, dropout_p=0.5
        )


# ---------------------------------------------------------------------------
# P5: bias
# ---------------------------------------------------------------------------


def _run_bias(q, k, v, bias):
    b, h, s, d = q.shape
    o = torch.empty_like(q)
    fn = build(num_heads=h, head_dim=d, causal=False, bias=True, dtype_str="bf16", num_kv_heads=k.shape[1])
    fn(q, k, v, o, b, s, seqlen_k=k.shape[2], scale=None, lse=None, bias=bias)
    return o


@pytest.mark.parametrize("hdim", [64, 128, 256, 384, 512], ids=["d64", "d128", "d256", "wide384", "wide512"])
def test_bias_matches_sdpa(hdim):
    """A `(B, H, Sq, Sk)` bias added to the scores, on both kernel bodies.

    Parametrized across the break at 256 because the two bodies reach the bias
    through different code: the dual-wave loop masks only its edge tiles, so
    its interior tiles take the bias through `bias_to_lists`, while the wide
    body masks every tile and takes it through the `seq_pad_mask_if_needed`
    override. A build that wired only one would pass half of this.
    """
    b, h, s = 1, 2, 512
    q, k, v = (_rand(b, h, s, hdim) for _ in range(3))
    bias = _rand(b, h, s, s)
    got = _run_bias(q, k, v, bias)
    want = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=bias.float())
    assert _err(got, want) < TOL


@pytest.mark.parametrize("hdim", [64, 384], ids=["dualwave", "wide"])
def test_bias_minus_inf_never_produces_nan(hdim):
    """`-inf` in the bias is how a caller says "never attend here".

    **This is the first configuration in which the `-inf - -inf -> NaN` bug is
    reachable at all**, which is why the test belongs to this phase and not an
    earlier one. A whole row of `-inf` drives that row's max to `-inf`, the
    denominator to zero, and a normalisation to `0/0`.

    It does not, because `ParitySoftmaxHelper` seeds `reduce_max` at `-3.0e38`
    rather than `-inf`, so no lane ever holds `-inf` as a max. The assertion on
    `isnan` is the point; a tolerance check alone would not tell the two apart.

    The one-tile case is the milder half: the row keeps a finite max from other
    tiles and the dead tile must simply contribute nothing.
    """
    b, h, s = 1, 4, 512
    q, k, v = (_rand(b, h, s, hdim) for _ in range(3))

    tile = torch.zeros(b, h, s, s, device="cuda", dtype=DT)
    tile[:, :, :, 64:128] = float("-inf")
    got = _run_bias(q, k, v, tile)
    assert not torch.isnan(got.float()).any()
    want = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=tile.float())
    assert _err(got, want) < TOL

    rows = torch.zeros(b, h, s, s, device="cuda", dtype=DT)
    dead = [7, 300]
    rows[:, :, dead, :] = float("-inf")
    got = _run_bias(q, k, v, rows)
    assert not torch.isnan(got.float()).any(), "a fully -inf bias row produced NaN"
    assert torch.count_nonzero(got[:, :, dead, :]) == 0, "a row with no live key must be exactly zero"
    live = torch.ones(s, dtype=torch.bool)
    live[dead] = False
    want = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=rows.float())
    assert _err(got[:, :, live], want[:, :, live]) < TOL


def test_bias_and_causal_are_rejected():
    """Undefined, not unimplemented -- the same call AOTriton and torch make."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        build(num_heads=8, head_dim=64, causal=True, bias=True, dtype_str="bf16")
    with pytest.raises(ValueError, match="mutually exclusive"):
        build(num_heads=8, head_dim=64, causal=True, window=True, bias=True, dtype_str="bf16")


def test_bias_tensor_presence_is_checked_both_ways():
    """A bias build needs one; a plain build must refuse one.

    Silently ignoring a bias returns dense attention -- right shape, wrong
    answer -- and a bias is only ever passed by a caller who believes it is
    being applied.
    """
    b, h, s, d = 1, 4, 256, 64
    q, k, v = (_rand(b, h, s, d) for _ in range(3))
    o = torch.empty_like(q)
    biased = build(num_heads=h, head_dim=d, causal=False, bias=True, dtype_str="bf16", num_kv_heads=h)
    with pytest.raises(ValueError, match="requires a"):
        biased(q, k, v, o, b, s, seqlen_k=s, scale=None, lse=None)
    plain = build(num_heads=h, head_dim=d, causal=False, dtype_str="bf16", num_kv_heads=h)
    with pytest.raises(ValueError, match="not compiled for bias"):
        plain(q, k, v, o, b, s, seqlen_k=s, scale=None, lse=None, bias=_rand(b, h, s, s))


# ---------------------------------------------------------------------------
# P4: varlen -- the five VarlenBits configurations
# ---------------------------------------------------------------------------

_VL_H, _VL_D = 4, 64
_VL_Q = [128, 377, 64, 500]
_VL_K = [100, 300, 64, 400]
_VL_N = len(_VL_Q)
_VL_MAX = max(_VL_Q)


def _i32(x):
    return torch.tensor(list(x), device="cuda", dtype=torch.int32)


def _cumsum(lens):
    out, run = [0], 0
    for value in lens:
        run += value
        out.append(run)
    return out


def _sdpa_bottom_right(q, k, v, causal):
    """SDPA with the kernel's causal convention: `col <= row + (Sk - Sq)`.

    `torch`'s `is_causal` is **top-left** aligned, and the two agree only when
    `Sq == Sk` -- which every dense test here happens to satisfy and no varlen
    test does. Using `is_causal` made two of the five modes look broken.
    """
    if not causal:
        return F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    sq, sk = q.shape[2], k.shape[2]
    rows = torch.arange(sq, device="cuda").view(-1, 1)
    cols = torch.arange(sk, device="cuda").view(1, -1)
    bias = torch.zeros(sq, sk, device="cuda", dtype=torch.float32)
    bias.masked_fill_(cols > rows + (sk - sq), float("-inf"))
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=bias)


def _run_varlen(q, k, v, o, bits, causal, batch_size, num_seqlens, **seqinfo):
    fn = build(
        num_heads=_VL_H,
        head_dim=_VL_D,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=_VL_H,
        varlen=True,
    )
    varlen = dict(bits=bits, max_seqlen_q=_VL_MAX, max_seqlen_k=_VL_MAX)
    varlen.update({name: t for name, t in seqinfo.items() if t is not None})
    fn(q, k, v, o, batch_size, _VL_MAX, seqlen_k=_VL_MAX, scale=None, lse=None, varlen=varlen, num_seqlens=num_seqlens)
    return o


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_varlen_compact(causal):
    """`0x0B0B` -- both sides packed, positions reused from the length array."""
    import fmha_abi_gfx1201 as _abi

    cu = _cumsum(_VL_Q)
    total = cu[-1]
    q, k, v = (_rand(1, _VL_H, total, _VL_D) for _ in range(3))
    o = torch.empty_like(q)
    side = _abi.VARLEN_COMPACT_SIDE
    _run_varlen(q, k, v, o, _abi.varlen_bits(side, side), causal, 1, _VL_N, seqinfo_q0=_i32(cu), seqinfo_k0=_i32(cu))
    for i in range(_VL_N):
        a, b = cu[i], cu[i + 1]
        want = _sdpa_bottom_right(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], causal)
        assert _err(o[:, :, a:b], want) < TOL, f"sequence {i}"


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_varlen_padded(causal):
    """`0x0202` -- a real batch axis with short sequences, lengths from cu."""
    import fmha_abi_gfx1201 as _abi

    cu = _cumsum(_VL_Q)
    q, k, v = (_rand(_VL_N, _VL_H, _VL_MAX, _VL_D) for _ in range(3))
    o = torch.zeros_like(q)
    side = _abi.VARLEN_PADDED_SIDE
    _run_varlen(q, k, v, o, _abi.varlen_bits(side, side), causal, _VL_N, 0, seqinfo_q0=_i32(cu), seqinfo_k0=_i32(cu))
    for i, length in enumerate(_VL_Q):
        want = _sdpa_bottom_right(q[i : i + 1, :, :length], k[i : i + 1, :, :length], v[i : i + 1, :, :length], causal)
        assert _err(o[i : i + 1, :, :length], want) < TOL, f"sequence {i}"


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_varlen_strided(causal):
    """`0x1313` -- packed, but positions from their own array, with gaps.

    The only configuration that reads a position array on both sides, and the
    one that separates POSITION from LENGTH: the starts here are deliberately
    not the prefix sums, so a decoder that reused the length array would land
    in the wrong rows.
    """
    import fmha_abi_gfx1201 as _abi

    cu = _cumsum(_VL_Q)
    starts = [0, 200, 700, 900]
    assert starts != cu[:-1], "the gaps are the point of this case"
    total = starts[-1] + _VL_Q[-1] + 64
    q, k, v = (_rand(1, _VL_H, total, _VL_D) for _ in range(3))
    o = torch.empty_like(q)
    pos = _i32(starts + [starts[-1] + _VL_Q[-1]])
    side = _abi.VARLEN_STRIDED_SIDE
    _run_varlen(
        q,
        k,
        v,
        o,
        _abi.varlen_bits(side, side),
        causal,
        1,
        _VL_N,
        seqinfo_q0=_i32(cu),
        seqinfo_q1=pos,
        seqinfo_k0=_i32(cu),
        seqinfo_k1=pos,
    )
    for i, length in enumerate(_VL_Q):
        a = starts[i]
        want = _sdpa_bottom_right(q[:, :, a : a + length], k[:, :, a : a + length], v[:, :, a : a + length], causal)
        assert _err(o[:, :, a : a + length], want) < TOL, f"sequence {i}"


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_varlen_seqused_k_on_packed_kv(causal):
    """`0x150B` -- packed Q, and a K side whose lengths are individual.

    `seqused_k` on packed KV must use ARRAY rather than REUSE: its length array
    holds individual lengths, so it is not a position array and there is
    nothing to reuse. That is plan section 1.4 showing up in the encoding.
    """
    import fmha_abi_gfx1201 as _abi

    cu, kcu = _cumsum(_VL_Q), _cumsum(_VL_K)
    q = _rand(1, _VL_H, cu[-1], _VL_D)
    k, v = (_rand(1, _VL_H, kcu[-1], _VL_D) for _ in range(2))
    o = torch.empty_like(q)
    _run_varlen(
        q,
        k,
        v,
        o,
        _abi.varlen_bits(_abi.VARLEN_COMPACT_SIDE, _abi.VARLEN_SEQUSED_PACKED_SIDE),
        causal,
        1,
        _VL_N,
        seqinfo_q0=_i32(cu),
        seqinfo_k0=_i32(_VL_K),
        seqinfo_k1=_i32(kcu),
    )
    for i in range(_VL_N):
        a, b = cu[i], cu[i + 1]
        c, d = kcu[i], kcu[i + 1]
        want = _sdpa_bottom_right(q[:, :, a:b], k[:, :, c:d], v[:, :, c:d], causal)
        assert _err(o[:, :, a:b], want) < TOL, f"sequence {i}"


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_varlen_seqused_k_on_bhsd_cache(causal):
    """`0x040B` -- packed Q against a *batched* K cache, in one call.

    The case the two sides genuinely disagree: Q is stacked, so its batch index
    is 0 and its row offset large; K is not, so its batch index is `z` and its
    row offset zero. A kernel that let one side's batch index serve both reads
    batch 0 of the cache for every sequence -- which is what this did until
    `kv_batch_idx` was split out, and it was the only one of the five modes
    that noticed.
    """
    import fmha_abi_gfx1201 as _abi

    cu = _cumsum(_VL_Q)
    kmax = max(_VL_K)
    q = _rand(1, _VL_H, cu[-1], _VL_D)
    k, v = (_rand(_VL_N, _VL_H, kmax, _VL_D) for _ in range(2))
    o = torch.empty_like(q)
    _run_varlen(
        q,
        k,
        v,
        o,
        _abi.varlen_bits(_abi.VARLEN_COMPACT_SIDE, _abi.VARLEN_SEQUSED_CACHE_SIDE),
        causal,
        1,
        _VL_N,
        seqinfo_q0=_i32(cu),
        seqinfo_k0=_i32(_VL_K),
    )
    for i in range(_VL_N):
        a, b = cu[i], cu[i + 1]
        length = _VL_K[i]
        want = _sdpa_bottom_right(q[:, :, a:b], k[i : i + 1, :, :length], v[i : i + 1, :, :length], causal)
        assert _err(o[:, :, a:b], want) < TOL, f"sequence {i}"


def _ref_lse_bottom_right(q, k, v, causal):
    """`logsumexp` of the scaled scores, with the kernel's causal convention."""
    sq, sk = q.shape[2], k.shape[2]
    scores = (q.float() @ k.float().transpose(-1, -2)) * (q.shape[3] ** -0.5)
    if causal:
        rows = torch.arange(sq, device="cuda").view(-1, 1)
        cols = torch.arange(sk, device="cuda").view(1, -1)
        scores = scores.masked_fill(cols > rows + (sk - sq), float("-inf"))
    return torch.logsumexp(scores, dim=-1)


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("layout", ["HT", "TH"])
def test_varlen_logsumexp_layouts(layout, causal):
    """LSE under varlen: token pitch, row origin and `VarlenBits` bits 17:16.

    LSE is always *compact* -- it is the one tensor whose strides are a
    function of the bits rather than a free variable, which is why no
    `lse_stride` is passed. Compact is not fixed, though, and the production
    formula `q_head_idx * seq_len_v + q_row` hardcodes all three things that
    move: the pitch is `max_seqlen_q` where a stacked side runs to the batch
    total, the row origin is 0 where a packed sequence starts at `q_row_off`,
    and the layout is `_HT` written out, so `_TH` was silently ignored.

    `_TH` is the case that fails loudest if the bits are dropped, since it is
    a transpose rather than an offset -- reading it back with `_HT` indexing
    would give a different value for every element but the first.
    """
    import fmha_abi_gfx1201 as _abi

    bits_layout = _abi.VARLEN_LSE_LAYOUT_HT if layout == "HT" else _abi.VARLEN_LSE_LAYOUT_TH
    cu = _cumsum(_VL_Q)
    total = cu[-1]
    q, k, v = (_rand(1, _VL_H, total, _VL_D) for _ in range(3))
    o = torch.empty_like(q)
    lse = torch.zeros(_VL_H * total, device="cuda", dtype=torch.float32)
    fn = build(
        num_heads=_VL_H,
        head_dim=_VL_D,
        causal=causal,
        dtype_str="bf16",
        num_kv_heads=_VL_H,
        varlen=True,
        return_lse=True,
    )
    side = _abi.VARLEN_COMPACT_SIDE
    fn(
        q,
        k,
        v,
        o,
        1,
        _VL_MAX,
        seqlen_k=_VL_MAX,
        scale=None,
        lse=lse,
        varlen=dict(
            bits=_abi.varlen_bits(side, side, bits_layout),
            max_seqlen_q=_VL_MAX,
            max_seqlen_k=_VL_MAX,
            seqinfo_q0=_i32(cu),
            seqinfo_k0=_i32(cu),
        ),
        num_seqlens=_VL_N,
    )
    for i in range(_VL_N):
        a, b = cu[i], cu[i + 1]
        want = _ref_lse_bottom_right(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], causal)[0]
        for h in range(_VL_H):
            got = lse[h * total + a : h * total + b] if layout == "HT" else lse[a * _VL_H + h : b * _VL_H + h : _VL_H]
            assert (got - want[h]).abs().max().item() < 5e-2, f"sequence {i} head {h}"


def test_varlen_causal_defaults_to_cross_seqlen():
    """A causal varlen build must turn `cross_seqlen` on by itself.

    Q and K lengths come from independent arrays read at runtime, so nothing at
    build time knows whether they match; where `seqlen_k < seqlen_q`,
    bottom-right causal leaves the leading Q blocks with no live key and the
    kernel has to zero them. Two of the five modes returned wrong answers until
    this defaulted on. Still pinnable off, since it is not free.
    """
    causal_vl = fmha_knobs("gfx950", varlen=True).resolve(FmhaInputMetadata(num_heads=8, head_dim=64, causal=True))
    assert causal_vl.cross_seqlen is True
    full_vl = fmha_knobs("gfx950", varlen=True).resolve(FmhaInputMetadata(num_heads=8, head_dim=64, causal=False))
    assert full_vl.cross_seqlen is False
    dense = fmha_knobs("gfx950").resolve(FmhaInputMetadata(num_heads=8, head_dim=64, causal=True))
    assert dense.cross_seqlen is False
    pinned = fmha_knobs("gfx950", varlen=True, cross_seqlen=False)
    assert pinned.resolve(FmhaInputMetadata(num_heads=8, head_dim=64, causal=True)).cross_seqlen is False


def test_packed_varlen_draws_a_dropout_plane_per_sequence():
    """**Each sequence of a packed batch must get its own dropout mask.**

    The philox plane is `sequence * num_head_q + q_head_idx`, and the sequence
    index has to come from `seq_idx_i32` -- the workgroup's raw grid `z` --
    rather than from `batch_idx`. `decode_addressing` resolves `batch_idx` to
    **0** for every sequence of a packed layout, which is right for addressing
    (a packed tensor carries its origin in `row_off` instead) and wrong for
    this: it made all N sequences draw plane `0*H + h`, one shared mask.

    **The oracle is a dense call of batch N, sliced -- not N calls of batch
    1.** That distinction is the whole test. A `batch_size=1` call cannot
    express "sequence 3's plane" by construction: it has `batch_idx == 0` and
    so also draws plane 0. Comparing a packed call against N such calls is
    comparing two things that collapse the same way, and it reports agreement
    while both are wrong. A dense call of batch N has `batch_idx == z`, so its
    row-group `i` carries plane `i*H + h`, which is the thing worth matching.

    Measured before the fix, with equal lengths so a packed and a dense call
    hold the same rows -- per-sequence bitwise equality against each reference:

        vs N dense batch-1 calls : [True, True, True]     <- the degenerate one
        vs one dense batch-N call: [True, False, False]

    Sequence 0 agrees either way, because its correct plane *is* 0. That
    `[True, False, False]` is the fingerprint of the bug, and it inverts
    exactly when the plane moves to `seq_idx_i32`.

    This case had no numeric coverage at all before: no test in any of the
    three kernels combined varlen with dropout, which is why a defect visible
    in the first non-zero sequence of every packed call survived every gate.
    Bitwise rather than `allclose` for the usual varlen reason -- only the base
    address differs between a varlen workgroup and its dense counterpart.
    """
    import fmha_abi_gfx1201 as _abi

    h, d, n, length = 4, 64, 3, 128
    torch.manual_seed(7)
    cu = _cumsum([length] * n)
    # One pool laid out two ways. Equal lengths, so the packed (1, H, N*S, D)
    # and dense (N, H, S, D) views hold the same rows in the same order.
    qp, kp, vp = (_rand(1, h, cu[-1], d) for _ in range(3))
    to_b = (
        lambda t: t.view(1, h, n, length, d).permute(2, 0, 1, 3, 4).reshape(n, h, length, d).contiguous()
    )  # noqa: E731
    qb, kb, vb = to_b(qp), to_b(kp), to_b(vp)

    p, seed = 0.5, 1234
    dense_fn = build(num_heads=h, head_dim=d, causal=False, dtype_str="bf16", num_kv_heads=h, dropout=True)
    packed_fn = build(
        num_heads=h, head_dim=d, causal=False, dtype_str="bf16", num_kv_heads=h, dropout=True, varlen=True
    )
    # `philox_offset2`, matching `_run_dropout`, so the batch-1 arm below is
    # drawn from the same counter and the comparison is about the plane alone.
    drop = dict(dropout_p=p, philox_seed=seed, philox_offset2=0)

    o_dense = torch.zeros_like(qb)
    dense_fn(qb, kb, vb, o_dense, n, length, seqlen_k=length, **drop)

    o_packed = torch.zeros_like(qp)
    side = _abi.VARLEN_COMPACT_SIDE
    packed_fn(
        qp,
        kp,
        vp,
        o_packed,
        1,
        length,
        seqlen_k=length,
        # Both sides get the cumulative array: `0x0B0B` is packed on Q *and*
        # K, and the K side dereferences its own pointer -- omitting it is a
        # null read, not a fallback.
        varlen=dict(
            bits=_abi.varlen_bits(side, side),
            max_seqlen_q=length,
            max_seqlen_k=length,
            seqinfo_q0=_i32(cu),
            seqinfo_k0=_i32(cu),
        ),
        num_seqlens=n,
        **drop,
    )
    torch.cuda.synchronize()

    for i in range(n):
        assert torch.equal(o_packed[0, :, cu[i] : cu[i + 1], :], o_dense[i]), (
            f"sequence {i} of a packed batch does not match dense row-group {i}. If sequence 0 passes and the "
            f"rest fail, the plane has collapsed to `batch_idx` and every sequence is sharing one mask."
        )

    # The degenerate reference, asserted to *disagree*. Without this the test
    # would still pass if the plane collapsed and the dense reference collapsed
    # with it -- which is precisely how the original oracle came to assert the
    # bug.
    o_b1 = torch.stack([_run_dropout(qb[i : i + 1], kb[i : i + 1], vb[i : i + 1], p, seed=seed)[0] for i in range(n)])
    assert torch.equal(o_b1[0], o_dense[0]), "sequence 0 shares plane 0 with the batch-1 reference by construction"
    assert not torch.equal(o_b1[1], o_dense[1]), (
        "a batch-1 call and dense row-group 1 drew the same mask, so `batch_idx` is not distinguishing "
        "sequences even in the dense path -- this test's oracle would be vacuous"
    )


# ---------------------------------------------------------------------------
# Split-K: correct only in the production output layout
# ---------------------------------------------------------------------------


def _splitk_workspace(b, h, s, splits, head_dim):
    from kernels.attention.flash_attn_utils import dualwave_splitk_workspace_elems

    n = dualwave_splitk_workspace_elems(b, h, s, splits, head_dim=head_dim)
    return torch.zeros(n, device="cuda", dtype=torch.float32)


@pytest.mark.parametrize("splits", [2, 4])
def test_splitk_matches_sdpa_in_production_layout(splits):
    """Split-K works, but only where the heads are packed adjacently.

    There was no device-level split-K test at all before this -- only a traits
    comparison, which checks the *configuration* matches production and never
    runs the kernel. That is why a wrong answer went unnoticed: `num_kv_splits`
    resolved, built and returned finite garbage.
    """
    b, h, s, d = 3, 4, 2048, 64
    q, k, v = (_rand(b, s, h, d).transpose(1, 2) for _ in range(3))
    o = torch.empty(b, s, h, d, device="cuda", dtype=DT).transpose(1, 2)
    assert o.stride(1) == d, "the point of this layout is heads adjacent"
    fn = build(num_heads=h, head_dim=d, causal=True, dtype_str="bf16", num_kv_heads=h, num_kv_splits=splits)
    fn(q, k, v, o, b, s, seqlen_k=s, scale=None, lse=None, workspace=_splitk_workspace(b, h, s, splits, d))
    assert _err(o, _ref(q, k, v, causal=True)) < TOL


def test_splitk_refuses_a_gapped_batch_stride():
    """Heads adjacent is necessary but not sufficient; batches must be packed.

    The combine kernel is handed only `(batch_size, seq_len, stride_o_seq)`, so
    it can do nothing but assume a batch origin of `seq_len * stride_o_seq`.
    This O is BSHD -- the layout that works -- with an over-allocated sequence
    axis sliced back, so its head stride is right and its batch stride is not.
    It passed the first version of this guard and returned error 4.0.
    """
    b, h, s, d = 3, 4, 2048, 64
    q, k, v = (_rand(b, s + 37, h, d)[:, :s, :, :].transpose(1, 2) for _ in range(3))
    o = _rand(b, s + 37, h, d)[:, :s, :, :].transpose(1, 2)
    assert o.stride(1) == d, "heads are adjacent, so only the batch stride is wrong"
    assert o.stride(0) != s * o.stride(2)
    fn = build(num_heads=h, head_dim=d, causal=True, dtype_str="bf16", num_kv_heads=h, num_kv_splits=2)
    with pytest.raises(ValueError, match="batch stride"):
        fn(q, k, v, o, b, s, seqlen_k=s, scale=None, lse=None, workspace=_splitk_workspace(b, h, s, 2, d))


def test_splitk_refuses_a_bhsd_output():
    """A BHSD-contiguous O must raise, because the combine would alias heads.

    The shared combine kernel stores at `seq * stride_q_n + head * HEAD_DIM`,
    which is the production BSHD form. With BHSD both terms scale by `HEAD_DIM`
    and heads land on top of tokens: measured error 3.5 against a 2e-2
    tolerance, finite, deterministic, no fault. Refusing is the only honest
    behaviour until the combine path is made stride-general.
    """
    b, h, s, d = 1, 4, 2048, 64
    q, k, v = (_rand(b, h, s, d) for _ in range(3))
    o = torch.empty(b, h, s, d, device="cuda", dtype=DT)
    fn = build(num_heads=h, head_dim=d, causal=True, dtype_str="bf16", num_kv_heads=h, num_kv_splits=2)
    with pytest.raises(ValueError, match="head stride"):
        fn(q, k, v, o, b, s, seqlen_k=s, scale=None, lse=None, workspace=_splitk_workspace(b, h, s, 2, d))


# ---------------------------------------------------------------------------
# P3: generalized sliding window (gSWA)
# ---------------------------------------------------------------------------


def _run_window(q, k, v, window, *, causal=True):
    """Dispatch a window build. `window` is the raw (left, right) wire pair."""
    b, hq, s, d = q.shape
    o = torch.empty(b, hq, s, v.shape[3], device="cuda", dtype=q.dtype)
    fn = build(
        num_heads=hq,
        head_dim=d,
        head_dim_v=v.shape[3],
        causal=causal,
        window=True,
        dtype_str="bf16",
        num_kv_heads=k.shape[1],
    )
    fn(q, k, v, o, b, s, seqlen_k=k.shape[2], scale=None, lse=None, window=window)
    return o


def _window_ref(q, k, v, left, right):
    """SDPA with the band `i - left <= c <= i + right` as an additive mask."""
    s, sk = q.shape[2], k.shape[2]
    i = torch.arange(s, device="cuda").view(-1, 1)
    c = torch.arange(sk, device="cuda").view(1, -1)
    live = (c >= i - left) & (c <= i + right)
    assert bool(live.any(dim=1).all()), "every row must keep a key, or the reference is NaN"
    bias = torch.zeros(s, sk, device="cuda", dtype=torch.float32)
    bias.masked_fill_(~live, float("-inf"))
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=bias)


@pytest.mark.parametrize("hdim", LADDER)
def test_window_sentinel_is_bitwise_causal(hdim):
    """A window build fed the causal sentinels must reproduce causal exactly.

    The sharpest oracle available for this phase, and the reason the sentinels
    are resolved on the device rather than the host: `WINDOW_BOTRIGHT` means
    `window_right = seqlen_k - seqlen_q`, which is precisely the `delta` the
    causal path already uses, and `window_left = seqlen_q`, which is wide
    enough that no row loses a key. So the left bound is exercised -- the
    compares are emitted and executed -- while masking nothing, and any
    disagreement is the window machinery rather than a tolerance.

    Parametrized over the whole ladder because a window reaches **both kernel
    bodies**. Widths up to 256 run the dual-wave pipeline; 384 and 512 run the
    staged/sharded one in `fmha_wide_gfx950.py`, which has its own KV loop and
    its own tile base. The window support is inherited by subclassing rather
    than duplicated, and this is what holds that claim at every rung -- the
    two extra checks below are here so a rung cannot pass on the sentinel path
    alone, which leaves the left bound inert by construction.
    """
    b, h, s = 1, 4, 512
    q, k, v = (_rand(b, h, s, hdim) for _ in range(3))
    want = _run(q, k, v, causal=True)
    got = _run_window(q, k, v, (fmha.WINDOW_BOTRIGHT, fmha.WINDOW_BOTRIGHT))
    assert torch.equal(got, want), (got.float() - want.float()).abs().max().item()
    # A real band at the same width, so a rung cannot pass on the sentinel path
    # alone -- the sentinels leave the left bound inert by construction.
    got = _run_window(q, k, v, (127, 0))
    assert _err(got, _window_ref(q, k, v, 127, 0)) < TOL
    # And the degenerate window that leaves every row with no key at all.
    got = _run_window(q, k, v, (-32, 0))
    assert not torch.isnan(got.float()).any()
    assert torch.count_nonzero(got) == 0, "a window admitting no key must give exactly zero"


@pytest.mark.parametrize("causal_shape", [(2, 8, 512), (1, 4, 1024)], ids=["s512", "s1024"])
@pytest.mark.parametrize("left,right", [(31, 0), (127, 0), (511, 0), (63, 63), (0, 0), (255, 32)])
def test_window_matches_masked_sdpa(causal_shape, left, right):
    """Real bands against SDPA with the same band as an additive mask.

    `(0, 0)` is the degenerate diagonal -- one key per row -- which is the case
    a left bound gets wrong most loudly, and `(63, 63)` is a symmetric band
    that is not causal at all, reachable only because `window_right` is a free
    bound rather than a fixed alignment.
    """
    b, h, s = causal_shape
    q, k, v = (_rand(b, h, s, 64) for _ in range(3))
    got = _run_window(q, k, v, (left, right))
    assert _err(got, _window_ref(q, k, v, left, right)) < TOL


def _window_ref_with_dead_rows(q, k, v, left, right):
    """`_window_ref`, but tolerating rows the window leaves with no key.

    Those rows are what a negative bound produces, and the reference cannot
    express them directly: an all-`-inf` row makes SDPA's softmax `0/0`. The
    convention -- FlashAttention's, and what this kernel produces -- is that a
    row with nothing to attend to contributes nothing, so its output is zero.
    """
    s, sk = q.shape[2], k.shape[2]
    i = torch.arange(s, device="cuda").view(-1, 1)
    c = torch.arange(sk, device="cuda").view(1, -1)
    live = (c >= i - left) & (c <= i + right)
    dead = ~live.any(dim=1)
    bias = torch.zeros(s, sk, device="cuda", dtype=torch.float32)
    bias.masked_fill_(~live, float("-inf"))
    bias[dead] = 0.0  # keep SDPA finite; the rows are zeroed below anyway
    out = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=bias)
    out[:, :, dead, :] = 0.0
    return out, int(dead.sum())


@pytest.mark.parametrize(
    "left,right",
    [(-32, 128), (-1, 64), (512, -32), (512, -1), (256, -128), (-32, 0), (-64, 32), (-32, -16)],
)
def test_window_negative_bounds(left, right):
    """Either bound may be negative, and then some rows keep no key at all.

    A negative `window_left` pushes the band *ahead* of the diagonal and a
    negative `window_right` pushes it behind; both are ordinary AOTriton
    inputs. The last three cases admit no key for any row, which is the one
    that would most easily produce `NaN`: the row max is never written, the
    denominator stays zero, and a division would be `0/0`.

    That it does not is `ParitySoftmaxHelper`'s floor-seeded `reduce_max`
    earning its keep -- no lane ever holds `-inf` as a max, so `exp2` gives a
    clean zero rather than `NaN`. The assertion on `isnan` is the point of the
    test; the tolerance check alone would not distinguish the two.
    """
    b, h, s = 1, 4, 512
    q, k, v = (_rand(b, h, s, 64) for _ in range(3))
    got = _run_window(q, k, v, (left, right))
    assert not torch.isnan(got.float()).any(), "a fully masked row produced NaN"
    want, n_dead = _window_ref_with_dead_rows(q, k, v, left, right)
    assert n_dead > 0, "this case is meant to exercise rows with no live key"
    assert _err(got, want) < TOL


@pytest.mark.parametrize("hdim", [64, 512], ids=["dualwave", "wide"])
def test_window_skips_leading_dead_tiles(hdim):
    """A narrow window must be *faster*, not merely correct.

    A dead tile masks to `-inf`, contributes zero and changes no output bit, so
    every correctness test here passes whether or not the walk skips it. Only a
    measurement can tell, which is why this one is a timing assertion.

    The margin is deliberately loose. At `left=127` over 8192 tokens the walk
    covers a handful of tiles instead of the full lower triangle and measures
    around 7x; anything under 3x means the tile range is not being cut.

    Both bodies, because they cut the range in different code. The wide body
    shipped with its KV loop starting at a literal tile 0, which masks every
    dead tile correctly and walks all of them -- so it was *correct* and got no
    speedup at all (0.92x measured). Only a timing test distinguishes that from
    a working cut.
    """
    b, h, s, d = 1, 4, 4096, hdim
    q, k, v = (_rand(b, h, s, d) for _ in range(3))
    o = torch.empty(b, h, s, d, device="cuda", dtype=DT)
    fn = build(num_heads=h, head_dim=d, causal=True, window=True, dtype_str="bf16", num_kv_heads=h)

    def timed(window):
        call = lambda: fn(q, k, v, o, b, s, seqlen_k=s, scale=None, lse=None, window=window)  # noqa: E731
        for _ in range(10):
            call()
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(20):
                call()
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t0)
        return best

    narrow = timed((127, 0))
    unbounded = timed((s, 0))
    assert unbounded / narrow > 3.0, f"narrow window only {unbounded / narrow:.1f}x faster; tiles not skipped"


def test_window_requires_a_window_build():
    """Passing a window to a build that ignores it must raise, not be dropped."""
    b, h, s = 1, 4, 256
    q, k, v = (_rand(b, h, s, 64) for _ in range(3))
    o = torch.empty(b, h, s, 64, device="cuda", dtype=DT)
    fn = build(num_heads=h, head_dim=64, causal=True, dtype_str="bf16", num_kv_heads=h)
    with pytest.raises(ValueError, match="not compiled for windows"):
        fn(q, k, v, o, b, s, seqlen_k=s, scale=None, lse=None, window=(31, 0))


def test_window_build_requires_causal():
    with pytest.raises(ValueError, match="window=True requires causal=True"):
        build(num_heads=8, head_dim=64, causal=False, window=True, dtype_str="bf16")


# ---------------------------------------------------------------------------
# R1: one factory, one resolve
# ---------------------------------------------------------------------------

_META = FmhaInputMetadata(num_heads=8, head_dim=128)


def test_factory_accepts_a_full_arch_string():
    """`gcnArchName` carries target features; the caller should not strip them."""
    assert fmha_knobs("gfx950:sramecc+:xnack-").resolve(_META).build_traits(_META).BLOCK_M == 256


def test_factory_rejects_unknown_arch_and_fields():
    with pytest.raises(ValueError, match="no FMHA knobs for arch"):
        fmha_knobs("gfx1201")
    with pytest.raises(TypeError, match="unknown Gfx950Knobs field"):
        fmha_knobs("gfx950", nonsense=1)
    with pytest.raises(TypeError, match="unknown Gfx950Knobs field"):
        # `traits` is no longer a field at all, so it lands in the generic
        # unknown-field arm rather than a special case of its own.
        fmha_knobs("gfx950", traits=object())


def test_resolve_produces_the_traits():
    """`resolve` still owns the derivation; the knobs just stop carrying it.

    R1 made one call produce both halves, so a caller could not hold knobs that
    disagreed with the traits actually built. That still holds -- the traits
    come from `build_traits` on the resolved knobs, and there is no other way
    to make them -- but the knob object stays plain scalars so the set can be
    recorded beside the compiled hsaco in a flat `k=v` wire format.
    """
    unresolved = fmha_knobs("gfx950")
    assert not hasattr(unresolved, "traits"), "a knob class is plain build options"
    assert unresolved.block_dmodel is None
    resolved = unresolved.resolve(_META)
    assert resolved.block_dmodel == 128
    assert resolved.build_traits(_META).HEAD_DIM == 128


def test_resolve_is_idempotent():
    once = fmha_knobs("gfx950").resolve(_META)
    twice = once.resolve(_META)
    assert once.build_traits(_META).cache_tag == twice.build_traits(_META).cache_tag
    assert (once.num_waves, once.block_m, once.block_n) == (twice.num_waves, twice.block_m, twice.block_n)


def test_cross_seqlen_is_an_ordinary_field():
    """R1's side effect: no keyword-only parameter, no `kwargs.pop`, no converter arg."""
    assert fmha_knobs("gfx950").resolve(_META).build_traits(_META).CROSS_SEQLEN is False
    assert fmha_knobs("gfx950", cross_seqlen=True).resolve(_META).build_traits(_META).CROSS_SEQLEN is True


@pytest.mark.parametrize(
    "block_dmodel,want",
    [
        (32, (4, 128, 64, 32)),  # family S -- granule 32 for widths off the 64 grid
        (96, (4, 128, 64, 32)),
        (64, (8, 256, 64, 64)),  # family A
        (128, (8, 256, 64, 64)),
        (192, (4, 128, 64, 64)),  # family B -- 4 waves for the register file
        (256, (4, 128, 64, 64)),
        # Family W. Also 4 waves, but BLOCK_M is `Q_TILES * 32` and the VO
        # shards consume waves, so 512 at 2 shards covers 64 rows, not 128.
        (384, (4, 128, 64, 64)),
        (512, (4, 64, 64, 64)),
    ],
)
def test_wave_geometry_selects_the_family(block_dmodel, want):
    """Three families, keyed on the tile width, granule included.

    `block_dmodel` off the built rungs is rejected by `_with_widths`, so the
    planned families are reached by setting the field directly -- which is also
    what pins the step's contract: it reads what the previous step wrote.
    """
    k = replace(fmha_knobs("gfx950"), block_dmodel=block_dmodel)._with_wave_geometry()
    assert (k.num_waves, k.block_m, k.block_n, k.head_dim_granule) == want


@pytest.mark.parametrize("block_dmodel", [32, 64, 96, 128, 192, 512])
def test_every_family_stages_coherently(block_dmodel):
    """Each family's (granule, BLOCK_N, waves) must divide a KV tile evenly.

    A wave moves a fixed 512 bf16 elements per DMA issue, so the granule fixes
    tokens-per-issue and BLOCK_N fixes how many lines a tile needs. This is the
    check a *new* family has to pass, which is why it is a rule rather than a
    table of expected numbers.
    """
    k = replace(fmha_knobs("gfx950"), block_dmodel=block_dmodel)._with_wave_geometry()
    per_issue, lines, issues = k.staging_shape()
    assert per_issue * lines == k.block_n
    assert issues >= 1 and lines == issues * k.num_waves


def test_granule_16_is_blocked_by_the_mfma_not_the_staging():
    """head_dim below 32 cannot have a native tile with `v_mfma_f32_32x32x16`.

    Worth pinning because the staging *is* regular at granule 16 -- BLOCK_N 256
    over 8 waves is one issue per wave -- so the temptation is to conclude it
    works. The PV MFMA emits 32 D columns, so `D_CHUNKS` would be below 1.
    """
    k = replace(fmha_knobs("gfx950"), block_dmodel=16)._with_wave_geometry()
    with pytest.raises(ValueError, match="narrower than the PV MFMA"):
        k.staging_shape()


def test_wave_geometry_requires_widths_first():
    """The step order is a data dependency, and says so instead of guessing."""
    with pytest.raises(ValueError, match="runs after _with_widths"):
        fmha_knobs("gfx950")._with_wave_geometry()


# Family S -- (4, 128, 64, 32) -- used to be here. It is addressable now, so
# only the geometries that remain unbuilt belong in this test. BLOCK_N 128 is
# not an oversight: `tooling/lds_model.py` shows a wave's K read covers 64
# tokens, so a 128-token tile would need a doubled score accumulator.
@pytest.mark.parametrize("geom", [(8, 256, 128, 64)], ids=["block_n_128"])
def test_unaddressable_geometry_fails_loudly(geom):
    """Describable is not the same as addressable, and the gap must not be silent.

    `make_traits` takes the geometry as parameters, so it will happily describe
    families S and B. The DMA and LDS-read helpers still assume
    `SMEM_N_RPT == NUM_WAVES` and a 64-element granule, so such a build would
    address the wrong LDS and produce plausible numbers -- which is exactly
    what head_dim 192 did before the ladder was reverted.
    """
    nw, bm, bn, gran = geom
    with pytest.raises(NotImplementedError, match="describable but not yet addressable"):
        fmha_knobs("gfx950", num_waves=nw, block_m=bm, block_n=bn, head_dim_granule=gran).resolve(_META)


def test_traits_constructor_matches_production_exactly():
    """The parity constructor is a transcription; this is what makes that mean something.

    Field-by-field against `_make_dualwave_swp_traits` at family A's geometry,
    across the modes, because a single configuration would not catch a term
    that happens to vanish there.
    """
    from fmha_traits_gfx950 import assert_matches_production

    for causal in (True, False):
        for lse in (True, False):
            for splits in (1, 2):
                for paged, layout in ((False, "linear"), (True, "linear"), (True, "vectorized")):
                    assert_matches_production(
                        # Family A's geometry is granule 64, so only the rungs
                        # that are multiples of 64 can be expressed in it --
                        # and production only builds 64/128 anyway.
                        head_dims=tuple(h for h in LADDER if h % 64 == 0),
                        num_heads=8,
                        num_kv_heads=8,
                        causal=causal,
                        return_lse=lse,
                        num_kv_splits=splits,
                        paged=paged,
                        kv_cache_layout=layout,
                        kv_vectorized=paged and layout == "vectorized",
                    )
    fam_a = tuple(h for h in LADDER if h % 64 == 0)
    assert_matches_production(head_dims=fam_a, num_heads=8, num_kv_heads=2, kv_vectorized=False)
    assert_matches_production(head_dims=fam_a, num_heads=8, num_kv_heads=8, dtype_str="f16", kv_vectorized=False)


def test_wave_geometry_must_be_pinned_whole():
    with pytest.raises(ValueError, match="together or not at all"):
        fmha_knobs("gfx950", num_waves=4).resolve(_META)


def test_builder_requires_resolved_knobs():
    from flash_attn_func_gfx950 import build_flash_attn_func_gfx950_module_primary as build_primary

    with pytest.raises(ValueError, match="must be resolved"):
        build_primary(_META, fmha_knobs("gfx950"))


def test_build_time_sm_scale_is_honoured():
    """`meta.sm_scale` bakes a scale into the build; a per-call `scale` still wins.

    R1 is what surfaced this: with the traits moved onto the knobs, `meta`
    became an unused parameter of the builder, and the reason was that
    `meta.sm_scale` had never been read at all -- a build could declare a scale
    and the kernel would silently use `1/sqrt(head_dim)`.
    """
    b, h, s, d = 1, 4, 256, 64
    q, k, v = (_rand(b, h, s, d) for _ in range(3))
    baked = 0.03

    o = torch.empty_like(q)
    fn = build(num_heads=h, head_dim=d, causal=False, dtype_str="bf16", sm_scale=baked)
    fn(q, k, v, o, b, s)
    assert _err(o, _ref(q, k, v, scale=baked)) < TOL

    # A per-call scale overrides the baked one.
    o2 = torch.empty_like(q)
    fn(q, k, v, o2, b, s, scale=0.2)
    assert _err(o2, _ref(q, k, v, scale=0.2)) < TOL


@pytest.mark.parametrize("sq,sk", [(256, 128), (512, 128), (256, 64), (384, 256)], ids=lambda p: str(p))
@pytest.mark.parametrize("head_dim", [64, 128])
def test_window_masks_the_kv_tail_when_q_overhangs_k(sq, sk, head_dim):
    """`seqlen_q > seqlen_k` under a top-left window, which was wrong before.

    `DualwaveSoftmaxHelper.causal_mask_pair_if_needed` replaces
    `seq_pad_mask_if_needed` on the argument that with `delta = seqlen_kv -
    seqlen_q` no row can reach a padding column. True of *that* delta -- but a
    window build re-points `delta_i32` at the resolved `window_right`, and
    top-left causal is `window_right == 0`, so the bound is `col <= row` and a
    row at or past `seqlen_kv` reaches columns the K buffer does not hold.
    They read back as **0, which is a logit and not `-inf`**, so each takes a
    real share of the softmax weight while contributing nothing to the
    numerator, and every such row comes out too small.

    Measured before `_KvTailCausalMaskMixin`, relative error on `O` against
    fp64 at `B1 H2 D64`: 1.1e-01 at Sq 256/Sk 128, 2.2e-01 at 512/128, 3.2e-01
    at 256/64 -- growing with the overhang, while plain bottom-right causal
    stayed at 2.3e-03 throughout. Nothing in the suite covered
    `seqlen_q > seqlen_k` under a window, which is why it shipped.

    The band here is top-left (`right = 0`), the one the kernel gets wrong; the
    left bound is unbounded so this is exactly "causal, aligned top-left".
    """
    torch.manual_seed(sq * 31 + sk + head_dim)
    b, h = 1, 2
    q = _rand(b, h, sq, head_dim)
    k, v = _rand(b, h, sk, head_dim), _rand(b, h, sk, head_dim)
    got = _run_window(q, k, v, (fmha.WINDOW_BOTRIGHT, 0))

    qq, kk, vv = (t.double() for t in (q, k, v))
    scores = (qq @ kk.transpose(-1, -2)) * (1.0 / head_dim**0.5)
    i = torch.arange(sq, device="cuda").view(-1, 1)
    c = torch.arange(sk, device="cuda").view(1, -1)
    scores = scores.masked_fill(c > i, float("-inf"))
    live = torch.isfinite(scores).any(-1, keepdim=True)
    ref = torch.where(live, torch.softmax(scores, -1), torch.zeros_like(scores)) @ vv

    err = ((got.double() - ref).norm() / ref.norm()).item()
    assert err < 1e-2, (
        f"Sq {sq} Sk {sk} D {head_dim}: {err:.3e}. Rows at or past seqlen_k are attending to "
        "columns the K buffer does not hold, which read as 0 and take softmax weight."
    )
