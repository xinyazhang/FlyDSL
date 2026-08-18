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

import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from flash_attn_func_gfx950 import build_flash_attn_func_gfx950_module as build
from fmha_tuning_gfx950 import LADDER, LADDER_PLANNED, FmhaInputMetadata, fmha_knobs, tile_width_for
from gfx950_standalone import dualwave  # noqa: F401  (puts the repo root on sys.path)

from kernels.attention.flash_attn_gfx950 import build_flash_attn_dualwave_swp_module as build_prod

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
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
        (1, 32), (16, 32), (32, 32), (33, 64), (64, 64), (65, 96), (96, 96), (97, 128), (128, 128),
        (129, 160), (160, 160), (161, 192), (192, 192), (193, 224), (224, 224),
        (225, 256), (256, 256), (257, 384), (384, 384), (385, 512), (512, 512),
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
# R1: one factory, one resolve
# ---------------------------------------------------------------------------

_META = FmhaInputMetadata(num_heads=8, head_dim=128)


def test_factory_accepts_a_full_arch_string():
    """`gcnArchName` carries target features; the caller should not strip them."""
    assert fmha_knobs("gfx950:sramecc+:xnack-").resolve(_META).traits.BLOCK_M == 256


def test_factory_rejects_unknown_arch_and_fields():
    with pytest.raises(ValueError, match="no FMHA knobs for arch"):
        fmha_knobs("gfx1201")
    with pytest.raises(TypeError, match="unknown Gfx950Knobs field"):
        fmha_knobs("gfx950", nonsense=1)
    with pytest.raises(TypeError, match="set by resolve"):
        fmha_knobs("gfx950", traits=object())


def test_resolve_produces_the_traits():
    """The whole point of R1: one object carries both halves.

    Before it, the traits came from a separate `make_parity_traits(meta,
    knobs, ...)` call, so a caller could hold knobs that disagreed with the
    traits actually built.
    """
    unresolved = fmha_knobs("gfx950")
    assert unresolved.traits is None
    resolved = unresolved.resolve(_META)
    assert resolved.traits is not None
    assert resolved.traits.HEAD_DIM == 128


def test_resolve_is_idempotent():
    once = fmha_knobs("gfx950").resolve(_META)
    twice = once.resolve(_META)
    assert once.traits.cache_tag == twice.traits.cache_tag
    assert (once.num_waves, once.block_m, once.block_n) == (twice.num_waves, twice.block_m, twice.block_n)


def test_cross_seqlen_is_an_ordinary_field():
    """R1's side effect: no keyword-only parameter, no `kwargs.pop`, no converter arg."""
    assert fmha_knobs("gfx950").resolve(_META).traits.CROSS_SEQLEN is False
    assert fmha_knobs("gfx950", cross_seqlen=True).resolve(_META).traits.CROSS_SEQLEN is True


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
