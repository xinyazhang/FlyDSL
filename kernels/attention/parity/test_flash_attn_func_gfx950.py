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

import pytest
import torch
import torch.nn.functional as F
from flash_attn_func_gfx950 import build_flash_attn_func_gfx950_module as build
from fmha_tuning_gfx950 import LADDER, FmhaInputMetadata, fmha_knobs, tile_width_for
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

_LADDER_CASES = [16, 17, 32, 33, 40, 48, 64, 80, 96, 100, 113, 128]


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


@pytest.mark.parametrize("hdim,hdim_v", [(128, 64), (96, 48), (64, 32), (128, 96)])
def test_asymmetric_hdim(hdim, hdim_v):
    """`hdim_qk != hdim_vo`. The reference asm ships this shape as hd192/hd128."""
    b, h, s = 2, 8, 512
    q, k, v, pitch = _padded_qkv(b, h, s, hdim, hdim_v)
    o = torch.empty(b, h, s, pitch, device="cuda", dtype=DT)[..., :hdim_v]
    _run(q, k, v, o=o)
    assert _err(o, _ref(q, k, v)) < TOL


@pytest.mark.parametrize("poison", [0.0, 1e4, float("inf"), float("nan")], ids=["zero", "big", "inf", "nan"])
@pytest.mark.parametrize("hdim", [17, 33, 80, 100, 113])
def test_padded_head_ignores_pad_contents(hdim, poison):
    """The answer must not depend on what sits in the D-axis padding.

    **This is the test that found the one real bug in P1.** Masking Q alone
    makes the padded columns contribute `0 * pad`, which is 0 for any finite
    pad -- so `big` and `zero` passed while `nan` and `inf` produced NaN, since
    `0 * NaN` is NaN. K carries its own mask now (`ParityKvLdsToVgprLoader`);
    without it these two ids fail and the other two do not, which is why the
    poison values are parametrized rather than folded into one case.
    """
    b, h, s = 2, 8, 256
    q, k, v, pitch = _padded_qkv(b, h, s, hdim, hdim, poison=poison)
    o = torch.empty(b, h, s, pitch, device="cuda", dtype=DT)[..., :hdim]
    _run(q, k, v, o=o)
    err = _err(o, _ref(q, k, v))
    assert math.isfinite(err) and err < TOL


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


@pytest.mark.parametrize("hdim,want", [(1, 64), (16, 64), (64, 64), (65, 128), (128, 128)])
def test_tile_width_rounds_up(hdim, want):
    assert tile_width_for(hdim) == want


def test_tile_width_reports_planned_rungs_distinctly():
    """A rung that exists in the plan but is not built is not "unsupported"."""
    with pytest.raises(NotImplementedError, match="4-wave"):
        tile_width_for(192)
    with pytest.raises(ValueError, match="exceeds"):
        tile_width_for(1024)


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


def test_wave_geometry_selects_the_family():
    """Family A up to 128; family B above it, which is where P2 stops."""
    assert fmha_knobs("gfx950")._wave_geometry(64) == (8, 256, 64)
    assert fmha_knobs("gfx950")._wave_geometry(128) == (8, 256, 64)
    assert fmha_knobs("gfx950")._wave_geometry(192) == (4, 128, 128)


def test_family_b_names_itself_rather_than_silently_building_family_a():
    """A pinned family-B geometry must fail loudly, not fall back.

    `_make_dualwave_swp_traits` hardcodes 8/256/64, so building it under
    family B's name would produce family A's kernel with family B's label --
    which would then be benchmarked as if it were the new geometry.
    """
    with pytest.raises(NotImplementedError, match="parity-side traits constructor"):
        fmha_knobs("gfx950", num_waves=4, block_m=128, block_n=128).resolve(_META)


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
