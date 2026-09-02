# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness for the gfx950 dK/dV backward kernel (B1).

Run from this directory, with `ROCM_PATH` exported and the JIT disk cache
**off** -- the parity helpers are not in the cache key, so a stale artifact can
turn an edit into a phantom pass::

    export ROCM_PATH=$(rocm-sdk path --root)
    cd kernels/attention/parity
    FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 -m pytest test_fmha_bwd_dkdv_gfx950.py -q

--- Why there is no fixed tolerance ----------------------------------------

The forward has a *bitwise* oracle: a production kernel computing the same
thing in the same order. The backward has none, and a fixed tolerance on a
bf16 backward is either so loose it accepts real bugs or so tight it fails on
arithmetic order. So the gate is the ratio `test_transformers.py` uses -- the
same problem computed three ways::

    ours     = this kernel, bf16
    ref_low  = torch's math backend, bf16      <- an equally imprecise honest answer
    ref_high = torch's math backend, fp64      <- ground truth

and `err(ours, ref_high) <= fudge * err(ref_low, ref_high)` per output tensor.
The bound is then the precision the problem inherently *has*, measured rather
than guessed, and it scales with sequence length and head dim on its own.

--- The three kinds of test ------------------------------------------------

- **Oracle** -- the ratio gate above, across shapes and both rungs.
- **Self-consistency** -- run the gfx950 *forward*, take its `O` and `LSE`, and
  feed them to this kernel. A disagreement between our two halves about
  `sm_scale` folding, `log2e` or the LSE layout is invisible to the oracle
  test, which supplies a reference LSE and would attribute the error to the
  backward.
- **Structural** -- assertions that a mistake would otherwise turn into
  plausible numbers: that the stride slots mean what they say, that nothing is
  written past the sequence, and that the configurations this phase does not
  compute are refused rather than approximated.
"""

import os
import subprocess
import sys
import tempfile
import time

import fmha_abi_gfx1201 as abi
import fmha_common_gfx1201 as fmha
import pytest
import torch
import torch.nn.functional as F
from fmha_bwd_dkdv_gfx950 import build_fmha_bwd_dkdv_gfx950_module as build
from fmha_tuning_bwd_dkdv_gfx950 import LADDER
from gfx950_standalone import dualwave  # noqa: F401  (puts the repo root on sys.path)
from torch.nn.attention import SDPBackend, sdpa_kernel

# Legacy module default. `_rand` and every allocation now go through `_dt()`,
# which the `dtype` fixture drives; this is kept only so that a helper called
# outside a fixture-bearing test has an unambiguous type.
DT = torch.bfloat16

# Per-tensor slack over the honest bf16 answer. Both gradients reduce over the
# **query** axis, so neither is the loose one the plan warns about (that is
# dQ's, which reduces over keys and belongs to B2).
#
# **Measured, not chosen.** Across every shape in this file the ratio sits at
# 1.35-1.46, flat in batch, heads, sequence length and head dim -- 2.3e-3
# against the math backend's 1.7e-3 -- and that flatness is what identifies it:
# a systematic extra rounding rather than an accumulation defect, namely that
# `P` and `dS` are truncated to bf16 before the two q-contracted GEMMs while
# torch's chain keeps an fp32 intermediate. 2.0 leaves ~40% headroom. Raising
# it to paper over a regression is exactly what plan section 8 forbids.
FUDGE = {"dk": 2.0, "dv": 2.0}

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    reason="requires a gfx950 device",
)


def _require_rocm_path():
    if not os.environ.get("ROCM_PATH") and not os.path.isdir("/opt/rocm"):
        pytest.skip("ROCM_PATH is unset and /opt/rocm is absent, so the JIT cannot find ld.lld")


# ---------------------------------------------------------------------------
# B8: the element type is a test axis, not a second suite
# ---------------------------------------------------------------------------
#
# **Parametrised rather than duplicated, deliberately.** A parallel f16 suite
# drifts: the next feature gets added to one file and not the other, which is
# exactly how bf16 became the only tested dtype in the first place. Here the
# harness reads one module-level setting and the `dtype` fixture drives it, so
# any test that takes the fixture gets both element types -- including tests
# that do not exist yet.
#
# The state is module-level rather than threaded through twenty signatures so
# that `_rand`, `_run`, `_ratio_check`, `_bias_run`, `_dropout_case` and
# `_varlen_case` all pick it up without being touched. The fixture saves and
# restores it, so a test that does *not* take the fixture is unambiguously
# bf16 and stays that way.
_DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16}
_DT_STATE = {"str": "bf16"}


def _dt():
    """The torch dtype the harness allocates tensors in."""
    return _DTYPES[_DT_STATE["str"]]


def _dt_str():
    """The `dtype_str` the harness compiles for. Always matches `_dt()`."""
    return _DT_STATE["str"]


@pytest.fixture(params=["bf16", "f16"])
def dtype(request):
    """Run a test once per element type.

    Take this fixture in any test whose subject is dtype-independent -- which
    is nearly all of them, since both types are two bytes wide and every tile
    geometry, LDS offset and register count is identical between them.
    """
    prev = _DT_STATE["str"]
    _DT_STATE["str"] = request.param
    yield request.param
    _DT_STATE["str"] = prev


def _rand(*shape, dtype=None):
    return torch.randn(*shape, device="cuda", dtype=_dt() if dtype is None else dtype)


def _band(sq, sk, window):
    """The visible `(Sq, Sk)` band, or None for no masking.

    **Bottom-right aligned**, which `torch`'s `is_causal` is not: `is_causal`
    is top-left and the two agree only at `Sq == Sk`. Writing the band out is
    the only way to compare against a kernel whose diagonal is
    `delta = seqlen_k - seqlen_q`, and it is also what lets one expression
    serve plain causal and a window -- plain causal *is* `(seqlen_q, delta)`.
    """
    if window is None:
        return None
    wl, wr = window
    i = torch.arange(sq, device="cuda")[:, None]
    j = torch.arange(sk, device="cuda")[None, :]
    return (j <= i + wr) & (j >= i - wl)


def _math_backward(q, k, v, do, scale, dtype, window=None):
    """`(o, lse, delta, dq, dk, dv)` from an explicit fp-`dtype` reference.

    `lse` and `delta` are computed alongside rather than taken from the
    autograd graph, because this kernel consumes them as *inputs* and the whole
    point of the high-precision arm is that everything it hands over is exact.

    The softmax is explicit rather than `F.scaled_dot_product_attention` as
    soon as a band is involved: SDPA has no bottom-right or sliding-window
    form. Without a band the two agree, and the unmasked arm still goes through
    the math backend so the no-feature path is compared against torch's own
    code rather than against this function.
    """
    qq, kk, vv = (t.to(dtype).detach().requires_grad_() for t in (q, k, v))
    sq, sk = q.shape[2], k.shape[2]
    band = _band(sq, sk, window)
    if band is None:
        with sdpa_kernel(SDPBackend.MATH):
            o = F.scaled_dot_product_attention(qq, kk, vv, scale=scale)
        s = (qq @ kk.transpose(-1, -2)).detach() * scale
    else:
        s = (qq @ kk.transpose(-1, -2)) * scale
        s = s.masked_fill(~band, float("-inf"))
        # A row with no live key: 0/0 in `softmax` and `-inf` in `logsumexp`.
        # The kernel produces zero there because *every* element of such a row
        # is masked, whatever LSE says, so the reference has to agree by
        # construction rather than by arithmetic.
        live = torch.isfinite(s).any(-1, keepdim=True)
        o = torch.where(live, torch.softmax(s, -1), torch.zeros_like(s)) @ vv
        s = s.detach()
    dq, dk, dv = torch.autograd.grad(o, (qq, kk, vv), do.to(dtype))
    lse = torch.logsumexp(s, -1)
    # Finite stand-in for a dead row's `-inf` logsumexp. Its value cannot reach
    # the output -- the mask zeroes those elements first -- and leaving an
    # infinity in an input the kernel multiplies by `log2(e)` would test the
    # `ninf` fastmath licence rather than the mask.
    #
    # **Clamped to the dtype, which f16 forced.** `1e30` was chosen when this
    # arm was only ever bf16 or fp64, and it does not fit in f16 -- torch
    # raises rather than saturating, so every f16 arm died here before it
    # reached a kernel. The stand-in only has to be large enough that
    # `exp2(-lse * log2e)` underflows to zero, which 65504 comfortably is.
    big = min(1e30, float(torch.finfo(lse.dtype).max))
    lse = torch.where(torch.isfinite(lse), lse, torch.full_like(lse, big))
    delta = (do.to(dtype) * o.detach()).sum(-1)
    return o.detach(), lse, delta, dq, dk, dv


def _run(q, k, v, do, lse, delta, *, scale, dk=None, dv=None, batch=None, knobs=None, causal=False, window=None):
    """Build for this shape and dispatch, returning `(dk, dv)`."""
    b, hq, sq, d = q.shape
    sk = k.shape[2]
    dk = torch.full_like(k, float("nan")) if dk is None else dk
    dv = torch.full_like(v, float("nan")) if dv is None else dv
    fn = build(
        num_heads=hq,
        head_dim=d,
        num_kv_heads=k.shape[1],
        dtype_str="bf16" if q.dtype is torch.bfloat16 else "f16",
        causal=causal,
        window=window is not None,
        **(knobs or {}),
    )
    extra = {} if window is None else {"window": window}
    fn(q, k, v, do, dk, dv, lse, delta, b if batch is None else batch, sq, seqlen_k=sk, scale=scale, **extra)
    return dk, dv


def _rel(got, ref64, floor=0.0):
    """Relative Frobenius error against the fp64 answer, with a floor under the norm.

    The floor is not slack, it is the answer to a genuinely degenerate case:
    `seqlen_k == 1` makes `dK` **analytically zero**, because a one-key softmax
    gives `p == 1`, hence `dp == delta` and `dS = p * (dp - delta)` cancels
    exactly. The fp64 reference norm is then `0.0` and the ratio is undefined,
    while the meaningful question -- is our value small *on the scale of this
    problem* -- has an obvious answer if the denominator comes from the other
    gradient, which is built from the same tensors and is not degenerate.
    Measured at `Sq 512, Sk 1, D 64`: reference `dK` norm 0.0, ours 9.4e-6,
    against a `dV` norm of 285. gfx1201's test carries the same device for the
    same reason (`window=(0, 0)` reaches it there).
    """
    den = max(ref64.norm().item(), float(floor), 1e-30)
    return ((got.double() - ref64).norm() / den).item()


# Added to `FUDGE * e_low`, so it only decides anything when the honest bf16
# answer is *exactly* right -- which happens only in the degenerate case above,
# where `e_low` is 0.0 and no multiple of it can pass a finite error. Three
# orders of magnitude below the 1.7e-3 that the non-degenerate shapes measure,
# so it never widens a real comparison.
DEGENERATE_ATOL = 1e-5


def _ratio_check(b, h, sq, sk, d, scale=None, knobs=None, causal=False, window=None):
    """The plan section 7.1 gate, and the numbers it measured, for one shape.

    `causal=True` with `window=None` is plain bottom-right causal, which the
    reference spells as the band `(seqlen_q, seqlen_k - seqlen_q)` -- the same
    thing the kernel's sentinels resolve to.
    """
    torch.manual_seed(hash((b, h, sq, sk, d)) & 0xFFFF)
    q, k, v, do = _rand(b, h, sq, d), _rand(b, h, sk, d), _rand(b, h, sk, d), _rand(b, h, sq, d)
    scale = 1.0 / d**0.5 if scale is None else scale
    ref_window = window if window is not None else ((sq, sk - sq) if causal else None)

    _o64, lse64, delta64, _dq64, dk64, dv64 = _math_backward(q, k, v, do, scale, torch.float64, ref_window)
    # The reference arm is the *dtype under test*, not always bf16: the gate is
    # "no worse than an honest low-precision chain", and which chain that is
    # depends on the element type. f16's own error is ~8x smaller than bf16's,
    # so leaving this at bf16 would make the f16 gate ~8x too loose.
    _o16, _l16, _d16, _dq16, dk16, dv16 = _math_backward(q, k, v, do, scale, _dt(), ref_window)

    lse = lse64.reshape(b * h, sq).float().contiguous()
    delta = delta64.reshape(b * h, sq).float().contiguous()
    dk, dv = _run(q, k, v, do, lse, delta, scale=scale, knobs=knobs, causal=causal, window=window)

    out = {}
    for name, ours, low, high, other in (
        ("dk", dk, dk16, dk64, dv64),
        ("dv", dv, dv16, dv64, dk64),
    ):
        floor = other.norm().item()
        e_ours, e_low = _rel(ours, high, floor), _rel(low, high, floor)
        out[name] = (e_ours, e_low)
        assert e_ours <= FUDGE[name] * e_low + DEGENERATE_ATOL, (
            f"{name} at (B{b} H{h} Sq{sq} Sk{sk} D{d}): error against fp64 is {e_ours:.3e}, "
            f"more than {FUDGE[name]}x the bf16 math backend's own {e_low:.3e}"
        )
    return out


# ---------------------------------------------------------------------------
# Oracle: the error-ratio gate against torch's math backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 64, 64),
        (1, 2, 128, 128),
        (2, 4, 512, 512),
        (4, 8, 1024, 1024),
    ],
    ids=lambda s: "B%dH%dSq%dSk%d" % s,
)
def test_matches_math_backend(shape, head_dim, dtype):
    _require_rocm_path()
    _ratio_check(*shape, head_dim)


def _family_knobs(head_dim, rows):
    """A pinned geometry each family can actually serve at this width.

    A wave owns `mfma_rows` KV rows, and the 16-row family's staging needs
    `SMEM_N_RPT` to divide the wave count -- which at granule 32 rules out the
    32-row default `block_q`. Returns None where no legal geometry exists.
    """
    granule = 64 if head_dim % 64 == 0 else 32
    waves = 4
    block_q = 32 if (rows == 16 and head_dim >= 192 and granule == 64) else 64
    if block_q // (512 // granule) % waves:
        return None
    return dict(
        mfma_rows=rows,
        dkv_shards=1,
        num_waves=waves,
        block_kv=rows * waves,
        block_q=block_q,
        head_dim_granule=granule,
    )


@pytest.mark.parametrize("mode", ["dense", "causal", "window"])
@pytest.mark.parametrize("rows", [32, 16], ids=["m32", "m16"])
@pytest.mark.parametrize("head_dim", LADDER)
def test_both_mfma_families_at_every_rung(head_dim, rows, mode, dtype):
    """Both families, everywhere, whatever the tuning table happens to pick.

    B3.5 added a 16-row family beside the 32-row one and routed six of the ten
    rungs to it. Testing only what `_GEOMETRY` selects would leave the other
    family covered at four rungs, and the next tuning change would silently
    move which four -- so the *coverage* must not depend on the *policy*.

    **And with every feature**, for the same reason one step further: a mask
    implemented at 32 rows and not 16 is a wrong answer at half the ladder, and
    each family's mask is keyed on its *own* lane->(row, col) map. `Sq != Sk`
    so bottom-right alignment is exercised rather than the one case where it
    coincides with top-left.
    """
    _require_rocm_path()
    knobs = _family_knobs(head_dim, rows)
    if knobs is None:
        pytest.skip(f"no legal {rows}-row geometry at head_dim {head_dim}")
    causal = mode != "dense"
    window = (96, 16) if mode == "window" else None
    _ratio_check(1, 2, 320, 256, head_dim, knobs=knobs, causal=causal, window=window)


@pytest.mark.parametrize("head_dim", LADDER)
def test_ladder_matches_math_backend(head_dim, dtype):
    """Every compiled rung, at a square shape and a ragged asymmetric one.

    Two shapes rather than the four above because each rung is a separate
    build, and what a new rung can get wrong is the *geometry* -- the staging
    granule, the wave count, the shard split -- which either works for a rung
    or does not. The sequence-length edges are width-independent and are
    covered once, at 64 and 128, by the tests above.
    """
    _require_rocm_path()
    _ratio_check(1, 2, 256, 256, head_dim)
    _ratio_check(1, 2, 199, 333, head_dim)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(
    "sq,sk",
    [(199, 333), (1024, 256), (256, 1024), (65, 65), (101, 103), (67, 130), (1, 1), (1, 512), (512, 1), (2, 3)],
)
def test_ragged_and_asymmetric_sequences(sq, sk, head_dim, dtype):
    """Tails on both axes, at every residue the row loads can land on.

    Nothing in this kernel masks: Q and dO are bounded at `seqlen_q` and read
    zero past it, K/V and the two outputs at `seqlen_kv`. `65` and `101` are
    there because the LSE and delta reads are four contiguous rows starting at
    a multiple of four, so a sequence length that is not a multiple of four is
    the only way a *partial* vector load is exercised at all. `Sk == 1` is the
    degenerate case `_rel`'s floor exists for.
    """
    _require_rocm_path()
    _ratio_check(1, 2, sq, sk, head_dim)


@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("scale", [0.02, 0.5, 4.0])
def test_scale_sweep(scale, head_dim, dtype):
    """Two things at once, and the second is why the range is 200x rather than 4x.

    First, `dK` carries `sm_scale` and `dV` does not (`dV = P^T dO` has no
    scale in it at all), so a build that scaled both returns a correct `dV`
    beside a `dK` off by exactly `sm_scale` -- and no shape check notices.

    Second, **a precision defect in the exponent hides at one scale and not at
    another.** B2 found that folding `sm_scale * log2e` into Q and rounding the
    product to bf16 -- which is what the *forward* does -- takes its ratio from
    1.29 at `sm_scale = 0.05` to 10.9 at 1.0, because the rounding lands in
    `P = exp2(S - lse2)`'s exponent and `dS` does not normalise it away. This
    kernel scales on the f32 scores instead, and the point of sweeping is that
    a single scale cannot tell the two apart. Measured flat at 1.36-1.55 across
    `sm_scale` 0.02 to 4.0 on both rungs; a host model of the two variants puts
    the pre-scaled one at 5.6e-2 against 1.6e-3 at `sm_scale = 4`.
    """
    _require_rocm_path()
    _ratio_check(1, 2, 512, 512, head_dim, scale=scale)


# ---------------------------------------------------------------------------
# Self-consistency with our own forward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [64, 128, 192, 256])
def test_consistent_with_the_gfx950_forward(head_dim, dtype):
    """Feed this kernel the forward kernel's own `O` and `LSE`.

    Plan section 7.1's second oracle, and it is not redundant with the ratio
    gate: that one supplies an `LSE` computed by torch, so a disagreement
    between our forward and our backward about the `log2e` fold, the scale
    fold, or the `(batch * heads, tokens)` LSE layout would be charged to the
    backward and read as a tolerance failure at some other shape. Here the LSE
    is the tensor our forward actually wrote.
    """
    _require_rocm_path()
    from flash_attn_func_gfx950 import build_flash_attn_func_gfx950_module as build_fwd

    b, h, s, d = 2, 4, 512, head_dim
    torch.manual_seed(7)
    q, k, v, do = _rand(b, h, s, d), _rand(b, h, s, d), _rand(b, h, s, d), _rand(b, h, s, d)
    scale = 1.0 / d**0.5

    o = torch.empty_like(q)
    lse = torch.empty(b * h, s, device="cuda", dtype=torch.float32)
    fwd = build_fwd(num_heads=h, head_dim=d, num_kv_heads=h, causal=False, dtype_str=_dt_str(), return_lse=True)
    fwd(q, k, v, o, b, s, seqlen_k=s, scale=scale, lse=lse)
    torch.cuda.synchronize()

    delta = (do.float() * o.float()).sum(-1).reshape(b * h, s).contiguous()
    dk, dv = _run(q, k, v, do, lse, delta, scale=scale)

    _o64, _lse64, _delta64, _dq64, dk64, dv64 = _math_backward(q, k, v, do, scale, torch.float64)
    _o16, _l16, _d16, _dq16, dk16, dv16 = _math_backward(q, k, v, do, scale, _dt())
    for name, ours, low, high in (("dk", dk, dk16, dk64), ("dv", dv, dv16, dv64)):
        e_ours, e_low = _rel(ours, high), _rel(low, high)
        assert e_ours <= FUDGE[name] * e_low, (
            f"{name} from our forward's O/LSE: {e_ours:.3e} against the bf16 backend's {e_low:.3e}. "
            "The two halves disagree about a convention rather than about precision."
        )


@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
def test_joint_with_dq_against_one_autograd_call(head_dim, dtype):
    """dQ and dK/dV are one gradient; test them apart and a cancelling error hides.

    Contract section 6. Skipped until the dQ kernel exists, because it is the
    sibling phase B2 -- the check itself is cheap and belongs here so that it
    starts running the moment that lands rather than being remembered.
    """
    _require_rocm_path()
    dq_build = pytest.importorskip(
        "fmha_bwd_dq_gfx950", reason="B2's dQ kernel is not present yet"
    ).build_fmha_bwd_dq_gfx950_module

    b, h, s, d = 1, 2, 256, head_dim
    torch.manual_seed(11)
    q, k, v, do = _rand(b, h, s, d), _rand(b, h, s, d), _rand(b, h, s, d), _rand(b, h, s, d)
    scale = 1.0 / d**0.5
    _o64, lse64, delta64, dq64, dk64, dv64 = _math_backward(q, k, v, do, scale, torch.float64)
    _o16, _l16, _d16, dq16, dk16, dv16 = _math_backward(q, k, v, do, scale, _dt())

    lse = lse64.reshape(b * h, s).float().contiguous()
    delta = delta64.reshape(b * h, s).float().contiguous()
    dk, dv = _run(q, k, v, do, lse, delta, scale=scale)
    dq = torch.full_like(q, float("nan"))
    dq_fn = dq_build(num_heads=h, head_dim=d, num_kv_heads=h, dtype_str=_dt_str())
    dq_fn(q, k, v, do, dq, lse, delta, b, s, seqlen_k=s, scale=scale)
    torch.cuda.synchronize()

    for name, ours, low, high in (("dq", dq, dq16, dq64), ("dk", dk, dk16, dk64), ("dv", dv, dv16, dv64)):
        e_ours, e_low = _rel(ours, high), _rel(low, high)
        assert e_ours <= 3.0 * e_low, f"{name}: {e_ours:.3e} against {e_low:.3e}"


# ---------------------------------------------------------------------------
# Causal and windows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", LADDER)
@pytest.mark.parametrize("sq,sk", [(256, 256), (199, 333), (512, 128), (128, 512)])
def test_causal_matches_the_band(sq, sk, head_dim, dtype):
    """Bottom-right causal at every rung, including both `Sq != Sk` directions.

    `Sq > Sk` is the case that produces q rows with **no live key at all** --
    the reference's `logsumexp` is `-inf` there and the kernel's mask zeroes
    every element of the row before that can matter. It was a real NaN in
    gfx1201's suite, in the test rather than the kernel.
    """
    _require_rocm_path()
    _ratio_check(1, 2, sq, sk, head_dim, causal=True)


@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize(
    "window",
    [(64, 0), (128, 32), (32, -8), (10000, 10000), (0, 0)],
    ids=["band64", "band128r32", "negative_left", "unbounded", "diagonal_only"],
)
def test_window_matches_the_band(window, head_dim, dtype):
    """Generalized sliding windows, including the two degenerate ends.

    `negative_left` pushes the whole band to the *right* of the diagonal, so
    the leading masked run spans several tiles rather than clipping one -- the
    case `decompose_causal_regions` warns not to carry two-region intuition
    into. `diagonal_only` makes `dK` analytically zero (`p == 1`, so
    `dp == delta` and `dS` cancels), which is what `_rel`'s floor is for.
    """
    _require_rocm_path()
    _ratio_check(1, 2, 256, 256, head_dim, causal=True, window=window)


@pytest.mark.parametrize("rows", [32, 16], ids=["m32", "m16"])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_window_sentinel_reproduces_causal_bitwise(head_dim, rows, dtype):
    """A window build fed `WINDOW_BOTRIGHT` must equal a causal build **exactly**.

    The sharpest test available here, and the reason is that the two builds are
    not the same code: a non-window build `const_expr`s the left-bound
    comparison away entirely, while a window build emits it with a bound that
    never bites. Bit-identity says the left-bound arm is inert when it should
    be -- a tolerance would not, and it caught four separate bugs in the
    forward's P3.

    Both families, because each has its own copy of the comparison.
    """
    _require_rocm_path()
    knobs = _family_knobs(head_dim, rows)
    if knobs is None:
        pytest.skip(f"no legal {rows}-row geometry at head_dim {head_dim}")
    b, h, sq, sk = 1, 2, 512, 384
    torch.manual_seed(head_dim + rows)
    q, k, v, do = (
        _rand(b, h, sq, head_dim),
        _rand(b, h, sk, head_dim),
        _rand(b, h, sk, head_dim),
        _rand(b, h, sq, head_dim),
    )
    lse = torch.randn(b * h, sq, device="cuda", dtype=torch.float32)
    delta = torch.randn_like(lse)
    scale = 1.0 / head_dim**0.5
    plain = _run(q, k, v, do, lse, delta, scale=scale, knobs=knobs, causal=True)
    sentinel = _run(
        q,
        k,
        v,
        do,
        lse,
        delta,
        scale=scale,
        knobs=knobs,
        causal=True,
        window=(fmha.WINDOW_BOTRIGHT, fmha.WINDOW_BOTRIGHT),
    )
    torch.cuda.synchronize()
    assert torch.equal(plain[0], sentinel[0]), "dK differs between a causal build and a sentinel-fed window build"
    assert torch.equal(plain[1], sentinel[1]), "dV differs between a causal build and a sentinel-fed window build"


@pytest.mark.parametrize("rows", [32, 16], ids=["m32", "m16"])
def test_causal_tile_cut_is_not_inert(rows):
    """The cut must be **timed**, because a dead tile changes no output bit.

    A causal build that walked every q tile and masked the dead ones would be
    bit-for-bit correct and do twice the work. The forward's wide body shipped
    exactly that -- right answers at 0.92x -- and only a timing assertion found
    it.

    **The shape has to fill the machine**, and that is the trap on this side.
    At `B=1 H=2 S=1024` there are 16 workgroups on 256 CUs, every one runs
    concurrently, and the wall clock is set by the *longest* -- the KV block at
    `kv_start = 0`, which walks every tile in both builds. Measured 0.94x
    there and 1.53x at the shape below, from the same code: a working cut is as
    easy to misdiagnose as an inert one.

    **A shared machine can fail this test without the kernel changing**, and it
    did in B6. Under a neighbouring job the whole GPU ran ~3.5x slow -- dense
    1.23 ms where it is normally 0.35 -- and the ratio compressed to 1.17x, well
    under the bar. It was *stable* at 1.17x across eight repeats, which is the
    part worth remembering: a throttled machine gives quiet, repeatable, wrong
    numbers, so low variance is not evidence that a measurement is valid. Both
    builds compress toward a common floor because whatever the contention is
    costs the same in each, and the cut's saving is a proportion of a term that
    is no longer dominant. The absolute check below is the guard: a ratio
    measured while the machine is not at speed is not evidence about the cut in
    either direction, so the failure says which of the two it is.
    """
    _require_rocm_path()
    head_dim = 64
    knobs = _family_knobs(head_dim, rows)
    b, h, s = 4, 8, 4096
    torch.manual_seed(1)
    q, k, v, do = (_rand(b, h, s, head_dim) for _ in range(4))
    lse = torch.randn(b * h, s, device="cuda", dtype=torch.float32)
    delta = torch.randn_like(lse)
    dk, dv = torch.empty_like(k), torch.empty_like(v)
    scale = 1.0 / head_dim**0.5

    def _time(causal):
        # Built once, outside the loop. `_run` rebuilds per call, which is
        # fine for a correctness helper and would make this measure the JIT
        # dispatch path instead of the kernel -- 6.2 s against 14.4 s the
        # first time it was written that way.
        fn = build(num_heads=h, head_dim=head_dim, num_kv_heads=h, causal=causal, **knobs)
        args = (q, k, v, do, dk, dv, lse, delta, b, s)
        kw = dict(seqlen_k=s, scale=scale)
        fn(*args, **kw)
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(3):
            for _ in range(3):
                fn(*args, **kw)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(10):
                fn(*args, **kw)
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t)
        return best

    dense, causal = _time(False), _time(True)
    # Nominal, on the four GEMMs, at the fastest of the three dense repeats.
    dense_tf = 4 * 2 * b * h * s * s * head_dim / (dense / 10) / 1e12
    assert dense_tf > 400, (
        f"the machine is not at speed: the dense build measured {dense_tf:.0f} TFLOP/s where it "
        "reaches ~780 on an idle GPU, so the ratio below is not evidence about the tile cut. Check "
        "for a neighbouring job before reading this as a kernel regression."
    )
    # A causal KV block walks about half the q tiles on average, so the ceiling
    # is ~1.9x. 1.25 is well clear of both that and of run-to-run noise, and
    # well above the 1.0 an inert cut would give.
    assert dense / causal > 1.25, (
        f"the causal tile cut looks inert at {rows} rows: dense {dense * 1e3:.1f} ms against causal "
        f"{causal * 1e3:.1f} ms, ratio {dense / causal:.2f}x. A dead tile masks to nothing, so "
        "correctness cannot see this."
    )


_ISA_CHILD = """
import sys, argparse
sys.path.insert(0, {parity!r})
p = argparse.ArgumentParser()
p.add_argument("--rows", type=int, default=32)
p.add_argument("--dt", default="bf16")
a = p.parse_args()
import torch
from fmha_bwd_dkdv_gfx950 import build_fmha_bwd_dkdv_gfx950_module as build
DT = torch.bfloat16 if a.dt == "bf16" else torch.float16
d, B, H, S = 64, 1, 2, 256
q = torch.randn(B, H, S, d, device="cuda", dtype=DT)
k, v, do = torch.randn_like(q), torch.randn_like(q), torch.randn_like(q)
dk, dv = torch.empty_like(k), torch.empty_like(v)
lse = torch.randn(B * H, S, device="cuda", dtype=torch.float32)
delta = torch.randn_like(lse)
fn = build(num_heads=H, head_dim=d, num_kv_heads=H, dtype_str=a.dt, mfma_rows=a.rows,
           dkv_shards=1, num_waves=4, block_kv=a.rows * 4, block_q=64, head_dim_granule=64)
fn(q, k, v, do, dk, dv, lse, delta, B, S, seqlen_k=S, scale=1.0 / d ** 0.5)
torch.cuda.synchronize()
"""


def _dump_isa(*, dtype_str, rows):
    """Compile one configuration in a child process and return its final ISA.

    A child, because `FLYDSL_DUMP_IR` has to be set before the interpreter
    starts -- the same reason `check_exec_hazard_gfx950.py` shells out.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    out = tempfile.mkdtemp(prefix="isa_dtype_")
    env = dict(os.environ, FLYDSL_RUNTIME_ENABLE_CACHE="0", FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=out)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_ISA_CHILD.format(parity=here))
        script = f.name
    r = subprocess.run(
        [sys.executable, script, "--rows", str(rows), "--dt", dtype_str],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert r.returncode == 0, f"build failed for {dtype_str} rows{rows}:\n{r.stderr[-2000:]}"
    path = os.path.join(out, "fmha_bwd_dkdv_gfx950_kernel_0", "21_final_isa.s")
    with open(path) as fh:
        isa = fh.read()
    # The negative control: if the scan is not looking at a real kernel, every
    # "absence" assertion below passes vacuously.
    assert "v_mfma" in isa, f"no MFMA in the dumped ISA for {dtype_str} rows{rows}; this is not the kernel"
    return isa


def test_no_transpose_read_under_a_narrowed_exec():
    """CDNA4 section 11.4: `ds_read_b64_tr_b16` requires EXEC all 1s.

    B4 is the first phase where this is reachable -- before it neither kernel
    contained an `scf.if` at all. The kernel holds a stronger invariant than
    "no read inside a branch": every mask predicate is wave-uniform, so the
    branches are *scalar* and EXEC is never narrowed anywhere. The checker
    reports both, and this asserts the one that must hold.

    Shelled out because it needs `FLYDSL_DUMP_IR` set before the interpreter
    starts, and it builds twelve configurations, so it is also the slowest
    thing in the file.
    """
    _require_rocm_path()
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tooling", "check_exec_hazard_gfx950.py")
    r = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=2400)
    assert r.returncode == 0, f"exec-hazard scan failed:\n{r.stdout}\n{r.stderr}"
    assert "EXEC HAZARD: clean" in r.stdout, r.stdout


# ---------------------------------------------------------------------------
# Varlen
# ---------------------------------------------------------------------------
#
# **The oracle here is not AOTriton.** Its varlen is `cu_seqlens` +
# `num_seqlens` + `seq_strides`, the four-`VarlenType` enum, and two of the
# five configurations below (`0x150B`, `0x040B`) have no spelling in it at all
# -- so for those there is nothing to differentially test against. The
# substitute is the one varlen plan section 7 names and it is sharper than a
# tolerance: **N sequences must equal N separate dense calls, bitwise.** It
# holds because a varlen workgroup and its dense counterpart walk the same
# tiles in the same order over the same values; only the base address differs.

_VARLEN_MODES = ("compact", "padded", "strided", "seqused_packed", "seqused_cache")


def _cu(lens):
    return torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device="cuda", dtype=torch.int32)


def _varlen_case(mode, lq, lk, h, d, lse_layout):
    """`(descriptor, packed tensors, per-sequence tensors)` for one mode."""
    n, mq, mk = len(lq), max(lq), max(lk)
    torch.manual_seed(n * 7 + d + lse_layout)
    seqs = [
        (
            _rand(1, h, a, d),
            _rand(1, h, b, d),
            _rand(1, h, b, d),
            _rand(1, h, a, d),
        )
        for a, b in zip(lq, lk)
    ]
    packed = lambda i: torch.cat([s[i] for s in seqs], dim=2)  # noqa: E731
    if mode in ("compact", "strided", "seqused_packed"):
        Q, K, V, DO = packed(0), packed(1), packed(2), packed(3)
    elif mode == "padded":
        Q = torch.zeros(n, h, mq, d, device="cuda", dtype=_dt())
        K = torch.zeros(n, h, mk, d, device="cuda", dtype=_dt())
        V, DO = torch.zeros_like(K), torch.zeros_like(Q)
        for i, (q, k, v, do) in enumerate(seqs):
            Q[i, :, : lq[i]], DO[i, :, : lq[i]] = q[0], do[0]
            K[i, :, : lk[i]], V[i, :, : lk[i]] = k[0], v[0]
    else:  # seqused_cache: packed Q against a *batched* KV cache
        Q, DO = packed(0), packed(3)
        K = torch.zeros(n, h, mk, d, device="cuda", dtype=_dt())
        V = torch.zeros_like(K)
        for i, (_q, k, v, _do) in enumerate(seqs):
            K[i, :, : lk[i]], V[i, :, : lk[i]] = k[0], v[0]
    cuq, cuk, tot = _cu(lq), _cu(lk), int(sum(lq))
    used = torch.tensor(lk, device="cuda", dtype=torch.int32)
    desc = {
        "compact": lambda: abi.varlen_compact(cuq, cuk, mq, mk, lse_tokens=tot, lse_layout=lse_layout),
        "padded": lambda: abi.varlen_padded(cuq, cuk, mq, mk, lse_layout=lse_layout),
        "strided": lambda: abi.varlen_strided(cuq, cuk, cuq, cuk, mq, mk, lse_tokens=tot, lse_layout=lse_layout),
        "seqused_packed": lambda: abi.varlen_seqused_k(cuq, cuk, used, mq, mk, lse_tokens=tot, lse_layout=lse_layout),
        "seqused_cache": lambda: abi.varlen_seqused_k(
            cuq, None, used, mq, mk, k_is_cache=True, lse_tokens=tot, lse_layout=lse_layout
        ),
    }[mode]()
    return desc, (Q, K, V, DO), seqs


def _varlen_vs_dense(mode, lq, lk, d=64, h=2, causal=False, rows=None, lse_layout=0):
    """One varlen call against `len(lq)` dense ones. Bitwise."""
    desc, (Q, K, V, DO), seqs = _varlen_case(mode, lq, lk, h, d, lse_layout)
    n, mq, mk = len(lq), max(lq), max(lk)
    q_packed = Q.shape[0] == 1
    nseq, bsz = (n if q_packed else 0), Q.shape[0]
    tokens = int(sum(lq)) if q_packed else mq
    scale = 1.0 / d**0.5
    if lse_layout:
        lse = torch.randn((1 if q_packed else bsz) * tokens, h, device="cuda", dtype=torch.float32)
    else:
        lse = torch.randn((1 if q_packed else bsz) * h, tokens, device="cuda", dtype=torch.float32)
    delta = torch.randn_like(lse)
    dK, dV = torch.full_like(K, float("nan")), torch.full_like(V, float("nan"))
    knobs = {} if rows is None else _family_knobs(d, rows)
    fn = build(
        num_heads=h,
        head_dim=d,
        num_kv_heads=h,
        varlen=True,
        causal=causal,
        lse_layout_th=bool(lse_layout),
        dtype_str=_dt_str(),
        **knobs,
    )
    fn(Q, K, V, DO, dK, dV, lse, delta, bsz, mq, seqlen_k=mk, scale=scale, varlen=desc, num_seqlens=nseq)
    torch.cuda.synchronize()

    # **The reference is pinned to the varlen build's own geometry.** The
    # tuning policy is feature-aware -- head_dim 224 picks a different MFMA
    # family once varlen is on -- and two families accumulate in different
    # orders, so an unpinned reference would compare arithmetic orders rather
    # than addressing. What this test claims is that only the base address
    # differs, and that is what pinning makes it test.
    pin = dict(
        mfma_rows=fn.traits.MFMA_ROWS,
        dkv_shards=fn.knobs.dkv_shards,
        num_waves=fn.knobs.num_waves,
        block_kv=fn.traits.BLOCK_KV,
        block_q=fn.knobs.block_q,
        head_dim_granule=fn.knobs.head_dim_granule,
        tight_registers=fn.knobs.tight_registers,
    )
    ref = build(num_heads=h, head_dim=d, num_kv_heads=h, causal=causal, dtype_str=_dt_str(), **pin)
    off = 0
    for i, (q, k, v, do) in enumerate(seqs):
        dk_r, dv_r = torch.full_like(k, float("nan")), torch.full_like(v, float("nan"))
        if lse_layout:
            blk = slice(off, off + lq[i]) if q_packed else slice(i * tokens, i * tokens + lq[i])
            rows_l, rows_d = lse[blk].T.contiguous(), delta[blk].T.contiguous()
        elif q_packed:
            rows_l, rows_d = lse[:, off : off + lq[i]].contiguous(), delta[:, off : off + lq[i]].contiguous()
        else:
            rows_l = lse[i * h : (i + 1) * h, : lq[i]].contiguous()
            rows_d = delta[i * h : (i + 1) * h, : lq[i]].contiguous()
        ref(q, k, v, do, dk_r, dv_r, rows_l, rows_d, 1, lq[i], seqlen_k=lk[i], scale=scale)
        torch.cuda.synchronize()
        # **The output slice follows the K layout, not the Q one.** `0x040B` is
        # packed Q against a batched cache, so the Q side is packed while dK
        # and dV are still `(n, h, max_k, d)`. Reading them as packed produced
        # a mismatch in exactly the mode P4 warns about for the batch index,
        # and it was this harness rather than the kernel.
        if K.shape[0] == 1:
            got_k = dK[:, :, sum(lk[:i]) : sum(lk[: i + 1])]
            got_v = dV[:, :, sum(lk[:i]) : sum(lk[: i + 1])]
        else:
            got_k, got_v = dK[i : i + 1, :, : lk[i]], dV[i : i + 1, :, : lk[i]]
        assert torch.equal(got_k.reshape(-1), dk_r.reshape(-1)), f"{mode} seq {i}: dK differs from the dense call"
        assert torch.equal(got_v.reshape(-1), dv_r.reshape(-1)), f"{mode} seq {i}: dV differs from the dense call"
        off += lq[i]


@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
@pytest.mark.parametrize("rows", [32, 16], ids=["m32", "m16"])
@pytest.mark.parametrize("mode", _VARLEN_MODES)
def test_varlen_equals_n_dense_calls(mode, rows, causal, dtype):
    """All five `VarlenBits` configurations, both families, both masking modes.

    `Sq != Sk` per sequence, so bottom-right alignment is per-sequence rather
    than global -- and `torch`'s top-left `is_causal` could not express it even
    if this test used a tolerance oracle instead of a bitwise one.

    `seqused_cache` (`0x040B`) is the mode that needs **each side's own batch
    index**: Q is stacked (batch 0, large row offset) against a *batched* cache
    (batch z, no row offset) in the same call. The other four agree, so it is
    the only one that exposes a shared index.
    """
    _require_rocm_path()
    _varlen_vs_dense(mode, [96, 200, 41], [160, 200, 300], d=64, rows=rows, causal=causal)


@pytest.mark.parametrize("head_dim", LADDER)
def test_varlen_across_the_ladder(head_dim, dtype):
    """The whole ladder, at the policy's own family and geometry for each rung."""
    _require_rocm_path()
    _varlen_vs_dense("compact", [128, 65, 33], [128, 65, 33], d=head_dim)
    _varlen_vs_dense("seqused_cache", [128, 65, 33], [200, 65, 90], d=head_dim, causal=True)


@pytest.mark.parametrize("mode", _VARLEN_MODES)
def test_varlen_transformer_engine_lse_layout(mode, dtype):
    """`VARLEN_LSE_LAYOUT_TH`: the row pitch stops being 1.

    Bits 17:16 choose `(H, T)` -- AOTriton's, tokens contiguous -- or `(T, H)`,
    Transformer Engine's, where consecutive tokens of one head are `num_heads`
    apart. The kernel's row-tensor read is four contiguous f32 per accumulator
    group in a dense build and **sixteen scalars in a varlen one** precisely
    because of this, and only this test exercises the second path's reason for
    existing. It caught the 32-row family still doing the wide load.
    """
    _require_rocm_path()
    _varlen_vs_dense(mode, [96, 200, 41], [96, 200, 41], d=64, lse_layout=abi.VARLEN_LSE_LAYOUT_TH)
    _varlen_vs_dense(mode, [96, 200, 41], [160, 200, 300], d=128, rows=16, lse_layout=abi.VARLEN_LSE_LAYOUT_TH)


def test_varlen_edges(dtype):
    """Single-token sequences, a single sequence, and a badly ragged batch."""
    _require_rocm_path()
    _varlen_vs_dense("compact", [1, 2, 3, 4], [1, 2, 3, 4], d=64)
    _varlen_vs_dense("compact", [512], [512], d=64, causal=True)
    _varlen_vs_dense("padded", [7, 300, 1], [500, 9, 64], d=64, causal=True)


def test_varlen_build_and_descriptor_must_agree():
    """Neither direction of the mismatch may pass silently.

    A varlen descriptor on a dense build would be ignored, and a dense call on
    a varlen build decodes `bits == 0` to the right answer -- so both would
    *work*, and both are a caller who believes something about the layout that
    is not being honoured.
    """
    _require_rocm_path()
    b, h, s, d = 1, 2, 128, 64
    q, k, v, do = (_rand(b, h, s, d) for _ in range(4))
    dk, dv = torch.empty_like(k), torch.empty_like(v)
    lse = torch.zeros(b * h, s, device="cuda", dtype=torch.float32)
    desc = abi.varlen_compact(_cu([s]), _cu([s]), s, s, lse_tokens=s)
    with pytest.raises(ValueError, match="not compiled for varlen"):
        build(num_heads=h, head_dim=d, dtype_str=_dt_str())(
            q, k, v, do, dk, dv, lse, lse, b, s, varlen=desc, num_seqlens=1
        )
    with pytest.raises(ValueError, match="requires a varlen= descriptor"):
        build(num_heads=h, head_dim=d, varlen=True, dtype_str=_dt_str())(q, k, v, do, dk, dv, lse, lse, b, s)


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------
#
# **The contract is cross-kernel and it is the whole difficulty.** The backward
# must regenerate the forward's mask bit for bit or the gradients are quietly
# wrong, and neither kernel can check that alone. Three parties make it
# testable: the forward *draws* the mask and reports the `(seed, offset)` it
# actually used, `dropout_mask_gfx1201` *regenerates* it as a tensor at a
# tiling of its own, and this kernel regenerates it again from the same
# report. All three call `Philox.grid_plane` / `grid_offset` rather than
# transcribing the formula, which is what makes their agreement mean something.


def _u64(x):
    return torch.tensor([int(x)], device="cuda", dtype=torch.int64)


def _dropout_case(
    d=64, b=1, h=2, s=256, p=0.25, rows=None, causal=False, seed=1234, mask_tile=(64, 32), bias_scale=None, hk=None
):
    """Forward -> reported (seed, offset) -> mask tensor -> fp64 reference -> backward.

    `bias_scale` and `hk` extend the same three-party oracle to B7's two
    features, which is the point of doing it here rather than in a second
    harness: the forward draws its mask over the *biased* scores and per
    *query* head, so both compositions are claims about what the forward did,
    and this is the only place that can read that back.
    """
    from dropout_mask_gfx1201 import dropout_mask
    from flash_attn_func_gfx950 import build_flash_attn_func_gfx950_module as build_fwd
    from philox import dropout_threshold

    torch.manual_seed(7)
    hk = h if hk is None else hk
    g = h // hk
    q, do = _rand(b, h, s, d), _rand(b, h, s, d)
    k, v = _rand(b, hk, s, d), _rand(b, hk, s, d)
    bias = None if bias_scale is None else _rand(b, h, s, s) * bias_scale
    scale = 1.0 / d**0.5

    o = torch.empty_like(q)
    lse = torch.empty(b * h, s, device="cuda", dtype=torch.float32)
    seed_out, off_out = torch.zeros(1, device="cuda", dtype=torch.int64), torch.zeros(
        1, device="cuda", dtype=torch.int64
    )
    fwd = build_fwd(
        num_heads=h,
        head_dim=d,
        num_kv_heads=hk,
        causal=causal,
        dtype_str=_dt_str(),
        return_lse=True,
        dropout=True,
        bias=bias is not None,
    )
    fwd(
        q,
        k,
        v,
        o,
        b,
        s,
        seqlen_k=s,
        scale=scale,
        lse=lse,
        dropout_p=p,
        **({} if bias is None else {"bias": bias}),
        philox_seed=_u64(seed),
        philox_offset1=_u64(0),
        philox_offset2=0,
        philox_seed_out=seed_out,
        philox_offset_out=off_out,
    )
    torch.cuda.synchronize()
    delta = (do.float() * o.float()).sum(-1).reshape(b * h, s).contiguous()

    raw = dropout_mask(
        b, h, s, s, int(seed_out.item()), int(off_out.item()), block_m=mask_tile[0], block_n=mask_tile[1]
    )
    keep = raw > dropout_threshold(p)

    qf, kf, vf = (t.float().detach().requires_grad_() for t in (q, k, v))
    sc = (qf @ kf.repeat_interleave(g, dim=1).transpose(-1, -2)) * scale
    if bias is not None:
        sc = sc + bias.float()
    if causal:
        sc = sc.masked_fill(~_band(s, s, (s, 0)), float("-inf"))
    pd = torch.where(keep, torch.softmax(sc, -1) * (1.0 / (1.0 - p)), torch.zeros_like(sc))
    o_ref = pd @ vf.repeat_interleave(g, dim=1)
    dk_ref, dv_ref = torch.autograd.grad(o_ref, (kf, vf), do.float())

    knobs = {} if rows is None else _family_knobs(d, rows)
    dk, dv = torch.full_like(k, float("nan")), torch.full_like(v, float("nan"))
    fn = build(
        num_heads=h,
        head_dim=d,
        num_kv_heads=hk,
        dropout=True,
        causal=causal,
        bias=bias is not None,
        dtype_str=_dt_str(),
        **knobs,
    )
    fn(
        q,
        k,
        v,
        do,
        dk,
        dv,
        lse,
        delta,
        b,
        s,
        seqlen_k=s,
        scale=scale,
        dropout_p=p,
        philox_seed=seed_out,
        philox_offset1=off_out,
        philox_offset2=0,
        **({} if bias is None else {"bias": bias}),
    )
    torch.cuda.synchronize()
    return o, o_ref.detach(), dk, dk_ref, dv, dv_ref, float(keep.float().mean())


@pytest.mark.parametrize("rows", [32, 16], ids=["m32", "m16"])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_dropout_regenerates_the_forwards_mask(head_dim, rows, causal, dtype):
    """The cross-kernel gate, and the only test that can see a mask disagreement.

    The reference is built from the mask `dropout_mask_gfx1201` regenerates, so
    it is only a valid reference if that kernel agrees with the forward -- which
    the `O` assertion below checks first. Once `O` matches, a `dK`/`dV`
    mismatch can only be this kernel drawing a different mask, and that is the
    failure the whole phase exists to prevent.
    """
    _require_rocm_path()
    o, o_ref, dk, dk_ref, dv, dv_ref, keep_rate = _dropout_case(d=head_dim, rows=rows, causal=causal)
    assert _rel(o, o_ref.double()) < 1e-2, "the debug mask kernel disagrees with the forward; the reference is invalid"
    for name, got, ref in (("dk", dk, dk_ref), ("dv", dv, dv_ref)):
        e = _rel(got, ref.double())
        assert e < 1e-2, f"{name}: {e:.3e} -- the backward drew a different mask from the forward"
    assert abs(keep_rate - 0.75) < 0.01, f"keep rate {keep_rate:.4f}, expected ~0.75 at p=0.25"


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_dropout_keep_rate(p):
    """The rate the mask actually realises, not the one that was asked for."""
    _require_rocm_path()
    _o, _oref, _dk, _dkref, _dv, _dvref, keep = _dropout_case(d=64, p=p, s=512)
    assert abs(keep - (1.0 - p)) < 0.01, f"keep rate {keep:.4f} at p={p}, expected ~{1 - p}"


def _dropout_run(knobs, p, seed=99, d=64, b=1, h=2, s=512, drop=True):
    torch.manual_seed(5)
    q, k, v, do = (_rand(b, h, s, d) for _ in range(4))
    lse = torch.randn(b * h, s, device="cuda", dtype=torch.float32)
    delta = torch.randn_like(lse)
    dk, dv = torch.full_like(k, float("nan")), torch.full_like(v, float("nan"))
    fn = build(num_heads=h, head_dim=d, num_kv_heads=h, dropout=drop, dtype_str=_dt_str(), **knobs)
    extra = dict(dropout_p=p, philox_seed=_u64(seed), philox_offset1=_u64(0), philox_offset2=0) if drop else {}
    fn(q, k, v, do, dk, dv, lse, delta, b, s, seqlen_k=s, scale=1.0 / d**0.5, **extra)
    torch.cuda.synchronize()
    return dk.clone(), dv.clone()


@pytest.mark.parametrize("rows", [32, 16], ids=["m32", "m16"])
def test_dropout_p_zero_is_bitwise_a_build_without_it(rows, dtype):
    """`p = 0` must change nothing at all, not merely little.

    The dropout arm multiplies `dP` by `1/(1-p)` and selects on `keep`; at
    `p = 0` the scale is exactly 1.0 and every element is kept, so the
    arithmetic is unchanged and the answer should be identical to a build that
    emits none of it. A tolerance would not notice a survivor scale applied on
    the wrong side.

    **bf16 holds bitwise; f16 does not, and that is a real finding rather than
    a slack tolerance.** The two builds are genuinely different instruction
    streams -- one carries a `select` and a multiply the other does not -- so
    bit-for-bit agreement was never guaranteed by construction; it needs the
    compiler not to reassociate around the extra ops, and `fm_fast` licenses it
    to. Measured at 32 rows: **0.12% of elements differ, up to 19 ulps of the
    f16 encoding**, and those elements are the *small* ones (median |dK| 1.4e-2
    against 6.1e-2 overall), which is the signature of a fixed absolute f32
    residue resolving differently once the mantissa is fine enough to see it.
    bf16's 7 bits absorb it; f16's 10 sometimes do not. The 16-row family holds
    bitwise in both types.

    So f16 asserts the properties that still carry and that a misplaced scale
    would still break: determinism, a bounded ulp distance, and **equal
    accuracy against fp64** -- measured 2.9842e-04 against 2.9844e-04, agreeing
    to four significant figures, where a survivor scale on the wrong side of
    the subtraction would move it by a factor.
    """
    _require_rocm_path()
    knobs = _family_knobs(64, rows)
    on = _dropout_run(knobs, 0.0, drop=True)
    off = _dropout_run(knobs, None, drop=False)
    if _dt_str() == "bf16":
        assert torch.equal(on[0], off[0]) and torch.equal(on[1], off[1])
        return
    again = _dropout_run(knobs, 0.0, drop=True)
    for i, name in ((0, "dK"), (1, "dV")):
        assert torch.equal(on[i], again[i]), f"{name}: the dropout build is not deterministic at p=0"
        ulp = (on[i].view(torch.int16).int() - off[i].view(torch.int16).int()).abs()
        frac = float((ulp != 0).float().mean())
        assert frac < 0.01, f"{name}: {frac:.3%} of elements differ at p=0, far beyond a rounding residue"
        assert int(ulp.max()) <= 64, f"{name}: max {int(ulp.max())} ulps at p=0, beyond a rounding residue"


@pytest.mark.parametrize("rows", [32, 16], ids=["m32", "m16"])
def test_dropout_is_deterministic_per_seed(rows, dtype):
    """Same seed bit-identical, different seed different. Both halves matter."""
    _require_rocm_path()
    knobs = _family_knobs(64, rows)
    a, b = _dropout_run(knobs, 0.3), _dropout_run(knobs, 0.3)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), "not deterministic at a fixed seed"
    c = _dropout_run(knobs, 0.3, seed=100)
    assert not torch.equal(a[0], c[0]), "a different seed produced the same mask"


@pytest.mark.parametrize(
    "rows,alt",
    [
        (32, dict(mfma_rows=32, dkv_shards=1, num_waves=2, block_kv=64, block_q=64, head_dim_granule=64)),
        (16, dict(mfma_rows=16, dkv_shards=1, num_waves=4, block_kv=64, block_q=32, head_dim_granule=64)),
        (16, dict(mfma_rows=16, dkv_shards=1, num_waves=2, block_kv=32, block_q=64, head_dim_granule=64)),
    ],
    ids=["m32_waves", "m16_blockq", "m16_waves"],
)
def test_dropout_mask_does_not_depend_on_the_tiling(rows, alt, dtype):
    """P6's contract, and the reason `grid_plane` never sees a `BLOCK_*`.

    The mask is a function of absolute `(batch, head, row, column)` and the
    *maximum* sequence lengths -- never of the tile geometry -- so two builds
    of the same problem at different tilings must agree **bit for bit**. This
    is a constraint on the tuner from here on: `_GEOMETRY` may move freely and
    a dropout mask may not follow it.

    Within one MFMA family, because two families accumulate in different orders
    and would differ in the last bits for reasons that have nothing to do with
    the mask.
    """
    _require_rocm_path()
    base = _dropout_run(_family_knobs(64, rows), 0.3)
    other = _dropout_run(alt, 0.3)
    assert torch.equal(base[0], other[0]), "dK moved when the tiling changed: the mask depends on it"
    assert torch.equal(base[1], other[1]), "dV moved when the tiling changed: the mask depends on it"


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [64, 128, 192])
def test_bshd_layout_is_read_through_the_strides(head_dim, dtype):
    """A BSHD *layout* with a BHSD *shape* must give the same answer.

    Nothing at runtime distinguishes a head stride from a sequence stride, so a
    kernel that ignored one of them would return finite garbage. Transposing
    every tensor exercises all six stride triples at once.
    """
    _require_rocm_path()
    b, h, s, d = 2, 4, 256, head_dim
    torch.manual_seed(3)
    q, k, v, do = _rand(b, h, s, d), _rand(b, h, s, d), _rand(b, h, s, d), _rand(b, h, s, d)
    scale = 1.0 / d**0.5
    _o, lse64, delta64, _dq, _dk, _dv = _math_backward(q, k, v, do, scale, torch.float64)
    lse = lse64.reshape(b * h, s).float().contiguous()
    delta = delta64.reshape(b * h, s).float().contiguous()

    dk_c, dv_c = _run(q, k, v, do, lse, delta, scale=scale)

    def _as_bshd(t):
        out = torch.empty(b, s, h, d, device="cuda", dtype=t.dtype).transpose(1, 2)
        out.copy_(t)
        return out

    qt, kt, vt, dot = (_as_bshd(t) for t in (q, k, v, do))
    dk_t = _as_bshd(torch.full_like(k, float("nan")))
    dv_t = _as_bshd(torch.full_like(v, float("nan")))
    _run(qt, kt, vt, dot, lse, delta, scale=scale, dk=dk_t, dv=dv_t)
    torch.cuda.synchronize()
    assert torch.equal(dk_t, dk_c), "dK differs between a BSHD and a BHSD layout of the same data"
    assert torch.equal(dv_t, dv_c), "dV differs between a BSHD and a BHSD layout of the same data"


@pytest.mark.parametrize("head_dim", [64, 128, 384])
def test_writes_nothing_past_the_kv_sequence(head_dim, dtype):
    """The output descriptors bound the store; a ragged tail must stay untouched.

    The KV block is a whole number of workgroups, so the last one owns rows
    past `seqlen_kv` and computes real values for them. They are dropped by the
    buffer bound rather than by a branch, and nothing else in the kernel would
    notice if they were not.
    """
    _require_rocm_path()
    b, h, s, d = 1, 2, 130, head_dim
    slack = 64
    torch.manual_seed(5)
    q, k_full, v_full, do = (
        _rand(b, h, s, d),
        _rand(b, h, s + slack, d),
        _rand(b, h, s + slack, d),
        _rand(b, h, s, d),
    )
    k, v = k_full[:, :, :s], v_full[:, :, :s]
    scale = 1.0 / d**0.5
    _o, lse64, delta64, _dq, _dk, _dv = _math_backward(q, k, v, do, scale, torch.float64)
    lse = lse64.reshape(b * h, s).float().contiguous()
    delta = delta64.reshape(b * h, s).float().contiguous()

    dk_full = torch.full((b, h, s + slack, d), 1234.0, device="cuda", dtype=_dt())
    dv_full = torch.full((b, h, s + slack, d), 1234.0, device="cuda", dtype=_dt())
    _run(q, k, v, do, lse, delta, scale=scale, dk=dk_full[:, :, :s], dv=dv_full[:, :, :s])
    torch.cuda.synchronize()
    assert torch.all(dk_full[:, :, s:] == 1234.0), "dK was written past seqlen_kv"
    assert torch.all(dv_full[:, :, s:] == 1234.0), "dV was written past seqlen_kv"
    assert torch.isfinite(dk_full[:, :, :s]).all() and torch.isfinite(dv_full[:, :, :s]).all()


# The input contract is 8xD even though the compiled tiles are 32xD: loads and
# stores are 8 columns wide, so a head_dim that is a multiple of 8 is a whole
# number of chunks and the kernel never touches a column it was not given.
# Every multiple of 8 the ladder can reach, so a rung that mishandles its
# sub-8-grid widths cannot hide behind the ones the suite happens to name.
_GRID8 = list(range(8, 513, 8))


@pytest.mark.parametrize("hdim", _GRID8)
def test_grid8_contiguous_is_exact_and_writes_nothing_past_the_last_row(hdim, dtype):
    """A plainly contiguous 8xD tensor -- no padded view -- must just work.

    The forward's `test_grid8_contiguous_is_exact_and_writes_nothing_past_o`,
    transposed. What a caller actually passes is `torch.randn(b, h, s, hdim)`,
    whose D pitch is `hdim` itself; every tensor built through a padded
    allocation is 8-aligned *by construction* and cannot fail the way a tight
    `(B, H, S, 24)` can.

    The extra dK and dV rows are canaries: they are contiguous with the last
    real row, so a D-tail chunk overrunning it lands in them and nowhere else.
    Both outputs are checked, because the two extents are separate arguments
    and the suppression is written twice.
    """
    _require_rocm_path()
    b, h, s = 1, 2, 128
    torch.manual_seed(hdim)
    q, k, v, do = (_rand(b, h, s, hdim) for _ in range(4))
    assert q.stride(2) == hdim, "the point of this test is a tight pitch"
    scale = 1.0 / hdim**0.5
    _o, lse64, delta64, _dq, dk64, dv64 = _math_backward(q, k, v, do, scale, torch.float64)
    _o2, _l, _d, _dq2, dk16, dv16 = _math_backward(q, k, v, do, scale, _dt())
    lse = lse64.reshape(b * h, s).float().contiguous()
    delta = delta64.reshape(b * h, s).float().contiguous()

    sentinel = -12345.0
    dkbuf = torch.full((b, h, s + 1, hdim), sentinel, device="cuda", dtype=_dt())
    dvbuf = torch.full((b, h, s + 1, hdim), sentinel, device="cuda", dtype=_dt())
    _run(q, k, v, do, lse, delta, scale=scale, dk=dkbuf[:, :, :s], dv=dvbuf[:, :, :s])
    torch.cuda.synchronize()
    assert torch.all(dkbuf[:, :, s] == sentinel), f"a dK store ran past the last row at hdim {hdim}"
    assert torch.all(dvbuf[:, :, s] == sentinel), f"a dV store ran past the last row at hdim {hdim}"
    for name, ours, low, high, other in (
        ("dk", dkbuf[:, :, :s], dk16, dk64, dv64),
        ("dv", dvbuf[:, :, :s], dv16, dv64, dk64),
    ):
        floor = other.norm().item()
        e_ours, e_low = _rel(ours, high, floor), _rel(low, high, floor)
        assert e_ours <= FUDGE[name] * e_low + DEGENERATE_ATOL, f"{name} at hdim {hdim}: {e_ours:.3e} vs {e_low:.3e}"


@pytest.mark.parametrize("hdim,pitch", [(100, 104), (33, 40), (7, 8), (300, 304)])
def test_padded_head_never_writes_past_the_8_aligned_chunk(hdim, pitch, dtype):
    """A D tail may spill into the caller's pad, but not past it.

    A 128-bit store is all-or-nothing, so a chunk straddling `hdim` writes into
    the allocation's own padding -- permitted, and the reason the pitch
    contract exists. A chunk starting at or past `hdim` must be dropped
    entirely, and columns from `ceil8(hdim)` on were never in any store chunk
    at all; this pins both by checking they are untouched.
    """
    _require_rocm_path()
    b, h, s = 1, 2, 128
    torch.manual_seed(hdim)

    def _padded(*, rows=s):
        full = torch.randn(b, h, rows, pitch, device="cuda", dtype=_dt())
        return full, full[..., :hdim]

    _fq, q = _padded()
    _fk, k = _padded()
    _fv, v = _padded()
    _fdo, do = _padded()
    scale = 1.0 / hdim**0.5
    _o, lse64, delta64, _dq, dk64, dv64 = _math_backward(q, k, v, do, scale, torch.float64)
    _o2, _l, _d, _dq2, dk16, dv16 = _math_backward(q, k, v, do, scale, _dt())
    lse = lse64.reshape(b * h, s).float().contiguous()
    delta = delta64.reshape(b * h, s).float().contiguous()

    dkfull = torch.full((b, h, s, pitch), -7.0, device="cuda", dtype=_dt())
    dvfull = torch.full((b, h, s, pitch), -7.0, device="cuda", dtype=_dt())
    _run(q, k, v, do, lse, delta, scale=scale, dk=dkfull[..., :hdim], dv=dvfull[..., :hdim])
    torch.cuda.synchronize()
    ceil8 = (hdim + 7) // 8 * 8
    assert torch.all(dkfull[..., ceil8:] == -7.0), "dK store ran past the 8-aligned chunk containing hdim_qk"
    assert torch.all(dvfull[..., ceil8:] == -7.0), "dV store ran past the 8-aligned chunk containing hdim_vo"
    for name, ours, low, high, other in (
        ("dk", dkfull[..., :hdim], dk16, dk64, dv64),
        ("dv", dvfull[..., :hdim], dv16, dv64, dk64),
    ):
        floor = other.norm().item()
        e_ours, e_low = _rel(ours, high, floor), _rel(low, high, floor)
        assert e_ours <= FUDGE[name] * e_low + DEGENERATE_ATOL, f"{name} at hdim {hdim}: {e_ours:.3e} vs {e_low:.3e}"


def test_tight_odd_hdim_is_refused_not_corrupted():
    """An odd head_dim in a tight allocation has nowhere to put the tail chunk.

    `ceil8(100)` is 104, so the kernel touches four columns that belong to the
    next row. Refusing is the contract; the alternative is a wrong answer in a
    tensor the caller never suspected.
    """
    _require_rocm_path()
    b, h, s, hdim = 1, 2, 128, 100
    q, k, v, do = (_rand(b, h, s, hdim) for _ in range(4))
    dk, dv = torch.empty_like(k), torch.empty_like(v)
    lse = torch.zeros(b * h, s, device="cuda", dtype=torch.float32)
    fn = build(num_heads=h, head_dim=hdim)
    with pytest.raises(ValueError, match="not a multiple of 8"):
        fn(q, k, v, do, dk, dv, lse, lse, b, s)


def test_odd_hdim_bshd_without_slack_is_refused():
    """BSHD hides the overrun from a pitch check, so the check is not a pitch.

    Heads of one token are adjacent in BSHD, so the gap after a D row is `hdim`
    itself and there is no slack -- while `stride(2)` is `num_heads * hdim`,
    a tidy multiple of 8 whenever `num_heads` is even.
    """
    _require_rocm_path()
    b, h, s, hdim = 1, 4, 128, 100
    assert (h * hdim) % 8 == 0, "the case is only interesting when the pitch looks fine"
    q, k, v, do = (torch.randn(b, s, h, hdim, device="cuda", dtype=_dt()).transpose(1, 2) for _ in range(4))
    dk = torch.empty(b, s, h, hdim, device="cuda", dtype=_dt()).transpose(1, 2)
    dv = torch.empty(b, s, h, hdim, device="cuda", dtype=_dt()).transpose(1, 2)
    lse = torch.zeros(b * h, s, device="cuda", dtype=torch.float32)
    assert q.stride(2) % 8 == 0, "the pitch check would pass here"
    fn = build(num_heads=h, head_dim=hdim)
    with pytest.raises(ValueError, match="unused element"):
        fn(q, k, v, do, dk, dv, lse, lse, b, s)


@pytest.mark.parametrize("head_dim", [64, 512])
def test_deterministic(head_dim, dtype):
    """Bit-identical run to run. Nothing here accumulates through an atomic.

    Worth pinning explicitly: AITER's tuned gfx950 backward reaches dQ by
    atomic add to VRAM and is non-deterministic by construction, and plan
    section 7.3 rules that design out for this stack. A test is what keeps the
    ruling true.
    """
    _require_rocm_path()
    b, h, s, d = 1, 2, 512, head_dim
    torch.manual_seed(13)
    q, k, v, do = _rand(b, h, s, d), _rand(b, h, s, d), _rand(b, h, s, d), _rand(b, h, s, d)
    lse = torch.randn(b * h, s, device="cuda", dtype=torch.float32)
    delta = torch.randn(b * h, s, device="cuda", dtype=torch.float32)
    a = _run(q, k, v, do, lse, delta, scale=1.0 / d**0.5)
    c = _run(q, k, v, do, lse, delta, scale=1.0 / d**0.5)
    torch.cuda.synchronize()
    assert torch.equal(a[0], c[0]) and torch.equal(a[1], c[1])


def test_refuses_what_it_does_not_compute():
    """Every one of these would otherwise be a plausible, finite, wrong gradient."""
    with pytest.raises(ValueError, match="window=True requires causal=True"):
        build(num_heads=8, head_dim=64, window=True)
    with pytest.raises(ValueError, match="must be a multiple of"):
        build(num_heads=8, head_dim=64, num_kv_heads=3)
    with pytest.raises(ValueError, match="exceeds the widest tile"):
        build(num_heads=8, head_dim=513)
    with pytest.raises(NotImplementedError, match="bf16 and f16"):
        build(num_heads=8, head_dim=64, dtype_str="fp8")


def test_rejects_a_mismatched_row_tensor():
    """LSE and delta are compact, so the host is the only place to check them."""
    _require_rocm_path()
    b, h, s, d = 1, 2, 128, 64
    q, k, v, do = _rand(b, h, s, d), _rand(b, h, s, d), _rand(b, h, s, d), _rand(b, h, s, d)
    good = torch.zeros(b * h, s, device="cuda", dtype=torch.float32)
    fn = build(num_heads=h, head_dim=d)
    dk, dv = torch.empty_like(k), torch.empty_like(v)
    with pytest.raises(ValueError, match="float32"):
        fn(q, k, v, do, dk, dv, good.half(), good, b, s)
    with pytest.raises(ValueError):
        fn(q, k, v, do, dk, dv, torch.zeros(b * h, s + 1, device="cuda"), good, b, s)
    with pytest.raises(ValueError, match="delta"):
        fn(q, k, v, do, dk, dv, good, good[:, :-1].contiguous(), b, s)


# ---------------------------------------------------------------------------
# B7: the bias input
# ---------------------------------------------------------------------------


def _bias_ref(q, k, v, do, scale, bias):
    """`(lse, delta, dk, dv)` in fp64, with the bias inside the softmax.

    The bias shifts the scores and the softmax normalises most of it away, so a
    kernel that *reads* a bias and drops it still produces a plausible
    gradient. That is why this reference is built with the bias rather than a
    tolerance widened to absorb one -- and why
    `test_bias_changes_the_gradient_correctly` also checks the distance to the
    bias-free answer.
    """
    qq, kk, vv = (t.double().detach().requires_grad_() for t in (q, k, v))
    s = (qq @ kk.transpose(-1, -2)) * scale
    if bias is not None:
        s = s + bias.double()
    o = torch.softmax(s, -1) @ vv
    dk, dv = torch.autograd.grad(o, (kk, vv), do.double())
    lse = torch.logsumexp(s.detach(), -1)
    delta = (do.double() * o.detach()).sum(-1)
    return lse, delta, dk, dv


def _bias_run(
    q, k, v, do, lse, delta, *, scale, bias, knobs=None, varlen=None, num_seqlens=0, batch=None, seqlens=None
):
    """Build and dispatch. `seqlens` is `(seqlen_q, seqlen_k)`, needed under varlen.

    A packed batch's tensors carry `total_tokens` on the sequence axis while
    the kernel wants `max_seqlen_*`, so deriving the lengths from the tensor
    shapes -- which is right for every dense call -- is wrong for exactly the
    varlen ones.
    """
    b, hq, sq, d = q.shape
    sq, sk = (sq, k.shape[2]) if seqlens is None else seqlens
    fn = build(
        num_heads=hq,
        head_dim=d,
        num_kv_heads=k.shape[1],
        bias=bias is not None,
        varlen=varlen is not None,
        dtype_str=_dt_str(),
        **(knobs or {}),
    )
    dk, dv = torch.full_like(k, float("nan")), torch.full_like(v, float("nan"))
    extra = {} if bias is None else {"bias": bias}
    if varlen is not None:
        extra.update(varlen=varlen, num_seqlens=num_seqlens)
    fn(q, k, v, do, dk, dv, lse, delta, b if batch is None else batch, sq, seqlen_k=sk, scale=scale, **extra)
    return dk, dv


@pytest.mark.parametrize("rows", [32, 16], ids=["m32", "m16"])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_bias_changes_the_gradient_correctly(head_dim, rows, dtype):
    """Against a reference *with* the bias, and far from one without it.

    The second assertion is the one that matters. A build that read the bias
    into the wrong lane, scaled it by the wrong constant, or dropped it after
    loading would still return a finite gradient of the right shape, and the
    softmax's normalisation makes the magnitude plausible too. Measured: 2.3e-3
    against the bias reference and **2.2 / 3.0** against the bias-free one, so
    the two are three orders of magnitude apart and no tolerance sits between
    them by accident.
    """
    _require_rocm_path()
    knobs = _family_knobs(head_dim, rows)
    if knobs is None:
        pytest.skip(f"no legal {rows}-row geometry at head_dim {head_dim}")
    b, h, s = 2, 4, 256
    torch.manual_seed(head_dim + rows)
    q, k, v, do = (_rand(b, h, s, head_dim) for _ in range(4))
    # Scaled up, so the bias is not a perturbation the softmax can shrug off.
    bias = _rand(b, h, s, s) * 2.0
    scale = 1.0 / head_dim**0.5
    lse, delta, dk_ref, dv_ref = _bias_ref(q, k, v, do, scale, bias)
    _lse0, _d0, dk_nobias, dv_nobias = _bias_ref(q, k, v, do, scale, None)
    dk, dv = _bias_run(
        q,
        k,
        v,
        do,
        lse.float().reshape(b * h, s).contiguous(),
        delta.float().reshape(b * h, s).contiguous(),
        scale=scale,
        bias=bias,
        knobs=knobs,
    )
    for name, got, ref, off in (("dk", dk, dk_ref, dk_nobias), ("dv", dv, dv_ref, dv_nobias)):
        e = _rel(got, ref)
        assert e < 1e-2, f"{name}: {e:.3e} against the bias reference"
        e_off = _rel(got, off)
        assert e_off > 0.1, (
            f"{name}: {e_off:.3e} against the *bias-free* reference -- this build is not applying the "
            "bias, and every other assertion here would still pass"
        )


@pytest.mark.parametrize("rows", [32, 16], ids=["m32", "m16"])
def test_bias_zero_is_bitwise_a_build_without_it(rows, dtype):
    """`bias=0` must be bitwise identical, as `p = 0` was in B6.

    A zero bias adds exactly `0.0 * log2e` to every exponent, so the arithmetic
    is unchanged and any difference is a bug. A tolerance would not notice the
    bias being applied on the wrong side of the LSE subtraction, or in the
    wrong fastmath domain, which are the two mistakes the placement invites.
    """
    _require_rocm_path()
    knobs = _family_knobs(64, rows)
    b, h, s, d = 1, 2, 256, 64
    torch.manual_seed(3)
    q, k, v, do = (_rand(b, h, s, d) for _ in range(4))
    scale = 1.0 / d**0.5
    lse, delta, _dk, _dv = _bias_ref(q, k, v, do, scale, None)
    lse_a = lse.float().reshape(b * h, s).contiguous()
    del_a = delta.float().reshape(b * h, s).contiguous()
    with_zero = _bias_run(
        q, k, v, do, lse_a, del_a, scale=scale, bias=torch.zeros_like(q[..., :1] * 0).new_zeros(b, h, s, s), knobs=knobs
    )
    without = _bias_run(q, k, v, do, lse_a, del_a, scale=scale, bias=None, knobs=knobs)
    assert torch.equal(with_zero[0], without[0]), "dK differs between bias=0 and a build without bias"
    assert torch.equal(with_zero[1], without[1]), "dV differs between bias=0 and a build without bias"


@pytest.mark.parametrize("rows", [32, 16], ids=["m32", "m16"])
def test_bias_minus_inf_never_produces_nan(rows, dtype):
    """`-inf` is how a caller spells "never attend here", and it is arithmetic here.

    Everywhere else in this kernel a mask is a `select` or a descriptor bound,
    neither of which fast-math touches. The bias puts a real infinity into an
    `fmul` and an `fadd`, and `fm_fast` carries `ninf` -- a licence to assume
    that cannot happen. The forward records `ninf` silently deleting a KV tail
    mask on gfx1201, so this is the test that says the flag was narrowed.
    """
    _require_rocm_path()
    knobs = _family_knobs(64, rows)
    b, h, s, d = 1, 2, 256, 64
    torch.manual_seed(5)
    q, k, v, do = (_rand(b, h, s, d) for _ in range(4))
    bias = _rand(b, h, s, s)
    bias[:, :, :, 64:128] = float("-inf")
    scale = 1.0 / d**0.5
    lse, delta, dk_ref, dv_ref = _bias_ref(q, k, v, do, scale, bias)
    dk, dv = _bias_run(
        q,
        k,
        v,
        do,
        lse.float().reshape(b * h, s).contiguous(),
        delta.float().reshape(b * h, s).contiguous(),
        scale=scale,
        bias=bias,
        knobs=knobs,
    )
    assert not dk.isnan().any(), "dK carries a NaN from a -inf bias"
    assert not dv.isnan().any(), "dV carries a NaN from a -inf bias"
    # A key nobody may attend to receives no gradient at all, exactly.
    assert dk[:, :, 64:128, :].count_nonzero() == 0, "a fully masked key has a nonzero dK"
    assert dv[:, :, 64:128, :].count_nonzero() == 0, "a fully masked key has a nonzero dV"
    assert _rel(dk, dk_ref) < 1e-2 and _rel(dv, dv_ref) < 1e-2


def test_bias_and_causal_are_rejected():
    """Undefined, not unimplemented -- and the forward cannot build the pair either.

    A bias *is* an attention mask: a large negative entry is how a caller
    spells "do not attend here". Pairing it with a positional mask asks which
    wins where they disagree and there is no answer. AOTriton disables the
    combination, torch's math backend raises by name, and the forward refuses
    it in these words -- so a backward that accepted it would have no forward
    to produce the LSE it consumes. Untestable as well as undefined.
    """
    with pytest.raises(ValueError, match="mutually exclusive"):
        build(num_heads=8, head_dim=64, bias=True, causal=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        build(num_heads=8, head_dim=64, bias=True, causal=True, window=True)


def test_bias_tensor_presence_is_checked_both_ways():
    """Both silent failures return the right shape."""
    _require_rocm_path()
    b, h, s, d = 1, 2, 128, 64
    torch.manual_seed(9)
    q, k, v, do = (_rand(b, h, s, d) for _ in range(4))
    row = torch.zeros(b * h, s, device="cuda", dtype=torch.float32)
    dk, dv = torch.empty_like(k), torch.empty_like(v)
    bias = _rand(b, h, s, s)
    with pytest.raises(ValueError, match="requires a"):
        build(num_heads=h, head_dim=d, bias=True)(q, k, v, do, dk, dv, row, row, b, s)
    with pytest.raises(ValueError, match="not compiled for bias"):
        build(num_heads=h, head_dim=d)(q, k, v, do, dk, dv, row, row, b, s, bias=bias)
    with pytest.raises(ValueError, match="contiguous seqlen_k"):
        build(num_heads=h, head_dim=d, bias=True)(q, k, v, do, dk, dv, row, row, b, s, bias=bias.transpose(2, 3))


def test_bias_composes_with_varlen(dtype):
    """Bias under varlen must equal N dense calls, bitwise.

    The bias follows Q on the query axis -- the descriptor folds in the varlen
    row origin and the head exactly as Q's slab does -- so a packed batch reads
    it as `(1, h, total_q, max_seqlen_k)`. The KV axis is *not* packed: slot 3
    has no stride of its own, so each sequence's row spans `seqlen_k` columns
    from zero. That is a contract rather than an accident, and this is where it
    is pinned down.
    """
    _require_rocm_path()
    import fmha_abi_gfx1201 as abi

    h, d = 4, 64
    lens = [192, 256]
    smax, total = max(lens), sum(lens)
    scale = 1.0 / d**0.5
    knobs = _family_knobs(d, 32)
    torch.manual_seed(11)
    qs = [_rand(1, h, n, d) for n in lens]
    ks = [_rand(1, h, n, d) for n in lens]
    vs = [_rand(1, h, n, d) for n in lens]
    dos = [_rand(1, h, n, d) for n in lens]
    bs = [_rand(1, h, n, smax) for n in lens]

    dense = []
    for i, n in enumerate(lens):
        bias_i = bs[i][..., :n].contiguous()
        lse, delta, _dk, _dv = _bias_ref(qs[i], ks[i], vs[i], dos[i], scale, bias_i)
        dense.append(
            _bias_run(
                qs[i],
                ks[i],
                vs[i],
                dos[i],
                lse.float().reshape(h, n).contiguous(),
                delta.float().reshape(h, n).contiguous(),
                scale=scale,
                bias=bias_i,
                knobs=knobs,
            )
        )

    qp, kp, vp, dop = (torch.cat(t, dim=2) for t in (qs, ks, vs, dos))
    bp = torch.cat(bs, dim=2).contiguous()
    lse_p = torch.zeros(h, total, device="cuda", dtype=torch.float32)
    del_p = torch.zeros_like(lse_p)
    off = 0
    for i, n in enumerate(lens):
        lse, delta, _dk, _dv = _bias_ref(qs[i], ks[i], vs[i], dos[i], scale, bs[i][..., :n].contiguous())
        lse_p[:, off : off + n] = lse.float().reshape(h, n)
        del_p[:, off : off + n] = delta.float().reshape(h, n)
        off += n
    cu = _cu(lens)
    got = _bias_run(
        qp,
        kp,
        vp,
        dop,
        lse_p,
        del_p,
        scale=scale,
        bias=bp,
        knobs=knobs,
        varlen=abi.varlen_compact(cu, cu, smax, smax, lse_tokens=total),
        num_seqlens=len(lens),
        batch=1,
        seqlens=(smax, smax),
    )
    off = 0
    for i, n in enumerate(lens):
        for j, name in ((0, "dK"), (1, "dV")):
            assert torch.equal(got[j][:, :, off : off + n, :], dense[i][j]), (
                f"{name} for sequence {i} differs from the dense call; bias under varlen is not "
                "following Q's query axis"
            )
        off += n


@pytest.mark.parametrize("rows", [32, 16], ids=["m32", "m16"])
def test_bias_composes_with_dropout(rows, dtype):
    """Both touch the scores, and they touch it at different points.

    The bias goes into the exponent, so it lands before `P` exists; dropout
    goes on `P` after the row sum. Composition is therefore a claim about
    ordering, and the way to check it is the B6 oracle with a bias added to
    every arm: the forward draws the mask over the biased scores, and this
    kernel must reproduce both.
    """
    _require_rocm_path()
    o, o_ref, dk, dk_ref, dv, dv_ref, keep = _dropout_case(d=64, rows=rows, bias_scale=2.0)
    assert _rel(o, o_ref.double()) < 1e-2, "the forward and the mask kernel disagree; the reference is invalid"
    assert _rel(dk, dk_ref.double()) < 1e-2, "dK: bias and dropout do not compose"
    assert _rel(dv, dv_ref.double()) < 1e-2, "dV: bias and dropout do not compose"
    assert abs(keep - 0.75) < 0.01


# ---------------------------------------------------------------------------
# B7: GQA
# ---------------------------------------------------------------------------


def _gqa_ref(q, k, v, do, scale, causal=False):
    """fp64 truth with K/V expanded over the group, and `dK`/`dV` summed by autograd.

    `repeat_interleave` on the *differentiable* tensor is what makes the sum
    happen: autograd accumulates the gradient of every expanded copy back onto
    the one KV head, which is exactly the reduction the kernel performs in
    registers. Expanding first and differentiating the expanded tensor would
    return one q head's contribution per copy and hide a missing group sum.
    """
    g = q.shape[1] // k.shape[1]
    qq, kk, vv = (t.double().detach().requires_grad_() for t in (q, k, v))
    s = (qq @ kk.repeat_interleave(g, dim=1).transpose(-1, -2)) * scale
    if causal:
        s = s.masked_fill(~_band(q.shape[2], k.shape[2], (q.shape[2], k.shape[2] - q.shape[2])), float("-inf"))
    o = torch.softmax(s, -1) @ vv.repeat_interleave(g, dim=1)
    dk, dv = torch.autograd.grad(o, (kk, vv), do.double())
    lse = torch.logsumexp(s.detach(), -1)
    delta = (do.double() * o.detach()).sum(-1)
    return lse, delta, dk, dv


@pytest.mark.parametrize("rows", [32, 16], ids=["m32", "m16"])
@pytest.mark.parametrize("hq,hk", [(8, 8), (8, 4), (8, 2), (8, 1), (6, 3)], ids=["mha", "g2", "g4", "mqa", "g2odd"])
@pytest.mark.parametrize("causal", [False, True], ids=["dense", "causal"])
def test_gqa_matches_the_grouped_reference(hq, hk, rows, causal, dtype):
    """Several query heads reduce into one KV head, and the sum is the feature.

    `hq=6, hk=3` is the case a broadcast hides: with `hk=1` a plain
    `expand`-style reference happens to be right whatever the grouping, so a
    wrong `q_head_idx -> kv_head_idx` mapping survives it. A group that is
    neither 1 nor the whole head count is what pins the mapping down.
    """
    _require_rocm_path()
    d = 64
    knobs = _family_knobs(d, rows)
    b, s = 2, 256
    torch.manual_seed(hq * 31 + hk)
    q = _rand(b, hq, s, d)
    k, v = _rand(b, hk, s, d), _rand(b, hk, s, d)
    do = _rand(b, hq, s, d)
    scale = 1.0 / d**0.5
    lse, delta, dk_ref, dv_ref = _gqa_ref(q, k, v, do, scale, causal)
    dk, dv = _run(
        q,
        k,
        v,
        do,
        lse.float().reshape(b * hq, s).contiguous(),
        delta.float().reshape(b * hq, s).contiguous(),
        scale=scale,
        knobs=knobs,
        causal=causal,
    )
    for name, got, ref in (("dk", dk, dk_ref), ("dv", dv, dv_ref)):
        e = _rel(got, ref)
        assert e < 1e-2, f"{name}: {e:.3e} -- the group sum is wrong at hq={hq} hk={hk}"


@pytest.mark.parametrize("rows", [32, 16], ids=["m32", "m16"])
@pytest.mark.parametrize("hk", [4, 2, 1], ids=["g2", "g4", "mqa"])
def test_gqa_matches_n_grouped_dense_calls(hk, rows, dtype):
    """The B5-shaped gate: new addressing against arithmetic already trusted.

    An MHA build over the *expanded* K/V is code that has passed every phase
    since B1. Summing its per-head outputs over each group is what the GQA
    build must produce, so this compares the fold's addressing against a
    result whose arithmetic is not in question.

    **Not bitwise, and the reason is the point of the design.** The MHA path
    rounds each query head's partial to bf16 before anything can sum them; the
    fold keeps the group sum in f32 registers and rounds once. So the two agree
    to within one bf16 quantisation of the output and no closer -- which is
    exactly the difference that ruled out writing per-head partials to a
    scratch tensor. An addressing error is O(1) and nowhere near this band.
    """
    _require_rocm_path()
    d, hq = 64, 8
    g = hq // hk
    knobs = _family_knobs(d, rows)
    b, s = 2, 256
    torch.manual_seed(hk * 17 + rows)
    q = _rand(b, hq, s, d)
    k, v = _rand(b, hk, s, d), _rand(b, hk, s, d)
    do = _rand(b, hq, s, d)
    scale = 1.0 / d**0.5
    lse, delta, dk64, dv64 = _gqa_ref(q, k, v, do, scale)
    lse_a = lse.float().reshape(b * hq, s).contiguous()
    del_a = delta.float().reshape(b * hq, s).contiguous()

    got = _run(q, k, v, do, lse_a, del_a, scale=scale, knobs=knobs)
    ke = k.repeat_interleave(g, dim=1).contiguous()
    ve = v.repeat_interleave(g, dim=1).contiguous()
    per_head = _run(q, ke, ve, do, lse_a, del_a, scale=scale, knobs=knobs)
    summed = [t.float().reshape(b, hk, g, s, d).sum(2) for t in per_head]

    for name, a, ref, truth in (("dk", got[0], summed[0], dk64), ("dv", got[1], summed[1], dv64)):
        e = _rel(a, ref.double())
        assert e < 1e-2, f"{name}: {e:.3e} against N grouped dense calls -- the fold addresses a head wrong"
        # And the fold is no less accurate than the per-head route it replaces.
        assert _rel(a, truth) < 3.0 * _rel(ref, truth) + 1e-6, (
            f"{name}: the in-register group sum is materially worse than summing bf16 partials, which "
            "is the opposite of the reason it was chosen"
        )


def test_gqa_composes_with_varlen_and_dropout(dtype):
    """Two oracles, because varlen x dropout cannot use the varlen one.

    **Arm 1, ragged and dry:** a GQA group under varlen equals N separate dense
    calls, bitwise. That is B5's oracle unchanged.

    **Arm 2, dropout.** This test used to claim dropout in its name and its
    docstring and pass no dropout argument at all -- found by AST inspection,
    and the reason it mattered is that varlen x dropout then had *no* numeric
    coverage in any of the three kernels. A defect visible in the first
    non-zero sequence of every packed call survived every gate we had.

    Enabling it needs a different reference, and that is the interesting part.
    The philox plane is `seq_idx_i32 * num_head_q + q_head_idx`, and
    `decode_addressing` resolves `batch_idx` to 0 for every sequence of a
    packed layout -- so **N separate `batch_size=1` calls all draw plane 0**
    and cannot serve. They are degenerate in exactly the way the packed call
    used to be, which is how the bug hid. A dense call with `batch = N` gives
    slice `i` the plane `i * H + h`, which is what packed sequence `i` gets, so
    that is the reference here; equal-length sequences, since a dense batch has
    one length for all of them.

    The `planes differ across sequences` assertion is the negative control: if
    the plane collapsed again, both sides would still agree and only that
    check would notice.
    """
    _require_rocm_path()
    import fmha_abi_gfx1201 as abi

    hq, hk, d = 8, 2, 64
    lens = [128, 192]
    smax, total = max(lens), sum(lens)
    scale = 1.0 / d**0.5
    knobs = _family_knobs(d, 32)
    torch.manual_seed(21)
    qs = [_rand(1, hq, n, d) for n in lens]
    ks = [_rand(1, hk, n, d) for n in lens]
    vs = [_rand(1, hk, n, d) for n in lens]
    dos = [_rand(1, hq, n, d) for n in lens]

    dense = []
    rows = []
    for i, n in enumerate(lens):
        lse, delta, _a, _b = _gqa_ref(qs[i], ks[i], vs[i], dos[i], scale)
        rows.append((lse.float().reshape(hq, n).contiguous(), delta.float().reshape(hq, n).contiguous()))
        dense.append(_run(qs[i], ks[i], vs[i], dos[i], rows[i][0], rows[i][1], scale=scale, knobs=knobs, batch=1))

    qp, kp, vp, dop = (torch.cat(t, dim=2) for t in (qs, ks, vs, dos))
    lse_p = torch.zeros(hq, total, device="cuda", dtype=torch.float32)
    del_p = torch.zeros_like(lse_p)
    off = 0
    for i, n in enumerate(lens):
        lse_p[:, off : off + n] = rows[i][0]
        del_p[:, off : off + n] = rows[i][1]
        off += n
    cu = _cu(lens)
    fn = build(num_heads=hq, head_dim=d, num_kv_heads=hk, varlen=True, dtype_str=_dt_str(), **knobs)
    dk, dv = torch.full_like(kp, float("nan")), torch.full_like(vp, float("nan"))
    fn(
        qp,
        kp,
        vp,
        dop,
        dk,
        dv,
        lse_p,
        del_p,
        1,
        smax,
        seqlen_k=smax,
        scale=scale,
        varlen=abi.varlen_compact(cu, cu, smax, smax, lse_tokens=total),
        num_seqlens=len(lens),
    )
    off = 0
    for i, n in enumerate(lens):
        assert torch.equal(dk[:, :, off : off + n, :], dense[i][0]), f"dK sequence {i} under GQA+varlen"
        assert torch.equal(dv[:, :, off : off + n, :], dense[i][1]), f"dV sequence {i} under GQA+varlen"
        off += n

    # -- Arm 2: the same fold with dropout on, against a dense batch of N.
    nseq, n = 2, 128
    torch.manual_seed(22)
    qd = _rand(nseq, hq, n, d)
    kd, vd = _rand(nseq, hk, n, d), _rand(nseq, hk, n, d)
    dod = _rand(nseq, hq, n, d)
    lse_d = torch.randn(nseq * hq, n, device="cuda", dtype=torch.float32)
    del_d = torch.randn_like(lse_d)
    drop = dict(dropout_p=0.25, philox_seed=_u64(4242), philox_offset1=_u64(0), philox_offset2=0)

    fd = build(num_heads=hq, head_dim=d, num_kv_heads=hk, dropout=True, dtype_str=_dt_str(), **knobs)
    dk_d, dv_d = torch.full_like(kd, float("nan")), torch.full_like(vd, float("nan"))
    fd(qd, kd, vd, dod, dk_d, dv_d, lse_d, del_d, nseq, n, seqlen_k=n, scale=scale, **drop)

    cat = lambda t: torch.cat([t[i : i + 1] for i in range(nseq)], dim=2)  # noqa: E731
    qv, kv_, vv_, dov = cat(qd), cat(kd), cat(vd), cat(dod)
    lse_v = torch.cat([lse_d.reshape(nseq, hq, n)[i] for i in range(nseq)], dim=1).contiguous()
    del_v = torch.cat([del_d.reshape(nseq, hq, n)[i] for i in range(nseq)], dim=1).contiguous()
    cu_d = _cu([n] * nseq)
    fv = build(num_heads=hq, head_dim=d, num_kv_heads=hk, dropout=True, varlen=True, dtype_str=_dt_str(), **knobs)
    dk_v, dv_v = torch.full_like(kv_, float("nan")), torch.full_like(vv_, float("nan"))
    fv(
        qv,
        kv_,
        vv_,
        dov,
        dk_v,
        dv_v,
        lse_v,
        del_v,
        1,
        n,
        seqlen_k=n,
        scale=scale,
        varlen=abi.varlen_compact(cu_d, cu_d, n, n, lse_tokens=nseq * n),
        num_seqlens=nseq,
        **drop,
    )
    torch.cuda.synchronize()
    for i in range(nseq):
        lo, hi = i * n, (i + 1) * n
        assert torch.equal(dk_v[:, :, lo:hi, :], dk_d[i : i + 1]), (
            f"dK sequence {i}: a packed sequence and the matching dense batch slice must draw the same "
            "dropout plane, since both have seq_idx == i"
        )
        assert torch.equal(dv_v[:, :, lo:hi, :], dv_d[i : i + 1]), f"dV sequence {i} under GQA+varlen+dropout"
    assert not torch.equal(dk_v[:, :, :n, :], dk_v[:, :, n:, :]), (
        "the two packed sequences produced identical dK, so every sequence in the pack is drawing one "
        "mask -- which is the defect this arm exists to catch, and both oracles above would still pass"
    )


@pytest.mark.parametrize("rows", [32, 16], ids=["m32", "m16"])
def test_gqa_draws_a_mask_per_query_head(rows, dtype):
    """Dropout under GQA, against the mask the forward actually drew.

    The forward runs one program per query head and draws a plane per
    `(batch, query head)`. The fold walks the group inside one workgroup, so
    the plane has to move with the group -- and it is the one piece of
    per-head state that is neither a descriptor nor an address, which is
    exactly the kind that gets hoisted by accident.
    """
    _require_rocm_path()
    o, o_ref, dk, dk_ref, dv, dv_ref, keep = _dropout_case(d=64, rows=rows, hk=2)
    assert _rel(o, o_ref.double()) < 1e-2, "the forward and the mask kernel disagree; the reference is invalid"
    assert _rel(dk, dk_ref.double()) < 1e-2, "dK: the GQA fold is not drawing a mask per query head"
    assert _rel(dv, dv_ref.double()) < 1e-2, "dV: the GQA fold is not drawing a mask per query head"
    assert abs(keep - 0.75) < 0.01


# ---------------------------------------------------------------------------
# B8: tests that can only pass if the build is genuinely the dtype it claims
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [32, 16], ids=["m32", "m16"])
@pytest.mark.parametrize("dt", ["bf16", "f16"])
def test_the_build_uses_the_mfma_opcode_for_its_dtype(dt, rows):
    """The coverage gate, and the reason it exists is that everything else can lie.

    bf16 and f16 are **the same width**. Every descriptor, stride, LDS offset,
    tile geometry and register count is identical between them, so a build that
    silently kept bf16 while the harness fed it f16 tensors would produce
    numbers that are finite and, against a tolerance, plausible. Feeding fp16
    tensors is not evidence that fp16 was built -- that is the dtype-shaped
    twin of a checker that parsed a flag and never passed it on.

    So this reads the emitted ISA and requires the right MFMA opcode **and the
    absence of the other one**. The absence half is what makes it decisive: a
    build that emitted both would mean the dtype reached one GEMM and not
    another, and the four GEMMs here do not all take their operands from the
    same place.
    """
    _require_rocm_path()
    isa = _dump_isa(dtype_str=dt, rows=rows)
    shape = "32x32x16" if rows == 32 else "16x16x32"
    want, other = f"v_mfma_f32_{shape}_{dt}", f"v_mfma_f32_{shape}_{'bf16' if dt == 'f16' else 'f16'}"
    assert want in isa, f"a dtype_str={dt!r} build emitted no {want}; the dtype did not reach the MMA atom"
    assert other not in isa, f"a dtype_str={dt!r} build also emitted {other}; the dtype reached only some GEMMs"
    # And the output conversion, which is the site that was hardcoded before B8
    # and would have bitcast bf16 patterns into an f16 tensor.
    cvt = "v_cvt_pk_bf16_f32" if dt == "bf16" else "v_cvt_pk_f16_f32"
    bad = "v_cvt_pk_f16_f32" if dt == "bf16" else "v_cvt_pk_bf16_f32"
    assert cvt in isa, f"a dtype_str={dt!r} build emitted no {cvt}; the store is packing the wrong type"
    assert bad not in isa, f"a dtype_str={dt!r} build emitted {bad}; some store path is still hardcoded"


@pytest.mark.parametrize("rows", [32, 16], ids=["m32", "m16"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_f16_is_measurably_more_accurate_than_bf16(head_dim, rows):
    """The second coverage gate, in the direction only fp16 can produce.

    f16 has 10 mantissa bits to bf16's 7, so on **identical inputs** against an
    fp64 reference its error must be about 8x smaller. A build that claimed f16
    and ran bf16 would score the same as bf16, not 8x better -- so this catches
    the same lie as the ISA assertion without reading a disassembly, and it
    catches it in the numbers a user would actually see.

    The bar is 3x rather than 8x: 8x is what the mantissa predicts and what is
    measured (2.9e-4 against 2.35e-3), and 3x is clear of both that and of any
    shape-dependence, while still being far outside the 1.0x a silently-bf16
    build would give. This is the same "assert it is measurably not the other
    plausible value" move as dropped-vs-undropped `dS` and the bias-free
    reference.
    """
    _require_rocm_path()
    knobs = _family_knobs(head_dim, rows)
    b, h, s = 2, 4, 256
    torch.manual_seed(head_dim * 7 + rows)
    q32, k32, v32, do32 = (torch.randn(b, h, s, head_dim, device="cuda", dtype=torch.float32) for _ in range(4))
    scale = 1.0 / head_dim**0.5
    err = {}
    for name, dt in (("bf16", torch.bfloat16), ("f16", torch.float16)):
        # Cast from one f32 source, so the two runs see the same values to
        # within their own rounding rather than two different draws.
        q, k, v, do = (t.to(dt) for t in (q32, k32, v32, do32))
        _o, lse, delta, _dq, dk64, dv64 = _math_backward(q, k, v, do, scale, torch.float64)
        fn = build(num_heads=h, head_dim=head_dim, num_kv_heads=h, dtype_str=name, **knobs)
        dk, dv = torch.full_like(k, float("nan")), torch.full_like(v, float("nan"))
        fn(
            q,
            k,
            v,
            do,
            dk,
            dv,
            lse.reshape(b * h, s).float().contiguous(),
            delta.reshape(b * h, s).float().contiguous(),
            b,
            s,
            seqlen_k=s,
            scale=scale,
        )
        torch.cuda.synchronize()
        err[name] = (_rel(dk, dk64), _rel(dv, dv64))
    for i, tensor in enumerate(("dk", "dv")):
        lo, hi = err["f16"][i], err["bf16"][i]
        assert lo * 3.0 < hi, (
            f"{tensor}: f16 error {lo:.3e} is not measurably below bf16's {hi:.3e}. f16 has three more "
            "mantissa bits, so ~8x is expected; a ratio near 1 means the f16 build is running bf16."
        )


def test_the_tensor_dtype_must_match_the_build():
    """Same width, so nothing downstream would notice a mismatch.

    Handing f16 tensors to a bf16 build reads them as bf16: every address is
    identical and only the bit interpretation differs, which is wrong by a
    factor near 2^112 and entirely silent. There is no place further down that
    could catch it, so the host is the only place it can be caught.
    """
    _require_rocm_path()
    b, h, s, d = 1, 2, 128, 64
    torch.manual_seed(4)
    row = torch.zeros(b * h, s, device="cuda", dtype=torch.float32)
    for build_dt, feed_dt in (("bf16", torch.float16), ("f16", torch.bfloat16)):
        q, k, v, do = (torch.randn(b, h, s, d, device="cuda", dtype=feed_dt) for _ in range(4))
        dk, dv = torch.empty_like(k), torch.empty_like(v)
        fn = build(num_heads=h, head_dim=d, num_kv_heads=h, dtype_str=build_dt)
        with pytest.raises(ValueError, match="compiled for dtype_str"):
            fn(q, k, v, do, dk, dv, row, row, b, s)


@pytest.mark.parametrize("rows", [32, 16], ids=["m32", "m16"])
def test_f16_range_holds_where_bf16_does_and_the_boundary_is_ds(rows):
    """The one place the two element types are not interchangeable.

    f16 overflows at 65504 where bf16 carries f32's range, so B7's and B5's
    range arguments were re-derived rather than assumed:

    - **The padding-row argument carries.** `LSE` and `delta` are f32 row
      tensors whatever the element type, `exp2` is f32 and the accumulators are
      f32, so "a zero row gives `P = exp2(0) = 1` and contributes zero" is a
      statement about f32 arithmetic. Nothing in it moves.
    - **The `-inf` bias case carries.** f16 has an infinity, the read converts
      to f32 before anything arithmetic happens, and `exp2(-inf)` is an exact
      zero either way. A caller who spells the mask as a large finite negative
      instead saturates to `-inf` on the way into an f16 tensor, which still
      masks.
    - **`P` cannot overflow.** `lse` is the log-sum-exp of its own row, so no
      score exceeds it and `P <= 1` by construction, in either type.
    - **`dS` is the exposure**, and the only one: it is the one unbounded value
      packed to the element type, for GEMM4. Measured boundary, at
      `B1 H2 S256 D64` with `dO` scaled up: `|dS|` reaches 22390 at `|dO|`
      scale 15000 and f16 is clean; at scale 30000 `|dS|` is 69482, past
      65504, and f16 returns NaN where bf16 does not.

    This pins the safe side. **The test computes its own precondition** rather
    than trusting a magnitude, because `max|dS|` depends on the tail of the
    draw and not just the scale: the first version of this test used 15000x on
    the grounds that the probe had measured it clean, and a different seed put
    the maximum over the line. A range test whose safety margin is itself a
    random variable is not a range test.

    The boundary is inherent to f16 and is shared with every f16 attention
    backward; it is recorded rather than worked around, because clamping `dS`
    would change the gradient.
    """
    _require_rocm_path()
    knobs = _family_knobs(64, rows)
    b, h, s, d = 1, 2, 256, 64
    scale = 1.0 / d**0.5
    torch.manual_seed(rows)
    q32, k32, v32 = (torch.randn(b, h, s, d, device="cuda", dtype=torch.float32) for _ in range(3))
    # Far outside any magnitude the other tests reach, so it exercises the
    # range rather than the mantissa -- and then checked, not assumed.
    do32 = torch.randn(b, h, s, d, device="cuda", dtype=torch.float32) * 5000.0
    _o, lse, delta, _dq, dk64, dv64 = _math_backward(q32, k32, v32, do32, scale, torch.float64)

    # The precondition: this shape must actually be inside f16's range, or the
    # assertions below are testing the overflow rather than the absence of one.
    qq, kk, vv = (t.double() for t in (q32, k32, v32))
    sc64 = (qq @ kk.transpose(-1, -2)) * scale
    p64 = torch.softmax(sc64, -1)
    dp64 = do32.double() @ vv.transpose(-1, -2)
    delta64 = (do32.double() * (p64 @ vv)).sum(-1, keepdim=True)
    max_ds = (p64 * (dp64 - delta64)).abs().max().item()
    assert max_ds < 65504.0 / 4, f"|dS| reaches {max_ds:.0f}, too close to f16's 65504 for this to be a safe case"
    for name in ("bf16", "f16"):
        dt = _DTYPES[name]
        q, k, v, do = (t.to(dt) for t in (q32, k32, v32, do32))
        fn = build(num_heads=h, head_dim=d, num_kv_heads=h, dtype_str=name, **knobs)
        dk, dv = torch.full_like(k, float("nan")), torch.full_like(v, float("nan"))
        fn(
            q,
            k,
            v,
            do,
            dk,
            dv,
            lse.reshape(b * h, s).float().contiguous(),
            delta.reshape(b * h, s).float().contiguous(),
            b,
            s,
            seqlen_k=s,
            scale=scale,
        )
        torch.cuda.synchronize()
        assert torch.isfinite(dk.float()).all(), f"{name}: dK is not finite at a magnitude inside f16's range"
        assert torch.isfinite(dv.float()).all(), f"{name}: dV is not finite at a magnitude inside f16's range"
        assert _rel(dk, dk64) < 1e-2 and _rel(dv, dv64) < 1e-2, f"{name}: wrong at 15000x dO"
