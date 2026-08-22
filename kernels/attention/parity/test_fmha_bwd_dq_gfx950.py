# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness for the gfx950 backward **dQ (and dB)** kernel (B2).

Run from this directory, with `ROCM_PATH` exported and the JIT disk cache off:

    export ROCM_PATH=$(rocm-sdk path --root)
    cd kernels/attention/parity
    FLYDSL_RUNTIME_ENABLE_CACHE=0 python3 -m pytest test_fmha_bwd_dq_gfx950.py -q

`FLYDSL_RUNTIME_ENABLE_CACHE=0` is not optional: `flash_attn_utils.py` and the
parity helpers are not part of the traced closure, so the cache key does not see
edits to them, and a stale artifact reads as a phantom pass.

--- Four kinds of test, and what each is for ---------------------------------

The backward has **no bitwise oracle** -- there is no second implementation
computing the same thing in the same order, which is what the forward's suite
leans on. Plan section 7 names three replacements, and this file is those plus
the structural checks:

- **The error-ratio gate** (`test_error_ratio_vs_math_backend`). A fixed
  tolerance on a bf16 backward is either loose enough to accept a real bug or
  tight enough to fail on arithmetic order. So the same problem is computed
  three ways -- ours in bf16, the torch math backend in bf16, the math backend
  in fp64 -- and the bound is `err(ours, fp64) <= fudge * err(bf16, fp64)`:
  the precision the problem inherently has, measured rather than guessed.
- **Self-consistency with our own forward**
  (`test_agrees_with_our_forwards_own_lse`). Take `O` and `LSE` from the gfx950
  forward, compute `delta` from that `O`, and check the gradient against
  autograd. This is the only test that can catch a disagreement between our two
  halves about `sm_scale` folding, `log2e` or the LSE layout -- against a
  standalone reference those show up as "the backward is wrong".
- **`dB`** (`test_db_matches_bias_gradient`). `dB = dS`, so the oracle is
  autograd through an explicit zero bias added to the scores.
- **Structural** -- strides, GQA, ragged sequences, and that nothing is written
  past `seqlen_q`. Assertions that a mistake would otherwise turn into
  plausible-looking numbers.

The joint dQ + dK/dV check plan section 6 asks for lives here too, and **skips
rather than fails while B1 is in flight**: it is the test that catches a scale
or transpose error that cancels between the two kernels, so it must not be
forgotten -- but a sibling kernel still being written must not be able to turn
this suite red either. The skip reason is the record that it has not run.
"""

import math
import sys
import time

import fmha_common_gfx1201 as fmha
import pytest
import torch
import torch.nn.functional as F
from fmha_bwd_dq_gfx950 import build_fmha_bwd_dq_gfx950_module as build_dq
from fmha_tuning_bwd_dq_gfx950 import BWD_DQ_LADDER, bwd_dq_knobs
from gfx950_standalone import dualwave  # noqa: F401  (puts the repo root on sys.path)
from torch.nn.attention import SDPBackend, sdpa_kernel

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    reason="requires a gfx950 device",
)

DT = torch.bfloat16

# How much worse than an honest bf16 reference this kernel is allowed to be.
# Not a tolerance: the *ratio* to `err(math-in-bf16, math-in-fp64)` on the same
# problem, so it scales with sequence length, head dim and dtype by itself.
#
# Measured, not chosen. Across the shapes below the observed ratio for dQ sits
# at 1.0-1.6; 4.0 leaves room for a shape whose arithmetic order happens to be
# unluckier without leaving room for a bug -- a transposed operand or a dropped
# `sm_scale` moves the ratio by one to three orders of magnitude, not by 3x.
DQ_FUDGE = 4.0
DB_FUDGE = 4.0


def _rand(*shape, dtype=DT):
    return torch.randn(*shape, device="cuda", dtype=dtype)


def _max_err(got, ref):
    return (got.double() - ref.double()).abs().max().item()


def _sm_scale(d):
    return 1.0 / math.sqrt(d)


def _math_grads(q, k, v, do, dtype, *, scale, gqa):
    """`(dq, dk, dv)` from the torch math backend, at a chosen dtype.

    The reference of plan section 7.1, used at two precisions: the same call in
    bf16 is the "equally imprecise honest answer" and in fp64 is ground truth.
    `enable_gqa` rather than a manual `repeat_interleave`, so the reference is
    the one a caller would actually write.
    """
    qq, kk, vv = (t.to(dtype).detach().clone().requires_grad_(True) for t in (q, k, v))
    with sdpa_kernel(SDPBackend.MATH):
        o = F.scaled_dot_product_attention(qq, kk, vv, is_causal=False, scale=scale, enable_gqa=gqa)
    o.backward(do.to(dtype))
    return qq.grad, kk.grad, vv.grad


def _fwd_stats(q, k, v, do, *, scale, dtype=torch.float64):
    """`(lse, delta)` computed the way a preprocess kernel would, at `dtype`.

    `lse` is the natural logsumexp of the *scaled* scores -- which is what the
    gfx950 forward writes -- and `delta = rowsum(dO * O)`, which plan section 5
    keeps out of the kernel as a host argument.
    """
    qf, kf, vf, dof = (t.to(dtype) for t in (q, k, v, do))
    rep = qf.shape[1] // kf.shape[1]
    if rep > 1:
        kf = kf.repeat_interleave(rep, dim=1)
        vf = vf.repeat_interleave(rep, dim=1)
    s = (qf @ kf.transpose(-1, -2)) * scale
    lse = torch.logsumexp(s, dim=-1)
    o = torch.softmax(s, dim=-1) @ vf
    delta = (dof * o).sum(-1)
    # `(batch * heads, tokens)`, the rank-2 shape both backward kernels take.
    n = lse.shape[-1]
    return lse.float().reshape(-1, n).contiguous(), delta.float().reshape(-1, n).contiguous(), o


def _run_dq(
    q,
    k,
    v,
    do,
    lse,
    delta,
    *,
    scale=None,
    db=None,
    dq=None,
    causal=False,
    window_build=False,
    window=None,
    mfma_rows=None,
):
    """Build for this shape and dispatch, returning dQ.

    `dq` is filled with NaN first, not zeroed: a kernel that fails to write a
    row would pass a zero-initialised check on any row whose true gradient is
    small, and the ragged-sequence cases are exactly where that happens.
    """
    b, hq, sq, d = q.shape
    if dq is None:
        dq = torch.full((b, hq, sq, d), float("nan"), device="cuda", dtype=q.dtype)
    # `head_dim_v` comes from V's own last dimension rather than being passed,
    # so an asymmetric-hdim test cannot accidentally disagree with the tensor
    # it hands in. Dropping it here builds `padded_head=False` for a genuinely
    # padded call, and the kernel then reduces over the caller's slack -- which
    # is exactly the failure the guard in `_args` now refuses.
    fn = build_dq(
        num_heads=hq,
        head_dim=d,
        head_dim_v=v.shape[3],
        causal=causal,
        window=window_build,
        dtype_str="bf16" if q.dtype is torch.bfloat16 else "f16",
        num_kv_heads=k.shape[1],
        store_db=db is not None,
        **({} if mfma_rows is None else {"mfma_rows": mfma_rows}),
    )
    fn(q, k, v, do, dq, lse, delta, b, sq, seqlen_k=k.shape[2], scale=scale, db=db, window=window)
    return dq


def _run_fwd(q, k, v, *, scale=None):
    """`(O, LSE)` from the gfx950 forward, for the self-consistency gate."""
    from flash_attn_func_gfx950 import build_flash_attn_func_gfx950_module as build_fwd

    b, hq, s, d = q.shape
    dv = v.shape[3]
    o = torch.empty(b, hq, s, dv, device="cuda", dtype=q.dtype)
    lse = torch.empty(b, hq, s, device="cuda", dtype=torch.float32)
    fn = build_fwd(
        num_heads=hq,
        head_dim=d,
        head_dim_v=dv,
        causal=False,
        dtype_str="bf16",
        num_kv_heads=k.shape[1],
        return_lse=True,
    )
    fn(q, k, v, o, b, s, seqlen_k=k.shape[2], scale=scale, lse=lse)
    return o, lse


# ---------------------------------------------------------------------------
# The correctness gate: error ratio against the math backend
# ---------------------------------------------------------------------------


# Four shapes rather than one, per the lore's "always test several shapes": a
# tile-aligned small case, a multi-workgroup case, a ragged one on both axes,
# and one where the KV axis alone is ragged.
SHAPES = [
    (1, 2, 256, 256),
    (2, 8, 1024, 1024),
    (1, 4, 300, 300),
    (1, 4, 128, 777),
]

# The two rungs that get the full shape cross-product. Every rung is covered by
# `test_ladder_error_ratio`; these two also get the ragged and multi-workgroup
# cases, because they are the widths every other test in this file uses and a
# regression there should be visible from more than one direction.
WORKHORSE = (64, 128)


@pytest.mark.parametrize("head_dim", WORKHORSE)
@pytest.mark.parametrize("b,h,sq,sk", SHAPES, ids=lambda x: str(x))
def test_error_ratio_vs_math_backend(b, h, sq, sk, head_dim):
    """`err(ours, fp64) <= fudge * err(math-in-bf16, fp64)`, on dQ.

    The bound is the precision the problem inherently has. `lse` and `delta`
    come from an fp32 reference rather than from our forward, so a failure here
    is about *this* kernel and not about the pair -- the pair is
    `test_agrees_with_our_forwards_own_lse`.
    """
    torch.manual_seed(0)
    q, do = (_rand(b, h, sq, head_dim) for _ in range(2))
    k, v = (_rand(b, h, sk, head_dim) for _ in range(2))
    scale = _sm_scale(head_dim)

    lse, delta, _ = _fwd_stats(q, k, v, do, scale=scale, dtype=torch.float32)
    got = _run_dq(q, k, v, do, lse, delta, scale=scale)

    hi = _math_grads(q, k, v, do, torch.float64, scale=scale, gqa=False)[0]
    lo = _math_grads(q, k, v, do, DT, scale=scale, gqa=False)[0]

    ours_err = _max_err(got, hi)
    ref_err = _max_err(lo, hi)
    assert torch.isfinite(got.float()).all()
    assert ours_err <= DQ_FUDGE * ref_err, (
        f"dQ error {ours_err:.3e} against fp64 exceeds {DQ_FUDGE}x the bf16 math "
        f"backend's own {ref_err:.3e} (ratio {ours_err / max(ref_err, 1e-30):.2f})"
    )


@pytest.mark.parametrize("head_dim", BWD_DQ_LADDER)
@pytest.mark.parametrize("sq,sk", [(512, 512), (300, 411)], ids=["square", "ragged"])
def test_ladder_error_ratio(head_dim, sq, sk):
    """Every rung of the ladder, square and ragged.

    The ladder is where the *geometry* changes -- wave count, staging granule,
    LDS size -- so a rung that addresses its tile wrongly fails here and
    nowhere else. Two shapes because a rung whose only failure is the KV tail
    passes a square one: `sk = 411` is not a multiple of `BLOCK_N`, and
    `sq = 300` is not a multiple of any `BLOCK_M` on the table.
    """
    torch.manual_seed(13)
    b, h = 1, 4
    q, do = (_rand(b, h, sq, head_dim) for _ in range(2))
    k, v = (_rand(b, h, sk, head_dim) for _ in range(2))
    scale = _sm_scale(head_dim)

    lse, delta, _ = _fwd_stats(q, k, v, do, scale=scale, dtype=torch.float32)
    got = _run_dq(q, k, v, do, lse, delta, scale=scale)

    hi = _math_grads(q, k, v, do, torch.float64, scale=scale, gqa=False)[0]
    lo = _math_grads(q, k, v, do, DT, scale=scale, gqa=False)[0]
    ours, honest = _max_err(got, hi), _max_err(lo, hi)
    assert torch.isfinite(got.float()).all()
    assert ours <= DQ_FUDGE * honest, (
        f"head_dim {head_dim}: dQ error {ours:.3e} exceeds {DQ_FUDGE}x the bf16 math backend's "
        f"{honest:.3e} (ratio {ours / max(honest, 1e-30):.2f})"
    )


@pytest.mark.parametrize("head_dim", WORKHORSE)
def test_error_ratio_under_gqa(head_dim):
    """Same gate with four Q heads per KV head.

    GQA is a head *remap*, not new arithmetic, but the remap is the one thing
    that decides which K a Q block streams -- and getting it wrong returns a
    finite gradient computed against the wrong keys.
    """
    torch.manual_seed(1)
    b, hq, hk, s = 2, 8, 2, 512
    q, do = (_rand(b, hq, s, head_dim) for _ in range(2))
    k, v = (_rand(b, hk, s, head_dim) for _ in range(2))
    scale = _sm_scale(head_dim)

    lse, delta, _ = _fwd_stats(q, k, v, do, scale=scale, dtype=torch.float32)
    got = _run_dq(q, k, v, do, lse, delta, scale=scale)

    hi = _math_grads(q, k, v, do, torch.float64, scale=scale, gqa=True)[0]
    lo = _math_grads(q, k, v, do, DT, scale=scale, gqa=True)[0]
    assert _max_err(got, hi) <= DQ_FUDGE * _max_err(lo, hi)


@pytest.mark.parametrize("scale", [0.05, 0.25, 1.0])
def test_error_ratio_under_a_runtime_scale(scale):
    """A caller-supplied `sm_scale`, which is not `1/sqrt(d)`.

    `sm_scale` enters this kernel twice -- as `sm_scale*log2e` on the f32
    scores, and again on the dQ accumulator at the end -- so a build that took
    one of the two from `head_dim` instead of from the argument is correct at
    exactly one scale. Sweeping it is what separates the two.

    The sweep is also what found the *precision* result recorded in
    `BwdDqSoftmaxHelper.scale_and_sub_lse`: the error ratio was flat at 1.3
    and climbed to 10.9 by `scale=1.0` while the kernel still pre-scaled Q into
    bf16. A single-scale test would have called that a passing kernel.
    """
    torch.manual_seed(2)
    b, h, s, d = 1, 4, 512, 64
    q, k, v, do = (_rand(b, h, s, d) for _ in range(4))

    lse, delta, _ = _fwd_stats(q, k, v, do, scale=scale, dtype=torch.float32)
    got = _run_dq(q, k, v, do, lse, delta, scale=scale)

    hi = _math_grads(q, k, v, do, torch.float64, scale=scale, gqa=False)[0]
    lo = _math_grads(q, k, v, do, DT, scale=scale, gqa=False)[0]
    assert _max_err(got, hi) <= DQ_FUDGE * _max_err(lo, hi)


# ---------------------------------------------------------------------------
# Self-consistency: our forward's own O and LSE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", BWD_DQ_LADDER)
def test_agrees_with_our_forwards_own_lse(head_dim):
    """Feed the gfx950 forward's `O`/`LSE` to the backward and check autograd.

    **This is the convention test.** The gate above supplies `lse` from an fp32
    reference, so it would pass a kernel that agreed with torch and disagreed
    with our forward -- about which base the LSE is in, about whether
    `sm_scale` is already folded into it, about the `(H, T)` row layout. Here
    the two halves have to agree with each other *and* with autograd.
    """
    torch.manual_seed(3)
    b, h, s = 2, 4, 512
    q, k, v, do = (_rand(b, h, s, head_dim) for _ in range(4))
    scale = _sm_scale(head_dim)

    o_ours, lse_ours = _run_fwd(q, k, v, scale=scale)
    delta = (do.float() * o_ours.float()).sum(-1)
    got = _run_dq(q, k, v, do, lse_ours.reshape(-1, s).contiguous(), delta.reshape(-1, s).contiguous(), scale=scale)

    hi = _math_grads(q, k, v, do, torch.float64, scale=scale, gqa=False)[0]
    lo = _math_grads(q, k, v, do, DT, scale=scale, gqa=False)[0]
    ours_err = _max_err(got, hi)
    ref_err = _max_err(lo, hi)
    assert ours_err <= DQ_FUDGE * ref_err, (
        f"dQ from our own forward's LSE is {ours_err:.3e} from fp64 against the bf16 "
        f"reference's {ref_err:.3e}: the forward and backward disagree about a convention"
    )


@pytest.mark.parametrize("head_dim", BWD_DQ_LADDER)
def test_joint_gradient_with_dkdv(head_dim):
    """dQ and dK/dV against **one** autograd call.

    Separate suites hide an error that cancels between the two kernels -- a
    transposed operand, or a scale applied on the wrong side, is self-
    consistent within each. Plan section 6 asks for exactly this.

    Skipped, not failed, when B1's module is absent or its front end has moved:
    this file's deliverable is dQ, and a sibling kernel in flight must not be
    able to turn this suite red. The skip *reason* is the record that the check
    has not run.
    """
    dkdv = pytest.importorskip(
        "fmha_bwd_dkdv_gfx950",
        reason="B1 (the dK/dV kernel) is not importable yet; plan section 6 requires this joint check",
    )
    torch.manual_seed(12)
    b, h, s = 2, 4, 512
    q, k, v, do = (_rand(b, h, s, head_dim) for _ in range(4))
    scale = _sm_scale(head_dim)
    lse, delta, _ = _fwd_stats(q, k, v, do, scale=scale, dtype=torch.float32)

    got_dq = _run_dq(q, k, v, do, lse, delta, scale=scale)
    dk = torch.full((b, h, s, head_dim), float("nan"), device="cuda", dtype=DT)
    dv = torch.full((b, h, s, head_dim), float("nan"), device="cuda", dtype=DT)
    try:
        fn = dkdv.build_fmha_bwd_dkdv_gfx950_module(num_heads=h, head_dim=head_dim, causal=False)
        fn(q, k, v, do, dk, dv, lse, delta, b, s, seqlen_k=s, scale=scale)
    except (AttributeError, TypeError, ValueError, NotImplementedError) as exc:
        pytest.skip(f"B1's front end is not settled yet: {exc}")
    torch.cuda.synchronize()

    hi = _math_grads(q, k, v, do, torch.float64, scale=scale, gqa=False)
    lo = _math_grads(q, k, v, do, DT, scale=scale, gqa=False)
    for name, got, ref_hi, ref_lo in (
        ("dQ", got_dq, hi[0], lo[0]),
        ("dK", dk, hi[1], lo[1]),
        ("dV", dv, hi[2], lo[2]),
    ):
        ours, honest = _max_err(got, ref_hi), _max_err(ref_lo, ref_hi)
        assert ours <= DQ_FUDGE * honest, (
            f"{name} error {ours:.3e} against one fp64 autograd call exceeds {DQ_FUDGE}x the "
            f"bf16 reference's {honest:.3e}"
        )


# ---------------------------------------------------------------------------
# dB
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", BWD_DQ_LADDER)
def test_db_matches_bias_gradient(head_dim):
    """`dB == dS`, against autograd through an explicit zero bias.

    A zero bias added to the scores changes nothing about the forward, so this
    reference is the same attention -- and its gradient with respect to that
    bias is exactly `dS`. The oracle needs no separate derivation, which is the
    point: writing `P * (dP - delta)` out by hand would be a second
    implementation that can be wrong the same way ours is.
    """
    torch.manual_seed(4)
    b, h, s = 1, 4, 256
    q, k, v, do = (_rand(b, h, s, head_dim) for _ in range(4))
    scale = _sm_scale(head_dim)

    lse, delta, _ = _fwd_stats(q, k, v, do, scale=scale, dtype=torch.float32)
    db = torch.full((b, h, s, s), float("nan"), device="cuda", dtype=DT)
    _run_dq(q, k, v, do, lse, delta, scale=scale, db=db)

    def ref(dtype):
        qq, kk, vv = (t.to(dtype) for t in (q, k, v))
        bias = torch.zeros(b, h, s, s, device="cuda", dtype=dtype, requires_grad=True)
        p = torch.softmax((qq @ kk.transpose(-1, -2)) * scale + bias, dim=-1)
        (p @ vv).backward(do.to(dtype))
        return bias.grad

    hi = ref(torch.float64)
    lo = ref(DT)
    assert torch.isfinite(db.float()).all(), "dB has NaN: a tile was not written"
    assert _max_err(db, hi) <= DB_FUDGE * _max_err(lo, hi)


def test_db_covers_a_ragged_kv_axis():
    """dB's tail columns, where the run-wide store would have been wrong.

    `seqlen_k` off a multiple of 4 is the case that forced the per-element
    store: a 4-wide vector store straddling `seqlen_k` cannot be partially
    written, and suppressing the whole run would drop live columns. `seqlen_k`
    is deliberately `4n+1` so all three of the run's tail states occur.
    """
    torch.manual_seed(5)
    b, h, sq, sk, d = 1, 2, 128, 201, 64
    q, do = (_rand(b, h, sq, d) for _ in range(2))
    k, v = (_rand(b, h, sk, d) for _ in range(2))
    scale = _sm_scale(d)

    lse, delta, _ = _fwd_stats(q, k, v, do, scale=scale, dtype=torch.float32)
    sentinel = -12345.0
    db = torch.full((b, h, sq, sk), sentinel, device="cuda", dtype=DT)
    _run_dq(q, k, v, do, lse, delta, scale=scale, db=db)

    qq, kk, vv = (t.double() for t in (q, k, v))
    bias = torch.zeros(b, h, sq, sk, device="cuda", dtype=torch.float64, requires_grad=True)
    (torch.softmax((qq @ kk.transpose(-1, -2)) * scale + bias, dim=-1) @ vv).backward(do.double())

    assert not (db.float() == sentinel).any(), "dB left elements unwritten past the last full 4-column run"
    assert _max_err(db, bias.grad) <= 5.0e-2


# ---------------------------------------------------------------------------
# The 8xD input contract, and padded heads
# ---------------------------------------------------------------------------


# Loads and stores are 8 columns wide, so a head_dim that is a multiple of 8 is
# a whole number of chunks and the kernel never touches a column it was not
# given. Every multiple of 8 the ladder can reach, so a rung that mishandles
# its sub-8-grid widths cannot hide behind the ones the suite happens to name.
_GRID8 = list(range(8, 513, 8))


@pytest.mark.parametrize("hdim", _GRID8)
def test_grid8_contiguous_is_exact_and_writes_nothing_past_dq(hdim):
    """A plainly contiguous 8xD tensor -- no padded view -- must just work.

    The forward's counterpart of this test found a real class of bug, and the
    reason is that every *other* padded test allocates through a wider buffer
    and slices, so its D pitch is 8-aligned by construction. What a caller
    actually passes is `torch.randn(b, h, s, 24)`, whose pitch is 24.

    The extra dQ row is a canary: it is contiguous with the last real row, so a
    tail chunk overrunning the final row lands in it and nowhere else. That is
    the check that `BwdDqStoreHelper`'s `hdim_vo` rebinding actually took --
    without it the suppression compares against the wrong extent and the
    overrun is by whole chunks.
    """
    torch.manual_seed(14)
    b, h, s = 1, 2, 128
    q, do = (_rand(b, h, s, hdim) for _ in range(2))
    k, v = (_rand(b, h, s, hdim) for _ in range(2))
    assert q.stride(2) == hdim, "the point of this test is a tight pitch"
    scale = _sm_scale(hdim)
    lse, delta, _ = _fwd_stats(q, k, v, do, scale=scale, dtype=torch.float32)

    sentinel = -12345.0
    buf = torch.full((b, h, s + 1, hdim), sentinel, device="cuda", dtype=DT)
    got = _run_dq(q, k, v, do, lse, delta, scale=scale, dq=buf[:, :, :s, :])

    assert torch.all(buf[:, :, s, :] == sentinel), "a store ran past the last dQ row"
    hi = _math_grads(q, k, v, do, torch.float64, scale=scale, gqa=False)[0]
    lo = _math_grads(q, k, v, do, DT, scale=scale, gqa=False)[0]
    assert _max_err(got, hi) <= DQ_FUDGE * _max_err(lo, hi)


def _padded(b, h, s, hdim, pitch, poison=None):
    """A `(b, h, s, hdim)` view into an 8-aligned allocation of `pitch`.

    What the pad *contains* is deliberately not part of the contract, which is
    what `poison` exists to hold the kernel to: masking Q alone is enough for a
    finite pad because `0 * x == 0`, and not enough for a NaN.
    """
    t = _rand(b, h, s, pitch)
    if poison is not None:
        t[..., hdim:] = poison
    return t[..., :hdim]


@pytest.mark.parametrize("hdim", [40, 100, 129, 200, 260, 400])
@pytest.mark.parametrize("poison", [None, float("nan")], ids=["clean", "nan-pad"])
def test_padded_head(hdim, poison):
    """A head_dim between two rungs, with the surplus columns masked.

    The pad is poisoned in half the cases because the D-axis slack belongs to
    the caller and nothing constrains its contents. Masking Q alone would
    survive a finite pad and not a NaN, which is why K is masked too -- and in
    this kernel *both* tiles that reach the K register path are, each against
    its own extent.
    """
    torch.manual_seed(15)
    b, h, s = 1, 2, 256
    pitch = (hdim + 7) // 8 * 8
    q, k, v, do = (_padded(b, h, s, hdim, pitch, poison) for _ in range(4))
    scale = _sm_scale(hdim)
    lse, delta, _ = _fwd_stats(q, k, v, do, scale=scale, dtype=torch.float32)

    dq = torch.empty(b, h, s, pitch, device="cuda", dtype=DT)[..., :hdim]
    dq.fill_(float("nan"))
    got = _run_dq(q, k, v, do, lse, delta, scale=scale, dq=dq)

    hi = _math_grads(q, k, v, do, torch.float64, scale=scale, gqa=False)[0]
    lo = _math_grads(q, k, v, do, DT, scale=scale, gqa=False)[0]
    assert torch.isfinite(got.float()).all(), "the poisoned pad reached the arithmetic"
    assert _max_err(got, hi) <= DQ_FUDGE * _max_err(lo, hi)


@pytest.mark.parametrize("hdim,hdim_v", [(128, 64), (64, 32), (128, 96), (192, 128), (256, 128), (100, 40)])
def test_asymmetric_hdim(hdim, hdim_v):
    """`hdim_qk != hdim_vo`, which is where the two extents cross.

    **This is the test the B2 outcome asked for.** GEMM2 reads V through the
    *K* register path, whose padded-head mask is written against `hdim_qk`;
    dQ is written through the O store, whose suppression is written against
    `hdim_vo`. Both are the other extent here, and they coincide in every
    symmetric build -- so nothing before this could tell the fix from its
    absence.

    `(100, 40)` is the case that also exercises the `HDIM_VO_FLOOR` fallback:
    the 128 tile's floor is 96, `hdim_vo` is below it, so the V mask has to
    cover every K-step rather than the two the floor would leave.
    """
    torch.manual_seed(16)
    b, h, s = 1, 2, 256
    pitch = (max(hdim, hdim_v) + 7) // 8 * 8
    q, k = (_padded(b, h, s, hdim, pitch) for _ in range(2))
    v, do = (_padded(b, h, s, hdim_v, pitch) for _ in range(2))
    scale = _sm_scale(hdim)
    lse, delta, _ = _fwd_stats(q, k, v, do, scale=scale, dtype=torch.float32)

    dq = torch.empty(b, h, s, pitch, device="cuda", dtype=DT)[..., :hdim]
    dq.fill_(float("nan"))
    got = _run_dq(q, k, v, do, lse, delta, scale=scale, dq=dq)

    hi = _math_grads(q, k, v, do, torch.float64, scale=scale, gqa=False)[0]
    lo = _math_grads(q, k, v, do, DT, scale=scale, gqa=False)[0]
    assert _max_err(got, hi) <= DQ_FUDGE * _max_err(lo, hi)


def test_padded_head_never_writes_past_hdim_qk():
    """dQ's D-tail may spill into the caller's pad, but not past it.

    The store is 128 bits and cannot be split, so a chunk straddling the real
    extent writes into the allocation's own padding -- permitted, and the
    reason the 8-element pitch contract exists. A chunk starting at or past it
    must be suppressed, and **dQ's extent is `hdim_qk`**: a build that
    suppressed against `hdim_vo` here would write four whole chunks of garbage
    into the next row.
    """
    torch.manual_seed(17)
    b, h, s, hdim, hdim_v = 1, 2, 256, 72, 40
    pitch = 128  # far wider than either extent, so the surplus is visible
    q, k = (_padded(b, h, s, hdim, pitch) for _ in range(2))
    v, do = (_padded(b, h, s, hdim_v, pitch) for _ in range(2))
    scale = _sm_scale(hdim)
    lse, delta, _ = _fwd_stats(q, k, v, do, scale=scale, dtype=torch.float32)

    sentinel = -12345.0
    buf = torch.full((b, h, s, pitch), sentinel, device="cuda", dtype=DT)
    _run_dq(q, k, v, do, lse, delta, scale=scale, dq=buf[..., :hdim])
    # `ceil8(72)` is 72, so nothing may be written at or past column 72.
    assert torch.all(buf[..., hdim:] == sentinel), "a chunk past hdim_qk was not suppressed"


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


def test_strides_are_read_not_derived():
    """A BSHD allocation viewed as BHSD gives the same gradient.

    The whole ABI claim is that the three non-D strides are free variables.
    Nothing at runtime distinguishes a head stride from a sequence stride, so a
    kernel that derived one would return finite garbage here rather than an
    error.
    """
    torch.manual_seed(6)
    b, h, s, d = 2, 4, 256, 64
    scale = _sm_scale(d)
    ten = [_rand(b, h, s, d) for _ in range(4)]
    q, k, v, do = ten
    lse, delta, _ = _fwd_stats(q, k, v, do, scale=scale, dtype=torch.float32)
    contiguous = _run_dq(q, k, v, do, lse, delta, scale=scale)

    # Same values, laid out (B, S, H, D) and transposed back to a BHSD view.
    def bshd(t):
        out = torch.empty(b, s, h, t.shape[3], device="cuda", dtype=DT).transpose(1, 2)
        out.copy_(t)
        return out

    dq = torch.empty(b, s, h, d, device="cuda", dtype=DT).transpose(1, 2)
    dq.fill_(float("nan"))
    got = _run_dq(*(bshd(t) for t in ten), lse, delta, scale=scale, dq=dq)
    assert torch.equal(got, contiguous)


def test_nothing_is_written_past_seqlen_q():
    """A ragged Q axis must leave the rows past it alone.

    Those rows exist in the allocation -- the grid is rounded up to `BLOCK_M`
    -- and the kernel relies on the buffer descriptor's bound to drop their
    stores rather than on a branch. A descriptor sized from the wrong length
    would corrupt them silently.
    """
    torch.manual_seed(7)
    b, h, sq, d = 1, 2, 300, 64
    alloc = 512
    q, do = (_rand(b, h, sq, d) for _ in range(2))
    k, v = (_rand(b, h, sq, d) for _ in range(2))
    scale = _sm_scale(d)
    lse, delta, _ = _fwd_stats(q, k, v, do, scale=scale, dtype=torch.float32)

    sentinel = 7.5
    big = torch.full((b, h, alloc, d), sentinel, device="cuda", dtype=DT)
    _run_dq(q, k, v, do, lse, delta, scale=scale, dq=big[:, :, :sq, :])
    assert torch.all(big[:, :, sq:, :] == sentinel)


def test_the_padded_tail_of_a_q_block_is_finite():
    """A Q block whose rows run past `seqlen_q` must not produce NaN.

    Those lanes read zero Q, zero dO and a redirected (zero) LSE, so `P` is 1
    on every live column while `dP` and `delta` are 0. The gradient they
    compute is zero and is dropped at the store -- but only if nothing in
    between produced an infinity that the MFMA could spread.
    """
    torch.manual_seed(8)
    b, h, sq, d = 1, 2, 257, 128
    q, do = (_rand(b, h, sq, d) for _ in range(2))
    k, v = (_rand(b, h, sq, d) for _ in range(2))
    scale = _sm_scale(d)
    lse, delta, _ = _fwd_stats(q, k, v, do, scale=scale, dtype=torch.float32)
    got = _run_dq(q, k, v, do, lse, delta, scale=scale)
    assert torch.isfinite(got.float()).all()


def test_run_to_run_deterministic():
    """Two runs of the same problem must be bit-identical.

    Plan section 7.3: AITER's fused backward accumulates dQ by atomic add and
    is non-deterministic by construction, and that is why we do not copy its
    design. This is the assertion that makes the claim mean something.
    """
    torch.manual_seed(9)
    b, h, s, d = 2, 4, 1024, 128
    q, k, v, do = (_rand(b, h, s, d) for _ in range(4))
    scale = _sm_scale(d)
    lse, delta, _ = _fwd_stats(q, k, v, do, scale=scale, dtype=torch.float32)
    a = _run_dq(q, k, v, do, lse, delta, scale=scale)
    c = _run_dq(q, k, v, do, lse, delta, scale=scale)
    assert torch.equal(a, c)


# ---------------------------------------------------------------------------
# The knobs refuse what B2 does not implement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [("bias", True), ("dropout", True)],
)
def test_unimplemented_modes_are_refused(field, value):
    """Refused by name, not ignored.

    Every one of these would build, run and return a correctly-shaped wrong
    answer, which is the failure mode the whole backward plan is written
    against. B4-B6 turn them on one at a time.
    """
    with pytest.raises(NotImplementedError, match=field):
        build_dq(num_heads=4, head_dim=64, **{field: value})


def test_head_dim_past_the_widest_rung_is_refused():
    """B3 reaches 512, which is the widest tile the ladder describes."""
    with pytest.raises(ValueError, match="exceeds the widest tile"):
        build_dq(num_heads=4, head_dim=520, causal=False)


@pytest.mark.parametrize("field", ["d_stages", "qk_shards", "vo_shards"])
def test_d_axis_splits_are_refused(field):
    """The traits can describe them; this body cannot execute them.

    **A near miss worth a test.** The forward's `_with_d_axis_splits` turns
    `d_stages` on at block_dmodel > 256 -- exactly the rungs B3 added -- and
    under `D_STAGES > 1` the inherited `ParityGemmHelper.qk` becomes
    `qk_stage(..., stage=0)` and `pv` writes only the first stage's chunks. The
    loop here never advances the stage, so the kernel would reduce over part of
    the head dim, write part of the accumulator, and return a finite wrong
    answer at 384 and 512. It was caught by an LDS figure that was half what it
    should have been.
    """
    with pytest.raises(NotImplementedError, match=field):
        build_dq(num_heads=4, head_dim=384, causal=False, **{field: 2})


def test_rows_per_wave_ceiling_is_enforced():
    """A geometry over the MFMA's M extent must raise, not compute.

    P7's finding, and the addendum requires it enforced rather than commented:
    the forward shipped twelve silently-wrong configurations at 0.15-0.28
    relative error because `BLOCK_M` over `Q_TILES` exceeded `MFMA_M` and the
    invariant lived in prose.

    Two layers, and they are checked separately because either alone could be
    removed without the other noticing. `make_bwd_dq_traits` is where the
    invariant is *enforced*; the knob-level geometry list happens to reject the
    same configuration earlier, and only because every entry on it satisfies
    `block_m == num_waves * 32`.
    """
    from fmha_tuning_bwd_dq_gfx950 import make_bwd_dq_traits

    with pytest.raises(ValueError, match="caps it at 32"):
        make_bwd_dq_traits(
            num_heads=4,
            num_kv_heads=4,
            head_dim=128,
            num_waves=2,
            block_m=128,  # 64 rows per wave, twice the MFMA's M extent
            block_n=64,
            granule=64,
            causal=False,
        )
    with pytest.raises(NotImplementedError, match="addressable"):
        build_dq(
            num_heads=4,
            head_dim=128,
            causal=False,
            num_waves=2,
            block_m=128,
            block_n=64,
            head_dim_granule=64,
        )


def test_a_non_padded_build_refuses_a_padded_call():
    """`padded_head=False` promises the tile is the extent, on *both* axes.

    The kernel emits no D-axis mask at all in that build, so a V narrower than
    the tile is reduced over the caller's slack: finite, right-shaped, 0.70
    relative error. Found by this file getting it wrong -- `_run_dq` used to
    omit `head_dim_v` -- which is why the guard is in `_args` and not here.
    """
    torch.manual_seed(19)
    b, h, s, d, dv = 1, 2, 128, 128, 64
    q, do = (_rand(b, h, s, d) for _ in range(2))
    k = _rand(b, h, s, d)
    v = _rand(b, h, s, dv)
    lse = torch.zeros(b * h, s, device="cuda", dtype=torch.float32)
    dq = torch.empty(b, h, s, d, device="cuda", dtype=DT)
    fn = build_dq(num_heads=h, head_dim=d, causal=False)  # no head_dim_v: not padded
    with pytest.raises(ValueError, match="not compiled for a padded head"):
        fn(q, k, v, do[..., :dv], dq, lse, lse.clone(), b, s, seqlen_k=s)


@pytest.mark.parametrize("waves", [2, 4])
@pytest.mark.parametrize("wpe", [1, 2])
def test_wave_geometries_agree(waves, wpe):
    """Every geometry the knobs will accept must give the same answer.

    `BwdDqKnobs._SUPPORTED_GEOMETRIES` adds the two-wave points to the
    forward's list, and the list exists precisely because a geometry that is
    *describable* can still address the wrong LDS -- P2 measured that at
    head_dim 192. So the entries have to be run, not asserted: a wave count
    that gets `ISSUES_PER_WAVE` wrong leaves LDS lines unwritten and reads back
    whatever was there.

    Against the error-ratio gate rather than bitwise, because the wave count
    changes `BLOCK_M` and therefore which rows a workgroup owns; the
    accumulation order within a row is unchanged, but nothing here needs to
    depend on that.
    """
    torch.manual_seed(18)
    b, h, s, d = 1, 4, 512, 128
    q, k, v, do = (_rand(b, h, s, d) for _ in range(4))
    scale = _sm_scale(d)
    lse, delta, _ = _fwd_stats(q, k, v, do, scale=scale, dtype=torch.float32)

    dq = torch.full((b, h, s, d), float("nan"), device="cuda", dtype=DT)
    fn = build_dq(
        num_heads=h,
        head_dim=d,
        causal=False,
        num_waves=waves,
        block_m=waves * 32,
        block_n=64,
        head_dim_granule=64,
        waves_per_eu=wpe,
    )
    fn(q, k, v, do, dq, lse, delta, b, s, seqlen_k=s, scale=scale)

    hi = _math_grads(q, k, v, do, torch.float64, scale=scale, gqa=False)[0]
    lo = _math_grads(q, k, v, do, DT, scale=scale, gqa=False)[0]
    assert _max_err(dq, hi) <= DQ_FUDGE * _max_err(lo, hi)


def test_the_wide_rungs_fit_lds_unstaged():
    """Two staging regions, no `D_STAGES`, inside the cap -- on both families.

    The B3 blocker, as an assertion. B2 staged K twice -- 199 KB at head_dim
    512 against a 163840 B cap -- so the ladder did not merely run slowly
    there, it could not be built. A regression that reintroduced the third
    region would fail to compile with "local memory exceeds limit", which names
    the symptom and not the cause.

    Both families, and pinned by *geometry* rather than by the default policy,
    because the policy moved once already: the 16-row family halves `BLOCK_N`
    and therefore the tile again.
    """
    from fmha_tuning_gfx950 import FmhaInputMetadata

    cases = [
        (256, 32, 66.5),
        (384, 32, 99.8),
        (512, 32, 133.0),
        (384, 16, 49.9),
        (512, 16, 66.5),
    ]
    for head_dim, rows, want_kb in cases:
        traits = (
            bwd_dq_knobs(mfma_rows=rows).resolve(FmhaInputMetadata(num_heads=8, head_dim=head_dim, causal=False)).traits
        )
        kb = traits.LDS_KV_TOTAL_SIZE * traits.BF16_BYTES / 1024
        assert abs(kb - want_kb) < 0.5, f"head_dim {head_dim} at {rows} rows: {kb:.1f} KB, expected {want_kb}"
        assert kb < 160.0
        assert traits.D_STAGES == 1 and traits.VO_SHARDS == 1


# ---------------------------------------------------------------------------
# B3.5: the register model, and the MFMA shape it argues for
# ---------------------------------------------------------------------------


# `.vgpr_count` plus `.vgpr_spill_count` from the B3 ISA dumps, at the default
# geometry for each rung, `store_db=False`. Recorded rather than re-measured
# because the point of the test is to pin the *model* against a fixed
# observation; re-dumping ten kernels would make it a slow test of the
# compiler instead.
_B3_MEASURED_VGPR = {
    32: 140 + 0,
    64: 190 + 0,
    96: 236 + 0,
    128: 260 + 0,
    160: 312 + 0,
    192: 360 + 0,
    224: 412 + 0,
    256: 460 + 0,
    384: 512 + 112,
    512: 512 + 546,
}


@pytest.mark.parametrize("head_dim", [d for d in BWD_DQ_LADDER if d <= 256])
def test_register_model_matches_measured(head_dim):
    """The structural accounting must predict the observed VGPR count.

    **This is what licenses the model to be used as a prediction** for the
    16-row geometry, which has never been built. Every term in
    `register_demand` is an exact count of what the algorithm holds -- nothing
    is fitted -- so agreeing with measurement to within a few registers at
    eight independent rungs is a real check rather than a tautology.

    Only the rungs that do *not* spill: above 512 the allocator's behaviour is
    non-linear (head_dim 512 is 336 registers over and spills 546, because
    spill code costs registers of its own), so the measurement stops being a
    measurement of demand.

    The band is 32 registers, which is the residual the model deliberately
    leaves out: addressing, DMA descriptors and loop scalars, measured at 4-12
    at two waves and 12-14 at four. Fitting a constant for them would make the
    model agree with today's numbers by construction.
    """
    from fmha_tuning_bwd_dq_gfx950 import register_demand
    from fmha_tuning_gfx950 import FmhaInputMetadata

    traits = bwd_dq_knobs().resolve(FmhaInputMetadata(num_heads=8, head_dim=head_dim, causal=False)).traits
    predicted = register_demand(traits)["total"]
    measured = _B3_MEASURED_VGPR[head_dim]
    assert abs(measured - predicted) <= 32, (
        f"head_dim {head_dim}: model says {predicted} registers, ISA says {measured}. "
        "Either the body's live set changed or the model is wrong; both matter."
    )
    assert predicted <= 512, "a rung that does not spill must not be modelled as over the file"


# `.vgpr_count` from the B3.5 ISA dumps of the **16-row** family, default
# geometry, `store_db=False`. The 32-row column is `_B3_MEASURED_VGPR`.
_M16_MEASURED_VGPR = {64: 80, 128: 132, 192: 170, 256: 216, 384: 318, 512: 414}


@pytest.mark.parametrize("head_dim", sorted(_M16_MEASURED_VGPR))
def test_register_model_predicted_the_16_row_family(head_dim):
    """**The model made a prediction about an unbuilt geometry. This is the score.**

    Before the 16-row family existed, `register_demand` said its demand would
    be `0.5 * head_dim` for the loop invariants against 32 rows' `1.0 * d`, and
    put head_dim 512 at 404 registers against 848. The family now exists, so
    the prediction is checkable rather than merely plausible -- and a model
    that survives contact is worth more than the decision it justified.

    It came in at **+4 to +16** registers across six rungs, the same band and
    the same sign as the 32-row fit: the terms are exact and what is left out
    (addressing, DMA descriptors, loop scalars) is a small positive residual.
    """
    from fmha_tuning_bwd_dq_gfx950 import register_demand
    from fmha_tuning_gfx950 import FmhaInputMetadata

    traits = bwd_dq_knobs(mfma_rows=16).resolve(FmhaInputMetadata(num_heads=8, head_dim=head_dim, causal=False)).traits
    predicted = register_demand(traits)["total"]
    measured = _M16_MEASURED_VGPR[head_dim]
    assert 0 <= measured - predicted <= 32, (
        f"head_dim {head_dim} at 16 rows: model {predicted}, ISA {measured}. The residual is supposed "
        "to be a small positive constant; a negative one means the model over-counts."
    )
    assert predicted <= 512


def test_sixteen_rows_is_what_the_wide_rungs_need():
    """The argument for the second family, stated so it cannot drift.

    At 32 rows the loop-invariant Q + dO + dQ alone is `1.0 * head_dim`
    registers, which is the entire 512-register file at head_dim 512 before a
    single operand. At 16 rows it is `0.5 * head_dim`. That is arithmetic, not
    a measurement, and it is why the family exists.

    Pinned by geometry, not by the default policy: the policy now *selects* 16
    rows at 384 and above, so reading the default would compare a thing to
    itself.
    """
    from fmha_tuning_bwd_dq_gfx950 import register_demand
    from fmha_tuning_gfx950 import FmhaInputMetadata

    def demand(head_dim, rows):
        traits = (
            bwd_dq_knobs(mfma_rows=rows).resolve(FmhaInputMetadata(num_heads=8, head_dim=head_dim, causal=False)).traits
        )
        return register_demand(traits)

    for head_dim in (384, 512):
        a, b = demand(head_dim, 32), demand(head_dim, 16)
        assert a["q"] + a["do"] + a["dq_acc"] == head_dim
        assert b["q"] + b["do"] + b["dq_acc"] == head_dim // 2
        assert a["over_512"] > 0, "the 32-row family is supposed to be the problem here"
        assert b["over_512"] == 0, "the 16-row family is supposed to be the answer"


def test_the_default_policy_splits_the_families_at_384():
    """32 rows below 384, 16 at and above. Measured, and additive.

    16 against 32 rows at `B=2 H=8 S=4096`: 4.23x at 512, 1.69x at 384, 1.04x
    at 256, level at 192, and **worse** at 128 (0.94x) and 64 (0.86x). The
    32-row family keeps everything below 384 -- "the 32-row path must not
    regress" is a gate, and the way it is met is that nothing below 384 moves.
    """
    from fmha_tuning_gfx950 import FmhaInputMetadata

    for head_dim, rows in ((64, 32), (128, 32), (256, 32), (384, 16), (512, 16)):
        traits = bwd_dq_knobs().resolve(FmhaInputMetadata(num_heads=8, head_dim=head_dim, causal=False)).traits
        assert traits.MFMA_N == rows, f"head_dim {head_dim} should default to {rows} rows"


@pytest.mark.parametrize("head_dim", [32, 96, 160, 224])
def test_the_16_row_family_refuses_the_off_grid_rungs(head_dim):
    """Its transpose read assumes granule-64 staging, and says so.

    `_kt_read_base` folds `tok_off(4 * group)` into `group * granule`, which
    needs `SMEM_N_RPT` to divide 4 -- true at granule 64 and not at granule 32.
    Those rungs are all comfortably served by the 32-row family, so this is a
    limit rather than a deferral, and refusing beats a silent wrong address.
    """
    with pytest.raises(NotImplementedError, match="multiples of 64"):
        build_dq(num_heads=4, head_dim=head_dim, causal=False, mfma_rows=16)


def test_the_16x16x16_shape_is_half_rate():
    """`16x16x16` costs half the machine; `16x16x32` does not.

    Read out of `SISchedule.td` under `SIDPGFX950FullSpeedModel`, not out of an
    ISA document, per the lore. It decides B3.5's shape question on its own and
    **without the lane-map probe**: a 16-row family built on `16x16x16` cannot
    beat the 32-row family anywhere the 32-row family fits, because its
    ceiling is 50% of peak. `16x16x32` gives 16 rows at full rate.

    A test rather than a comment because the numbers are a claim about the
    toolchain, and a toolchain can change.
    """
    from fmha_tuning_bwd_dq_gfx950 import mfma_flops_per_pass

    today = mfma_flops_per_pass(32, 32, 16)
    assert mfma_flops_per_pass(16, 16, 16) == today / 2
    assert mfma_flops_per_pass(16, 16, 32) == today


def test_knobs_resolve_is_idempotent():
    """Resolving an already-resolved knob object re-derives the same answer.

    The forward's `Gfx950Knobs.resolve` promises this and the backward inherits
    the pipeline, so it must hold here too -- a step that read back a derived
    field instead of recomputing it would break it.
    """
    from fmha_tuning_gfx950 import FmhaInputMetadata

    meta = FmhaInputMetadata(num_heads=8, head_dim=128, causal=False)
    once = bwd_dq_knobs("gfx950:sramecc+:xnack-").resolve(meta)
    twice = once.resolve(meta)
    assert once == twice
    assert once.traits.STORE_DB is False
    assert bwd_dq_knobs("gfx950", store_db=True).resolve(meta).traits.STORE_DB is True


# ---------------------------------------------------------------------------
# B4: causal and windows
# ---------------------------------------------------------------------------


# Which MFMA family serves which rung is a *tuning* decision, and a feature
# implemented in one family and not the other is a wrong answer at half the
# ladder. So every B4 test names the family rather than taking the default.
# `16` is legal only on the multiples of 64; see the off-grid refusal.
_FAMILIES = (32, 16)


def _causal_ref(q, k, v, do, *, scale, window=None, dtype=torch.float64):
    """`(lse, delta, dq)` for a causal or windowed problem, at `dtype`.

    The mask is written out rather than taken from `is_causal`, because
    **torch's `is_causal` is top-left and these kernels are bottom-right**;
    they agree only at `Sq == Sk`. Spelling it means a shape that violates that
    fails loudly here rather than quietly disagreeing.
    """
    qf, kf, vf, dof = (t.to(dtype) for t in (q, k, v, do))
    sq, sk = q.shape[2], k.shape[2]
    rows = torch.arange(sq, device=q.device)[:, None]
    cols = torch.arange(sk, device=q.device)[None, :]
    right = sk - sq if window is None else window[1]
    keep = cols <= rows + right
    if window is not None:
        keep = keep & (cols >= rows - window[0])
    scores = ((qf @ kf.transpose(-1, -2)) * scale).masked_fill(~keep, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    p = torch.softmax(scores, dim=-1)
    o = p @ vf
    delta = (dof * o).sum(-1)
    ds = p * ((dof @ vf.transpose(-1, -2)) - delta.unsqueeze(-1))
    n = lse.shape[-1]
    return (
        lse.float().reshape(-1, n).contiguous(),
        delta.float().reshape(-1, n).contiguous(),
        ((ds @ kf) * scale).double(),
    )


@pytest.mark.parametrize("rows", _FAMILIES)
@pytest.mark.parametrize("head_dim", [d for d in BWD_DQ_LADDER if d % 64 == 0])
def test_causal_error_ratio(head_dim, rows):
    """Causal, every rung that both families serve, against the error-ratio gate.

    `Sq == Sk` throughout, and that is not a convenience: these kernels are
    **bottom-right** aligned and torch's `is_causal` is top-left, so any other
    shape compares two different problems. The forward's P4 lost time to this.
    """
    torch.manual_seed(20)
    b, h, s = 1, 4, 512
    q, k, v, do = (_rand(b, h, s, head_dim) for _ in range(4))
    scale = _sm_scale(head_dim)
    lse, delta, hi = _causal_ref(q, k, v, do, scale=scale)
    _, _, lo = _causal_ref(q, k, v, do, scale=scale, dtype=DT)
    got = _run_dq(q, k, v, do, lse, delta, scale=scale, causal=True, mfma_rows=rows)
    assert torch.isfinite(got.float()).all()
    assert _max_err(got, hi) <= DQ_FUDGE * _max_err(lo, hi)


@pytest.mark.parametrize("rows", _FAMILIES)
@pytest.mark.parametrize("window", [(64, 0), (128, 32), (16, 16), (0, 0)], ids=str)
def test_window_error_ratio(window, rows):
    """Generalized sliding windows, both families.

    `(0, 0)` is the degenerate band -- one live column per row -- which is the
    case where a leading masked run spans whole tiles rather than clipping one,
    and where `_skip_dead_leading_tiles` has the most to do.
    """
    torch.manual_seed(21)
    b, h, s, d = 1, 4, 512, 64
    q, k, v, do = (_rand(b, h, s, d) for _ in range(4))
    scale = _sm_scale(d)
    lse, delta, hi = _causal_ref(q, k, v, do, scale=scale, window=window)
    _, _, lo = _causal_ref(q, k, v, do, scale=scale, window=window, dtype=DT)
    got = _run_dq(q, k, v, do, lse, delta, scale=scale, causal=True, window_build=True, window=window, mfma_rows=rows)
    assert torch.isfinite(got.float()).all()
    assert _max_err(got, hi) <= DQ_FUDGE * _max_err(lo, hi)


@pytest.mark.parametrize("rows", _FAMILIES)
@pytest.mark.parametrize("head_dim", [64, 512])
def test_sentinel_window_is_bitwise_plain_causal(head_dim, rows):
    """**The sharpest test in this codebase**, and it earned that in the forward's P3.

    A window build fed `WINDOW_BOTRIGHT` on both bounds describes exactly
    bottom-right causal, so it must reproduce a plain causal build *bit for
    bit* -- not to a tolerance. It fails on any difference in the tile walk, in
    which elements the mask touches, or in the order the KV tiles accumulate,
    none of which a tolerance would notice.

    **One set of inputs for both builds.** The first version of this check
    re-randomised between them and reported a difference of 2.28 that was
    entirely the inputs; the lore's "always run a known-good control" is about
    exactly that.
    """
    torch.manual_seed(22)
    b, h, s = 1, 4, 512
    q, k, v, do = (_rand(b, h, s, head_dim) for _ in range(4))
    scale = _sm_scale(head_dim)
    lse, delta, _ = _causal_ref(q, k, v, do, scale=scale)

    plain = _run_dq(q, k, v, do, lse, delta, scale=scale, causal=True, mfma_rows=rows)
    sentinel = _run_dq(
        q,
        k,
        v,
        do,
        lse,
        delta,
        scale=scale,
        causal=True,
        window_build=True,
        window=(fmha.WINDOW_BOTRIGHT, fmha.WINDOW_BOTRIGHT),
        mfma_rows=rows,
    )
    assert torch.equal(plain, sentinel), "a window build fed the causal sentinels is not the causal build"


def _time_us(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


@pytest.mark.parametrize("rows", _FAMILIES)
def test_the_causal_tile_cut_is_not_inert(rows):
    """**A dead tile is a no-op, so only timing can show the cut works.**

    The forward's wide body shipped with its tile cut inert: right answers,
    0.92x, and nothing but a measurement found it. Causal halves the KV tiles a
    Q block walks on average, so a working cut is worth close to 2x at a long
    sequence; the bound here is 0.80x, loose enough to survive noise and the
    residual load imbalance, and tight enough that an inert cut (1.0x) fails.

    This is also what the `init_tile_bounds` override is for -- the inherited
    bound rounds the tile count up to even and floors it at four, which is up
    to three dead tiles per Q block and enough to hide the cut at short
    sequences.
    """
    torch.manual_seed(23)
    # **1024 workgroups, and the count is the point.** At `b=1` this shape
    # dispatches 256 of them onto ~256 CUs, one each -- so the *longest* Q
    # block sets the clock and halving the total work is invisible (measured
    # 0.92x, with the cut demonstrably working). Causal load is `2i + 2` tiles
    # for block `i`, so the saving only appears once several workgroups share a
    # CU. A timing assertion that does not account for that is a flaky test
    # about occupancy wearing a correctness hat.
    #
    # Measured here: 269 us dense against 187 causal, 0.69x. The ideal is 0.5x
    # and the gap is the residual imbalance.
    b, h, s, d = 4, 8, 4096, 64
    q, k, v, do = (_rand(b, h, s, d) for _ in range(4))
    dq = torch.empty_like(q)
    lse = torch.zeros(b * h, s, device="cuda", dtype=torch.float32)
    scale = _sm_scale(d)

    def build(causal):
        fn = build_dq(num_heads=h, head_dim=d, causal=causal, mfma_rows=rows)
        return lambda: fn(q, k, v, do, dq, lse, lse, b, s, seqlen_k=s, scale=scale)

    dense = _time_us(build(False))
    causal = _time_us(build(True))
    assert causal < 0.80 * dense, (
        f"causal {causal:.0f} us against dense {dense:.0f} us at {rows} rows: the tile cut is inert. "
        "A dead tile changes no output bit, so no correctness test can see this."
    )


@pytest.mark.parametrize("rows", _FAMILIES)
def test_the_window_tile_cut_is_not_inert(rows):
    """The left bound must move the *base* of the walk, not just mask.

    `max_num_tiles` truncates the walk on the right; only
    `_skip_dead_leading_tiles` moves it on the left, and P3 found the forward's
    wide body walking from a literal tile 0 with a correct mask on top -- 6.7x
    left on the table and every answer right.
    """
    torch.manual_seed(24)
    b, h, s, d = 1, 8, 4096, 64
    q, k, v, do = (_rand(b, h, s, d) for _ in range(4))
    dq = torch.empty_like(q)
    lse = torch.zeros(b * h, s, device="cuda", dtype=torch.float32)
    scale = _sm_scale(d)
    fn = build_dq(num_heads=h, head_dim=d, causal=True, window=True, mfma_rows=rows)

    def at(window):
        return lambda: fn(q, k, v, do, dq, lse, lse, b, s, seqlen_k=s, scale=scale, window=window)

    unbounded = _time_us(at((fmha.WINDOW_BOTRIGHT, fmha.WINDOW_BOTRIGHT)))
    narrow = _time_us(at((128, 0)))
    assert narrow < 0.5 * unbounded, (
        f"a 128-wide left band takes {narrow:.0f} us against {unbounded:.0f} for an unbounded one at "
        f"{rows} rows: the left bound is masking but not cutting."
    )


@pytest.mark.parametrize("rows,head_dim", [(32, 128), (16, 512)])
@pytest.mark.parametrize("window_build", [False, True], ids=["causal", "window"])
def test_no_transpose_read_under_a_restricted_exec(rows, head_dim, window_build):
    """CDNA4 11.4: `ds_read_b64_tr_b16` requires **EXEC all 1s**.

    B4 is the first phase where a branch exists to violate it, and the failure
    mode is the house speciality -- finite, wrong, no diagnostic. Both kernels
    reported at B3 that no `scf.if` existed yet and that this is the phase to
    check rather than assume.

    The invariant checked is stronger and simpler than "no transpose read
    inside a divergent region": **the kernel restricts EXEC nowhere at all.**
    Every masking guard here is wave-uniform -- the tile index and the wave's
    first row, both `readfirstlane`d -- so the compiler emits scalar branches
    (`s_cbranch`) and never a `saveexec` pair. If a guard ever became lane-
    varying, `s_and_saveexec_b64` would appear and this fails, whether or not
    a transpose read happened to land inside it that day.
    """
    import glob
    import os
    import re
    import subprocess
    import tempfile

    script = (
        "import math, torch\n"
        "from fmha_bwd_dq_gfx950 import build_fmha_bwd_dq_gfx950_module as b\n"
        "import fmha_common_gfx1201 as fmha\n"
        f"d, rows, win = {head_dim}, {rows}, {int(window_build)}\n"
        "DT=torch.bfloat16; bb,h,s=1,2,256\n"
        "q,k,v,do=(torch.randn(bb,h,s,d,device='cuda',dtype=DT) for _ in range(4))\n"
        "dq=torch.zeros(bb,h,s,d,device='cuda',dtype=DT)\n"
        "l=torch.zeros(bb*h,s,device='cuda',dtype=torch.float32)\n"
        "fn=b(num_heads=h,head_dim=d,causal=True,window=bool(win),mfma_rows=rows)\n"
        "w=(fmha.WINDOW_BOTRIGHT,fmha.WINDOW_BOTRIGHT) if win else None\n"
        "fn(q,k,v,do,dq,l,l,bb,s,seqlen_k=s,scale=1.0/math.sqrt(d),window=w)\n"
        "torch.cuda.synchronize()\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "build.py")
        open(src, "w").write(script)
        env = dict(os.environ, FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=tmp, FLYDSL_RUNTIME_ENABLE_CACHE="0")
        env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run([sys.executable, src], env=env, cwd=tmp, capture_output=True, timeout=900)
        dumps = glob.glob(os.path.join(tmp, "*", "*final_isa.s"))
        assert dumps, "no ISA dump; the build did not get far enough for this test to mean anything"
        isa = open(dumps[0]).read()

    assert isa.count("ds_read_b64_tr") > 0, "no transpose read in the dump: this test would pass vacuously"
    saveexec = re.findall(r"^\s+s_\w*saveexec\w*\b.*$", isa, re.M)
    exec_write = re.findall(r"^\s+s_[a-z0-9_]+\s+exec\b.*$", isa, re.M)
    assert not saveexec and not exec_write, (
        f"the kernel restricts EXEC ({len(saveexec)} saveexec, {len(exec_write)} exec writes), so a "
        "transpose read may execute under a partial mask. Every masking guard must be wave-uniform."
    )
