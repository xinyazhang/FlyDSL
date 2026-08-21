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


def _run_dq(q, k, v, do, lse, delta, *, scale=None, db=None, dq=None):
    """Build for this shape and dispatch, returning dQ.

    `dq` is filled with NaN first, not zeroed: a kernel that fails to write a
    row would pass a zero-initialised check on any row whose true gradient is
    small, and the ragged-sequence cases are exactly where that happens.
    """
    b, hq, sq, d = q.shape
    if dq is None:
        dq = torch.full((b, hq, sq, d), float("nan"), device="cuda", dtype=q.dtype)
    fn = build_dq(
        num_heads=hq,
        head_dim=d,
        causal=False,
        dtype_str="bf16" if q.dtype is torch.bfloat16 else "f16",
        num_kv_heads=k.shape[1],
        store_db=db is not None,
    )
    fn(q, k, v, do, dq, lse, delta, b, sq, seqlen_k=k.shape[2], scale=scale, db=db)
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


@pytest.mark.parametrize("head_dim", BWD_DQ_LADDER)
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
    [("causal", True), ("window", True), ("bias", True), ("dropout", True)],
)
def test_unimplemented_modes_are_refused(field, value):
    """Refused by name, not ignored.

    Every one of these would build, run and return a correctly-shaped wrong
    answer, which is the failure mode the whole backward plan is written
    against. B4-B6 turn them on one at a time.
    """
    with pytest.raises(NotImplementedError, match=field):
        build_dq(num_heads=4, head_dim=64, **{field: value})


@pytest.mark.parametrize("head_dim", [32, 192, 256, 512])
def test_unbuilt_rungs_are_refused(head_dim):
    """B2 is head_dim 64 and 128. The rest is B3, and says so."""
    with pytest.raises(NotImplementedError, match="head_dim"):
        build_dq(num_heads=4, head_dim=head_dim, causal=False)


@pytest.mark.parametrize("head_dim,head_dim_v", [(100, None), (48, None), (64, 32)])
def test_padded_and_asymmetric_heads_are_refused(head_dim, head_dim_v):
    """The two extents are used the other way round here, so refuse until B3.

    GEMM2 reads V through the *K* register path, whose padded-head mask is
    written against `hdim_qk`; the dQ store suppresses against `hdim_vo`. Both
    are the wrong extent for this kernel, and a build that ignored that would
    return a finite gradient with the pad folded in.
    """
    with pytest.raises(NotImplementedError):
        build_dq(num_heads=4, head_dim=head_dim, head_dim_v=head_dim_v, causal=False)


def test_lse_and_delta_layout_is_checked_on_the_host():
    """The kernel derives their pitches rather than reading strides.

    So the host is the only place the caller's actual layout can be verified,
    and a silently-wrong pitch reads a plausible LSE for the wrong row.
    """
    torch.manual_seed(10)
    b, h, s, d = 1, 2, 256, 64
    q, k, v, do = (_rand(b, h, s, d) for _ in range(4))
    scale = _sm_scale(d)
    lse, delta, _ = _fwd_stats(q, k, v, do, scale=scale, dtype=torch.float32)
    with pytest.raises(ValueError, match="logsumexp"):
        _run_dq(q, k, v, do, lse.to(torch.float16), delta, scale=scale)
    with pytest.raises(ValueError, match="delta"):
        _run_dq(q, k, v, do, lse, delta[:, : s // 2].contiguous(), scale=scale)
    with pytest.raises(ValueError, match="logsumexp"):
        # Rank 3 `(B, H, S)` -- the shape the forward writes, and the one a
        # caller reaches for. It has to be viewed at rank 2, so say so.
        _run_dq(q, k, v, do, lse.reshape(b, h, s), delta, scale=scale)


def test_db_tensor_and_build_must_agree():
    """A dB build needs a tensor, and a non-dB build must refuse one.

    Silently ignoring a dB tensor returns gradients that are the right shape
    with the bias gradient missing, and it is only ever passed by a caller who
    believes it is being written.
    """
    torch.manual_seed(11)
    b, h, s, d = 1, 2, 128, 64
    q, k, v, do = (_rand(b, h, s, d) for _ in range(4))
    scale = _sm_scale(d)
    lse, delta, _ = _fwd_stats(q, k, v, do, scale=scale, dtype=torch.float32)
    db = torch.zeros(b, h, s, s, device="cuda", dtype=DT)
    dq = torch.empty(b, h, s, d, device="cuda", dtype=DT)

    plain = build_dq(num_heads=h, head_dim=d, causal=False)
    with pytest.raises(ValueError, match="not compiled for dB"):
        plain(q, k, v, do, dq, lse, delta, b, s, seqlen_k=s, db=db)

    with_db = build_dq(num_heads=h, head_dim=d, causal=False, store_db=True)
    with pytest.raises(ValueError, match="requires a"):
        with_db(q, k, v, do, dq, lse, delta, b, s, seqlen_k=s)


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
