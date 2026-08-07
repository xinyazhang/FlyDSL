# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness for the gfx1201 fused backward kernel.

The reference is PyTorch autograd on an **fp32** attention written out
explicitly -- not `F.scaled_dot_product_attention` with `is_causal=True`.
Writing the mask out is what makes the bottom-right and sliding-window cases
unambiguous: `is_causal` is top-left, and there is a standing argument about
whether it should be (pytorch/pytorch#108108), so a test that relied on it
would be testing the reference's opinion rather than the kernel.

Everything is compared as a relative Frobenius error against that fp32
reference. The tolerance is 2e-2 for f16 and 4e-2 for bf16 -- loose, because
the quantity being bounded is a *gradient* of a 16-bit computation, and P and
dS are both narrowed to 16 bits before they reach the matrix cores (gfx1201
WMMA has no f32 A/B operand). Measured errors are two orders of magnitude
inside that; the test prints them so a regression that stays under tolerance is
still visible.

Two things are checked beyond raw accuracy:

- **Interoperation with our forward kernel.** ``test_matches_forward_kernel_lse``
  takes O and LSE from `flydsl_flash_attn_func_gfx1201` rather than from torch,
  which is the pairing that actually ships. A backward pass that only ever sees
  a torch-computed LSE would not notice a disagreement about the logsumexp
  convention (natural log here) or about its layout.
- **Varlen equivalence.** Each packed sequence's gradients must match the same
  sequence run densely, which is the property the whole VarlenBits decode
  exists to provide.
"""

from __future__ import annotations

import math

import pytest
import torch
from fmha_bwd_fuse_gfx1201_interface import flydsl_fmha_bwd_fuse_gfx1201 as bwd_fuse
from fmha_tuning_bwd_fuse_gfx1201 import BwdInputMetadata, BwdKnobs
from fmha_tuning_bwd_fuse_gfx1201 import plan as bwd_plan

_TOL = {torch.float16: 2e-2, torch.bfloat16: 4e-2}


def _require_env():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.lower().split(":")[0]
    if not arch.startswith("gfx1201"):
        pytest.skip(f"needs gfx1201, got {arch}")


def _mask(seq_q, seq_k, causal_type, window, device):
    """The visible-key predicate, as a (Sq, Sk) bool.

    One expression covers all four modes because they are all the same
    two-sided band: a key `j` is visible to query `i` iff
    ``i - left <= j <= i + right``. Non-causal is the band with both edges
    unbounded, top-left is ``right = 0``, bottom-right is
    ``right = Sk - Sq``.
    """
    i = torch.arange(seq_q, device=device)[:, None]
    j = torch.arange(seq_k, device=device)[None, :]
    if causal_type == 0:
        return torch.ones(seq_q, seq_k, dtype=torch.bool, device=device)
    if causal_type == 1:
        left, right = seq_q, 0
    elif causal_type == 2:
        left, right = seq_q, seq_k - seq_q
    else:
        left, right = window
    return (j <= i + right) & (j >= i - left)


def _reference(q, k, v, do, scale, causal_type, window, group):
    """(o, lse, dq, dk, dv) in fp32, by autograd on an explicit formulation."""
    qf = q.float().detach().requires_grad_()
    kf = k.float().detach().requires_grad_()
    vf = v.float().detach().requires_grad_()
    kx = kf.repeat_interleave(group, dim=1) if group > 1 else kf
    vx = vf.repeat_interleave(group, dim=1) if group > 1 else vf
    s = (qf @ kx.transpose(-1, -2)) * scale
    m = _mask(q.shape[2], k.shape[2], causal_type, window, q.device)
    s = s.masked_fill(~m, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    # A row that sees no key at all: the kernel writes +inf there, so that
    # `exp(qk - lse)` is zero for exactly the rows that must contribute
    # nothing. `logsumexp` of an all -inf row is -inf, which would make the
    # same expression NaN.
    lse = torch.where(torch.isfinite(lse), lse, torch.full_like(lse, float("inf")))
    p = torch.exp(s - lse[..., None])
    o = p @ vx
    dq, dk, dv = torch.autograd.grad(o, (qf, kf, vf), do.float())
    return o.detach(), lse.detach().contiguous(), dq, dk, dv


def _rel(got, want):
    n = want.norm()
    if n == 0:
        return float(got.float().norm())
    return float((got.float() - want).norm() / n)


def _check(label, dtype, got, want):
    tol = _TOL[dtype]
    errs = {name: _rel(g, w) for name, g, w in zip(("dq", "dk", "dv"), got, want)}
    print(f"  {label}: " + " ".join(f"{n}={e:.3e}" for n, e in errs.items()))
    for name, err in errs.items():
        assert err < tol, f"{label}: {name} rel err {err:.3e} >= {tol}"


def _run(
    batch=1,
    nhq=2,
    nhk=None,
    seq_q=128,
    seq_k=None,
    head_dim=64,
    dtype=torch.float16,
    causal_type=0,
    window=None,
    scale=None,
    seed=0,
    use_fwd_kernel=False,
):
    nhk = nhq if nhk is None else nhk
    seq_k = seq_q if seq_k is None else seq_k
    scale = 1.0 / math.sqrt(head_dim) if scale is None else scale
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(batch, nhq, seq_q, head_dim, device="cuda", dtype=dtype, generator=g)
    k = torch.randn(batch, nhk, seq_k, head_dim, device="cuda", dtype=dtype, generator=g)
    v = torch.randn(batch, nhk, seq_k, head_dim, device="cuda", dtype=dtype, generator=g)
    do = torch.randn(batch, nhq, seq_q, head_dim, device="cuda", dtype=dtype, generator=g)

    group = nhq // nhk
    o_ref, lse_ref, dq_r, dk_r, dv_r = _reference(q, k, v, do, scale, causal_type, window, group)

    if use_fwd_kernel:
        from flash_attn_func_gfx1201_interface import flydsl_flash_attn_func_gfx1201 as fwd

        o, lse = fwd(q, k, v, causal=causal_type != 0, return_lse=True)
        o = o.contiguous()
        lse = lse.contiguous()
    else:
        o, lse = o_ref.to(dtype), lse_ref

    dq, dk, dv = bwd_fuse(
        q,
        k,
        v,
        o,
        do,
        lse,
        causal=causal_type != 0,
        causal_type=causal_type or None,
        window=window,
        sm_scale=scale,
    )
    torch.cuda.synchronize()
    return (dq, dk, dv), (dq_r, dk_r, dv_r)


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [16, 32, 48, 64, 80, 128])
@pytest.mark.parametrize("causal_type", [0, 1], ids=["full", "topleft"])
def test_head_dim_ladder(head_dim, causal_type):
    _require_env()
    got, want = _run(head_dim=head_dim, causal_type=causal_type, seq_q=128)
    _check(f"d{head_dim}/ct{causal_type}", torch.float16, got, want)


@pytest.mark.parametrize("seq", [1, 7, 33, 64, 100, 129, 256, 384])
@pytest.mark.parametrize("causal_type", [0, 1], ids=["full", "topleft"])
def test_ragged_seqlen(seq, causal_type):
    """Sequence lengths that are not multiples of any tile.

    Both roles have a tail: the dK/dV role's is a partial *query* block and the
    dQ role's a partial *key* block, and the two are masked by different code.
    A length that divides both tiles exercises neither.
    """
    _require_env()
    got, want = _run(seq_q=seq, head_dim=64, causal_type=causal_type)
    _check(f"S{seq}/ct{causal_type}", torch.float16, got, want)


@pytest.mark.parametrize("nhq,nhk", [(8, 8), (8, 4), (8, 1), (2, 1)])
@pytest.mark.parametrize("causal_type", [0, 1], ids=["full", "topleft"])
def test_gqa(nhq, nhk, causal_type):
    """MQA/GQA. dK/dV are computed per *query* head and reduced by the
    interface, so this is the only test of that reduction."""
    _require_env()
    got, want = _run(nhq=nhq, nhk=nhk, seq_q=128, head_dim=64, causal_type=causal_type)
    _check(f"gqa{nhq}/{nhk}/ct{causal_type}", torch.float16, got, want)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize("causal_type", [0, 1], ids=["full", "topleft"])
def test_dtypes(dtype, causal_type):
    _require_env()
    got, want = _run(dtype=dtype, head_dim=128, seq_q=192, causal_type=causal_type)
    _check(f"{dtype}/ct{causal_type}", dtype, got, want)


@pytest.mark.parametrize("batch", [1, 3])
def test_batch(batch):
    _require_env()
    got, want = _run(batch=batch, nhq=3, seq_q=96, head_dim=64, causal_type=1)
    _check(f"B{batch}", torch.float16, got, want)


def test_non_default_scale():
    """`sm_scale` must reach both the exponent and the dK/dQ epilogue.

    A scale left out of the epilogue is invisible at the default
    `1/sqrt(head_dim)` only if the reference also uses it there -- which it
    does -- so a distinct value is the only thing that separates the two uses.
    """
    _require_env()
    got, want = _run(seq_q=128, head_dim=64, causal_type=1, scale=0.137)
    _check("scale", torch.float16, got, want)


# ---------------------------------------------------------------------------
# Masking modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_q,seq_k", [(64, 128), (128, 128), (96, 160), (128, 64)])
def test_bottom_right_causal(seq_q, seq_k):
    """`causal_type=2`: the diagonal sits at `seqlen_k - seqlen_q`.

    `(128, 64)` is the case where the leading `seq_q - seq_k` rows see **no
    key at all**. The kernel writes zero dQ there, which is right, and the
    reference agrees only because `_reference` replaces the `-inf` logsumexp of
    an empty row with `+inf` -- the same convention the forward kernel writes.
    Left in rather than split out: it is the one configuration that exercises
    the empty-row path at all.
    """
    _require_env()
    got, want = _run(seq_q=seq_q, seq_k=seq_k, head_dim=64, causal_type=2)
    _check(f"botright{seq_q}/{seq_k}", torch.float16, got, want)


@pytest.mark.parametrize("window", [(32, 0), (16, 16), (0, 0), (256, 8)])
def test_sliding_window(window):
    """`causal_type=3`: an explicit two-sided band.

    `(0, 0)` is the diagonal alone -- every row sees exactly one key -- which
    is the sharpest test of the region decomposition, since the fully-live
    region is empty for every query block and both masked runs are non-trivial.
    """
    _require_env()
    got, want = _run(seq_q=192, seq_k=192, head_dim=64, causal_type=3, window=window)
    _check(f"win{window}", torch.float16, got, want)


def test_cross_attention_lengths():
    _require_env()
    for sq, sk in ((64, 192), (192, 64), (100, 37)):
        got, want = _run(seq_q=sq, seq_k=sk, head_dim=64, causal_type=0)
        _check(f"cross{sq}/{sk}", torch.float16, got, want)


# ---------------------------------------------------------------------------
# Interoperation with the forward kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("causal_type", [0, 1], ids=["full", "topleft"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_matches_forward_kernel_lse(causal_type, head_dim):
    """O and LSE from our forward kernel, not from torch.

    This is the pairing that ships, and it is the only test that would catch a
    disagreement about the logsumexp convention -- the forward writes natural
    log, and reading it as base-2 produces gradients that are wrong by a
    smooth per-row factor no shape check notices.
    """
    _require_env()
    got, want = _run(seq_q=256, head_dim=head_dim, nhq=4, nhk=4, causal_type=causal_type, use_fwd_kernel=True)
    _check(f"fwd-lse/d{head_dim}/ct{causal_type}", torch.float16, got, want)


# ---------------------------------------------------------------------------
# Varlen
# ---------------------------------------------------------------------------


def _cu(lens):
    out = [0]
    for x in lens:
        out.append(out[-1] + x)
    return torch.tensor(out, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("causal_type", [0, 1], ids=["full", "topleft"])
def test_varlen_compact_matches_dense(causal_type):
    """Suite: every packed sequence's gradients equal its own dense call.

    Dense-equivalence rather than reference-equivalence, because it isolates
    the *addressing*: both sides run the identical kernel over the identical
    numbers, so any difference is the VarlenBits decode and nothing else.
    """
    _require_env()
    from flash_attn_func_gfx1201_aiw import build_flash_attn_func_aiw_module

    head_dim, nh = 64, 2
    lens_q = [96, 40, 5, 0, 137]
    lens_k = [x + 11 * i if x else 0 for i, x in enumerate(lens_q)]
    cq, ck = _cu(lens_q), _cu(lens_k)
    Tq, Tk = int(cq[-1]), int(ck[-1])
    n = len(lens_q)
    mq, mk = max(lens_q), max(lens_k)
    g = torch.Generator(device="cuda").manual_seed(3)
    q = torch.randn(1, nh, Tq, head_dim, device="cuda", dtype=torch.float16, generator=g)
    k = torch.randn(1, nh, Tk, head_dim, device="cuda", dtype=torch.float16, generator=g)
    v = torch.randn(1, nh, Tk, head_dim, device="cuda", dtype=torch.float16, generator=g)
    do = torch.randn(1, nh, Tq, head_dim, device="cuda", dtype=torch.float16, generator=g)
    scale = 1.0 / math.sqrt(head_dim)

    # Forward, packed, to get O and LSE in the packed layout.
    fwd = build_flash_attn_func_aiw_module(num_heads=nh, head_dim=head_dim, causal=bool(causal_type), dtype_str="f16")
    o = torch.zeros_like(q)
    lse = torch.zeros(nh, max(Tq, 1), dtype=torch.float32, device="cuda")
    vl = fwd.varlen_compact(cq, ck, mq, mk)
    fwd(q, k, v, o, n, mq, mk, lse=lse, varlen=vl)
    torch.cuda.synchronize()

    dq, dk, dv = bwd_fuse(q, k, v, o, do, lse, causal=bool(causal_type), sm_scale=scale, varlen=vl, num_seqlens=n)
    torch.cuda.synchronize()

    for z, (lq, lk) in enumerate(zip(lens_q, lens_k)):
        if lq == 0 or lk == 0:
            continue
        qs, ks = int(cq[z]), int(ck[z])
        qz = q[:, :, qs : qs + lq].contiguous()
        kz = k[:, :, ks : ks + lk].contiguous()
        vz = v[:, :, ks : ks + lk].contiguous()
        doz = do[:, :, qs : qs + lq].contiguous()
        oz = torch.zeros_like(qz)
        lsez = torch.zeros(nh, lq, dtype=torch.float32, device="cuda")
        fwd(qz, kz, vz, oz, 1, lq, lk, lse=lsez)
        torch.cuda.synchronize()
        rq, rk, rv = bwd_fuse(qz, kz, vz, oz, doz, lsez, causal=bool(causal_type), sm_scale=scale)
        torch.cuda.synchronize()
        assert torch.equal(rq, dq[:, :, qs : qs + lq]), f"dq sequence {z} differs from its dense call"
        assert torch.equal(rk, dk[:, :, ks : ks + lk]), f"dk sequence {z} differs from its dense call"
        assert torch.equal(rv, dv[:, :, ks : ks + lk]), f"dv sequence {z} differs from its dense call"


def test_varlen_compact_accuracy():
    """The packed path against the fp32 reference, per sequence."""
    _require_env()
    from flash_attn_func_gfx1201_aiw import build_flash_attn_func_aiw_module

    head_dim, nh = 64, 2
    lens = [64, 96, 33]
    cq = _cu(lens)
    T = int(cq[-1])
    n = len(lens)
    m = max(lens)
    g = torch.Generator(device="cuda").manual_seed(5)
    q = torch.randn(1, nh, T, head_dim, device="cuda", dtype=torch.float16, generator=g)
    k = torch.randn(1, nh, T, head_dim, device="cuda", dtype=torch.float16, generator=g)
    v = torch.randn(1, nh, T, head_dim, device="cuda", dtype=torch.float16, generator=g)
    do = torch.randn(1, nh, T, head_dim, device="cuda", dtype=torch.float16, generator=g)
    scale = 1.0 / math.sqrt(head_dim)

    fwd = build_flash_attn_func_aiw_module(num_heads=nh, head_dim=head_dim, causal=True, dtype_str="f16")
    o = torch.zeros_like(q)
    lse = torch.zeros(nh, T, dtype=torch.float32, device="cuda")
    vl = fwd.varlen_compact(cq, cq, m, m)
    fwd(q, k, v, o, n, m, m, lse=lse, varlen=vl)
    torch.cuda.synchronize()
    dq, dk, dv = bwd_fuse(q, k, v, o, do, lse, causal=True, sm_scale=scale, varlen=vl, num_seqlens=n)
    torch.cuda.synchronize()

    for z, ln in enumerate(lens):
        s0 = int(cq[z])
        sl = slice(s0, s0 + ln)
        _, _, rq, rk, rv = _reference(q[:, :, sl], k[:, :, sl], v[:, :, sl], do[:, :, sl], scale, 1, None, 1)
        _check(f"varlen[{z}]", torch.float16, (dq[:, :, sl], dk[:, :, sl], dv[:, :, sl]), (rq, rk, rv))


# ---------------------------------------------------------------------------
# Build-time contracts
# ---------------------------------------------------------------------------


def test_head_dim_over_the_register_wall_is_rejected():
    """head_dim 256 must fail at plan time with a message naming the reason.

    Not a slow path: the dK/dV role's two accumulators would occupy every VGPR
    a lane has, so the alternative is spilling a loop-carried accumulator every
    iteration.
    """
    with pytest.raises(ValueError, match="head_dim"):
        bwd_plan(BwdInputMetadata(num_head_q=1, num_head_k=1, head_dim=256))


def test_dropout_is_rejected_not_ignored():
    from fmha_bwd_fuse_gfx1201_kernel import build_fmha_bwd_fuse_module

    meta = BwdInputMetadata(num_head_q=1, num_head_k=1, head_dim=64, dropout=True)
    p = bwd_plan(meta)
    with pytest.raises(NotImplementedError, match="dropout"):
        build_fmha_bwd_fuse_module(p.meta, p.knobs)


def test_lds_budget_is_checked_at_plan_time():
    """A knob combination that overflows LDS is a `ValueError`, not a launch
    failure with no indication of which knob caused it."""
    with pytest.raises(ValueError, match="LDS"):
        bwd_plan(BwdInputMetadata(num_head_q=1, num_head_k=1, head_dim=128), BwdKnobs(q_step=256))


def test_gqa_head_counts_are_validated():
    with pytest.raises(ValueError, match="num_head_q"):
        bwd_plan(BwdInputMetadata(num_head_q=6, num_head_k=4, head_dim=64))
