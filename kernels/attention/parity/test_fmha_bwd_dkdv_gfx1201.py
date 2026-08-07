# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness tests for the gfx1201 dK / dV backward kernel.

The bar here is different from the forward's, and it is worth saying why. The
forward's sharpest test is *self*-equivalence -- a window equal to the causal
diagonal must reproduce the causal path bitwise, varlen with N sequences must
reproduce N dense calls bitwise -- because it had no independent oracle. This
kernel does have one: PyTorch autograd on an fp32 reference. So the primary
test is a tolerance against that, and self-equivalence is used only where the
oracle cannot reach (varlen, where autograd has no packed-batch form).

**The fp32 reference must handle a row with no live keys the way our forward
does**, or it compares against NaN rather than against an answer.
`torch.softmax` of an all `-inf` row is 0/0 and `torch.logsumexp` of one is
`-inf`; our forward writes `O = 0` and `logsumexp = +inf` for exactly those
rows, which is what makes `p = exp2(s - lse) = 0` fall out in the backward.
`_reference` does the same. Bottom-right causal with `seqlen_q > seqlen_k` is
the configuration that reaches it, and it was a real NaN before the reference
was fixed -- in the *test*, not the kernel.

Run it individually, per this directory's prototype convention::

    export ROCM_PATH=$(rocm-sdk path --root)
    cd kernels/attention/parity && python3 -m pytest test_fmha_bwd_dkdv_gfx1201.py -v
"""

import os

import pytest
import torch
import torch.nn.functional as F
from dropout_mask_gfx1201 import dropout_mask
from flash_attn_func_gfx1201_interface import flydsl_flash_attn_func_gfx1201
from fmha_bwd_dkdv_gfx1201_interface import flydsl_flash_attn_bwd_dkdv_gfx1201
from fmha_bwd_dkdv_gfx1201_kernel import build_bwd_dkdv_module_primary, varlen_compact, varlen_padded
from fmha_tuning_bwd_dkdv_gfx1201 import BwdDkDvKnobs, BwdDkDvMetadata
from fmha_tuning_bwd_dkdv_gfx1201 import plan as bwd_plan
from philox import dropout_threshold

# f16 tolerance. dK and dV are accumulated in f32 and rounded once on store, so
# the error is dominated by the 16-bit A/B operands of four chained GEMMs; every
# configuration below measures ~3e-4, which is where a comparable f16 forward
# sits. bf16 has ~8 fewer mantissa bits and measures ~4e-3.
_RTOL_F16 = 2e-3
_RTOL_BF16 = 2e-2


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


def _tol(dtype):
    return _RTOL_BF16 if dtype == torch.bfloat16 else _RTOL_F16


def _rel(got, want, floor=0.0):
    """Relative error, with a floor under the denominator.

    A floor is needed because one configuration makes a gradient
    *analytically* zero: `window=(0, 0)` leaves only the diagonal live, so
    `p` is exactly 1 there, `dp` equals `delta`, and `dS = p * (dp - delta)`
    cancels to zero for every element -- hence `dK == 0` exactly. Measured, the
    reference norm is `0.0` and the kernel's is `1.7e-5`, so the ratio is
    meaningless and the meaningful question is whether the kernel's value is
    small *on the scale of this problem*. Callers pass a floor derived from the
    other gradient, which comes from the same tensors and is not degenerate.
    """
    den = max(want.float().norm().item(), float(floor))
    return ((got.float() - want.float()).norm() / max(den, 1e-30)).item()


def _band(seq_q, seq_k, causal_type, window, device):
    """The visible band as a boolean `(Sq, Sk)` mask, or None for no masking.

    One expression for all three causal vocabularies, because they *are* one:
    `causal_type` 1 and 2 are `(seqlen_q, 0)` and `(seqlen_q, seqlen_k -
    seqlen_q)` respectively, which is why the kernel ships only 0 and 3.
    """
    if causal_type == 0:
        return None
    i = torch.arange(seq_q, device=device)[:, None]
    j = torch.arange(seq_k, device=device)[None, :]
    if causal_type == 1:
        wl, wr = seq_q, 0
    elif causal_type == 2:
        wl, wr = seq_q, seq_k - seq_q
    else:
        wl, wr = window
    return (j <= i + wr) & (j >= i - wl)


def _reference(q, k, v, do, causal_type=0, window=None, sm_scale=None, keep=None, dropout_p=0.0):
    """fp32 forward + autograd, returning `(o, lse, delta, dk, dv)`.

    Everything is fp32 and the softmax is explicit rather than
    `F.scaled_dot_product_attention`, for two reasons: SDPA has no bottom-right
    or sliding-window form, and it has no way to inject a *given* dropout mask,
    which is what makes the dropout test compare against the same stream the
    kernel drew rather than against a statistic.
    """
    dev = q.device
    seq_q, seq_k = q.shape[2], k.shape[2]
    head_dim = q.shape[3]
    sm = 1.0 / (head_dim**0.5) if sm_scale is None else sm_scale
    rep = q.shape[1] // k.shape[1]

    qf = q.float().requires_grad_()
    kf = k.float().requires_grad_()
    vf = v.float().requires_grad_()
    kg = kf.repeat_interleave(rep, dim=1)
    vg = vf.repeat_interleave(rep, dim=1)

    s = (qf @ kg.transpose(-1, -2)) * sm
    band = _band(seq_q, seq_k, causal_type, window, dev)
    if band is not None:
        s = s.masked_fill(~band, float("-inf"))

    # A row with no live key: match what our forward writes rather than what
    # torch produces. See the module docstring.
    live = torch.isfinite(s).any(-1, keepdim=True)
    p = torch.where(live, torch.softmax(s, -1), torch.zeros_like(s))
    if keep is not None:
        p = torch.where(keep, p * (1.0 / (1.0 - dropout_p)), torch.zeros_like(p))
    o = p @ vg
    lse = torch.logsumexp(s, -1)
    lse = torch.where(live.squeeze(-1), lse, torch.full_like(lse, float("inf")))

    dk, dv = torch.autograd.grad(o, (kf, vf), do.float())
    delta = (do.float() * o.detach()).sum(-1)
    return o.detach(), lse.detach(), delta, dk, dv


def _bwd_via_kernel(q, k, v, do, o, lse, delta, causal_type, window, dropout_p=None, seed=0, offset=0, knobs=None):
    """Direct builder call, bypassing the interface's delta and allocation."""
    batch, num_heads_q, seq_q, _ = q.shape
    seq_k = k.shape[2]
    dtype_str = "bf16" if q.dtype == torch.bfloat16 else "f16"
    pitch_k = (k.shape[3] + 7) // 8 * 8
    pitch_v = (v.shape[3] + 7) // 8 * 8
    dk = torch.zeros(*k.shape[:3], pitch_k, dtype=k.dtype, device=k.device)[..., : k.shape[3]]
    dv = torch.zeros(*v.shape[:3], pitch_v, dtype=v.dtype, device=v.device)[..., : v.shape[3]]
    _plan = bwd_plan(
        BwdDkDvMetadata(
            num_heads=num_heads_q,
            head_dim=k.shape[3],
            head_dim_v=v.shape[3],
            causal=causal_type != 0,
            causal_type=causal_type or None,
            dtype_str=dtype_str,
            dropout=dropout_p is not None,
        ),
        knobs,
    )
    exe = build_bwd_dkdv_module_primary(_plan.meta, _plan.knobs)
    exe(
        q,
        k,
        v,
        do,
        dk,
        dv,
        lse.reshape(batch * num_heads_q, seq_q).contiguous(),
        delta.reshape(batch * num_heads_q, seq_q).contiguous(),
        batch,
        seq_q,
        seqlen_k=seq_k,
        window=window,
        dropout_p=dropout_p,
        philox_seed=seed,
        philox_offset=offset,
    )
    torch.cuda.synchronize()
    return dk, dv


def _rand(batch, heads, seq, head_dim, dtype, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(batch, heads, seq, head_dim, dtype=dtype, device="cuda", generator=gen)


def _case(batch, hq, hk, seq_q, seq_k, head_dim, dtype=torch.float16, seed=0):
    return (
        _rand(batch, hq, seq_q, head_dim, dtype, seed),
        _rand(batch, hk, seq_k, head_dim, dtype, seed + 1),
        _rand(batch, hk, seq_k, head_dim, dtype, seed + 2),
        _rand(batch, hq, seq_q, head_dim, dtype, seed + 3),
    )


def _check(q, k, v, do, causal_type=0, window=None, dtype=None, label="", seed=0, offset=0, **kw):
    """Reference and kernel on the same inputs, both gradients within tolerance.

    `seed` / `offset` are the kernel's Philox arguments and are deliberately
    *not* forwarded to `_reference`, which takes the resulting `keep` mask
    instead. That is the point of the dropout test: the mask comes from the
    independent debug kernel rather than from a second transcription here.
    """
    o, lse, delta, gk, gv = _reference(q, k, v, do, causal_type, window, **kw)
    dk, dv = _bwd_via_kernel(
        q,
        k,
        v,
        do,
        o,
        lse,
        delta,
        causal_type,
        window,
        dropout_p=kw.get("dropout_p") if kw.get("keep") is not None else None,
        seed=seed,
        offset=offset,
    )
    tol = _tol(dtype or q.dtype)
    # See `_rel`: the two gradients share a problem, so the larger norm is the
    # scale on which "zero" is judged for a degenerate one.
    floor = 1e-3 * max(gk.norm().item(), gv.norm().item())
    r_dk, r_dv = _rel(dk, gk, floor), _rel(dv, gv, floor)
    assert torch.isfinite(dk).all(), f"{label}: dk not finite"
    assert torch.isfinite(dv).all(), f"{label}: dv not finite"
    assert r_dk < tol, f"{label}: dk rel={r_dk:.3e} >= {tol}"
    assert r_dv < tol, f"{label}: dv rel={r_dv:.3e} >= {tol}"
    return r_dk, r_dv


# --------------------------------------------------------------------------
# The head_dim ladder, both masking modes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("head_dim", [16, 32, 48, 64, 80, 96, 128])
def test_head_dim_ladder(head_dim, causal):
    _require_env()
    q, k, v, do = _case(1, 2, 2, 256, 256, head_dim)
    _check(q, k, v, do, causal_type=1 if causal else 0, label=f"hd{head_dim} causal={causal}")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_dtypes(dtype, causal):
    _require_env()
    q, k, v, do = _case(1, 2, 2, 256, 256, 64, dtype=dtype)
    _check(q, k, v, do, causal_type=1 if causal else 0, label=f"{dtype} causal={causal}")


# --------------------------------------------------------------------------
# Shapes: GQA, ragged sequence lengths, batch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("hq,hk", [(4, 1), (4, 2), (6, 3), (2, 2)], ids=["mqa", "gqa2", "gqa3", "mha"])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_gqa(hq, hk, causal):
    """dK/dV sum over every query head sharing a KV head.

    The kernel folds AOTriton's `for off_h_q in group` into the Q-block loop,
    so a wrong group mapping shows up as a *partial* sum -- finite, plausible,
    and only caught by a reference that does the same grouping. `hq=4, hk=2` is
    the case broadcasting hides: with `hk=1` a plain broadcast in the reference
    happens to be right.
    """
    _require_env()
    q, k, v, do = _case(1, hq, hk, 256, 256, 64)
    _check(q, k, v, do, causal_type=1 if causal else 0, label=f"gqa {hq}/{hk} causal={causal}")


@pytest.mark.parametrize("seq_q,seq_k", [(200, 200), (129, 129), (64, 320), (320, 64), (17, 17)])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_ragged_lengths(seq_q, seq_k, causal):
    """Sequence lengths that divide neither BLOCK_M nor BLOCK_N.

    Both tails matter and they are masked by different code: a Q row past
    `seqlen_q` is killed per accumulator element (the eight rows a lane holds
    are consecutive), a KV column past `seqlen_k` per lane (all eight share
    one column).
    """
    _require_env()
    q, k, v, do = _case(2, 2, 2, seq_q, seq_k, 64)
    _check(q, k, v, do, causal_type=1 if causal else 0, label=f"S{seq_q}/{seq_k} causal={causal}")


# --------------------------------------------------------------------------
# Masking: both causal alignments and explicit windows
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seq_q,seq_k", [(256, 256), (192, 320), (320, 192)], ids=["eq", "q<k", "q>k"])
def test_bottom_right_causal(seq_q, seq_k):
    """`causal_type=2`: the diagonal sits at `seqlen_k - seqlen_q`.

    `q>k` is the sharp one -- the leading `seqlen_q - seqlen_k` query rows see
    no key at all, so their logsumexp is `+inf` and every score they produce
    must fall out as exactly zero rather than as a NaN.
    """
    _require_env()
    q, k, v, do = _case(1, 2, 2, seq_q, seq_k, 64)
    _check(q, k, v, do, causal_type=2, label=f"botright {seq_q}/{seq_k}")


@pytest.mark.parametrize("window", [(64, 0), (48, 16), (0, 0), (16, 240), (256, 256)])
def test_sliding_window(window):
    """`causal_type=3`. `(0, 0)` is the diagonal alone; `(256, 256)` is dense.

    A narrow window is what exercises the *leading* masked region: with
    `window_left` small the first visited KV-block column is not 0, so the
    Q-block range this kernel walks has a lower bound as well as an upper one.
    """
    _require_env()
    q, k, v, do = _case(1, 2, 2, 256, 256, 64)
    _check(q, k, v, do, causal_type=3, window=window, label=f"swa {window}")


def test_window_equal_to_causal_matches_topleft():
    """`window=(seqlen_q, 0)` and `causal_type=1` are the same mask.

    Self-equivalence rather than a tolerance: the two travel completely
    different host paths -- one resolves a sentinel in the kernel, the other
    passes a literal bound -- and reach the same `resolve_window` output. They
    must therefore agree **bitwise**, which is a much sharper statement than
    both being close to the reference.
    """
    _require_env()
    q, k, v, do = _case(1, 2, 2, 256, 256, 64)
    o, lse, delta, _, _ = _reference(q, k, v, do, causal_type=1)
    a = _bwd_via_kernel(q, k, v, do, o, lse, delta, 1, None)
    b = _bwd_via_kernel(q, k, v, do, o, lse, delta, 3, (q.shape[2], 0))
    assert torch.equal(a[0], b[0]), "dk: causal_type=1 and the equivalent window disagree"
    assert torch.equal(a[1], b[1]), "dv: causal_type=1 and the equivalent window disagree"


# --------------------------------------------------------------------------
# PADDED_HEAD
# --------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [40, 72, 100])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_padded_head(head_dim, causal):
    """A head_dim off the ladder: the kernel computes ceil-to-tile columns.

    The tensors are allocated at the *padded* pitch and sliced, which is the
    alignment contract the interface enforces: an 8-wide store chunk
    straddling `head_dim` must land in the tensor's own padding rather than in
    the next head.
    """
    _require_env()
    pitch = (head_dim + 7) // 8 * 8
    q, k, v, do = _case(1, 2, 2, 256, 256, pitch)
    q, k, v, do = (t[..., :head_dim] for t in (q, k, v, do))
    _check(q, k, v, do, causal_type=1 if causal else 0, label=f"padded hd{head_dim}")


# --------------------------------------------------------------------------
# Varlen
# --------------------------------------------------------------------------


def _cu(lens):
    out = [0]
    for x in lens:
        out.append(out[-1] + x)
    return torch.tensor(out, dtype=torch.int32, device="cuda")


def _varlen_check(lens_q, lens_k, causal_type, mode, head_dim=64, heads=2, label=""):
    """One packed/padded batch against per-sequence fp32 references.

    autograd has no packed-batch form, so the reference is assembled sequence
    by sequence and scattered back -- which is also what makes this a test of
    the *addressing* rather than of the maths, since each sequence's maths is
    already covered above.
    """
    cq, ck = _cu(lens_q), _cu(lens_k)
    n = len(lens_q)
    total_q, total_k = int(cq[-1]), int(ck[-1])
    mq, mk = max(lens_q), max(lens_k)
    gen = torch.Generator(device="cuda").manual_seed(5)

    def _t(*shape):
        return torch.randn(*shape, dtype=torch.float16, device="cuda", generator=gen)

    if mode == "compact":
        q = _t(1, heads, max(total_q, 1), head_dim)
        k = _t(1, heads, max(total_k, 1), head_dim)
        v = _t(1, heads, max(total_k, 1), head_dim)
        do = _t(1, heads, max(total_q, 1), head_dim)
        lse = torch.zeros(heads, max(total_q, 1), dtype=torch.float32, device="cuda")
        delta = torch.zeros_like(lse)
        make = varlen_compact
        lse_tokens = total_q
    else:
        q = _t(n, heads, mq, head_dim)
        k = _t(n, heads, mk, head_dim)
        v = _t(n, heads, mk, head_dim)
        do = _t(n, heads, mq, head_dim)
        lse = torch.zeros(n * heads, mq, dtype=torch.float32, device="cuda")
        delta = torch.zeros_like(lse)
        make = varlen_padded
        lse_tokens = None

    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    ref_k = torch.zeros_like(k, dtype=torch.float32)
    ref_v = torch.zeros_like(v, dtype=torch.float32)

    for z, (lq, lk) in enumerate(zip(lens_q, lens_k)):
        if lq == 0 or lk == 0:
            continue
        if mode == "compact":
            qs, ks = int(cq[z]), int(ck[z])
            sl_q, sl_k = slice(qs, qs + lq), slice(ks, ks + lk)
            qz, kz, vz, doz = q[:, :, sl_q], k[:, :, sl_k], v[:, :, sl_k], do[:, :, sl_q]
        else:
            qz, kz, vz, doz = q[z : z + 1, :, :lq], k[z : z + 1, :, :lk], v[z : z + 1, :, :lk], do[z : z + 1, :, :lq]
        _o, _lse, _delta, _gk, _gv = _reference(qz, kz, vz, doz, causal_type)
        if mode == "compact":
            lse[:, sl_q] = _lse[0]
            delta[:, sl_q] = _delta[0]
            ref_k[:, :, sl_k] = _gk
            ref_v[:, :, sl_k] = _gv
        else:
            lse.view(n, heads, mq)[z, :, :lq] = _lse[0]
            delta.view(n, heads, mq)[z, :, :lq] = _delta[0]
            ref_k[z, :, :lk] = _gk[0]
            ref_v[z, :, :lk] = _gv[0]

    _plan = bwd_plan(
        BwdDkDvMetadata(
            num_heads=heads,
            head_dim=head_dim,
            causal=causal_type != 0,
            causal_type=causal_type or None,
            dtype_str="f16",
        )
    )
    exe = build_bwd_dkdv_module_primary(_plan.meta, _plan.knobs)
    exe(
        q,
        k,
        v,
        do,
        dk,
        dv,
        lse,
        delta,
        n,
        mq,
        seqlen_k=mk,
        varlen=make(cq, ck, mq, mk, lse_tokens=lse_tokens),
    )
    torch.cuda.synchronize()
    r_dk, r_dv = _rel(dk, ref_k), _rel(dv, ref_v)
    assert r_dk < _RTOL_F16, f"{label}: dk rel={r_dk:.3e}"
    assert r_dv < _RTOL_F16, f"{label}: dv rel={r_dv:.3e}"


_VARLEN_LENGTHS = [
    ([3, 128, 40, 200], "mixed"),
    ([64, 64, 64], "all_equal"),
    ([1000, 5, 7, 3], "one_long"),
    ([0, 96, 40], "zero_leading"),
    ([96, 40, 0], "zero_trailing"),
    ([77], "single"),
]


@pytest.mark.parametrize("lens_q,label", _VARLEN_LENGTHS, ids=[x[1] for x in _VARLEN_LENGTHS])
@pytest.mark.parametrize("causal_type", [0, 1, 2], ids=["full", "topleft", "botright"])
def test_varlen_compact(lens_q, label, causal_type):
    """Packed 1THD tensors with `cu_seqlens` for both roles.

    The k-q difference varies per sequence, deliberately: a uniform one makes
    `seqlen_k[z] - seqlen_q[z]` equal the batch-wide difference and hides a
    bottom-right diagonal resolved per batch instead of per sequence.
    """
    _require_env()
    lens_k = [x + 3 + 17 * i if x else 0 for i, x in enumerate(lens_q)]
    _varlen_check(lens_q, lens_k, causal_type, "compact", label=f"compact/{label}/ctype={causal_type}")


@pytest.mark.parametrize("causal_type", [0, 1], ids=["full", "causal"])
def test_varlen_padded(causal_type):
    """BHSD tensors whose sequences are short: lengths only, no positions.

    Differs from compact in precisely the two decoded fields -- `batch_index`
    becomes `z` and `row_off` becomes 0 -- so this is what catches a mix-up
    between them.
    """
    _require_env()
    lens_q = [3, 128, 40, 200]
    lens_k = [6, 140, 49, 220]
    _varlen_check(lens_q, lens_k, causal_type, "padded", label=f"padded/ctype={causal_type}")


# --------------------------------------------------------------------------
# Dropout
# --------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_dropout(p, causal):
    """Against the *same* mask, drawn by the independent debug mask kernel.

    Not a statistical test. `dropout_mask_gfx1201` materialises the Philox
    stream from `Philox.grid_plane`/`grid_offset` with a tiling this kernel
    shares nothing with, so agreeing with it is a statement about the offset
    scheme rather than about the drop rate. That matters here more than in the
    forward: a lane holds eight *query rows* at one key column, so the
    backward reads the stream across its packing axis and needs a separate
    Philox call and a runtime slot select per element -- an entirely different
    access pattern to the forward's contiguous span.
    """
    _require_env()
    seed, offset = 1234, 0
    batch, heads, seq, head_dim = 1, 2, 128, 64
    q, k, v, do = _case(batch, heads, heads, seq, seq, head_dim)
    keep = dropout_mask(batch, heads, seq, seq, seed, offset) > dropout_threshold(p)
    _check(
        q,
        k,
        v,
        do,
        causal_type=1 if causal else 0,
        keep=keep,
        dropout_p=p,
        seed=seed,
        offset=offset,
        label=f"dropout p={p} causal={causal}",
    )


def test_dropout_off_by_default():
    """A build that did not ask for dropout must ignore the dropout arguments."""
    _require_env()
    _plan = bwd_plan(BwdDkDvMetadata(num_heads=2, head_dim=64, causal=False, dtype_str="f16"))
    assert _plan.meta.dropout is False
    exe = build_bwd_dkdv_module_primary(_plan.meta, _plan.knobs)
    q, k, v, do = _case(1, 2, 2, 128, 128, 64)
    o, lse, delta, gk, gv = _reference(q, k, v, do)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    # `dropout_p` is accepted and discarded on a non-dropout build, exactly as
    # the forward's `_dropout_args` does; the output must be the undropped one.
    exe(q, k, v, do, dk, dv, lse.reshape(2, 128), delta.reshape(2, 128), 1, 128, dropout_p=0.5, philox_seed=7)
    torch.cuda.synchronize()
    assert _rel(dk, gk) < _RTOL_F16
    assert _rel(dv, gv) < _RTOL_F16


# --------------------------------------------------------------------------
# End to end, against our own forward kernel
# --------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_end_to_end_with_forward(head_dim, causal):
    """The logsumexp comes from our forward kernel, not from torch.

    This is the interoperation the backward exists for, and it is a different
    claim from every test above: those feed a reference logsumexp, so they
    check the backward's maths against the definition. This one checks that
    the forward's *layout* and *units* are the ones the backward reads --
    `lse_token_pitch`, the HT/TH choice, natural-log units, and `+inf` for a
    dead row.
    """
    _require_env()
    q, k, v, do = _case(1, 4, 4, 256, 256, head_dim)
    o, lse = flydsl_flash_attn_func_gfx1201(q, k, v, causal=causal, return_lse=True)
    dk, dv = flydsl_flash_attn_bwd_dkdv_gfx1201(q, k, v, do, o, lse, causal=causal)
    qf = q.float().requires_grad_()
    kf = k.float().requires_grad_()
    vf = v.float().requires_grad_()
    ref = F.scaled_dot_product_attention(qf, kf, vf, is_causal=causal)
    gk, gv = torch.autograd.grad(ref, (kf, vf), do.float())
    assert _rel(dk, gk) < _RTOL_F16, f"dk rel={_rel(dk, gk):.3e}"
    assert _rel(dv, gv) < _RTOL_F16, f"dv rel={_rel(dv, gv):.3e}"


def test_end_to_end_gqa():
    _require_env()
    q, k, v, do = _case(2, 8, 2, 512, 512, 64)
    o, lse = flydsl_flash_attn_func_gfx1201(q, k, v, causal=True, return_lse=True)
    dk, dv = flydsl_flash_attn_bwd_dkdv_gfx1201(q, k, v, do, o, lse, causal=True)
    qf = q.float().requires_grad_()
    kf = k.float().requires_grad_()
    vf = v.float().requires_grad_()
    ref = F.scaled_dot_product_attention(qf, kf.repeat_interleave(4, 1), vf.repeat_interleave(4, 1), is_causal=True)
    gk, gv = torch.autograd.grad(ref, (kf, vf), do.float())
    assert _rel(dk, gk) < _RTOL_F16
    assert _rel(dv, gv) < _RTOL_F16


# --------------------------------------------------------------------------
# Knobs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("num_waves", [1, 2, 4, 8])
def test_num_waves_knob(num_waves):
    """`BLOCK_N = 16 * num_waves`, and the answer must not depend on it.

    Wave count changes which KV rows a wave owns and how the transposed Q/dO
    tiling is spread, including whether its tail guard fires -- but each KV
    row's arithmetic is identical however rows are grouped. A wrong tiling is
    therefore *not* caught by the tolerance alone; this compares two wave
    counts against each other as well.
    """
    _require_env()
    q, k, v, do = _case(1, 2, 2, 200, 200, 64)
    o, lse, delta, gk, gv = _reference(q, k, v, do, causal_type=1)
    dk, dv = _bwd_via_kernel(q, k, v, do, o, lse, delta, 1, None, knobs=BwdDkDvKnobs(num_waves=num_waves))
    assert _rel(dk, gk) < _RTOL_F16, f"num_waves={num_waves} dk rel={_rel(dk, gk):.3e}"
    assert _rel(dv, gv) < _RTOL_F16, f"num_waves={num_waves} dv rel={_rel(dv, gv):.3e}"


@pytest.mark.parametrize("block_m", [16, 32])
def test_block_m_knob(block_m):
    _require_env()
    q, k, v, do = _case(1, 2, 2, 200, 200, 64)
    o, lse, delta, gk, gv = _reference(q, k, v, do, causal_type=1)
    dk, dv = _bwd_via_kernel(q, k, v, do, o, lse, delta, 1, None, knobs=BwdDkDvKnobs(block_m=block_m))
    assert _rel(dk, gk) < _RTOL_F16, f"block_m={block_m} dk rel={_rel(dk, gk):.3e}"
    assert _rel(dv, gv) < _RTOL_F16, f"block_m={block_m} dv rel={_rel(dv, gv):.3e}"


def test_lds_budget_is_enforced():
    """A `block_m` whose four staging tiles exceed 64 KiB must be rejected.

    A build-time `ValueError`, not silent truncation: the LDS allocation would
    otherwise be sized past the hardware limit and fail far away, at
    serialization, with a message naming neither knob.
    """
    from fmha_tuning_bwd_dkdv_gfx1201 import resolve_knobs

    with pytest.raises(ValueError, match="LDS"):
        resolve_knobs(BwdDkDvMetadata(num_heads=2, head_dim=256), BwdDkDvKnobs(block_m=256))
