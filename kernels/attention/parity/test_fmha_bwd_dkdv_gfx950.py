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

import pytest
import torch
import torch.nn.functional as F
from fmha_bwd_dkdv_gfx950 import build_fmha_bwd_dkdv_gfx950_module as build
from fmha_tuning_bwd_dkdv_gfx950 import LADDER
from gfx950_standalone import dualwave  # noqa: F401  (puts the repo root on sys.path)
from torch.nn.attention import SDPBackend, sdpa_kernel

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


def _rand(*shape, dtype=DT):
    return torch.randn(*shape, device="cuda", dtype=dtype)


def _math_backward(q, k, v, do, scale, dtype):
    """`(o, lse, delta, dq, dk, dv)` from torch's math backend at `dtype`.

    `lse` and `delta` are computed alongside rather than taken from the
    autograd graph, because this kernel consumes them as *inputs* and the whole
    point of the high-precision arm is that everything it hands over is exact.
    """
    qq, kk, vv = (t.to(dtype).detach().requires_grad_() for t in (q, k, v))
    with sdpa_kernel(SDPBackend.MATH):
        o = F.scaled_dot_product_attention(qq, kk, vv, scale=scale)
    dq, dk, dv = torch.autograd.grad(o, (qq, kk, vv), do.to(dtype))
    lse = torch.logsumexp((qq @ kk.transpose(-1, -2)).detach() * scale, dim=-1)
    delta = (do.to(dtype) * o.detach()).sum(-1)
    return o.detach(), lse, delta, dq, dk, dv


def _run(q, k, v, do, lse, delta, *, scale, dk=None, dv=None, batch=None):
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
    )
    fn(q, k, v, do, dk, dv, lse, delta, b if batch is None else batch, sq, seqlen_k=sk, scale=scale)
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


def _ratio_check(b, h, sq, sk, d, scale=None):
    """The plan section 7.1 gate, and the numbers it measured, for one shape."""
    torch.manual_seed(hash((b, h, sq, sk, d)) & 0xFFFF)
    q, k, v, do = _rand(b, h, sq, d), _rand(b, h, sk, d), _rand(b, h, sk, d), _rand(b, h, sq, d)
    scale = 1.0 / d**0.5 if scale is None else scale

    _o64, lse64, delta64, _dq64, dk64, dv64 = _math_backward(q, k, v, do, scale, torch.float64)
    _o16, _l16, _d16, _dq16, dk16, dv16 = _math_backward(q, k, v, do, scale, torch.bfloat16)

    lse = lse64.reshape(b * h, sq).float().contiguous()
    delta = delta64.reshape(b * h, sq).float().contiguous()
    dk, dv = _run(q, k, v, do, lse, delta, scale=scale)

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
def test_matches_math_backend(shape, head_dim):
    _require_rocm_path()
    _ratio_check(*shape, head_dim)


@pytest.mark.parametrize("head_dim", LADDER)
def test_ladder_matches_math_backend(head_dim):
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
def test_ragged_and_asymmetric_sequences(sq, sk, head_dim):
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
def test_scale_sweep(scale, head_dim):
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
def test_consistent_with_the_gfx950_forward(head_dim):
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
    fwd = build_fwd(num_heads=h, head_dim=d, num_kv_heads=h, causal=False, dtype_str="bf16", return_lse=True)
    fwd(q, k, v, o, b, s, seqlen_k=s, scale=scale, lse=lse)
    torch.cuda.synchronize()

    delta = (do.float() * o.float()).sum(-1).reshape(b * h, s).contiguous()
    dk, dv = _run(q, k, v, do, lse, delta, scale=scale)

    _o64, _lse64, _delta64, _dq64, dk64, dv64 = _math_backward(q, k, v, do, scale, torch.float64)
    _o16, _l16, _d16, _dq16, dk16, dv16 = _math_backward(q, k, v, do, scale, torch.bfloat16)
    for name, ours, low, high in (("dk", dk, dk16, dk64), ("dv", dv, dv16, dv64)):
        e_ours, e_low = _rel(ours, high), _rel(low, high)
        assert e_ours <= FUDGE[name] * e_low, (
            f"{name} from our forward's O/LSE: {e_ours:.3e} against the bf16 backend's {e_low:.3e}. "
            "The two halves disagree about a convention rather than about precision."
        )


@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
def test_joint_with_dq_against_one_autograd_call(head_dim):
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
    _o16, _l16, _d16, dq16, dk16, dv16 = _math_backward(q, k, v, do, scale, torch.bfloat16)

    lse = lse64.reshape(b * h, s).float().contiguous()
    delta = delta64.reshape(b * h, s).float().contiguous()
    dk, dv = _run(q, k, v, do, lse, delta, scale=scale)
    dq = torch.full_like(q, float("nan"))
    dq_fn = dq_build(num_heads=h, head_dim=d, num_kv_heads=h)
    dq_fn(q, k, v, do, dq, lse, delta, b, s, seqlen_k=s, scale=scale)
    torch.cuda.synchronize()

    for name, ours, low, high in (("dq", dq, dq16, dq64), ("dk", dk, dk16, dk64), ("dv", dv, dv16, dv64)):
        e_ours, e_low = _rel(ours, high), _rel(low, high)
        assert e_ours <= 3.0 * e_low, f"{name}: {e_ours:.3e} against {e_low:.3e}"


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_dim", [64, 128, 192])
def test_bshd_layout_is_read_through_the_strides(head_dim):
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
def test_writes_nothing_past_the_kv_sequence(head_dim):
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

    dk_full = torch.full((b, h, s + slack, d), 1234.0, device="cuda", dtype=DT)
    dv_full = torch.full((b, h, s + slack, d), 1234.0, device="cuda", dtype=DT)
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
def test_grid8_contiguous_is_exact_and_writes_nothing_past_the_last_row(hdim):
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
    _o2, _l, _d, _dq2, dk16, dv16 = _math_backward(q, k, v, do, scale, torch.bfloat16)
    lse = lse64.reshape(b * h, s).float().contiguous()
    delta = delta64.reshape(b * h, s).float().contiguous()

    sentinel = -12345.0
    dkbuf = torch.full((b, h, s + 1, hdim), sentinel, device="cuda", dtype=DT)
    dvbuf = torch.full((b, h, s + 1, hdim), sentinel, device="cuda", dtype=DT)
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
def test_padded_head_never_writes_past_the_8_aligned_chunk(hdim, pitch):
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
        full = torch.randn(b, h, rows, pitch, device="cuda", dtype=DT)
        return full, full[..., :hdim]

    _fq, q = _padded()
    _fk, k = _padded()
    _fv, v = _padded()
    _fdo, do = _padded()
    scale = 1.0 / hdim**0.5
    _o, lse64, delta64, _dq, dk64, dv64 = _math_backward(q, k, v, do, scale, torch.float64)
    _o2, _l, _d, _dq2, dk16, dv16 = _math_backward(q, k, v, do, scale, torch.bfloat16)
    lse = lse64.reshape(b * h, s).float().contiguous()
    delta = delta64.reshape(b * h, s).float().contiguous()

    dkfull = torch.full((b, h, s, pitch), -7.0, device="cuda", dtype=DT)
    dvfull = torch.full((b, h, s, pitch), -7.0, device="cuda", dtype=DT)
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
    q, k, v, do = (torch.randn(b, s, h, hdim, device="cuda", dtype=DT).transpose(1, 2) for _ in range(4))
    dk = torch.empty(b, s, h, hdim, device="cuda", dtype=DT).transpose(1, 2)
    dv = torch.empty(b, s, h, hdim, device="cuda", dtype=DT).transpose(1, 2)
    lse = torch.zeros(b * h, s, device="cuda", dtype=torch.float32)
    assert q.stride(2) % 8 == 0, "the pitch check would pass here"
    fn = build(num_heads=h, head_dim=hdim)
    with pytest.raises(ValueError, match="unused element"):
        fn(q, k, v, do, dk, dv, lse, lse, b, s)


@pytest.mark.parametrize("head_dim", [64, 512])
def test_deterministic(head_dim):
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
    with pytest.raises(NotImplementedError, match="causal"):
        build(num_heads=8, head_dim=64, causal=True)
    with pytest.raises(NotImplementedError, match="dropout"):
        build(num_heads=8, head_dim=64, dropout=True)
    with pytest.raises(NotImplementedError, match="GQA"):
        build(num_heads=8, head_dim=64, num_kv_heads=2)
    with pytest.raises(ValueError, match="exceeds the widest tile"):
        build(num_heads=8, head_dim=513)
    with pytest.raises(NotImplementedError, match="bf16"):
        build(num_heads=8, head_dim=64, dtype_str="f16")


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
