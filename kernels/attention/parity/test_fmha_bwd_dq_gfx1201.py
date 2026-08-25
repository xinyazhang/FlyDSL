# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness tests for the gfx1201 backward-dQ kernel.

Three kinds of bar, sharpest last:

1. **Against torch autograd.** ``torch.autograd.grad`` on an fp32
   reimplementation of attention, with the *same* fp32 ``lse`` and ``delta``
   fed to the kernel. This isolates dQ: any error is this kernel's, not the
   forward's.
2. **Against the real forward kernel.** The same comparison, but ``lse`` and
   ``O`` come from ``flash_attn_func_gfx1201_aiw``. This is what says the two
   kernels agree about the logsumexp *layout* and its scaling convention,
   which a torch-supplied ``lse`` cannot test.
3. **Self-equivalence, bitwise.** ``kt_lds_layout`` "scalar" against
   "transposed" must be bit-identical -- they are the same values multiplied
   in the same order and differ only in how K reaches the WMMA operand. And
   varlen with N sequences must equal N separate dense calls, bit for bit, for
   the reason the forward's suite gives: an addressing bug that lands inside
   the right allocation reads *plausible* data, and a tolerance comparison
   accepts it.

Run it individually, per this directory's prototype convention::

    export ROCM_PATH=$(rocm-sdk path --root)
    cd kernels/attention/parity && python3 -m pytest test_fmha_bwd_dq_gfx1201.py -v
"""

import math
import os

import pytest
import torch
from dropout_mask_gfx1201 import dropout_mask
from flash_attn_func_gfx1201_aiw import build_flash_attn_func_aiw_module
from fmha_bwd_dq_gfx1201_interface import bwd_dq_delta, flydsl_bwd_dq_gfx1201
from fmha_bwd_dq_gfx1201_kernel import build_bwd_dq_module
from fmha_tuning_bwd_dq_gfx1201 import BwdDqInputMetadata, BwdDqKnobs
from fmha_tuning_bwd_dq_gfx1201 import plan as bwd_dq_plan
from philox import dropout_threshold

_NUM_HEADS = 2

# Tolerance for dQ against an fp32 reference, as max|delta| / max|reference|.
#
# dQ is a longer chain than O -- three GEMMs, one of which takes a
# 16-bit-truncated dS operand -- so it is looser than the forward's 5e-3.
# Measured over the whole ladder at seqlen 192, both masking modes:
#
#   f16    2.3e-4 .. 6.4e-4
#   bf16   3.1e-3 .. 8.4e-3
#
# The bf16 gap is the dS truncation, the same round-toward-zero pack the
# forward uses for P (`fmha.bf16_trunc_pack_v8`, which records why it is not
# round-to-nearest). 2e-2 is the brief's bar; it leaves a factor of 2 over the
# worst measurement and still rejects anything structurally wrong, which lands
# at O(1) -- the K^T-tile overwrite this suite caught during bring-up scored
# 0.6 here.
_RTOL = 2e-2

# Below this the *reference* is degenerate and a relative comparison measures
# only fp16 noise. It happens for real: when every row sees at most one key --
# a 1x1 problem, or a window of exactly the diagonal -- the softmax is
# identically 1, dP equals delta exactly, and dQ is analytically **zero**. Both
# sides then compute a difference of two roundings of the same dot product, and
# their ratio is unbounded while both are around 1e-4. dQ in every
# non-degenerate case here runs 0.1 to 10, so a floor three orders below that
# turns those cases into an absolute check without loosening any other.
_NOISE_FLOOR = 1e-2


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


def _rel(got, ref):
    """max|delta| against the reference's own scale, floored -- see `_NOISE_FLOOR`."""
    return (got.float() - ref).abs().max().item() / max(ref.abs().max().item(), _NOISE_FLOOR)


def _causal_bias(nq, nk, ctype, device):
    """Additive mask for AOTriton's CAUSAL_TYPE 1 (top-left) / 2 (bottom-right)."""
    i = torch.arange(nq, device=device)[:, None]
    j = torch.arange(nk, device=device)[None, :]
    diag = 0 if ctype == 1 else (nk - nq)
    return torch.where(j <= i + diag, 0.0, float("-inf"))


def _fp32_forward(q, k, v, sm_scale, ctype, requires_grad=False):
    """(dq_ref-capable graph, lse, o) in fp32, with GQA expanded.

    One function for both the reference gradient and the ``lse``/``delta`` the
    kernel is fed, so the two cannot disagree about the masking or the scale.
    """
    nq, nk = q.shape[2], k.shape[2]
    rep = q.shape[1] // k.shape[1]
    kf = k.float().repeat_interleave(rep, dim=1)
    vf = v.float().repeat_interleave(rep, dim=1)
    qf = q.float()
    if requires_grad:
        qf = qf.clone().requires_grad_(True)
    s = qf @ kf.transpose(-1, -2) * sm_scale
    if ctype:
        s = s + _causal_bias(nq, nk, ctype, q.device)
    return qf, s, vf


def _reference(q, k, v, do, sm_scale, ctype):
    """(dq, lse, delta) -- the fp32 answer and the two row tensors it implies."""
    qg, s, vf = _fp32_forward(q, k, v, sm_scale, ctype, requires_grad=True)
    og = torch.softmax(s, dim=-1) @ vf
    (dq_ref,) = torch.autograd.grad(og, qg, do.float())
    with torch.no_grad():
        _, s2, vf2 = _fp32_forward(q, k, v, sm_scale, ctype)
        lse = torch.logsumexp(s2, dim=-1)
        o = torch.softmax(s2, dim=-1) @ vf2
        delta = (do.float() * o).sum(-1)
    return dq_ref, lse, delta


def _qkv(batch, nhq, nhk, nq, nk, head_dim, dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def _t(h, n):
        return torch.randn(batch, h, n, head_dim, dtype=dtype, device="cuda", generator=gen)

    return _t(nhq, nq), _t(nhk, nk), _t(nhk, nk), _t(nhq, nq)


def _run(q, k, v, do, lse, delta, causal, ctype=None, dtype_str="f16", sm_scale=None, **knobs):
    """Build and launch the kernel directly, bypassing the interface."""
    b, nhq, nq, d = q.shape
    nk = k.shape[2]
    sm_scale = 1.0 / math.sqrt(d) if sm_scale is None else sm_scale
    exe = build_bwd_dq_module(
        num_heads=nhq,
        head_dim=d,
        causal=causal,
        causal_type=ctype,
        dtype_str=dtype_str,
        sm_scale=sm_scale,
        **knobs,
    )
    dq = torch.zeros_like(q)
    exe(
        q,
        k,
        v,
        do,
        dq,
        lse.reshape(b * nhq, nq).contiguous(),
        delta.reshape(b * nhq, nq).contiguous(),
        b,
        nq,
        nk,
    )
    torch.cuda.synchronize()
    return dq


# ---------------------------------------------------------------------------
# 1. Against torch autograd, with an fp32 lse/delta
# ---------------------------------------------------------------------------

# The compiled tile widths this kernel ships. dQ carries three
# head_dim-proportional register sets (Q, dO, dQ) where the forward carries
# two, so the top two rungs run head-dim *sharded* -- four waves cooperating on
# one Q row sub-tile, each holding a quarter of all three sets and reducing the
# partial S and dP through LDS. 384 and 512 are the only widths that shard.
_LADDER = [16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 384, 512]


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("head_dim", _LADDER)
def test_ladder_matches_autograd(head_dim, causal):
    """Every supported head_dim matches an fp32 autograd reference."""
    _require_env()
    q, k, v, do = _qkv(1, _NUM_HEADS, _NUM_HEADS, 192, 192, head_dim, torch.float16)
    dq_ref, lse, delta = _reference(q, k, v, do, 1.0 / math.sqrt(head_dim), 1 if causal else 0)
    dq = _run(q, k, v, do, lse, delta, causal)
    assert torch.isfinite(dq).all(), f"head_dim={head_dim} causal={causal} not finite"
    rel = _rel(dq, dq_ref)
    assert rel < _RTOL, f"head_dim={head_dim} causal={causal} rel={rel:.3e}"


@pytest.mark.parametrize("dtype,dtype_str", [(torch.float16, "f16"), (torch.bfloat16, "bf16")], ids=["f16", "bf16"])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_dtypes(dtype, dtype_str, causal):
    """Both 16-bit input types. bf16 loses more, because dS is truncated into
    a bf16 WMMA operand -- see `fmha.bf16_trunc_pack_v8`."""
    _require_env()
    q, k, v, do = _qkv(1, _NUM_HEADS, _NUM_HEADS, 128, 128, 64, dtype)
    dq_ref, lse, delta = _reference(q, k, v, do, 1.0 / 8.0, 1 if causal else 0)
    dq = _run(q, k, v, do, lse, delta, causal, dtype_str=dtype_str)
    rel = _rel(dq, dq_ref)
    assert rel < _RTOL, f"{dtype_str} causal={causal} rel={rel:.3e}"


@pytest.mark.parametrize("nhq,nhk", [(8, 8), (8, 4), (8, 1)], ids=["mha", "gqa4", "mqa"])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_gqa(nhq, nhk, causal):
    """Several query heads may share one KV head; K/V then carry num_head_k."""
    _require_env()
    q, k, v, do = _qkv(2, nhq, nhk, 128, 128, 64, torch.float16)
    dq_ref, lse, delta = _reference(q, k, v, do, 1.0 / 8.0, 1 if causal else 0)
    exe = build_bwd_dq_module(num_heads=nhq, head_dim=64, causal=causal, dtype_str="f16")
    dq = torch.zeros_like(q)
    exe(
        q,
        k,
        v,
        do,
        dq,
        lse.reshape(2 * nhq, 128).contiguous(),
        delta.reshape(2 * nhq, 128).contiguous(),
        2,
        128,
        128,
    )
    torch.cuda.synchronize()
    rel = _rel(dq, dq_ref)
    assert rel < _RTOL, f"nhq={nhq} nhk={nhk} causal={causal} rel={rel:.3e}"


# `seqlen % BLOCK_M != 0` on the Q side leaves a partial workgroup whose rows
# run past the sequence; `seqlen_k % BLOCK_N != 0` leaves a ragged final KV
# tile. 100 and 77 are coprime to both 64 (BLOCK_M) and 32 (BLOCK_N), and 1
# exercises the degenerate tile.
_RAGGED = [(100, 100), (100, 77), (77, 100), (65, 33), (1, 1), (129, 257)]


@pytest.mark.parametrize("nq,nk", _RAGGED, ids=[f"{a}x{b}" for a, b in _RAGGED])
@pytest.mark.parametrize("ctype", [0, 1, 2], ids=["full", "topleft", "botright"])
def test_ragged_lengths(nq, nk, ctype):
    """Sequence lengths that divide neither BLOCK_M nor BLOCK_N.

    ``ctype`` 2 (bottom-right) is the interesting one here: with nq != nk its
    diagonal sits at ``nk - nq``, which is negative for ``77x100`` reversed and
    so leaves whole leading Q rows with no visible key at all. Those rows get
    ``+inf`` from the forward's logsumexp convention and must produce dQ = 0
    rather than NaN.
    """
    _require_env()
    q, k, v, do = _qkv(1, _NUM_HEADS, _NUM_HEADS, nq, nk, 64, torch.float16)
    dq_ref, lse, delta = _reference(q, k, v, do, 1.0 / 8.0, ctype)
    # A fully masked row has lse = -inf in torch; the forward kernel writes
    # +inf there instead, deliberately, so that exp2(x - lse) is 0. Match the
    # kernel's convention, which is what it will actually be fed.
    lse = torch.where(torch.isinf(lse) & (lse < 0), torch.full_like(lse, float("inf")), lse)
    delta = torch.nan_to_num(delta, nan=0.0)
    dq = _run(q, k, v, do, lse, delta, bool(ctype), ctype=ctype or None)
    assert torch.isfinite(dq).all(), f"{nq}x{nk} ctype={ctype} not finite"
    rel = _rel(dq, torch.nan_to_num(dq_ref, nan=0.0))
    assert rel < _RTOL, f"{nq}x{nk} ctype={ctype} rel={rel:.3e}"


@pytest.mark.parametrize("window", [(16, 0), (32, 16), (0, 0), (-8, 64)], ids=["w16_0", "w32_16", "diag", "neg_left"])
def test_sliding_window(window):
    """Generalized sliding-window attention: a two-sided band.

    ``(-8, 64)`` is the sharp case -- a negative left bound pushes the whole
    band right of the diagonal, so the *leading* KV tiles are masked as well as
    the trailing ones and `decompose_causal_regions`'s three-region split is
    doing real work.
    """
    _require_env()
    nq = nk = 128
    wl, wr = window
    q, k, v, do = _qkv(1, _NUM_HEADS, _NUM_HEADS, nq, nk, 64, torch.float16)
    i = torch.arange(nq, device="cuda")[:, None]
    j = torch.arange(nk, device="cuda")[None, :]
    live = (j <= i + wr) & (j >= i - wl)
    bias = torch.where(live, 0.0, float("-inf"))

    qg = q.float().clone().requires_grad_(True)
    s = qg @ k.float().transpose(-1, -2) / 8.0 + bias
    og = torch.softmax(s, dim=-1) @ v.float()
    (dq_ref,) = torch.autograd.grad(og, qg, do.float())
    with torch.no_grad():
        s2 = q.float() @ k.float().transpose(-1, -2) / 8.0 + bias
        lse = torch.logsumexp(s2, dim=-1)
        lse = torch.where(torch.isinf(lse) & (lse < 0), torch.full_like(lse, float("inf")), lse)
        o = torch.nan_to_num(torch.softmax(s2, dim=-1), nan=0.0) @ v.float()
        delta = (do.float() * o).sum(-1)

    exe = build_bwd_dq_module(num_heads=_NUM_HEADS, head_dim=64, causal=True, causal_type=3, dtype_str="f16")
    dq = torch.zeros_like(q)
    exe(
        q,
        k,
        v,
        do,
        dq,
        lse.reshape(_NUM_HEADS, nq).contiguous(),
        delta.reshape(_NUM_HEADS, nq).contiguous(),
        1,
        nq,
        nk,
        window=window,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(dq).all(), f"window={window} not finite"
    rel = _rel(dq, torch.nan_to_num(dq_ref, nan=0.0))
    assert rel < _RTOL, f"window={window} rel={rel:.3e}"


# ---------------------------------------------------------------------------
# 2. Against the real forward kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_end_to_end_with_forward_kernel(head_dim, causal):
    """``lse`` and ``O`` from ``flash_attn_func_gfx1201_aiw``, not from torch.

    This is the only test that exercises the logsumexp *contract* between the
    two kernels: its layout (`lse_token_pitch`), its natural-log units, and the
    ``+inf`` it writes for a row with no live keys. A torch-supplied ``lse``
    would pass even if the two disagreed about all three.
    """
    _require_env()
    B, N = 2, 192
    q, k, v, do = _qkv(B, _NUM_HEADS, _NUM_HEADS, N, N, head_dim, torch.float16, seed=7)
    fwd = build_flash_attn_func_aiw_module(num_heads=_NUM_HEADS, head_dim=head_dim, causal=causal, dtype_str="f16")
    o = torch.empty_like(q)
    lse = torch.zeros(B * _NUM_HEADS, N, dtype=torch.float32, device="cuda")
    fwd(q, k, v, o, B, N, lse=lse)
    torch.cuda.synchronize()

    dq = flydsl_bwd_dq_gfx1201(q, k, v, o, do, lse, causal=causal)
    torch.cuda.synchronize()

    dq_ref, _, _ = _reference(q, k, v, do, 1.0 / math.sqrt(head_dim), 1 if causal else 0)
    assert torch.isfinite(dq).all()
    rel = _rel(dq, dq_ref)
    assert rel < _RTOL, f"head_dim={head_dim} causal={causal} rel={rel:.3e}"


def test_interface_accepts_bhs_lse_and_precomputed_delta():
    """`(B, H, S)` and `(B*H, S)` are the same buffer, and `delta=` skips `o`."""
    _require_env()
    B, N, D = 2, 128, 64
    q, k, v, do = _qkv(B, _NUM_HEADS, _NUM_HEADS, N, N, D, torch.float16, seed=11)
    dq_ref, lse, _ = _reference(q, k, v, do, 1.0 / 8.0, 0)
    o = (torch.softmax(q.float() @ k.float().transpose(-1, -2) / 8.0, -1) @ v.float()).half()
    a = flydsl_bwd_dq_gfx1201(q, k, v, o, do, lse)  # (B, H, S) lse, delta derived from o
    # The *same* delta, spelled by hand. Recomputing it from an fp32 `o` would
    # be a different tensor -- `bwd_dq_delta` casts to fp32 from whatever it is
    # given -- and this test is about the two call shapes, not about delta.
    b = flydsl_bwd_dq_gfx1201(
        q,
        k,
        v,
        None,
        do,
        lse.reshape(B * _NUM_HEADS, N).contiguous(),
        delta=bwd_dq_delta(o, do),
    )
    torch.cuda.synchronize()
    assert torch.equal(a, b), "the two spellings of the same inputs disagree"
    assert _rel(a, dq_ref) < _RTOL


def test_delta_helper_matches_the_definition():
    """`bwd_dq_delta` is `rowsum(dO * O)` in fp32, in the logsumexp layout."""
    _require_env()
    o = torch.randn(2, 3, 16, 8, dtype=torch.float16, device="cuda")
    do = torch.randn_like(o)
    got = bwd_dq_delta(o, do)
    want = (o.float() * do.float()).sum(-1).reshape(6, 16)
    assert got.shape == (6, 16) and got.dtype == torch.float32
    assert torch.equal(got, want)


# ---------------------------------------------------------------------------
# 3. Self-equivalence, bitwise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("head_dim", [32, 64, 128, 512])
def test_kt_layouts_are_bitwise_identical(head_dim, causal):
    """ "scalar" and "transposed" must agree bit for bit.

    They multiply the same K values into the same accumulator in the same
    order; only the route from VRAM to the WMMA A-operand differs -- eight
    strided LDS reads out of the row-major tile against one vector read out of
    a hardware-transposed copy. A tolerance test would accept a transpose that
    is off by a lane, which is the failure this catches.

    512 is here for a second reason: it is a *sharded* build, so it also says
    the transposed staging still covers the whole K tile when each wave reads
    only its own head-dim slice of it, and that BLOCK_N 16 tiles the transpose
    correctly.
    """
    _require_env()
    q, k, v, do = _qkv(1, _NUM_HEADS, _NUM_HEADS, 160, 160, head_dim, torch.float16, seed=3)
    _, lse, delta = _reference(q, k, v, do, 1.0 / math.sqrt(head_dim), 1 if causal else 0)
    a = _run(q, k, v, do, lse, delta, causal, kt_lds_layout="scalar")
    b = _run(q, k, v, do, lse, delta, causal, kt_lds_layout="transposed")
    assert torch.equal(a, b), (
        f"kt_lds_layout arms differ at head_dim={head_dim} causal={causal}: "
        f"max |delta| {(a.float() - b.float()).abs().max():.3e}"
    )


@pytest.mark.parametrize("num_waves", [1, 2, 4, 8], ids=lambda w: f"w{w}")
def test_wave_count_is_invisible_to_the_output(num_waves):
    """Non-causal, BLOCK_M is a bitwise no-op.

    Each Q row's arithmetic is identical however rows are grouped into
    workgroups, and non-causally every workgroup walks the same KV tiles in
    the same increasing order, so the accumulation order does not move either.
    Only a benchmark can see BLOCK_M at all -- which is exactly why a wrong
    wave count is worth pinning here.

    **Causal is deliberately not bitwise**, and that is not a weakness in this
    kernel. `decompose_causal_regions` cuts the visited range using BLOCK_M, so
    a different BLOCK_M puts different tiles in the full run versus the masked
    one -- and the loop order is full-then-masked, not increasing in KV. The
    values are the same; the *order* they are summed into dQ is not. Checked
    below with a tolerance instead.
    """
    _require_env()
    q, k, v, do = _qkv(1, _NUM_HEADS, _NUM_HEADS, 160, 160, 64, torch.float16, seed=5)
    _, lse, delta = _reference(q, k, v, do, 1.0 / 8.0, 0)
    ref = _run(q, k, v, do, lse, delta, False, num_waves=4)
    got = _run(q, k, v, do, lse, delta, False, num_waves=num_waves)
    assert torch.equal(ref, got), f"num_waves={num_waves} changed the output"


@pytest.mark.parametrize("num_waves", [1, 2, 8], ids=lambda w: f"w{w}")
def test_wave_count_causal_matches_within_tolerance(num_waves):
    """Causal: BLOCK_M moves the summation order, so compare to autograd."""
    _require_env()
    q, k, v, do = _qkv(1, _NUM_HEADS, _NUM_HEADS, 160, 160, 64, torch.float16, seed=5)
    dq_ref, lse, delta = _reference(q, k, v, do, 1.0 / 8.0, 1)
    got = _run(q, k, v, do, lse, delta, True, num_waves=num_waves)
    rel = _rel(got, dq_ref)
    assert rel < _RTOL, f"num_waves={num_waves} rel={rel:.3e}"


def _cu(lens):
    return torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32, device="cuda")


_VARLEN_LENGTHS = [
    ([3, 128, 40, 200], "mixed"),
    ([64, 64, 64], "all_equal"),
    ([96, 0, 40], "zero_middle"),
    ([77], "single"),
]


# ---------------------------------------------------------------------------
# Distance-1 K/V prefetch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window", [(64, 0), (96, 32), (128, 64)], ids=["wl64", "wl96wr32", "wl128wr64"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_kv_prefetch_matches_distance_zero_across_the_seam(head_dim, window):
    """`kv_prefetch_dist=1` must be bit-identical to 0, *on a windowed shape*.

    The window is the whole point. A distance-1 schedule fetches tile `i+1`
    while computing tile `i`, so it needs the successor of the tile it is on --
    and gSWA's masked region walks two disjoint runs, so the successor is
    piecewise. Get it wrong and the prefetch reads the wrong tile, which is
    **invisible on any contiguous shape** because the value is overwritten
    before use.

    Both bugs this test caught were exactly that. The first version passed
    every existing dQ case at seqlen 256 and was wrong by `rel ~ 1.0` here:
    the masked loop used the contiguous successor. The second seeded the
    pre-loop fetch from the full run's origin, which is wrong whenever there
    are no full tiles -- and `window=(64, 0)` at seqlen 512 is exactly that
    shape.

    Bitwise, not tolerance: the two arms multiply the same values in the same
    order and differ only in when the load is issued.
    """
    _require_env()
    B, H, N = 1, 2, 512
    wl, wr = window
    gen = torch.Generator(device="cuda").manual_seed(7)
    q, k, v, do = (torch.randn(B, H, N, head_dim, dtype=torch.float16, device="cuda", generator=gen) for _ in range(4))
    i = torch.arange(N, device="cuda")[:, None]
    j = torch.arange(N, device="cuda")[None, :]
    band = (j <= i + wr) & (j >= i - wl)

    qg = q.float().clone().requires_grad_(True)
    sg = qg @ k.float().transpose(-1, -2) / math.sqrt(head_dim)
    sg = sg.masked_fill(~band, float("-inf"))
    live = torch.isfinite(sg).any(-1, keepdim=True)
    og = torch.where(live, torch.softmax(sg, -1), torch.zeros_like(sg)) @ v.float()
    (dq_ref,) = torch.autograd.grad(og, qg, do.float())
    with torch.no_grad():
        s = q.float() @ k.float().transpose(-1, -2) / math.sqrt(head_dim)
        s = s.masked_fill(~band, float("-inf"))
        o = torch.where(live, torch.softmax(s, -1), torch.zeros_like(s)) @ v.float()
        lse = torch.logsumexp(s, -1)
        lse = torch.where(live.squeeze(-1), lse, torch.full_like(lse, float("inf")))
        lse = lse.reshape(B * H, N).contiguous()
        delta = (do.float() * o).sum(-1).reshape(B * H, N).contiguous()

    out = {}
    for dist in (0, 1):
        exe = build_bwd_dq_module(
            num_heads=H,
            head_dim=head_dim,
            causal=True,
            causal_type=3,
            dtype_str="f16",
            kv_prefetch_dist=dist,
        )
        dq = torch.zeros_like(q)
        exe(q, k, v, do, dq, lse, delta, B, N, N, window=window)
        torch.cuda.synchronize()
        out[dist] = dq

    assert torch.equal(out[0], out[1]), (
        f"prefetch changed the result at head_dim={head_dim} window={window}: "
        f"max|delta|={(out[1].float() - out[0].float()).abs().max().item():.3e}"
    )
    # And the shared answer is the right one, so the test cannot pass by both
    # arms being broken identically -- which is the failure mode a pure A/B has.
    rel = _rel(out[1], dq_ref)
    assert rel < _RTOL, f"prefetch arm disagrees with autograd: rel={rel:.3e}"


@pytest.mark.parametrize("lens_q,label", _VARLEN_LENGTHS, ids=[x[1] for x in _VARLEN_LENGTHS])
@pytest.mark.parametrize("ctype", [0, 1, 2], ids=["full", "topleft", "botright"])
def test_varlen_compact_matches_dense(lens_q, label, ctype):
    """Packed varlen with N sequences equals N separate dense calls, bitwise.

    That holds for a reason rather than by luck: a varlen workgroup and its
    dense counterpart cover the same tiles in the same order with the same
    values, and only the base address differs. It is the right gate here
    because an addressing bug that lands inside the right allocation reads
    plausible data, which a tolerance comparison accepts.

    The k-q length difference **varies per sequence**, deliberately: a uniform
    difference makes `seqlen_k[z] - seqlen_q[z]` equal the batch-wide value,
    which hides any bottom-right implementation that resolves the diagonal per
    batch instead of per sequence.
    """
    _require_env()
    head_dim = 64
    lens_k = [x + 3 + 17 * i if x else 0 for i, x in enumerate(lens_q)]
    N = len(lens_q)
    cq, ck = _cu(lens_q), _cu(lens_k)
    Tq, Tk = int(cq[-1]), int(ck[-1])
    mq, mk = max(lens_q), max(lens_k)
    gen = torch.Generator(device="cuda").manual_seed(13)

    def _t(n, h=_NUM_HEADS):
        return torch.randn(1, h, max(n, 1), head_dim, dtype=torch.float16, device="cuda", generator=gen)

    q, k, v, do = _t(Tq), _t(Tk), _t(Tk), _t(Tq)
    # `lse` and `delta` are fabricated rather than derived: this test compares
    # two *kernel* launches against each other, so any well-formed pair of row
    # tensors exercises the addressing equally well, and a torch reference
    # would only add its own approximation to both sides.
    lse = torch.randn(_NUM_HEADS, max(Tq, 1), dtype=torch.float32, device="cuda", generator=gen)
    delta = torch.randn(_NUM_HEADS, max(Tq, 1), dtype=torch.float32, device="cuda", generator=gen)

    exe = build_bwd_dq_module(
        num_heads=_NUM_HEADS,
        head_dim=head_dim,
        causal=bool(ctype),
        causal_type=ctype or None,
        dtype_str="f16",
    )
    dq = torch.zeros_like(q)
    exe(
        q,
        k,
        v,
        do,
        dq,
        lse,
        delta,
        1,
        mq,
        mk,
        num_seqlens=N,
        varlen=exe.varlen_compact(cq, ck, mq, mk, lse_tokens=max(Tq, 1)),
    )
    torch.cuda.synchronize()

    for z, (lq, lk) in enumerate(zip(lens_q, lens_k)):
        if lq == 0:
            continue
        qs, ks = int(cq[z]), int(ck[z])
        qz = q[:, :, qs : qs + lq].contiguous()
        kz = k[:, :, ks : ks + lk].contiguous()
        vz = v[:, :, ks : ks + lk].contiguous()
        doz = do[:, :, qs : qs + lq].contiguous()
        lz = lse[:, qs : qs + lq].contiguous()
        dz = delta[:, qs : qs + lq].contiguous()
        ref = torch.zeros_like(qz)
        exe(qz, kz, vz, doz, ref, lz, dz, 1, lq, lk)
        torch.cuda.synchronize()
        got = dq[:, :, qs : qs + lq]
        assert torch.equal(ref, got), (
            f"varlen/{label}/ctype={ctype} sequence {z} (Lq={lq}, Lk={lk}) differs "
            f"from its dense call: max |delta| {(ref.float() - got.float()).abs().max():.3e}"
        )


def test_varlen_padded_matches_dense():
    """The BHSD-with-short-sequences mode: lengths only, no positions."""
    _require_env()
    lens = [40, 96, 128]
    N, head_dim = len(lens), 64
    mq = max(lens)
    gen = torch.Generator(device="cuda").manual_seed(17)

    def _t():
        return torch.randn(N, _NUM_HEADS, mq, head_dim, dtype=torch.float16, device="cuda", generator=gen)

    q, k, v, do = _t(), _t(), _t(), _t()
    lse = torch.randn(N * _NUM_HEADS, mq, dtype=torch.float32, device="cuda", generator=gen)
    delta = torch.randn(N * _NUM_HEADS, mq, dtype=torch.float32, device="cuda", generator=gen)
    exe = build_bwd_dq_module(num_heads=_NUM_HEADS, head_dim=head_dim, causal=True, dtype_str="f16")
    dq = torch.zeros_like(q)
    exe(q, k, v, do, dq, lse, delta, N, mq, mq, varlen=exe.varlen_padded(_cu(lens), _cu(lens), mq, mq))
    torch.cuda.synchronize()

    for z, ln in enumerate(lens):
        qz = q[z : z + 1, :, :ln].contiguous()
        kz = k[z : z + 1, :, :ln].contiguous()
        vz = v[z : z + 1, :, :ln].contiguous()
        doz = do[z : z + 1, :, :ln].contiguous()
        lz = lse[z * _NUM_HEADS : (z + 1) * _NUM_HEADS, :ln].contiguous()
        dz = delta[z * _NUM_HEADS : (z + 1) * _NUM_HEADS, :ln].contiguous()
        ref = torch.zeros_like(qz)
        exe(qz, kz, vz, doz, ref, lz, dz, 1, ln, ln)
        torch.cuda.synchronize()
        assert torch.equal(ref, dq[z : z + 1, :, :ln]), f"padded varlen sequence {z} differs from its dense call"


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.1, 0.5])
def test_dropout_matches_the_reported_mask(p):
    """dQ under dropout, against autograd through the *same* mask.

    ``dropout_mask_gfx1201`` reports which element gets which random, and
    ``_scaled_dot_product_attention_math(dropout_mask=...)`` applies it exactly
    as the kernels do -- after the softmax, survivors scaled by ``1/(1-p)``,
    denominator left undropped. So the only thing that can differ is *which*
    entries were dropped, which is the agreement being tested: the backward
    pass regenerates the forward's stream and a single misplaced random moves a
    row of dQ by an O(1) amount.

    Note the kernel masks **dP**, not P: P is the undropped probability, since
    the logsumexp it comes from is the undropped sum.
    """
    _require_env()
    B, H, N, D = 1, 4, 128, 64
    seed, off = 20250807, 5
    gen = torch.Generator(device="cuda").manual_seed(23)
    q, k, v, do = (torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", generator=gen) for _ in range(4))
    keep = dropout_mask(B, H, N, N, seed, off) > dropout_threshold(p)

    qg = q.float().clone().requires_grad_(True)
    og, _ = torch.ops.aten._scaled_dot_product_attention_math(qg, k.float(), v.float(), dropout_p=p, dropout_mask=keep)
    (dq_ref,) = torch.autograd.grad(og, qg, do.float())
    with torch.no_grad():
        o, _ = torch.ops.aten._scaled_dot_product_attention_math(
            q.float(), k.float(), v.float(), dropout_p=p, dropout_mask=keep
        )
        # The *undropped* logsumexp, which is what the forward kernel writes.
        lse = torch.logsumexp(q.float() @ k.float().transpose(-1, -2) / math.sqrt(D), dim=-1)
        delta = (do.float() * o).sum(-1)

    exe = build_bwd_dq_module(num_heads=H, head_dim=D, causal=False, dtype_str="f16", dropout=True)
    dq = torch.zeros_like(q)
    exe(
        q,
        k,
        v,
        do,
        dq,
        lse.reshape(B * H, N).contiguous(),
        delta.reshape(B * H, N).contiguous(),
        B,
        N,
        N,
        dropout_p=p,
        philox_seed=seed,
        philox_offset2=off,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(dq).all()
    rel = _rel(dq, dq_ref)
    assert rel < 4e-2, f"p={p} rel={rel:.3e}"


def test_split_philox_offset_regenerates_the_same_stream():
    """A `(pointer, immediate)` offset pair must reach the same stream as the sum.

    This is the agreement that matters for a captured graph: the reference mask
    comes from `dropout_mask_gfx1201`, which takes one scalar offset, so if the
    backward's two halves were summed anywhere other than on the device -- or
    in the wrong order, or at the wrong width -- dQ would move by an O(1)
    amount against the same tolerance the seed test relies on.
    """
    _require_env()
    B, H, N, D, p = 1, 4, 128, 64, 0.5
    seed, off = 20250807, 5
    gen = torch.Generator(device="cuda").manual_seed(23)
    q, k, v, do = (torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", generator=gen) for _ in range(4))
    keep = dropout_mask(B, H, N, N, seed, off) > dropout_threshold(p)

    qg = q.float().clone().requires_grad_(True)
    og, _ = torch.ops.aten._scaled_dot_product_attention_math(qg, k.float(), v.float(), dropout_p=p, dropout_mask=keep)
    (dq_ref,) = torch.autograd.grad(og, qg, do.float())
    with torch.no_grad():
        o, _ = torch.ops.aten._scaled_dot_product_attention_math(
            q.float(), k.float(), v.float(), dropout_p=p, dropout_mask=keep
        )
        lse = torch.logsumexp(q.float() @ k.float().transpose(-1, -2) / math.sqrt(D), dim=-1)
        delta = (do.float() * o).sum(-1)

    exe = build_bwd_dq_module(num_heads=H, head_dim=D, causal=False, dtype_str="f16", dropout=True)
    # off == 5 == 3 + 2, with the 3 living in device memory the way a captured
    # graph's counter does.
    dq = torch.zeros_like(q)
    exe(
        q,
        k,
        v,
        do,
        dq,
        lse.reshape(B * H, N).contiguous(),
        delta.reshape(B * H, N).contiguous(),
        B,
        N,
        N,
        dropout_p=p,
        philox_seed=seed,
        philox_offset1=torch.tensor([3], dtype=torch.int64, device=q.device),
        philox_offset2=2,
    )
    torch.cuda.synchronize()
    rel = _rel(dq, dq_ref)
    assert rel < 4e-2, f"the split offset reached a different stream: rel={rel:.3e}"


def test_backward_replays_the_pair_the_forward_reported():
    """The write-back round trip, which is the only way a captured graph works.

    Under capture the effective offset is `*offset1 + offset2`, summed on the
    device out of a counter the host cannot read without synchronising. So the
    forward is asked to record what it drew from, and the backward is driven
    from those two words rather than from anything the host believed it
    passed. Here the host deliberately does *not* know the answer -- the
    counter is 900 and the immediate 100, and the reference mask is built from
    whatever the forward reports.
    """
    _require_env()
    B, H, N, D, p, seed = 1, 4, 128, 64, 0.5, 4242
    gen = torch.Generator(device="cuda").manual_seed(23)
    q, k, v, do = (torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", generator=gen) for _ in range(4))

    def u64(n):
        return torch.tensor([n], dtype=torch.int64, device=q.device)

    fwd = build_flash_attn_func_aiw_module(num_heads=H, head_dim=D, causal=False, dtype_str="f16", dropout=True)
    seed_out, off_out = u64(0), u64(0)
    fwd(
        q,
        k,
        v,
        torch.empty_like(q),
        B,
        N,
        dropout_p=p,
        philox_seed=u64(seed),
        philox_offset1=u64(900),
        philox_offset2=100,
        philox_seed_output=seed_out,
        philox_offset_output=off_out,
    )
    torch.cuda.synchronize()
    assert (seed_out.item(), off_out.item()) == (seed, 1000), "the forward did not report what it used"

    keep = dropout_mask(B, H, N, N, seed_out.item(), off_out.item()) > dropout_threshold(p)
    qg = q.float().clone().requires_grad_(True)
    og, _ = torch.ops.aten._scaled_dot_product_attention_math(qg, k.float(), v.float(), dropout_p=p, dropout_mask=keep)
    (dq_ref,) = torch.autograd.grad(og, qg, do.float())
    with torch.no_grad():
        o, _ = torch.ops.aten._scaled_dot_product_attention_math(
            q.float(), k.float(), v.float(), dropout_p=p, dropout_mask=keep
        )
        lse = torch.logsumexp(q.float() @ k.float().transpose(-1, -2) / math.sqrt(D), dim=-1)
        delta = (do.float() * o).sum(-1)

    exe = build_bwd_dq_module(num_heads=H, head_dim=D, causal=False, dtype_str="f16", dropout=True)
    dq = torch.zeros_like(q)
    exe(
        q,
        k,
        v,
        do,
        dq,
        lse.reshape(B * H, N).contiguous(),
        delta.reshape(B * H, N).contiguous(),
        B,
        N,
        N,
        dropout_p=p,
        philox_seed=seed_out,
        philox_offset1=off_out,
        philox_offset2=0,
    )
    torch.cuda.synchronize()
    rel = _rel(dq, dq_ref)
    assert rel < 4e-2, f"the backward did not replay the forward's stream: rel={rel:.3e}"


def test_dropout_with_a_wrong_seed_is_detected():
    """The negative control. A comparison against a reference is only worth
    its tolerance if a wrong mask actually breaks it."""
    _require_env()
    B, H, N, D, p = 1, 4, 128, 64, 0.5
    gen = torch.Generator(device="cuda").manual_seed(23)
    q, k, v, do = (torch.randn(B, H, N, D, dtype=torch.float16, device="cuda", generator=gen) for _ in range(4))
    keep = dropout_mask(B, H, N, N, 99, 0) > dropout_threshold(p)  # wrong seed
    qg = q.float().clone().requires_grad_(True)
    og, _ = torch.ops.aten._scaled_dot_product_attention_math(qg, k.float(), v.float(), dropout_p=p, dropout_mask=keep)
    (dq_ref,) = torch.autograd.grad(og, qg, do.float())
    with torch.no_grad():
        o, _ = torch.ops.aten._scaled_dot_product_attention_math(
            q.float(), k.float(), v.float(), dropout_p=p, dropout_mask=keep
        )
        lse = torch.logsumexp(q.float() @ k.float().transpose(-1, -2) / math.sqrt(D), dim=-1)
        delta = (do.float() * o).sum(-1)

    exe = build_bwd_dq_module(num_heads=H, head_dim=D, causal=False, dtype_str="f16", dropout=True)
    dq = torch.zeros_like(q)
    exe(
        q,
        k,
        v,
        do,
        dq,
        lse.reshape(B * H, N).contiguous(),
        delta.reshape(B * H, N).contiguous(),
        B,
        N,
        N,
        dropout_p=p,
        philox_seed=1,
        philox_offset2=0,
    )
    torch.cuda.synchronize()
    assert _rel(dq, dq_ref) > 4e-2, "a wrong dropout mask was not detected; the tolerance is doing the work"


def test_dropout_off_build_ignores_p():
    """A build without dropout emits no PRNG at all, so it must reject `p`."""
    _require_env()
    exe = build_bwd_dq_module(num_heads=_NUM_HEADS, head_dim=64, causal=False, dtype_str="f16")
    q, k, v, do = _qkv(1, _NUM_HEADS, _NUM_HEADS, 64, 64, 64, torch.float16)
    _, lse, delta = _reference(q, k, v, do, 1.0 / 8.0, 0)
    dq = torch.zeros_like(q)
    # `dropout_p` is silently ignored rather than rejected, matching the
    # forward: the argument exists in the launch ABI for every build.
    exe(
        q,
        k,
        v,
        do,
        dq,
        lse.reshape(_NUM_HEADS, 64).contiguous(),
        delta.reshape(_NUM_HEADS, 64).contiguous(),
        1,
        64,
        64,
        dropout_p=0.5,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(dq).all()


# ---------------------------------------------------------------------------
# Host-side contracts
# ---------------------------------------------------------------------------


def test_rejects_head_dim_over_the_ladder():
    """dQ tops out at 512, the same width the forward reaches.

    It gets there by sharding: a dQ wave carries Q, dO *and* the dQ
    accumulator, all proportional to head_dim, which is `head_dim` VGPRs per
    lane before anything else, so 384 and 512 run four waves per Q row
    sub-tile. Past 512 there is no rung. `plan` is where the ladder lives, so
    that is where the rejection belongs -- a direct builder call trips the
    kernel's own assertion instead, which is the tuning module contradicting
    itself rather than a caller mistake.
    """
    from fmha_tuning_bwd_dq_gfx1201 import BwdDqInputMetadata
    from fmha_tuning_bwd_dq_gfx1201 import plan as _plan

    # 384 was rejected before sharding existed; it is now a compiled width.
    assert _plan(BwdDqInputMetadata(num_heads=1, head_dim=384)).knobs.shards == 4
    with pytest.raises(ValueError, match="head_dim"):
        _plan(BwdDqInputMetadata(num_heads=1, head_dim=640))


def test_block_m_and_num_waves_must_agree():
    """They state one fact twice; pinning both inconsistently is an error."""
    from fmha_tuning_bwd_dq_gfx1201 import BwdDqInputMetadata, resolve_knobs

    meta = BwdDqInputMetadata(num_heads=1, head_dim=64)
    with pytest.raises(ValueError, match="num_waves"):
        resolve_knobs(meta, BwdDqKnobs(block_m=64, num_waves=8))
    # Pinning only `block_m` moves the wave count with it.
    assert resolve_knobs(meta, BwdDqKnobs(block_m=128)).num_waves == 8


def test_strides_are_read_not_assumed():
    """A BSHD-laid-out tensor passed as `t.transpose(1, 2)` must give the same
    answer as its contiguous BHSD copy. The kernel reads strides; nothing here
    derives the layout from the shape."""
    _require_env()
    B, H, N, D = 2, _NUM_HEADS, 128, 64
    gen = torch.Generator(device="cuda").manual_seed(29)
    bshd = [torch.randn(B, N, H, D, dtype=torch.float16, device="cuda", generator=gen) for _ in range(4)]
    q, k, v, do = (t.transpose(1, 2) for t in bshd)
    _, lse, delta = _reference(q, k, v, do, 1.0 / 8.0, 1)
    lse = lse.reshape(B * H, N).contiguous()
    delta = delta.reshape(B * H, N).contiguous()

    exe = build_bwd_dq_module(num_heads=H, head_dim=D, causal=True, dtype_str="f16")
    strided = torch.zeros(B, N, H, D, dtype=torch.float16, device="cuda").transpose(1, 2)
    exe(q, k, v, do, strided, lse, delta, B, N, N)
    packed = torch.zeros(B, H, N, D, dtype=torch.float16, device="cuda")
    exe(*(t.contiguous() for t in (q, k, v, do)), packed, lse, delta, B, N, N)
    torch.cuda.synchronize()
    assert torch.equal(strided, packed), "the strided and packed layouts disagree"


# --------------------------------------------------------------------------
# Attention bias, and dB
#
# The forward supports a bias; for a long time this kernel did not, and the
# failure mode was silent rather than loud. The forward folds the bias into the
# score *and* into the logsumexp it stores, and this kernel recomputes P from
# that logsumexp -- so a build without the bias term produced `P * exp(-bias)`
# and a plausible, wrong dQ. `test_zero_bias_matches_no_bias` is the cheap
# check that would have caught it, and it is here for that reason.
# --------------------------------------------------------------------------


def _bias_reference(q, k, v, do, bias):
    """fp32 forward + autograd with an additive bias, returning what both sides need."""
    qf, kf, vf = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    bf = bias.detach().float().requires_grad_(True)
    sm = 1.0 / (q.shape[3] ** 0.5)
    s = (qf @ kf.transpose(-1, -2)) * sm + bf
    p = torch.softmax(s, -1)
    o = p @ vf
    o.backward(do.float())
    return (
        o.detach(),
        torch.logsumexp(s, -1).detach(),
        (do.float() * o.detach()).sum(-1),
        qf.grad.detach(),
        bf.grad.detach(),
    )


def _bias_dq(q, k, v, do, bias, dbias=None):
    """dQ (and dB) through a bias build, bypassing the interface as `_bwd_via_kernel` does."""
    batch, heads, seq, head_dim = q.shape
    _, lse, delta, _, _ = _bias_reference(q, k, v, do, bias)
    exe = build_bwd_dq_module(
        num_heads=heads,
        head_dim=head_dim,
        causal=False,
        dtype_str="bf16" if q.dtype == torch.bfloat16 else "f16",
        bias=True,
    )
    pitch = (head_dim + 7) // 8 * 8
    dq = torch.zeros(batch, heads, seq, pitch, dtype=q.dtype, device=q.device)[..., :head_dim]
    exe(
        q,
        k,
        v,
        do,
        dq,
        lse.reshape(batch * heads, seq).contiguous(),
        delta.reshape(batch * heads, seq).contiguous(),
        batch,
        seq,
        seqlen_k=seq,
        bias=bias,
        dbias=dbias,
    )
    return dq


@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_bias_dq_matches_autograd(head_dim):
    """dQ under a bias, against fp32 autograd on the same bias."""
    _require_env()
    q, k, v, do = _qkv(1, 2, 2, 128, 128, head_dim, torch.float16)
    gen = torch.Generator(device=q.device).manual_seed(7)
    bias = torch.randn(1, 2, 128, 128, dtype=q.dtype, device=q.device, generator=gen)
    _, _, _, dq_ref, _ = _bias_reference(q, k, v, do, bias)
    dq = _bias_dq(q, k, v, do, bias, dbias=torch.zeros_like(bias))
    assert _rel(dq, dq_ref) < _RTOL, f"dQ rel={_rel(dq, dq_ref):.3e}"


@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_dbias_matches_autograd(head_dim):
    """dB = dS. The kernel already forms it for GEMM3, so this checks the store."""
    _require_env()
    q, k, v, do = _qkv(1, 2, 2, 128, 128, head_dim, torch.float16)
    gen = torch.Generator(device=q.device).manual_seed(7)
    bias = torch.randn(1, 2, 128, 128, dtype=q.dtype, device=q.device, generator=gen)
    _, _, _, _, db_ref = _bias_reference(q, k, v, do, bias)
    dbias = torch.zeros_like(bias)
    _bias_dq(q, k, v, do, bias, dbias=dbias)
    assert _rel(dbias, db_ref) < _RTOL, f"dB rel={_rel(dbias, db_ref):.3e}"


def test_zero_bias_matches_no_bias():
    """An all-zero bias must reproduce the no-bias build.

    This is the test whose absence let the missing-bias bug live: a bias build
    that ignored the bias entirely would still pass every *other* test here,
    because every other test runs without one.
    """
    _require_env()
    q, k, v, do = _qkv(1, 2, 2, 128, 128, 64, torch.float16)
    zeros = torch.zeros(1, 2, 128, 128, dtype=q.dtype, device=q.device)
    _, lse, delta, _, _ = _bias_reference(q, k, v, do, zeros)
    plain = _run(q, k, v, do, lse, delta, causal=False)
    biased = _bias_dq(q, k, v, do, zeros, dbias=torch.zeros_like(zeros))
    assert _rel(biased, plain) < 1e-3, f"a zero bias moved dQ by rel={_rel(biased, plain):.3e}"


def test_nonzero_bias_actually_changes_dq():
    """The converse: a bias that was silently dropped would pass the test above."""
    _require_env()
    q, k, v, do = _qkv(1, 2, 2, 128, 128, 64, torch.float16)
    gen = torch.Generator(device=q.device).manual_seed(11)
    bias = torch.randn(1, 2, 128, 128, dtype=q.dtype, device=q.device, generator=gen)
    zeros = torch.zeros_like(bias)
    with_b = _bias_dq(q, k, v, do, bias, dbias=torch.zeros_like(bias))
    with_z = _bias_dq(q, k, v, do, zeros, dbias=torch.zeros_like(zeros))
    assert _rel(with_b, with_z) > 1e-2


def test_bias_and_causal_are_rejected_together():
    """As in the forward: a bias already is an additive mask."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        bwd_dq_plan(BwdDqInputMetadata(num_heads=2, head_dim=64, causal=True, bias=True))


def test_dbias_with_a_different_layout_from_bias():
    """B and DB carry separate strides, so DB need not be laid out like B.

    The kernel briefly derived one from the other, which is invisible while
    both are contiguous and writes to the wrong addresses the moment either is
    a view. `dbias` here is a column slice of a wider buffer: its last axis is
    still contiguous, as the ABI requires, but its row pitch is twice B's.
    """
    _require_env()
    q, k, v, do = _qkv(1, 2, 2, 128, 128, 64, torch.float16)
    gen = torch.Generator(device=q.device).manual_seed(31)
    bias = torch.randn(1, 2, 128, 128, dtype=q.dtype, device=q.device, generator=gen)
    _, _, _, _, db_ref = _bias_reference(q, k, v, do, bias)

    wide = torch.zeros(1, 2, 128, 256, dtype=q.dtype, device=q.device)
    dbias = wide[..., :128]
    assert dbias.stride(3) == 1 and dbias.stride(2) != bias.stride(2)
    _bias_dq(q, k, v, do, bias, dbias=dbias)

    assert _rel(dbias, db_ref) < _RTOL, f"dB rel={_rel(dbias, db_ref):.3e}"
    # Nothing may have landed outside the slice.
    assert wide[..., 128:].abs().max().item() == 0.0, "dB wrote past its own columns"


# --------------------------------------------------------------------------
# Regressions found by static reading of an AOTriton pass-1 failure log
# (`flyati/flyc_pass1.out`, 1334 FAILED). The log carried no tracebacks, so
# these were derived from the failing parametrizations plus the resolved tile
# sizes, and each one names the mechanism it is meant to catch.
#
# The tiles are the reason the shapes look arbitrary: at every failing head_dim
# `plan()` resolves BLOCK_M=256 and BLOCK_N=32, so seqlen_q 13 or 16 leaves
# 240+ padding Q rows and seqlen_k 8 or 11 leaves 21+ padding KV columns.
# --------------------------------------------------------------------------


def _ne_reference(q, k, v, do):
    """fp32 autograd with hdim_vo != hdim_qk."""
    qf, kf, vf = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    s = (qf @ kf.transpose(-1, -2)) * (1.0 / (q.shape[3] ** 0.5))
    o = torch.softmax(s, -1) @ vf
    o.backward(do.float())
    return torch.logsumexp(s, -1).detach(), (do.float() * o.detach()).sum(-1), qf.grad.detach()


@pytest.mark.parametrize("hdim_qk,hdim_vo", [(192, 128), (128, 192), (64, 32), (32, 64), (256, 64)])
def test_hdim_qk_ne_vo(hdim_qk, hdim_vo):
    """dQ when the QK and V/O head dims differ.

    One compiled tile serves both axes, so it must cover the wider one and the
    narrower rides as a masked runtime extent. The kernel's `vo_cols` is
    `MaskedAxis(hdim_vo, active=PADDED_HEAD)`, so if `PADDED_HEAD` is derived
    from `head_dim` alone -- as it was -- an equal-to-tile `hdim_qk` turns the
    masking off and the narrower tensor is walked to the full tile width.
    """
    _require_env()
    b, h, n = 1, 2, 128
    gen = torch.Generator(device="cuda").manual_seed(17)

    def _t(d):
        return torch.randn(b, h, n, d, dtype=torch.float16, device="cuda", generator=gen)

    q, k = _t(hdim_qk), _t(hdim_qk)
    v, do = _t(hdim_vo), _t(hdim_vo)
    lse, delta, dq_ref = _ne_reference(q, k, v, do)
    exe = build_bwd_dq_module(num_heads=h, head_dim=hdim_qk, head_dim_v=hdim_vo, causal=False, dtype_str="f16")
    dq = torch.zeros_like(q)
    exe(q, k, v, do, dq, lse.reshape(b * h, n).contiguous(), delta.reshape(b * h, n).contiguous(), b, n, seqlen_k=n)
    assert _rel(dq, dq_ref) < _RTOL, f"qk={hdim_qk} vo={hdim_vo} rel={_rel(dq, dq_ref):.3e}"


@pytest.mark.parametrize("seqlen_q,seqlen_k", [(13, 11), (16, 64), (64, 8), (2048, 8), (71, 37)])
def test_bias_with_ragged_seqlens(seqlen_q, seqlen_k):
    """Bias and dB where neither extent fills its tile.

    Both the bias load and the dB store index by absolute Q row and KV column.
    With BLOCK_M 256 and BLOCK_N 32 these shapes leave most of the tile out of
    bounds on both axes, and dB's KV pitch is exactly `seqlen_k` -- so an
    unbounded store lands in the *next row* rather than in padding.
    """
    _require_env()
    b, h, d = 1, 2, 64
    gen = torch.Generator(device="cuda").manual_seed(19)

    def _t(n):
        return torch.randn(b, h, n, d, dtype=torch.float16, device="cuda", generator=gen)

    q, do = _t(seqlen_q), _t(seqlen_q)
    k, v = _t(seqlen_k), _t(seqlen_k)
    bias = torch.randn(b, h, seqlen_q, seqlen_k, dtype=torch.float16, device="cuda", generator=gen)

    qf, kf, vf = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    bf = bias.detach().float().requires_grad_(True)
    s = (qf @ kf.transpose(-1, -2)) * (1.0 / d**0.5) + bf
    o = torch.softmax(s, -1) @ vf
    o.backward(do.float())
    lse = torch.logsumexp(s, -1).detach()
    delta = (do.float() * o.detach()).sum(-1)

    exe = build_bwd_dq_module(num_heads=h, head_dim=d, causal=False, dtype_str="f16", bias=True)
    dq = torch.zeros_like(q)
    dbias = torch.zeros_like(bias)
    exe(
        q,
        k,
        v,
        do,
        dq,
        lse.reshape(b * h, seqlen_q).contiguous(),
        delta.reshape(b * h, seqlen_q).contiguous(),
        b,
        seqlen_q,
        seqlen_k=seqlen_k,
        bias=bias,
        dbias=dbias,
    )
    assert _rel(dq, qf.grad.detach()) < _RTOL, f"dQ rel={_rel(dq, qf.grad.detach()):.3e}"
    assert _rel(dbias, bf.grad.detach()) < _RTOL, f"dB rel={_rel(dbias, bf.grad.detach()):.3e}"


def test_dbias_does_not_write_past_its_own_extent():
    """The direct out-of-bounds detector, rather than an indirect wrong answer.

    dB is a `(seqlen_q, seqlen_k)` window of a buffer padded on both axes and
    pre-filled with a sentinel. Anything the kernel writes outside its extent
    lands in the padding and is visible; with a tightly-sized dB the same
    overflow silently lands on a neighbouring row's real gradient, which is
    what made the original bug read as a plain accuracy failure.
    """
    _require_env()
    b, h, d, sq, sk = 1, 2, 64, 13, 11
    gen = torch.Generator(device="cuda").manual_seed(23)

    def _t(n):
        return torch.randn(b, h, n, d, dtype=torch.float16, device="cuda", generator=gen)

    q, do = _t(sq), _t(sq)
    k, v = _t(sk), _t(sk)
    bias = torch.randn(b, h, sq, sk, dtype=torch.float16, device="cuda", generator=gen)

    qf, kf, vf = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    s = (qf @ kf.transpose(-1, -2)) * (1.0 / d**0.5) + bias.float()
    o = torch.softmax(s, -1) @ vf
    o.backward(do.float())
    lse = torch.logsumexp(s, -1).detach()
    delta = (do.float() * o.detach()).sum(-1)

    SENTINEL = 7.0
    padded = torch.full((b, h, sq + 64, sk + 64), SENTINEL, dtype=torch.float16, device="cuda")
    dbias = padded[:, :, :sq, :sk]

    exe = build_bwd_dq_module(num_heads=h, head_dim=d, causal=False, dtype_str="f16", bias=True)
    exe(
        q,
        k,
        v,
        do,
        torch.zeros_like(q),
        lse.reshape(b * h, sq).contiguous(),
        delta.reshape(b * h, sq).contiguous(),
        b,
        sq,
        seqlen_k=sk,
        bias=bias,
        dbias=dbias,
    )

    assert (padded[:, :, :, sk:] == SENTINEL).all(), "dB wrote past seqlen_k"
    assert (padded[:, :, sq:, :] == SENTINEL).all(), "dB wrote past seqlen_q"


def test_hdim_qk_ne_vo_sets_padded_head():
    """The policy half of `test_hdim_qk_ne_vo`, without a GPU.

    One flag gates both column axes, so it has to be on when *either* is short
    of the tile. Deriving it from `head_dim` alone is what left the V/O axis
    unmasked whenever the QK axis happened to land exactly on a ladder rung.
    """
    for qk, vo, want in ((192, 192, False), (192, 128, True), (128, 192, True), (64, 32, True)):
        knobs = bwd_dq_plan(BwdDqInputMetadata(num_heads=2, head_dim=qk, head_dim_v=vo, causal=False)).knobs
        assert knobs.padded_head is want, f"qk={qk} vo={vo}: padded_head={knobs.padded_head}"
        assert knobs.block_dmodel >= max(qk, vo), "the single tile must cover the wider axis"
