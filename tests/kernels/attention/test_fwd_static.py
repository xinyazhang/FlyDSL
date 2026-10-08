# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Forward JIT-only baking knobs (FWD-40..44): STATIC_HEADS, STATIC_HDIM, STATIC_STRIDES, STATIC_LAYOUT, STATIC_SCALE.

Each bakes a group of per-call kernel arguments (the Leading_upper_snake_case parameters `Num_head_*`, `Hdim_*`,
`Stride_*`, `Varlen_bits`, `Sm_scale`) into the binary, one compile per value set. At the defaults every one of them is a
real kernarg (the ABI goldens in `test_gfx950.py`), so these builds are never an AOT build. They must

* answer within the fp64 rounding floor, like any forward (`fwd_check`), on every input shape the baked values describe, and
* agree with the runtime build: the program is the same arithmetic with constants folded, so the outputs match to one bf16
  ulp in at most 0.1% of the elements (fast-math reassociation can round a different binary differently; Q25).
"""

import pytest
import torch

from tests.kernels.attention.attn_testlib import (
    DTYPES,
    PERMS,
    WINDOW_BOTRIGHT,
    VarlenCase,
    alloc,
    fwd_check,
    meta_of,
    randn,
    run_fwd,
    seeded,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

BR = (WINDOW_BOTRIGHT, WINDOW_BOTRIGHT)
KNOBS = ("STATIC_HEADS", "STATIC_HDIM", "STATIC_STRIDES", "STATIC_LAYOUT", "STATIC_SCALE")
PINS = {k: {k: True} for k in KNOBS}
PINS["RAW_SCORES"] = {"RAW_SCORES": True}
PINS["all"] = {k: True for k in KNOBS}
PINS["all+raw"] = {**{k: True for k in KNOBS}, "RAW_SCORES": True}


def _agree(got, want, ctx):
    assert (got != want).sum().item() <= 0.001 * got.numel(), ctx
    assert torch.allclose(got.float(), want.float(), rtol=2**-7, atol=2**-10), ctx


@pytest.mark.parametrize("window", [None, BR], ids=["dense", "causal"])
@pytest.mark.parametrize("pins", list(PINS))
def test_static_knob_is_correct_and_matches_the_runtime_build(fwd_build, pins, window):
    """FWD-40: every knob alone, and all of them, under GQA (`num_head_k != num_head_q`), a gapped BSHD layout and a
    non-default scale, against the fp64 floor and against the runtime build."""
    dtype = DTYPES["bf16"]
    meta = meta_of(head_dim=64, window=window is not None)
    gen = seeded(40)
    b, hq, hk, s, d = 2, 8, 2, 256, 64
    big = randn(b, hq, s + 37, d, dtype, (0, 2, 1), gen=gen)
    q = big[:, :, :s, :]
    k, v = (randn(b, hk, s, d, dtype, (0, 2, 1), gen=gen) for _ in range(2))
    kw = dict(
        b=b,
        hq=hq,
        hk=hk,
        sq=s,
        d=d,
        dtype=dtype,
        qkv=(q, k, v),
        window=window,
        scale=0.2,
        perms=(None,) * 3 + ((0, 2, 1),),
    )
    fwd_check(fwd_build(meta, **PINS[pins]), ctx=f"{pins} static", **kw)
    o_static, o_runtime = alloc(b, hq, s, d, dtype, (0, 2, 1)), alloc(b, hq, s, d, dtype, (0, 2, 1))
    run_fwd(fwd_build(meta, **PINS[pins]), q, k, v, o_static, scale=0.2, window=window)
    run_fwd(fwd_build(meta), q, k, v, o_runtime, scale=0.2, window=window)
    _agree(o_static, o_runtime, f"{pins} vs runtime")


@pytest.mark.parametrize("window", [None, BR], ids=["dense", "causal"])
@pytest.mark.parametrize("pins", ["STATIC_SCALE", "RAW_SCORES", "STATIC_SCALE+RAW_SCORES"])
@pytest.mark.parametrize("scale", [0.3, 1.0, 0.0, -0.2], ids=["pos", "one", "zero", "neg"])
def test_scale_values(fwd_build, scale, window, pins):
    """FWD-41: the scale as a baked value (`STATIC_SCALE`), as the raw-score path (`RAW_SCORES`: the masks and the row max
    run unscaled, the scale is one multiply per row and the exp subtract's FMA; a negative scale flips Q's sign, a zero one
    runs at a vanishing positive one), and both. Every scale is correct, a masked `-inf` score never meets the multiply as
    a NaN at scale 0, and the answer is the default build's to the usual tolerance."""
    dtype = DTYPES["bf16"]
    meta = meta_of(head_dim=64, window=window is not None)
    fn = fwd_build(meta, **{k: True for k in pins.split("+")})
    fwd_check(fn, b=2, hq=4, sq=257, sk=300, d=64, dtype=dtype, window=window, scale=scale, ctx=f"scale {scale}")
    gen = seeded(41)
    q, k, v = (randn(2, 4, 257, 64, dtype, gen=gen) for _ in range(3))
    want, got = alloc(2, 4, 257, 64, dtype), alloc(2, 4, 257, 64, dtype)
    run_fwd(fwd_build(meta), q, k, v, want, scale=scale, window=window)
    run_fwd(fn, q, k, v, got, scale=scale, window=window)
    _agree(got, want, f"scale {scale}")


def test_raw_scores_refuses_what_adds_in_the_scaled_domain(backend, arch):
    """FWD-42: a bias (or ALiBi) adds in the scaled domain and the wide body scales its own staged scores, so RAW_SCORES
    refuses them at `resolve` instead of returning a plausible wrong answer."""
    for meta in (meta_of(head_dim=64, bias=True), meta_of(head_dim=512)):
        with pytest.raises(NotImplementedError, match="RAW_SCORES"):
            backend.fwd_knobs(arch, RAW_SCORES=True).resolve(meta)


@pytest.mark.parametrize("hdim", [72, 128], ids=lambda x: f"d{x}")
def test_static_hdim_with_padded_and_exact_heads(fwd_build, hdim):
    """FWD-43: a baked head dim over a padded tile (72 in a 96-wide one: the column masks use the baked extent) and over
    an exact one."""
    dtype = DTYPES["bf16"]
    meta = meta_of(head_dim=hdim, window=True)
    fn = fwd_build(meta, STATIC_HDIM=True, STATIC_STRIDES=True)
    fwd_check(fn, b=2, hq=4, sq=257, sk=300, d=hdim, dtype=dtype, window=BR, perms=(PERMS[1],) * 4, ctx=f"hdim {hdim}")


def test_static_layout_refuses_varlen_and_dense_stays_correct(fwd_build):
    """FWD-44: `Varlen_bits` is baked as 0 (dense), compiling the varlen decode away, so the host refuses a varlen call
    like STATIC_SEQLEN does; the dense call is right."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=64, window=True), STATIC_LAYOUT=True)
    case = VarlenCase("0x0B0B", [64, 64], [64, 64], 2, 2, 64, dtype)
    with pytest.raises(ValueError, match="STATIC_LAYOUT"):
        case.launch(fn, window=BR)
    fwd_check(fn, b=2, hq=4, sq=130, sk=300, d=64, dtype=dtype, window=BR, ctx="static layout dense")


def test_static_values_each_get_their_own_binary(fwd_build):
    """FWD-45: two calls of one STATIC_HEADS/STATIC_STRIDES build with different head counts or strides must not reuse
    the first one's binary (the baked tuple keys the compile)."""
    dtype = DTYPES["bf16"]
    fn = fwd_build(meta_of(head_dim=64), STATIC_HEADS=True, STATIC_STRIDES=True, STATIC_SCALE=True)
    for hq, hk, sq in ((4, 4, 128), (8, 2, 128), (4, 4, 192)):
        fwd_check(fn, b=2, hq=hq, hk=hk, sq=sq, d=64, dtype=dtype, ctx=f"heads {hq}/{hk} sq {sq}")
