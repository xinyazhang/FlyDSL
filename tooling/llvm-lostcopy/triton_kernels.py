#!/usr/bin/env python
"""Candidate Triton kernels for the gfx950 lost-SGPR-definition reproducer.

`bwd_preprocess_min` is a self-contained reduction of AOTriton's
`bwd_preprocess`, which is the Triton kernel family already observed to carry
the defect.
"""
import triton
import triton.language as tl


@triton.jit
def bwd_preprocess_min(
    Out,
    DO,
    Delta,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_don,
    cu_seqlens_q,
    num_seqlens,
    max_seqlen_q,
    hdim_vo,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M
    offs_m = off_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    num_h = tl.num_programs(1)

    o_ptrs = (
        Out
        + off_z * stride_oz
        + off_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_on
    )
    if num_seqlens == 0:
        seqlen_q = max_seqlen_q
    else:
        cu_start = tl.load(cu_seqlens_q + off_z)
        cu_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_end - cu_start
    do_ptrs = (
        DO
        + off_z * stride_doz
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :] * stride_don
    )

    mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD:
        mask = mask & (offs_d[None, :] < hdim_vo)
    o = tl.load(o_ptrs, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)

    off_zh = off_z * num_h + off_h
    delta_ptrs = Delta + off_zh * max_seqlen_q + off_m + tl.arange(0, BLOCK_M)
    overflow = off_m + BLOCK_M - seqlen_q
    if overflow > 0:
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow, dtype=tl.int32)
        store_mask = boundary > tl.arange(0, BLOCK_M)
        tl.store(delta_ptrs, delta, mask=store_mask)
    else:
        tl.store(delta_ptrs, delta)
