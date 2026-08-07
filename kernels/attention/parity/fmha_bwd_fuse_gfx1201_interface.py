# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Public API for the gfx1201 fused flash-attention backward pass.

``flydsl_fmha_bwd_fuse_gfx1201(q, k, v, o, do, lse, ...) -> (dq, dk, dv)``

BHSD in, BHSD out, matching the forward kernel's convention: only the *shape*
is fixed at ``(batch, num_heads, seq_len, head_dim)``; any memory layout with D
innermost is accepted, so a BSHD-laid-out tensor is passed as
``t.transpose(1, 2)``.

Three things this layer owns, all of them because they are not the kernel's
business:

**delta.** The backward pass needs ``delta = rowsum(dO * O)``. It is computed
here with PyTorch, in f32, and handed over as a tensor sharing LSE's layout. A
fused preprocess kernel (AOTriton's ``bwd_preprocess``) would be a later
optimisation; it is a separate kernel there too.

**The GQA reduction.** The kernel gives the dK/dV role one program per *query*
head, so under GQA it writes ``(B, Hq, S, D)`` and this layer sums each group
down to ``(B, Hk, S, D)``. AOTriton instead loops the group inside the kernel
with K and V held resident, which saves the scratch buffer and the K/V
re-reads; the cost there is a third level of loop-carried accumulator, which is
the thing this port most wanted to avoid. See the kernel module docstring.

**Where LSE comes from.** ``lse`` must be the forward kernel's output --
natural-log, f32, compact, with the pitch VarlenBits describes. Passing
`torch.logsumexp` of a reference computation works too and is what the test
does for one case, but the intended pairing is
``flydsl_flash_attn_func_gfx1201(..., return_lse=True)``.

Not supported: attention bias, dropout (rejected at build time), and head_dim
above 128 (rejected at plan time -- see "The register wall" in
`fmha_tuning_bwd_fuse_gfx1201`).
"""

from __future__ import annotations

from functools import lru_cache

import torch
from fmha_bwd_fuse_gfx1201_kernel import build_fmha_bwd_fuse_module
from fmha_tuning_bwd_fuse_gfx1201 import BwdInputMetadata, BwdKnobs
from fmha_tuning_bwd_fuse_gfx1201 import plan as _plan_build

__all__ = ["flydsl_fmha_bwd_fuse_gfx1201", "compute_delta"]


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "f16"
    raise ValueError(f"flydsl_fmha_bwd_fuse_gfx1201 only supports bf16/f16, got {dtype!r}")


@lru_cache(maxsize=32)
def _get_kernel(meta: BwdInputMetadata, knobs: BwdKnobs):
    """Build cache. Both arguments are frozen dataclasses, so the pair hashes
    and every knob that reaches the builder is inside the key."""
    return build_fmha_bwd_fuse_module(meta, knobs)


def compute_delta(o: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
    """``rowsum(dO * O)`` in f32, shaped ``(B, H, S)`` and contiguous.

    In f32 regardless of the input dtype: it is a reduction over head_dim of a
    product of two 16-bit tensors, and accumulating that at 16 bits loses
    several bits of the answer at head_dim 128. AOTriton's ``bwd_preprocess``
    does the same, in a kernel.
    """
    return (do.float() * o.float()).sum(-1).contiguous()


def flydsl_fmha_bwd_fuse_gfx1201(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    causal_type: int | None = None,
    window: tuple[int, int] | None = None,
    varlen: dict | None = None,
    num_seqlens: int | None = None,
    stream: torch.cuda.Stream | None = None,
    knobs: BwdKnobs | None = None,
    delta: torch.Tensor | None = None,
):
    """Gradients of flash attention with respect to q, k and v.

    Args:
        q, k, v: BHSD tensors, f16 or bf16. ``k``/``v`` may carry fewer heads
            than ``q`` (GQA); ``num_head_q`` must be a multiple of
            ``num_head_k``.
        o:  the forward pass's output, same shape and dtype as ``q``.
        do: the incoming gradient, same shape and dtype as ``o``.
        lse: the forward pass's logsumexp, f32, natural-log, shape
            ``(B, H, S)`` or ``(B*H, S)`` and contiguous.
        causal: causal masking. ``causal_type`` selects the alignment
            (1 top-left, the default; 2 bottom-right; 3 an explicit
            ``window=(left, right)``).
        sm_scale: softmax scale; defaults to ``1/sqrt(head_dim)``.
        varlen: the forward kernel's VarlenBits dict, or ``None`` for dense.
        num_seqlens: how many sequences the batch holds. Required with
            ``varlen`` and forbidden without it: a stacked layout puts every
            sequence in one rank-4 tensor whose batch extent is 1, so the
            sequence count cannot be read off the shape -- and taking it from
            the shape anyway would silently run one sequence's worth of the
            grid over a packed batch.
        delta: an already-computed ``rowsum(dO*O)``; computed here if omitted.

    Returns:
        ``(dq, dk, dv)`` with the shapes and dtypes of ``q``, ``k`` and ``v``.
    """
    for name, t in (("q", q), ("k", k), ("v", v), ("o", o), ("do", do)):
        if not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA/HIP tensor")
    if not (q.dtype == k.dtype == v.dtype == o.dtype == do.dtype):
        raise ValueError("q/k/v/o/do must share a dtype")
    if q.dim() != 4:
        raise ValueError(f"expected 4D BHSD tensors, got rank {q.dim()}")
    if lse.dtype != torch.float32:
        raise ValueError(f"lse must be float32, got {lse.dtype}")
    if not lse.is_contiguous():
        raise ValueError("lse must be contiguous: the kernel derives its pitch from VarlenBits")

    try:
        arch = torch.cuda.get_device_properties(q.device.index).gcnArchName
    except Exception:
        arch = ""
    if not arch.lower().split(":")[0].startswith("gfx1201"):
        raise ValueError(f"flydsl_fmha_bwd_fuse_gfx1201 requires gfx1201, got {arch!r}")

    batch, nhq, seq_q, head_dim = q.shape
    nhk, seq_k = k.shape[1], k.shape[2]
    if varlen is None:
        if num_seqlens is not None and num_seqlens != batch:
            raise ValueError(f"num_seqlens={num_seqlens} contradicts the dense batch extent {batch}")
        n_seq = batch
    else:
        if num_seqlens is None:
            raise ValueError("varlen= requires num_seqlens=: a stacked batch's sequence count is not in the shape")
        n_seq = int(num_seqlens)

    _plan = _plan_build(
        BwdInputMetadata(
            num_head_q=nhq,
            num_head_k=nhk,
            head_dim=head_dim,
            causal=causal,
            dtype_str=_torch_dtype_to_str(q.dtype),
            sm_scale=sm_scale,
            causal_type=causal_type,
        ),
        knobs,
    )

    if delta is None:
        delta = compute_delta(o, do)
    if delta.dtype != torch.float32 or not delta.is_contiguous():
        raise ValueError("delta must be contiguous float32 with lse's layout")

    group = nhq // nhk
    dq = torch.empty_like(q)
    # Expanded dK/dV. `zeros`, not `empty`: a key row past `seqlen_k` in the
    # final tile is never written by any program, and under varlen whole
    # programs exit without writing. Reducing uninitialised memory would poison
    # the group sum.
    dk_e = torch.zeros(batch, nhq, seq_k, head_dim, dtype=q.dtype, device=q.device)
    dv_e = torch.zeros(batch, nhq, seq_k, head_dim, dtype=q.dtype, device=q.device)
    # dQ likewise: rows past `seqlen_q` in the final tile are skipped, and the
    # caller sees the whole tensor.
    dq.zero_()

    with torch.cuda.device(q.device.index):
        launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
        exe = _get_kernel(_plan.meta, _plan.knobs)
        exe(
            q,
            k,
            v,
            do,
            dk_e,
            dv_e,
            dq,
            lse,
            delta,
            n_seq,
            seq_q,
            seq_k,
            scale=_plan.meta.sm_scale,
            stream=launch_stream,
            window=window,
            varlen=varlen,
        )

    if group == 1:
        return dq, dk_e, dv_e
    # Sum each GQA group back onto its KV head. f32 for the sum: the group can
    # be 8 wide and the terms are same-signed often enough that a 16-bit
    # accumulation visibly loses precision against the reference.
    dk = dk_e.view(batch, nhk, group, seq_k, head_dim).float().sum(2).to(q.dtype)
    dv = dv_e.view(batch, nhk, group, seq_k, head_dim).float().sum(2).to(q.dtype)
    return dq, dk, dv
