# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""High-level FlyDSL backward-dQ API for gfx1201 / RDNA4.

Wraps ``fmha_bwd_dq_gfx1201_kernel.build_bwd_dq_module_primary`` behind a
single function, ``flydsl_bwd_dq_gfx1201(q, k, v, o, do, lse, ...)``, which
returns ``dQ``.

The signature is the backward pass's, not the forward's, and every argument it
adds is something the forward already produced:

    o     the forward output, needed only for ``delta``
    do    the incoming gradient
    lse   the forward's logsumexp, in **our** layout -- see below

``delta = rowsum(dO * O)`` is computed here in PyTorch
(``(do.float() * o.float()).sum(-1)``), which is one extra pass over two
``(B, H, S, D)`` tensors per call. AOTriton fuses it into a ``bwd_preprocess``
kernel and so should we eventually; the kernel takes it as a tensor either way,
so that change lands entirely on this side of the boundary.

**On ``lse``.** This is the forward kernel's own output and its layout is the
one ``lse_token_pitch`` describes, not AOTriton's independent convention. Pass
either ``(B, H, S)`` (what
``flydsl_flash_attn_func_gfx1201(..., return_lse=True)`` returns) or the flat
``(B*H, S)`` the kernel writes; both are the same buffer.

BHSD (``[B, H, S, D]``) shape convention throughout, matching the forward. Only
the *shape* is fixed -- any memory layout with D innermost is accepted, so a
BSHD-laid-out tensor is passed as ``t.transpose(1, 2)``.

This computes **dQ only**. dK and dV come from the sibling kernels; a full
backward pass runs all of them.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from fmha_bwd_dq_gfx1201_kernel import build_bwd_dq_module_primary
from fmha_tuning_bwd_dq_gfx1201 import BwdDqInputMetadata, BwdDqKnobs
from fmha_tuning_bwd_dq_gfx1201 import plan as _plan_build

__all__ = ["flydsl_bwd_dq_gfx1201", "bwd_dq_delta"]


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "f16"
    raise ValueError(f"flydsl_bwd_dq_gfx1201 only supports bf16/f16, got {dtype!r}")


@lru_cache(maxsize=32)
def _get_kernel(meta: BwdDqInputMetadata, knobs: BwdDqKnobs):
    """Build cache. Both arguments are frozen dataclasses, so the pair hashes.

    The *pair* is the key, deliberately: an earlier version of the forward
    interface listed the knobs it happened to care about, which is how tuning
    changes managed to alter the emitted kernel without changing the key.
    """
    return build_bwd_dq_module_primary(meta, knobs)


def bwd_dq_delta(o: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
    """``rowsum(dO * O)`` as ``(B*H, S)`` float32 -- AOTriton's ``bwd_preprocess``.

    Exposed rather than inlined because all three backward kernels need the
    same tensor and must agree on it exactly: dQ, dK and dV each subtract this
    same per-row scalar, and two callers computing it two ways would diverge
    silently. Accumulated in float32 whatever the inputs are, for the usual
    reason -- it is a reduction over head_dim of a product of two 16-bit
    tensors.
    """
    if o.shape != do.shape:
        raise ValueError(f"o and do must share shape, got {tuple(o.shape)} and {tuple(do.shape)}")
    b, h, s, _ = o.shape
    return (o.float() * do.float()).sum(-1).reshape(b * h, s).contiguous()


def _flat_rows(t: torch.Tensor, name: str, b: int, h: int, s: int) -> torch.Tensor:
    """A ``(B, H, S)`` or ``(B*H, S)`` float32 tensor as contiguous ``(B*H, S)``."""
    if t.dtype != torch.float32:
        raise ValueError(f"{name} must be float32, got {t.dtype}")
    if t.dim() == 3:
        if tuple(t.shape) != (b, h, s):
            raise ValueError(f"{name} must be ({b}, {h}, {s}) or ({b * h}, {s}), got {tuple(t.shape)}")
        t = t.reshape(b * h, s)
    elif t.dim() == 2:
        if tuple(t.shape) != (b * h, s):
            raise ValueError(f"{name} must be ({b}, {h}, {s}) or ({b * h}, {s}), got {tuple(t.shape)}")
    else:
        raise ValueError(f"{name} must be rank 2 or 3, got rank {t.dim()}")
    # `.contiguous()` is a no-op for the common case and a copy for a sliced
    # view. The kernel derives the row pitch from VarlenBits rather than
    # reading strides, so a non-contiguous row tensor is not merely slower --
    # it would be read wrong.
    return t.contiguous()


def flydsl_bwd_dq_gfx1201(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    causal: bool = False,
    sm_scale: float | None = None,
    stream: torch.cuda.Stream | None = None,
    knobs: BwdDqKnobs | None = None,
    delta: torch.Tensor | None = None,
    dq: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the FlyDSL backward-dQ kernel on RDNA4 (gfx1201).

    Args:
        q, k, v: the forward pass's inputs, ``[batch, num_heads, seq_len,
            head_dim]`` (BHSD). Only the shape is fixed; any layout with D
            innermost is accepted. ``k``/``v`` may carry fewer heads than
            ``q`` (MQA/GQA), and ``seq_len`` may differ between Q and KV.
        o: the forward pass's output, same shape as ``q``. Used only to build
            ``delta``; pass ``delta=`` directly to skip it.
        do: the incoming gradient, same shape as ``q``.
        lse: the forward pass's logsumexp, ``(B, H, S)`` or ``(B*H, S)``
            float32, in **natural** units. This is exactly what
            ``flydsl_flash_attn_func_gfx1201(..., return_lse=True)`` returns.
        causal: top-left causal masking when ``True``, matching the forward's
            default alignment (PyTorch's ``is_causal``).
        sm_scale: defaults to ``1/sqrt(head_dim)``. Must match the forward's.
        knobs: ``None`` uses the tuning policy for this shape; a
            ``BwdDqKnobs`` with some fields set pins those and lets policy
            resolve the rest.
        delta: precomputed ``rowsum(dO * O)``, as ``bwd_dq_delta`` returns it.
        dq: optional output buffer; allocated like ``q`` when omitted.

    Returns:
        ``dQ``, same shape and dtype as ``q``.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda and do.is_cuda):
        raise ValueError("flydsl_bwd_dq_gfx1201 requires CUDA/HIP tensors")
    if not (q.device == k.device == v.device == do.device):
        raise ValueError("q/k/v/do must reside on the same device")
    try:
        arch = torch.cuda.get_device_properties(q.device.index).gcnArchName
    except Exception:
        arch = ""
    if not (arch.lower().split(":")[0] if arch else "").startswith("gfx1201"):
        raise ValueError(f"flydsl_bwd_dq_gfx1201 requires gfx1201, got {arch!r}")
    if q.dim() != 4:
        raise ValueError(f"expected 4D BHSD tensor, got rank {q.dim()} ({tuple(q.shape)})")
    if k.shape != v.shape:
        raise ValueError(f"k and v must share shape, got k={tuple(k.shape)} v={tuple(v.shape)}")
    if do.shape != q.shape:
        raise ValueError(f"do must share q's shape, got {tuple(do.shape)} and {tuple(q.shape)}")
    if not (q.dtype == k.dtype == v.dtype == do.dtype):
        raise ValueError(f"q/k/v/do dtype must match: {q.dtype}/{k.dtype}/{v.dtype}/{do.dtype}")
    if q.shape[0] != k.shape[0] or q.shape[3] != k.shape[3]:
        raise ValueError(f"q/k must share batch and head_dim, got q={tuple(q.shape)} k={tuple(k.shape)}")
    if q.shape[1] % k.shape[1]:
        raise ValueError(f"num_heads_q ({q.shape[1]}) must be divisible by num_heads_k ({k.shape[1]})")

    batch, num_heads, seq_len_q, head_dim = q.shape
    seq_len_k = k.shape[2]
    dtype_str = _torch_dtype_to_str(q.dtype)

    _plan = _plan_build(
        BwdDqInputMetadata(
            num_heads=num_heads,
            head_dim=head_dim,
            causal=causal,
            dtype_str=dtype_str,
            sm_scale=sm_scale,
        ),
        knobs,
    )
    if _plan.knobs.padded_head:
        # The D-axis pitch must be a multiple of 16 bytes -- 8 elements at
        # f16/bf16 -- because loads and stores are 8 columns wide, so the chunk
        # containing `head_dim` runs to ceil8(head_dim). This is the forward's
        # alignment contract and it is load-bearing on the *store* side too:
        # dQ's last chunk would otherwise write into the next head.
        for name, t in (("q", q), ("k", k), ("v", v), ("do", do)):
            if t.stride(2) % 8:
                raise ValueError(
                    f"{name} has a D-axis pitch of {t.stride(2)} elements, which is not a "
                    f"multiple of 8 (16 bytes). head_dim {head_dim} is not a compiled tile "
                    f"width, so the kernel operates on ceil8({head_dim})="
                    f"{(head_dim + 7) // 8 * 8} columns and needs the allocation padded to "
                    f"match. Pad the last dimension before calling."
                )

    lse_p = _flat_rows(lse, "lse", batch, num_heads, seq_len_q)
    if delta is None:
        if o is None:
            raise ValueError("pass either o= (to derive delta) or delta=")
        if o.shape != q.shape:
            raise ValueError(f"o must share q's shape, got {tuple(o.shape)} and {tuple(q.shape)}")
        delta_p = bwd_dq_delta(o, do)
    else:
        delta_p = _flat_rows(delta, "delta", batch, num_heads, seq_len_q)

    if dq is None:
        # Same D-pitch rule as the forward's O allocation: `torch.empty_like`
        # on a padded head_dim would give a tightly packed tensor whose last
        # 8-wide store chunk spills into the next head.
        _pitch = (head_dim + 7) // 8 * 8
        if _pitch == head_dim:
            dq = torch.empty_like(q)
        else:
            dq = torch.empty(batch, num_heads, seq_len_q, _pitch, dtype=q.dtype, device=q.device)[..., :head_dim]

    with torch.cuda.device(q.device.index):
        launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
        if launch_stream.device != q.device:
            raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
        exe = _get_kernel(_plan.meta, _plan.knobs)
        exe(
            q,
            k,
            v,
            do,
            dq,
            lse_p,
            delta_p,
            batch,
            seq_len_q,
            seq_len_k,
            scale=sm_scale,
            stream=launch_stream,
        )
    return dq
