# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""High-level FlyDSL dK / dV backward API for gfx1201 / RDNA4.

Wraps `fmha_bwd_dkdv_gfx1201_kernel.build_bwd_dkdv_module_primary` behind a
single function, `flydsl_flash_attn_bwd_dkdv_gfx1201(q, k, v, do, o, lse, ...)`:

- BHSD (`[B, H, S, D]`) input/output *shape* convention, matching the forward
  and AOTriton's `attn_bwd`. Only the shape is fixed; any memory layout with D
  innermost is accepted, so a BSHD-laid-out tensor is passed as
  `t.transpose(1, 2)`.
- Build cache keyed by the whole `(BwdDkDvMetadata, BwdDkDvKnobs)` pair, both
  frozen dataclasses so the pair *is* the key -- the forward's interface
  records what happens when only some fields are listed.
- `delta = rowsum(dO * O)` computed here in torch, in fp32. AOTriton has a
  fused `bwd_preprocess` kernel; that is a later optimisation and is
  deliberately not on this path, so the reduction is one obvious tensor
  expression rather than a second kernel to keep in sync.
- dK and dV are allocated with the D axis padded to the same 8-element
  multiple the inputs must have, then sliced. `torch.empty_like` would give a
  tightly packed tensor whose last 8-wide store chunk spills into the next
  head -- the same trap the forward's O allocation documents.

**This computes dK and dV only.** dQ is a separate kernel; a full backward
pass needs both. The three-way split is AOTriton's (`bwd_kernel_dk_dv`,
`bwd_kernel_dq`) and exists because the two accumulate along different axes.

The `lse` argument must be the logsumexp *our* forward kernel wrote, in the
layout its VarlenBits describe. It is not interchangeable with a value
recomputed by another implementation unless that one also uses natural-log
units and writes `+inf` for a row with no live keys.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from fmha_bwd_dkdv_gfx1201_kernel import build_bwd_dkdv_module_primary
from fmha_tuning_bwd_dkdv_gfx1201 import BwdDkDvKnobs, BwdDkDvMetadata
from fmha_tuning_bwd_dkdv_gfx1201 import plan as _plan_build

__all__ = ["flydsl_flash_attn_bwd_dkdv_gfx1201"]


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "f16"
    raise ValueError(f"flydsl_flash_attn_bwd_dkdv_gfx1201 only supports bf16/f16, got {dtype!r}")


@lru_cache(maxsize=32)
def _get_kernel(meta: BwdDkDvMetadata, knobs: BwdDkDvKnobs):
    """Build cache. Both arguments are frozen dataclasses, so both hash."""
    return build_bwd_dkdv_module_primary(meta, knobs)


def _as_row_2d(t: torch.Tensor, name: str) -> torch.Tensor:
    """A `(B, H, S)` or `(B*H, S)` row-wise tensor as the compact 2-D form.

    `.reshape` and not `.view`: the caller may hand us a slice of a longer
    logsumexp buffer, which is exactly what the forward's `return_lse=True`
    produces when it trims a padded sequence length. A reshape copies in that
    case, which is correct and cheap; a view would refuse.
    """
    if t.dtype != torch.float32:
        raise ValueError(f"{name} must be float32, got {t.dtype}")
    if t.dim() == 3:
        t = t.reshape(t.shape[0] * t.shape[1], t.shape[2])
    elif t.dim() != 2:
        raise ValueError(f"{name} must be rank 2 (rows, tokens) or rank 3 (B, H, S), got {tuple(t.shape)}")
    return t.contiguous()


def flydsl_flash_attn_bwd_dkdv_gfx1201(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    *,
    causal: bool = False,
    causal_type: int | None = None,
    window: tuple[int, int] | None = None,
    sm_scale: float | None = None,
    stream: torch.cuda.Stream | None = None,
    knobs: BwdDkDvKnobs | None = None,
    varlen: dict | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    batch_size: int | None = None,
    dropout_p: float | None = None,
    philox_seed: int | torch.Tensor = 0,
    philox_offset1: torch.Tensor | None = None,
    philox_offset2: int = 0,
    delta: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """dK and dV for one attention call on RDNA4 (gfx1201).

    Args:
        q, k, v: the forward's inputs, `[B, H, S, D]` (BHSD). `k`/`v` may carry
            fewer heads than `q` (MQA/GQA); `num_heads_q` must be divisible by
            `num_heads_k`.
        do: gradient of the loss w.r.t. the forward's output, shaped like `q`.
        o: the forward's output, shaped like `q`. Used only to form `delta`.
        lse: the forward's logsumexp, `(B, H, S)` or `(B*H, S)` fp32 under the
            default `VARLEN_LSE_LAYOUT_HT`. Natural-log units, `+inf` for a row
            with no live keys -- i.e. exactly what our forward writes.
        causal / causal_type / window: the same masking vocabulary the forward
            takes. `causal_type` 1 is top-left, 2 bottom-right, 3 an explicit
            `window=(left, right)`; both alignments travel to the kernel as
            sentinels and are resolved against *this sequence's* lengths.
        varlen: `None` for the dense case, else the dict one of the kernel
            module's `varlen_*` constructors returns. The forward's identically
            named constructors produce the same dict, and passing the forward's
            is the cheapest way to be certain the two agree.
        philox_seed: the dropout key, as an int or a one-element int64
            device tensor. The tensor form is the captured-graph one and is
            read on the device; an int is materialised here, as AOTriton's
            own caller does. The forward's `philox_seed_output` is exactly
            this argument's value and can be handed straight back.
        philox_offset1 / philox_offset2: the dropout counter, split the way
            `at::cuda::PhiloxCudaState` splits it -- a one-element int64 device
            tensor the kernel adds in, plus an immediate. `None` and an
            immediate is the uncaptured case; under CUDA graph capture the
            tensor is the counter the graph re-reads on replay, which is what
            keeps a replayed step from repeating one frozen mask. Must match
            what the forward was given, or the regenerated stream differs and
            the gradient is silently wrong.
        delta: `rowsum(dO * O)` if the caller has already formed it -- a fused
            preprocess kernel would supply this. Computed here when omitted.

    Returns:
        `(dk, dv)`, same shape and dtype as `k` and `v`.
    """
    for name, t in (("q", q), ("k", k), ("v", v), ("do", do), ("o", o)):
        if not t.is_cuda:
            raise ValueError(f"flydsl_flash_attn_bwd_dkdv_gfx1201 requires CUDA/HIP tensors, {name} is not")
        if t.device != q.device:
            raise ValueError(f"all tensors must share a device, got q={q.device} {name}={t.device}")
        if t.dim() != 4:
            raise ValueError(f"expected 4D BHSD {name}, got rank {t.dim()} ({tuple(t.shape)})")
        if t.dtype != q.dtype:
            raise ValueError(f"q/k/v/do/o dtype must match: {name} is {t.dtype}, q is {q.dtype}")
    try:
        arch = torch.cuda.get_device_properties(q.device.index).gcnArchName
    except Exception:
        arch = ""
    if not (arch.lower().split(":")[0] if arch else "").startswith("gfx1201"):
        raise ValueError(f"flydsl_flash_attn_bwd_dkdv_gfx1201 requires gfx1201, got {arch!r}")
    if k.shape != v.shape:
        raise ValueError(f"k and v must share shape, got k={tuple(k.shape)} v={tuple(v.shape)}")
    if do.shape != q.shape or o.shape != q.shape:
        raise ValueError(f"do and o must share q's shape {tuple(q.shape)}")
    if q.shape[0] != k.shape[0] or q.shape[3] != k.shape[3]:
        raise ValueError(f"q/k must share batch and head_dim, got q={tuple(q.shape)} k={tuple(k.shape)}")
    if q.shape[1] % k.shape[1]:
        raise ValueError(f"num_heads_q ({q.shape[1]}) must be divisible by num_heads_k ({k.shape[1]})")

    batch, num_heads, seq_len_q, head_dim = q.shape
    seq_len_k = k.shape[2]
    head_dim_v = v.shape[3]
    dtype_str = _torch_dtype_to_str(q.dtype)

    _plan = _plan_build(
        BwdDkDvMetadata(
            num_heads=num_heads,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            causal=causal,
            causal_type=causal_type,
            dtype_str=dtype_str,
            dropout=dropout_p is not None and float(dropout_p) > 0.0,
        ),
        knobs,
    )
    if _plan.knobs.padded_head:
        # The D-axis pitch must be a multiple of 16 bytes (8 elements at
        # f16/bf16). Same alignment contract as the forward, and load-bearing
        # for the same reason: accesses are 8 columns wide, so the chunk
        # containing `head_dim` runs to ceil8(head_dim) and must land inside
        # the tensor's own padding rather than in the next head.
        for name, t in (("q", q), ("k", k), ("v", v), ("do", do)):
            if t.stride(2) % 8:
                raise ValueError(
                    f"{name} has a D-axis pitch of {t.stride(2)} elements, not a multiple of 8 "
                    f"(16 bytes). head_dim {head_dim} is not a compiled tile width, so the kernel "
                    f"operates on ceil8 columns and needs the allocation padded to match."
                )

    lse2d = _as_row_2d(lse, "lse")
    if delta is None:
        # fp32 throughout: dO and O are 16-bit and their product summed over
        # head_dim loses too much in half precision, and this value is
        # subtracted from dP where the two are the same order of magnitude.
        delta2d = (do.float() * o.float()).sum(-1)
        delta2d = delta2d.reshape(delta2d.shape[0] * delta2d.shape[1], delta2d.shape[2]).contiguous()
    else:
        delta2d = _as_row_2d(delta, "delta")
    if delta2d.shape != lse2d.shape:
        raise ValueError(f"delta {tuple(delta2d.shape)} and lse {tuple(lse2d.shape)} must agree")

    def _alloc_like(t):
        pitch = (t.shape[3] + 7) // 8 * 8
        if pitch == t.shape[3]:
            return torch.zeros(*t.shape, dtype=t.dtype, device=t.device)
        return torch.zeros(t.shape[0], t.shape[1], t.shape[2], pitch, dtype=t.dtype, device=t.device)[..., : t.shape[3]]

    dk = _alloc_like(k)
    dv = _alloc_like(v)

    _mq = seq_len_q if max_seqlen_q is None else int(max_seqlen_q)
    _mk = seq_len_k if max_seqlen_k is None else int(max_seqlen_k)
    _bs = batch if batch_size is None else int(batch_size)

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
            dk,
            dv,
            lse2d,
            delta2d,
            _bs,
            _mq,
            seqlen_k=_mk,
            scale=sm_scale,
            stream=launch_stream,
            window=window,
            varlen=varlen,
            dropout_p=dropout_p,
            philox_seed=philox_seed,
            philox_offset1=philox_offset1,
            philox_offset2=philox_offset2,
        )
    return dk, dv
