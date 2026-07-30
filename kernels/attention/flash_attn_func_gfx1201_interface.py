# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""High-level FlyDSL Flash Attention API for gfx1201 / RDNA4.

Wraps ``flash_attn_func_gfx1201.build_flash_attn_func_module`` behind a single
function, ``flydsl_flash_attn_func_gfx1201(q, k, v, ...)``:

- Build cache keyed by (num_heads, head_dim, causal, dtype, waves_per_eu, daz)
  so repeated calls with the same static config compile only once per process.
- BSHD (``[B, S, H, D]``) input/output convention, matching upstream
  flash-attention layout.
- Automatic seq_len padding to the kernel's ``BLOCK_M`` tile (128).
- Non-causal padding-ratio safety guard: padded K/V tokens produce
  ``QK^T = 0``, but ``exp(0) = 1`` still contributes to the softmax
  denominator and silently scales the output. Calls with
  ``n_pad / seq_len_pad > 0.005`` (0.5%) and ``causal=False`` are rejected
  with a ``ValueError``.
- Unified device / stream context so multi-GPU callers whose current device
  differs from ``q.device`` still compile and launch on the right device.

The kernel implements self-attention only (Lq == Lk). Cross-attention
(Lq != Lk) is rejected; callers should fall back to PyTorch SDPA.

The gfx950 / gfx942 equivalents live in ``flash_attn_interface.py``; this
module is deliberately separate because the gfx1201 kernel has its own
builder, tile size, and padding contract.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

from flash_attn_func_gfx1201 import build_flash_attn_func_module, default_block_m
from flash_attn_func_gfx1201_bp import build_flash_attn_func_bp_module, bp_block_m
from flash_attn_func_gfx1201_m32 import build_flash_attn_func_m32_module

__all__ = ["flydsl_flash_attn_func_gfx1201"]


# Tile size baked into the gfx1201 kernel (BLOCK_M). Seq_len must be a
# multiple of this; padding is invisible to callers.
# BLOCK_M is head_dim-dependent; see default_block_m() in the kernel module.
_KERNEL_BLOCK_M = 128

# A wave's output accumulator is head_dim/2 VGPRs, so an unsliced head_dim 512
# would need the whole 256-VGPR file before anything else, and its K+V LDS tile
# would exceed the 64 KB workgroup limit (66048 B). Both are avoided by slicing
# the V/output width -- see "head_dim above 256" in gfx1201_fmha.md.
_MAX_HEAD_DIM = 512

# Above this QK width the V/output side is computed in column slices of this
# size, so that o_accs (head_dim_v/2 VGPRs) and the V LDS tile stay in budget.
_V_SLICE_ABOVE = 256
_V_SLICE_WIDTH = 128


# head_dims routed to the binding-prefetch kernel by default. Its head-dim
# sharding beats the baseline's launch-level V slicing here, measured at
# B=1 H=8 N=4096 f16 non-causal: 256 67.5 -> 75.5, 384 36.9 -> 45.1. head_dim
# 512 joined once V staging was chunked (vo_chunks): staging half the V
# columns at a time keeps the padded K+V tile inside 64 KiB, which restores
# conflict-free LDS and took it 22.4 -> 41.4.
_BP_HEAD_DIMS = frozenset({256, 384, 512})


def _use_bp(head_dim: int, use_binding_prefetch: bool, variant: str) -> bool:
    return variant != "m32" and (use_binding_prefetch or head_dim in _BP_HEAD_DIMS)


def _v_slice_width(head_dim: int) -> int:
    """Column-slice width for V/O; head_dim itself means no slicing."""
    if head_dim <= _V_SLICE_ABOVE:
        return head_dim
    assert head_dim % _V_SLICE_WIDTH == 0, head_dim
    return _V_SLICE_WIDTH

# Maximum tolerated ratio of padded tokens for non-causal attention.
# 0.5% is the bf16 mantissa precision floor (~0.4%) plus 1 bit of margin.
# Above this the relative error grows quickly (50% pad -> 37% rel_err).
# Causal mode masks future tokens including the padded ones, so it is
# unaffected. (Rationale documented in aiter's 2969_padded_softmax_rca.md.)
_MAX_NONCAUSAL_PAD_RATIO = 0.005


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "f16"
    raise ValueError(f"flydsl_flash_attn_func_gfx1201 only supports bf16/f16, got {dtype!r}")


@lru_cache(maxsize=32)
def _get_kernel(
    num_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    waves_per_eu: int,
    daz: bool,
    use_bp: bool,
    variant: str = "",
    head_dim_v: int | None = None,
    d_offset: int = 0,
):
    if variant == "m32":
        builder = build_flash_attn_func_m32_module
    elif use_bp:
        builder = build_flash_attn_func_bp_module
    else:
        builder = build_flash_attn_func_module
    kwargs = {}
    if head_dim_v is not None:
        # Only the baseline builder splits the V/output width; bp and m32 do not
        # take these, and are gated to head_dim <= 128 anyway.
        kwargs = {"head_dim_v": head_dim_v, "d_offset": d_offset}
    return builder(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        waves_per_eu=waves_per_eu,
        daz=daz,
        **kwargs,
    )


def flydsl_flash_attn_func_gfx1201(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    waves_per_eu: int = 2,
    daz: bool = True,
    stream: torch.cuda.Stream | None = None,
    use_binding_prefetch: bool = False,
    variant: str = "",
) -> torch.Tensor:
    """Run FlyDSL Flash Attention on RDNA4 (gfx1201).

    Args:
        q, k, v: tensors with shape ``[batch, seq_len, num_heads, head_dim]``
            (BSHD). All three must share dtype, batch, num_heads, head_dim,
            and seq_len. Must reside on a CUDA/HIP device. ``head_dim`` must be
            a multiple of 16 and at most 512; above 256 it is computed in
            128-wide V column slices and so must be a multiple of 128.
        causal: apply causal masking when ``True``.
        waves_per_eu: kernel occupancy hint passed to the FlyDSL builder.
        daz: enable denormals-are-zero on the kernel.
        stream: optional CUDA/HIP stream to launch on. Defaults to the current
            stream for ``q.device``.
        use_binding_prefetch: select the binding-prefetch scheduling variant
            (``flash_attn_func_gfx1201_bp``), which carries both K and V tiles
            in registers at prefetch distance 1 instead of loading K at
            distance 0. Stage 1: correctness-oriented and not yet tuned, and it
            only accepts head_dim 64/128. Defaults to the baseline kernel.

    Returns:
        Output tensor with the same shape and dtype as ``q``.

    Raises:
        ValueError: if shapes/dtypes/devices are incompatible, the kernel's
            ``head_dim`` constraints are not met, or the non-causal padding
            ratio ``n_pad / seq_len_pad`` exceeds 0.5% (see module docstring
            for rationale).
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_func_gfx1201 requires CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(f"q/k/v must reside on the same device, got q={q.device} k={k.device} v={v.device}")
    try:
        arch = torch.cuda.get_device_properties(q.device.index).gcnArchName
    except Exception:
        arch = ""
    arch_base = arch.lower().split(":")[0] if arch else ""
    if not arch_base.startswith("gfx1201"):
        raise ValueError(f"flydsl_flash_attn_func_gfx1201 requires gfx1201, got {arch!r}")
    if not (q.shape == k.shape == v.shape):
        raise ValueError(
            "flydsl_flash_attn_func_gfx1201 is self-attention; q/k/v must share "
            f"shape, got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"q/k/v dtype must match: {q.dtype}/{k.dtype}/{v.dtype}")
    if q.dim() != 4:
        raise ValueError(f"expected 4D BSHD tensor, got rank {q.dim()} ({tuple(q.shape)})")

    batch, seq_len_real, num_heads, head_dim = q.shape
    if head_dim < 16 or head_dim % 16 != 0 or head_dim > _MAX_HEAD_DIM:
        raise ValueError(
            f"kernel requires 16 <= head_dim <= {_MAX_HEAD_DIM} and head_dim % 16 == 0, "
            f"got {head_dim}"
        )
    if head_dim > _V_SLICE_ABOVE and head_dim % _V_SLICE_WIDTH != 0:
        raise ValueError(
            f"head_dim above {_V_SLICE_ABOVE} is computed in V column slices of "
            f"{_V_SLICE_WIDTH}, so it must be a multiple of {_V_SLICE_WIDTH}; got {head_dim}"
        )

    dtype_str = _torch_dtype_to_str(q.dtype)

    # Pad seq_len up to the kernel's tile size. Tight padding (<= 0.5% of
    # S_pad) is empirically below the bf16 noise floor on production shapes.
    # Higher ratios are rejected above: padded K/V tokens produce QK^T = 0
    # but exp(0) = 1 still contributes to the softmax denominator and would
    # scale the output. Padded queries produce garbage rows that we slice
    # off before returning.
    # Must match the kernel's own choice: it sets the seq_len padding below.
    use_bp = _use_bp(head_dim, use_binding_prefetch, variant)
    if variant == "m32":
        block_m = 256
    elif use_bp:
        block_m = bp_block_m(head_dim)
    else:
        block_m = default_block_m(head_dim)
    seq_len_pad = ((seq_len_real + block_m - 1) // block_m) * block_m
    n_pad = seq_len_pad - seq_len_real
    if not causal and n_pad > 0 and n_pad / seq_len_pad > _MAX_NONCAUSAL_PAD_RATIO:
        raise ValueError(
            "flydsl_flash_attn_func_gfx1201: non-causal path with padding ratio "
            f"{n_pad}/{seq_len_pad}={n_pad / seq_len_pad:.4f} exceeds 0.5% "
            "safety threshold; padded K/V tokens contribute to softmax "
            "denominator and would scale outputs. Either set causal=True, "
            "pad seq_len to a multiple of 128 before calling, or use a "
            "self-attn kernel with explicit attention masking."
        )
    if seq_len_pad != seq_len_real:
        # F.pad pads from the last dim; for BSHD (last=head_dim) the seq dim
        # is dim 1, so we pad (D_left, D_right, H_left, H_right, S_left, S_right).
        q_p = F.pad(q.contiguous(), (0, 0, 0, 0, 0, n_pad))
        k_p = F.pad(k.contiguous(), (0, 0, 0, 0, 0, n_pad))
        v_p = F.pad(v.contiguous(), (0, 0, 0, 0, 0, n_pad))
    else:
        q_p = q.contiguous()
        k_p = k.contiguous()
        v_p = v.contiguous()

    o_p = torch.empty_like(q_p)

    # Wrap kernel build + launch in q.device context so multi-GPU callers
    # whose current device differs from q.device get the kernel compiled
    # and launched on the right device/stream.
    with torch.cuda.device(q.device.index):
        launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
        if launch_stream.device != q.device:
            raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
        # A wave's output accumulator is head_dim_v/2 VGPRs and the V LDS tile
        # scales with head_dim_v, so wide heads are computed one column slice at
        # a time. Sound because attention is column-separable in V:
        # O[:, s] = P @ V[:, s], and P does not depend on V. GEMM1 and the K
        # traffic repeat per slice, which is what makes this a fallback rather
        # than the default.
        slice_w = head_dim if use_bp else _v_slice_width(head_dim)
        for d_off in range(0, head_dim, slice_w):
            exe = _get_kernel(
                num_heads=num_heads,
                head_dim=head_dim,
                causal=causal,
                dtype_str=dtype_str,
                waves_per_eu=waves_per_eu,
                daz=daz,
                use_bp=use_bp,
                variant=variant,
                head_dim_v=None if slice_w == head_dim else slice_w,
                d_offset=0 if slice_w == head_dim else d_off,
            )
            exe(
                q_p.reshape(-1),
                k_p.reshape(-1),
                v_p.reshape(-1),
                o_p.reshape(-1),
                batch,
                seq_len_pad,
                stream=launch_stream,
            )

    if seq_len_pad != seq_len_real:
        return o_p[:, :seq_len_real, :, :].contiguous()
    return o_p
