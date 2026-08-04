# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""High-level FlyDSL Flash Attention API for gfx1201 / RDNA4.

Wraps ``flash_attn_func_gfx1201_aiw.build_flash_attn_func_aiw_module`` behind a
single function, ``flydsl_flash_attn_func_gfx1201(q, k, v, ...)``:

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

from flash_attn_func_gfx1201_aiw import (
    build_flash_attn_func_aiw_module,
    default_block_m as aiw_block_m,
    default_prefetch_dist as aiw_prefetch_dist,
)

__all__ = ["flydsl_flash_attn_func_gfx1201"]

# The unified kernel (``flash_attn_func_gfx1201_aiw``) is the only path. The
# three kernels it replaced were kept for a while as correctness oracles and
# have been retired; their final numbers are recorded under N2 in
# ``sdpa-close-gap-plan1.md``.


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


# The binding-prefetch kernel wins from head_dim 48 up; 16 and 32 still prefer
# the baseline (bp 35.4/60.7 against 37.3/61.2). Measured B=1 H=8 N=4096 f16
# non-causal. Below the threshold the tiles are small enough that bp's extra
# register-carried prefetch buys nothing.
_BP_MIN_HEAD_DIM = 48


def _use_bp(head_dim: int, use_binding_prefetch: bool, variant: str) -> bool:
    return variant != "m32" and (
        use_binding_prefetch or head_dim >= _BP_MIN_HEAD_DIM
    )


def _aiw_knobs(head_dim: int, use_binding_prefetch: bool, variant: str) -> dict:
    """Map the public options onto aiw's knob space.

    ``use_binding_prefetch`` forces prefetch distance 1 below the threshold
    where the policy would pick 0; ``variant="m32"`` selects two Q row-tiles
    per wave. Everything else follows the tuning policy in the kernel module.
    """
    dist = 1 if use_binding_prefetch else aiw_prefetch_dist(head_dim)
    knobs = {
        "k_prefetch_dist": dist,
        "v_lds_layout": "transposed" if dist else "row",
        "q_row_tiles": 2 if variant == "m32" else 1,
    }
    if variant == "m32":
        # Two row-tiles doubles o_accs + q_b_packs + s_accs; above head_dim 64
        # that crosses the 256-VGPR cap and spills (-27% measured at 128).
        if head_dim != 64:
            raise ValueError(
                f"variant='m32' (q_row_tiles=2) is head_dim 64 only, got {head_dim}"
            )
        knobs["k_prefetch_dist"] = 1
        knobs["v_lds_layout"] = "transposed"
    return knobs


# The compiled tile widths. `head_dim` is rounded up to the smallest of these
# that covers it; the real extent then rides along as a runtime argument and the
# kernel masks the difference. Mirrors AOTriton's `block_dmodel_values()`, which
# is the same list minus 384.
_BLOCK_DMODEL_LADDER = (16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 384, 512)


def _round_to_ladder(head_dim: int) -> int:
    """Smallest compiled tile width covering `head_dim`."""
    for w in _BLOCK_DMODEL_LADDER:
        if w >= head_dim:
            return w
    raise ValueError(
        f"head_dim {head_dim} exceeds the largest compiled tile "
        f"({_BLOCK_DMODEL_LADDER[-1]})"
    )




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
    padded_head: bool = False,
):
    kwargs = {}
    if head_dim_v is not None:
        # The V/output column window.
        kwargs = {"head_dim_v": head_dim_v, "d_offset": d_offset}
    kwargs.update(_aiw_knobs(head_dim, use_bp, variant))
    kwargs["padded_head"] = padded_head
    return build_flash_attn_func_aiw_module(
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
    return_lse: bool = False,
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
        use_binding_prefetch: force K prefetch distance 1 below the head_dim
            where the tuning policy would pick 0 (48). K and V prefetch
            distances are independent -- V is always staged one tile ahead --
            so this only moves K. Above the threshold it is already the default
            and the flag is a no-op.
        variant: ``""`` selects the default schedule. ``"m32"`` gives each
            wave two Q row-tiles, so one K/V operand feeds two WMMAs;
            head_dim 64 only.

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
    if k.shape != v.shape:
        raise ValueError(f"k and v must share shape, got k={tuple(k.shape)} v={tuple(v.shape)}")
    if q.dim() == 4 and k.dim() == 4:
        # MQA/GQA: several query heads share one KV head. Everything except the
        # head axis must still agree -- this is self-attention, so Lq == Lk.
        if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[3] != k.shape[3]:
            raise ValueError(
                "q/k must share batch, seq_len and head_dim; only num_heads may "
                f"differ (GQA). Got q={tuple(q.shape)} k={tuple(k.shape)}"
            )
        if q.shape[2] % k.shape[2]:
            raise ValueError(
                f"num_heads_q ({q.shape[2]}) must be divisible by num_heads_k "
                f"({k.shape[2]})"
            )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"q/k/v dtype must match: {q.dtype}/{k.dtype}/{v.dtype}")
    if q.dim() != 4:
        raise ValueError(f"expected 4D BSHD tensor, got rank {q.dim()} ({tuple(q.shape)})")
    if variant not in ("", "m32"):
        raise ValueError(
            f"unknown variant {variant!r}; expected '' (unified kernel) or 'm32'"
        )

    batch, seq_len_real, num_heads, head_dim = q.shape
    if head_dim < 1 or head_dim > _MAX_HEAD_DIM:
        raise ValueError(f"kernel requires 1 <= head_dim <= {_MAX_HEAD_DIM}, got {head_dim}")
    # The tile width is a build axis; the real extent is a runtime argument.
    block_dmodel = _round_to_ladder(head_dim)
    padded_head = block_dmodel != head_dim
    if padded_head:
        # The D-axis pitch must be a multiple of 16 bytes -- 8 elements at
        # f16/bf16. This is the alignment contract (see sdpa-close-gap-plan1.md
        # section 3), upheld upstream by the shim layer shared by the CUDA and
        # ROCm backends, which pads the last dimension before dispatch exactly
        # as `pad_last_dim` does.
        #
        # It is load-bearing here, not decorative. Loads and stores are 8
        # columns wide, so the chunk containing `head_dim` runs to
        # ceil8(head_dim); an 8-aligned pitch guarantees that lands inside the
        # tensor's own padding. A tightly-packed tensor -- e.g. contiguous
        # (B, S, H, 100), pitch 100 -- has no such padding, and the store at
        # column 96 would write columns 100..103 of the *next head*.
        for name, t in (("q", q), ("k", k), ("v", v)):
            if t.stride(2) % 8:
                raise ValueError(
                    f"{name} has a D-axis pitch of {t.stride(2)} elements, which is "
                    f"not a multiple of 8 (16 bytes). head_dim {head_dim} is not a "
                    f"compiled tile width, so the kernel operates on "
                    f"ceil8({head_dim})={(head_dim + 7) // 8 * 8} columns and needs "
                    f"the allocation padded to match. Pad the last dimension before "
                    f"calling, as PyTorch's SDPA shim does."
                )
    dtype_str = _torch_dtype_to_str(q.dtype)

    # Everything below keys on the *tile* width, per N3: BLOCK_DMODEL is the
    # single tuning key, and hdim rides along as a runtime argument.
    use_bp = _use_bp(block_dmodel, use_binding_prefetch, variant)
    # BLOCK_M is invariant to q_row_tiles by construction (see the
    # Q_TILES_PER_BLOCK comment in the kernel), so only the prefetch distance
    # matters here.
    block_m = aiw_block_m(
        block_dmodel, 1 if use_bp else aiw_prefetch_dist(block_dmodel)
    )
    # The KV tail is masked in-kernel, so seq_len is passed through as-is.
    #
    # It used to be rounded up to BLOCK_M with F.pad -- three tensor copies per
    # call (~25 MB at B=1 H=8 N=4096 d=128 f16) -- and non-causal calls with
    # more than 0.5% padding were *rejected*, because a zero-padded key gives
    # QK^T = 0 and exp(0) = 1 still lands in the softmax denominator, silently
    # scaling the output. Neither the copy nor the rejection is needed now: the
    # tail columns are masked to -inf and contribute nothing.
    seq_len_pad = seq_len_real
    # No `.contiguous()`: the kernel reads strides, so any xxxD permutation is
    # already supported, and forcing a copy would both cost a copy and destroy
    # a padded D pitch (re-packing a (.., 100)-in-104 view back to a tight 100).
    q_p, k_p, v_p = q, k, v

    # Allocate O with the D axis padded to the same 8-element multiple the
    # inputs are required to have. `torch.empty_like` would give a tightly
    # packed tensor, whose last 8-wide store chunk would spill into the next
    # head. See the pitch check above.
    _o_pitch = (head_dim + 7) // 8 * 8
    if _o_pitch == head_dim:
        o_p = torch.empty(
            batch, seq_len_pad, num_heads, head_dim, dtype=q.dtype, device=q.device
        )
    else:
        o_p = torch.empty(
            batch, seq_len_pad, num_heads, _o_pitch, dtype=q.dtype, device=q.device
        )[..., :head_dim]

    # logsumexp, (B*H, S_pad) fp32 -- AOTriton's non-varlen layout. Sliced back
    # to the real seq_len on return, like O.
    if return_lse:
        lse_p = torch.empty(
            batch * num_heads, seq_len_pad, dtype=torch.float32, device=q.device
        )
    else:
        lse_p = None

    # Wrap kernel build + launch in q.device context so multi-GPU callers
    # whose current device differs from q.device get the kernel compiled
    # and launched on the right device/stream.
    with torch.cuda.device(q.device.index):
        launch_stream = torch.cuda.current_stream(q.device) if stream is None else stream
        if launch_stream.device != q.device:
            raise ValueError(f"`stream` must be on {q.device}, got {launch_stream.device}")
        exe = _get_kernel(
            num_heads=num_heads,
            head_dim=block_dmodel,
            causal=causal,
            dtype_str=dtype_str,
            waves_per_eu=waves_per_eu,
            daz=daz,
            use_bp=use_bp,
            variant=variant,
            head_dim_v=None,
            d_offset=0,
            padded_head=padded_head,
        )
        # Whole tensors, not `.reshape(-1)`: the kernel reads strides, and
        # flattening materialises a copy for any non-contiguous input, which
        # would silently defeat the point of reading them.
        exe(q_p, k_p, v_p, o_p, batch, seq_len_pad,
            stream=launch_stream, lse=lse_p)

    out = (
        o_p[:, :seq_len_real, :, :].contiguous()
        if seq_len_pad != seq_len_real
        else o_p
    )
    if not return_lse:
        return out
    lse = lse_p.view(batch, num_heads, seq_len_pad)[:, :, :seq_len_real]
    return out, lse
