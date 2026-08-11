# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host side of the gfx1201 attention ABI: the wire spec, and marshalling to it.

Everything the *caller* has to get right, in one place. Four kernels now share
this ABI -- the forward and the three backward ones -- and before this module
existed each carried its own copy of all of it. That is not merely repetitive:
`VarlenBits` is a **wire encoding**, and three copies of a wire encoding are
three chances for it to drift while every test still passes, because each
kernel is validated against its own copy.

**Not `fmha_common_gfx1201.py`.** That module is device-side: it emits IR, and
its central constraint is what the AST rewriter will and will not do to a
module-level function. Nothing here emits IR. These are plain Python functions
that run on the host before a launch, turning torch tensors and Python
arguments into the scalars and pointers the kernel signature wants. The two
concerns fail differently and are read at different times.

The three functions that take a build knob as their first argument
(`resolve_window`, `dropout_args`, `varlen_args`) are the only ones that are
not pure. They were closures over the builder's `const_expr` state; making the
dependency an argument is what let them move, and it also makes visible which
host behaviour a knob reaches -- something the closure form hid.
"""

from __future__ import annotations

import contextlib
import weakref

import fmha_common_gfx1201 as fmha
import torch
from philox import dropout_threshold
from torch import float32 as torch_f32

import flydsl.compiler as flyc
import flydsl.expr as fx

__all__ = [
    "VARLEN_STACKED",
    "VARLEN_LENGTH_MAX",
    "VARLEN_LENGTH_CUMULATIVE",
    "VARLEN_LENGTH_INDIVIDUAL",
    "VARLEN_POSITION_IMPLIED",
    "VARLEN_POSITION_REUSE",
    "VARLEN_POSITION_ARRAY",
    "VARLEN_LSE_LAYOUT_HT",
    "VARLEN_LSE_LAYOUT_TH",
    "VARLEN_DENSE",
    "VARLEN_COMPACT_SIDE",
    "CAUSAL_SENTINEL",
    "varlen_bits",
    "varlen_compact",
    "varlen_padded",
    "varlen_strided",
    "varlen_seqused_k",
    "NULL_PTR",
    "ptr_arg",
    "strides_of",
    "lse_args",
    "resolve_window",
    "dropout_args",
    "dropout_outputs",
    "u64_scalar",
    "varlen_args",
    "run_compiled",
    "new_compiled_cache",
    "dtype_to_elem_type",
]


# Resolving in one place removes the class of bug rather than the
# instance.
CAUSAL_SENTINEL = {
    1: fmha.WINDOW_TOPLEFT,  # j <= i
    2: fmha.WINDOW_BOTRIGHT,  # j <= i + (seqlen_k - seqlen_q)
}

# ---- VarlenBits, sdpa-varlen-plan.md section 2 ----
#
# One byte per side, decoded by the same kernel-side function twice, plus
# the LSE layout in byte 2. `0` is BHSD / MAX / IMPLIED on both sides with
# an (H, T) logsumexp -- the conventional dense case, and the default.
VARLEN_STACKED = 1
VARLEN_LENGTH_MAX = 0 << 1  # noqa: F841  (the table is the wire spec)
VARLEN_LENGTH_CUMULATIVE = 1 << 1
VARLEN_LENGTH_INDIVIDUAL = 2 << 1
VARLEN_POSITION_IMPLIED = 0 << 3
VARLEN_POSITION_REUSE = 1 << 3
VARLEN_POSITION_ARRAY = 2 << 3
# `_HT` is AOTriton's and this kernel's default: shape (H, T), T
# contiguous. `_TH` is Transformer Engine's (T, H).
VARLEN_LSE_LAYOUT_HT = 0 << 16
VARLEN_LSE_LAYOUT_TH = 1 << 16  # noqa: F841  (ditto)


def varlen_bits(q_side=0, k_side=0, lse_layout=VARLEN_LSE_LAYOUT_HT):
    """Assemble VarlenBits from per-side bytes."""
    for name, b in (("q_side", q_side), ("k_side", k_side)):
        if not 0 <= b <= 0xFF:
            raise ValueError(f"{name} must fit in a byte, got {b:#x}")
        if (b >> 3) & 3 == 1 and (b >> 1) & 3 != 1:
            # REUSE takes a *position* out of the length array, which is
            # only a position when the lengths are cumulative.
            raise ValueError(f"{name}={b:#04x}: POSITION=REUSE requires " f"LENGTH=CUMULATIVE (plan section 1, axis C)")
        if (b >> 1) & 3 == 3 or (b >> 3) & 3 == 3:
            raise ValueError(f"{name}={b:#04x} uses a reserved code")
    return q_side | (k_side << 8) | lse_layout


VARLEN_DENSE = 0  # noqa: F841  (ditto)
VARLEN_COMPACT_SIDE = VARLEN_STACKED | VARLEN_LENGTH_CUMULATIVE | VARLEN_POSITION_REUSE  # 0x0B
VARLEN_PADDED_SIDE = VARLEN_LENGTH_CUMULATIVE | VARLEN_POSITION_IMPLIED
# 0x02

VARLEN_STRIDED_SIDE = VARLEN_STACKED | VARLEN_LENGTH_CUMULATIVE | VARLEN_POSITION_ARRAY  # 0x13
VARLEN_SEQUSED_PACKED_SIDE = VARLEN_STACKED | VARLEN_LENGTH_INDIVIDUAL | VARLEN_POSITION_ARRAY  # 0x15
VARLEN_SEQUSED_CACHE_SIDE = VARLEN_LENGTH_INDIVIDUAL | VARLEN_POSITION_IMPLIED  # 0x04


def varlen_compact(
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, lse_tokens=None, lse_layout=VARLEN_LSE_LAYOUT_HT
):
    """Classical packed varlen: 1THD tensors, `cu_seqlens` for both roles.

    `seqinfo_?1` is deliberately **not** passed: `POSITION = REUSE` takes
    the position out of the cumulative length value already loaded, so
    this configuration reads no position array at all (plan section 1.2).
    """
    return dict(
        bits=varlen_bits(VARLEN_COMPACT_SIDE, VARLEN_COMPACT_SIDE, lse_layout),
        seqinfo_q0=cu_seqlens_q,
        seqinfo_q1=None,
        seqinfo_k0=cu_seqlens_k,
        seqinfo_k1=None,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        lse_tokens=lse_tokens,
    )


def varlen_padded(
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, lse_tokens=None, lse_layout=VARLEN_LSE_LAYOUT_HT
):
    """BHSD tensors whose sequences are short: lengths only, no positions."""
    return dict(
        bits=varlen_bits(VARLEN_PADDED_SIDE, VARLEN_PADDED_SIDE, lse_layout),
        seqinfo_q0=cu_seqlens_q,
        seqinfo_q1=None,
        seqinfo_k0=cu_seqlens_k,
        seqinfo_k1=None,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        lse_tokens=lse_tokens,
    )


def varlen_strided(
    cu_seqlens_q,
    cu_seqlens_k,
    seq_strides_q,
    seq_strides_k,
    max_seqlen_q,
    max_seqlen_k,
    lse_tokens=None,
    lse_layout=VARLEN_LSE_LAYOUT_HT,
):
    """Packed tensors with padding *between* sequences (TE's layout).

    Differs from `varlen_compact` in one thing only: positions come from a
    second array instead of being reused from the length array. That is
    the whole of AOTriton's `StridedVarlen`, and the reason the two roles
    must never be swapped -- `seq_strides` differences are *padded*
    extents, not lengths.
    """
    return dict(
        bits=varlen_bits(VARLEN_STRIDED_SIDE, VARLEN_STRIDED_SIDE, lse_layout),
        seqinfo_q0=cu_seqlens_q,
        seqinfo_q1=seq_strides_q,
        seqinfo_k0=cu_seqlens_k,
        seqinfo_k1=seq_strides_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        lse_tokens=lse_tokens,
    )


def varlen_seqused_k(
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    k_is_cache=False,
    lse_tokens=None,
    lse_layout=VARLEN_LSE_LAYOUT_HT,
):
    """Packed Q against a KV cache with per-sequence *used* lengths.

    `torch.nn.attention.varlen`'s `seqused_k`, and the configuration no
    `VarlenType` can express: the K side takes its **length** from an
    individual array and its **position** from a cumulative one, so the
    two axes read different tensors.

    `k_is_cache=True` is the rectangular variant -- a BHSD cache with no
    `cu_seqlens_k` at all, where the position is implied by the batch
    index.
    """
    k_side = VARLEN_SEQUSED_CACHE_SIDE if k_is_cache else VARLEN_SEQUSED_PACKED_SIDE
    return dict(
        bits=varlen_bits(VARLEN_COMPACT_SIDE, k_side, lse_layout),
        seqinfo_q0=cu_seqlens_q,
        seqinfo_q1=None,
        seqinfo_k0=seqused_k,
        seqinfo_k1=None if k_is_cache else cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        lse_tokens=lse_tokens,
    )


NULL_PTR = flyc.from_c_void_p(fx.Uint8, 0)


def ptr_arg(t):
    if hasattr(t, "data_ptr"):
        type_name = type(t).__name__
        module_name = type(t).__module__
        ptr = 0 if type_name == "FakeTensor" or "fake_tensor" in module_name else t.data_ptr()
        return flyc.from_c_void_p(fx.Uint8, ptr)
    return t


def strides_of(t, name):
    """`(batch, head, seq)` strides of a rank-4 BHSD tensor, in elements.

    **Shape and layout are different things**, and this reads the second.
    *Shape* is `t.shape`, and the kernel requires it to be BHSD --
    `(batch, num_heads, seq_len, head_dim)` -- because that is what fixes
    which axis each of the three returned slots describes. *Layout* is how
    the data actually sits in memory, and any permutation is accepted so
    long as D is innermost, which is what `stride(3) == 1` checks. A BSHD
    *layout* is therefore fine; pass it as `t.transpose(1, 2)`, which has
    BHSD shape.

    The order is the ABI. AOTriton dispatches the compiled hsaco directly
    rather than through the Python wrapper, so these three slots are the
    contract, and a caller that fills them from a BSHD shape swaps head
    with sequence -- reading heads where the kernel means tokens, which
    produces finite garbage and never faults.
    """
    if not hasattr(t, "stride"):
        raise TypeError(f"{name} must be a rank-4 tensor so its strides can be read, " f"got {type(t).__name__}")
    if t.dim() != 4:
        raise ValueError(f"{name} must be rank 4, got shape {tuple(t.shape)}")
    if t.stride(3) != 1:
        raise ValueError(f"{name} must have a contiguous last dimension, got " f"stride(3)={t.stride(3)}")
    return t.stride(0), t.stride(1), t.stride(2)


def lse_args(lse, seq_len, varlen, num_head_q):
    """The logsumexp pointer, and a check that its layout matches the bits.

    Returns only a pointer: unlike Q/K/V/O this tensor is always compact,
    so the kernel derives both pitches from `LSE_LAYOUT`, `num_head_q` and
    the token count (sdpa-varlen-plan.md section 4.2). Inferring strides
    here as well would give one fact two sources.

    What the host can do instead -- and could not while it was inferring --
    is verify the caller's tensor actually has the declared layout.
    """
    if lse is None:
        return NULL_PTR
    if lse.dtype != torch_f32:
        raise ValueError(f"logsumexp must be float32, got {lse.dtype}")
    if lse.dim() != 2:
        raise ValueError(f"logsumexp must be rank 2, got shape {tuple(lse.shape)}")
    if not lse.is_contiguous():
        raise ValueError(
            "logsumexp must be contiguous: the kernel derives its pitches "
            "from VarlenBits rather than reading strides"
        )
    _layout = 0 if varlen is None else (int(varlen["bits"]) >> 16) & 3
    if _layout == 0:
        # The token pitch the kernel will derive. Only checkable when the
        # caller supplies it: under a stacked layout it lives in
        # `seqinfo[N]` on the device, and reading that back would cost a
        # sync for a validation.
        want_last = int(seq_len) if varlen is None else varlen.get("lse_tokens")
        if want_last is not None and lse.shape[1] != int(want_last):
            raise ValueError(f"VARLEN_LSE_LAYOUT_HT wants (*, {int(want_last)}), got " f"{tuple(lse.shape)}")
    else:
        if lse.shape[1] != num_head_q:
            raise ValueError(f"VARLEN_LSE_LAYOUT_TH wants (*, {num_head_q}), got " f"{tuple(lse.shape)}")
    return ptr_arg(lse)


def resolve_window(causal_type, host_causal_type, window, seqlen_q, seqlen_k):
    """(window_left, window_right), signed, as the kernel wants them.

    Non-causal still forwards a pair so both arms share one ABI and stay
    directly comparable -- the same reason the strides are always passed
    even under strides_constexpr.

    Causal alignments forward a *sentinel*, not a bound: the kernel
    resolves it per sequence, which is the only correct thing to do once
    there is more than one sequence.
    """
    if causal_type == 0:
        if window is not None:
            # Silently dropping it would return dense attention that is
            # the right shape, finite, and wrong -- and a window is only
            # ever passed by a caller who believes it is being applied.
            # The non-causal arm has no left-masked region to apply one
            # with, so this is a build-time choice, not a runtime one.
            raise ValueError(
                "window= requires a causal build; this one has "
                "causal=False. Pass causal=True, causal_type=3 for "
                "generalized sliding-window attention"
            )
        return 0, 0
    if host_causal_type in CAUSAL_SENTINEL:
        if window is not None:
            raise ValueError(
                f"causal_type={host_causal_type} already fixes the window; " "pass causal_type=3 to choose one"
            )
        _s = CAUSAL_SENTINEL[host_causal_type]
        wl, wr = _s, _s
    else:
        if window is None:
            raise ValueError(
                "causal_type=3 is generalized sliding-window attention and "
                "requires window=(left, right); "
                f"pass ({seqlen_q}, 0) for top-left causal or "
                f"({seqlen_q}, {seqlen_k - seqlen_q}) for bottom-right"
            )
        wl, wr = window
    return int(wl), int(wr)


def u64_scalar(value, device, stream=None):
    """A one-element device u64 holding `value`, or `value` if already one.

    `None` stays `None` (the null case). A tensor is taken as-is and *not*
    copied, which is the point: under graph capture the caller owns a counter
    the graph re-reads, and copying it here would freeze it again.

    An int is materialised, which is what AOTriton's host does with its own
    `DEFAULT_PHILOX_SEED`. `torch.int64` and not a uint64: torch has no
    unsigned 64-bit dtype, and the kernel reads the same eight bytes either
    way. Allocated on `stream` when one is given, because the kernel that
    reads it runs there.

    **The returned tensor must stay referenced until the launch is enqueued.**
    Only its raw pointer reaches the kernel, so nothing else keeps it alive --
    callers bind it to a local that outlives the launch call.
    """
    if value is None or hasattr(value, "data_ptr"):
        if value is not None and value.numel() < 1:
            raise ValueError("a philox scalar tensor must hold at least one element")
        if value is not None and value.element_size() != 8:
            raise ValueError(f"a philox scalar tensor must be 8 bytes per element, got {value.dtype}")
        return value
    with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
        return torch.tensor([int(value)], dtype=torch.int64, device=device)


def dropout_outputs(enable_dropout, seed_output, offset_output):
    """(seed_output, offset_output) in launch order -- the forward's write-back.

    The forward records the `(seed, offset)` it actually drew from so the
    backward can be handed them rather than re-deriving them. That matters
    only under graph capture, where the effective offset is a sum formed on
    the device out of a counter the host cannot read without synchronising.
    Either may be `None`, meaning the caller does not want the value.
    """
    if not enable_dropout:
        return NULL_PTR, NULL_PTR
    return (
        NULL_PTR if seed_output is None else ptr_arg(seed_output),
        NULL_PTR if offset_output is None else ptr_arg(offset_output),
    )


def dropout_args(enable_dropout, dropout_p, seed, offset1, offset2, device=None, stream=None):
    """(seed, offset1, offset2, threshold, 1/(1-p), keepalive) in launch order.

    The trailing `keepalive` is not a kernel argument. It is whatever tensor
    `u64_scalar` had to materialise for an int seed, returned so the caller
    can hold it across the launch; see that function.

    The counter is the pair torch splits it into, not one pre-summed scalar:
    `offset1` is a device pointer the kernel adds in when non-null, `offset2`
    an immediate. See `fmha.philox_offset_base` for why a captured CUDA graph
    needs the pointer half, and `philox_offset1`/`philox_offset2` in AOTriton
    for the ABI this matches.

    `offset1` may be a tensor holding one u64, or None for the uncaptured
    case. A one-element `torch.int64` tensor is the same 8 bytes as the `u64`
    the kernel reads; torch has no unsigned 64-bit dtype to spell it with, and
    the counter is far from the sign bit.

    The threshold and the scale are computed here, once per call, rather
    than per element in the kernel -- `philox.dropout_threshold` turns the
    probability into an i32 the raw random can be compared against, so the
    hot path never converts a random to a float.
    """
    if not enable_dropout:
        return NULL_PTR, NULL_PTR, 0, 0, 1.0, None
    if dropout_p is None:
        raise ValueError("this build has dropout=True and requires dropout_p=")
    p = float(dropout_p)
    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout_p must be in [0, 1), got {p}")
    seed_t = u64_scalar(seed, device, stream)
    off1_t = u64_scalar(offset1, device, stream)
    return (
        NULL_PTR if seed_t is None else ptr_arg(seed_t),
        NULL_PTR if off1_t is None else ptr_arg(off1_t),
        int(offset2),
        dropout_threshold(p),
        1.0 / (1.0 - p),
        (seed_t, off1_t),
    )


def varlen_args(strides_constexpr, varlen, seqlen_q, seqlen_k):
    """(bits, q0, q1, k0, k1, max_q, max_k) in launch order.

    `varlen` is None for the dense case, else a dict with `bits` and
    whichever `seqinfo_*` tensors that configuration reads. Unread slots
    stay **null**, which is safe because the kernel's decode branches
    rather than selects -- see the prologue.
    """
    if varlen is None:
        return (0, NULL_PTR, NULL_PTR, NULL_PTR, NULL_PTR, int(seqlen_q), int(seqlen_k))
    if strides_constexpr:
        raise ValueError(
            "strides_constexpr derives the layout from the shape, which "
            "varlen invalidates; it is a dense-only diagnostic arm"
        )
    # No implemented-subset gate: every encodable side byte now decodes,
    # since the decoder is one function covering all three axis values.
    # `varlen_bits` rejects the combinations that are not *meaningful*
    # (reserved codes, REUSE without cumulative lengths).
    bits = int(varlen["bits"])
    got = tuple(
        ptr_arg(varlen[k]) if varlen.get(k) is not None else NULL_PTR
        for k in ("seqinfo_q0", "seqinfo_q1", "seqinfo_k0", "seqinfo_k1")
    )
    return (bits,) + got + (int(varlen["max_seqlen_q"]), int(varlen["max_seqlen_k"]))


def run_compiled(cache, exe, *args):
    """First call compiles and runs; later calls fast-dispatch the compiled fn.

    `cache` is the caller's `weakref.WeakKeyDictionary`, not a module-level one:
    four kernels share this function and a single dict keyed by launcher would
    work, but passing it keeps the lifetime obviously the caller's.

    `flyc.compile` caches internally (`JitFunction._mem_cache`, keyed on the
    full argument signature), so this is not about avoiding recompilation --
    it is about dispatch overhead, measured at **157 us per call** at
    (head_dim 64, N 512), where the whole call is 63 us with the cache and
    220 us without. The gap closes as the kernel grows.

    A `WeakKeyDictionary` rather than an `exe._cf` attribute: that attribute is
    not one FlyDSL defines, so writing it mutates a FlyDSL-owned object. Weak
    keys so an entry dies with its launcher instead of pinning one compiled
    artifact -- and the GPU code object it owns -- per configuration for the
    life of the process.
    """
    cf = cache.get(exe)
    if cf is None:
        cache[exe] = flyc.compile(exe, *args)
    else:
        cf(*args)


def new_compiled_cache():
    """A cache for `run_compiled`. One per kernel module."""
    return weakref.WeakKeyDictionary()


def dtype_to_elem_type(dtype_str):
    """`"f16"` / `"bf16"` -> the FlyDSL Numeric class the kernel builds with."""
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f16' or 'bf16')")
