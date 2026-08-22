# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tuning policy for the gfx1201 *fused* backward kernel.

Same split as `fmha_tuning_gfx1201.py` does for the forward pass, and for the
same reason: the kernel file is about correctness, this one is about speed.
Nothing here can make a build wrong, only slow -- with one exception noted
below, and it is spelled as a hard error rather than a knob.

**No flydsl import.** Everything here is plain Python arithmetic over ints,
which is what makes this module trivially stable-only: it cannot use an
unstable API because it uses no API at all.

--- What the knobs mean ----------------------------------------------------

The fused kernel is one launch with two program roles, selected by
``block_idx.x`` (AOTriton's ``bwd_kernel_fuse`` does the same thing):

  role      owns                       iterates over    accumulates
  ---------------------------------------------------------------------
  dK/dV     ``KV_TILE`` key rows       Q blocks         dK and dV
  dQ        ``Q_TILE``  query rows     KV blocks        dQ

Both roles give **one 16-row WMMA tile to each wave**, so ``KV_TILE`` and
``Q_TILE`` are both ``16 * NUM_WAVES`` and are not free parameters -- only
``num_waves`` is. The *other* axis is the inner step: ``q_step`` is how many
query rows the dK/dV role stages per iteration, ``kv_step`` how many key rows
the dQ role stages. Both are 16 by default, i.e. exactly one WMMA tile, which
is what keeps the LDS footprint down (see `lds_elems_dkdv`).

--- The register wall ------------------------------------------------------

This is the constraint that shapes the whole design, so it is written down
here rather than discovered per build.

A wave that owns 16 rows x ``head_dim`` of f32 accumulator pays
``head_dim / 2`` VGPRs for it (8 lanes' worth of v8f32 per 16 columns). The
dK/dV role holds **two** such accumulators, dK and dV, so it pays

    2 * head_dim / 2 = head_dim VGPRs

before a single operand is live. At head_dim 128 that is 128 of the 256
VGPRs a gfx1201 lane has; at 192 it is 192 and the operands no longer fit; at
256 the accumulators alone would fill the register file. `MAX_HEAD_DIM` is 128
for exactly this reason, and it is a `ValueError` rather than a slow path
because the alternative -- spilling a loop-carried accumulator to scratch every
iteration -- is not a degradation, it is a different kernel.

The dQ role holds one accumulator (``head_dim / 2``) plus the Q and dO operand
packs it keeps resident across the KV loop (``head_dim / 4`` each), so it is
``head_dim`` as well and is not the binding constraint.
"""

from dataclasses import dataclass, fields, replace

__all__ = [
    "BwdInputMetadata",
    "BwdKnobs",
    "BwdPlan",
    "MAX_HEAD_DIM",
    "HEAD_DIM_LADDER",
    "acc_vgprs_dkdv",
    "lds_elems_dkdv",
    "lds_elems_dq",
    "plan",
    "resolve_knobs",
]

# See "The register wall" above. Not a policy number: raising it without
# changing the accumulator layout produces a kernel that spills every
# iteration.
MAX_HEAD_DIM = 128

# Compiled tile widths. A head_dim between two rungs is served by the next one
# up with `padded_head` set, exactly as the forward kernel's ladder works.
HEAD_DIM_LADDER = (16, 32, 48, 64, 80, 96, 112, 128)

# LDS padding, in elements, added to every tile's row pitch. 4 elements = 8
# bytes, which is what the forward kernel uses and for the same reason: it
# breaks the power-of-two row stride that would otherwise put every row on the
# same LDS banks. Not swizzling, which was measured a net loss on this part
# (`sdpa_lore_gfx1201.md`).
LDS_PAD = 4

# gfx1201 workgroup LDS limit. A build over this is rejected at plan time
# rather than at kernel launch, where the failure is a HIP error with no
# indication of which knob caused it.
LDS_BYTES_LIMIT = 65536

# Waves per workgroup. 4 gives a 128-thread workgroup, which is what makes the
# cooperative loads land one 8-element vector per lane per row-batch at
# head_dim 128 (128 threads / (128/8) threads-per-row = 8 rows per batch).
# Not swept -- the fused kernel has not been benchmarked, only verified.
_DEFAULT_NUM_WAVES = 4


@dataclass(frozen=True)
class BwdInputMetadata:
    """What to compute. Set by the caller; never by policy."""

    num_head_q: int
    num_head_k: int
    head_dim: int
    causal: bool = False
    dtype_str: str = "f16"
    sm_scale: float | None = None
    # AOTriton's CAUSAL_TYPE: 0 none, 1 top-left, 2 bottom-right, 3 explicit
    # window. `None` derives 1/0 from `causal`, matching the forward kernel.
    causal_type: int | None = None
    dropout: bool = False
    philox_width: int | None = None

    # Attention bias, matching the forward's `BIAS_TYPE`: a (B, H, Sq, Sk)
    # matrix added to the scores after the scale and before the mask.
    #
    # An input only. The forward folds the bias into the score *and* into the
    # logsumexp it stores, and this kernel recomputes P from that logsumexp, so
    # without the bias term every gradient is wrong by `exp(-bias)`.
    #
    # **dB is not emitted here**, as it is not by the standalone dK/dV kernel;
    # `fmha_bwd_dq_gfx1201_kernel`'s `return_dbias` is where it comes from.
    bias: bool = False


@dataclass(frozen=True)
class BwdKnobs:
    """How to compute it. Every field `None` means "policy decides"."""

    # Compile-time head width. The *real* extent rides along as a runtime
    # argument and may be smaller, which is what `padded_head` records.
    block_dmodel: int | None = None
    padded_head: bool | None = None

    num_waves: int | None = None
    q_step: int | None = None  # query rows staged per dK/dV iteration
    kv_step: int | None = None  # key rows staged per dQ iteration

    waves_per_eu: int | None = None
    flat_work_group_size: int | None = None
    sched_strategy: str | None = None
    lpt_tile_order: bool | None = None

    fp_mode: str | None = None
    denormals_are_zero: bool | None = None
    unsafe_fp_math: bool | None = None
    fast_fp_math: bool | None = None

    def merge(self, other: "BwdKnobs") -> "BwdKnobs":
        """`other`'s set fields win; `None` leaves this one's value alone."""
        upd = {f.name: getattr(other, f.name) for f in fields(other) if getattr(other, f.name) is not None}
        return replace(self, **upd)


@dataclass(frozen=True)
class BwdPlan:
    meta: BwdInputMetadata
    knobs: BwdKnobs


def _round_to_ladder(head_dim: int) -> int:
    for rung in HEAD_DIM_LADDER:
        if head_dim <= rung:
            return rung
    raise ValueError(
        f"the fused backward kernel supports head_dim <= {MAX_HEAD_DIM}, got {head_dim}. "
        "See 'The register wall' in this module: dK and dV are both live across "
        "the Q loop, which costs head_dim VGPRs before any operand"
    )


def acc_vgprs_dkdv(head_dim: int) -> int:
    """VGPRs the dK/dV role's two loop-carried accumulators occupy, per lane.

    `head_dim`, not `head_dim / 2`: there are two of them. Exposed so a caller
    -- or a future sweep -- can state the budget without re-deriving it.
    """
    return head_dim


def lds_elems_dkdv(block_dmodel: int, kv_tile: int, q_step: int) -> int:
    """LDS elements the dK/dV role needs, in `dtype_str` units.

    Six tiles, and the two that look redundant are not. The WMMA operand
    layout puts the *contracted* index on `klane*8 + e` and the free index on
    `lane16`, so a matrix can only be an operand of a product that contracts
    over the axis it was loaded along. dK and dV contract over the query axis
    while S and dP contract over the head axis, which means Q and dO are each
    needed in both orientations. There is no rearrangement that avoids it --
    see the operand-layout note in the kernel module.
    """
    row = block_dmodel + LDS_PAD  # K, V, Q, dO: row = token, col = d
    col = q_step + LDS_PAD  # Q^T, dO^T:      row = d,     col = token
    return 2 * kv_tile * row + 2 * q_step * row + 2 * block_dmodel * col


def lds_elems_dq(block_dmodel: int, kv_step: int) -> int:
    """LDS elements the dQ role needs. Three tiles: K, V and K^T."""
    row = block_dmodel + LDS_PAD
    col = kv_step + LDS_PAD
    return 2 * kv_step * row + block_dmodel * col


def resolve_knobs(meta: BwdInputMetadata, overrides: BwdKnobs | None = None) -> BwdKnobs:
    """Fill in every `None` with the policy's answer.

    Every field of the result is set: the kernel builder falls back to nothing
    and treats an unresolved knob as a caller error.
    """
    k = BwdKnobs().merge(overrides or BwdKnobs())

    block_dmodel = k.block_dmodel if k.block_dmodel is not None else _round_to_ladder(meta.head_dim)
    padded_head = block_dmodel != meta.head_dim

    num_waves = k.num_waves if k.num_waves is not None else _DEFAULT_NUM_WAVES
    q_step = k.q_step if k.q_step is not None else 16
    kv_step = k.kv_step if k.kv_step is not None else 16

    # Longest-processing-time-first, the forward kernel's `lpt_tile_order`
    # rather than AOTriton's `remap_xcd`. Under causal masking a dQ program's
    # cost grows with its query block (block 0 walks one KV tile, block N-1
    # walks N) and grid.x issues in increasing order, so the expensive work
    # lands in the tail. Reversing puts it first.
    #
    # **Only the dQ half is reversed.** A dK/dV program's cost grows the other
    # way -- key block 0 is visible to every query, key block N-1 to almost
    # none -- so increasing order is *already* longest-first for it. Reversing
    # both would move the imbalance rather than remove it. This asymmetry is
    # why the knob is a single bool applied at one site in the kernel and not a
    # grid permutation.
    #
    # With uniform cost -- every non-causal program -- the reversal is a
    # permutation with no load-balancing content, so `and causal` is part of
    # the knob's definition and resolved here.
    lpt = k.lpt_tile_order if k.lpt_tile_order is not None else True
    lpt = bool(lpt) and bool(meta.causal)

    flat = k.flat_work_group_size if k.flat_work_group_size is not None else num_waves * 32

    # Copied from the forward kernel's policy, not measured here. The backward
    # loop has the same shape -- LDS-fed WMMA with the loads hoisted above the
    # matrix ops -- so max-memory-clause should behave the same way, but that
    # is an expectation and not a measurement. Left on for causal only, which
    # is where the forward's gain was largest.
    sched = k.sched_strategy if k.sched_strategy is not None else ("max-memory-clause" if meta.causal else "")

    return BwdKnobs(
        block_dmodel=block_dmodel,
        padded_head=padded_head,
        num_waves=num_waves,
        q_step=q_step,
        kv_step=kv_step,
        waves_per_eu=k.waves_per_eu,
        flat_work_group_size=flat,
        sched_strategy=sched,
        lpt_tile_order=lpt,
        fp_mode=k.fp_mode if k.fp_mode is not None else "noninf",
        denormals_are_zero=k.denormals_are_zero if k.denormals_are_zero is not None else True,
        unsafe_fp_math=k.unsafe_fp_math if k.unsafe_fp_math is not None else False,
        fast_fp_math=k.fast_fp_math if k.fast_fp_math is not None else False,
    )


def plan(request: BwdInputMetadata, overrides: BwdKnobs | None = None) -> BwdPlan:
    """The one entry point into tuning: caller's inputs in, a full plan out.

    `request` comes back as `BwdPlan.meta` unchanged. Everything policy decided
    lands in `BwdPlan.knobs`, including the rounding up to a compiled tile.

    Raises for a head_dim past the register wall, for a knob combination whose
    LDS footprint does not fit, and for the two structural divisibility
    requirements. All three are build-time facts, so they are errors here
    rather than assertions inside the traced kernel where the message would
    arrive without the knobs that caused it.
    """
    if request.bias and (request.causal or request.causal_type):
        raise ValueError(
            "bias and causal masking are mutually exclusive, as in the forward: a bias "
            "already is an additive mask, so the pair has no defined meaning. Fold the "
            "causal pattern into the bias tensor, or drop the bias"
        )
    if request.head_dim < 1:
        raise ValueError(f"head_dim must be positive, got {request.head_dim}")
    if request.num_head_k < 1 or request.num_head_q % request.num_head_k:
        raise ValueError(
            f"num_head_q ({request.num_head_q}) must be a positive multiple of " f"num_head_k ({request.num_head_k})"
        )
    knobs = resolve_knobs(request, overrides)

    if knobs.block_dmodel % 16:
        raise ValueError(f"block_dmodel must be a multiple of 16, got {knobs.block_dmodel}")
    if knobs.block_dmodel > MAX_HEAD_DIM:
        raise ValueError(f"block_dmodel {knobs.block_dmodel} exceeds MAX_HEAD_DIM {MAX_HEAD_DIM}")
    for name, step in (("q_step", knobs.q_step), ("kv_step", knobs.kv_step)):
        if step % 16 or step & (step - 1):
            # A power of two because the region decomposition divides by it
            # with an arithmetic shift (`sdiv_rd_pow2`), which is only a
            # floor-division for a power of two.
            raise ValueError(f"{name} must be a power-of-two multiple of 16, got {step}")

    elem_bytes = 2  # f16 / bf16; f32 inputs are not supported
    lds = max(
        lds_elems_dkdv(knobs.block_dmodel, 16 * knobs.num_waves, knobs.q_step),
        lds_elems_dq(knobs.block_dmodel, knobs.kv_step),
    )
    if lds * elem_bytes > LDS_BYTES_LIMIT:
        raise ValueError(
            f"this configuration needs {lds * elem_bytes} B of LDS, over the "
            f"{LDS_BYTES_LIMIT} B workgroup limit. Lower num_waves "
            f"({knobs.num_waves}) or q_step ({knobs.q_step})"
        )
    return BwdPlan(request, replace(knobs, padded_head=knobs.block_dmodel != request.head_dim))
