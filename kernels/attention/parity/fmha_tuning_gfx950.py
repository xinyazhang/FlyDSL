# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tuning policy for the gfx950 parity attention kernel: which knobs, for which shape.

The gfx950 twin of `fmha_tuning_gfx1201.py`, split from the kernel for the same
reason: the kernel file is about correctness and this one is about speed. A
number here moves when a sweep says so, and nothing here can make a build
*wrong*, only slow.

Two kinds of thing live here, and the distinction decides whether a change
needs a benchmark or a correctness argument:

- **Policy** -- `LADDER`, `_TILE_FOR_HEAD_DIM`, the knob fallbacks. Measured
  choices; any of them could change without making a build incorrect.
- **Geometry** -- `tile_width_for`, and the divisibility rules it enforces.
  These compute what is *legal*, and changing one can make a build invalid.

**The ladder is the whole design.** gfx950's dualwave LDS staging is built on a
128-byte (64-element bf16) row -- `SMEM_D_RPT = head_dim // 64` in
`_make_dualwave_swp_traits` -- so head_dim is not a free parameter, it is a
multiple of 64. Everything between the rungs is served by compiling the next
rung up and passing the real extent as a runtime argument, which is what
`padded_head` records. See `hdim_mode` for what the kernel then does with it.
"""

from dataclasses import dataclass, fields, replace

# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

# Compiled tile widths. Only the first two exist today: 192 and above need the
# 4-wave / 512-VGPR / 128x128 geometry that `fwd_hd192_hd128_bf16.co` uses, and
# that is P2. Listed here so `tile_width_for` reports "not built yet" rather
# than "unsupported", which are different problems for the caller.
LADDER = (64, 128)
LADDER_PLANNED = (192, 256, 384, 512)

# gfx950 dualwave geometry constant. Not a knob -- it is the 128-byte LDS row
# the staging is built on, and `_make_dualwave_swp_traits` derives SMEM_D_RPT
# from it.
HEAD_DIM_GRANULE = 64


def tile_width_for(head_dim):
    """The compiled tile width serving `head_dim`, or raise saying why not.

    Rounds *up* to the next rung. head_dim 40 compiles as a 64-wide build with
    `hdim_qk=40`; head_dim 129 would need the 192 rung, which is P2.
    """
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    for rung in LADDER:
        if head_dim <= rung:
            return rung
    for rung in LADDER_PLANNED:
        if head_dim <= rung:
            raise NotImplementedError(
                f"head_dim {head_dim} needs the {rung}-wide tile, which is the 4-wave "
                f"geometry (P2 in sdpa-close-gap-gfx950.md). Built rungs: {LADDER}."
            )
    raise ValueError(f"head_dim {head_dim} exceeds the largest planned tile ({LADDER_PLANNED[-1]})")


# ---------------------------------------------------------------------------
# What to compute, and how
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FmhaInputMetadata:
    """What to compute. Set by the caller; never by policy."""

    num_heads: int
    head_dim: int
    causal: bool = True
    dtype_str: str = "bf16"
    num_kv_heads: int | None = None
    head_dim_v: int | None = None
    sm_scale: float | None = None


@dataclass(frozen=True)
class FmhaKnobs:
    """How to compute it. Every field `None` means "policy decides".

    `None` rather than a literal default on purpose: it is the only way
    `resolve_knobs` can tell "the caller wants 1" from "the caller did not
    say", and that difference is the whole point of the overrides argument.
    """

    # Compile-time widths. `block_dmodel` is the tile the hsaco serves; the
    # *real* extent travels as a runtime argument, which is what `padded_head`
    # records.
    block_dmodel: int | None = None
    block_dmodel_v: int | None = None
    padded_head: bool | None = None

    # How a head_dim narrower than the tile is handled. See the kernel's
    # `hdim_mode` note; the two GEMMs are not symmetric about this.
    #
    #   "zero_fill"        loads skip whole chunks past hdim and zero those
    #                      registers; the MFMA count stays tile-shaped.
    #   "runtime_qk_steps" additionally shortens the QK reduction to
    #                      ceil(hdim_qk/16) MFMA K-steps at runtime.
    #
    # `None` resolves to "zero_fill" unless the head is not padded at all, in
    # which case the two are identical and the cheaper one is picked.
    hdim_mode: str | None = None

    # Whether the strides fold to literals. False is the parity ABI; True
    # exists so the fast path can be shown to emit unchanged code.
    strides_constexpr: bool | None = None

    # Dualwave schedule knobs, passed through to `_make_dualwave_swp_traits`.
    waves_per_eu: int | None = None
    daz: bool | None = None
    lazy_rescale: bool | None = None
    setprio: bool | None = None
    stagger: bool | None = None

    # Modes carried from the production kernel. Not part of the parity
    # surface; see the plan's split-K note.
    num_kv_splits: int | None = None
    paged: bool | None = None
    kv_cache_layout: str | None = None
    return_lse: bool | None = None

    def merge(self, other: "FmhaKnobs | None") -> "FmhaKnobs":
        """`other`'s set fields win; its `None`s leave this one's alone."""
        if other is None:
            return self
        set_fields = {f.name: getattr(other, f.name) for f in fields(other) if getattr(other, f.name) is not None}
        return replace(self, **set_fields)


# Defaults the policy has no shape-dependent opinion about. These match the
# production `build_flash_attn_dualwave_swp_module` signature exactly, so a
# default-knob parity build is the same schedule the baseline was measured on.
_KNOBS_FALLBACK = FmhaKnobs(
    waves_per_eu=2,
    daz=True,
    lazy_rescale=True,
    setprio=True,
    stagger=True,
    num_kv_splits=1,
    paged=False,
    kv_cache_layout="linear",
    return_lse=False,
    strides_constexpr=False,
)


def resolve_knobs(meta: FmhaInputMetadata, overrides: "FmhaKnobs | None" = None) -> FmhaKnobs:
    """The complete configuration for `meta`.

    `overrides` is applied *first*, so a pinned knob participates in deriving
    the ones downstream of it rather than being stamped on afterwards --
    pinning `block_dmodel` therefore also decides `padded_head`.
    """
    knobs = _KNOBS_FALLBACK.merge(overrides)

    block_dmodel = knobs.block_dmodel
    if block_dmodel is None:
        block_dmodel = tile_width_for(meta.head_dim)
    elif block_dmodel not in LADDER:
        raise ValueError(f"block_dmodel must be one of the built rungs {LADDER}, got {block_dmodel}")
    if meta.head_dim > block_dmodel:
        raise ValueError(f"head_dim {meta.head_dim} does not fit the pinned block_dmodel {block_dmodel}")

    head_dim_v = meta.head_dim if meta.head_dim_v is None else meta.head_dim_v
    if head_dim_v > block_dmodel:
        raise ValueError(f"head_dim_v {head_dim_v} does not fit block_dmodel {block_dmodel}")
    block_dmodel_v = knobs.block_dmodel_v
    if block_dmodel_v is None:
        block_dmodel_v = block_dmodel

    padded_head = knobs.padded_head
    if padded_head is None:
        padded_head = (meta.head_dim != block_dmodel) or (head_dim_v != block_dmodel_v)

    hdim_mode = knobs.hdim_mode
    if hdim_mode is None:
        # With nothing to pad, the modes are identical; pick the one that
        # emits no runtime trip count.
        hdim_mode = "zero_fill"
    if hdim_mode not in ("zero_fill", "runtime_qk_steps"):
        raise ValueError(f"hdim_mode must be 'zero_fill' or 'runtime_qk_steps', got {hdim_mode!r}")

    return replace(
        knobs,
        block_dmodel=block_dmodel,
        block_dmodel_v=block_dmodel_v,
        padded_head=bool(padded_head),
        hdim_mode=hdim_mode,
    )


@dataclass(frozen=True)
class FmhaPlan:
    meta: FmhaInputMetadata
    knobs: FmhaKnobs


def plan(meta: FmhaInputMetadata, overrides: "FmhaKnobs | None" = None) -> FmhaPlan:
    """`meta` plus its fully-resolved schedule, as one hashable build key."""
    return FmhaPlan(meta=meta, knobs=resolve_knobs(meta, overrides))
