# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tuning policy for the gfx950 parity attention kernel: which knobs, for which shape.

Split from the kernel for the usual reason: the kernel file is about
correctness and this one is about speed. A number here moves when a sweep says
so, and nothing here can make a build *wrong*, only slow.

--- One object, one call (R1) ---------------------------------------------

    knobs = fmha_knobs(arch, **overrides)   # factory -> arch-specific subclass
    cfg   = knobs.resolve(meta)             # fully resolved, ready to build

**Knobs and traits were always the same thing at two points on one axis.**
Knobs are the *partially resolved* form -- `None` means "policy decides" --
and the dualwave traits are the *fully resolved* one. Keeping them as separate
types with a converter between them meant every feature knob was declared
twice and threaded through a function call, which is why `varlen` and
`cross_seqlen` used to travel as keyword arguments *beside* the knob object
instead of inside it.

So `resolve` is a method, and it owns the whole derivation: the ladder, the
padded-head decision, the wave geometry, and the traits the kernel is built
from. `FmhaInputMetadata` stays arch-neutral -- it is *what to compute*, and
no arch has an opinion about it. `FmhaKnobs` is *how*, and is subclassed per
arch.

Two kinds of thing still live here, and the distinction decides whether a
change needs a benchmark or a correctness argument:

- **Policy** -- `LADDER`, the fallbacks, `_wave_geometry`. Measured choices;
  any of them could change without making a build incorrect.
- **Geometry** -- `tile_width_for` and the divisibility rules `resolve`
  enforces. These compute what is *legal*, and changing one can make a build
  invalid.

**The ladder is the whole design.** gfx950's dualwave LDS staging is built on a
128-byte (64-element bf16) row -- `SMEM_D_RPT = head_dim // 64` -- so head_dim
is not a free parameter, it is a multiple of 64. Everything between the rungs
is served by compiling the next rung up and passing the real extent as a
runtime argument, which is what `padded_head` records.
"""

from dataclasses import dataclass, fields, replace

from gfx950_standalone import dualwave

__all__ = [
    "LADDER",
    "LADDER_PLANNED",
    "FmhaInputMetadata",
    "FmhaKnobs",
    "Gfx950Knobs",
    "fmha_knobs",
    "tile_width_for",
]

# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------

# Compiled tile widths. Only the first two exist today, and the reason is
# measured rather than assumed -- see the P2 section of
# `sdpa-close-gap-gfx950.md`. On the 8-wave geometry:
#
#   head_dim   VGPR   spills   LDS
#     64        164      0     34 KB
#    128        248      0     68 KB     <- saturated, zero spills
#    192        256    506    102 KB
#    256        256    963    136 KB
#
# LDS is not the binding constraint at 192/256; the register file is. That is
# why `fwd_hd192_hd128_bf16.co` runs 4 waves at 512 VGPRs/lane -- halving the
# wave count doubles the per-lane register file. Listed here so
# `tile_width_for` reports "not built yet" rather than "unsupported", which are
# different problems for the caller.
LADDER = (64, 128)
LADDER_PLANNED = (192, 256, 384, 512)

# gfx950 dualwave geometry constant. Not a knob -- it is the 128-byte LDS row
# the staging is built on.
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
# What to compute
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FmhaInputMetadata:
    """What to compute. Set by the caller; never by policy, and never by arch."""

    num_heads: int
    head_dim: int
    causal: bool = True
    dtype_str: str = "bf16"
    num_kv_heads: int | None = None
    head_dim_v: int | None = None
    sm_scale: float | None = None


# ---------------------------------------------------------------------------
# How to compute it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FmhaKnobs:
    """How to compute it, arch-neutral part. `None` means "policy decides".

    `None` rather than a literal default on purpose: it is the only way
    `resolve` can tell "the caller wants 1" from "the caller did not say", and
    that difference is the whole point of the overrides.

    Subclass per arch and implement `resolve`. Nothing may be added here that
    only one arch understands -- that is what the subclass is for.
    """

    # Compile-time widths. `block_dmodel` is the tile the hsaco serves; the
    # *real* extent travels as a runtime argument, which `padded_head` records.
    block_dmodel: int | None = None
    block_dmodel_v: int | None = None
    padded_head: bool | None = None

    # How a head_dim narrower than the tile is handled. The two GEMMs are not
    # symmetric about this:
    #
    #   "zero_fill"        loads skip whole chunks past hdim and zero those
    #                      registers; the MFMA count stays tile-shaped.
    #   "runtime_qk_steps" additionally shortens the QK reduction to
    #                      ceil(hdim_qk/16) MFMA K-steps at runtime. Legal
    #                      because the S accumulator shape depends on
    #                      BLOCK_M/BLOCK_N, not on D.
    hdim_mode: str | None = None

    # Whether the strides fold to literals. False is the parity ABI; True
    # exists so the fast path can be shown to emit unchanged code.
    strides_constexpr: bool | None = None

    return_lse: bool | None = None

    def merge(self, other):
        """`other`'s set fields win; its `None`s leave this one's alone."""
        if other is None:
            return self
        set_fields = {f.name: getattr(other, f.name) for f in fields(other) if getattr(other, f.name) is not None}
        return replace(self, **set_fields)

    def resolve(self, meta: FmhaInputMetadata):
        raise NotImplementedError(f"{type(self).__name__} does not implement resolve()")

    # -- resolution steps ------------------------------------------------
    #
    # Each `_with_*` takes knobs and returns knobs with some fields decided,
    # so `resolve` is a pipeline rather than a place where scattered tuples
    # get reassembled. Three things follow, and each removes a class of bug:
    #
    # - **The field names appear once.** Returning `(block_dmodel,
    #   block_dmodel_v, padded_head, hdim_mode)` and `replace`-ing them at the
    #   call site spelled all four twice, in an order nothing checked.
    # - **The step order is the data dependency.** `_with_wave_geometry` reads
    #   `self.block_dmodel`, so it *must* run after `_with_widths`; it no
    #   longer takes it as an argument and cannot be handed a stale one.
    # - **`dataclasses.replace` preserves the subclass**, so a base-class step
    #   returns `Gfx950Knobs` and the pipeline can mix inherited and
    #   arch-specific steps freely.

    def _with_widths(self, meta):
        """Decide `block_dmodel`, `block_dmodel_v`, `padded_head`, `hdim_mode`.

        Every arch has a ladder and a padded-head rule; only the ladder's
        contents differ, and those are `tile_width_for`'s. Kept on the base so
        a second arch cannot accidentally decide `padded_head` by another rule.
        """
        block_dmodel = self.block_dmodel
        if block_dmodel is None:
            block_dmodel = tile_width_for(meta.head_dim)
        elif block_dmodel not in LADDER:
            raise ValueError(f"block_dmodel must be one of the built rungs {LADDER}, got {block_dmodel}")
        if meta.head_dim > block_dmodel:
            raise ValueError(f"head_dim {meta.head_dim} does not fit the pinned block_dmodel {block_dmodel}")

        head_dim_v = meta.head_dim if meta.head_dim_v is None else meta.head_dim_v
        if head_dim_v > block_dmodel:
            raise ValueError(f"head_dim_v {head_dim_v} does not fit block_dmodel {block_dmodel}")
        block_dmodel_v = block_dmodel if self.block_dmodel_v is None else self.block_dmodel_v

        padded_head = self.padded_head
        if padded_head is None:
            padded_head = (meta.head_dim != block_dmodel) or (head_dim_v != block_dmodel_v)

        hdim_mode = "zero_fill" if self.hdim_mode is None else self.hdim_mode
        if hdim_mode not in ("zero_fill", "runtime_qk_steps"):
            raise ValueError(f"hdim_mode must be 'zero_fill' or 'runtime_qk_steps', got {hdim_mode!r}")

        return replace(
            self,
            block_dmodel=block_dmodel,
            block_dmodel_v=block_dmodel_v,
            padded_head=bool(padded_head),
            hdim_mode=hdim_mode,
        )


@dataclass(frozen=True)
class Gfx950Knobs(FmhaKnobs):
    """The gfx950 dual-wave schedule.

    `varlen`, `cross_seqlen`, `paged` and `num_kv_splits` are ordinary fields
    here, not arguments threaded past the object. That is the point of R1:
    before it, `cross_seqlen` was a keyword-only parameter on the builder, a
    `kwargs.pop` in the front end, and an argument to a converter -- three
    places to keep in step for one boolean, and P4 would have added a fourth
    passenger in `varlen`.
    """

    # Dual-wave schedule.
    waves_per_eu: int | None = None
    daz: bool | None = None
    lazy_rescale: bool | None = None
    setprio: bool | None = None
    stagger: bool | None = None

    # Problem modes. Ordinary fields; see the class docstring.
    varlen: bool | None = None
    cross_seqlen: bool | None = None
    paged: bool | None = None
    kv_cache_layout: str | None = None
    num_kv_splits: int | None = None

    # Wave geometry. `None` means "the family for this tile width", which is
    # what `_wave_geometry` decides. Pinnable so a sweep can cross the family
    # boundary without editing the table.
    num_waves: int | None = None
    block_m: int | None = None
    block_n: int | None = None

    # Set by `resolve`; never by a caller. Holding the traits *on* the resolved
    # knobs is what makes "knobs and traits are one thing" true at the use
    # site: the builder takes one object and reads both from it.
    traits: object | None = None

    def resolve(self, meta: FmhaInputMetadata) -> "Gfx950Knobs":
        """The complete build configuration for `meta`, traits included.

        Subsumes what used to be `resolve_knobs` *and* `make_parity_traits`.
        Idempotent: resolving an already-resolved object re-derives the same
        answer, since every derived field is recomputed from `meta` and the
        pinned fields rather than read back.
        """
        return (
            _GFX950_FALLBACK.merge(self)
            ._checked_modes()
            ._with_widths(meta)
            ._with_wave_geometry()
            ._with_traits(meta)
        )

    def _checked_modes(self):
        """Reject mode combinations the kernel does not implement.

        First in the pipeline because none of it depends on a derived field --
        anything that fails here would fail whatever the ladder decided, so
        failing before the derivation keeps the message about the caller's
        input rather than about something computed from it.
        """
        if self.varlen and self.num_kv_splits > 1:
            raise ValueError("varlen is not supported together with num_kv_splits > 1")
        if self.kv_cache_layout not in ("linear", "vectorized"):
            raise ValueError(f"kv_cache_layout must be 'linear' or 'vectorized', got {self.kv_cache_layout!r}")
        return self

    def _with_wave_geometry(self):
        """Decide `num_waves`, `block_m`, `block_n` from the resolved tile width.

        **The selection lives here, not in the traits constructor**, which is
        the concrete reason R1 came before P2: family B (4 waves, 128x128, 512
        VGPRs/lane) differs from family A only in these three numbers plus what
        they imply, and a constructor that hardcodes them cannot host both.

        Family A is measured saturated at head_dim 128 -- 248 of 256 VGPRs,
        zero spills -- so the next rung has to halve the wave count to double
        the per-lane register file. See the table above `LADDER`.

        Reads `self.block_dmodel`, so it runs after `_with_widths`. That
        ordering is the whole reason the width is no longer an argument: an
        argument can be stale, a field the previous step wrote cannot.
        """
        pinned = (self.num_waves, self.block_m, self.block_n)
        if all(x is not None for x in pinned):
            return self
        if any(x is not None for x in pinned):
            raise ValueError(f"pin num_waves, block_m and block_n together or not at all, got {pinned}")
        if self.block_dmodel is None:
            raise ValueError("_with_wave_geometry runs after _with_widths; block_dmodel is not resolved")
        if self.block_dmodel <= 128:
            return replace(self, num_waves=8, block_m=256, block_n=64)  # family A
        return replace(self, num_waves=4, block_m=128, block_n=128)  # family B

    def _with_traits(self, meta):
        """Build the dualwave traits this configuration implies.

        The last step, and the one family B replaces. Family A delegates to the
        production constructor, which hardcodes 8/256/64 -- usable *because*
        family A's geometry is exactly those numbers, not by coincidence.
        Family B needs its own, which is work P2 still owes; failing here names
        that, rather than silently building family A's geometry under family
        B's name and then benchmarking it as the new one.
        """
        if (self.num_waves, self.block_m, self.block_n) != (8, 256, 64):
            raise NotImplementedError(
                f"wave geometry {(self.num_waves, self.block_m, self.block_n)} (family B) needs a "
                "parity-side traits constructor; `_make_dualwave_swp_traits` hardcodes 8/256/64. "
                "See P2 in sdpa-close-gap-gfx950.md."
            )
        num_kv_heads = meta.num_heads if meta.num_kv_heads is None else meta.num_kv_heads
        traits = dualwave._make_dualwave_swp_traits(
            meta.num_heads,
            num_kv_heads,
            self.block_dmodel,
            causal=meta.causal,
            dtype_str=meta.dtype_str,
            waves_per_eu=self.waves_per_eu,
            daz=self.daz,
            dualwave_swp_lazy_rescale=self.lazy_rescale,
            dualwave_swp_setprio=self.setprio,
            dualwave_swp_debug_lazy_counts=False,
            dualwave_swp_enable_stagger=self.stagger,
            num_kv_splits=self.num_kv_splits,
            varlen=self.varlen,
            cross_seqlen=self.cross_seqlen,
            paged=self.paged,
            kv_cache_layout=self.kv_cache_layout,
            kv_vectorized=self.paged and self.kv_cache_layout == "vectorized",
            return_lse=self.return_lse,
        )
        return replace(self, traits=traits)


# Defaults the policy has no shape-dependent opinion about. These match the
# production `build_flash_attn_dualwave_swp_module` signature exactly, so a
# default-knob parity build is the schedule the baseline was measured on.
_GFX950_FALLBACK = Gfx950Knobs(
    waves_per_eu=2,
    daz=True,
    lazy_rescale=True,
    setprio=True,
    stagger=True,
    varlen=False,
    cross_seqlen=False,
    paged=False,
    kv_cache_layout="linear",
    num_kv_splits=1,
    return_lse=False,
    strides_constexpr=False,
)

_BY_ARCH = {"gfx950": Gfx950Knobs}


def fmha_knobs(arch: str, **overrides) -> FmhaKnobs:
    """The knob object for `arch`, with `overrides` pinned.

    The one entry point. A caller names an architecture and the fields it
    cares about; everything else stays `None` until `resolve` decides it.
    Keyed on the arch *prefix* so a full `gcnArchName` --
    "gfx950:sramecc+:xnack-" -- works without the caller stripping it.
    """
    base = arch.split(":")[0].lower() if arch else ""
    for prefix, cls in _BY_ARCH.items():
        if base.startswith(prefix):
            if "traits" in overrides:
                raise TypeError("`traits` is set by resolve(), not by the caller")
            known = {f.name for f in fields(cls)}
            unknown = set(overrides) - known
            if unknown:
                raise TypeError(f"unknown {cls.__name__} field(s): {sorted(unknown)}")
            return cls(**overrides)
    raise ValueError(f"no FMHA knobs for arch {arch!r}; known: {sorted(_BY_ARCH)}")
