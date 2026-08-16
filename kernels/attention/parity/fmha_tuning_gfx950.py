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

- **Policy** -- `LADDER`, the fallbacks, `_with_wave_geometry`. Measured
  choices; any of them could change without making a build incorrect.
- **Geometry** -- `tile_width_for`, `staging_shape`, and the divisibility rules
  `resolve` enforces. These compute what is *legal*, and changing one can make
  a build invalid.

**The ladder is the design, and the granule is a knob within it.** A head_dim
between two rungs is served by compiling the next rung up and passing the real
extent as a runtime argument, which is what `padded_head` records. What decides
the rungs is the *staging granule* -- how many D elements one DMA issue covers
-- and that was long assumed to be 64 because the production kernel is built
that way. It is not a constant: `_with_wave_geometry` picks it per family, and
`staging_shape` states the rule it has to satisfy. Only `PV_MFMA_N` is a real
floor, and it is an instruction limit rather than a staging one.
"""

from dataclasses import dataclass, fields, replace

from fmha_traits_gfx950 import make_traits

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
#
# **32 is a planned rung too, below the built ones**, and it is deliberately
# not in `LADDER_PLANNED`: that list is consulted only after `LADDER` misses,
# so putting 32 there would never be reached -- while putting it in `LADDER`
# would route head_dim <= 32 to a tile that does not exist yet and break a path
# that works today, slowly. It arrives when family S is built; until then
# head_dim <= 32 rounds up to 64 and pays for it (see `_with_wave_geometry`).
LADDER = (64, 128)
LADDER_PLANNED = (192, 256, 384, 512)

# The D-axis staging granule: how many bf16 elements of one token a single DMA
# issue covers. **Not a constant, and not 64 by necessity** -- see
# `_with_wave_geometry`. A wave moves 512 elements per issue (64 lanes x 8), so
# the granule and BLOCK_N together decide how many lines a KV tile occupies and
# how many issues each wave makes:
#
#     tokens per issue = 512 / granule
#     lines            = BLOCK_N / tokens_per_issue
#     issues per wave  = lines / NUM_WAVES        (must be a positive integer)
#
# Granule 64 with BLOCK_N 64 is one point in that space, not the only one.
DEFAULT_HEAD_DIM_GRANULE = 64

# The PV MFMA is `v_mfma_f32_32x32x16`, whose output is 32 D columns wide, so
# `D_CHUNKS = head_dim / 32` cannot go below 1. **This is what makes a granule
# of 16 impossible**, and it is an instruction limit rather than a staging one:
# the staging is perfectly regular at granule 16 (BLOCK_N 256 over 8 waves is
# one issue per wave), but the output accumulator cannot be narrower than 32.
# Serving head_dim 16 natively would need `v_mfma_f32_16x16x16`, whose
# accumulator is v4f32 rather than v16f32 -- a different register layout
# through every helper, i.e. its own family.
PV_MFMA_N = 32

# Hardware shape of one DMA issue, used by `staging_shape`.
WARP_SIZE = 64
DMA_BYTES = 16
BF16_BYTES = 2


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
    # what `_with_wave_geometry` decides. Pinnable so a sweep can cross a
    # family boundary without editing the table.
    num_waves: int | None = None
    block_m: int | None = None
    block_n: int | None = None
    head_dim_granule: int | None = None

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
        """Decide the wave geometry and staging granule from the tile width.

        **The selection lives here, not in the traits constructor**, which is
        the concrete reason R1 came before P2: the families differ only in
        these four numbers plus what they imply, and a constructor that
        hardcodes them cannot host more than one.

        Three families, and the granule is a *choice* in each rather than the
        constant it used to be:

        | family | tile width | waves | BLOCK_M | BLOCK_N | granule |
        |---|---|---|---|---|---|
        | S | <= 32 | 8 | 256 | 128 | 32 |
        | A | <= 128 | 8 | 256 | 64 | 64 |
        | B | > 128 | 4 | 128 | 128 | 64 |

        **A** is measured saturated at head_dim 128 -- 248 of 256 VGPRs, zero
        spills -- so **B** halves the wave count to double the per-lane
        register file. **S** exists because padding a small head into A's
        64-wide tile is expensive, not cheap: at B=4 H=8 N=4096 non-causal,
        head_dim 16/32/48 each take *longer* than head_dim 64 (203/205/215 us
        against 172), doing the full 64-wide MFMA plus the padded-head masking
        on top -- 169 real TFLOPS at head_dim 16 against 801 at 64.

        S keeps A's wave count, BLOCK_M and one-issue-per-wave DMA structure;
        only the granule and BLOCK_N move. BLOCK_N must go to 128 to keep the
        DMA full, since halving the granule doubles the tokens one issue
        covers -- and independently, gfx1201's tuning table reached BLOCK_N 128
        for small head_dim by measurement, on the grounds that a wider KV tile
        amortises the per-tile softmax cost.

        Reads `self.block_dmodel`, so it runs after `_with_widths`. That
        ordering is the whole reason the width is no longer an argument: an
        argument can be stale, a field the previous step wrote cannot.
        """
        pinned = (self.num_waves, self.block_m, self.block_n, self.head_dim_granule)
        if all(x is not None for x in pinned):
            return self
        if any(x is not None for x in pinned):
            raise ValueError(
                f"pin num_waves, block_m, block_n and head_dim_granule together or not at all, got {pinned}"
            )
        if self.block_dmodel is None:
            raise ValueError("_with_wave_geometry runs after _with_widths; block_dmodel is not resolved")
        if self.block_dmodel % 64:
            return replace(self, num_waves=4, block_m=128, block_n=64, head_dim_granule=32)  # family S
        if self.block_dmodel <= 128:
            return replace(self, num_waves=8, block_m=256, block_n=64, head_dim_granule=64)  # family A
        return replace(self, num_waves=4, block_m=128, block_n=64, head_dim_granule=64)  # family B

    # Geometries whose *address helpers* are known correct, which is a stricter
    # set than the ones `make_traits` can describe.
    _SUPPORTED_GEOMETRIES = ((8, 256, 64, 64), (4, 128, 64, 64))

    def _check_helpers_support_geometry(self):
        """Refuse a geometry the kernel's addressing cannot actually serve.

        `fmha_traits_gfx950.make_traits` takes the geometry as parameters, so
        it will happily *describe* families S and B. The addressing has not
        caught up, and the gap is specific:

        - `_k_dma_m0_base` / `_v_dma_m0_base` place a tile line per wave per
          d-band (`wave * LINE + d * N_RPT * LINE`), which assumes
          `SMEM_N_RPT == NUM_WAVES` -- one issue per wave. Family B needs four,
          and waves 4..15's lines would simply never be written.
        - `init_dma_thread_offsets` splits a lane as `lane // VEC_KV` tokens by
          `lane % VEC_KV` D-buckets, which is the right split only when the
          granule is `VEC_KV * VEC_KV == 64`.
        - `_k_lds_read_base_per_lane` and `_swizzled_ks_offset` fold `%8`,
          `//8` and `//4` constants that are `SMEM_N_RPT` and
          `granule // K_STEP_QK` at family A's numbers.

        Failing here rather than at those sites keeps the diagnosis at the
        level of the decision. A geometry that builds and runs but addresses
        the wrong LDS produces plausible numbers, which is the failure mode
        this whole guard exists to avoid -- P2 measured exactly that at
        head_dim 192.
        """
        geom = (self.num_waves, self.block_m, self.block_n, self.head_dim_granule)
        if geom not in self._SUPPORTED_GEOMETRIES:
            raise NotImplementedError(
                f"geometry (waves, BLOCK_M, BLOCK_N, granule) = {geom} is describable but not yet "
                f"addressable: the DMA and LDS-read helpers assume SMEM_N_RPT == NUM_WAVES and a "
                f"64-element granule. Supported: {self._SUPPORTED_GEOMETRIES}. "
                "See P2 in sdpa-close-gap-gfx950.md."
            )
        return self

    def staging_shape(self):
        """`(tokens_per_issue, lines, issues_per_wave)` for this geometry.

        The coherence check the family table has to satisfy, written once so a
        new family is validated rather than asserted. A wave moves 512 bf16
        elements per DMA issue (64 lanes x 8), so the granule fixes how many
        tokens that covers, and BLOCK_N fixes how many such lines a KV tile
        needs. `issues_per_wave` must be a positive integer or the tile does
        not divide across the waves.
        """
        per_issue = (WARP_SIZE * DMA_BYTES // BF16_BYTES) // self.head_dim_granule
        if self.block_n % per_issue:
            raise ValueError(f"BLOCK_N {self.block_n} is not a multiple of {per_issue} tokens per DMA issue")
        lines = self.block_n // per_issue
        if lines % self.num_waves:
            raise ValueError(f"{lines} KV tile lines do not divide across {self.num_waves} waves")
        if self.block_dmodel < PV_MFMA_N:
            raise ValueError(
                f"block_dmodel {self.block_dmodel} is narrower than the PV MFMA's {PV_MFMA_N}-column "
                f"output; head_dim below {PV_MFMA_N} cannot have a native tile with v_mfma_f32_32x32x16"
            )
        return per_issue, lines, lines // self.num_waves

    def _with_traits(self, meta):
        """Build the dualwave traits this configuration implies.

        The last step. `fmha_traits_gfx950.make_traits` takes the geometry as
        parameters where the production constructor hardcodes it, and is
        checked field-by-field against production at family A's numbers -- so
        family A goes through this path today and the bitwise-vs-production
        test covers it.
        """
        self.staging_shape()  # the geometry must divide a KV tile evenly
        self._check_helpers_support_geometry()
        num_kv_heads = meta.num_heads if meta.num_kv_heads is None else meta.num_kv_heads
        traits = make_traits(
            num_heads=meta.num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=self.block_dmodel,
            num_waves=self.num_waves,
            block_m=self.block_m,
            block_n=self.block_n,
            granule=self.head_dim_granule,
            causal=meta.causal,
            dtype_str=meta.dtype_str,
            waves_per_eu=self.waves_per_eu,
            daz=self.daz,
            lazy_rescale=self.lazy_rescale,
            setprio=self.setprio,
            stagger=self.stagger,
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
