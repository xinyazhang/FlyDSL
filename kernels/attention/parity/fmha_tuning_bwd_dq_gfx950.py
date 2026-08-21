# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Traits and tuning policy for the gfx950 backward **dQ (and dB)** kernel.

Split from `fmha_bwd_dq_gfx950.py` for the reason `fmha_tuning_gfx950.py`
gives: the kernel file is about correctness, this one is about speed, and a
number here can only make a build slow, never wrong.

--- What the dQ kernel adds to the forward's traits --------------------------

Two fields, and both are *build* axes rather than tuning ones:

- `STORE_DB` -- whether the kernel writes `dB = dS`. AOTriton's `bwd_kernel_dq`
  emits `DQ, DB` because `dS` is materialised per (q, k) element only there;
  our gfx1201 port dropped it. A build without it emits no store at all, which
  matters because the store is per element (see the kernel's
  `BwdDbStoreHelper` for why it cannot be a vector store).
- `BWD_LDS_SLOTS` -- documentation, not arithmetic. The dQ kernel needs *three*
  KV-shaped LDS tiles per KV block, not two, and the field records which of the
  forward's four staging slots each lands in. See the kernel's module
  docstring.

--- Why the geometry is the forward's, for now -------------------------------

`Gfx950Knobs._with_wave_geometry` picks family A (8 waves, BLOCK_M 256,
BLOCK_N 64, granule 64) at head_dim <= 128, and this phase reuses that
unchanged. The plan (section 4) is emphatic that the MFMA shape must not be
baked in, and it is not: everything below reads `MFMA_M` / `D_CHUNK` from
`fmha_traits_gfx950`. What *is* pinned here is the wave count, and it is
pinned by measurement rather than by assumption -- see the outcome section of
`sdpa-bwd-plan-gfx950.md`.
"""

from dataclasses import dataclass, fields, replace

from fmha_traits_gfx950 import ParityDualwaveTraits, make_traits
from fmha_tuning_gfx950 import Gfx950Knobs, tile_width_for

__all__ = [
    "BWD_DQ_LADDER",
    "BwdDqKnobs",
    "BwdDqTraits",
    "bwd_dq_knobs",
    "make_bwd_dq_traits",
]

# The rungs this kernel is built and tested for. A strict subset of the
# forward's `LADDER`: B2 is dense, non-causal, head_dim 64 and 128, and a rung
# that has never been run is not a rung. `tile_width_for` still owns the
# rounding rule, so a head_dim of 96 rounds to 128 exactly as it does forward.
BWD_DQ_LADDER = (64, 128)


@dataclass(frozen=True)
class BwdDqTraits(ParityDualwaveTraits):
    """The forward's parity traits plus what the dQ kernel alone needs.

    A subclass for the same reason `ParityDualwaveTraits` is one: the parent is
    frozen with no defaulted fields, so added fields need defaults and those
    defaults are the "behave like the forward" values. Nothing that reads the
    parent's fields has to know these exist.
    """

    # Write `dB = dS`. Off by default; the store is per element and costs 32
    # `buffer_store`s per lane per KV tile, so a build that does not want it
    # must not pay for it.
    STORE_DB: bool = False

    # Which of the forward's four LDS staging slots each of the dQ kernel's
    # three tiles occupies, as `(k_buf, v_buf, k_buf)` indices. Not read by any
    # arithmetic -- it is here so the assignment is stated in one place that the
    # kernel and a reader of the traits can both point at.
    BWD_LDS_SLOTS: tuple = (("k", 0), ("v", 0), ("k", 1))


def make_bwd_dq_traits(*, store_db=False, **kwargs):
    """`fmha_traits_gfx950.make_traits`, widened to `BwdDqTraits`.

    Delegating rather than transcribing is the whole point: every derived
    field -- the LDS line strides, the DMA split, the V transpose strides the
    third GEMM reads K through -- comes from the one constructor the forward's
    `assert_matches_production` pins against production. A second copy of any
    of them would be a second thing to keep in step, and the read and write
    sides of an LDS layout drifting apart is the specific failure this avoids.
    """
    base = make_traits(**kwargs)
    return BwdDqTraits(
        **{f.name: getattr(base, f.name) for f in fields(base)},
        STORE_DB=bool(store_db),
    )


@dataclass(frozen=True)
class BwdDqKnobs(Gfx950Knobs):
    """The forward's knob pipeline, with the dQ build axes added.

    `resolve` is inherited whole; only the last step (`_with_traits`) is
    replaced, because the pipeline before it -- mode checks, the ladder, the
    wave geometry -- asks exactly the same questions for the backward.
    """

    store_db: bool | None = None

    def resolve(self, meta) -> "BwdDqKnobs":
        """The forward's pipeline, seeded from *this* kernel's fallback.

        Overridden only for the seed. `Gfx950Knobs.resolve` merges into
        `_GFX950_FALLBACK`, which is a `Gfx950Knobs` and knows nothing about
        `store_db` -- `dataclasses.replace` on it would raise rather than drop
        the field, but the failure would come from a stack frame two files away
        from the cause. The step sequence below is the parent's, verbatim.
        """
        return (
            _BWD_DQ_FALLBACK.merge(self)
            ._checked_modes()
            ._with_mode_defaults(meta)
            ._with_widths(meta)
            ._with_wave_geometry()
            ._with_traits(meta)
        )

    def _with_traits(self, meta):
        """The forward's `_with_traits`, against `make_bwd_dq_traits`.

        Copied in structure rather than called through, because the parent
        hardcodes `make_traits` as its last expression and there is no seam
        below it. The divergences are named: the rung check, `store_db`, and
        the modes B2 does not implement.
        """
        if self.block_dmodel not in BWD_DQ_LADDER:
            raise NotImplementedError(
                f"the backward dQ kernel is built for head_dim tiles {BWD_DQ_LADDER}, not "
                f"{self.block_dmodel}. head_dim {meta.head_dim} rounds to {tile_width_for(meta.head_dim)}; "
                "the wider rungs are B3."
            )
        # Refused rather than ignored. Each of these would build and run and
        # return the right *shape*, which is the failure mode the whole
        # backward plan is written to avoid.
        for name, value in (
            ("causal", meta.causal),
            ("window", meta.window),
            ("bias", meta.bias),
            ("dropout", meta.dropout),
            ("varlen", self.varlen),
            ("paged", self.paged),
        ):
            if value:
                raise NotImplementedError(
                    f"{name}=True is not implemented by the backward dQ kernel yet; B2 is dense, "
                    "non-causal, head_dim 64/128, bf16. See the phase ladder in sdpa-bwd-plan-gfx950.md."
                )
        if self.num_kv_splits != 1:
            raise NotImplementedError("split-K is out of scope for every backward kernel; see plan section 9")
        if meta.head_dim_v is not None and meta.head_dim_v != meta.head_dim:
            # **Asymmetric hdim is a real gap, not a missing flag.** The dQ
            # kernel reads V through the *K* path (`load_k(1)` for GEMM2),
            # whose padded-head mask is written against `hdim_qk`; and it
            # writes dQ through the O store, whose suppression is written
            # against `hdim_vo`. Both are backwards for this kernel the moment
            # the two extents differ. Two named sites, both one line, and
            # neither is testable inside B2's dense head_dim 64/128 scope.
            raise NotImplementedError(
                f"head_dim_v {meta.head_dim_v} != head_dim {meta.head_dim} is not implemented by the "
                "backward dQ kernel: GEMM2 masks V's columns against hdim_qk and the dQ store "
                "suppresses against hdim_vo, and both are the other extent here. B3."
            )
        if self.padded_head:
            # Same two sites. A padded head with `hdim_qk == hdim_vo` may well
            # already be correct -- the Q, K and store masks are all inherited
            # -- but "may well be" is what this plan exists to refuse. B3 turns
            # it on with the ladder, where it can be tested against the rungs
            # it is for.
            raise NotImplementedError(
                f"head_dim {meta.head_dim} needs the padded-head path (tile {self.block_dmodel}), which "
                "the backward dQ kernel does not claim yet. B2 serves exact head_dim 64 and 128; B3 "
                "brings the ladder and the padded head together."
            )

        self.staging_shape()
        self._check_helpers_support_geometry()
        num_kv_heads = meta.num_heads if meta.num_kv_heads is None else meta.num_kv_heads
        traits = make_bwd_dq_traits(
            store_db=bool(self.store_db),
            num_heads=meta.num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=self.block_dmodel,
            num_waves=self.num_waves,
            block_m=self.block_m,
            block_n=self.block_n,
            granule=self.head_dim_granule,
            d_stages=self.d_stages,
            qk_shards=self.qk_shards,
            vo_shards=self.vo_shards,
            v_half_wave=self.v_half_wave,
            v_n_group=self.v_n_group,
            v_k_substep=self.v_k_substep,
            v_dc_in_pair=self.v_dc_in_pair,
            causal=False,
            dtype_str=meta.dtype_str,
            waves_per_eu=self.waves_per_eu,
            daz=self.daz,
            lazy_rescale=self.lazy_rescale,
            setprio=self.setprio,
            stagger=self.stagger,
            lpt_tile_order=self.lpt_tile_order,
            num_kv_splits=1,
            varlen=False,
            cross_seqlen=False,
            paged=False,
            kv_cache_layout=self.kv_cache_layout,
            kv_vectorized=False,
            return_lse=False,
        )
        # **The dQ kernel occupies three of the four staging slots, not two**,
        # so the LDS bill is the forward's, not three quarters of it: it uses
        # the whole `LDS_KV_TOTAL_SIZE` allocation and leaves the fourth slot
        # unused. Checked against the same cap for that reason.
        lds_bytes = traits.LDS_KV_TOTAL_SIZE * traits.BF16_BYTES
        if lds_bytes > self.LDS_CAP_BYTES:
            raise ValueError(
                f"KV staging needs {lds_bytes} B of LDS, over the {self.LDS_CAP_BYTES} B cap, for "
                f"block_dmodel {self.block_dmodel} at BLOCK_N {self.block_n}"
            )
        return replace(self, traits=traits)


def bwd_dq_knobs(arch: str = "gfx950", **overrides) -> BwdDqKnobs:
    """The knob object for the backward dQ kernel on `arch`.

    Mirrors `fmha_tuning_gfx950.fmha_knobs`, including its arch-*prefix* match
    so a full `gcnArchName` -- "gfx950:sramecc+:xnack-" -- works unstripped.
    """
    base = arch.split(":")[0].lower() if arch else ""
    if not base.startswith("gfx950"):
        raise ValueError(f"the backward dQ kernel is gfx950-only, got arch {arch!r}")
    if "traits" in overrides:
        raise TypeError("`traits` is set by resolve(), not by the caller")
    known = {f.name for f in fields(BwdDqKnobs)}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(f"unknown BwdDqKnobs field(s): {sorted(unknown)}")
    return BwdDqKnobs(**overrides)


# Defaults the policy has no shape-dependent opinion about. `lpt_tile_order`
# and the causal-only schedule knobs are inert in a dense build; they are
# spelled out anyway so a resolved object never carries a `None` past
# `resolve`, which is what `Gfx950Knobs._checked_modes` reads.
_BWD_DQ_FALLBACK = BwdDqKnobs(
    waves_per_eu=2,
    daz=True,
    lazy_rescale=True,
    setprio=True,
    stagger=True,
    lpt_tile_order=False,
    varlen=False,
    cross_seqlen=False,
    paged=False,
    kv_cache_layout="linear",
    num_kv_splits=1,
    return_lse=False,
    strides_constexpr=False,
    store_db=False,
)
