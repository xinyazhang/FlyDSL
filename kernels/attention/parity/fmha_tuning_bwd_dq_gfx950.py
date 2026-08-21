# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Traits and tuning policy for the gfx950 backward **dQ (and dB)** kernel.

Split from `fmha_bwd_dq_gfx950.py` for the reason `fmha_tuning_gfx950.py`
gives: the kernel file is about correctness, this one is about speed, and a
number here can only make a build slow, never wrong.

--- What the dQ kernel adds to the forward's traits --------------------------

Three things, and all of them are *build* axes rather than tuning ones:

- `STORE_DB` -- whether the kernel writes `dB = dS`. AOTriton's `bwd_kernel_dq`
  emits `DQ, DB` because `dS` is materialised per (q, k) element only there;
  our gfx1201 port dropped it. A build without it emits no store at all, which
  matters because the store is per element (see the kernel's
  `BwdDbStoreHelper` for why it cannot be a vector store).
- `HDIM_VO_FLOOR` -- the `hdim_vo` counterpart of `HDIM_QK_FLOOR`. This kernel
  reads *two* tiles through the K register path, one whose D axis is the qk
  extent and one whose is the vo extent, so a single floor cannot serve both.
- `LDS_KV_TOTAL_SIZE`, cut to one buffer. `make_bwd_dq_traits` says why; it is
  what makes head_dim 512 fit at all.

--- The geometry is dQ's own, and it is measured -----------------------------

`BwdDqKnobs._with_wave_geometry` replaces the forward's family table rather
than extending it, and the docstring there carries the full
`(num_waves, waves_per_eu)` sweep it came from. The headline is that
`waves_per_eu = 1` is worth up to 4x and the forward's default of 2 is wrong
for this kernel at every rung measured.

The MFMA shape is still not baked in anywhere: `ROWS_PER_WAVE`, `D_CHUNK` and
`K_STEP_QK` all come from `fmha_traits_gfx950`, and `BLOCK_M` is derived as
`num_waves * MFMA_M` rather than chosen -- which is the P7 invariant this phase
had to stop stating in prose.
"""

from dataclasses import dataclass, fields, replace

from fmha_traits_gfx950 import ParityDualwaveTraits, make_traits
from fmha_tuning_gfx950 import LADDER, Gfx950Knobs, tile_width_for

__all__ = [
    "BWD_DQ_LADDER",
    "BwdDqKnobs",
    "BwdDqTraits",
    "bwd_dq_knobs",
    "make_bwd_dq_traits",
]

# The rungs this kernel is built and tested for. B3 brings it to the forward's
# `LADDER` in full; `tile_width_for` still owns the rounding rule, so a
# head_dim of 100 rounds to 128 exactly as it does forward.
#
# Kept as its own name rather than an alias for `LADDER` so a rung can be
# withdrawn here without touching the forward -- which is what "a rung that has
# never been run is not a rung" needs to stay enforceable.
BWD_DQ_LADDER = LADDER


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

    # The `hdim_vo` counterpart of `HDIM_QK_FLOOR`, for the V tile's own
    # padded-head mask. A field rather than a knob-only value because the two
    # extents are read by *different* helper instances in this kernel and
    # nothing else would keep them together.
    HDIM_VO_FLOOR: int = 0


def make_bwd_dq_traits(*, store_db=False, hdim_vo_floor=0, **kwargs):
    """`fmha_traits_gfx950.make_traits`, widened, and cut to **one** LDS buffer.

    Delegating rather than transcribing is the whole point: every derived
    field -- the LDS line strides, the DMA split, the V transpose strides the
    third GEMM reads K through -- comes from the one constructor the forward's
    `assert_matches_production` pins against production. A second copy of any
    of them would be a second thing to keep in step, and the read and write
    sides of an LDS layout drifting apart is the specific failure this avoids.

    **The one field this overrides is `LDS_KV_TOTAL_SIZE`.** The forward sizes
    its allocation for `NUM_PREFETCH_K` KV tiles in flight; the dQ kernel keeps
    one, and after B3's single-staging of K it needs exactly one K-pitch region
    plus one V-pitch region -- which is `DUALWAVE_SWP_KV_PER_BUFFER`, already
    derived. Overriding the *total* and nothing else is what keeps the buffer
    bases, the `m0` tables and the tile sizes the shared derivation's.

    At head_dim 512, BLOCK_N 64: 66560 + 69632 = 136192 B against a 163840 B
    cap. The three-slot B2 layout needed 199 KB there, which is why the second
    staging of K was a B3 blocker rather than a B7 optimisation.
    """
    base = make_traits(**kwargs)
    carried = {f.name: getattr(base, f.name) for f in fields(base) if f.name != "LDS_KV_TOTAL_SIZE"}
    return BwdDqTraits(
        **carried,
        STORE_DB=bool(store_db),
        HDIM_VO_FLOOR=int(hdim_vo_floor),
        LDS_KV_TOTAL_SIZE=base.DUALWAVE_SWP_KV_PER_BUFFER,
    )


@dataclass(frozen=True)
class BwdDqKnobs(Gfx950Knobs):
    """The forward's knob pipeline, with the dQ build axes added.

    `resolve` is inherited whole; only the last step (`_with_traits`) is
    replaced, because the pipeline before it -- mode checks, the ladder, the
    wave geometry -- asks exactly the same questions for the backward.
    """

    store_db: bool | None = None

    # The forward's list plus the two-wave points. **Not a relaxation of the
    # check** -- the check exists because `_k_dma_m0_base` assumes one DMA
    # issue per wave, and `_ParityKvStaging` is the generalisation that lifts
    # it (`ISSUES_PER_WAVE = SMEM_N_RPT // NUM_WAVES`, 4 at two waves and
    # granule 64). What makes these entries legitimate is that every one of
    # them is *run* by `test_wave_geometries_agree`, which requires them to
    # agree with the default build to the error-ratio gate.
    #
    # Two waves is here because B1 measured it as the best point at head_dim
    # 128 (788 TF against 403 at four waves and `waves_per_eu=2`), and a
    # geometry the tuner cannot express is a lever that does not exist.
    _SUPPORTED_GEOMETRIES = Gfx950Knobs._SUPPORTED_GEOMETRIES + (
        (2, 64, 64, 64),
        (2, 64, 64, 32),
    )

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

    def _with_d_axis_splits(self):
        """Both D-axis cuts stay at 1. **This is load-bearing, not tidiness.**

        The forward's version turns `d_stages` on at block_dmodel > 256 and
        `vo_shards` on at > 384, because its LDS does not hold two KV tiles
        that wide and its O accumulator is the whole AGPR file. This body has
        one tile in flight and 133 KB at head_dim 512, so neither applies --
        and neither is *implemented* here.

        The failure if this is left to the parent is not a build error. With
        `D_STAGES = 2`, `ParityGemmHelper.qk` silently becomes `qk_stage(...,
        stage=0)` and `pv` writes only `v_o[0 : D_CHUNKS/2]`: the kernel
        reduces over half the head dim, writes half the accumulator, and
        returns a finite wrong answer. Caught here by an LDS figure that was
        half what it should have been, which is the only reason it was caught
        at all -- so `_with_traits` also refuses a pinned `d_stages > 1`
        rather than relying on this default holding.
        """
        return replace(
            self,
            d_stages=1 if self.d_stages is None else self.d_stages,
            qk_shards=1 if self.qk_shards is None else self.qk_shards,
            vo_shards=1 if self.vo_shards is None else self.vo_shards,
        )

    def _hdim_vo_floor(self, meta):
        """The V tile's counterpart of `hdim_qk_floor`, exclusive.

        The floor exists so the padded-head mask can skip the D columns the
        dispatcher guarantees are real -- it is what keeps a padded build from
        paying the 27-54% the forward measured for masking every K-step. The
        *qk* floor comes from the ladder rung; the *vo* one has to come from
        the same rung, because both extents live in the same compiled tile.

        It is dropped to 0 when `head_dim_v` is at or below it. That happens
        for an asymmetric call like `head_dim 128, head_dim_v 40`: the 128 tile
        promises `hdim_qk > 96` and promises nothing at all about `hdim_vo`, so
        the V mask has to cover every step. Slower, and the alternative is
        reducing over the caller's padding.
        """
        head_dim_v = meta.head_dim if meta.head_dim_v is None else meta.head_dim_v
        floor = int(self.hdim_qk_floor or 0)
        return floor if head_dim_v > floor else 0

    def _with_wave_geometry(self):
        """dQ's own geometry table. Measured, not inherited.

        The forward's families answer questions this body does not ask: family
        W exists to shard its O accumulator (`VO_SHARDS`) and stages the D axis
        at 384 because two KV tiles in flight do not fit LDS there. Neither
        holds after B3 -- one tile in flight, and head_dim 512 is 133 KB.

        Measured at `B=2 H=8 S=2048` bf16 non-causal, TFLOP/s on `6*S^2*d`,
        over the whole `(num_waves, waves_per_eu)` grid:

            head_dim   w2/e1  w2/e2  w4/e1  w4/e2  w8/e1  w8/e2
                32       335    334    423    420      -      -
                64       501    500    582    580    436    435
                96       599    597    678    681      -      -
               128       715    622    685    702    496    505
               160       665    313    645    429      -      -
               192       672    260    657    350    311    316
               224       682    667    677    171      -      -
               256       712    694    675    169    162    162
               384       339    336    335    338    338    336
               512       115    113    113    117    115    114

        (`-` is `SMEM_N_RPT % NUM_WAVES != 0`: a granule-32 tile has four KV
        lines, so eight waves cannot each own one.)

        Two things in that table are policy and one is a finding.

        **`waves_per_eu = 1` unconditionally.** It is never worse than 2 and it
        is worth up to 4x -- head_dim 256 at four waves is 675 against 169. The
        addendum's discriminator explains it exactly: the ISA dump for the slow
        build reports 256 VGPRs, **0 AGPRs and 191 spills**, against 460 VGPR +
        204 AGPR and zero spills for the fast one. It was not short of
        registers, it was forbidden from using half of them.

        **Two waves from 128 to 256, four elsewhere.** Four wins below 128 and
        two wins above it, both by margins the table can support. Between the
        neighbouring points the difference is often under 10%, which per the
        lore is not a result a sweep can settle -- those are left where they
        fall rather than tuned.

        **384 and 512 are flat across the entire grid**, which is the finding:
        they are not occupancy-limited and no wave count reaches them. See the
        B3 outcome in `sdpa-bwd-plan-gfx950.md`.
        """
        me = self._with_d_axis_splits()
        # Decided before the pin check, and separately from it: the wave count
        # is a *geometry* choice that has to move with BLOCK_M, while this is a
        # register-budget hint that is right at 1 whatever the geometry. A
        # sweep pinning the four geometry fields still gets it.
        if me.waves_per_eu is None:
            me = replace(me, waves_per_eu=1)
        pinned = (me.num_waves, me.block_m, me.block_n, me.head_dim_granule)
        if all(x is not None for x in pinned):
            return me
        if any(x is not None for x in pinned):
            raise ValueError(
                f"pin num_waves, block_m, block_n and head_dim_granule together or not at all, got {pinned}"
            )
        if me.block_dmodel is None:
            raise ValueError("_with_wave_geometry runs after _with_widths; block_dmodel is not resolved")
        waves = 2 if 128 <= me.block_dmodel <= 256 else 4
        return replace(
            me,
            num_waves=waves,
            # `ROWS_PER_WAVE` is pinned at the MFMA's M extent, so BLOCK_M is
            # derived rather than chosen -- `make_traits` raises on any other
            # value, which is the P7 invariant this phase had to stop stating
            # in prose.
            block_m=waves * 32,
            block_n=64,
            head_dim_granule=32 if me.block_dmodel % 64 else 64,
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
                f"{self.block_dmodel}. head_dim {meta.head_dim} rounds to {tile_width_for(meta.head_dim)}."
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
                    f"{name}=True is not implemented by the backward dQ kernel yet; B3 is dense and "
                    "non-causal over the full ladder. See the phase ladder in sdpa-bwd-plan-gfx950.md."
                )
        if self.num_kv_splits != 1:
            raise NotImplementedError("split-K is out of scope for every backward kernel; see plan section 9")
        for name, value in (("d_stages", self.d_stages), ("qk_shards", self.qk_shards), ("vo_shards", self.vo_shards)):
            if value != 1:
                # **Refused, because the body does not implement it and the
                # inherited helpers half-do.** `D_STAGES > 1` makes
                # `ParityGemmHelper.qk` reduce over one stage of the head dim
                # and `pv` write one stage of the accumulator, and this loop
                # never advances the stage -- so the answer is finite, the
                # right shape, and computed over a fraction of `d`. The
                # forward's default policy turns `d_stages` on at
                # block_dmodel > 256, which is exactly the rungs B3 added.
                raise NotImplementedError(
                    f"{name}={value} is described by the traits but not implemented by the dQ body: it "
                    "would reduce over part of the head dim and return a finite wrong answer. The wide "
                    "rungs do not need it -- head_dim 512 is 133 KB of LDS unstaged."
                )
        if self.block_dmodel_v != self.block_dmodel:
            # Not the same thing as an asymmetric *head_dim*, which B3 does
            # serve: this is an asymmetric compiled *tile*, which would give
            # GEMM3's K reads and the dQ store two different `D_CHUNKS`. B2
            # refused it as part of a wider refusal; keep the narrow one.
            raise NotImplementedError(
                f"block_dmodel_v {self.block_dmodel_v} != block_dmodel {self.block_dmodel}: dQ's output "
                "width is the *qk* extent, so a second tile width has nothing here to describe"
            )

        self.staging_shape()
        self._check_helpers_support_geometry()
        num_kv_heads = meta.num_heads if meta.num_kv_heads is None else meta.num_kv_heads
        traits = make_bwd_dq_traits(
            store_db=bool(self.store_db),
            hdim_vo_floor=self._hdim_vo_floor(meta),
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
        # One K-pitch region plus one V-pitch region, which is what
        # `make_bwd_dq_traits` cut `LDS_KV_TOTAL_SIZE` down to. Checked against
        # the same cap the forward uses, because the cap is the hardware's.
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
    # **Deliberately absent**: `waves_per_eu`. The forward pins 2 here; this
    # kernel wants 1 and `_with_wave_geometry` says why, with the measurement.
    # Left `None` so that step decides it and a caller can still override.
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
