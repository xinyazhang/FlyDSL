# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tuning policy for the gfx950 dK/dV backward kernel: which knobs, for which shape.

The backward twin of `fmha_tuning_gfx950.py`, and split from the kernel for the
same reason: a number here moves when a sweep says so, and nothing here can
make a build *wrong*, only slow.

--- The role swap, stated once ---------------------------------------------

**dK/dV is the forward loop transposed.** K and V stay resident in registers
and Q and dO stream past through LDS, where the forward keeps Q resident and
streams K/V. The dualwave traits are written against the forward's roles, and
this module reuses them rather than forking them, so two of their names mean
the other thing here:

| traits field | forward | dK/dV |
|---|---|---|
| `BLOCK_M` | Q rows per workgroup | **KV rows per workgroup** (resident) |
| `BLOCK_N` | KV rows per streamed tile | **Q rows per streamed tile** |
| `ROWS_PER_WAVE` | Q rows per wave | **KV rows per wave** |

`BwdDkDvTraits` re-exposes the first two under `BLOCK_KV` / `BLOCK_Q`, and the
kernel body uses only those. Everything derived from `BLOCK_N` -- `SMEM_N_RPT`,
the DMA issue count, the LDS line layout -- is about *a tile of 64 rows staged
through LDS* and is correct unchanged, because the staging does not care whose
rows they are.

--- What B1 is, and is not -------------------------------------------------

Dense, non-causal, head_dim 64 and 128, bf16, `num_heads_q == num_heads_k`.
The ladder (B3), causal and windows (B4), varlen (B5) and dropout/bias (B6)
are later phases; the `const_expr` hooks are left where they go, and nothing
here half-implements one. `resolve` raises rather than silently serving a
configuration the kernel does not compute.
"""

from dataclasses import dataclass, fields, replace

from fmha_traits_gfx950 import ParityDualwaveTraits, make_traits

__all__ = [
    "LADDER",
    "BwdDkDvInputMetadata",
    "BwdDkDvKnobs",
    "BwdDkDvTraits",
    "bwd_dkdv_knobs",
    "tile_width_for",
]

# The compiled tile widths this phase builds. Deliberately two entries rather
# than the forward's ten: B1's gate is correctness against autograd, and every
# extra rung is a second shape for a wrong answer to hide in. B3 widens it, and
# `tile_width_for` is the one place that has to change.
LADDER = (64, 128)

# The streamed tile is always 64 q rows: `ds_read_b64_tr_b16` covers 16 rows per
# k-substep and the MFMA takes four of them, so 64 is what one pass of the
# transposed operand spans. The granule is family A's, which is what keeps the
# LDS staging and the lane->(row, d) map the *validated* ones rather than a
# re-derivation.
DEFAULT_BLOCK_Q = 64
DEFAULT_HEAD_DIM_GRANULE = 64

# Waves per workgroup, by tile width. `BLOCK_KV` follows: a wave owns the
# MFMA's 32-row M extent, so `block_kv = 32 * num_waves`.
#
# **This is the AGPR cliff, and it is sharper here than in the forward.** A
# dK/dV wave carries *two* accumulators where the forward carries one, so at
# head_dim 128 they are 128 VGPRs before anything else is live. Measured at
# `B=4 H=8 S=4096` bf16 on real FLOPs:
#
#   head_dim  waves  BLOCK_KV  waves_per_eu  AGPR  spills   TFLOP/s
#      64       8      256          2          0      0       657
#      64       4      128          2          0      0       698
#      64       2       64          2          0      0       666
#      64       1       32          2          0      0       595
#     128       8      256          2          0    118       444
#     128       4      128          2          0    126       403
#     128       4      128          1          -      -        721
#     128       2       64          2        108      0       788
#     128       1       32          2          -      -        475
#
# At 8 waves (2 per SIMD) a wave may address 256 registers total, so the
# allocator cannot reach the AGPR file at all and spills to scratch instead;
# at 2 it takes 108 AGPRs and spills nothing, which is worth 1.8x. That is the
# lore's "check `agpr_count` alongside spills" arriving one width earlier than
# it does on the forward. head_dim 64 fits either way and prefers 4 by 5%,
# confirmed by interleaved single-GPU A/B (360.5 us against 379.3, nine
# interleaved reps) rather than by the sweep above, per plan section B7.
_WAVES_BY_WIDTH = {64: 4, 128: 2}

# LDS is `2 tensors * NUM_STREAM_BUFFERS * BLOCK_Q * head_dim * ~8.5 B`, and the
# addressable cap is 163840 B. head_dim 128 needs 69632 B, so both rungs fit
# with room; the check exists so B3's wider rungs fail at the decision rather
# than as "local memory exceeds limit" with no indication of which knob to move.
LDS_CAP_BYTES = 163840


def tile_width_for(head_dim):
    """The compiled tile width serving `head_dim`, or raise saying why not."""
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    for rung in LADDER:
        if head_dim <= rung:
            return rung
    raise NotImplementedError(
        f"head_dim {head_dim} needs a rung this phase does not build. B1 serves {LADDER}; "
        "the full ladder is B3 (see sdpa-bwd-plan-gfx950.md)."
    )


@dataclass(frozen=True)
class BwdDkDvTraits(ParityDualwaveTraits):
    """The forward's traits, read with the backward's role names.

    A subclass with no new *derived* state: every field is the parent's, and
    the two properties below are aliases that make the kernel body readable
    rather than new numbers. That is deliberate -- the moment the LDS layout
    here stops being the forward's, the transpose read (which is the forward's
    V read, unmodified) stops being validated.

    `NUM_STREAM_BUFFERS` is the one addition. It is 2 because the body
    prefetches one tile ahead: the DMA for tile `j+2` is issued once every wave
    has finished reading tile `j`, so it overlaps tile `j+1`'s compute.
    """

    NUM_STREAM_BUFFERS: int = 2

    @property
    def BLOCK_KV(self):
        """KV rows one workgroup owns, resident in registers across the Q loop."""
        return self.BLOCK_M

    @property
    def BLOCK_Q(self):
        """Q rows one streamed tile carries through LDS."""
        return self.BLOCK_N

    @property
    def STREAM_LINE_STRIDE(self):
        """LDS elements per staged line.

        **Both streamed tiles use the forward's *V* line stride, not its K
        one**, and that is a correctness requirement rather than a padding
        preference. Q and dO are each read two ways -- row-major for the
        d-contracted GEMMs and column-major through `ds_read_b64_tr_b16` for
        the q-contracted ones -- and the transpose read path is only validated
        against `SMEM_V_LINE_STRIDE`. The row-major read does not care which
        stride it is given, so making both tiles V-shaped is what lets one
        staged tile serve both readers.
        """
        return self.SMEM_V_LINE_STRIDE

    @property
    def STREAM_TILE_ELEMS(self):
        """LDS elements one staged tile occupies."""
        return self.SMEM_V_TILE_ELEMS

    @property
    def LDS_STREAM_TOTAL_SIZE(self):
        """LDS elements for both tensors across all buffers."""
        return 2 * self.NUM_STREAM_BUFFERS * self.STREAM_TILE_ELEMS


def _as_bwd_traits(base, **extra):
    """Re-wrap a `ParityDualwaveTraits` as `BwdDkDvTraits`, field for field."""
    return BwdDkDvTraits(**{f.name: getattr(base, f.name) for f in fields(base)}, **extra)


@dataclass(frozen=True)
class BwdDkDvInputMetadata:
    """What to compute. Set by the caller; never by policy, and never by arch."""

    num_heads: int
    head_dim: int
    dtype_str: str = "bf16"
    num_kv_heads: int | None = None
    sm_scale: float | None = None

    # B4. Left where the feature goes, so the ABI and the tuning surface do not
    # move when it arrives. `resolve` rejects a True.
    causal: bool = False
    window: bool = False
    # B6.
    dropout: bool = False
    bias: bool = False


@dataclass(frozen=True)
class BwdDkDvKnobs:
    """How to compute it. `None` means "policy decides".

    `None` rather than a literal default for the same reason as the forward's
    knobs: it is the only way `resolve` can tell "the caller wants 1" from "the
    caller did not say".
    """

    block_dmodel: int | None = None
    padded_head: bool | None = None

    num_waves: int | None = None
    block_kv: int | None = None
    block_q: int | None = None
    head_dim_granule: int | None = None
    num_stream_buffers: int | None = None

    waves_per_eu: int | None = None
    daz: bool | None = None
    strides_constexpr: bool | None = None

    # Set by `resolve`; never by a caller.
    traits: object | None = None

    def merge(self, other):
        """`other`'s set fields win; its `None`s leave this one's alone."""
        if other is None:
            return self
        set_fields = {f.name: getattr(other, f.name) for f in fields(other) if getattr(other, f.name) is not None}
        return replace(self, **set_fields)

    def resolve(self, meta: BwdDkDvInputMetadata) -> "BwdDkDvKnobs":
        """The complete build configuration for `meta`, traits included.

        Idempotent: every derived field is recomputed from `meta` and the
        pinned fields rather than read back.
        """
        return _FALLBACK.merge(self)._checked_scope(meta)._with_widths(meta)._with_geometry()._with_traits(meta)

    def _checked_scope(self, meta):
        """Refuse what B1 does not compute, at the decision rather than at an address.

        Every one of these would otherwise produce a plausible, finite, wrong
        gradient: a dropped causal mask returns dense attention's gradient, and
        a GQA call returns one q head's contribution where the sum over the
        group belongs. Both are the right shape.
        """
        for name in ("causal", "window", "dropout", "bias"):
            if getattr(meta, name):
                raise NotImplementedError(
                    f"{name}=True is not implemented in B1. This phase is dense, non-causal, "
                    "head_dim 64/128, bf16; see sdpa-bwd-plan-gfx950.md phases B4-B6."
                )
        if meta.dtype_str != "bf16":
            raise NotImplementedError(f"B1 builds bf16 only, got {meta.dtype_str!r}")
        num_kv_heads = meta.num_heads if meta.num_kv_heads is None else meta.num_kv_heads
        if num_kv_heads != meta.num_heads:
            raise NotImplementedError(
                f"GQA (num_heads {meta.num_heads}, num_kv_heads {num_kv_heads}) needs dK/dV summed "
                "over every q head sharing a kv head, which this kernel does not do: one workgroup "
                "owns one (q head, kv block) and would write, not accumulate. B1 is MHA only."
            )
        return self

    def _with_widths(self, meta):
        """Decide `block_dmodel` and `padded_head`."""
        block_dmodel = self.block_dmodel
        if block_dmodel is None:
            block_dmodel = tile_width_for(meta.head_dim)
        elif block_dmodel not in LADDER:
            raise ValueError(f"block_dmodel must be one of the built rungs {LADDER}, got {block_dmodel}")
        if meta.head_dim > block_dmodel:
            raise ValueError(f"head_dim {meta.head_dim} does not fit the pinned block_dmodel {block_dmodel}")
        padded_head = self.padded_head
        if padded_head is None:
            padded_head = meta.head_dim != block_dmodel
        if padded_head:
            raise NotImplementedError(
                f"head_dim {meta.head_dim} would need the {block_dmodel}-wide tile with the surplus D "
                "columns masked. B1 serves exact rung widths only; the padded-head path is B3."
            )
        return replace(self, block_dmodel=block_dmodel, padded_head=False)

    def _with_geometry(self):
        """Decide the wave geometry and staging granule from the tile width.

        `_WAVES_BY_WIDTH` is the whole policy, and its table is where the
        measurement lives. `BLOCK_KV` is *derived* from it rather than pinned
        beside it -- a wave owns exactly the MFMA's 32-row M extent, so the two
        cannot disagree, which is the failure P7 found twelve instances of on
        the forward. Pinnable as a set so a sweep can cross the boundary
        without editing this.

        Reads `self.block_dmodel`, so it runs after `_with_widths`.
        """
        pinned = (self.num_waves, self.block_kv, self.block_q, self.head_dim_granule)
        if all(x is not None for x in pinned):
            return self
        if any(x is not None for x in pinned):
            raise ValueError(
                f"pin num_waves, block_kv, block_q and head_dim_granule together or not at all, got {pinned}"
            )
        num_waves = _WAVES_BY_WIDTH[self.block_dmodel]
        return replace(
            self,
            num_waves=num_waves,
            block_kv=32 * num_waves,
            block_q=DEFAULT_BLOCK_Q,
            head_dim_granule=DEFAULT_HEAD_DIM_GRANULE,
        )

    def _with_traits(self, meta):
        """Build the traits this configuration implies.

        `make_traits` is the forward's, called with `block_m=block_kv` and
        `block_n=block_q` -- see the module docstring for why those two slots
        change meaning and nothing else does.
        """
        num_kv_heads = meta.num_heads if meta.num_kv_heads is None else meta.num_kv_heads
        base = make_traits(
            num_heads=meta.num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=self.block_dmodel,
            num_waves=self.num_waves,
            block_m=self.block_kv,
            block_n=self.block_q,
            granule=self.head_dim_granule,
            causal=False,
            dtype_str=meta.dtype_str,
            waves_per_eu=self.waves_per_eu,
            daz=self.daz,
        )
        traits = _as_bwd_traits(base, NUM_STREAM_BUFFERS=self.num_stream_buffers)
        if traits.ROWS_PER_WAVE != 32:
            # P7's ceiling, restated for the backward: a wave holds at most the
            # MFMA's M extent in KV rows, and fewer than 32 is legal but runs a
            # full 32-row MFMA and discards the rest. `make_traits` already
            # rejects more; this rejects less, because the dK/dV accumulators
            # are indexed by the full 32 and a short block would store rows the
            # workgroup does not own.
            raise ValueError(
                f"block_kv {self.block_kv} over {self.num_waves} waves gives {traits.ROWS_PER_WAVE} KV "
                f"rows per wave; the dK/dV accumulator and its store are written against the MFMA's "
                f"32-row M extent. Pass block_kv={self.num_waves * 32}."
            )
        lds_bytes = traits.LDS_STREAM_TOTAL_SIZE * traits.BF16_BYTES
        if lds_bytes > LDS_CAP_BYTES:
            raise ValueError(
                f"Q + dO staging needs {lds_bytes} B of LDS, over the {LDS_CAP_BYTES} B cap, for "
                f"block_dmodel {self.block_dmodel} at block_q {self.block_q} with "
                f"{self.num_stream_buffers} buffers. Lower block_q, or stage the D axis (B3)."
            )
        return replace(self, traits=traits)


# Defaults the policy has no shape-dependent opinion about. `waves_per_eu` and
# `daz` are the forward's, so a default build runs under the same scheduling
# and denormal regime the forward was measured on.
_FALLBACK = BwdDkDvKnobs(
    num_stream_buffers=2,
    waves_per_eu=2,
    daz=True,
    strides_constexpr=False,
)


def bwd_dkdv_knobs(arch="gfx950", **overrides):
    """Knobs for `arch`, with `overrides` pinned. Mirrors `fmha_knobs`."""
    if arch != "gfx950":
        raise ValueError(f"this tuning module is gfx950-only, got {arch!r}")
    return BwdDkDvKnobs(**overrides)
