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
this module reuses them rather than forking them, so three of their names mean
the other thing here:

| traits field | forward | dK/dV |
|---|---|---|
| `BLOCK_M` | Q rows per workgroup | **KV rows per workgroup** |
| `BLOCK_N` | KV rows per streamed tile | **Q rows per streamed tile** |
| `ROWS_PER_WAVE` | Q rows per wave | **KV rows per wave** (always 32) |
| `VO_SHARDS` | waves splitting O's D axis | **waves splitting dK/dV's D axis** |

`BwdDkDvTraits` re-exposes them under `BLOCK_KV`, `BLOCK_Q` and `DKV_SHARDS`,
and the kernel body uses only those. Everything derived from `BLOCK_N` --
`SMEM_N_RPT`, the DMA issue count, the LDS line layout -- is about *a tile of
64 rows staged through LDS* and is correct unchanged, because the staging does
not care whose rows they are. Reusing `VO_SHARDS` rather than adding a field is
deliberate: `make_traits` already derives and *validates* the shard split
(`D_CHUNKS_PER_STAGE_SHARD`, the even-chunk rule the LDS offset decomposition
needs), and a second field would be a second derivation to keep in step.

--- What this phase is, and is not -----------------------------------------

B3: the full ladder `(32 … 512)`, the 8xD input contract, padded heads. Still
dense, non-causal, MHA, bf16. Causal and windows (B4), varlen (B5) and
dropout/bias (B6) are later; the `const_expr` hooks are left where they go and
`resolve` raises rather than silently serving a configuration the kernel does
not compute.

--- The two walls, and which one binds -------------------------------------

Per staged slot at `BLOCK_Q` 64: `68 * head_dim` elements, i.e. `136 * d`
bytes. Two tensors and `NUM_STREAM_BUFFERS` buffers, against a 163840 B cap.
**LDS never binds below 512 and binds only the second buffer at 512**, which is
why this kernel needs no `D_STAGES` at all -- see `_with_buffers`.

Registers do bind. A wave holds `d/4` VGPRs of resident K, `d/4` of V, and
**two** accumulators of `d/2` each: `1.5 * d` before a single transient. The
levers, in the addendum's order, are `(num_waves, waves_per_eu)` -- which is
free and decided head_dim 128 -- and then `DKV_SHARDS`, which divides the
accumulator pair by the shard count at the cost of every shard recomputing S
and dP. `_GEOMETRY` is the table that resolves all three, and the measurements
behind it are recorded there.
"""

from dataclasses import dataclass, fields, replace

from fmha_traits_gfx950 import ParityDualwaveTraits, make_traits

__all__ = [
    "LADDER",
    "BwdDkDvInputMetadata",
    "BwdDkDvKnobs",
    "BwdDkDvTraits",
    "bwd_dkdv_knobs",
    "rung_below",
    "tile_width_for",
]

# The forward's ladder, entire. A head_dim between two rungs is served by the
# next rung up with the real extent passed at runtime and the surplus D columns
# masked, which is what `padded_head` records.
LADDER = (32, 64, 96, 128, 160, 192, 224, 256, 384, 512)

# The streamed tile is always 64 q rows: `ds_read_b64_tr_b16` covers 16 rows per
# k-substep and the MFMA takes four of them, so 64 is what one pass of the
# transposed operand spans. A 128-row tile would need eight substeps -- legal,
# and a second geometry to validate; not built.
DEFAULT_BLOCK_Q = 64

# The addressable LDS cap. Measured elsewhere, not inferred: the compiler
# reports "local memory (N) exceeds limit (163840)" and says nothing about
# which knob to move, which is why `_with_buffers` decides this here.
LDS_CAP_BYTES = 163840

# Per-wave register demand, in VGPRs, as a function of the tile width and the
# shard count. Not a hardware limit -- **it is demand, and B1 measured demand
# and grant differing by 1.8x** -- but it is what orders the search.
#
#     resident K + V          d/4 + d/4  = d/2      (loop-invariant)
#     dK acc + dV acc         d/2 + d/2  = d        (loop-invariant, / shards)
#
# so `d/2 + d/shards` before a single transient: 384 at head_dim 256 unsharded,
# and 384 again at 512 with four shards.
#
# The transients (one Q pack, the two score accumulators, P and dS, one
# transposed chunk) add roughly 100 more and do not scale with the tile width
# the same way, because the d-contraction reads are interleaved with their
# MFMAs rather than gathered first.


# (num_waves, waves_per_eu, dkv_shards) by tile width. `BLOCK_KV` follows as
# `32 * num_waves / shards` and is *not* stored beside them; see
# `_with_geometry`.
#
# **The AGPR cliff is the first lever and it decided head_dim 128**, which
# carries two accumulators where the forward carries one. At 8 waves (2 per
# SIMD) a wave may address 256 registers *in total*, so the allocator cannot
# reach the AGPR file at all and spills to scratch instead:
#
#   head_dim  waves  waves_per_eu  AGPR  spills   TFLOP/s
#     128       8         2          0    118       444
#     128       4         2          0    126       403
#     128       4         1          -      -       721
#     128       2         2        108      0       788
#
# Note the third row: at 4 waves the only thing between 403 and 721 TF is the
# occupancy *hint*. Sweep the pair, never the wave count alone. A build at 0
# AGPRs with a nonzero spill count is not short of registers, it is forbidden
# from using half of them.
#
# **The second lever is `BLOCK_KV`, and B3 found it matters more than the
# first above head_dim 128.** Every workgroup streams the whole of Q and dO
# for its head, so the read traffic is `seqlen / BLOCK_KV` copies of that slab
# -- and `BLOCK_KV` is `32 * waves / shards`, so *raising* the wave count is
# free traffic relief whenever the registers still fit at one wave per SIMD:
#
#   head_dim  waves  BLOCK_KV   TFLOP/s
#     160       2       64        408
#     160       4      128        690      <- same registers per wave, half the traffic
#     224       2       64        486
#     224       4      128        723
#
# That is why every rung but 128 lands on 4 waves. 8 crosses back over the
# cliff (2 waves per SIMD) and is worse everywhere it was tried: 256 at 8
# waves is 198 TF against 451, 384 is 91 against 283, 512 is 81 against 260.
#
# **`dkv_shards` is last and only 256 upward need it**, because it buys
# accumulator space by making every shard recompute S and dP -- 1.5x the MFMA
# work at 2 shards, 2.5x at 4 -- *and* it divides `BLOCK_KV`, so it pays the
# traffic lever back. Measured at `B=4 H=8 S=4096` bf16 non-causal on nominal
# FLOPs (`8 * B * H * Sq * Sk * d`, so the duplicated work is *not* credited):
#
#   head_dim  waves  wpe  shards  tight  AGPR  spills   TFLOP/s
#      32       4     2     1      yes      0      0       508
#      64       4     1     1      no       0      0       713
#      96       4     2     1      no       0      4       805
#     128       2     2     1      no      74      0       744
#     160       4     2     1      no     100      0       751
#     192       4     1     1      no     232      0       636
#     224       4     1     1      no     230      0       737
#     256       4     1     2      no     168      0       451
#     384       4     1     2      yes    256     31       283
#     512       4     1     4      yes    256     38       260
#
# Every build here compiles in under two seconds, far inside plan B3's
# eight-minute cap; no configuration in the sweep ever approached it.
# Whether the tile body is written for minimum live registers rather than
# maximum instruction-level parallelism; see `_with_register_pressure`. The
# crossover is where the two accumulators stop leaving room for the transients,
# and below it the tight arm gives up real overlap for a saving nothing needed:
#
#   head_dim   loose            tight
#      32      472 TF, 0 sp     511 TF, 0 sp
#      64      712 TF, 0 sp     703 TF, 0 sp
#     128      744 TF, 0 sp     606 TF, 0 sp
#     224      723 TF, 0 sp     645 TF, 0 sp
#     256      148 TF, 246 sp   375 TF, 166 sp   (at shards 1; both want a shard)
#     384      197 TF, 368 sp   283 TF,  31 sp
#     512      213 TF, 294 sp   261 TF,  38 sp
#
# head_dim 32 is the odd one at the narrow end and it is not a register story:
# with `D_CHUNKS == 1` there is barely any independent work for the loose arm
# to overlap, so all it does is lengthen the live ranges.
_TIGHT_REGISTERS = {
    32: True,
    64: False,
    96: False,
    128: False,
    160: False,
    192: False,
    224: False,
    256: False,
    384: True,
    512: True,
}

_GEOMETRY = {
    32: (4, 2, 1),
    64: (4, 1, 1),
    96: (4, 2, 1),
    128: (2, 2, 1),
    160: (4, 2, 1),
    192: (4, 1, 1),
    224: (4, 1, 1),
    256: (4, 1, 2),
    384: (4, 1, 2),
    512: (4, 1, 4),
}


def _granule_for(block_dmodel):
    """The D-axis staging granule for a tile width.

    One DMA issue moves 512 bf16 per wave, so the granule fixes how many tokens
    that covers and therefore how many LDS lines a tile occupies. 64 is family
    A's; a width off the 64 grid needs 32, which is the forward's family S and
    is validated by `tooling/lds_model.py` against the same read helpers this
    kernel uses. The floor is the PV MFMA's 32-column output, so 16 is not
    available at any width.
    """
    return 64 if block_dmodel % 64 == 0 else 32


def rung_below(block_dmodel):
    """The widest rung strictly narrower than `block_dmodel`, or 0.

    `tile_width_for` rounds *up*, so a build it chose serves exactly the
    half-open range `(rung_below(R), R]`. That lower bound is what lets the
    kernel skip masking the D columns it knows are real -- see `HDIM_QK_FLOOR`
    in the kernel. A property of the ladder rather than of any build, so adding
    a rung silently tightens every wider build's floor, which is correct.
    """
    below = [r for r in LADDER if r < block_dmodel]
    return max(below) if below else 0


def tile_width_for(head_dim):
    """The compiled tile width serving `head_dim`, or raise saying why not."""
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    for rung in LADDER:
        if head_dim <= rung:
            return rung
    raise ValueError(f"head_dim {head_dim} exceeds the widest tile ({max(LADDER)})")


@dataclass(frozen=True)
class BwdDkDvTraits(ParityDualwaveTraits):
    """The forward's traits, read with the backward's role names.

    A subclass with no new *derived* state: every field is the parent's, and
    the properties below are aliases that make the kernel body readable rather
    than new numbers. That is deliberate -- the moment the LDS layout here
    stops being the forward's, the transpose read (which is the forward's V
    read, unmodified) stops being validated.
    """

    # How many stream buffers the body cycles. 2 prefetches one tile ahead: the
    # DMA for tile `j+2` is issued once every wave has finished reading tile
    # `j`, so it overlaps tile `j+1`'s compute. 1 is the same body with the
    # prefetch distance collapsed to zero, which is what head_dim 384 and 512
    # buy their LDS with. See `_with_buffers`.
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
    def DKV_SHARDS(self):
        """Waves splitting the D axis of the dK and dV accumulators.

        `VO_SHARDS` under its forward name, and the same bargain: D is an
        *output* axis for both accumulators, so shards write disjoint columns
        and **never have to agree on anything** -- no cross-wave reduction, no
        extra barrier, no summation order to get wrong. The price is that every
        shard recomputes the whole of S and dP, which is half the arithmetic.
        """
        return self.VO_SHARDS

    @property
    def KV_BLOCKS_PER_WG(self):
        """Distinct 32-row KV blocks in one workgroup: `num_waves / shards`."""
        return self.Q_TILES

    @property
    def D_CHUNKS_PER_SHARD(self):
        """Output chunks one shard owns. Even, which the LDS offset needs."""
        return self.D_CHUNKS_PER_STAGE_SHARD

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
    head_dim_v: int | None = None
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
    block_dmodel_v: int | None = None
    padded_head: bool | None = None
    hdim_qk_floor: int | None = None

    # Pinned as a set of four; see `_with_geometry`.
    num_waves: int | None = None
    block_kv: int | None = None
    block_q: int | None = None
    head_dim_granule: int | None = None

    # Per-width policy a sweep wants to vary on its own, so not part of the
    # four-field set above.
    dkv_shards: int | None = None
    num_stream_buffers: int | None = None
    waves_per_eu: int | None = None
    tight_registers: bool | None = None

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
        return (
            _FALLBACK.merge(self)
            ._checked_scope(meta)
            ._with_widths(meta)
            ._with_geometry()
            ._with_buffers()
            ._with_register_pressure()
            ._with_traits(meta)
        )

    def _checked_scope(self, meta):
        """Refuse what this phase does not compute, at the decision rather than at an address.

        Every one of these would otherwise produce a plausible, finite, wrong
        gradient: a dropped causal mask returns dense attention's gradient, and
        a GQA call returns one q head's contribution where the sum over the
        group belongs. Both are the right shape.
        """
        for name in ("causal", "window", "dropout", "bias"):
            if getattr(meta, name):
                raise NotImplementedError(
                    f"{name}=True is not implemented. B1-B3 are dense, non-causal, MHA, bf16; see "
                    "sdpa-bwd-plan-gfx950.md phases B4-B6."
                )
        if meta.dtype_str != "bf16":
            raise NotImplementedError(f"this kernel builds bf16 only, got {meta.dtype_str!r}")
        num_kv_heads = meta.num_heads if meta.num_kv_heads is None else meta.num_kv_heads
        if num_kv_heads != meta.num_heads:
            raise NotImplementedError(
                f"GQA (num_heads {meta.num_heads}, num_kv_heads {num_kv_heads}) needs dK/dV summed "
                "over every q head sharing a kv head, which this kernel does not do: one workgroup "
                "owns one (q head, kv block) and would write, not accumulate. MHA only."
            )
        return self

    def _with_widths(self, meta):
        """Decide `block_dmodel`, `block_dmodel_v`, `padded_head` and the mask floor."""
        block_dmodel = self.block_dmodel
        derived = block_dmodel is None
        if derived:
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

        # Only a *derived* tile carries the ladder's guarantee. A caller that
        # pins `block_dmodel` may pin 256 for head_dim 64, so there is no floor
        # to claim and the kernel masks every column.
        hdim_qk_floor = self.hdim_qk_floor
        if hdim_qk_floor is None:
            hdim_qk_floor = rung_below(block_dmodel) if derived else 0
        if not 0 <= hdim_qk_floor < block_dmodel:
            raise ValueError(f"hdim_qk_floor {hdim_qk_floor} must be in [0, block_dmodel={block_dmodel})")
        if meta.head_dim <= hdim_qk_floor:
            raise ValueError(
                f"head_dim {meta.head_dim} is at or below this build's hdim_qk_floor {hdim_qk_floor}; "
                f"the {block_dmodel}-wide tile only serves ({hdim_qk_floor}, {block_dmodel}]"
            )
        return replace(
            self,
            block_dmodel=block_dmodel,
            block_dmodel_v=block_dmodel_v,
            padded_head=bool(padded_head),
            hdim_qk_floor=int(hdim_qk_floor),
        )

    def _with_geometry(self):
        """Decide waves, `BLOCK_KV`, the granule and the shard count from the tile width.

        `_GEOMETRY` is the whole policy and its table is where the measurement
        lives. **`BLOCK_KV` is derived rather than pinned beside the wave
        count** -- a wave owns exactly the MFMA's 32-row M extent and shards
        split the waves, not the rows, so `BLOCK_KV = 32 * num_waves / shards`
        and the two cannot disagree. That is the failure P7 found twelve
        instances of on the forward, where the invariant was prose.

        Reads `self.block_dmodel`, so it runs after `_with_widths`.
        """
        shards = self.dkv_shards
        pinned = (self.num_waves, self.block_kv, self.block_q, self.head_dim_granule)
        if all(x is not None for x in pinned):
            if shards is None:
                raise ValueError("a pinned geometry must pin dkv_shards too; it decides BLOCK_KV")
            return self
        if any(x is not None for x in pinned):
            raise ValueError(
                f"pin num_waves, block_kv, block_q and head_dim_granule together or not at all, got {pinned}"
            )
        num_waves, waves_per_eu, table_shards = _GEOMETRY[self.block_dmodel]
        if shards is None:
            shards = table_shards
        return replace(
            self,
            num_waves=num_waves,
            waves_per_eu=self.waves_per_eu if self.waves_per_eu is not None else waves_per_eu,
            dkv_shards=shards,
            block_kv=32 * (num_waves // shards),
            block_q=DEFAULT_BLOCK_Q,
            head_dim_granule=_granule_for(self.block_dmodel),
        )

    def _with_buffers(self):
        """Two stream buffers if LDS allows, one if it does not.

        **The second buffer is the only thing LDS ever costs this kernel**, and
        it is worth stating why there is no `D_STAGES` here at all. A staged
        slot is `68 * head_dim` elements, so two tensors double-buffered are
        `544 * head_dim` bytes: 139264 at head_dim 256 and 278528 at 512
        against a 163840 cap. Single-buffered they are `272 * head_dim`, which
        is 139264 at 512 -- so head_dim 512 fits *with a whole tile of each
        tensor resident*, and the D axis never has to be split in time.

        What is lost at one buffer is the prefetch distance: the DMA for the
        next tile is issued at the end of this one and waited on immediately.
        The body is otherwise identical, which is why this is a number rather
        than a second code path.
        """
        if self.num_stream_buffers is not None:
            return self
        per_slot = self._stream_slot_bytes()
        return replace(self, num_stream_buffers=2 if 4 * per_slot <= LDS_CAP_BYTES else 1)

    def _stream_slot_bytes(self):
        """Bytes one staged tile occupies, without building the traits.

        Duplicates two lines of `make_traits`' derivation, and does so
        knowingly: `_with_buffers` has to decide the buffer count *before* the
        traits exist, and the alternative is building throwaway traits to ask
        them. Guarded -- `_with_traits` asserts the two agree.
        """
        granule = self.head_dim_granule
        smem_n_rpt = self.block_q // (512 // granule)
        line = 512 + 32
        return smem_n_rpt * (self.block_dmodel // granule) * line * 2

    def _with_register_pressure(self):
        """Trade instruction-level parallelism for live registers, or not.

        One knob for two linked choices in the tile body -- whether a staged
        half is read pack-by-pack into its MFMA or all at once, and whether the
        two halves go through the softmax together. Both hold roughly the same
        thing live and both cost the same thing: the scheduler's freedom to
        overlap an MFMA burst with the loads and VALU of independent work.
        `_TIGHT_REGISTERS` records where they cross.
        """
        if self.tight_registers is not None:
            return self
        return replace(self, tight_registers=_TIGHT_REGISTERS[self.block_dmodel])

    def _with_traits(self, meta):
        """Build the traits this configuration implies.

        `make_traits` is the forward's, called with `block_m=block_kv`,
        `block_n=block_q` and `vo_shards=dkv_shards`. Those three slots change
        meaning; nothing else does. Using its `vo_shards` rather than a field
        of our own is what gets the shard validation -- the even-chunk rule the
        LDS offset decomposition depends on -- for free.
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
            vo_shards=self.dkv_shards,
            causal=False,
            dtype_str=meta.dtype_str,
            waves_per_eu=self.waves_per_eu,
            daz=self.daz,
        )
        traits = _as_bwd_traits(base, NUM_STREAM_BUFFERS=self.num_stream_buffers)

        # **P7's ceiling, enforced rather than commented.** A wave holds
        # exactly the MFMA's 32-row M extent in KV rows: more would address
        # rows the accumulator does not have, fewer would run a full 32-row
        # MFMA and store rows the workgroup does not own. `make_traits` rejects
        # the first; this rejects the second, and together they make the
        # invariant checkable instead of stated. The forward shipped twelve
        # silently-wrong configurations for want of exactly this.
        if traits.ROWS_PER_WAVE != 32:
            raise ValueError(
                f"block_kv {self.block_kv} over {self.num_waves} waves at {self.dkv_shards} shards gives "
                f"{traits.ROWS_PER_WAVE} KV rows per wave, and this kernel serves exactly 32. It is "
                f"`v_mfma_f32_32x32x16`'s N extent, and everything downstream is written against it: the "
                f"v16f32 accumulator pair, the `8*(r//4) + 4*(lane//32) + (r%4)` row map the LSE and delta "
                f"reads are grouped by, the `_pack_p_v8_slices` operand packing, and the `permlane32_swap` "
                f"store. **16 rows per wave is the accumulator lever plan section 3 names and it is not "
                f"built** -- it needs `v_mfma_f32_16x16x16`, whose v4f32 accumulator is a second operand "
                f"algebra and a re-derived transpose-read lane map, not a parameter. "
                f"Pass block_kv={32 * (self.num_waves // self.dkv_shards)}."
            )
        if traits.STREAM_TILE_ELEMS * traits.BF16_BYTES != self._stream_slot_bytes():
            raise AssertionError(
                f"`_stream_slot_bytes` says {self._stream_slot_bytes()} B per slot but the traits derive "
                f"{traits.STREAM_TILE_ELEMS * traits.BF16_BYTES}; the buffer-count decision was made "
                "against a stale copy of the LDS derivation"
            )
        lds_bytes = traits.LDS_STREAM_TOTAL_SIZE * traits.BF16_BYTES
        if lds_bytes > LDS_CAP_BYTES:
            raise ValueError(
                f"Q + dO staging needs {lds_bytes} B of LDS, over the {LDS_CAP_BYTES} B cap, for "
                f"block_dmodel {self.block_dmodel} at block_q {self.block_q} with "
                f"{self.num_stream_buffers} buffers. Drop to one buffer, or lower block_q."
            )
        return replace(self, traits=traits)


# Defaults the policy has no shape-dependent opinion about. `daz` is the
# forward's, so a default build runs under the same denormal regime it was
# measured on. `waves_per_eu` is deliberately absent: it is per-width and
# `_with_geometry` supplies it.
_FALLBACK = BwdDkDvKnobs(
    daz=True,
    strides_constexpr=False,
)


def bwd_dkdv_knobs(arch="gfx950", **overrides):
    """Knobs for `arch`, with `overrides` pinned. Mirrors `fmha_knobs`."""
    if arch != "gfx950":
        raise ValueError(f"this tuning module is gfx950-only, got {arch!r}")
    return BwdDkDvKnobs(**overrides)
