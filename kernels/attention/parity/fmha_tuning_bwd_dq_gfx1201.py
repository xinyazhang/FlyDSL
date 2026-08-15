# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tuning policy for the gfx1201 backward-dQ kernel: which knobs, for which shape.

Split out of ``fmha_bwd_dq_gfx1201_kernel.py`` the way
``fmha_tuning_gfx1201.py`` is split out of the forward kernel, and for the same
reason: the kernel file is about correctness and this one is about speed. They
change on different evidence -- a number here moves when a sweep says so, and
nothing here can make a build *wrong*, only slow.

It imports nothing from ``flydsl``, so it is trivially stable-only and can be
read (or edited) without a GPU in the room. The forward tuning module has the
same property; keep it.

**Lightly tuned.** Two knobs were swept once, at B=1 H=8 N=4096 f16, both
masking modes, on a single un-interleaved run: the wave count and
``kt_lds_layout``. Those two tables are measurements. Everything else --
``block_n``, ``kv_addr_hoist``, ``sched_strategy`` -- is an unswept default and
says so at its definition. Treat those as hypotheses.

One caveat on every number here: they come from one run each, and
``sdpa_lore_gfx1201.md`` records that this board drifts about 5%, so
alternatives closer than that are not separated. The two effects below are much
larger than 5% where they matter at all.
"""

from dataclasses import dataclass, fields, replace

# ---------------------------------------------------------------------------
# The compiled tile widths.
#
# Shorter than the forward's ladder, and the missing top is not an oversight.
# A dQ wave carries, per lane:
#
#   q packs    head_dim / 4  VGPRs   (BLOCK_DMODEL/16 operands of v8f16)
#   dO packs   head_dim / 4  VGPRs
#   dq accs    head_dim / 2  VGPRs   (BLOCK_DMODEL/16 accumulators of v8f32)
#
# i.e. head_dim VGPRs before the S and dP accumulators (32 more at BLOCK_N 32),
# the addressing, and the LDS staging registers. 256 is therefore already at
# the 256-VGPR wall and 384/512 cannot be expressed without either head-dim
# sharding or a D-column window, neither of which this kernel implements yet.
# The forward escapes this because it carries only *one* head_dim-proportional
# register set (O) plus Q; dQ carries three.
# ---------------------------------------------------------------------------

_BLOCK_DMODEL_LADDER = (16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256)

MAX_HEAD_DIM = _BLOCK_DMODEL_LADDER[-1]


def _round_to_ladder(head_dim: int) -> int:
    """Smallest compiled tile width covering `head_dim`."""
    for w in _BLOCK_DMODEL_LADDER:
        if w >= head_dim:
            return w
    raise ValueError(f"head_dim {head_dim} exceeds the largest compiled tile ({MAX_HEAD_DIM})")


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

# Q rows a single wave owns. Fixed by the WMMA tile, not a choice: one 16x16
# WMMA per (Q row block, KV column block), and this kernel does not implement
# the forward's ROW_SUBTILES knob.
_ROWS_PER_WAVE = 16

# Waves per workgroup, i.e. BLOCK_M / 16.
#
# The trade is *not* the forward's. Here a wave's register cost does not depend
# on the wave count at all -- every wave owns 16 Q rows whatever BLOCK_M is --
# so more waves is purely: fewer workgroups, hence less K/V global traffic and
# fewer LDS stagings per KV tile, against more waves sharing one workgroup's
# LDS bandwidth and a coarser Q tail. The measured curve is accordingly much
# flatter than the forward's, with two cliffs.
#
# TFLOPS, B=1 H=8 N=4096 f16, non-causal / causal:
#
#   head_dim    w2           w4           w8           w16          chosen
#   16       51.3/34.5    48.6/38.9    53.9/45.5    54.0/51.2       16
#   32       62.7/62.4    70.6/60.7    75.5/78.5    70.2/66.8        8
#   48          --        79.6/71.7    82.9/83.3    77.2/65.8        8
#   64       75.6/74.5    85.6/89.9    89.8/88.0    83.2/74.3        8
#   80          --        86.1/80.5    92.8/86.5    94.4/86.4        8
#   96          --        86.6/83.9    90.2/87.4    85.6/78.3        8
#   128      35.8/39.2    93.9/90.8    99.5/93.2    94.5/83.7        8
#   160         --        50.0/52.2    90.5/84.1    87.3/76.8        8
#   192      41.4/42.0    75.3/81.8    26.7/23.1    92.8/83.6       16
#   224         --        46.4/44.4    26.0/20.8    91.2/44.8       16
#   256      13.9/15.2    39.9/40.6    65.0/58.8    72.0/60.9       16
#
# The two collapses -- 128 at w2, and 192/224 at w8 -- are three to four times
# slower than their neighbours, far outside the board's drift, and they are the
# reason this is a table rather than a constant. They are almost certainly
# spills (the 192/224 pair spills at 8 waves and *recovers* at 16, which is the
# same non-monotone shape the forward's `_KV_ADDR_HOIST_HEAD_DIMS` comment
# describes: whether a hoist or a wider tile survives is decided by the
# register allocator one width at a time). Not confirmed against an ISA dump.
#
# 80 is a coin toss (w16 is +1.7% non-causal and -0.1% causal, inside the
# drift) and stays at the default rather than gaining an entry for noise.
_NUM_WAVES_BY_HEAD_DIM: dict[int, int] = {16: 16, 192: 16, 224: 16, 256: 16}
_DEFAULT_NUM_WAVES = 8

# KV columns per tile.
#
# Pinned at 32 and the kernel asserts it. BLOCK_N is the width of the S and dP
# accumulators, which are *dead* register pressure in this kernel -- unlike the
# forward, where a wider tile amortises the per-tile softmax. Widening it would
# add 2 * BLOCK_N/16 * 8 VGPRs against a head_dim-proportional budget that is
# already the binding constraint. Revisit only at head_dim <= 32.
_DEFAULT_BLOCK_N = 32

# How K reaches GEMM3 (`dq += K^T @ dS^T`).
#
#   "scalar"      read K^T out of the row-major K tile with 8 strided 16-bit
#                 LDS loads per operand. No extra LDS, no extra global traffic,
#                 head_dim scalar LDS reads per KV tile per wave.
#   "transposed"  stage a second, transposed copy K^T[d][kv] with
#                 `global_load_tr_b128`, exactly as the forward stages V^T, and
#                 read one vector per operand. Costs a second full K global
#                 load and `head_dim * (BLOCK_N + 4)` more elements of LDS.
#
# The forward measured the equivalent choice for V at +2.7% at N >= 4096, on a
# loop that reads V *once* per operand, and the expectation here was a larger
# win because GEMM3 does 8 scalar reads per operand. **The opposite happened.**
# TFLOPS, B=1 H=8 N=4096 f16, scalar -> transposed:
#
#   head_dim   non-causal        causal        ratio (nc / c)
#   64        86.7 -> 87.7    90.6 -> 88.4     1.01 / 0.98
#   128       93.5 -> 56.4    91.0 -> 55.9     0.60 / 0.61
#   256       39.9 -> 23.3    40.4 -> 20.5     0.59 / 0.51
#
# So the transposed arm is a wash at 64 and loses 40-50% from 128 up. The
# forward's V^T pays for itself because V is loaded once either way; here K^T
# is a **second** full K tile -- a second global load and
# `head_dim * (BLOCK_N + 4)` more LDS -- to serve a GEMM that already had the
# data resident. The LDS reads it saves are cheaper than the traffic it adds,
# and the gap grows with head_dim exactly as the extra tile does.
#
# Kept as a knob rather than deleted: the two arms are bitwise identical, so
# it is a free A/B for anything that changes the LDS or traffic budget (a KV
# prefetch, a wider BLOCK_N, a fused dK/dV that has K^T resident anyway).
_DEFAULT_KT_LDS_LAYOUT = "scalar"

# Distance-1 K/V prefetch, as the forward's `k_prefetch_dist`/`v_prefetch_dist`.
#
# 0 until measured. The loop today issues both global reads inside the barrier
# pair and waits on them immediately: at head_dim 128 the ISA puts 0..4
# instructions and **zero** WMMA between each `global_load` and the
# `s_wait_loadcnt` that consumes it, where the forward gets 35-36 instructions
# and 7 WMMA, plus two loads it does not wait on at all. So the latency is
# fully exposed here and there are three GEMMs per tile to hide it behind --
# one more than the forward has.
#
# The carry is cheap: 1-2 batches each for K and V at every head_dim on the
# ladder, so 8-16 VGPRs. dQ measures 150/221/231/256 VGPRs at head_dim
# 64/128/192/256, so it fits everywhere except 256, which already spills 132.
_DEFAULT_KV_PREFETCH_DIST = 0


def default_num_waves(head_dim: int) -> int:
    """Waves per workgroup at this head_dim."""
    return _NUM_WAVES_BY_HEAD_DIM.get(head_dim, _DEFAULT_NUM_WAVES)


def default_block_m(head_dim: int) -> int:
    """Q rows per workgroup at this head_dim."""
    return _ROWS_PER_WAVE * default_num_waves(head_dim)


def default_block_n(head_dim: int, causal: bool) -> int:
    """KV columns per tile at this head_dim."""
    return _DEFAULT_BLOCK_N


# Whether the KV address hoists `row * stride_seq` out of the loop; see
# `kv_off` in `fmha_common_gfx1201.make_addr_pair` for the two forms.
#
# The forward keys this off head_dim from a measured table. Nothing here is
# measured, and the two kernels have different loop bodies, so copying its
# table would be borrowing a conclusion rather than a fact. Off everywhere
# until swept.
_KV_ADDR_HOIST_HEAD_DIMS: frozenset[int] = frozenset()


def _kv_addr_hoist(head_dim: int, causal: bool) -> bool:
    return head_dim in _KV_ADDR_HOIST_HEAD_DIMS


# ---------------------------------------------------------------------------
# The two halves of a build request. Same split as the forward's, for the same
# reason: a caller states a problem, the tuning policy answers with a schedule.
# Both frozen so the pair can be an `lru_cache` key.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BwdDqInputMetadata:
    """What to compute. Set by the caller; never by policy."""

    num_heads: int
    head_dim: int
    causal: bool = True
    dtype_str: str = "bf16"
    sm_scale: float | None = None
    causal_type: int | None = None
    dropout: bool = False
    philox_width: int | None = None


@dataclass(frozen=True)
class BwdDqKnobs:
    """How to compute it. Every field `None` means "policy decides"."""

    # Compile-time width baked into the binary. The *real* extent travels as a
    # runtime argument and may be smaller, which is what `padded_head` records.
    block_dmodel: int | None = None

    block_m: int | None = None
    block_n: int | None = None
    num_waves: int | None = None
    kt_lds_layout: str | None = None
    kv_addr_hoist: bool | None = None
    kv_prefetch_dist: int | None = None

    waves_per_eu: int | None = None
    flat_work_group_size: int | None = None
    sched_strategy: str | None = None

    # Three floating-point knobs acting at three levels; see the forward tuning
    # module's note. Only `fp_mode` is ever varied in practice.
    #
    # "noninf" is load-bearing here for the same reason it is there and one
    # more: this kernel writes -inf into the scores of masked columns *and*
    # reads a +inf logsumexp for rows the forward found no keys for. `ninf`
    # licenses the compiler to assume neither exists.
    fp_mode: str | None = None
    denormals_are_zero: bool | None = None
    unsafe_fp_math: bool | None = None
    fast_fp_math: bool | None = None

    padded_head: bool | None = None
    lpt_tile_order: bool | None = None

    def merge(self, other: "BwdDqKnobs | None") -> "BwdDqKnobs":
        """`other`'s set fields win; its `None`s leave this one's alone."""
        if other is None:
            return self
        set_fields = {f.name: getattr(other, f.name) for f in fields(other) if getattr(other, f.name) is not None}
        return replace(self, **set_fields)


_KNOBS_FALLBACK = BwdDqKnobs(
    kt_lds_layout=_DEFAULT_KT_LDS_LAYOUT,
    kv_prefetch_dist=_DEFAULT_KV_PREFETCH_DIST,
    waves_per_eu=2,
    fp_mode="noninf",
    denormals_are_zero=True,
    unsafe_fp_math=True,
    fast_fp_math=True,
    padded_head=False,
    lpt_tile_order=True,
)


def resolve_knobs(meta: BwdDqInputMetadata, overrides: "BwdDqKnobs | None" = None) -> BwdDqKnobs:
    """The complete configuration for `meta`.

    `overrides` is applied *first*, so a pinned knob participates in deriving
    the ones downstream of it rather than being stamped on afterwards --
    pinning `num_waves` therefore also moves `block_m`.
    """
    s = _KNOBS_FALLBACK.merge(overrides)
    if s.block_dmodel is None:
        s = replace(s, block_dmodel=meta.head_dim)
    hd = s.block_dmodel

    if s.num_waves is None:
        s = replace(s, num_waves=default_num_waves(hd))
    if s.block_m is None:
        s = replace(s, block_m=_ROWS_PER_WAVE * s.num_waves)
    if s.block_n is None:
        s = replace(s, block_n=default_block_n(hd, meta.causal))
    if s.kv_addr_hoist is None:
        s = replace(s, kv_addr_hoist=_kv_addr_hoist(hd, meta.causal))
    if s.flat_work_group_size is None:
        s = replace(s, flat_work_group_size=32 * s.num_waves)
    # `block_m` and `num_waves` state one fact twice, so they must agree. A
    # caller who pins only `block_m` gets the wave count that goes with it.
    if s.block_m != _ROWS_PER_WAVE * s.num_waves:
        if overrides is not None and overrides.num_waves is None:
            s = replace(s, num_waves=s.block_m // _ROWS_PER_WAVE, flat_work_group_size=2 * s.block_m)
        else:
            raise ValueError(
                f"block_m ({s.block_m}) must be {_ROWS_PER_WAVE} * num_waves " f"({s.num_waves}); one wave owns 16 rows"
            )
    return s


@dataclass(frozen=True)
class BwdDqPlan:
    """Everything a host needs for one build, from one call."""

    meta: BwdDqInputMetadata
    knobs: BwdDqKnobs


def plan(request: BwdDqInputMetadata, overrides: BwdDqKnobs | None = None) -> BwdDqPlan:
    """The one entry point into tuning: the caller's inputs in, a full plan out.

    `request` is returned as `BwdDqPlan.meta` **unchanged**. The rounding up to
    a compiled tile lands in `BwdDqPlan.knobs.block_dmodel`, and
    `knobs.padded_head` records whether the two differ.
    """
    head_dim = request.head_dim
    if head_dim < 1 or head_dim > MAX_HEAD_DIM:
        raise ValueError(f"kernel requires 1 <= head_dim <= {MAX_HEAD_DIM}, got {head_dim}")
    block_dmodel = _round_to_ladder(head_dim)
    knobs = replace(
        resolve_knobs(request, (overrides or BwdDqKnobs()).merge(BwdDqKnobs(block_dmodel=block_dmodel))),
        padded_head=block_dmodel != head_dim,
    )
    return BwdDqPlan(request, knobs)
