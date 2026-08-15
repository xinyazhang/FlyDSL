# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tuning policy for the gfx1201 dK/dV backward kernel: which knobs, which shape.

Same split as `fmha_tuning_gfx1201.py` and for the same reason -- the kernel
file is about correctness and this one is about speed, and a number here moves
when a sweep says so. It imports nothing from `flydsl`, so it is trivially
stable-only and importable from anywhere.

**Only `block_m` has been swept.** `num_waves` and the floating-point knobs are
one default each, carried over from the forward or picked once; treat them as
placeholders. `_lds_bytes` / `_fits_lds` is not a placeholder -- it is a
legality calculation and must stay correct.

Geometry the kernel and this module must agree on
-------------------------------------------------

The dK/dV kernel is the transpose of the forward's loop: it holds a K/V tile
and streams Q/dO past it. That inverts which tensor is register-resident and
which transits LDS, and it fixes the wave decomposition:

- **one wave owns 16 KV rows**, the WMMA M extent, so `BLOCK_N = 16 *
  num_waves`. Waves own *disjoint* KV rows, which is what keeps dK/dV
  accumulators wave-private and removes any cross-wave reduction. The
  alternative -- waves splitting the Q stream over a shared KV tile -- needs a
  `BLOCK_N x head_dim` f32 reduction through LDS per workgroup, which does not
  fit.
- **Q and dO are staged in LDS twice each**, row-major and transposed, because
  each is read once as a WMMA A operand along `d` (for S and dP) and once along
  `q` (for dK^T and dV^T). See the kernel's LDS-layout comment. That is what
  makes `_lds_bytes` the binding constraint on `block_m`.

Register floor, per wave, in VGPRs:

    k_packs + v_packs    (head_dim + head_dim_v) / 4
    dk + dv accumulators (head_dim + head_dim_v) / 2

i.e. **1.5 * head_dim** with head_dim_v == head_dim, before any transient. At
head_dim 128 that is 192 of the 256 available and the kernel spills; at 192 the
floor alone is 288 and it spills hard. Sharding `d` across waves is the fix and
is not implemented -- see the module docstring of the kernel.
"""

from dataclasses import dataclass, fields, replace

# One wave owns one WMMA M tile of KV rows. Not a knob: it is the instruction.
ROWS_PER_WAVE = 16

# Workgroup LDS budget on gfx1201. The hardware allows 64 KiB per workgroup.
_LDS_LIMIT = 65536

# LDS row padding, in elements. Same value and same reason as the forward's
# `_LDS_PAD`: it moves consecutive rows off the same bank group, and a swizzle
# was measured a net loss there.
LDS_PAD = 4

# The compiled tile widths, mirroring the forward's ladder so a backward build
# can be requested for any head_dim the forward accepts. Widths above 128 are
# *buildable but spill*; see the register floor in the module docstring.
_BLOCK_DMODEL_LADDER = (16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256)

MAX_HEAD_DIM = _BLOCK_DMODEL_LADDER[-1]

# Waves per workgroup; `BLOCK_N = 16 * this`.
#
# The knob that matters most at large head_dim, and the reason is traffic
# rather than registers: every workgroup streams the *whole* Q/dO sequence past
# its KV tile, so that traffic scales as `1/BLOCK_N`. Widening the workgroup
# quarters it. Per-wave register state is untouched -- each wave still owns 16
# KV rows and its own dK/dV accumulators -- so this is orthogonal to the spills
# at head_dim >= 192 and composes with anything that fixes them.
#
# Interleaved, three reps, median, at two shapes. Milliseconds, lower better;
# the previous default 4 against the two alternatives:
#
#              B=4 H=16 N=2048           B=1 H=8 N=4096
#   hd  mask   nw4     nw8    nw16     nw4     nw8    nw16
#   64  no     2.03    2.04    2.09    1.04    1.01    0.99
#   64  yes    1.15    1.24    1.26    0.65    0.71    0.78
#   80  no     2.90    2.54    2.71    1.47    1.24    1.24
#   80  yes    1.63    1.47    1.53    0.95    0.86    0.97
#   128 no     4.73    4.23    4.62    2.32    1.84    2.01
#   128 yes    2.32    2.31    2.57    1.25    1.21    1.25
#   160 no     6.69    6.65    6.17    3.47    3.21    2.98
#   192 no    10.05    9.54    8.83    4.95    4.72    4.55
#   224 no    17.22   16.92   13.24      --      --      --
#   256 no    28.28   21.18   19.08   14.09   10.85    9.53
#   256 yes   15.39   11.69   11.11    7.86    5.89    5.46
#
# Three bands, and the same three at both shapes, which is why this is a table
# and not a constant:
#
#   <= 64    4    widening loses; the tile is small enough that fewer, fatter
#                 workgroups just cost occupancy
#   80-128   8    1.05-1.26x
#   >= 160  16    1.06-1.48x, growing with head_dim exactly as the Q/dO
#                 traffic per KV row does
#
# Deliberately **not** keyed on `causal`. The two masking modes disagree only
# at 128 and 160, by at most 4% -- inside the board's own drift
# (`sdpa_lore_gfx1201.md`) -- and at 192 causal the wider setting is faster
# anyway. A second axis would be two more numbers to maintain for noise.
#
# The previous default of 4 everywhere came from one un-interleaved sweep at a
# third shape, whose comment said "re-sweep interleaved before moving it".
# This is that re-sweep.
# Reverse the KV-tile axis of the grid, the forward's `lpt_tile_order`.
#
# **The direction is inverted here, which is why this defaults to off.** The
# forward reverses because its cost *rises* with the tile index -- query tile i
# attends keys j <= i, so late tiles are the expensive ones, and the natural
# order dispatches the cheap ones first. Longest-processing-time scheduling
# wants the opposite.
#
# dK/dV is the transpose of that loop: key j is attended by queries i >= j, so
# a *low* KV tile is visited by nearly every Q block and a high one by few. The
# natural order already dispatches longest-first, and reversing it would be
# anti-LPT. Kept as a knob rather than not ported at all, because that
# reasoning is a prediction about a 64-CU dispatcher, and the measurement is
# cheap. Measured, interleaved x3, median, natural against reversed:
#
#                       head_dim 64      head_dim 128     head_dim 256
#   B=4 H=16 N=2048  full   0.99x         0.97x            1.00x
#                  causal   1.00x         0.94x            0.99x
#   B=1 H=8  N=4096  full   1.05x         1.00x            0.99x
#                  causal   0.96x         0.91x            0.92x
#
# The prediction holds, and the shape of the result is the confirmation: the
# loss is **causal-only** -- 0.91-0.96x across both shapes -- while non-causal
# is a wash, because there every KV tile has the same cost and the order cannot
# matter. Reversing turns longest-first into shortest-first exactly where the
# cost is skewed.
#
# Kept rather than deleted: the reversal is bitwise identical (it is a grid
# permutation, verified), so it is a free A/B for anything that changes the
# per-tile cost profile -- a fused dK/dV/dQ, or a window that makes the tail
# tiles expensive instead of cheap.
_DEFAULT_LPT_TILE_ORDER = False

_NUM_WAVES_SMALL_MAX_HEAD_DIM = 64
_NUM_WAVES_MEDIUM_MAX_HEAD_DIM = 128

# Head dims that stage 32 Q rows per pass rather than 16. Measured, and the
# effect is far outside the board's drift at the wide end.
#
# Two 16-row sub-tiles per staging pass amortise the pass's two barriers over
# twice the work, which is why 32 looks like the obvious default -- right up to
# the point where the extra live values spill. The per-wave register floor is
# `1.5 * head_dim` VGPRs before any transient, so that point arrives early.
#
# Measured B=2 H=12 N=4096 f16, best of two or three alternating reps
# (`bm32/bm16`, below 1.0 means 32 wins):
#
#   head_dim   non-causal   causal
#   16           0.930       0.903
#   32             --        0.937
#   48           1.022       1.032
#   64           0.989       0.947
#   80             --        1.300
#   96             --        1.378
#   128            --        1.528
#   160            --        2.211
#
# So the penalty grows monotonically from 80 up and there is no threshold to
# key on below it: **48 loses and 64 wins**, which is not an ordering any
# formula produces. 48 is the awkward width -- 6 threads per cooperative-load
# row against 64's 8 -- and the forward's tables record the same width class
# misbehaving (its head_dim 224, at 14 threads per row, spills 101 registers).
# Hence a set rather than a bound. 64's margin is 1-5% and 48's 2-3%, both
# close to the drift floor; re-measure interleaved before moving either.
#
# Note the sign is the *opposite* of the lesson the forward's tables record
# three times ("spill count is not a proxy for speed"). Here it is a good
# proxy, because what spills is an accumulator read and written by every WMMA
# rather than an operand preloaded once.
_BLOCK_M_32_HEAD_DIMS = frozenset({16, 32, 64})
_DEFAULT_BLOCK_M = 32


def _round_to_ladder(head_dim: int) -> int:
    """Smallest compiled tile width covering `head_dim`."""
    for w in _BLOCK_DMODEL_LADDER:
        if w >= head_dim:
            return w
    raise ValueError(f"head_dim {head_dim} exceeds the largest compiled tile ({MAX_HEAD_DIM})")


def lds_bytes(block_m: int, head_dim: int, head_dim_v: int, elem_bytes: int = 2) -> int:
    """LDS a Q/dO staging pass needs, in bytes.

    Four tiles, because both tensors are needed in both orientations:

        Q  row-major   block_m         x (head_dim   + pad)
        Q  transposed  head_dim        x (block_m    + pad)
        dO row-major   block_m         x (head_dim_v + pad)
        dO transposed  head_dim_v      x (block_m    + pad)

    The transposed copies are not an optimisation. A WMMA A operand's eight
    per-lane elements run along the *contraction* axis, and the four GEMMs in
    this kernel contract over `d` twice and over `q` twice, so each tensor is
    read both ways. Deriving one from the other in LDS costs eight strided
    scalar reads per operand -- the forward's `V_LDS_LAYOUT="row"` path, which
    it measured 2.7% slower for a single tensor.
    """
    rm = block_m * (head_dim + LDS_PAD) + block_m * (head_dim_v + LDS_PAD)
    tr = head_dim * (block_m + LDS_PAD) + head_dim_v * (block_m + LDS_PAD)
    return (rm + tr) * elem_bytes


def _fits_lds(block_m: int, head_dim: int, head_dim_v: int) -> bool:
    return lds_bytes(block_m, head_dim, head_dim_v) <= _LDS_LIMIT


def default_block_m(head_dim: int, head_dim_v: int) -> int:
    """Q rows per staging pass: the measured choice, reduced until it fits LDS.

    Walks *down* rather than failing, because block_m is a pure schedule
    parameter -- 16 is always legal (one WMMA M tile) and always correct.
    """
    want = _DEFAULT_BLOCK_M if max(head_dim, head_dim_v) in _BLOCK_M_32_HEAD_DIMS else 16
    for bm in (want, 16):
        if _fits_lds(bm, head_dim, head_dim_v):
            return bm
    return 16


def default_num_waves(head_dim: int, head_dim_v: int) -> int:
    """Waves per workgroup, hence `BLOCK_N = 16 * this`.

    Independent of LDS: the Q/dO tiles are shared by every wave, so widening
    the workgroup costs registers (each wave's own K/V and dK/dV) rather than
    LDS. See the table above for the three bands and the measurements.
    """
    wide = max(head_dim, head_dim_v)
    if wide <= _NUM_WAVES_SMALL_MAX_HEAD_DIM:
        return 4
    if wide <= _NUM_WAVES_MEDIUM_MAX_HEAD_DIM:
        return 8
    return 16


@dataclass(frozen=True)
class BwdDkDvMetadata:
    """What to compute. Set by the caller; never by policy."""

    num_heads: int
    head_dim: int
    causal: bool = False
    dtype_str: str = "f16"
    head_dim_v: int | None = None
    sm_scale: float | None = None
    causal_type: int | None = None
    dropout: bool = False
    philox_width: int | None = None


@dataclass(frozen=True)
class BwdDkDvKnobs:
    """How to compute it. Every field `None` means "policy decides"."""

    # Compile-time widths baked into the binary; the real extents ride along as
    # runtime `hdim_qk` / `hdim_vo` arguments and `padded_head` records whether
    # they differ.
    block_dmodel: int | None = None
    block_dmodel_v: int | None = None

    block_m: int | None = None
    num_waves: int | None = None
    lpt_tile_order: bool | None = None

    # Function attributes and floating-point latitude. Same three-level split
    # as the forward's: `fp_mode` is the explicit flag set on the softmax-ish
    # arithmetic, `fast_fp_math` the ambient default, `unsafe_fp_math` a
    # whole-compilation backend option.
    waves_per_eu: int | None = None
    sched_strategy: str | None = None
    fp_mode: str | None = None
    denormals_are_zero: bool | None = None
    unsafe_fp_math: bool | None = None
    fast_fp_math: bool | None = None

    # Whether the Q/dO address hoists `row * stride_seq` out of the loop. See
    # `kv_off` in `fmha_common_gfx1201.make_addr_pair`.
    addr_hoist: bool | None = None

    # Derived, not chosen: true when the caller's head_dim is not itself a
    # compiled tile width.
    padded_head: bool | None = None

    def merge(self, other: "BwdDkDvKnobs | None") -> "BwdDkDvKnobs":
        """`other`'s set fields win; its `None`s leave this one's alone."""
        if other is None:
            return self
        set_fields = {f.name: getattr(other, f.name) for f in fields(other) if getattr(other, f.name) is not None}
        return replace(self, **set_fields)


_KNOBS_FALLBACK = BwdDkDvKnobs(
    lpt_tile_order=_DEFAULT_LPT_TILE_ORDER,
    # "noninf" and not "fast": `ninf` lets the compiler assume no operand is
    # infinite, and this kernel reads a logsumexp that is deliberately +inf for
    # a row with no live keys. The forward records the same hazard deleting its
    # KV tail mask.
    fp_mode="noninf",
    denormals_are_zero=True,
    unsafe_fp_math=True,
    fast_fp_math=True,
    # The loop body holds four WMMA chains and two LDS staging passes; the
    # default GCN scheduler sinks each LDS load next to its consumer. Same
    # reasoning as the forward's causal builds, untested here.
    sched_strategy="max-memory-clause",
    waves_per_eu=None,
    addr_hoist=False,
    padded_head=False,
)


def resolve_knobs(meta: BwdDkDvMetadata, overrides: "BwdDkDvKnobs | None" = None) -> BwdDkDvKnobs:
    """The complete configuration for `meta`.

    `overrides` is applied first, so a pinned knob participates in deriving the
    ones downstream of it -- pinning `num_waves` changes `BLOCK_N` and hence
    nothing else here, but pinning `block_dmodel` changes `block_m`.
    """
    s = _KNOBS_FALLBACK.merge(overrides)
    if s.block_dmodel is None:
        s = replace(s, block_dmodel=meta.head_dim)
    if s.block_dmodel_v is None:
        s = replace(s, block_dmodel_v=meta.head_dim_v if meta.head_dim_v is not None else s.block_dmodel)
    hd, hdv = s.block_dmodel, s.block_dmodel_v
    if s.num_waves is None:
        s = replace(s, num_waves=default_num_waves(hd, hdv))
    if s.block_m is None:
        s = replace(s, block_m=default_block_m(hd, hdv))
    if not _fits_lds(s.block_m, hd, hdv):
        raise ValueError(
            f"block_m={s.block_m} with head_dim=({hd}, {hdv}) needs "
            f"{lds_bytes(s.block_m, hd, hdv)} B of LDS, over the {_LDS_LIMIT} B cap"
        )
    return s


@dataclass(frozen=True)
class BwdDkDvPlan:
    """Everything a host needs for one build, from one call."""

    meta: BwdDkDvMetadata
    knobs: BwdDkDvKnobs


def plan(request: BwdDkDvMetadata, overrides: BwdDkDvKnobs | None = None) -> BwdDkDvPlan:
    """The one entry point into tuning: caller's inputs in, a full plan out.

    `request` comes back as `BwdDkDvPlan.meta` unchanged. The rounding up to a
    compiled tile lands in `knobs.block_dmodel`, and `knobs.padded_head`
    records whether the two differ -- exactly as `fmha_tuning_gfx1201.plan`
    does, so the two passes round a given head_dim the same way.
    """
    head_dim = request.head_dim
    if head_dim < 1 or head_dim > MAX_HEAD_DIM:
        raise ValueError(f"kernel requires 1 <= head_dim <= {MAX_HEAD_DIM}, got {head_dim}")
    head_dim_v = request.head_dim_v if request.head_dim_v is not None else head_dim
    block_dmodel = _round_to_ladder(head_dim)
    block_dmodel_v = _round_to_ladder(head_dim_v)
    knobs = replace(
        resolve_knobs(
            request,
            (overrides or BwdDkDvKnobs()).merge(BwdDkDvKnobs(block_dmodel=block_dmodel, block_dmodel_v=block_dmodel_v)),
        ),
        padded_head=(block_dmodel != head_dim) or (block_dmodel_v != head_dim_v),
    )
    return BwdDkDvPlan(request, knobs)
