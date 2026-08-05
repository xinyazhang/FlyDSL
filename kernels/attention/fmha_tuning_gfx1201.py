# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tuning policy for the gfx1201 attention kernel: which knobs, for which shape.

Split out of `flash_attn_func_gfx1201_aiw.py` so that the kernel file is about
correctness and this one is about speed. They change for different reasons and
on different evidence -- a number here moves when a sweep says so, and nothing
here can make the kernel *wrong*, only slow.

**Why a third module rather than folding this into the interface**, which is
where `sdpa-readability.md` asked for it: the kernel needs these values too. It
resolves `k_prefetch_dist=None` and friends into defaults, and the geometry
helpers below are called with values -- `VO_WIDTH`, `BLOCK_N` -- that only
exist part-way through the builder. Moving the policy into the interface, which
already imports the kernel, would be either a circular import or a second copy
of that derivation. A module importing neither is importable by both.

Two kinds of thing live here, and the distinction is worth keeping when
deciding whether a change needs a benchmark or a correctness argument:

- **Policy** -- `default_prefetch_dist`, `qk_shards`, `default_block_m`,
  `default_block_n`, and the tables they read. Measured *choices*. Any of them
  could be changed without making a build incorrect.
- **Geometry** -- `q_tiles_per_block`, `vo_chunks`, `resolve_shards`. These
  compute what is *legal* given a choice, out of LDS capacity and divisibility
  constraints. Changing one can make a build invalid, and they are here only
  because the policy functions call them.
"""

# ---------------------------------------------------------------------------
# Tuning policy
#
# These tables come from measured sweeps, not from a formula; see the comments
# on each. They are the *default* knob settings for a given head_dim -- every
# one can be overridden per build.
# ---------------------------------------------------------------------------

# The binding-prefetch schedule wins from head_dim 48 up; 16 and 32 still prefer
# distance 0 (35.4/60.7 with prefetch against 37.3/61.2 without). Measured
# B=1 H=8 N=4096 f16 non-causal.
_PREFETCH_MIN_HEAD_DIM = 48

_TARGET_WAVES = 8
_ROWS_PER_Q_TILE = 16

# Measured (shards, q_tiles) per head_dim. There is no clean formula: more
# waves helps while registers and LDS allow, and hurts the moment it pushes
# either over. Both effects are only visible after compiling, so these come
# from a sweep (B=1 H=8 N=4096 f16 non-causal, TFLOPS at 8 vs 16 waves):
#
#   hdim   8 waves  16 waves           chosen         why not more
#    48      79.5     76.3             8 waves
#    64      84.3     90.1            16 waves
#    80      91.4     95.7            16 waves
#    96      98.4    100.3            16 waves
#   128      97.5    102.0            16 waves
#   160      92.9     94.2            16 waves
#   192      99.5     90.6             8 waves        16 spills 8 registers
#   224      62.2     79.7 (2 shards) 16 waves        1 shard spills at any count
#   256      81.9      rej             8 waves        reduction buffer over LDS
#   384      53.9     63.8 (12 waves) 12 waves        16 waves rejected
#   512      53.3      rej             8 waves        reduction buffer over LDS
#
# The 16-wave rejections are LDS: the cross-shard reduction buffer scales with
# NUM_WAVES, and past 8 waves it no longer fits inside the V window it aliases.
#
# head_dim 256 re-swept after the softmax correction and LSE moved the budget.
# The old entry (2 shards, 4 q_tiles -> BLOCK_M 64, 8 waves) came from a sweep
# whose note reads "16 waves rejected: reduction buffer over LDS" -- true only
# *with* sharding, since that buffer exists only when QK_SHARDS > 1. Unsharded
# 16 waves was therefore never tried, and it is much better:
#
#   (shards, q_tiles)   BLOCK_M  waves | non-causal  causal
#   (2, 4)  <- old         64       8  |    74.1      71.8
#   (1, 16) <- new        256      16  |    92.6      74.9
#
# i.e. 1.25x non-causal and 1.05x causal, despite spilling 3 registers where
# the old config spilled none (241 -> 256 VGPRs). BLOCK_M 64 -> 256 quarters
# the workgroup count and the K/V traffic with it, which outweighs the spills
# by a wide margin -- a reminder that spill count is not a proxy for speed.
#
# 384 and 512 were swept the same way and are already at their optimum
# (384: 61.4/60.9 at (3,4), best alternative 54.2; 512: 52.2/44.6 at (4,2),
# best alternative 36.7). Only 256 was mistuned.
#
# Re-swept in full after gSWA and varlen, both of which moved register
# pressure. Nine of eleven entries were confirmed unchanged. Two were not, and
# **both were mistuned the same way head_dim 256 had been**: a configuration
# rejected on spills or LDS was never retried after the surrounding budget
# changed, so the table kept a choice whose reason had expired.
#
#   head_dim  was      now      non-causal  causal   (interleaved, 5 reps)
#   224       (2, 8)   (1, 16)     1.269     1.230
#   384       (3, 4)   (2, 4)      1.089     1.062
#
# head_dim 224's old note reads "1 shard spills at any count" -- true, and
# irrelevant: unsharded 16 waves is 27% faster despite the spills. That is the
# third time this file has recorded that spill count is not a proxy for speed,
# and the second time the lesson was written down and then not applied to a
# neighbouring entry.
#
# head_dim 160 screened as a win for (1, 8) and is *not* one: interleaved it
# is 0.948 non-causal and 1.000 causal. Kept at (1, 16). Recorded because the
# screen and the confirmation disagreed by more than the effect being measured
# -- a single undrifted measurement cannot resolve a 1% difference on this
# board, and two of the three candidates it produced were real.
#
# head_dim 128 re-swept after the P2 region split, which duplicates the loop
# body and pushed its VGPR count 149 -> 212. 16 q_tiles (BLOCK_M 256, 16 waves)
# no longer fits; 8 is 92.5/92.0 against 84.4/81.2 TFLOPS non-causal/causal.
# 48/80/96/160 were swept at the same time and are unchanged; 192 is already
# at its optimum.
_SHARDS_BY_HEAD_DIM = {224: 1, 256: 1, 384: 2}
_Q_TILES_BY_HEAD_DIM = {48: 8, 64: 16, 80: 16, 96: 16, 128: 8, 160: 16,
                        192: 8, 224: 16, 256: 16, 384: 4, 512: 2}

# BLOCK_M for the distance-0 schedule. Per-wave register use is dominated by
# two head_dim-proportional terms -- o_accs = VO_WIDTH/2 VGPRs and
# q_b_packs = head_dim/4 -- neither of which depends on BLOCK_M or BLOCK_N, so
# tile size is a weak lever on spilling. It is not a null one, though, because
# it changes the cooperative-load geometry. Measured spills / TFLOPS at
# B=1 H=8 N=4096 f16 non-causal:
#
#   head_dim  BM=128        BM=64        BM=32
#   160       0sp / 80.8    - / 74.4     - / 48.7
#   192      24sp / 67.2    - / 59.0     - / 37.9
#   224     101sp / 33.5   64sp / 50.9   38sp / compile-fail
#   256      36sp / 67.2   20sp / 46.9   53sp / 19.9
#
# BLOCK_M=128 wins everywhere except head_dim=224, whose awkward
# THREADS_PER_ROW_LOAD=14 spills 101 registers and loses ~40%.
_DIST0_BLOCK_M_BY_HEAD_DIM = {224: 64}

# Small head_dim is softmax-bound, not saturation-bound: the per-(row, KV tile)
# softmax cost does not scale with head_dim, so at head_dim 16 a wave does only
# 4 WMMA against 17 v_exp_f32 plus 2 barriers. A wider KV tile amortises the
# per-tile part of that -- the correction exp, the m/l update, the O rescale and
# the barriers -- across more KV columns. Measured B=1 H=8 N=4096 f16
# non-causal:
#
#   head_dim   BN=32   BN=64   BN=128
#   16          37.4    44.6    48.2
#   32          61.4    72.5    70.7
#
# Causal is excluded: its mask is unrolled into 16 explicitly named scalars, so
# it requires NUM_S_VALS == 16, i.e. BLOCK_N == 32. Widening it would mean
# rewriting that unroll (planned for the interval-decomposition work).
_DIST0_BLOCK_N_BY_HEAD_DIM_NONCAUSAL = {16: 128, 32: 64}


def default_prefetch_dist(head_dim):
    """K/V prefetch distance for this head_dim."""
    return 1 if head_dim >= _PREFETCH_MIN_HEAD_DIM else 0


def qk_shards(head_dim):
    """Waves cooperating on one Q row-tile at this head_dim."""
    return _SHARDS_BY_HEAD_DIM.get(head_dim, max(1, head_dim // 128))


def q_tiles_per_block(head_dim, shards=None):
    """Q row-tiles per workgroup: TARGET_WAVES traded against the shard count.

    The V transpose tiling does not have to divide evenly across the waves --
    tail tiles are guarded at the LDS store -- so this is otherwise free.
    """
    shards = qk_shards(head_dim) if shards is None else shards
    return _Q_TILES_BY_HEAD_DIM.get(head_dim, max(1, _TARGET_WAVES // shards))


def vo_chunks(vo_width, block_n, shards, pad=4):
    """V staging passes needed to keep the *padded* K+V tile inside 64 KiB.

    Sharding V/O across waves means every wave's slice is live at once, so the
    full width of V^T would have to be resident -- 69888 B at 512 columns, over
    the cap. Staging V in `nc` passes makes only vo_width/nc columns resident,
    which restores the padding and with it conflict-free LDS. Costs one extra
    barrier pair per extra pass. Returns 1 whenever one pass fits.
    """
    for nc in (1, 2, 4, 8):
        if vo_width % nc:
            continue
        cols = vo_width // nc
        if cols % (shards * 16):        # each wave needs whole 16-col chunks
            continue
        if block_n * (vo_width + pad) * 2 + cols * (block_n + pad) * 2 <= 65536:
            return nc
    return 1


def resolve_shards(head_dim, vo_width, block_n, want=None):
    """Largest valid shard count no greater than the policy's preference.

    The policy table keys off head_dim alone, but the shard count also has to
    divide the *V/O window* into whole 16-column chunks. Those two constraints
    only diverge when the window is narrower than head_dim: head_dim 384 wants
    3 shards, which splits a 128-wide window into 42-column slices and is
    rejected downstream. Walk down from the preference to the first count that
    satisfies both, rather than failing the build.
    """
    want = qk_shards(head_dim) if want is None else want
    for s in range(want, 0, -1):
        if head_dim % s or (head_dim // s) % 16:
            continue
        cols = vo_width // vo_chunks(vo_width, block_n, s)
        if cols % s or (cols // s) % 16:
            continue
        return s
    return 1


def default_block_m(head_dim, prefetch_dist=None):
    """BLOCK_M for this head_dim under the default schedule."""
    dist = default_prefetch_dist(head_dim) if prefetch_dist is None else prefetch_dist
    if dist == 0:
        return _DIST0_BLOCK_M_BY_HEAD_DIM.get(head_dim, 128)
    return _ROWS_PER_Q_TILE * q_tiles_per_block(head_dim)


def default_block_n(head_dim, causal, prefetch_dist=None):
    """BLOCK_N for this head_dim under the default schedule."""
    dist = default_prefetch_dist(head_dim) if prefetch_dist is None else prefetch_dist
    if dist == 0 and not causal:
        return _DIST0_BLOCK_N_BY_HEAD_DIM_NONCAUSAL.get(head_dim, 32)
    return 32
