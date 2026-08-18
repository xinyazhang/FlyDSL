# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Parity-side dualwave traits, with the tile geometry as parameters.

`_make_dualwave_swp_traits` hardcodes `num_waves=8`, `block_m=256`,
`block_n=64` and derives the D-axis staging granule from a fixed 128-byte row.
Those four numbers are the entire difference between the tile families (see
`Gfx950Knobs._with_wave_geometry`), so a constructor that fixes them can serve
exactly one. This is that constructor with them opened up.

--- How this is kept honest -----------------------------------------------

Every derivation here is a transcription of the production one, and
`assert_matches_production` checks that claim the only way that means anything:
by building both for a range of head_dims and comparing **every field**. That
test is what makes this file safe to put on the live path -- family A goes
through it today, so the existing bitwise-against-production test covers it.

**What that check cannot tell you** is whether a *parameterization* is right,
only whether it reproduces family A. Several constants have more than one
plausible formula that coincide at granule 64, and the ones below are flagged
`UNVERIFIED`. They are the first suspects if a new family produces plausible
but wrong numbers, and none of them is exercised until one is built.
"""

from dataclasses import dataclass, fields

from gfx950_standalone import dualwave

__all__ = ["make_traits", "assert_matches_production", "ParityDualwaveTraits"]


@dataclass(frozen=True)
class ParityDualwaveTraits(dualwave.DualwaveSwpTraits):
    """Production's traits plus the two axes that let a tile exceed head_dim 256.

    A subclass rather than new fields on `DualwaveSwpTraits`, because that
    dataclass lives in `flash_attn_utils.py` and is shared by four production
    kernels. Subclassing is free here: the parent is `frozen=True` with no
    defaulted fields, so added fields need defaults and those defaults are
    exactly the "behave like today" values. `assert_matches_production`
    iterates the *parent's* fields, so it keeps pinning family A to production
    without knowing these exist.

    Both default to 1, and **1 must mean the kernel is unchanged** -- every
    construct they gate sits behind `const_expr(... > 1)` so a default build
    traces to identical IR. That is what makes the four working rungs safe.

    - `D_STAGES` -- how many passes the KV tile's D axis is staged through LDS
      in. LDS is `BLOCK_N * head_dim * ~8.3 B` and the cap is 163840, so
      head_dim 384 (204288 B) and 512 (272384 B) do not fit in one pass. It
      also bounds the K/V register window, since only one stage is live.
    - `VO_SHARDS` -- how many waves split the *output* D axis of one Q tile.
      Wave *s* accumulates only `O[:, slice_s]`, so O drops by the shard count.
      Every wave still computes the whole S, which is why this needs **no
      cross-wave reduction at all** -- the shards never have to agree on
      anything, they just write disjoint columns. The price is that QK is
      recomputed per shard.
    - `QK_SHARDS` -- additionally splits the *reduction* D axis, so wave *s*
      holds only `Q[:, slice_s]` and computes a partial S that must then be
      summed across shards through LDS. Strictly more powerful than
      `VO_SHARDS` (it is the only thing that shrinks Q) and strictly more
      machinery. Not implemented yet; `VO_SHARDS` alone is what makes head_dim
      512 fit, because O is 256 VGPRs against Q's 128.

    The two are separate fields rather than one number because they buy
    different things at different prices, and conflating them would hide that
    the cheap one is sufficient today.
    """

    D_STAGES: int = 1
    QK_SHARDS: int = 1
    VO_SHARDS: int = 1
    STAGE_DIM: int = 0  # head_dim // D_STAGES; 0 means "unset", fixed up below
    K_STEPS_PER_STAGE: int = 0
    D_CHUNKS_PER_STAGE: int = 0
    D_CHUNKS_PER_STAGE_SHARD: int = 0
    Q_TILES: int = 0  # NUM_WAVES // VO_SHARDS


# Hardware constants. Not parameters: these are the wave, the DMA width and the
# MFMA shape, and a build that changed one would not be this algorithm.
WARP_SIZE = 64
DMA_BYTES = 16
BF16_BYTES = 2
VEC_KV = 8  # bf16 elements one lane moves per DMA issue (16 B)
MFMA_LANE_K = 8
K_STEP_QK = 16  # MFMA K extent
D_CHUNK = 32  # PV MFMA N extent -- the O accumulator's width
PV_K_STEP = 16
K_SUB_N = 32


def make_traits(
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    num_waves,
    block_m,
    block_n,
    granule,
    d_stages=1,
    qk_shards=1,
    vo_shards=1,
    causal=True,
    dtype_str="bf16",
    waves_per_eu=2,
    daz=True,
    lazy_rescale=True,
    setprio=True,
    stagger=True,
    debug_lazy_counts=False,
    num_kv_splits=1,
    varlen=False,
    cross_seqlen=False,
    paged=False,
    kv_cache_layout="linear",
    kv_vectorized=None,
    return_lse=False,
    xcd_swizzle=False,
):
    """Dualwave traits for an arbitrary (waves, BLOCK_M, BLOCK_N, granule)."""
    if num_waves % vo_shards:
        raise ValueError(f"vo_shards {vo_shards} must divide num_waves {num_waves}")
    # With `vo_shards` waves sharing one Q tile, the workgroup covers
    # `num_waves // vo_shards` tiles, not `num_waves`. Rows per wave is *not*
    # what shrinks -- it is pinned at 32 by the MFMA's M extent -- so BLOCK_M
    # falls instead, and what each wave saves is D columns of O.
    q_tiles = num_waves // vo_shards
    if block_m % q_tiles:
        raise ValueError(f"BLOCK_M {block_m} does not divide across {q_tiles} Q tiles")
    if head_dim % granule:
        raise ValueError(f"head_dim {head_dim} is not a multiple of the granule {granule}")
    if head_dim % D_CHUNK:
        raise ValueError(f"head_dim {head_dim} is not a multiple of the PV MFMA width {D_CHUNK}")

    block_size = num_waves * WARP_SIZE
    rows_per_wave = block_m // q_tiles

    k_steps_qk = head_dim // K_STEP_QK
    d_chunks = head_dim // D_CHUNK
    pv_k_steps = K_SUB_N // PV_K_STEP

    # The D axis is cut two independent ways, and they compose: `d_stages`
    # splits it *in time* (one LDS residency per pass) and `qk_shards` splits
    # it *across waves* (one Q/O slice per wave). Validated together here so an
    # illegal pair fails at the decision rather than at an address.
    if d_stages < 1 or head_dim % d_stages:
        raise ValueError(f"head_dim {head_dim} is not a multiple of d_stages {d_stages}")
    stage_dim = head_dim // d_stages
    if stage_dim % granule:
        raise ValueError(f"stage extent {stage_dim} (head_dim/{d_stages}) is not a multiple of granule {granule}")
    if k_steps_qk % d_stages or d_chunks % d_stages:
        raise ValueError(
            f"d_stages {d_stages} must divide both K_STEPS_QK {k_steps_qk} and D_CHUNKS {d_chunks}; "
            "a stage that splits an MFMA step has no meaning"
        )
    if vo_shards < 1 or d_chunks % (d_stages * vo_shards):
        raise ValueError(
            f"vo_shards {vo_shards} x d_stages {d_stages} must divide D_CHUNKS {d_chunks}; "
            "each (stage, shard) owns a whole number of 32-column PV output chunks"
        )
    d_chunks_per_stage_shard = d_chunks // (d_stages * vo_shards)
    if d_chunks_per_stage_shard % 2:
        raise ValueError(
            f"D_CHUNKS per (stage, shard) is {d_chunks_per_stage_shard}, which must be even: the shard's "
            "LDS offset is folded into `urv_base`, and `_swizzled_v_dc_off` only decomposes that way "
            "when the shard starts on an even chunk"
        )
    if qk_shards < 1 or head_dim % qk_shards:
        raise ValueError(f"head_dim {head_dim} is not a multiple of qk_shards {qk_shards}")
    if num_waves % qk_shards:
        raise ValueError(f"qk_shards {qk_shards} must divide num_waves {num_waves}")
    k_steps_per_stage = k_steps_qk // d_stages
    d_chunks_per_stage = d_chunks // d_stages

    gqa_group_size = num_heads // num_kv_heads
    default_stride_q_n = num_heads * head_dim
    default_stride_kv_n = num_kv_heads * head_dim

    # One DMA issue per wave moves `smem_linear_wave` elements; the granule
    # decides how that splits into (tokens, D).
    smem_linear_wave = WARP_SIZE * DMA_BYTES // BF16_BYTES
    smem_n_per_wave = smem_linear_wave // granule
    smem_n_rpt = block_n // smem_n_per_wave
    # `stage_dim`, not `head_dim`: this is the sole term through which
    # `d_stages` reaches LDS. Everything downstream (tile elems, buffer bases,
    # LDS_KV_TOTAL_SIZE) is derived from it, so one substitution sizes the
    # whole allocation to a single pass. At `d_stages == 1` it is `head_dim`
    # and every derived number is bit-for-bit what it was.
    smem_d_rpt = stage_dim // granule
    if smem_n_rpt == 0 or block_n % smem_n_per_wave:
        raise ValueError(f"BLOCK_N {block_n} is not a multiple of {smem_n_per_wave} tokens per DMA issue")
    if smem_n_rpt % num_waves:
        raise ValueError(f"{smem_n_rpt} KV tile lines do not divide across {num_waves} waves")

    smem_k_pad = DMA_BYTES // BF16_BYTES
    smem_v_pad = 64 // BF16_BYTES
    smem_k_line_stride = smem_linear_wave + smem_k_pad
    smem_v_line_stride = smem_linear_wave + smem_v_pad
    smem_k_tile_elems = smem_n_rpt * smem_d_rpt * smem_k_line_stride
    smem_v_tile_elems = smem_n_rpt * smem_d_rpt * smem_v_line_stride

    num_prefetch_k = 2
    kv_per_buffer = smem_k_tile_elems + smem_v_tile_elems
    lds_kv_total_size = num_prefetch_k * kv_per_buffer

    # K LDS->VGPR. `n_strip_stride` is the offset from a lane's lo pack to its
    # hi pack: half a wave of lanes further on, each holding VEC_KV elements.
    k_lds_to_reg_n_strip_stride = (WARP_SIZE // 2) * VEC_KV
    k_lds_to_reg_kstep_inner_stride = K_STEP_QK
    k_lds_to_reg_kstep_outer_stride = smem_n_rpt * smem_k_line_stride

    # V LDS->VGPR. UNVERIFIED parameterizations, all exact at granule 64:
    #   half_wave  = (smem_n_rpt // 2) * line   -- could be a fixed 4 lines
    #   n_group    = granule // 4               -- could be 2 * VEC_KV
    #   k_substep  = 2 * granule                -- could be 4 * D_CHUNK
    #   dchunk_in_pair = D_CHUNK                -- could be granule // 2
    v_lds_to_reg_half_wave_stride = (smem_n_rpt // 2) * smem_v_line_stride
    v_lds_to_reg_lane_quad_stride = smem_v_line_stride
    v_lds_to_reg_n_group_stride = granule // 4
    v_lds_to_reg_lane_in_quad_stride = 4
    v_lds_to_reg_k_substep_stride = 2 * granule
    v_lds_to_reg_dchunk_pair_stride = smem_n_rpt * smem_v_line_stride
    v_lds_to_reg_dchunk_in_pair_stride = D_CHUNK
    v_lds_to_reg_transpose_pair_stride = granule

    kv_vec_size = DMA_BYTES // BF16_BYTES
    # Stored verbatim, `None` included -- the production constructor does the
    # same and the caller computes it. Defaulting it here would be more
    # defensive and would make this a *different* traits object, which the
    # field-by-field check would then have to be loosened to accept.

    return ParityDualwaveTraits(
        D_STAGES=d_stages,
        QK_SHARDS=qk_shards,
        VO_SHARDS=vo_shards,
        D_CHUNKS_PER_STAGE_SHARD=d_chunks_per_stage_shard,
        Q_TILES=q_tiles,
        STAGE_DIM=stage_dim,
        K_STEPS_PER_STAGE=k_steps_per_stage,
        D_CHUNKS_PER_STAGE=d_chunks_per_stage,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_N_OUT=block_n,
        K_SUB_N=K_SUB_N,
        WARP_SIZE=WARP_SIZE,
        NUM_WAVES=num_waves,
        BLOCK_SIZE=block_size,
        ROWS_PER_WAVE=rows_per_wave,
        HEAD_DIM=head_dim,
        K_STEP_QK=K_STEP_QK,
        K_STEPS_QK=k_steps_qk,
        D_CHUNK=D_CHUNK,
        D_CHUNKS=d_chunks,
        PV_K_STEP=PV_K_STEP,
        PV_K_STEPS=pv_k_steps,
        MFMA_LANE_K=MFMA_LANE_K,
        NUM_HEADS_Q=num_heads,
        NUM_HEADS_KV=num_kv_heads,
        GQA_GROUP_SIZE=gqa_group_size,
        CAUSAL=causal,
        DTYPE_STR=dtype_str,
        WAVES_PER_EU=waves_per_eu,
        DAZ=bool(daz),
        DUALWAVE_SWP_LAZY_RESCALE=bool(lazy_rescale),
        DUALWAVE_SWP_SETPRIO=bool(setprio),
        DUALWAVE_SWP_DEBUG_LAZY_COUNTS=bool(debug_lazy_counts),
        DUALWAVE_SWP_ENABLE_STAGGER=bool(stagger),
        NUM_KV_SPLITS=num_kv_splits,
        SPLITK=num_kv_splits > 1,
        PAGED=bool(paged),
        VARLEN=bool(varlen),
        CROSS_SEQLEN=bool(cross_seqlen),
        KV_CACHE_LAYOUT=kv_cache_layout,
        KV_VECTORIZED=kv_vectorized,
        DEFAULT_STRIDE_Q_N=default_stride_q_n,
        DEFAULT_STRIDE_KV_N=default_stride_kv_n,
        DMA_BYTES=DMA_BYTES,
        BF16_BYTES=BF16_BYTES,
        D_128B_SIZE=granule,
        VEC_KV=VEC_KV,
        SMEM_LINEAR_WAVE=smem_linear_wave,
        SMEM_N_PER_WAVE=smem_n_per_wave,
        SMEM_N_RPT=smem_n_rpt,
        SMEM_D_RPT=smem_d_rpt,
        SMEM_K_PAD=smem_k_pad,
        SMEM_V_PAD=smem_v_pad,
        SMEM_K_LINE_STRIDE=smem_k_line_stride,
        SMEM_V_LINE_STRIDE=smem_v_line_stride,
        SMEM_K_TILE_ELEMS=smem_k_tile_elems,
        SMEM_V_TILE_ELEMS=smem_v_tile_elems,
        NUM_PREFETCH_K=num_prefetch_k,
        DUALWAVE_SWP_KV_PER_BUFFER=kv_per_buffer,
        LDS_KV_TOTAL_SIZE=lds_kv_total_size,
        DUALWAVE_SWP_K_BUF_BASE=(0, kv_per_buffer),
        DUALWAVE_SWP_V_BUF_BASE=(smem_k_tile_elems, smem_k_tile_elems + kv_per_buffer),
        K_LDS_TO_REG_N_STRIP_STRIDE=k_lds_to_reg_n_strip_stride,
        K_LDS_TO_REG_KSTEP_INNER_STRIDE=k_lds_to_reg_kstep_inner_stride,
        K_LDS_TO_REG_KSTEP_OUTER_STRIDE=k_lds_to_reg_kstep_outer_stride,
        V_LDS_TO_REG_HALF_WAVE_STRIDE=v_lds_to_reg_half_wave_stride,
        V_LDS_TO_REG_LANE_QUAD_STRIDE=v_lds_to_reg_lane_quad_stride,
        V_LDS_TO_REG_N_GROUP_STRIDE=v_lds_to_reg_n_group_stride,
        V_LDS_TO_REG_LANE_IN_QUAD_STRIDE=v_lds_to_reg_lane_in_quad_stride,
        V_LDS_TO_REG_K_SUBSTEP_STRIDE=v_lds_to_reg_k_substep_stride,
        V_LDS_TO_REG_DCHUNK_PAIR_STRIDE=v_lds_to_reg_dchunk_pair_stride,
        V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE=v_lds_to_reg_dchunk_in_pair_stride,
        V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE=v_lds_to_reg_transpose_pair_stride,
        PAGED_BT_LDS_SIZE=2048,
        DUALWAVE_SWP_RESCALE_THRESHOLD=8.0,
        KV_VEC_SIZE=kv_vec_size,
        VEC_V_ROW_STRIDE=smem_v_line_stride,
        SCHED_MFMA_MASK=0x008,
        SCHED_VALU_MASK=0x002,
        SCHED_EXP_MASK=0x400,
        LDS_SCOPE_NAMES=("lds_k0", "lds_k1", "lds_v0", "lds_v1"),
        NEG_INF_F32_BITS=0xFF800000,
        LGKMCNT_0_ONLY=0xC07F,
        RETURN_LSE=bool(return_lse),
        XCD_SWIZZLE=bool(xcd_swizzle),
    )


def assert_matches_production(head_dims=(64, 128), **kwargs):
    """Every field must equal the production constructor's, at family A's geometry.

    The only check that makes "this is a transcription" mean anything. Called
    from the test suite across the built rungs and a spread of modes, because a
    single head_dim would not catch a term that happens to vanish there.
    """
    for head_dim in head_dims:
        mine = make_traits(
            head_dim=head_dim,
            num_waves=8,
            block_m=256,
            block_n=64,
            granule=64,
            **kwargs,
        )
        theirs = dualwave._make_dualwave_swp_traits(
            kwargs["num_heads"],
            kwargs["num_kv_heads"],
            head_dim,
            causal=kwargs.get("causal", True),
            dtype_str=kwargs.get("dtype_str", "bf16"),
            waves_per_eu=kwargs.get("waves_per_eu", 2),
            daz=kwargs.get("daz", True),
            dualwave_swp_lazy_rescale=kwargs.get("lazy_rescale", True),
            dualwave_swp_setprio=kwargs.get("setprio", True),
            dualwave_swp_debug_lazy_counts=kwargs.get("debug_lazy_counts", False),
            dualwave_swp_enable_stagger=kwargs.get("stagger", True),
            num_kv_splits=kwargs.get("num_kv_splits", 1),
            varlen=kwargs.get("varlen", False),
            cross_seqlen=kwargs.get("cross_seqlen", False),
            paged=kwargs.get("paged", False),
            kv_cache_layout=kwargs.get("kv_cache_layout", "linear"),
            kv_vectorized=kwargs.get("kv_vectorized"),
            return_lse=kwargs.get("return_lse", False),
        )
        diffs = [
            (f.name, getattr(mine, f.name), getattr(theirs, f.name))
            for f in fields(dualwave.DualwaveSwpTraits)
            if getattr(mine, f.name) != getattr(theirs, f.name)
        ]
        if diffs:
            raise AssertionError(f"head_dim {head_dim}: {len(diffs)} field(s) differ from production: {diffs}")
