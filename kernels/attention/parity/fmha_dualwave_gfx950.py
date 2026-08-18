# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Parity subclasses of the gfx950 dualwave helpers.

`kernels/attention/flash_attn_utils.py` is imported by four production kernels
(`flash_attn_generic`, `flash_attn_gfx950`, `flash_attn_fp8_gfx950`, and the
split-K combine), so it is **imported, never edited**. Everything this port
needs that differs from it lives here as a subclass.

--- What actually differs -------------------------------------------------

Less than the feature list suggests, because the production addressing is a
*special case* of the parity one rather than a different scheme. The dualwave
kernel addresses a flattened BSHD tensor:

    element(token, head, d) = token * stride_q_n + head * HEAD_DIM + d

which is the general BHSD-strided form with two slots pinned:

    element(b, h, s, d) = b * stride_0 + h * stride_1 + s * stride_2 + d
                                              ^^^^^^^^^^^^^^^^^^^^
                              stride_1 == HEAD_DIM, stride_2 == stride_q_n

So generalizing is a **change of variables, not new machinery**. Both `b` and
`h` are workgroup-uniform, so they fold into the buffer descriptor's base
address, which the production code already rebases per batch. What remains
per-access is `s * stride_2 + d`, which is the shape the existing helpers
already compute -- they just have to be handed `stride_2` where they currently
read `stride_q_n`, and a zero head offset where they currently add
`head * HEAD_DIM`.

That is why the overrides below are small and why none of them re-implements a
loop body. Three consequences worth naming, since each removes a whole class of
change:

- **`num_records` still bounds the sequence axis.** Rebasing at
  `(b, h)` and bounding at `seqlen * stride_2` elements makes an out-of-range
  row an out-of-buffer access, which returns zero in hardware rather than
  faulting. The production kernel relies on this for the ragged tail and it
  keeps working unmodified.
- **K and V get independent strides**, which the production code cannot express
  (it has one `stride_kv_n` for both). `load_k` and `load_v` are already
  separate methods that read `self.stride_kv_n_v`, so the subclass swaps the
  attribute and delegates rather than copying either body.
- **The head remap survives.** `q_head_idx = h_kv * gqa_group + group_id` is a
  permutation of head order for locality, not a correctness device. It is kept
  with runtime head counts so a parity build schedules its workgroups exactly
  as the measured baseline does.
"""

import contextlib
from dataclasses import replace

import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value
from fmha_common_gfx1201 import MaskedAxis
from gfx950_standalone import dualwave

__all__ = [
    "ParityGemmHelper",
    "ParityKernelContext",
    "ParityKvGmemToLdsLoader",
    "ParityKvLdsToVgprLoader",
    "ParityQLoader",
    "ParitySoftmaxHelper",
    "ParityStoreHelper",
]


# --- granule-general addressing --------------------------------------------
#
# Three production helpers fold constants that are only correct at granule 64.
# Each is replaced here by the same expression with the constant named, so at
# granule 64 they are the *same* arithmetic and a default build is unchanged --
# the bit-identity gate is what holds that claim.
#
# `_k_read_base` and `_ks_offset` are validated offline by
# `tooling/lds_model.py`, which reproduces family A exactly and confirms the
# granule-32 K read covers the tile once. The V pair below has no such model;
# it is measured instead.


def _anchor_v_o(traits, v_o):
    """`dualwave._anchor_v_o`, with the one-accumulator case spelled out.

    The production anchor asks for `!llvm.struct<(vector<16xf32>) x D_CHUNKS>`
    from an inline asm with `D_CHUNKS` outputs. At `D_CHUNKS == 1` LLVM rejects
    that outright -- *"inline asm with one output cannot return struct"* -- and
    the compiler aborts rather than diagnosing, so it surfaces as a crash.

    head_dim 32 is the first width to reach it: `D_CHUNKS = 32 / PV_MFMA_N = 1`.
    A single output returns the value's own type, so this is the same anchor
    with the struct wrapper dropped, not a weaker one.
    """
    if const_expr(traits.D_CHUNKS != 1):
        return dualwave._anchor_v_o(traits, v_o)
    acc = as_mlir_value(v_o[0])
    return [dualwave.llvm.inline_asm(acc.type, [acc], "", "=v,0", has_side_effects=True)]


def _k_read_base(traits, lane_mod_32, lane_div_32):
    """`_k_lds_read_base_per_lane` with `SMEM_N_RPT` in place of a literal 8."""
    return (
        (lane_mod_32 % traits.SMEM_N_RPT) * traits.SMEM_K_LINE_STRIDE
        + (lane_mod_32 // traits.SMEM_N_RPT) * traits.D_128B_SIZE
        + lane_div_32 * traits.VEC_KV
    )


def _ks_offset(traits, ks):
    """`_swizzled_ks_offset` with `K_STEPS_PER_BAND` in place of a literal 4."""
    per_band = traits.K_STEPS_PER_BAND
    return (ks // per_band) * traits.K_LDS_TO_REG_KSTEP_OUTER_STRIDE + (
        ks % per_band
    ) * traits.K_LDS_TO_REG_KSTEP_INNER_STRIDE


def _v_dc_offset(traits, dc):
    """`_swizzled_v_dc_off` with `D_CHUNKS_PER_BAND` in place of a literal 2."""
    per_band = traits.D_CHUNKS_PER_BAND
    return (dc // per_band) * traits.V_LDS_TO_REG_DCHUNK_PAIR_STRIDE + (
        dc % per_band
    ) * traits.V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE


def _v_imm_lo(traits, dc, k_substep):
    """`_swizzled_v_imm_lo`, in bytes, over the general dc offset."""
    return (k_substep * traits.V_LDS_TO_REG_K_SUBSTEP_STRIDE + _v_dc_offset(traits, dc)) * traits.BF16_BYTES


class ParityGemmHelper(dualwave.DualwaveGemmHelper):
    """The two GEMMs, addressed one D stage at a time.

    Under `D_STAGES > 1` a KV tile's D axis is covered in several passes, so
    neither GEMM sees all of it at once. The two are asymmetric about what that
    means, because D is a *reduction* axis for QK and an *output* axis for PV:

    - `qk_stage` accumulates into a running S that the caller carries across
      the stages, rather than seeding a fresh zero. All stages contribute to
      every element of S, and softmax cannot run until the last one has.
    - `pv_step_k` writes a disjoint slice of the O accumulator per stage, so
      the stages never meet; the stage only shifts which `v_o[dc]` is hit.

    Both take stage-relative register lists (that is what the loaders return)
    and map them to global indices here, so a stage index never has to be
    threaded into the loaders' addressing.
    """

    def qk_stage(self, v_k, q_all_scaled_bf16, acc, stage=0):
        k_lo, k_hi = v_k
        v_s_lo, v_s_hi = acc
        steps = self.traits.K_STEPS_PER_STAGE
        for ks in range_constexpr(steps):
            q_pack = dualwave._get_q_pack(self.traits, q_all_scaled_bf16, stage * steps + ks)
            v_s_lo = dualwave._mfma_acc(k_lo[ks], q_pack, v_s_lo, self.mma_atom, self.mfma_acc_vec_type)
            v_s_hi = dualwave._mfma_acc(k_hi[ks], q_pack, v_s_hi, self.mma_atom, self.mfma_acc_vec_type)
        return (v_s_lo, v_s_hi)

    def qk(self, v_k, q_all_scaled_bf16, stage=0):
        """Unstaged entry point: seed at zero and run the one stage there is."""
        if const_expr(self.traits.D_STAGES == 1):
            return super().qk(v_k, q_all_scaled_bf16)
        return self.qk_stage(v_k, q_all_scaled_bf16, (self.c_zero_v16f32, self.c_zero_v16f32), stage)

    def pv_step_k(self, step, v_p, v_v, v_o, stage=0):
        if const_expr(self.traits.D_STAGES == 1):
            return super().pv_step_k(step, v_p, v_v, v_o)
        v_p_lo, v_p_hi = v_p
        v_pk = v_v[step]
        p_pk = v_p_lo[step] if const_expr(step < 2) else v_p_hi[step - 2]
        per_stage = self.traits.D_CHUNKS_PER_STAGE
        for dc in range_constexpr(per_stage):
            out = stage * per_stage + dc
            v_o[out] = dualwave._mfma_acc(v_pk[dc], p_pk, v_o[out], self.mma_atom, self.mfma_acc_vec_type)
        return v_o

    def pv(self, v_p, v_v, v_o, stage=0):
        for step in range_constexpr(4):
            v_o = self.pv_step_k(step, v_p, v_v, v_o, stage=stage)
        return v_o


class _ParityKvStaging:
    """KV DMA addressing that allows more than one issue per wave.

    A mixin because **two objects need it**: the context builds the m0 tables
    in `init_dma_m0_tables`, and the loader computes source addresses in
    `_async_load_kv_linear`. The loader subclasses the *production* context
    rather than `ParityKernelContext`, so inheritance alone would not share
    them -- and a second copy is exactly how the write and read sides of an
    LDS layout drift apart.

    **What the production formula assumes.** It places one KV tile line per
    wave per d-band, `line = wave + d * SMEM_N_RPT`, which is correct exactly
    when `SMEM_N_RPT == NUM_WAVES`. Family A satisfies that by arithmetic
    coincidence: 8 waves, and BLOCK_N 64 at 8 tokens per issue is 8 lines. A
    4-wave family covering the same BLOCK_N needs **two issues per wave**, and
    under the production formula lines 4..7 are never written at all -- the
    reads then return whatever LDS happened to hold, which is how head_dim 192
    produced non-deterministic NaN.

    **The generalisation is a change of index, not of scheme.** The flat DMA
    index is band-major, `d_flat = band * ISSUES + issue`, and

        line  = (wave + issue * NUM_WAVES) + band * SMEM_N_RPT
        token = n_in_warp * SMEM_N_RPT + (wave + issue * NUM_WAVES)

    Both collapse to the production form at `ISSUES == 1`, where `SMEM_N_RPT`
    and `NUM_WAVES` are equal. Verified against a model of the write/read
    mapping before being written: the model reproduces family A exactly, and
    is what ruled out the BLOCK_N 128 variants -- a wave's K read covers 64
    tokens, so BLOCK_N 128 would need a doubled score accumulator.
    """

    def _issue_split(self, d_flat):
        """`(band, issue)` for a flat DMA index. Band-major."""
        return d_flat // self.ISSUES_PER_WAVE, d_flat % self.ISSUES_PER_WAVE

    def _dma_line(self, d_flat):
        """The KV tile line this wave writes for `d_flat`."""
        band, issue = self._issue_split(d_flat)
        return self.wave_id_uni + issue * self.traits.NUM_WAVES + band * self.traits.SMEM_N_RPT

    def _dma_m0(self, buf_base_elems, line_stride, d_flat):
        addr = self.lds_kv_base_idx + (buf_base_elems + self._dma_line(d_flat) * line_stride) * self.traits.BF16_BYTES
        return rocdl.readfirstlane(T.i32, as_mlir_value(fx.Int32(addr)))

    def k_dma_base(self, buf_id, d):
        return self._dma_m0(dualwave._k_buf_base(self.traits, buf_id), self.traits.SMEM_K_LINE_STRIDE, d)

    def v_dma_base(self, buf_id, d):
        return self._dma_m0(dualwave._v_buf_base(self.traits, buf_id), self.traits.SMEM_V_LINE_STRIDE, d)

    # Which D stage the next DMA reads from global. Set by the loader
    # immediately before delegating, the same way `stride_kv_n_v` is, and safe
    # for the same reason: tracing is eager, so it is read while `super()`
    # runs and no branch is open across the swap.
    dma_stage = 0

    def kv_src_elem(self, src_base, d_flat):
        """Global element index for this lane's `d_flat` chunk of a KV tile.

        `band` only spans `SMEM_D_RPT` d-bands, and under `D_STAGES > 1` that
        is one *stage* of the head dim rather than all of it -- LDS holds a
        stage at a time. So the stage's base offset is the single term that
        makes staging reach global memory; everything else is unchanged, and
        at `D_STAGES == 1` the term is zero.
        """
        band, issue = self._issue_split(d_flat)
        line_n = self.wave_id + issue * self.traits.NUM_WAVES
        n_in_tile = self.n_in_warp * self.traits.SMEM_N_RPT + line_n
        global_d = self.d_bucket * self.traits.VEC_KV + band * self.traits.D_128B_SIZE
        if const_expr(self.traits.D_STAGES > 1):
            global_d = global_d + self.dma_stage * self.traits.STAGE_DIM
        return src_base + n_in_tile * self.stride_kv_n_v + global_d


class ParityKernelContext(_ParityKvStaging, dualwave.DualwaveKernelContext):
    """Dualwave context addressing arbitrary BHSD strides, with a runtime scale."""

    def __init__(
        self,
        traits,
        *,
        strides,
        sm_scale,
        num_head_q,
        num_head_k,
        hdim_qk,
        hdim_vo,
        padded_head=False,
        **kwargs,
    ):
        super().__init__(traits, **kwargs)
        # 12 strides in launch order: Q, K, V, O, each (batch, head, seq).
        # Numerically named per `sdpa-feature-gap.md`'s porting instruction --
        # the `z/h/m/k` suffixes it warns about have caused real bugs.
        (
            self.stride_q0,
            self.stride_q1,
            self.stride_q2,
            self.stride_k0,
            self.stride_k1,
            self.stride_k2,
            self.stride_v0,
            self.stride_v1,
            self.stride_v2,
            self.stride_o0,
            self.stride_o1,
            self.stride_o2,
        ) = strides
        self.sm_scale_arg = sm_scale
        self.num_head_q = num_head_q
        self.num_head_k = num_head_k
        # P1. `hdim_qk` is the real reduction extent, which may be narrower
        # than the compiled tile; `hdim_vo` is the real output width. They are
        # separate because the two GEMMs are not symmetric -- see
        # `ParityQLoader` and `ParityStoreHelper`.
        self.hdim_qk = hdim_qk
        self.hdim_vo = hdim_vo
        self.PADDED_HEAD = bool(padded_head)

    # -- runtime softmax scale -------------------------------------------
    #
    # The production kernel derives the scale from head_dim
    # (`rsqrt(head_dim) * log2e`), which is only correct when the compiled
    # tile *is* the real extent. Under a padded head it is wrong, and AOTriton
    # passes `Sm_scale` regardless, so it becomes an argument.
    #
    # `* log2e` is kept folded in: every downstream exp is `exp2`, and folding
    # here means the conversion is paid once per kernel rather than per tile.
    # Pre-scaling Q by it *before* the row max -- which the production kernel
    # already does -- is the anti-FMA correction, so nothing moves.

    def init_types_and_constants(self, head_dim_runtime=None):
        super().init_types_and_constants(head_dim_runtime=head_dim_runtime)
        # DMA issues each wave makes per d-band, and the flat count that
        # follows. `NUM_DMA_*` feed both the m0 tables and the `s_waitcnt`
        # budget the pipeline is balanced against, so they have to agree.
        self.ISSUES_PER_WAVE = self.traits.SMEM_N_RPT // self.traits.NUM_WAVES
        self.NUM_DMA_K = self.traits.SMEM_D_RPT * self.ISSUES_PER_WAVE
        self.NUM_DMA_V = self.NUM_DMA_K
        self.c_sm_scale = fx.Float32(self.sm_scale_arg)
        self.c_sm_scale_log2e = fx.Float32(
            arith.mulf(
                fx.as_ir_value(fx.Float32(self.sm_scale_arg)),
                fx.as_ir_value(fx.Float32(dualwave._LOG2E)),
                fastmath=self.fm_fast,
            )
        )

    # -- runtime head counts ---------------------------------------------

    def init_thread_mapping(self):
        super().init_thread_mapping()
        # Re-derive the four head indices with runtime counts. The *mapping* is
        # the production one verbatim: `h_idx` is decomposed against the KV head
        # count and recomposed against the group size, which groups the Q heads
        # sharing a KV head. Only the operands change from constexpr to runtime,
        # so a build with matching counts schedules identically.
        num_head_k = fx.Index(self.num_head_k)
        gqa_group = fx.Index(self.num_head_q) // num_head_k
        self.h_kv_idx = self.h_idx % num_head_k
        self.group_id = self.h_idx // num_head_k
        self.q_head_idx = self.h_kv_idx * gqa_group + self.group_id
        self.kv_head_idx = self.h_kv_idx

    # -- granule-general staging ------------------------------------------

    def init_dma_thread_offsets(self):
        """Split a lane into (token, d-bucket) for the granule it stages.

        Production splits `lane // VEC_KV` by `lane % VEC_KV`, which is right
        only when a granule spans exactly `VEC_KV` lanes -- true at 64, which
        is `VEC_KV * VEC_KV`, and nowhere else. A lane always moves `VEC_KV`
        contiguous D elements, so `granule // VEC_KV` lanes cover one token's
        granule and the rest of the wave advances the token.
        """
        traits = self.traits
        self.lane_in_warp = self.tid % traits.WARP_SIZE
        self.n_in_warp = self.lane_in_warp // traits.SMEM_D_BUCKETS
        self.d_bucket = self.lane_in_warp % traits.SMEM_D_BUCKETS

    def init_lds_read_bases(self):
        super().init_lds_read_bases()
        # `_k_lds_read_base_per_lane` folds `SMEM_N_RPT` as a literal 8.
        self.k_lds_read_base_per_lane = _k_read_base(self.traits, self.lane_mod_32, self.lane_div_32)

    # -- per-tensor strides ----------------------------------------------

    def init_runtime_indices(self, **kwargs):
        super().init_runtime_indices(**kwargs)
        self.stride_q2_v = fx.Index(self.stride_q2)
        self.stride_k2_v = fx.Index(self.stride_k2)
        self.stride_v2_v = fx.Index(self.stride_v2)
        self.stride_o2_v = fx.Index(self.stride_o2)
        # The production helpers read these two names. Q's seq stride is the
        # default; `ParityKvGmemToLdsLoader` swaps the KV one per tensor.
        self.stride_q_n_v = self.stride_q2_v
        self.stride_kv_n_v = self.stride_k2_v

    def _slab_byte_base(self, s0, s1, s2, row_off, head_idx):
        """Byte offset of this workgroup's (batch, head) slab.

        Both axes are workgroup-uniform, so folding them into the descriptor
        costs scalar arithmetic once instead of a per-access add.

        **`row_off` is the varlen token origin, not the batch's.** The
        production kernel's `q_tok_base` is `batch * seqlen` in dense mode,
        because there the batch axis *is* a token offset into one flat
        allocation. Here the batch has its own stride, so passing `q_tok_base`
        would count it twice. Dense passes 0; varlen will pass the cumulative
        offset with `stride_0` set to 0, which is the same decomposition
        `fmha.decode_addressing` produces on gfx1201.
        """
        elems = self.batch_idx * fx.Index(s0) + head_idx * fx.Index(s1) + row_off * fx.Index(s2)
        return elems * fx.Index(self.traits.BF16_BYTES)

    def _slab_view(self, tensor, s0, s1, s2, row_off, head_idx, rows):
        """A buffer view over one (batch, head) slab, bounded at `rows` rows.

        The bound is `rows * stride_seq`, so a row past the sequence is out of
        the descriptor and reads as zero rather than faulting -- the same
        mechanism the production kernel uses for its ragged tail, restated over
        a stride the caller chose.
        """
        span_elems = rows * fx.Index(s2)
        return dualwave._make_rebased_view(
            fx.get_iter(tensor),
            self._slab_byte_base(s0, s1, s2, row_off, head_idx),
            span_elems * fx.Index(self.traits.BF16_BYTES),
            fx.make_layout(fx.Int32(span_elems), fx.Int32(1)),
            _buf_flags_i32=self.buf_flags_i32,
            _elem_ir=self.elem_ir,
        )

    def init_descriptors(self, **kwargs):
        """Rebuild Q/K/V/O over arbitrary strides; everything else stays.

        `super()` runs first for the state that does not depend on the
        addressing scheme -- `delta_i32`, `buf_flags_i32`, `elem_ir`, the
        paged page-view constants and the debug resource -- and its four dense
        views are then replaced. The discarded ones are pure descriptor
        arithmetic with no side effects, so they fold away; the alternative is
        duplicating the half of the method that has nothing to do with strides.
        """
        traits = self.traits
        super().init_descriptors(**kwargs)

        # Varlen token origins. Dense is 0 on both sides: the batch axis has a
        # real stride here, so it must not also be spent as a token offset.
        self.q_row_off = fx.Index(0)
        self.kv_row_off = fx.Index(0)

        # Head folded into the base, so what remains per access is `s * stride`.
        self.q_gmem_elem_offset = self.q_start * self.stride_q2_v
        self.kv_gmem_elem_offset = fx.Index(0)

        # First element past O's descriptor. A store redirected here is dropped
        # by the hardware bound, which is how `ParityStoreHelper` suppresses
        # the D-tail chunks without branching.
        self.o_oob_off = self.seqlen_q_v * self.stride_o2_v

        self.q_div = self._slab_view(
            self.Q, self.stride_q0, self.stride_q1, self.stride_q2, self.q_row_off, self.q_head_idx, self.seqlen_q_v
        )
        self.o_div = self._slab_view(
            self.O, self.stride_o0, self.stride_o1, self.stride_o2, self.q_row_off, self.q_head_idx, self.seqlen_q_v
        )
        if const_expr(not traits.PAGED):
            self.k_div = self._slab_view(
                self.K,
                self.stride_k0,
                self.stride_k1,
                self.stride_k2,
                self.kv_row_off,
                self.kv_head_idx,
                self.seqlen_kv_v,
            )
            self.v_div = self._slab_view(
                self.V,
                self.stride_v0,
                self.stride_v1,
                self.stride_v2,
                self.kv_row_off,
                self.kv_head_idx,
                self.seqlen_kv_v,
            )


class ParityQLoader(dualwave.DualwaveQLoader):
    """Q staging that zeroes the columns past `hdim_qk`.

    **Masking Q is what makes a padded head correct, and masking it is
    enough** for any finite pad -- `QK^T = sum_d Q[d] * K[d]`, so a zero in Q
    annihilates whatever K holds at the same column. K is left unmasked
    deliberately: Q is loaded once in the prologue, K once per KV tile, so this
    is the one of the two that is free. See `sdpa-close-gap-gfx950.md` for the
    residual case (a pad holding NaN or Inf, where `0 * NaN` is NaN) and the
    test that pins it.

    Whole 8-element chunks are still *loaded* past the extent, and are allowed
    to be: the D-axis pitch is contractually a multiple of 8 elements, so the
    chunk containing `hdim_qk` lands inside the allocation. What must not
    happen is those elements reaching the MFMA, which is what `discard` stops.
    """

    def load_pack(self, q_row_in_block, ks):
        pack = super().load_pack(q_row_in_block, ks)
        if const_expr(not self.PADDED_HEAD):
            return pack
        col_base = fx.Index(ks * self.traits.K_STEP_QK) + self.lane_div_32 * fx.Index(self.traits.MFMA_LANE_K)
        # Q is masked once, before the KV loop, so the mask registers die
        # immediately and the bitmask form is pure win here.
        return MaskedAxis(fx.Index(self.hdim_qk), elem_dtype=self.elem_dtype, bitmask=True).discard(
            pack, col_base, self.traits.MFMA_LANE_K
        )

    def load_all(self):
        """Q rows for this wave, for any `K_STEPS_QK`.

        The production version assembles the packs through a fixed 8 -> 16 ->
        32 tree and then takes either one 32-pack or a concatenation of two,
        which covers `K_STEPS_QK` 4 and 8 -- head_dim 64 and 128 -- and nothing
        else. head_dim 192 wants 12 packs and 256 wants 16, so the tail of that
        tree simply drops the remainder and the `Vec` constructor rejects the
        result ("shape (96,) has 96 elements, but value has type
        vector<64xbf16>"). It fails loudly, which is the good case.

        A left fold over the packs replaces the tree. `_concat_vectors` builds
        an explicit shuffle index list, so it does not need equal widths and
        the uneven steps 192 needs (32+32 -> 64, 64+32 -> 96) are fine.
        """
        traits = self.traits
        ctx = self.ctx_ref
        ctx.init_q_row()
        acc = self.load_pack(ctx.q_row_in_block, 0)
        for ks in range_constexpr(traits.K_STEPS_QK - 1):
            acc = dualwave._concat_vectors(acc, self.load_pack(ctx.q_row_in_block, ks + 1))
        return Vec(acc, (traits.K_STEPS_QK * traits.MFMA_LANE_K,), self.elem_dtype)


class ParityKvLdsToVgprLoader(dualwave.DualwaveKvLdsToVgprLoader):
    """K register reads that zero the columns past `hdim_qk`.

    Masking Q alone is enough for a *finite* pad, since `0 * x == 0`. It is not
    enough for a pad holding NaN or Inf, and a caller's D-axis padding is
    allocation slack whose contents nothing constrains -- so K is masked too.
    V needs nothing: its D columns are O's columns, and those are suppressed at
    the store instead.

    **Why the mask is the same expression as Q's.** The K tile transits LDS in
    a swizzled layout, so the column a register holds is not obviously its
    linear D index. Working the two halves against each other: the DMA writes
    LDS element `base + w*LINE + d*N_RPT*LINE + l*8 + i` holding D column
    `(l % 8) * 8 + d * 64 + i`, while `load_k` reads at
    `(lm % 8) * LINE + (lm // 8) * 64 + ld * 8 + (ks // 4) * N_RPT * LINE +
    (ks % 4) * 16`. Matching term by term gives `w = lm % 8`, `d = ks // 4` and
    `l = (lm // 8) * 8 + ld + (ks % 4) * 2`, and since `(lm // 8) * 8` vanishes
    mod 8,

        D = (ld + (ks % 4) * 2) * 8 + (ks // 4) * 64 + i
          = ks * 16 + ld * 8 + i

    which is `_q_pack_col` exactly -- the swizzle permutes *tokens* across LDS
    lines and leaves D in linear order. `k_hi` differs from `k_lo` by
    `K_LDS_TO_REG_N_STRIP_STRIDE`, an N offset, so it carries the same columns.
    """

    @contextlib.contextmanager
    def _scoped_to_stage(self):
        """Narrow `K_STEPS_QK` / `D_CHUNKS` to one stage for the duration.

        Under `D_STAGES > 1` an LDS buffer holds one stage of the head dim, so
        the inherited readers -- which loop to `K_STEPS_QK` and `D_CHUNKS` --
        would run off the end of it. The *offsets* need no adjustment:
        `_swizzled_ks_offset` and `_swizzled_v_dc_off` address d-bands within
        the buffer, and a stage-sized buffer has exactly `SMEM_D_RPT` of them.
        Only the counts are wrong.

        So this swaps the two counts rather than reimplementing the read
        loops. Copying them is the specific thing to avoid here: the write side
        (`kv_src_elem`) and the read side have to describe one LDS layout, and
        a second copy of either is how they drift apart.

        A no-op at `D_STAGES == 1`, where the two values are already equal --
        deliberately not merely equivalent, so a default build cannot differ.
        """
        if const_expr(self.traits.D_STAGES == 1):
            yield
            return
        full = self.traits
        self.traits = replace(
            full,
            K_STEPS_QK=full.K_STEPS_PER_STAGE,
            D_CHUNKS=full.D_CHUNKS_PER_STAGE,
        )
        try:
            yield
        finally:
            self.traits = full

    def _read_k_packs(self, buf_id, urk_base):
        """The inherited non-vectorized K read, with a granule-general swizzle.

        Six lines rather than a `super()` call because the production loop
        calls `_swizzled_ks_offset`, which folds `K_STEPS_PER_BAND` as a
        literal 4 -- a module function, so there is nothing to override but the
        loop that calls it. Identical arithmetic at granule 64.
        """
        traits = self.traits
        k_base = dualwave._k_buf_base(traits, buf_id)
        k_lo = [None] * traits.K_STEPS_QK
        k_hi = [None] * traits.K_STEPS_QK
        for ks in range_constexpr(traits.K_STEPS_QK):
            k_lo[ks], k_hi[ks] = self._load_k_pair(buf_id, k_base + urk_base + _ks_offset(traits, ks))
        return k_lo, k_hi

    def load_k(self, buf_id, urk_base=None, stage=0):
        with self._scoped_to_stage():
            if const_expr(self.traits.KV_VECTORIZED):
                k_lo, k_hi = super().load_k(buf_id, urk_base=urk_base)
            else:
                base = self.k_lds_read_base_per_lane if urk_base is None else urk_base
                k_lo, k_hi = self._read_k_packs(buf_id, base)
            steps = self.traits.K_STEPS_QK
        if const_expr(not self.PADDED_HEAD):
            return (k_lo, k_hi)
        # K is masked *inside* the KV loop, so the hoisted masks stay live and
        # compete with everything else. Worth it only where they fit: measured
        # +21% in the 64-wide tile and -43% in the 128-wide one, where 32 extra
        # live registers turn a spill-free build into 61 spills.
        cols = MaskedAxis(
            fx.Index(self.hdim_qk),
            elem_dtype=self.elem_dtype,
            bitmask=self.traits.K_STEPS_QK * (self.traits.MFMA_LANE_K // 2) <= 16,
        )
        width = self.traits.MFMA_LANE_K
        for ks in range_constexpr(steps):
            # The mask is against the *global* D column, so the stage's base
            # has to come back in here -- `ks` is stage-relative above.
            col_base = fx.Index((stage * steps + ks) * self.traits.K_STEP_QK) + self.lane_div_32 * fx.Index(width)
            k_lo[ks] = cols.discard(k_lo[ks], col_base, width)
            k_hi[ks] = cols.discard(k_hi[ks], col_base, width)
        return (k_lo, k_hi)

    def read_v_packs(self, buf_id, urv_base):
        """The inherited non-vectorized V read, with a granule-general swizzle.

        Same reason as `_read_k_packs`: the production loop calls
        `_swizzled_v_imm_lo`, which reaches `_swizzled_v_dc_off` and its
        literal 2. Identical arithmetic at granule 64.
        """
        traits = self.traits
        lds_base = dualwave._v_buf_base(traits, buf_id) + urv_base
        v_scope = dualwave._dualwave_lds_scope("v", buf_id)
        packs = [[None] * traits.D_CHUNKS for _ in range(4)]
        for dc in range_constexpr(traits.D_CHUNKS):
            for k_substep in range_constexpr(4):
                imm_lo = _v_imm_lo(traits, dc, k_substep)
                pair = traits.V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE * traits.BF16_BYTES
                read = lambda off: dualwave._ds_read_tr_v4f16_imm(  # noqa: E731
                    lds_base,
                    off,
                    lds_kv_base_idx=self.lds_kv_base_idx,
                    v_lds_read_vec4_type=self.v_lds_read_vec4_type,
                    scope_name=v_scope,
                    scope_names=traits.LDS_SCOPE_NAMES,
                )
                a, b = read(imm_lo), read(imm_lo + pair)
                packs[k_substep][dc] = Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
        return packs

    def load_v(self, buf_id, urv_base=None, stage=0):
        with self._scoped_to_stage():
            if const_expr(self.traits.KV_VECTORIZED):
                return super().load_v(buf_id, urv_base=urv_base)
            base = self.v_lds_read_base_per_lane if urv_base is None else urv_base
            return self.read_v_packs(buf_id, base)


class ParityKvGmemToLdsLoader(_ParityKvStaging, dualwave.DualwaveKvGmemToLdsLoader):
    """K/V staging with independent per-tensor sequence strides.

    The production loader has one `stride_kv_n` for both tensors, which the
    BHSD ABI can contradict -- K and V are separate allocations and a caller
    may hand us one as BHSD and the other as BSHD. `load_k` and `load_v` are
    already separate methods that read `self.stride_kv_n_v` on the way down, so
    selecting the right stride is an attribute swap before delegating, not a
    second copy of the DMA body.

    Safe because tracing is eager: the attribute is read while `super()` runs,
    and no branch is open across the swap.
    """

    def load_k(self, tile_start, buf_id, page_id=None, stage=0):
        self.stride_kv_n_v = self.stride_k2_v
        self.dma_stage = stage
        return super().load_k(tile_start, buf_id, page_id=page_id)

    def load_v(self, tile_start, buf_id, page_id=None, stage=0):
        self.stride_kv_n_v = self.stride_v2_v
        self.dma_stage = stage
        return super().load_v(tile_start, buf_id, page_id=page_id)

    # `load_*_tile` resolve a tile index to a token offset and delegate; the
    # stage has to ride along or it is lost at that hop.
    def load_k_tile(self, tile_idx, buf_id, page_id=None, stage=0):
        self.load_k(self.tile_start(tile_idx), buf_id, page_id=page_id, stage=stage)

    def load_v_tile(self, tile_idx, buf_id, page_id=None, stage=0):
        self.load_v(self.tile_start(tile_idx), buf_id, page_id=page_id, stage=stage)

    def _async_load_kv_linear(self, dma_m0, buf_id, src_div, src_base, soffset, num_dma):
        """Issue this wave's KV DMAs, addressed through `kv_src_elem`.

        The production version calls `_linear_kv_src_elem`, which interleaves
        tokens across `NUM_WAVES` and offsets D by the flat index -- both true
        only at one issue per wave. This is the same loop with the address
        redirected, not a second copy of the DMA sequence.
        """
        for d in range_constexpr(num_dma):
            self._issue_kv_dma(src_div, dma_m0[buf_id][d], self.kv_src_elem(src_base, d), soffset)


class ParitySoftmaxHelper(dualwave.DualwaveSoftmaxHelper):
    """Softmax whose running max can never be `-inf`.

    The production `reduce_max` seeds the reduction with `-inf` and the kernel
    floors the result to `-3.0e38` only under `CAUSAL`. That is the pattern
    `sdpa-feature-gap.md` flags:

        m_i = tl.full([BLOCK_M], -3.40282e+38)   # do NOT use -inf

    A row whose scores are all `-inf` -- every key masked -- gives
    `m_i = -inf`, and then `exp2(-inf - -inf)` is `NaN` rather than 0. On
    gfx1201 this is *preventative* today and only becomes reachable with bias,
    since causal tile 0 always contains `kv = 0 <= q_row`. Here it is cheaper
    to be unconditional: seeding the reduction at the floor costs nothing (it
    replaces one constant with another) and removes the case for every masking
    mode at once, including the windows and bias still to come.

    Seeding rather than flooring afterwards is the part that matters. A floor
    applied to the *result* still lets an all-`-inf` tile reach the subtract;
    seeding means no lane ever holds `-inf` as a max in the first place.
    """

    def reduce_max(self, v_s):
        return dualwave._score_pair_max(v_s, self.c_neg_floor, self.fm_fast)

    def rescale_o_serial(self, v_o, m_row, l_row, m_tile_max):
        """`rescale_o` without the `v_p` term, for the staged (unpipelined) loop.

        The dualwave schedule rescales O *and* the previous tile's P, because
        its softmax is split across clusters and a P from the last iteration is
        still in flight. The staged loop has no such P: it finishes each tile
        before starting the next, so at rescale time the only live state is O,
        the running max and the running sum. Passing a dummy `v_p` to
        `rescale_o` would work and would also emit a real multiply over it.

        Otherwise identical to `rescale_o`, term for term.
        """
        m_new = dualwave._fmax(m_row, m_tile_max, self.fm_fast)
        corr = rocdl.exp2(T.f32, as_mlir_value(dualwave._fsub(m_row, m_new, self.fm_fast)))
        self.scale_o(v_o, corr)
        v_o = _anchor_v_o(self.traits, v_o)
        l_row = dualwave._fmul(l_row, corr, self.fm_fast)
        return v_o, m_new, l_row


class ParityStoreHelper(dualwave.DualwaveStoreHelper):
    """O stores addressed by O's own strides.

    `_final_o_base` is the one place the production code spells O's address,
    and it spells it with *Q's* token stride plus `q_head_idx * HEAD_DIM` --
    correct only where O and Q share a layout. Overriding this single method is
    the whole change; the 128-bit store path above it is untouched.
    """

    def _final_o_base(self, q_row):
        return q_row * self.stride_o2_v + self.lane_div_32 * 8

    def _final_o_global(self, o_base, dc, g):
        """The store's element offset, redirected out of the buffer if past `hdim_vo`.

        A 128-bit store is all-or-nothing, so a chunk straddling `hdim_vo`
        cannot be partially written -- and does not need to be. The D pitch is
        contractually a multiple of 8 elements, so a chunk that *starts* inside
        `hdim_vo` ends inside the allocation, and writing its tail into the
        caller's own padding is exactly what that contract permits. Only
        chunks starting at or past `hdim_vo` must be suppressed.

        Suppressed by *address*, not by a branch: pushing the offset past the
        descriptor's `num_records` makes the hardware drop the store, which
        costs one select on a lane-varying value instead of an `scf.if` around
        a 128-bit store. `store_lse` uses the same device.
        """
        off = super()._final_o_global(o_base, dc, g)
        if const_expr(not self.PADDED_HEAD):
            return off
        col_base = fx.Index(dc * self.traits.D_CHUNK + 2 * g * 8) + self.lane_div_32 * fx.Index(8)
        in_range = MaskedAxis(fx.Index(self.hdim_vo)).valid(col_base)
        return fx.Index(in_range.select(fx.Index(off), self.o_oob_off))
