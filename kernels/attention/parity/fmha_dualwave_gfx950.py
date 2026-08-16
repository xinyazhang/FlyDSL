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

import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value
from fmha_common_gfx1201 import MaskedAxis
from gfx950_standalone import dualwave

__all__ = [
    "ParityKernelContext",
    "ParityKvGmemToLdsLoader",
    "ParityKvLdsToVgprLoader",
    "ParityQLoader",
    "ParityStoreHelper",
]


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

    def kv_src_elem(self, src_base, d_flat):
        """Global element index for this lane's `d_flat` chunk of a KV tile."""
        band, issue = self._issue_split(d_flat)
        line_n = self.wave_id + issue * self.traits.NUM_WAVES
        n_in_tile = self.n_in_warp * self.traits.SMEM_N_RPT + line_n
        global_d = self.d_bucket * self.traits.VEC_KV + band * self.traits.D_128B_SIZE
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
        runtime_qk_steps=False,
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
        self.RUNTIME_QK_STEPS = bool(runtime_qk_steps)

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
        return MaskedAxis(fx.Index(self.hdim_qk), elem_dtype=self.elem_dtype).discard(
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

    def load_k(self, buf_id, urk_base=None):
        k_lo, k_hi = super().load_k(buf_id, urk_base=urk_base)
        if const_expr(not self.PADDED_HEAD):
            return (k_lo, k_hi)
        cols = MaskedAxis(fx.Index(self.hdim_qk), elem_dtype=self.elem_dtype)
        width = self.traits.MFMA_LANE_K
        for ks in range_constexpr(self.traits.K_STEPS_QK):
            col_base = fx.Index(ks * self.traits.K_STEP_QK) + self.lane_div_32 * fx.Index(width)
            k_lo[ks] = cols.discard(k_lo[ks], col_base, width)
            k_hi[ks] = cols.discard(k_hi[ks], col_base, width)
        return (k_lo, k_hi)


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

    def load_k(self, tile_start, buf_id, page_id=None):
        self.stride_kv_n_v = self.stride_k2_v
        return super().load_k(tile_start, buf_id, page_id=page_id)

    def load_v(self, tile_start, buf_id, page_id=None):
        self.stride_kv_n_v = self.stride_v2_v
        return super().load_v(tile_start, buf_id, page_id=page_id)

    def _async_load_kv_linear(self, dma_m0, buf_id, src_div, src_base, soffset, num_dma):
        """Issue this wave's KV DMAs, addressed through `kv_src_elem`.

        The production version calls `_linear_kv_src_elem`, which interleaves
        tokens across `NUM_WAVES` and offsets D by the flat index -- both true
        only at one issue per wave. This is the same loop with the address
        redirected, not a second copy of the DMA sequence.
        """
        for d in range_constexpr(num_dma):
            self._issue_kv_dma(src_div, dma_m0[buf_id][d], self.kv_src_elem(src_base, d), soffset)


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
