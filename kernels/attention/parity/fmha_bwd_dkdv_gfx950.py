# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""dK / dV for gfx950 -- AOTriton's `bwd_kernel_dk_dv` on the dualwave helpers.

B1 of `sdpa-bwd-plan-gfx950.md`: dense, non-causal, head_dim 64 and 128, bf16,
MHA. The contract is `sdpa-bwd-contract-gfx950.md`; read that first.

--- The whole design in one paragraph --------------------------------------

**dK/dV is the forward loop transposed.** K and V stay resident in registers
while Q and dO stream past through LDS, exactly mirroring the forward's
resident Q and streaming K/V. Every helper in `fmha_dualwave_gfx950.py` was
written against "a tensor, its bounds and where it lands" rather than against
K or V by name, so the roles swap by re-pointing four descriptors and nothing
else. The two tensor slots the staging machinery calls *K* and *V* carry **Q**
and **dO** here, and `BLOCK_M` / `BLOCK_N` in the traits mean the KV block and
the Q tile -- see `fmha_tuning_bwd_dkdv_gfx950`'s docstring for the full map.

--- Four GEMMs, and where each operand comes from ---------------------------

    S   = Q . K^T      contract d      A = Q  (LDS, row-major)  B = K  (regs)
    dP  = dO . V^T     contract d      A = dO (LDS, row-major)  B = V  (regs)
    dV^T = dO^T . P    contract q      A = dO (LDS, **column**) B = P  (regs)
    dK^T = Q^T  . dS   contract q      A = Q  (LDS, **column**) B = dS (regs)

`v_mfma_f32_32x32x16_bf16`'s A and B operands have the *same* per-lane layout
-- 32 outer rows selected by `lane % 32`, 16 contraction elements from
`lane // 32` and the element index -- so the output is `[A's row][B's row]`
with B's row landing on `lane % 32`. That symmetry is why the forward's K
reader and its Q loader produce interchangeable packs, and it is what makes
the first two GEMMs above a rename of `DualwaveGemmHelper.qk`.

**The two column-major operands are the forward's V read, unmodified.** The
forward stages V as `[token][d]` and reads it back through
`ds_read_b64_tr_b16` as `A[d][token]`, which is precisely what the
q-contracted GEMMs want from a `[q_row][d]` tile. So both streamed tiles are
staged in the *V* LDS shape and read two ways: `ds_read_b128` for the
row-major operand and the transpose path for the column-major one. Deriving a
second lane map was the alternative, and contract section 3 says not to.

--- Why nothing is masked ---------------------------------------------------

Dense and non-causal, so the only edge is the ragged tail, and the buffer
descriptors already answer it. Q and dO are bounded at `seqlen_q` rows, so a
staged row past the end reads **zero**; K and V at `seqlen_kv`, and dK/dV
stores past it are dropped by the same bound. A padding q row therefore gets
`S = 0` and `LSE = 0` (also out of its buffer), hence `P = exp2(0) = 1` -- and
contributes `1 * dO = 0` to dV and `dS * Q = 0` to dK. Finite, exact, and no
`scf.if` anywhere near a transpose read, which contract section 3 warns is
undefined under a divergent EXEC.

--- Phase status -----------------------------------------------------------

Causal and windows (B4), varlen (B5), dropout and bias (B6), the head_dim
ladder and the wide body (B3), and GQA are **not implemented**. The ABI
carries their argument slots so the wire format does not move when they
arrive, and `BwdDkDvKnobs.resolve` refuses a build that would need one; there
is no half-implemented arm anywhere below.
"""

import weakref

import fmha_abi_gfx1201 as abi
from fmha_dualwave_gfx950 import (
    ParityGemmHelper,
    ParityKernelContext,
    ParityKvGmemToLdsLoader,
    ParitySoftmaxHelper,
    _score_column_runs,
    _v_imm_lo,
)
from fmha_tuning_bwd_dkdv_gfx950 import BwdDkDvInputMetadata, bwd_dkdv_knobs
from gfx950_standalone import buffer_ops, dualwave

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

__all__ = [
    "KERNEL_NAME",
    "build_fmha_bwd_dkdv_gfx950_module",
    "build_fmha_bwd_dkdv_gfx950_module_primary",
]

KERNEL_NAME = "fmha_bwd_dkdv_gfx950_kernel"

_s_barrier = dualwave._s_barrier
_s_waitcnt = dualwave._s_waitcnt
_sched_barrier = dualwave._sched_barrier
_waitcnt_vm_n = dualwave._waitcnt_vm_n

_COMPILED = weakref.WeakKeyDictionary()

_COMPILE_HINTS = {
    "fast_fp_math": True,
    "unsafe_fp_math": True,
    "llvm_options": {
        "enable-post-misched": False,
        "lsr-drop-solution": True,
    },
}

# `_causal_pair_thresholds(False)` flattened is `8 * (r // 4) + (r % 4)`, which
# is the MFMA accumulator's row map, and grouping it into runs gives four spans
# of four. So one `buffer_load` of 4 f32 covers one span of LSE (or delta) rows
# exactly, and the row set a lane needs is 4 loads per accumulator half. The
# forward derives the same runs for its bias reads; deriving them once is
# contract section 4's rule, and this is the second consumer that proves it.
_ROW_RUNS = _score_column_runs(False)


def _stream_row_read_base(traits, lane_mod_32, lane_div_32):
    """Per-lane LDS base for the row-major read of a staged tile.

    `fmha_dualwave_gfx950._k_read_base` with `STREAM_LINE_STRIDE` in place of
    `SMEM_K_LINE_STRIDE`: both streamed tiles are V-shaped (see
    `BwdDkDvTraits.STREAM_LINE_STRIDE`), and the line stride is the only term
    of the K read that knows which shape it is addressing.
    """
    return (
        (lane_mod_32 % traits.SMEM_N_RPT) * traits.STREAM_LINE_STRIDE
        + (lane_mod_32 // traits.SMEM_N_RPT) * traits.D_128B_SIZE
        + lane_div_32 * traits.VEC_KV
    )


def _stream_ks_offset(traits, ks):
    """LDS offset of MFMA k-step `ks` within a staged tile, in elements."""
    per_band = traits.K_STEPS_PER_BAND
    return (ks // per_band) * (traits.SMEM_N_RPT * traits.STREAM_LINE_STRIDE) + (ks % per_band) * traits.K_STEP_QK


def _lds_pack_read(traits, lds_ptr, elem_idx, scope_name, pack_type):
    """One 128-bit LDS read of 8 bf16, alias-scoped to the buffer it touches.

    `dualwave._load_k_pack_aligned` with the scope name as a parameter. It
    derives its own as `lds_k{buf}`, which is right for the forward's one
    row-major reader and wrong here, where two tensors are staged and the
    second lives in the `v` scopes. The alias scopes are not decoration: the
    backend drains *all* outstanding `buffer_load ... lds` before any DS read
    that may alias one, and that drain was 26% of the forward's runtime.
    """
    ptr = buffer_ops.get_element_ptr(lds_ptr, byte_offset=elem_idx * traits.BF16_BYTES, elem_type=T.i8)
    return dualwave.llvm.LoadOp(
        pack_type,
        ptr,
        alignment=16,
        alias_scopes=dualwave._dualwave_lds_alias_scopes(scope_name),
        noalias_scopes=dualwave._dualwave_lds_noalias_scopes(scope_name, traits.LDS_SCOPE_NAMES),
    ).result


class BwdDkDvKernelContext(ParityKernelContext):
    """Parity context whose resident tensor is K/V and whose stream is Q/dO.

    Inherits the BHSD stride ABI, the runtime head counts, the varlen decode
    (pinned dense until B5) and the philox prologue (inert until B6) by
    subclassing rather than porting, per contract section 2.

    Five things move, and they are all descriptors or bounds:

    - `strides` carries **18** slots, not 12: Q, K, V, dO, dK, dV. The first
      twelve go to the base class, which means its `stride_o_*` names hold
      **dO's** strides. Nothing below reads those names; `stride_do_*` is
      spelled out instead.
    - The staging machinery's K slot carries Q and its V slot carries dO.
    - Both staged tiles are V-shaped, so the LDS bases and the row-major read
      base are recomputed against `STREAM_LINE_STRIDE`.
    - The tile loop walks **q** tiles, so `init_tile_bounds` counts them from
      `seqlen_q`.
    - LSE and delta get f32 row resources, which the forward has no reader for.
    """

    def __init__(self, traits, *, strides, DO, DK, DV, Delta, **kwargs):
        super().__init__(traits, strides=strides[:12], **kwargs)
        self.DO = DO
        self.DK = DK
        self.DV = DV
        self.Delta = Delta
        (
            self.stride_do_batch,
            self.stride_do_head,
            self.stride_do_seq,
            self.stride_dk_batch,
            self.stride_dk_head,
            self.stride_dk_seq,
            self.stride_dv_batch,
            self.stride_dv_head,
            self.stride_dv_seq,
        ) = strides[9:18]

    # -- LDS ---------------------------------------------------------------

    def init_lds(self, shared_storage):
        """Allocate the staged tiles, keeping the base class's attribute names.

        `lds_kv_base_idx` / `lds_kv_base_ptr` are read by every DMA and LDS
        helper in the stack, so they stay -- only the storage field is renamed
        to say what it holds.
        """
        lds = fx.SharedAllocator().allocate(shared_storage).peek()
        self.lds = lds
        self.lds_kv_base_idx = fx.Index(fx.ptrtoint(lds.stream.ptr))
        self.lds_kv_base_ptr = lds.stream.ptr.llvm_ptr
        self.lds_bt_base_ptr = None

    def q_buf_base(self, buf_id):
        """First element of the Q tile in stream buffer `buf_id`."""
        return (2 * buf_id) * self.traits.STREAM_TILE_ELEMS

    def do_buf_base(self, buf_id):
        """First element of the dO tile in stream buffer `buf_id`."""
        return (2 * buf_id + 1) * self.traits.STREAM_TILE_ELEMS

    def k_dma_base(self, buf_id, d):
        """m0 for the Q staging DMA. The forward's K slot; see the class docstring."""
        return self._dma_m0(self.q_buf_base(buf_id), self.traits.STREAM_LINE_STRIDE, d)

    def v_dma_base(self, buf_id, d):
        """m0 for the dO staging DMA. The forward's V slot."""
        return self._dma_m0(self.do_buf_base(buf_id), self.traits.STREAM_LINE_STRIDE, d)

    def init_lds_read_bases(self):
        super().init_lds_read_bases()
        self.stream_row_read_base = _stream_row_read_base(self.traits, self.lane_mod_32, self.lane_div_32)
        # The column-major base is the forward's V one *unchanged*, which is
        # the whole of contract section 3: the tile is V-shaped and the read is
        # the same instruction sequence, so the validated lane map carries over
        # rather than being re-derived.
        self.stream_col_read_base = self.v_lds_read_base_per_lane

    # -- indices -----------------------------------------------------------

    def init_runtime_indices(self, **kwargs):
        super().init_runtime_indices(**kwargs)
        # The base class names dO's seq stride `stride_o_seq_v`, since dO
        # occupies its O slot. The two outputs have no slot there at all.
        self.stride_do_seq_v = self.stride_o_seq_v
        self.stride_dk_seq_v = fx.Index(self.stride_dk_seq)
        self.stride_dv_seq_v = fx.Index(self.stride_dv_seq)

    def init_thread_mapping(self):
        super().init_thread_mapping()
        # Aliases, not new values: `grid.y` selects a KV block here and the
        # base class's `q_*` names already hold exactly that arithmetic.
        self.kv_block_idx = self.q_block_idx
        self.kv_start = self.q_start
        self.wave_kv_offset = self.wave_q_offset

    def init_kv_row(self):
        """The 32 KV rows this wave owns. `init_q_row`'s arithmetic, renamed."""
        self.init_q_row()
        self.kv_row_in_block = self.q_row_in_block
        self.kv_row = self.q_row

    # -- descriptors -------------------------------------------------------

    def init_descriptors(self, **kwargs):
        """Six tensor views plus two f32 row resources.

        `super()` runs first for the state that is not about addressing --
        `delta_i32`, `buf_flags_i32`, `elem_ir` -- and its four dense views are
        replaced below. They are pure descriptor arithmetic with no side
        effects, so they fold away; the forward's `init_descriptors` makes the
        same trade for the same reason.
        """
        super().init_descriptors(**kwargs)
        traits = self.traits
        self.q_row_off = fx.Index(0)
        self.kv_row_off = fx.Index(0)
        # The staged tiles address from token 0 of the slab; the tile offset
        # rides in the DMA's `soffset`, which is what `_kv_tile_addr` produces.
        self.q_gmem_elem_offset = fx.Index(0)
        self.kv_gmem_elem_offset = fx.Index(0)

        # Streamed. Bounded at `seqlen_q` rows, which is what makes the ragged
        # tail stage as zeros instead of faulting.
        self.k_div = self._slab_view(
            self.Q,
            self.stride_q_batch,
            self.stride_q_head,
            self.stride_q_seq,
            fx.Index(0),
            self.q_head_idx,
            self.seqlen_q_v,
        )
        self.v_div = self._slab_view(
            self.DO,
            self.stride_do_batch,
            self.stride_do_head,
            self.stride_do_seq,
            fx.Index(0),
            self.q_head_idx,
            self.seqlen_q_v,
        )
        self.q_div = self.k_div

        # Resident, and the two outputs: all four are (batch, kv head) slabs
        # bounded at `seqlen_kv` rows.
        self.k_res_div = self._slab_view(
            self.K,
            self.stride_k_batch,
            self.stride_k_head,
            self.stride_k_seq,
            fx.Index(0),
            self.kv_head_idx,
            self.seqlen_kv_v,
        )
        self.v_res_div = self._slab_view(
            self.V,
            self.stride_v_batch,
            self.stride_v_head,
            self.stride_v_seq,
            fx.Index(0),
            self.kv_head_idx,
            self.seqlen_kv_v,
        )
        self.dk_div = self._slab_view(
            self.DK,
            self.stride_dk_batch,
            self.stride_dk_head,
            self.stride_dk_seq,
            fx.Index(0),
            self.kv_head_idx,
            self.seqlen_kv_v,
        )
        self.dv_div = self._slab_view(
            self.DV,
            self.stride_dv_batch,
            self.stride_dv_head,
            self.stride_dv_seq,
            fx.Index(0),
            self.kv_head_idx,
            self.seqlen_kv_v,
        )
        self.o_div = self.dk_div

        self.k_res_elem_base = self.kv_start * self.stride_k_seq_v
        self.v_res_elem_base = self.kv_start * self.stride_v_seq_v

        # LSE and delta. Compact `(batch * heads, tokens)` f32, the layout the
        # forward writes LSE in -- `q_head_idx * seq_len_v + q_row` inside a
        # per-batch slab. Bounding each resource at one head's row means a
        # token past `seqlen_q` reads zero, which is the value the "nothing is
        # masked" argument in the module docstring depends on.
        row_bytes = self.seq_len_v * fx.Index(4)
        per_batch_bytes = fx.Index(self.num_head_q) * row_bytes
        head_bytes = self.batch_idx * per_batch_bytes + self.q_head_idx * row_bytes
        self.lse_rsrc = dualwave._make_ws_rsrc(fx.Int64(fx.ptrtoint(fx.get_iter(self.LSE))), head_bytes, row_bytes)
        self.delta_rsrc = dualwave._make_ws_rsrc(fx.Int64(fx.ptrtoint(fx.get_iter(self.Delta))), head_bytes, row_bytes)
        del traits

    # -- bounds ------------------------------------------------------------

    def init_tile_bounds(self, **kwargs):
        """Count **q** tiles, rounded up to an even number.

        The body consumes two tiles per iteration, one per LDS buffer, so an
        odd count would leave the second half of the last iteration reading a
        tile that does not exist. Rounding up costs one dead tile rather than
        an epilogue: its rows are past `seqlen_q`, so Q and dO stage as zeros
        and it contributes exactly nothing (module docstring).
        """
        traits = self.traits
        self.kv_tile_size = traits.BLOCK_Q
        tiles = (self.seqlen_q_v + fx.Index(traits.BLOCK_Q - 1)) // fx.Index(traits.BLOCK_Q)
        tiles = ((tiles + fx.Index(1)) // fx.Index(2)) * fx.Index(2)
        tiles = fx.Index((tiles < fx.Index(2)).select(fx.Index(2), tiles))
        self.num_q_tiles = tiles
        self.max_num_tiles = tiles
        self.causal_end_raw_i32 = None
        self.split_t0 = 0
        self.split_t_end = tiles


class BwdDkDvStreamLoader(ParityKvGmemToLdsLoader):
    """Stages Q and dO into the K and V LDS slots.

    Two wrappers rather than an override of `load_k` / `load_v`, because the
    parity loader's job -- swapping in the right per-tensor sequence stride
    before delegating -- is exactly what has to change, and the tensor it must
    swap in is not the one the method is named for. Calling the *production*
    method directly keeps one implementation of the DMA sequence and states the
    substitution at the call site instead of hiding it in an attribute.
    """

    def stage_q_tile(self, tile_idx, buf_id):
        self.stride_kv_n_v = self.ctx_ref.stride_q_seq_v
        self.dma_stage = 0
        dualwave.DualwaveKvGmemToLdsLoader.load_k(self, self.tile_start(tile_idx), buf_id)

    def stage_do_tile(self, tile_idx, buf_id):
        self.stride_kv_n_v = self.ctx_ref.stride_do_seq_v
        self.dma_stage = 0
        dualwave.DualwaveKvGmemToLdsLoader.load_v(self, self.tile_start(tile_idx), buf_id)


class BwdDkDvResidentLoader(dualwave.DualwaveKernelContext):
    """K and V rows for this wave, loaded once and held for the whole q loop.

    The same shape of read as `DualwaveQLoader.load_pack` -- 128 bits per lane
    at `row * stride + ks * K_STEP_QK + (lane // 32) * MFMA_LANE_K` -- because
    the MFMA's A and B operands take identical per-lane layouts, so what the
    forward builds for Q is what this needs for K and V. Kept as a list of
    per-`ks` packs rather than one concatenated vector: the forward concatenates
    so that `_get_q_pack` can slice a loop-invariant register, and here the
    packs feed two different GEMMs and never need to be adjacent.
    """

    def load_rows(self, div, elem_base, stride_seq, row_in_block):
        traits = self.traits
        packs = []
        for ks in range_constexpr(traits.K_STEPS_QK):
            elem = (
                elem_base
                + row_in_block * stride_seq
                + fx.Index(ks * traits.K_STEP_QK)
                + self.lane_div_32 * fx.Index(traits.MFMA_LANE_K)
            )
            raw = dualwave._buffer_load_128(
                elem,
                _load_atom_128=self.load_atom_128,
                q_div=div,
                q_load_i32x4_type=self.q_load_i32x4_type,
            )
            packs.append(Vec(raw, (4,), fx.Int32).bitcast(self.elem_dtype).ir_value())
        return packs


class BwdDkDvStreamReader(dualwave.DualwaveKernelContext):
    """The two ways one staged tile is read back.

    Row-major for the d-contracted GEMMs and column-major for the q-contracted
    ones, from the *same* LDS bytes. That is the architectural bet of plan
    section 1 -- two LDS tiles, not gfx1201's four -- and it holds because
    `ds_read_b64_tr_b16` was designed for exactly this operand.
    """

    def read_row_half(self, tile_base, scope_name, half):
        """`A[row][d]` packs for rows `half * 32 .. + 32` of the tile.

        `K_LDS_TO_REG_N_STRIP_STRIDE` is 32 tokens along the intra-line token
        axis, so the two halves are the same columns of consecutive row blocks
        -- the relationship the forward's `_load_k_pair` has between the halves
        of a KV tile, hoisted into a parameter.

        One half at a time, rather than both, because the caller consumes each
        into its own accumulator and a whole half is `K_STEPS_QK * 4` VGPRs:
        64 at head_dim 128, against a register budget the two dK/dV
        accumulators have already spent 128 of.
        """
        traits = self.traits
        base = tile_base + self.stream_row_read_base + half * traits.K_LDS_TO_REG_N_STRIP_STRIDE
        return [
            _lds_pack_read(
                traits,
                self.lds_kv_base_ptr,
                base + _stream_ks_offset(traits, ks),
                scope_name,
                self.kv_mfma_pack_type,
            )
            for ks in range_constexpr(traits.K_STEPS_QK)
        ]

    def read_column_chunk(self, tile_base, scope_name, dc):
        """The four `A[d][row]` packs for output columns `dc * 32 .. + 32`.

        One chunk at a time, not the forward's whole `[4][D_CHUNKS]` table: the
        transposed operand feeds an accumulator that is already 64 VGPRs at
        head_dim 128, and reading all of it up front puts 128 more beside it
        for no benefit -- each pack is consumed by one MFMA and dies.
        """
        traits = self.traits
        lds_base = tile_base + self.stream_col_read_base
        pair = traits.V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE * traits.BF16_BYTES
        packs = []
        for k_substep in range_constexpr(4):
            imm_lo = _v_imm_lo(traits, dc, k_substep)
            a = dualwave._ds_read_tr_v4f16_imm(
                lds_base,
                imm_lo,
                lds_kv_base_idx=self.lds_kv_base_idx,
                v_lds_read_vec4_type=self.v_lds_read_vec4_type,
                scope_name=scope_name,
                scope_names=traits.LDS_SCOPE_NAMES,
            )
            b = dualwave._ds_read_tr_v4f16_imm(
                lds_base,
                imm_lo + pair,
                lds_kv_base_idx=self.lds_kv_base_idx,
                v_lds_read_vec4_type=self.v_lds_read_vec4_type,
                scope_name=scope_name,
                scope_names=traits.LDS_SCOPE_NAMES,
            )
            packs.append(Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value())
        return packs


class BwdDkDvGemmHelper(ParityGemmHelper):
    """The four GEMMs, as two shapes.

    Both are `sum_k mfma(A_k, B_k, acc)`; they differ only in what indexes `k`
    and whether the accumulator is fresh. Naming them for the axis they
    contract rather than for the tensors keeps `S` and `dP` -- and `dV` and
    `dK` -- one piece of code each instead of two, which contract section 4
    asks for and which is what stops a transcription drifting.
    """

    def contract_d(self, a_packs, b_packs):
        """`S` or `dP`: reduce over the head dim into a fresh accumulator."""
        acc = self.c_zero_v16f32
        for ks in range_constexpr(self.traits.K_STEPS_QK):
            acc = dualwave._mfma_acc(a_packs[ks], b_packs[ks], acc, self.mma_atom, self.mfma_acc_vec_type)
        return acc

    def contract_q(self, a_packs, b_packs, acc):
        """`dV` or `dK`: reduce over the tile's 64 q rows into a carried accumulator.

        Four k-substeps of 16 rows. `a_packs[step]` and `b_packs[step]` must
        carry the *same* permutation of the q axis within a substep, and they
        do: the transpose read's element order is `[0,1,2,3,8,9,10,11]` and
        `_pack_p_v8_slices` slices the accumulator in exactly that order. The
        forward relies on the identical coincidence between its V read and its
        P packing, one axis over.
        """
        for step in range_constexpr(4):
            acc = dualwave._mfma_acc(a_packs[step], b_packs[step], acc, self.mma_atom, self.mfma_acc_vec_type)
        return acc


class BwdDkDvSoftmaxHelper(ParitySoftmaxHelper):
    """`P` and `dS` from the two score accumulators, plus their row constants.

    Nothing online here: the forward already ran the softmax and its `LSE` is
    the whole of the state this needs, which is why the backward has no running
    max, no rescale and no `l`. What it does have that the forward does not is
    a **per-element** row index -- the accumulator's 16 f32 are 16 different q
    rows of one kv row, where the forward's are 16 kv columns of one q row -- so
    `lse` and `delta` arrive as 16 registers rather than one.
    """

    def load_row_values(self, rsrc, tile_base, scale):
        """`(lo, hi)` lists of 16 f32, one per accumulator element, times `scale`.

        Four `buffer_load_dwordx4` per half. The row an element holds is
        `8 * (r // 4) + 4 * (lane // 32) + (r % 4)`, so the four `r % 4` values
        of a group are four *contiguous* rows -- which is why this is 8 loads
        and not 32. `_ROW_RUNS` is where that grouping is stated.
        """
        out = []
        for half in range_constexpr(2):
            values = [None] * 16
            row_base = fx.Int32(tile_base + fx.Index(half * 32)) + fx.Int32(self.lane_div_32) * fx.Int32(4)
            for elem0, col_off, width in _ROW_RUNS:
                span = buffer_ops.buffer_load(
                    rsrc,
                    as_mlir_value(row_base + fx.Int32(col_off)),
                    vec_width=width,
                    dtype=fx.Float32,
                )
                vec = Vec(span, (width,), fx.Float32)
                for j in range_constexpr(width):
                    values[elem0 + j] = dualwave._fmul(vec[j], scale, self.fm_fast)
            out.append(values)
        return out

    def probabilities(self, v_s, neg_lse2):
        """`P = exp2(qk_scale * S - log2(e) * LSE)`, element by element.

        Q is **not** pre-scaled, which is the one place this departs from the
        forward. There it is free -- Q is loaded once in the prologue -- while
        here Q streams, so pre-scaling would cost a multiply and a bf16
        rounding per element per tile. Folding the scale into the subtraction
        that has to happen anyway costs one FMA and rounds once less.
        """
        lo, hi = dualwave._score_pair_to_lists(v_s)
        scale = self.ctx_ref.c_sm_scale_log2e
        out = []
        for half, values in ((0, lo), (1, hi)):
            row = neg_lse2[half]
            out.append(
                [
                    dualwave.rocdl.exp2(
                        T.f32,
                        as_mlir_value(
                            dualwave._fadd(dualwave._fmul(values[r], scale, self.fm_fast), row[r], self.fm_fast)
                        ),
                    )
                    for r in range_constexpr(16)
                ]
            )
        return out

    def dscores(self, p_lists, v_dp, delta):
        """`dS = P * (dP - delta)`.

        `dP` is deliberately unscaled: it is `dO . V^T` exactly, and `sm_scale`
        belongs to `dK` alone -- `dS` is the gradient with respect to the
        *scaled* logits, which is what `P` was built from. Applying the scale
        here instead would be a plausible, finite, wrong answer of the kind the
        joint autograd check in contract section 6 exists to catch.
        """
        lo, hi = dualwave._score_pair_to_lists(v_dp)
        out = []
        for half, values in ((0, lo), (1, hi)):
            row = delta[half]
            out.append(
                [
                    dualwave._fmul(
                        p_lists[half][r],
                        dualwave._fsub(values[r], row[r], self.fm_fast),
                        self.fm_fast,
                    )
                    for r in range_constexpr(16)
                ]
            )
        return out

    def pack_operand(self, lists):
        """The four bf16 B-operand packs for one q tile, in k-substep order.

        `_pack_p_v8_slices` is the forward's, and the flattening below is its
        `pv_step_k` dispatch (`lo[step]` under 2, else `hi[step - 2]`) written
        out once instead of at every use.
        """
        lo_packs, hi_packs = dualwave._pack_p_v8_slices(
            self.traits,
            (lists[0], lists[1]),
            lambda vals: dualwave._bf16_trunc_pack_v8(self.traits, vals, elem_dtype=self.elem_dtype),
        )
        return [lo_packs[0], lo_packs[1], hi_packs[0], hi_packs[1]]


class BwdDkDvStoreHelper(dualwave.DualwaveStoreHelper):
    """dK and dV, through the forward's O store path.

    A dK/dV accumulator is `[d][kv_row]` -- the same shape as the forward's
    `O` accumulator, one axis renamed -- so the 128-bit packing, the
    `permlane32_swap` half-wave exchange and the store are reused verbatim.
    Only the descriptor and the row stride are parameters, because two output
    tensors go through one helper.
    """

    def store_accs(self, accs, div, row, stride_seq):
        traits = self.traits
        base = row * stride_seq + self.lane_div_32 * fx.Index(8)
        for dc in range_constexpr(traits.D_CHUNKS):
            for g in range_constexpr(2):
                dualwave._buffer_store_128(
                    dualwave._packed_o_128_vec(traits, accs, dc, g, self.lane_div_32, self.elem_dtype),
                    base + fx.Index(dc * traits.D_CHUNK + 2 * g * 8),
                    _o_store_reg_128=self.o_store_reg_128,
                    _store_atom_128=self.store_atom_128,
                    o_div=div,
                )


class BwdDkDvTileBody(dualwave.DualwaveKernelContext):
    """One streamed q tile: four GEMMs, one softmax, two barriers.

    A helper object rather than a nested `def`, for the reason
    `fmha_wide_gfx950.make_wide_body` gives: the loop this is called from uses
    the `range(..., init=[...])` / `yield` protocol, which only exists after
    the AST rewrite, so the body has to stay inside the traced function while
    the code it calls stays out of it.
    """

    def __init__(self, ctx, *, stream, reader, gemm, softmax):
        super().__init__(ctx)
        self.stream = stream
        self.reader = reader
        self.gemm = gemm
        self.softmax = softmax

    def run(self, tile_idx, buf_id, prefetch_idx, v_k, v_v, dv, dk):
        """Accumulate this tile into `(dv, dk)` and prefetch `prefetch_idx`.

        Barrier discipline, and both are needed:

        - the **top** barrier publishes this buffer's DMA, which every wave has
          just waited for;
        - the **bottom** one says every wave has finished reading it, which is
          what makes the prefetch below safe to overwrite it with.

        The prefetch therefore lands one tile ahead and overlaps the *next*
        tile's compute, not this one's. Two buffers cannot do better: the other
        buffer already holds the tile after this one.
        """
        traits = self.traits
        ctx = self.ctx_ref
        q_base = ctx.q_buf_base(buf_id)
        do_base = ctx.do_buf_base(buf_id)
        q_scope = dualwave._dualwave_lds_scope("k", buf_id)
        do_scope = dualwave._dualwave_lds_scope("v", buf_id)
        tile_base = tile_idx * fx.Index(traits.BLOCK_Q)

        # `vmcnt` retires in issue order, so leaving exactly the next tile's
        # groups outstanding retires this one and nothing else. The forward's
        # `NUM_DMA_K + NUM_DMA_V` idiom, with the two tensors being Q and dO.
        _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
        _sched_barrier(0)
        _s_barrier()

        # -- GEMM 1 and 2. D is the reduction axis for both, and the resident
        #    operands are loop-invariant, so each is one burst.
        s_lo = self.gemm.contract_d(self.reader.read_row_half(q_base, q_scope, 0), v_k)
        s_hi = self.gemm.contract_d(self.reader.read_row_half(q_base, q_scope, 1), v_k)
        dp_lo = self.gemm.contract_d(self.reader.read_row_half(do_base, do_scope, 0), v_v)
        dp_hi = self.gemm.contract_d(self.reader.read_row_half(do_base, do_scope, 1), v_v)

        neg_lse2 = self.softmax.load_row_values(ctx.lse_rsrc, tile_base, ctx.c_neg_log2e)
        delta = self.softmax.load_row_values(ctx.delta_rsrc, tile_base, ctx.c_one_f)
        p_lists = self.softmax.probabilities((s_lo, s_hi), neg_lse2)
        ds_lists = self.softmax.dscores(p_lists, (dp_lo, dp_hi), delta)
        p_packs = self.softmax.pack_operand(p_lists)
        ds_packs = self.softmax.pack_operand(ds_lists)

        # -- GEMM 3 and 4. Q is the reduction axis, so the transposed operand
        #    is read one output chunk at a time and dies with its MFMAs.
        for dc in range_constexpr(traits.D_CHUNKS):
            dv[dc] = self.gemm.contract_q(self.reader.read_column_chunk(do_base, do_scope, dc), p_packs, dv[dc])
        for dc in range_constexpr(traits.D_CHUNKS):
            dk[dc] = self.gemm.contract_q(self.reader.read_column_chunk(q_base, q_scope, dc), ds_packs, dk[dc])

        _s_waitcnt(traits.LGKMCNT_0_ONLY)
        _sched_barrier(0)
        _s_barrier()
        # Reading past the last tile is harmless and is what keeps this
        # branch-free: the descriptor bounds it, the zeros land in a buffer
        # nothing consumes, and the alternative is an `scf.if` around a DMA.
        self.stream.stage_q_tile(prefetch_idx, buf_id)
        self.stream.stage_do_tile(prefetch_idx, buf_id)
        return dv, dk


def build_fmha_bwd_dkdv_gfx950_module_primary(meta, knobs):
    """Build the dK/dV kernel for a resolved (meta, knobs) pair."""
    if knobs.traits is None:
        raise ValueError("knobs must be resolved: call `bwd_dkdv_knobs(arch, ...).resolve(meta)` first")
    traits = knobs.traits
    BLOCK_DMODEL = knobs.block_dmodel
    STRIDES_CONSTEXPR = knobs.strides_constexpr
    BUILD_SM_SCALE = meta.sm_scale

    _cache_tag = (
        traits.cache_tag,
        BLOCK_DMODEL,
        STRIDES_CONSTEXPR,
        BUILD_SM_SCALE,
        (knobs.num_waves, knobs.block_kv, knobs.block_q, knobs.head_dim_granule),
        knobs.num_stream_buffers,
    )

    _lds_elem_dtype = dualwave.dtype_to_elem_type(traits.DTYPE_STR)

    @fx.struct
    class SharedStorage:
        # Q and dO tiles, interleaved per buffer: `[Q(0), dO(0), Q(1), dO(1)]`.
        stream: fx.Array[_lds_elem_dtype, traits.LDS_STREAM_TOTAL_SIZE, 16]

    @flyc.kernel(known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def fmha_bwd_dkdv_gfx950_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        LSE: fx.Tensor,
        Delta: fx.Tensor,
        Bias: fx.Tensor,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        philox_seed_ptr: fx.Pointer,
        philox_offset1: fx.Pointer,
        philox_offset2: fx.Int64,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        sm_scale: fx.Float32,
        stride_q_batch: fx.Int64,
        stride_q_head: fx.Int64,
        stride_q_seq: fx.Int64,
        stride_k_batch: fx.Int64,
        stride_k_head: fx.Int64,
        stride_k_seq: fx.Int64,
        stride_v_batch: fx.Int64,
        stride_v_head: fx.Int64,
        stride_v_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stride_dk_batch: fx.Int64,
        stride_dk_head: fx.Int64,
        stride_dk_seq: fx.Int64,
        stride_dv_batch: fx.Int64,
        stride_dv_head: fx.Int64,
        stride_dv_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq: fx.Int64,
    ):
        ctx = BwdDkDvKernelContext(
            traits,
            strides=(
                stride_q_batch,
                stride_q_head,
                stride_q_seq,
                stride_k_batch,
                stride_k_head,
                stride_k_seq,
                stride_v_batch,
                stride_v_head,
                stride_v_seq,
                stride_do_batch,
                stride_do_head,
                stride_do_seq,
                stride_dk_batch,
                stride_dk_head,
                stride_dk_seq,
                stride_dv_batch,
                stride_dv_head,
                stride_dv_seq,
            ),
            sm_scale=sm_scale,
            num_head_q=num_head_q,
            num_head_k=num_head_k,
            hdim_qk=hdim_qk,
            hdim_vo=hdim_vo,
            window_left=window_left,
            window_right=window_right,
            seqinfo=(seqinfo_q0, seqinfo_q1, seqinfo_k0, seqinfo_k1),
            varlen_bits=varlen_bits,
            num_seqlens=num_seqlens,
            Bias=Bias,
            bias_strides=(stride_b_batch, stride_b_head, stride_b_seq),
            philox=(philox_seed_ptr, philox_offset1, philox_offset2, None, None),
            idropout_p=idropout_p,
            dropout_scale=dropout_scale,
            Q=Q,
            K=K,
            V=V,
            O=DK,
            DO=DO,
            DK=DK,
            DV=DV,
            Delta=Delta,
            LSE=LSE,
            DebugCounts=DK,
            CuSeqQ=Q,
            CuSeqKv=Q,
            BlockTable=Q,
            seq_len=max_seqlen_q,
            seq_len_kv=max_seqlen_k,
            stride_q_n=stride_q_seq,
            stride_kv_n=stride_k_seq,
            head_dim_runtime=hdim_qk,
            block_table_stride=0,
        )
        ctx.init_types_and_constants()
        ctx.init_runtime_indices()
        ctx.init_lds(SharedStorage)
        ctx.init_thread_mapping()
        ctx.init_sequence_lengths()
        ctx.init_descriptors()
        ctx.init_workspace()
        ctx.init_atoms_and_lds_ptrs()
        ctx.init_dma_thread_offsets()
        ctx.init_tile_bounds()
        ctx.init_lds_read_bases()
        ctx.init_dma_m0_tables()
        ctx.init_kv_row()
        # `-log2(e)` and `1.0`, so `load_row_values` is one function for both
        # row tensors: LSE has to cross into the base-2 domain the exponent
        # lives in, and delta is already in dP's units.
        ctx.c_neg_log2e = fx.Float32(-dualwave._LOG2E)
        ctx.c_one_f = fx.Float32(1.0)

        stream = BwdDkDvStreamLoader(ctx)
        reader = BwdDkDvStreamReader(ctx)
        resident = BwdDkDvResidentLoader(ctx)
        gemm = BwdDkDvGemmHelper(ctx)
        softmax = BwdDkDvSoftmaxHelper(ctx)
        store = BwdDkDvStoreHelper(ctx)
        tile = BwdDkDvTileBody(ctx, stream=stream, reader=reader, gemm=gemm, softmax=softmax)

        zero_acc = ctx.c_zero_v16f32
        t_end = ctx.split_t_end

        @flyc.jit
        def _dkdv_body():
            """K/V resident, Q/dO streaming, two tiles per iteration."""
            v_k = resident.load_rows(ctx.k_res_div, ctx.k_res_elem_base, ctx.stride_k_seq_v, ctx.kv_row_in_block)
            v_v = resident.load_rows(ctx.v_res_div, ctx.v_res_elem_base, ctx.stride_v_seq_v, ctx.kv_row_in_block)

            # Prime both buffers. From here every tile's DMA is issued by the
            # body two tiles earlier, so this is the only staging whose latency
            # is not covered by compute.
            stream.stage_q_tile(fx.Index(0), 0)
            stream.stage_do_tile(fx.Index(0), 0)
            stream.stage_q_tile(fx.Index(1), 1)
            stream.stage_do_tile(fx.Index(1), 1)

            init_args = []
            for _ in range_constexpr(2 * traits.D_CHUNKS):
                init_args.append(zero_acc)
            loop_results = init_args

            for j, loop_args in range(fx.Index(0), t_end, fx.Index(2), init=init_args):
                dv = [loop_args[i] for i in range_constexpr(traits.D_CHUNKS)]
                dk = [loop_args[traits.D_CHUNKS + i] for i in range_constexpr(traits.D_CHUNKS)]
                dv, dk = tile.run(j, 0, j + fx.Index(2), v_k, v_v, dv, dk)
                dv, dk = tile.run(j + fx.Index(1), 1, j + fx.Index(3), v_k, v_v, dv, dk)
                loop_results = yield dv + dk

            dv = [loop_results[i] for i in range_constexpr(traits.D_CHUNKS)]
            dk = [loop_results[traits.D_CHUNKS + i] for i in range_constexpr(traits.D_CHUNKS)]

            # `dS` is the gradient of the *scaled* logits, so `sm_scale` belongs
            # to `dK` and to nothing else. Once at the end rather than per tile:
            # a linear factor commutes with the accumulation.
            dualwave._scale_o_accs(dk, ctx.c_sm_scale, traits, ctx.fm_fast)
            store.store_accs(dv, ctx.dv_div, ctx.kv_row, ctx.stride_dv_seq_v)
            store.store_accs(dk, ctx.dk_div, ctx.kv_row, ctx.stride_dk_seq_v)

        _dkdv_body()

    @flyc.jit
    def launch_fmha_bwd_dkdv_gfx950(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        LSE: fx.Tensor,
        Delta: fx.Tensor,
        Bias: fx.Tensor,
        batch_size: fx.Int32,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        philox_seed_ptr: fx.Pointer,
        philox_offset1: fx.Pointer,
        philox_offset2: fx.Int64,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        sm_scale: fx.Float32,
        stride_q_batch: fx.Int64,
        stride_q_head: fx.Int64,
        stride_q_seq: fx.Int64,
        stride_k_batch: fx.Int64,
        stride_k_head: fx.Int64,
        stride_k_seq: fx.Int64,
        stride_v_batch: fx.Int64,
        stride_v_head: fx.Int64,
        stride_v_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stride_dk_batch: fx.Int64,
        stride_dk_head: fx.Int64,
        stride_dk_seq: fx.Int64,
        stride_dv_batch: fx.Int64,
        stride_dv_head: fx.Int64,
        stride_dv_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = _cache_tag
        num_kv_blocks = (fx.Index(max_seqlen_k) + fx.Index(traits.BLOCK_KV - 1)) // fx.Index(traits.BLOCK_KV)
        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(traits.DAZ)
            else None
        )
        fmha_bwd_dkdv_gfx950_kernel(
            Q,
            K,
            V,
            DO,
            DK,
            DV,
            LSE,
            Delta,
            Bias,
            seqinfo_q0,
            seqinfo_q1,
            seqinfo_k0,
            seqinfo_k1,
            varlen_bits,
            num_seqlens,
            max_seqlen_q,
            max_seqlen_k,
            window_left,
            window_right,
            philox_seed_ptr,
            philox_offset1,
            philox_offset2,
            idropout_p,
            dropout_scale,
            num_head_q,
            num_head_k,
            hdim_qk,
            hdim_vo,
            sm_scale,
            stride_q_batch,
            stride_q_head,
            stride_q_seq,
            stride_k_batch,
            stride_k_head,
            stride_k_seq,
            stride_v_batch,
            stride_v_head,
            stride_v_seq,
            stride_do_batch,
            stride_do_head,
            stride_do_seq,
            stride_dk_batch,
            stride_dk_head,
            stride_dk_seq,
            stride_dv_batch,
            stride_dv_head,
            stride_dv_seq,
            stride_b_batch,
            stride_b_head,
            stride_b_seq,
            value_attrs={
                "rocdl.waves_per_eu": traits.WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{traits.BLOCK_SIZE},{traits.BLOCK_SIZE}",
                "passthrough": passthrough_entries,
            },
        ).launch(
            # Head fastest, matching the forward: on MI355X's 8 XCDs the grid
            # axis order is an L2-locality lever, and the forward measured 7%
            # for getting it this way round.
            grid=(num_head_q, num_kv_blocks, fx.Index(batch_size)),
            block=(traits.BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    def _args(
        Q,
        K,
        V,
        DO,
        DK,
        DV,
        LSE,
        Delta,
        batch_size,
        seqlen_q,
        seqlen_k=None,
        scale=None,
        stream=None,
    ):
        """Every kernel argument but the stream, in launch order.

        One place that turns tensors into the wire format, so `_launch` and
        `_compile` cannot drift apart -- the same shape, and the same reason,
        as the forward's `_args`.
        """
        seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
        _ptrs, shape_meta, st = abi.prep_tensors(
            [("Q", Q), ("K", K), ("V", V), ("DO", DO), ("DK", DK), ("DV", DV)],
            q_heads=("DO",),
            k_heads=("DK", "DV"),
        )
        del _ptrs  # gfx950 addresses through buffer descriptors, so it wants the tensors
        num_head_q, num_head_k, hdim_qk, hdim_vo = shape_meta
        if num_head_q != num_head_k:
            raise ValueError(
                f"num_heads_q {num_head_q} != num_heads_k {num_head_k}: dK/dV under GQA must be summed "
                "over the q heads sharing a kv head, which this kernel does not do. See "
                "`BwdDkDvKnobs._checked_scope`."
            )
        if hdim_qk != BLOCK_DMODEL or hdim_vo != BLOCK_DMODEL:
            raise ValueError(
                f"this build serves head_dim {BLOCK_DMODEL} exactly, got hdim_qk {hdim_qk} and hdim_vo "
                f"{hdim_vo}. Padded heads and asymmetric hdim are B3."
            )
        for name, t in (("DK", DK), ("DV", DV)):
            if t.dtype != Q.dtype:
                raise ValueError(f"{name} must have Q's dtype ({Q.dtype}), got {t.dtype}")
        # `row_tensor_arg` checks both the same way, which is what makes it safe
        # for the kernel to address them with one offset computation. Its return
        # is discarded: this kernel takes them as tensors, because it builds
        # buffer descriptors rather than dereferencing a raw pointer.
        abi.row_tensor_arg(LSE, "logsumexp", num_head_q, seqlen_q, None)
        abi.row_tensor_arg(Delta, "delta", num_head_q, seqlen_q, None)
        if int(batch_size) != int(Q.shape[0]):
            raise ValueError(f"batch_size={int(batch_size)} but Q.size(0)={int(Q.shape[0])}")
        if LSE.shape[0] != int(batch_size) * num_head_q:
            raise ValueError(f"logsumexp must be ({int(batch_size) * num_head_q}, {seqlen_q}); got {tuple(LSE.shape)}")
        if Delta.shape != LSE.shape:
            raise ValueError(f"delta must have logsumexp's shape {tuple(LSE.shape)}, got {tuple(Delta.shape)}")

        return (
            Q,
            K,
            V,
            DO,
            DK,
            DV,
            LSE,
            Delta,
            DK,  # Bias placeholder: a build without bias never reads the slot
            int(batch_size),
            abi.NULL_PTR,
            abi.NULL_PTR,
            abi.NULL_PTR,
            abi.NULL_PTR,
            0,  # varlen_bits
            0,  # num_seqlens
            int(seqlen_q),
            int(seqlen_k),
            0,  # window_left
            0,  # window_right
            abi.NULL_PTR,
            abi.NULL_PTR,
            0,  # philox_offset2
            0,  # idropout_p
            1.0,  # dropout_scale
            num_head_q,
            num_head_k,
            hdim_qk,
            hdim_vo,
            abi.resolve_scale(Q, scale if scale is not None else BUILD_SM_SCALE, False, 1.0 / (BLOCK_DMODEL**0.5)),
            *st,
            0,
            0,
            0,  # bias strides
        ), stream

    def _launch(*args, **kwargs):
        packed, stream = _args(*args, **kwargs)
        with CompilationContext.compile_hints(_COMPILE_HINTS):
            return abi.run_compiled(
                _COMPILED,
                launch_fmha_bwd_dkdv_gfx950,
                *packed,
                stream if stream is not None else fx.Stream(None),
            )

    def _compile(*args, **kwargs):
        packed, stream = _args(*args, **kwargs)
        with CompilationContext.compile_hints(_COMPILE_HINTS):
            return flyc.compile(launch_fmha_bwd_dkdv_gfx950, *packed, fx.Stream(stream))

    _launch.compile = _compile
    _launch.traits = traits
    _launch.knobs = knobs
    return _launch


def build_fmha_bwd_dkdv_gfx950_module(arch="gfx950", **kwargs):
    """Keyword front end: name a problem, get the policy's schedule."""
    from dataclasses import fields as _fields

    meta_fields = {f.name for f in _fields(BwdDkDvInputMetadata)}
    meta = BwdDkDvInputMetadata(**{k: v for k, v in kwargs.items() if k in meta_fields})
    knob_kwargs = {k: v for k, v in kwargs.items() if k not in meta_fields}
    return build_fmha_bwd_dkdv_gfx950_module_primary(meta, bwd_dkdv_knobs(arch, **knob_kwargs).resolve(meta))
