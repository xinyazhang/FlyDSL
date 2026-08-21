# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""dK / dV for gfx950 -- AOTriton's `bwd_kernel_dk_dv` on the dualwave helpers.

B1 + B3 of `sdpa-bwd-plan-gfx950.md`: dense, non-causal, MHA, bf16, over the
forward's whole head_dim ladder with the 8xD input contract and padded heads.
The contract is `sdpa-bwd-contract-gfx950.md` and its B3 addendum; read those
first.

--- The whole design in one paragraph --------------------------------------

**dK/dV is the forward loop transposed.** K and V stay resident in registers
while Q and dO stream past through LDS, exactly mirroring the forward's
resident Q and streaming K/V. Every helper in `fmha_dualwave_gfx950.py` was
written against "a tensor, its bounds and where it lands" rather than against
K or V by name, so the roles swap by re-pointing four descriptors and nothing
else. The two tensor slots the staging machinery calls *K* and *V* carry **Q**
and **dO** here, and `BLOCK_M` / `BLOCK_N` / `VO_SHARDS` in the traits mean the
KV block, the Q tile and the dK/dV D-axis split -- see
`fmha_tuning_bwd_dkdv_gfx950`'s docstring for the full map.

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

--- What binds at the wide rungs, and what does not -------------------------

**LDS never forces `D_STAGES` here.** A staged slot is `68 * head_dim`
elements, so two tensors single-buffered are `272 * head_dim` bytes -- 139264
at head_dim 512, inside the 163840 cap. The second buffer is the only thing
LDS ever costs, and `_with_buffers` spends it while it can afford to.

**Registers do bind**, because a wave holds two accumulators of `d/2` VGPRs
each on top of `d/4` of resident K and `d/4` of V. Two levers answer it, in
the addendum's order: `(num_waves, waves_per_eu)` as a pair, which is free and
decided head_dim 128; and then `DKV_SHARDS`, which divides the accumulator
pair across waves that write disjoint D columns -- no cross-wave reduction, no
extra barrier, at the price of every shard recomputing S and dP.

--- Padded heads: which operand is masked, and why it is the cheap one ------

The forward masks Q once in its prologue and K on the hot path, and pays for
the second one. Here the roles are swapped, so **the cheap masks are K and V**
-- resident, masked once, every step. Q and dO are the hot ones and are masked
only on the k-steps that can contain pad, which `HDIM_QK_FLOOR` bounds to two
on the 32-spaced ladder.

Only the two *d-contracted* GEMMs need masked operands at all. In `dV` and
`dK` the head dim is the **output** axis, so a pad column of dO can only reach
a pad column of dV, and the store suppresses it by address.

--- Why nothing else is masked ----------------------------------------------

Dense and non-causal, so the only edge is the ragged tail, and the buffer
descriptors already answer it. Q and dO are bounded at `seqlen_q` rows, so a
staged row past the end reads **zero**; K and V at `seqlen_kv`, and dK/dV
stores past it are dropped by the same bound. A padding q row therefore gets
`S = 0` and `LSE = 0` (also out of its buffer), hence `P = exp2(0) = 1` -- and
contributes `1 * dO = 0` to dV and `dS * Q = 0` to dK. Finite, exact, and no
`scf.if` anywhere near a transpose read, which contract section 3 warns is
undefined under a divergent EXEC.

--- Phase status -----------------------------------------------------------

Causal and windows (B4), varlen (B5), dropout and bias (B6) and GQA are **not
implemented**. The ABI carries their argument slots so the wire format does not
move when they arrive, and `BwdDkDvKnobs.resolve` refuses a build that would
need one; there is no half-implemented arm anywhere below.
"""

import weakref

import fmha_abi_gfx1201 as abi
from fmha_common_gfx1201 import MaskedAxis
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


def _masked_ks_steps(traits, hdim_floor):
    """The k-steps whose D columns can reach past the real head dim.

    A build serves `(HDIM_QK_FLOOR, BLOCK_DMODEL]` and the dispatcher enforces
    it, so a step whose columns all lie at or below the floor is reading real
    data whatever the runtime extent. On the 32-spaced ladder consecutive rungs
    are 32 apart and `K_STEP_QK` is 16, so exactly two steps survive at every
    width -- 2 of 32 at head_dim 512 rather than 32.

    The forward found this to be the whole cost of a padded head: masking every
    step ran 27-54% below the rung's native rate, near-independently of how
    much pad there was, which is what identified the masking rather than the
    wasted MFMA columns.
    """
    return [ks for ks in range(traits.K_STEPS_QK) if (ks + 1) * traits.K_STEP_QK > hdim_floor]


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

    Six things move, and they are all descriptors, bounds or indices:

    - `strides` carries **18** slots, not 12: Q, K, V, dO, dK, dV. The first
      twelve go to the base class, which means its `stride_o_*` names hold
      **dO's** strides. Nothing below reads those names; `stride_do_*` is
      spelled out instead.
    - The staging machinery's K slot carries Q and its V slot carries dO.
    - Both staged tiles are V-shaped, so the LDS bases and the row-major read
      base are recomputed against `STREAM_LINE_STRIDE`.
    - The tile loop walks **q** tiles, so `init_tile_bounds` counts them from
      `seqlen_q`.
    - `wave_id` splits into a KV row block and a D-axis shard.
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

    def dma_m0_table(self, base_fn, count):
        """One row per *stream buffer*, not the base class's fixed two.

        At `NUM_STREAM_BUFFERS == 1` the second row would address LDS past the
        allocation. It is never issued, so it folds away -- but a table that
        contains an illegal address is a thing a later reader has to reason
        about, and building the right number is free.
        """
        return tuple(tuple(base_fn(buf, d) for d in range(count)) for buf in range(self.traits.NUM_STREAM_BUFFERS))

    def init_lds_read_bases(self):
        super().init_lds_read_bases()
        traits = self.traits
        self.stream_row_read_base = _stream_row_read_base(traits, self.lane_mod_32, self.lane_div_32)
        # The column-major base is the forward's V one *unchanged*, which is
        # the whole of contract section 3: the tile is V-shaped and the read is
        # the same instruction sequence, so the validated lane map carries over
        # rather than being re-derived.
        col_base = self.v_lds_read_base_per_lane
        if const_expr(traits.DKV_SHARDS > 1):
            # The shard's D offset folds into the base rather than being added
            # per read, because `_v_imm_lo` needs a *compile-time* chunk index
            # and `shard_id` is `wave_id % SHARDS`. `_v_dc_offset` splits as
            # `_v_dc_offset(a + i) = (a // PER_BAND) * PAIR + _v_dc_offset(i)`
            # whenever `a` is a multiple of `D_CHUNKS_PER_BAND`, which
            # `make_traits` enforces for a shard's first chunk. The forward's
            # `WideKvLdsToVgprLoader.load_v_shard` folds it the same way.
            per_band = traits.D_CHUNKS_PER_BAND
            col_base = col_base + self.dkv_shard_id * (
                (traits.D_CHUNKS_PER_SHARD // per_band) * traits.V_LDS_TO_REG_DCHUNK_PAIR_STRIDE
            )
        self.stream_col_read_base = col_base

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
        traits = self.traits
        # Aliases, not new values: `grid.y` selects a KV block here and the
        # base class's `q_*` names already hold exactly that arithmetic.
        self.kv_block_idx = self.q_block_idx
        self.kv_start = self.q_start
        if const_expr(traits.DKV_SHARDS > 1):
            # With `SHARDS` waves per KV row block, `wave_id` no longer names a
            # row block -- `wave_id // SHARDS` does, and the remainder picks
            # which D columns of dK/dV the wave owns. The base class set
            # `wave_q_offset` from the raw wave id; this is the same correction
            # `WideKernelContext` makes for `VO_SHARDS`.
            self.dkv_shard_id = self.wave_id % traits.DKV_SHARDS
            self.wave_q_offset = (self.wave_id // traits.DKV_SHARDS) * traits.ROWS_PER_WAVE
        else:
            self.dkv_shard_id = fx.Index(0)
        self.wave_kv_offset = self.wave_q_offset
        # Global first D column of this wave's accumulators. Runtime, because
        # the shard comes from `wave_id`; zero and folded away when unsharded.
        self.dkv_col_base = self.dkv_shard_id * fx.Index(traits.D_CHUNKS_PER_SHARD * traits.D_CHUNK)

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
        # First element past each output's descriptor. A store redirected here
        # is dropped by the hardware bound, which is how the padded-head D tail
        # is suppressed without a branch.
        self.dk_oob_off = self.seqlen_kv_v * self.stride_dk_seq_v
        self.dv_oob_off = self.seqlen_kv_v * self.stride_dv_seq_v

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

    # -- bounds ------------------------------------------------------------

    def init_tile_bounds(self, **kwargs):
        """Count **q** tiles, rounded up to an even number.

        The body consumes two tiles per iteration, so an odd count would leave
        the second half of the last iteration reading a tile that does not
        exist. Rounding up costs one dead tile rather than an epilogue: its
        rows are past `seqlen_q`, so Q and dO stage as zeros and it contributes
        exactly nothing (module docstring).
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
    per-`ks` packs rather than one concatenated vector: the forward
    concatenates so `_get_q_pack` can slice a loop-invariant register, and here
    the packs feed two different GEMMs and never need to be adjacent.

    **This is where a padded head is masked**, and it is the cheap side of the
    trade. K and V are read once per kernel, so masking every k-step here costs
    nothing measurable -- against the forward, which has to mask its streamed
    K on the hot path and measured 27-54% for it.
    """

    def load_rows(self, div, elem_base, stride_seq, row_in_block, hdim):
        traits = self.traits
        cols = None
        if const_expr(self.PADDED_HEAD):
            # Loop-invariant and dead immediately after the prologue, so the
            # bitmask form is pure win here for the same reason it is in the
            # forward's Q loader.
            cols = MaskedAxis(fx.Index(hdim), elem_dtype=self.elem_dtype, bitmask=True)
        packs = []
        for ks in range_constexpr(traits.K_STEPS_QK):
            col = fx.Index(ks * traits.K_STEP_QK) + self.lane_div_32 * fx.Index(traits.MFMA_LANE_K)
            raw = dualwave._buffer_load_128(
                elem_base + row_in_block * stride_seq + col,
                _load_atom_128=self.load_atom_128,
                q_div=div,
                q_load_i32x4_type=self.q_load_i32x4_type,
            )
            pack = Vec(raw, (4,), fx.Int32).bitcast(self.elem_dtype).ir_value()
            if const_expr(self.PADDED_HEAD):
                pack = cols.discard(pack, col, traits.MFMA_LANE_K)
            packs.append(pack)
        return packs


class BwdDkDvStreamReader(dualwave.DualwaveKernelContext):
    """The two ways one staged tile is read back.

    Row-major for the d-contracted GEMMs and column-major for the q-contracted
    ones, from the *same* LDS bytes. That is the architectural bet of plan
    section 1 -- two LDS tiles, not gfx1201's four -- and it holds because
    `ds_read_b64_tr_b16` was designed for exactly this operand.
    """

    # Set by the kernel once the floor is known; see `_masked_ks_steps`.
    masked_steps = ()

    def read_row_pack(self, tile_base, scope_name, half, ks, hdim):
        """One `A[row][d]` pack for k-step `ks` of rows `half * 32 .. + 32`.

        `K_LDS_TO_REG_N_STRIP_STRIDE` is 32 tokens along the intra-line token
        axis, so the two halves are the same columns of consecutive row blocks
        -- the relationship the forward's `_load_k_pair` has between the halves
        of a KV tile, hoisted into a parameter.

        **One pack, not a whole half.** The caller feeds each straight into its
        MFMA, so nothing outlives its use; reading the half up front would put
        `K_STEPS_QK * 4` VGPRs live at once, which is 128 at head_dim 512
        beside accumulators that have already spent most of the file. Splitting
        the pair into halves was worth 2.2x at head_dim 128 before the wave
        count was touched at all, and this is the same move one step further.
        """
        traits = self.traits
        idx = (
            tile_base
            + self.stream_row_read_base
            + half * traits.K_LDS_TO_REG_N_STRIP_STRIDE
            + _stream_ks_offset(traits, ks)
        )
        pack = _lds_pack_read(traits, self.lds_kv_base_ptr, idx, scope_name, self.kv_mfma_pack_type)
        if const_expr(self.PADDED_HEAD) and ks in self.masked_steps:
            # The bitmask form ANDs one precomputed dword per pair instead of
            # selecting per element, but each mask stays live for the whole q
            # loop. Gated on how many steps actually survive the floor rather
            # than on `K_STEPS_QK`: the forward measured +21% in its 64-wide
            # tile and -43% in the 128-wide one, where 32 extra live registers
            # turned a spill-free build into 61 spills.
            width = traits.MFMA_LANE_K
            col = fx.Index(ks * traits.K_STEP_QK) + self.lane_div_32 * fx.Index(width)
            cols = MaskedAxis(
                fx.Index(hdim),
                elem_dtype=self.elem_dtype,
                bitmask=len(self.masked_steps) * (width // 2) <= 16,
            )
            pack = cols.discard(pack, col, width)
        return pack

    def read_column_chunk(self, tile_base, scope_name, dc):
        """The four `A[d][row]` packs for this shard's output chunk `dc`.

        `dc` is **shard-local**: the shard's first chunk is folded into
        `stream_col_read_base` (see `init_lds_read_bases`), because the
        immediate offset `_v_imm_lo` builds has to be a compile-time constant
        and the shard index is not.

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

    def contract_d(self, read_a, b_packs):
        """`S` or `dP`: reduce over the head dim into a fresh accumulator.

        `read_a` is a *callable*, not a list, so each A pack is read into the
        MFMA that consumes it rather than the whole set being gathered first.
        See `BwdDkDvStreamReader.read_row_pack`.
        """
        acc = self.c_zero_v16f32
        for ks in range_constexpr(self.traits.K_STEPS_QK):
            acc = dualwave._mfma_acc(read_a(ks), b_packs[ks], acc, self.mma_atom, self.mfma_acc_vec_type)
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

    def load_row_values(self, rsrc, tile_base, half, scale):
        """16 f32, one per accumulator element of `half`, times `scale`.

        Four `buffer_load_dwordx4`. The row an element holds is
        `8 * (r // 4) + 4 * (lane // 32) + (r % 4)`, so the four `r % 4` values
        of a group are four *contiguous* rows -- which is why this is 4 loads
        and not 16. `_ROW_RUNS` is where that grouping is stated.
        """
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
        return values

    def probabilities(self, v_s, neg_lse2):
        """`P = exp2(qk_scale * S - log2(e) * LSE)`, element by element.

        Q is **not** pre-scaled, which is the one place this departs from the
        forward, and B2 measured what it costs to get wrong: folding
        `sm_scale * log2e` into Q and rounding to bf16 puts the error in the
        exponent, taking the error ratio from 1.29 at `sm_scale = 0.05` to 10.9
        at 1.0. `O` is a normalised average and absorbs it; `dS` is not and does
        not. Here the scale rides the subtraction that had to happen anyway, so
        it is also free.
        """
        values = [Vec(v_s)[r] for r in range_constexpr(16)]
        scale = self.ctx_ref.c_sm_scale_log2e
        return [
            dualwave.rocdl.exp2(
                T.f32,
                as_mlir_value(
                    dualwave._fadd(dualwave._fmul(values[r], scale, self.fm_fast), neg_lse2[r], self.fm_fast)
                ),
            )
            for r in range_constexpr(16)
        ]

    def dscores(self, p_list, v_dp, delta):
        """`dS = P * (dP - delta)`.

        `dP` is deliberately unscaled: it is `dO . V^T` exactly, and `sm_scale`
        belongs to `dK` alone -- `dS` is the gradient with respect to the
        *scaled* logits, which is what `P` was built from. Applying the scale
        here instead would be a plausible, finite, wrong answer of the kind the
        joint autograd check in contract section 6 exists to catch.
        """
        values = [Vec(v_dp)[r] for r in range_constexpr(16)]
        return [
            dualwave._fmul(p_list[r], dualwave._fsub(values[r], delta[r], self.fm_fast), self.fm_fast)
            for r in range_constexpr(16)
        ]

    def pack_half(self, values):
        """The two bf16 B-operand packs for one 32-row half of a q tile.

        `_pack_p_v8_slices`' inner slicing, done a half at a time. The forward
        packs both halves together because its two accumulators are live
        together anyway; here they deliberately are not (see
        `BwdDkDvTileBody.run`), and a whole-tile packer would force them to be.
        """
        return [
            dualwave._bf16_trunc_pack_v8(
                self.traits, [values[p * 8 + s] for s in range_constexpr(8)], elem_dtype=self.elem_dtype
            )
            for p in range_constexpr(self.traits.PV_K_STEPS)
        ]


class BwdDkDvStoreHelper(dualwave.DualwaveStoreHelper):
    """dK and dV, through the forward's O store path.

    A dK/dV accumulator is `[d][kv_row]` -- the same shape as the forward's
    `O` accumulator, one axis renamed -- so the 128-bit packing, the
    `permlane32_swap` half-wave exchange and the store are reused verbatim.
    The descriptor, the row stride, the shard's column origin and the head
    extent are parameters, because two output tensors go through one helper.
    """

    def store_accs(self, accs, div, row, stride_seq, hdim, oob_off):
        """Store this wave's `D_CHUNKS_PER_SHARD` chunks of one output.

        **Address the chunk from its global column, not its local one.** Under
        sharding `dc` restarts at 0 while the columns it addresses do not, and
        the forward's wide body records what happens if the padded-head
        suppression is derived from the local index: the descriptor spans the
        whole tensor rather than one row, so a store past the row pitch lands
        in the *next row* and corrupts it -- head_dim 300 came out at 0.58
        absolute error, finite and with no fault.
        """
        traits = self.traits
        base = row * stride_seq + self.dkv_col_base + self.lane_div_32 * fx.Index(8)
        cols = MaskedAxis(fx.Index(hdim)) if const_expr(self.PADDED_HEAD) else None
        for dc in range_constexpr(traits.D_CHUNKS_PER_SHARD):
            for g in range_constexpr(2):
                off = base + fx.Index(dc * traits.D_CHUNK + 2 * g * 8)
                if const_expr(self.PADDED_HEAD):
                    # A 128-bit store is all-or-nothing, so a chunk straddling
                    # `hdim` writes into the caller's own 8-element pad, which
                    # the input contract permits. Only a chunk *starting* at or
                    # past `hdim` must be dropped, and pushing its offset past
                    # `num_records` is what drops it -- one select on a
                    # lane-varying value instead of an `scf.if` around a store.
                    col = self.dkv_col_base + fx.Index(dc * traits.D_CHUNK + 2 * g * 8) + self.lane_div_32 * fx.Index(8)
                    off = fx.Index(cols.valid(col).select(fx.Index(off), oob_off))
                dualwave._buffer_store_128(
                    dualwave._packed_o_128_vec(traits, accs, dc, g, self.lane_div_32, self.elem_dtype),
                    off,
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

    def __init__(self, ctx, *, stream, reader, gemm, softmax, hdim_qk, hdim_vo, tight_registers):
        super().__init__(ctx)
        self.stream = stream
        self.reader = reader
        self.gemm = gemm
        self.softmax = softmax
        self.hdim_qk = hdim_qk
        self.hdim_vo = hdim_vo
        self.tight_registers = tight_registers
        self.half_groups = ((0,), (1,)) if tight_registers else ((0, 1),)

    def _row_reader(self, base, scope, half, hdim):
        """A `ks -> pack` closure over one 32-row half of a staged tile.

        **The one line where `TIGHT_REGISTERS` changes what is emitted.** The
        tight arm reads each pack inside the MFMA that consumes it, so one is
        live at a time; the loose arm reads the whole half up front, which is
        `K_STEPS_QK * 4` VGPRs but lets every `ds_read_b128` issue before the
        first MFMA waits on it. Same `contract_d` either way -- it takes a
        callable precisely so this can be the only difference.
        """
        if const_expr(self.tight_registers):
            return lambda ks: self.reader.read_row_pack(base, scope, half, ks, hdim)
        packs = [
            self.reader.read_row_pack(base, scope, half, ks, hdim) for ks in range_constexpr(self.traits.K_STEPS_QK)
        ]
        return lambda ks: packs[ks]

    def run(self, tile_idx, buf_id, prefetch_idx, v_k, v_v, dv, dk):
        """Accumulate this tile into `(dv, dk)` and prefetch `prefetch_idx`.

        Barrier discipline, and both are needed:

        - the **top** barrier publishes this buffer's DMA, which every wave has
          just waited for;
        - the **bottom** one says every wave has finished reading it, which is
          what makes the prefetch below safe to overwrite it with.

        At two buffers the prefetch lands one tile ahead and overlaps the
        *next* tile's compute; at one it is issued immediately before the wait
        for it, which is the price head_dim 384 and 512 pay for their LDS. The
        body is the same either way.
        """
        traits = self.traits
        ctx = self.ctx_ref
        q_base = ctx.q_buf_base(buf_id)
        do_base = ctx.do_buf_base(buf_id)
        q_scope = dualwave._dualwave_lds_scope("k", buf_id)
        do_scope = dualwave._dualwave_lds_scope("v", buf_id)
        tile_base = tile_idx * fx.Index(traits.BLOCK_Q)

        # `vmcnt` retires in issue order, so leaving exactly the newer tiles'
        # groups outstanding retires this one and nothing else. The forward's
        # `NUM_DMA_K + NUM_DMA_V` idiom, over however many buffers are in play.
        _waitcnt_vm_n((traits.NUM_STREAM_BUFFERS - 1) * (ctx.NUM_DMA_K + ctx.NUM_DMA_V))
        _sched_barrier(0)
        _s_barrier()

        # -- GEMMs 1 and 2, then the softmax. `TIGHT_REGISTERS` decides
        #    whether the tile's two 32-row halves go through together or one
        #    at a time, the other half of the trade `_row_reader` makes.
        #
        # The halves are independent all the way to the packs. Together, the
        # peak holds two score accumulators, two dP, two LSE vectors, two
        # delta, two P and two dS -- about 96 f32 that a half-at-a-time order
        # never has live at once, since only the bf16 packs (8 VGPRs each)
        # survive a half. Apart, the scheduler has half as much independent
        # work to overlap the MFMA bursts with.
        #
        # `delta` is loaded *after* `probabilities` in both arms -- it is not
        # needed until `dscores`, and loading it earlier would hold its 16
        # registers across the LSE vector's.
        p_packs = [None] * 4
        ds_packs = [None] * 4
        for group in self.half_groups:
            s = {h: self.gemm.contract_d(self._row_reader(q_base, q_scope, h, self.hdim_qk), v_k) for h in group}
            dp = {h: self.gemm.contract_d(self._row_reader(do_base, do_scope, h, self.hdim_vo), v_v) for h in group}
            neg_lse2 = {h: self.softmax.load_row_values(ctx.lse_rsrc, tile_base, h, ctx.c_neg_log2e) for h in group}
            p_list = {h: self.softmax.probabilities(s[h], neg_lse2[h]) for h in group}
            delta = {h: self.softmax.load_row_values(ctx.delta_rsrc, tile_base, h, ctx.c_one_f) for h in group}
            ds_list = {h: self.softmax.dscores(p_list[h], dp[h], delta[h]) for h in group}
            for h in group:
                p_packs[2 * h : 2 * h + 2] = self.softmax.pack_half(p_list[h])
                ds_packs[2 * h : 2 * h + 2] = self.softmax.pack_half(ds_list[h])

        # -- GEMM 3 and 4. Q is the reduction axis, so the transposed operand
        #    is read one output chunk at a time and dies with its MFMAs.
        for dc in range_constexpr(traits.D_CHUNKS_PER_SHARD):
            dv[dc] = self.gemm.contract_q(self.reader.read_column_chunk(do_base, do_scope, dc), p_packs, dv[dc])
        for dc in range_constexpr(traits.D_CHUNKS_PER_SHARD):
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
    PADDED_HEAD = knobs.padded_head
    # D columns at or below this are guaranteed real, so the streamed masks can
    # skip them. See `_masked_ks_steps`.
    HDIM_QK_FLOOR = knobs.hdim_qk_floor
    STRIDES_CONSTEXPR = knobs.strides_constexpr
    BUILD_SM_SCALE = meta.sm_scale
    NBUF = traits.NUM_STREAM_BUFFERS
    # One knob for the whole register-against-ILP trade; see
    # `BwdDkDvTileBody._row_reader` and `_with_register_pressure`.
    TIGHT_REGISTERS = knobs.tight_registers
    MASKED_STEPS = tuple(_masked_ks_steps(traits, HDIM_QK_FLOOR)) if PADDED_HEAD else ()

    _cache_tag = (
        traits.cache_tag,
        BLOCK_DMODEL,
        PADDED_HEAD,
        HDIM_QK_FLOOR,
        STRIDES_CONSTEXPR,
        BUILD_SM_SCALE,
        (knobs.num_waves, knobs.block_kv, knobs.block_q, knobs.head_dim_granule),
        (knobs.dkv_shards, NBUF, knobs.waves_per_eu, TIGHT_REGISTERS),
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
            padded_head=PADDED_HEAD,
            hdim_qk_floor=HDIM_QK_FLOOR,
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
        reader.masked_steps = MASKED_STEPS
        resident = BwdDkDvResidentLoader(ctx)
        gemm = BwdDkDvGemmHelper(ctx)
        softmax = BwdDkDvSoftmaxHelper(ctx)
        store = BwdDkDvStoreHelper(ctx)
        tile = BwdDkDvTileBody(
            ctx,
            stream=stream,
            reader=reader,
            gemm=gemm,
            softmax=softmax,
            hdim_qk=hdim_qk,
            hdim_vo=hdim_vo,
            tight_registers=TIGHT_REGISTERS,
        )

        zero_acc = ctx.c_zero_v16f32
        t_end = ctx.split_t_end
        n_acc = traits.D_CHUNKS_PER_SHARD
        scale_vec = Vec.from_elements([ctx.c_sm_scale], fx.Float32).broadcast_to(16)

        @flyc.jit
        def _dkdv_body():
            """K/V resident, Q/dO streaming, two tiles per iteration."""
            v_k = resident.load_rows(
                ctx.k_res_div, ctx.k_res_elem_base, ctx.stride_k_seq_v, ctx.kv_row_in_block, hdim_qk
            )
            v_v = resident.load_rows(
                ctx.v_res_div, ctx.v_res_elem_base, ctx.stride_v_seq_v, ctx.kv_row_in_block, hdim_vo
            )

            # Prime every buffer. From here each tile's DMA is issued by the
            # body `NUM_STREAM_BUFFERS` tiles earlier, so this is the only
            # staging whose latency is not covered by compute.
            for b in range_constexpr(NBUF):
                stream.stage_q_tile(fx.Index(b), b)
                stream.stage_do_tile(fx.Index(b), b)

            init_args = []
            for _ in range_constexpr(2 * n_acc):
                init_args.append(zero_acc)
            loop_results = init_args

            for j, loop_args in range(fx.Index(0), t_end, fx.Index(2), init=init_args):
                dv = [loop_args[i] for i in range_constexpr(n_acc)]
                dk = [loop_args[n_acc + i] for i in range_constexpr(n_acc)]
                # Two tiles per iteration, one LDS buffer each when there are
                # two. At one buffer both slots use it and the prefetch
                # distance collapses; the arithmetic below is the same.
                for slot in range_constexpr(2):
                    dv, dk = tile.run(
                        j + fx.Index(slot),
                        slot % NBUF,
                        j + fx.Index(slot + NBUF),
                        v_k,
                        v_v,
                        dv,
                        dk,
                    )
                loop_results = yield dv + dk

            dv = [loop_results[i] for i in range_constexpr(n_acc)]
            dk = [loop_results[n_acc + i] for i in range_constexpr(n_acc)]

            # `dS` is the gradient of the *scaled* logits, so `sm_scale` belongs
            # to `dK` and to nothing else. Once at the end rather than per tile:
            # a linear factor commutes with the accumulation.
            for dc in range_constexpr(n_acc):
                dk[dc] = dualwave._fmul(Vec(dk[dc]), scale_vec, ctx.fm_fast)
            store.store_accs(dv, ctx.dv_div, ctx.kv_row, ctx.stride_dv_seq_v, hdim_vo, ctx.dv_oob_off)
            store.store_accs(dk, ctx.dk_div, ctx.kv_row, ctx.stride_dk_seq_v, hdim_qk, ctx.dk_oob_off)

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
            # **Head on the fast axis, and measured rather than inherited.**
            # Every workgroup here streams the whole of Q and dO, so putting
            # the KV block on the fast axis to make concurrent workgroups share
            # that slab is the obvious move -- and it is 12-15% *slower* at
            # every rung tried (512: 230 TF against 260; 384: 260 against 283;
            # 256: 390 against 433). The eight XCDs have separate L2s, so
            # sharing a slab duplicates it across all of them instead of
            # spreading distinct work. Same conclusion as the forward's, for a
            # different reason.
            grid=(fx.Index(num_head_q), num_kv_blocks, fx.Index(batch_size)),
            block=(traits.BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    def _check_8x_d_contract(named):
        """Refuse a layout the kernel's 8-wide accesses would overrun.

        **The input contract is 8xD, not 32xD.** Loads and stores are 8 columns
        wide, so a head_dim that is a multiple of 8 is a whole number of chunks
        and a plainly contiguous `(B, H, S, 24)` needs no padding of any kind.
        An odd head_dim still works, but only in an allocation with slack: the
        kernel touches `ceil8(hdim)` columns per row, so those extra elements
        must belong to the caller.

        Two separate requirements, and the pitch is only the first. *Alignment*:
        a row starts at `sum(index * stride)`, so every non-D stride must be a
        multiple of 8 for the 16-byte access to land aligned. *Slack*: the gap
        after a row is the smallest non-D stride, which for a BHSD tensor is
        the D pitch but for a BSHD one is `D` itself -- consecutive heads of the
        same token are adjacent, so a pitch check alone waves that through
        while the store corrupts the next head.
        """
        for name, t in named:
            d = t.shape[3]
            need = (d + 7) // 8 * 8
            if need == d:
                continue
            outer = [t.stride(i) for i in range(3) if t.shape[i] > 1]
            aligned = t.stride(3) == 1 and all(s % 8 == 0 for s in outer)
            slack = min(outer, default=need)
            if not aligned or slack < need:
                raise ValueError(
                    f"{name} has shape {tuple(t.shape)} strides {tuple(t.stride())}, which cannot hold "
                    f"a head_dim of {d}. {d} is not a multiple of 8, so the kernel reads and writes "
                    f"{need} columns per row and needs the D axis innermost, every other stride a "
                    f"multiple of 8, and {need - d} unused element(s) after each row. Allocate the last "
                    f"dimension as {need} and pass a [..., :{d}] view -- or use a head_dim that is a "
                    f"multiple of 8, which needs no padding at all."
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
        # dK follows Q's head dim and dV follows V's, which is what makes an
        # asymmetric build meaningful at all.
        if DK.shape[3] != hdim_qk or DO.shape[3] != hdim_vo or DV.shape[3] != hdim_vo:
            raise ValueError(
                f"dK must carry hdim_qk ({hdim_qk}) and dO/dV hdim_vo ({hdim_vo}); got "
                f"dK {DK.shape[3]}, dO {DO.shape[3]}, dV {DV.shape[3]}"
            )
        if hdim_qk > BLOCK_DMODEL or hdim_vo > BLOCK_DMODEL:
            raise ValueError(
                f"this build serves head dims up to {BLOCK_DMODEL}, got hdim_qk {hdim_qk} and " f"hdim_vo {hdim_vo}"
            )
        if not PADDED_HEAD and (hdim_qk != BLOCK_DMODEL or hdim_vo != BLOCK_DMODEL):
            raise ValueError(
                f"this build is not compiled for a padded head; it serves head_dim {BLOCK_DMODEL} "
                f"exactly, got hdim_qk {hdim_qk} and hdim_vo {hdim_vo}"
            )
        # Both extents share one mask floor, so both must sit above it.
        if HDIM_QK_FLOOR and min(hdim_qk, hdim_vo) <= HDIM_QK_FLOOR:
            raise ValueError(
                f"this build serves head dims in ({HDIM_QK_FLOOR}, {BLOCK_DMODEL}], got hdim_qk "
                f"{hdim_qk} and hdim_vo {hdim_vo}; build for the narrower rung, or pin "
                "hdim_qk_floor=0 to mask every column"
            )
        if PADDED_HEAD:
            _check_8x_d_contract((("Q", Q), ("K", K), ("V", V), ("DO", DO), ("DK", DK), ("DV", DV)))
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
            abi.resolve_scale(
                Q, scale if scale is not None else BUILD_SM_SCALE, PADDED_HEAD, 1.0 / (BLOCK_DMODEL**0.5)
            ),
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
