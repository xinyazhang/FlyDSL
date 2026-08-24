# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused flash-attention backward for gfx1201 (RDNA4) -- dK, dV and dQ, one launch.

A port of AOTriton's ``bwd_kernel_fuse`` (with ``bwd_inner_fuse`` and
``bwd_inner_dq``) onto the FlyDSL gfx1201 stack, sharing the forward kernel's
ABI: BHSD shapes, ``fx.Pointer`` arguments, ``(batch, head, seq)`` stride
triples, and **our** VarlenBits rather than AOTriton's ``cu_seqlens``.

--- What "fused" means here ------------------------------------------------

Not two accumulator sets in one program. AOTriton's fused kernel is a single
launch with **two program roles** selected by ``block_idx.x``:

    block_idx.x <  num_kv_blocks   ->  dK/dV role: own KV_TILE key rows,
                                       iterate over query blocks
    block_idx.x >= num_kv_blocks   ->  dQ role:    own Q_TILE query rows,
                                       iterate over key blocks

so a given wave carries dK+dV *or* dQ, never all three. That matters for the
register budget and it is the first thing to know before reading the two
bodies below, because "fused" suggests the harder thing.

What the fusion actually buys is one dispatch instead of two. What it costs is
recorded here because it is not obvious: the two roles need *different* LDS
layouts, and a workgroup's LDS allocation is static, so **every program pays
the larger of the two**. Measured at head_dim 128 f16: the dK/dV role needs
52480 B and the dQ role 13568 B, and the emitted binary reserves 52480 B for
both -- so the dQ programs run at a quarter of the occupancy a split kernel
would give them.

--- The operand-layout constraint, which shapes everything -----------------

gfx1201 WMMA is 16x16x16 on wave32. Every A/B operand is a v8f16 per lane laid
out as

    lane l, element e  <->  (free index = l % 16, contracted index = (l/16)*8 + e)

which is the *same* register layout for A and B -- the difference is only
whether the free index becomes the result's row or its column. The result is

    lane l, element e  <->  D[(l/16)*8 + e][l % 16]

i.e. the result's **row** index lands on ``klane*8 + e`` and its column on
``lane16``. Two consequences run through this whole file:

1. A WMMA result is immediately a valid operand pack for a product that
   contracts over **its row index**. That is how P feeds dV and dS feeds dK/dQ
   with no rearrangement, and it is why the forward kernel computes ``S = K
   Q^T`` rather than ``Q K^T``.

2. A matrix can only be an operand of a product that contracts over the axis
   it was *loaded* along. dK and dV contract over the query axis; S and dP
   contract over the head axis. So the dK/dV role needs **Q and dO in both
   orientations** -- there is no shuffle that avoids it, and the transposed
   copies are staged with ``global_load_tr_b128`` exactly as the forward kernel
   stages V^T.

The five products, with the operand each side supplies (``pack(free|contracted)``):

    dK/dV role                                       result element e <->
      S  = A:Q(q|d)      B:K(kv|d)                     (q=klane*8+e, kv=lane16)
      dP = A:dO(q|d)     B:V(kv|d)                     same
      dV^T += A:dO^T(d|q) B:P(kv|q)                    (d=klane*8+e, kv=lane16)
      dK^T += A:Q^T(d|q)  B:dS(kv|q)                   same
    dQ role
      S^T  = A:K(kv|d)    B:Q(q|d)                     (kv=klane*8+e, q=lane16)
      dP^T = A:V(kv|d)    B:dO(q|d)                    same
      dQ^T += A:K^T(d|kv) B:dS^T(q|kv)                 (d=klane*8+e, q=lane16)

Every accumulator therefore ends up as "8 contiguous head-dim values at one
token row", which is a v8 store -- the same epilogue shape the forward kernel
uses for O.

--- Register budget --------------------------------------------------------

A wave owning 16 rows x head_dim of f32 costs ``head_dim / 2`` VGPRs. The
dK/dV role holds two of those across the query loop, so **head_dim VGPRs are
gone before any operand is live**: 64 at head_dim 64, 128 at head_dim 128, 192
at 192. That last one leaves 64 VGPRs for K/V/Q/dO packs, S, dP, dS, addresses
and the epilogue, which is not enough -- so `fmha_tuning_bwd_fuse_gfx1201`
caps head_dim at 128 and says so as a `ValueError`. The dQ role is lighter
(one accumulator plus the resident Q and dO packs, also ``head_dim`` total)
and is not the binding constraint.

--- Deltas, and what this kernel does not do -------------------------------

``delta = rowsum(dO * O)`` is **not** computed here. The interface computes it
with PyTorch; a fused preprocess kernel is a later optimisation. LSE is read as
produced by our forward kernel, natural-log, with the pitch
``fmha.lse_token_pitch`` describes; delta is required to share that layout.

Not implemented: attention bias (``BIAS_TYPE`` is 0 throughout), and dropout,
which ``build_fmha_bwd_fuse_module`` rejects at build time -- the comment on
that rejection says exactly what is missing and why it is a rejection rather
than an untested code path.

GQA is handled by giving the dK/dV role one program per **query** head and
letting the caller reduce over the group, rather than by AOTriton's in-kernel
loop over the group with K/V held resident. That trades K/V re-reads and a
``(B, Hq, S, D)`` scratch buffer for one fewer level of loop-carried
accumulator nesting; see `fmha_bwd_fuse_gfx1201_interface` for the reduction.
"""

import math as host_math

import fmha_abi_gfx1201 as abi
import fmha_common_gfx1201 as fmha
from fmha_tuning_bwd_fuse_gfx1201 import (  # noqa: F401
    LDS_PAD,
    BwdInputMetadata,
    BwdKnobs,
    acc_vgprs_dkdv,
    lds_elems_dkdv,
    lds_elems_dq,
)
from gfx1201_standalone import buffer_ops
from gfx1201_standalone import utils as common_utils

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

KERNEL_NAME = "fmha_bwd_fuse_gfx1201_kernel"
_LOG2E = host_math.log2(host_math.e)

# `dtype_to_elem_type`, `_run_compiled`, `_pointer_load` and `_pointer_store`
# are inlined copies of the ones in `flash_attn_func_gfx1201_aiw.py`. Duplicated
# on purpose and for the reason recorded there: this directory is a
# self-contained prototype that must run with the cwd set to it and no
# PYTHONPATH. Importing them from the forward kernel module would also make a
# backward build pull in the whole forward builder, which is a dependency in
# the wrong direction. Fold all four into one place if this graduates.


_COMPILED = abi.new_compiled_cache()


def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return _llvm.LoadOp(result_type, fmha.llvm_value(ptr)).result


def _pointer_store(value: ir.Value, ptr: ir.Value):
    return _llvm.StoreOp(fmha.llvm_value(value), fmha.llvm_value(ptr))


def listify(carried, n):
    """A loop's carried state as a list of exactly `n` values.

    `scf.for` hands back a bare value rather than a sequence when it carries
    exactly one, which happens at head_dim 16 where the dQ role has a single
    accumulator. Indexing that bare value extracts a vector *element*, so the
    next WMMA silently receives an f32 scalar and the failure surfaces far from
    its cause.

    A plain Python function, and called rather than open-coded, because the
    obvious spelling -- rebinding the loop variable inside the body -- makes
    `ast_rewriter._collect_assigned_vars` treat the loop's own carried name as
    newly assigned state. That fails with a bare
    `'list' object has no attribute '_CAPIPtr'`.
    """
    if isinstance(carried, ir.Value):
        # A single value, already unwrapped. Indexing it would reach into the
        # *vector* and hand the next WMMA an f32 scalar.
        return [carried]
    return [carried[i] for i in range(n)]


def plain_regions(seq_len, block, alive):
    """`[full][one masked tail]` over `seq_len`, shaped like `CausalRegions`.

    The non-causal counterpart of `fmha.decompose_causal_regions`, returning
    the *same* seven fields so that the two loops that walk them are written
    once. Without this the kernel would carry a second pair of region loops
    differing only in how three of the counts were derived.

    `block` must be a power of two: `sdiv_rd_pow2` is an arithmetic shift.

    Module-level, like everything in `fmha_common_gfx1201`, because the rewrite
    from Python `if` to `scf.if` is lexical per `@flyc.kernel` function -- this
    uses only selects, so it needs no branch and is safe anywhere.

    **A likely shared helper.** The dK/dV split kernel and the dQ split kernel
    both need exactly this; see the consolidation note in the module docstring
    of the interface.
    """
    zero = fx.Int32(0)
    bn = fx.Int32(block)
    n_full = common_utils.smax(common_utils.sdiv_rd_pow2(seq_len, block), zero)
    full_end = n_full * bn
    n_right = common_utils.ssel(seq_len > full_end, fx.Int32(1), zero)
    n_full = common_utils.ssel(alive, n_full, zero)
    n_right = common_utils.ssel(alive, n_right, zero)
    return fmha.CausalRegions(
        n_left=zero,
        n_full=n_full,
        n_right=n_right,
        left_col0=zero,
        full_col0=zero,
        right_col0=full_end,
        masked_col0=full_end,
    )


def build_fmha_bwd_fuse_module(meta: BwdInputMetadata, knobs: BwdKnobs):
    """Build the fused gfx1201 backward kernel for one (problem, schedule) pair.

    Both arguments must be fully resolved -- use
    `fmha_tuning_bwd_fuse_gfx1201.plan`. Nothing here invents a default; an
    unresolved knob is a caller error.
    """
    # ---- Problem ----
    HEAD_DIM = knobs.block_dmodel
    PADDED_HEAD = knobs.padded_head
    dtype_str = meta.dtype_str
    CAUSAL = bool(meta.causal)

    # Attention bias. An input only -- dB comes from the standalone dQ kernel.
    BIAS_TYPE = 1 if meta.bias else 0
    assert not (BIAS_TYPE and CAUSAL), "bias and causal are mutually exclusive, as in the forward"
    sm_scale = meta.sm_scale

    # ---- Schedule ----
    NUM_WAVES = knobs.num_waves
    Q_STEP = knobs.q_step
    KV_STEP = knobs.kv_step
    WAVES_PER_EU = knobs.waves_per_eu
    FLAT_WORK_GROUP_SIZE = knobs.flat_work_group_size
    SCHED_STRATEGY = knobs.sched_strategy
    LPT_TILE_ORDER = knobs.lpt_tile_order
    FP_MODE = knobs.fp_mode
    DENORMALS_ARE_ZERO = knobs.denormals_are_zero
    UNSAFE_FP_MATH = knobs.unsafe_fp_math
    FAST_FP_MATH = knobs.fast_fp_math

    if dtype_str not in ("f16", "bf16"):
        raise ValueError(f"the fused backward kernel supports f16/bf16, got {dtype_str!r}")

    # Dropout. **Rejected, not silently ignored**, and not shipped untested.
    #
    # The forward kernel's scheme transfers directly to the dQ role -- there
    # the eight elements of an accumulator are eight consecutive *key* columns
    # of one query row, which is exactly what `Philox.keep_span` produces. It
    # does *not* transfer to the dK/dV role: there the eight elements are eight
    # consecutive *query* rows at one key column, so each needs its own
    # `grid_offset` and a runtime-indexed slot within the four randoms an
    # offset carries. That is implementable (eight philox4x32 per accumulator
    # plus a 4-way select) but it is a hot-path feature with no reference to
    # check it against short of reconstructing the mask with
    # `dropout_mask_gfx1201`, and an unverified backward pass is worse than an
    # absent one.
    if meta.dropout:
        raise NotImplementedError(
            "dropout is not implemented in the fused backward kernel. The dQ "
            "role maps onto Philox.keep_span unchanged; the dK/dV role needs "
            "per-element offsets because its accumulator elements run along "
            "the query axis, not the key axis. See the comment here."
        )

    # ---- WMMA / wave32 constants (hardware, not knobs) ----
    WARP_SIZE = 32
    WMMA_M = WMMA_N = WMMA_K = 16
    # K elements each lane holds of an A/B operand: WMMA_K / (WARP_SIZE/WMMA_M).
    WMMA_LANE_K = WMMA_K // (WARP_SIZE // WMMA_M)

    BLOCK_SIZE = NUM_WAVES * WARP_SIZE
    if FLAT_WORK_GROUP_SIZE is None:
        FLAT_WORK_GROUP_SIZE = BLOCK_SIZE

    # One 16-row WMMA tile per wave, in both roles.
    KV_TILE = WMMA_N * NUM_WAVES  # key rows owned by a dK/dV program
    Q_TILE = WMMA_M * NUM_WAVES  # query rows owned by a dQ program

    ND = HEAD_DIM // WMMA_N  # head-dim 16-blocks == accumulators per matrix

    assert HEAD_DIM % 16 == 0 and 0 < HEAD_DIM <= 128, f"head tile must be a multiple of 16 in (0, 128], got {HEAD_DIM}"
    assert Q_STEP % 16 == 0 and KV_STEP % 16 == 0
    assert Q_STEP & (Q_STEP - 1) == 0 and KV_STEP & (KV_STEP - 1) == 0

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(HEAD_DIM)

    # Causal alignment. Same vocabulary as the forward kernel: the *caller*
    # says 0/1/2/3, the kernel only ever sees 0 or 3, and 1/2 travel as window
    # sentinels resolved per sequence by `fmha.resolve_window`. Resolving them
    # on the host is wrong the moment there is more than one sequence.
    HOST_CAUSAL_TYPE = (1 if CAUSAL else 0) if meta.causal_type is None else meta.causal_type
    if HOST_CAUSAL_TYPE not in (0, 1, 2, 3):
        raise ValueError(f"causal_type must be 0, 1, 2 or 3, got {HOST_CAUSAL_TYPE}")
    if bool(HOST_CAUSAL_TYPE) != CAUSAL:
        raise ValueError(f"causal={CAUSAL} disagrees with causal_type={HOST_CAUSAL_TYPE}")

    fastmath = fmha.FastMath(FP_MODE)

    # ---- LDS layout ----
    #
    # Two disjoint plans over one allocation, one per role, because only one
    # role runs in a given workgroup. `LDS_TOTAL` is the max, which is what the
    # module docstring calls the price of fusing.
    #
    # Padded rather than XOR-swizzled: a swizzle was implemented and measured a
    # net loss on this part (`sdpa_lore_gfx1201.md`).
    ROW_STRIDE = HEAD_DIM + LDS_PAD  # tiles indexed [token][d]
    QT_STRIDE = Q_STEP + LDS_PAD  # Q^T / dO^T, indexed [d][token]
    KT_STRIDE = KV_STEP + LDS_PAD  # K^T,        indexed [d][token]

    # dK/dV role
    A_K = 0
    A_V = A_K + KV_TILE * ROW_STRIDE
    A_Q = A_V + KV_TILE * ROW_STRIDE
    A_DO = A_Q + Q_STEP * ROW_STRIDE
    A_QT = A_DO + Q_STEP * ROW_STRIDE
    A_DOT = A_QT + HEAD_DIM * QT_STRIDE
    LDS_DKDV = A_DOT + HEAD_DIM * QT_STRIDE
    assert LDS_DKDV == lds_elems_dkdv(HEAD_DIM, KV_TILE, Q_STEP)

    # dQ role
    B_K = 0
    B_V = B_K + KV_STEP * ROW_STRIDE
    B_KT = B_V + KV_STEP * ROW_STRIDE
    LDS_DQ = B_KT + HEAD_DIM * KT_STRIDE
    assert LDS_DQ == lds_elems_dq(HEAD_DIM, KV_STEP)

    LDS_TOTAL = max(LDS_DKDV, LDS_DQ)

    # Cooperative-load vector width, in elements. Fixed at 8 (16 B), which is
    # exactly what the D-axis alignment contract guarantees -- see the long
    # note on `VEC_WIDTH` in the forward kernel.
    VEC_WIDTH = 8
    TPR = HEAD_DIM // VEC_WIDTH  # threads cooperating on one row
    ROWS_PER_BATCH = BLOCK_SIZE // TPR
    assert ROWS_PER_BATCH >= 1, f"head_dim {HEAD_DIM} needs more than {BLOCK_SIZE} threads per row-batch"

    def _batches(rows):
        """`(num_batches, needs_guard)` covering `rows` with ceil() batches.

        Ceil, not floor: flooring silently drops rows whenever
        `ROWS_PER_BATCH` neither reaches `rows` nor divides it, which surfaces
        as stale LDS rather than as an error.
        """
        nb = (rows + ROWS_PER_BATCH - 1) // ROWS_PER_BATCH
        return nb, nb * ROWS_PER_BATCH != rows

    NB_KV, GUARD_KV = _batches(KV_TILE)
    NB_QSTEP, GUARD_QSTEP = _batches(Q_STEP)
    NB_KVSTEP, GUARD_KVSTEP = _batches(KV_STEP)

    # Transposed staging, `global_load_tr_b128`. One wave-wide load produces a
    # 16(d) x 16(token) block already in operand order; these say how many such
    # blocks there are and how they divide over the waves. The tiling need not
    # divide evenly -- tail tiles are guarded at the LDS store.
    def _tr_geom(tokens):
        tiles = ND * (tokens // WMMA_K)
        loads = (tiles + NUM_WAVES - 1) // NUM_WAVES
        return tiles, loads, loads * NUM_WAVES != tiles

    TR_Q_TILES, TR_Q_LOADS, TR_Q_GUARD = _tr_geom(Q_STEP)
    TR_K_TILES, TR_K_LOADS, TR_K_GUARD = _tr_geom(KV_STEP)

    elem_numeric_cls = abi.dtype_to_elem_type(dtype_str)

    @fx.struct
    class SharedStorage:
        buf: fx.Array[elem_numeric_cls, LDS_TOTAL, 16]

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def bwd_fuse_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        Bias: fx.Pointer,
        DO: fx.Pointer,
        DK: fx.Pointer,
        DV: fx.Pointer,
        DQ: fx.Pointer,
        LSE: fx.Pointer,
        Delta: fx.Pointer,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        batch_size: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        num_kv_blocks: fx.Int32,
        num_q_blocks: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
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
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
    ):
        elem_type = elem_numeric_cls.ir_type
        elem_dtype = elem_numeric_cls
        f32_ty = ir.F32Type.get()
        v8f16_type = Vec.make_type(8, elem_dtype)

        q_ptr = fmha.pointer_to_llvm_ptr(Q)
        k_ptr = fmha.pointer_to_llvm_ptr(K)
        v_ptr = fmha.pointer_to_llvm_ptr(V)
        do_ptr = fmha.pointer_to_llvm_ptr(DO)
        dk_ptr = fmha.pointer_to_llvm_ptr(DK)
        dv_ptr = fmha.pointer_to_llvm_ptr(DV)
        dq_ptr = fmha.pointer_to_llvm_ptr(DQ)
        l_ptr = fmha.pointer_to_llvm_ptr(LSE)
        dlt_ptr = fmha.pointer_to_llvm_ptr(Delta)
        q_ptr_i64 = fx.as_ir_value(fx.Int64(fx.ptrtoint(Q)))
        k_ptr_i64 = fx.as_ir_value(fx.Int64(fx.ptrtoint(K)))
        do_ptr_i64 = fx.as_ir_value(fx.Int64(fx.ptrtoint(DO)))

        # ---- Varlen prologue: VarlenBits -> six scalars ----
        # The only place the layout is examined; everything below reads the
        # scalars and cannot tell which mode it is in. Identical to the forward
        # kernel's prologue, deliberately -- the backward pass must address the
        # same tensors the same way or it silently differentiates a different
        # function.
        z_i32 = fx.Int32(gpu.block_idx.z)
        seqlen_q_i32, q_row_off, q_batch = fmha.decode_addressing(
            varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, z_i32
        )
        seqlen_k_i32, k_row_off, k_batch = fmha.decode_addressing(
            varlen_bits, 8, max_seqlen_k, seqinfo_k0, seqinfo_k1, z_i32
        )
        lse_tokens = fmha.lse_token_pitch(varlen_bits, 0, max_seqlen_q, seqinfo_q0, seqinfo_q1, num_seqlens)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_buf = lds.buf.ptr

        tid, wave_id, lane, lane16, klane = fmha.wave_lanes(WARP_SIZE)

        load_row_in_batch = tid // TPR
        load_col_base = (tid % TPR) * VEC_WIDTH

        # Grid: (num_kv_blocks + num_q_blocks, num_head_q, batch).
        #
        # grid.y is the **query** head in both roles. AOTriton uses the KV head
        # and loops the GQA group inside the dK/dV branch, holding K and V
        # resident across it; here the group is spread over grid.y and the
        # caller reduces. That removes a loop-carried-accumulator nesting level
        # at the cost of re-reading K/V per query head -- see the docstring.
        pid = fx.Int32(gpu.block_idx.x)
        head_q = fx.Index(gpu.block_idx.y)
        head_k = head_q // (fx.Index(num_head_q) // fx.Index(num_head_k))

        sm_log2e = fastmath.mul(sm_scale, fx.Float32(_LOG2E))

        q_st = (fx.Index(stride_q_batch), fx.Index(stride_q_head), fx.Index(stride_q_seq))
        k_st = (fx.Index(stride_k_batch), fx.Index(stride_k_head), fx.Index(stride_k_seq))
        v_st = (fx.Index(stride_v_batch), fx.Index(stride_v_head), fx.Index(stride_v_seq))
        do_st = (fx.Index(stride_do_batch), fx.Index(stride_do_head), fx.Index(stride_do_seq))
        dk_st = (fx.Index(stride_dk_batch), fx.Index(stride_dk_head), fx.Index(stride_dk_seq))
        dv_st = (fx.Index(stride_dv_batch), fx.Index(stride_dv_head), fx.Index(stride_dv_seq))
        dq_st = (fx.Index(stride_dq_batch), fx.Index(stride_dq_head), fx.Index(stride_dq_seq))

        _q_batch_v = fx.Index(q_batch)
        _k_batch_v = fx.Index(k_batch)
        _q_row_off_v = fx.Index(q_row_off)
        # Bias is (B, H, Sq, Sk): its last axis is the KV column and is the
        # contiguous one. `head_q` is a grid dimension here rather than a loop
        # variable, so unlike the standalone dK/dV kernel the base is built
        # once.
        if const_expr(BIAS_TYPE):
            bias_ptr = fmha.pointer_to_llvm_ptr(Bias)
            _b_head = _q_batch_v * fx.Index(stride_b_batch) + head_q * fx.Index(stride_b_head)
        _k_row_off_v = fx.Index(k_row_off)

        # `max(seqlen - 1, 0)`: `fx.Index` is unsigned, so a bare `seqlen - 1`
        # at seqlen 0 wraps and the row clamp then pins every address to
        # 2**64-1.
        q_last = fx.Index(common_utils.smax(seqlen_q_i32 - fx.Int32(1), fx.Int32(0)))
        k_last = fx.Index(common_utils.smax(seqlen_k_i32 - fx.Int32(1), fx.Int32(0)))
        seqlen_q_v = fx.Index(seqlen_q_i32)
        seqlen_k_v = fx.Index(seqlen_k_i32)

        # Address pairs. Every tile staged through LDS is read with the clamped
        # `kv_addr` form so a row past the sequence end lands on a live row
        # rather than outside the allocation; the value it returns is discarded
        # by the score mask or by the store guard.
        _q_kw = dict(seqlen_k=seqlen_q_v, seq_last=q_last, hoist=False, clamp=True)
        _k_kw = dict(seqlen_k=seqlen_k_v, seq_last=k_last, hoist=False, clamp=True)
        q_tbase, q_toff, q_addr = fmha.make_addr_pair(q_st, head_q, _q_batch_v, _q_row_off_v, **_q_kw)
        _, _, k_addr = fmha.make_addr_pair(k_st, head_k, _k_batch_v, _k_row_off_v, **_k_kw)
        _, _, v_addr = fmha.make_addr_pair(v_st, head_k, _k_batch_v, _k_row_off_v, **_k_kw)
        do_tbase, do_toff, do_addr = fmha.make_addr_pair(do_st, head_q, _q_batch_v, _q_row_off_v, **_q_kw)
        # Outputs are never over-read: the store is inside a row guard, so no
        # clamp is emitted for them.
        dk_tbase, dk_toff, _ = fmha.make_addr_pair(
            dk_st, head_q, _k_batch_v, _k_row_off_v, seqlen_k=seqlen_k_v, seq_last=k_last, hoist=False, clamp=False
        )
        dv_tbase, dv_toff, _ = fmha.make_addr_pair(
            dv_st, head_q, _k_batch_v, _k_row_off_v, seqlen_k=seqlen_k_v, seq_last=k_last, hoist=False, clamp=False
        )
        dq_tbase, dq_toff, _ = fmha.make_addr_pair(
            dq_st, head_q, _q_batch_v, _q_row_off_v, seqlen_k=seqlen_q_v, seq_last=q_last, hoist=False, clamp=False
        )

        def load_global_v8(ptr, base64, off):
            return _pointer_load(v8f16_type, fmha.split_ptr(ptr, base64, off, elem_type))

        def store_global_v8(ptr, base64, off, val):
            _pointer_store(val, fmha.split_ptr(ptr, base64, off, elem_type))

        # ---- Row-wise f32 side inputs: LSE and delta ----
        #
        # One value per (batch, head, query row), and both tensors are
        # *compact*, so their pitch is a function of VarlenBits rather than
        # something the caller passes -- the same argument
        # `fmha.lse_token_pitch` makes for LSE alone. The two layouts are the
        # same indices arranged two ways, so they collapse into a base and a
        # per-row pitch chosen once:
        #
        #   _HT  (H, T)  offset = (b*H + h)*tokens + row,      pitch 1
        #   _TH  (T, H)  offset = (b*tokens + row)*H + h,      pitch H
        #
        # Delta is *required* to share LSE's layout. It is produced beside it
        # by the same caller, and giving it its own would double the decode for
        # no expressiveness.
        row_base, row_pitch = fmha.lse_row_addressing(
            varlen_bits, _q_batch_v, head_q, num_head_q, lse_tokens, _q_row_off_v
        )

        def load_bias_1(off64):
            """One bias element, for the half whose eight values are eight q rows."""
            return _pointer_load(elem_type, buffer_ops.get_element_ptr(bias_ptr, fx.Int64(off64), elem_type=elem_type))

        def load_bias_8(off64):
            """Eight adjacent KV columns, for the half that walks the key axis."""
            return _pointer_load(v8f16_type, fmha.split_ptr(bias_ptr, fx.Int64(off64), fx.Index(0), elem_type))

        def load_row_f32(base_ptr, q_row):
            """`tensor[q_row]` of a row-wise f32 side input, address always legal.

            `q_row` is an `fx.Int32` absolute query index which may be past the
            sequence -- the score mask discards the value, but the *address*
            must still be inside the allocation, so an out-of-range row reads
            row 0. Row 0 always exists: a sequence with no query rows has no
            live program.
            """
            ok = q_row < seqlen_q_i32
            safe = fx.Index(fx.Int32(q_row) if ok else fx.Int32(0))
            off = row_base + safe * row_pitch
            return _pointer_load(f32_ty, buffer_ops.get_element_ptr(base_ptr, fx.Int64(off), elem_type=f32_ty))

        # ---- Column masking for a padded head ----
        qk_cols = fmha.MaskedAxis(fx.Index(hdim_qk), active=PADDED_HEAD, elem_dtype=elem_dtype)
        vo_cols = fmha.MaskedAxis(fx.Index(hdim_vo), active=PADDED_HEAD, elem_dtype=elem_dtype)

        def _tile_ap(cols, base, rows_tile):
            """An LDS-staged aperture over a `[token][d]` tile."""
            nb, guard = _batches(rows_tile)
            return fmha.Aperture(
                cols,
                lds_base=base,
                lds_stride=ROW_STRIDE,
                vec_width=VEC_WIDTH,
                threads_per_row=TPR,
                rows_per_batch=ROWS_PER_BATCH,
                num_batches=nb,
                needs_guard=guard,
            )

        def _tr_ap(cols, base, stride):
            """An LDS-staged aperture over a transposed `[d][token]` tile.

            No cooperative geometry: transposed tiles are filled by
            `global_load_tr_b128`, whose distribution is `TransposedTiling`'s.
            """
            return fmha.Aperture(cols, lds_base=base, lds_stride=stride, vec_width=VEC_WIDTH)

        # dK/dV role apertures
        a_k_ap = _tile_ap(qk_cols, A_K, KV_TILE)
        a_v_ap = _tile_ap(vo_cols, A_V, KV_TILE)
        a_q_ap = _tile_ap(qk_cols, A_Q, Q_STEP)
        a_do_ap = _tile_ap(vo_cols, A_DO, Q_STEP)
        a_qt_ap = _tr_ap(qk_cols, A_QT, QT_STRIDE)
        a_dot_ap = _tr_ap(vo_cols, A_DOT, QT_STRIDE)
        # dQ role apertures
        b_k_ap = _tile_ap(qk_cols, B_K, KV_STEP)
        b_v_ap = _tile_ap(vo_cols, B_V, KV_STEP)
        b_kt_ap = _tr_ap(qk_cols, B_KT, KT_STRIDE)

        # The two lane-offset pairs are different mappings and must not be
        # interchanged: the *load* pair is the address each lane supplies so
        # the hardware 8x8 transpose lands the right 16x16 block, the *store*
        # pair is where the lane's transposed result then belongs in LDS.
        _tr_lane = dict(
            num_waves=NUM_WAVES,
            d_step=WMMA_N,
            kv_step=WMMA_K,
            wave_id=wave_id,
            load_d_off=((lane // 8) % 2) * WMMA_LANE_K,
            load_kv_off=(lane // 16) * WMMA_LANE_K + (lane % 8),
            store_d_off=lane16,
            store_kv_off=klane * WMMA_LANE_K,
        )
        qt_tr = fmha.TransposedTiling(
            d_blocks=ND, tiles=TR_Q_TILES, loads=TR_Q_LOADS, needs_guard=TR_Q_GUARD, **_tr_lane
        )
        kt_tr = fmha.TransposedTiling(
            d_blocks=ND, tiles=TR_K_TILES, loads=TR_K_LOADS, needs_guard=TR_K_GUARD, **_tr_lane
        )

        # Readers: `start -> (row, col) -> value`, one per (tensor, load kind).
        fetch_q = fmha.reader(q_addr, lambda b, o: load_global_v8(q_ptr, b, o))
        fetch_k = fmha.reader(k_addr, lambda b, o: load_global_v8(k_ptr, b, o))
        fetch_v = fmha.reader(v_addr, lambda b, o: load_global_v8(v_ptr, b, o))
        fetch_do = fmha.reader(do_addr, lambda b, o: load_global_v8(do_ptr, b, o))
        fetch_q_tr = fmha.reader(q_addr, lambda b, o: fmha.global_load_tr_v8(q_ptr_i64, b, o, v8f16_type))
        fetch_k_tr = fmha.reader(k_addr, lambda b, o: fmha.global_load_tr_v8(k_ptr_i64, b, o, v8f16_type))
        fetch_do_tr = fmha.reader(do_addr, lambda b, o: fmha.global_load_tr_v8(do_ptr_i64, b, o, v8f16_type))

        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)

        def pack_v8(f32_vals):
            """Eight f32 scores as one 16-bit WMMA operand.

            There is no F32xF32 WMMA on gfx1201 (the ISA's A/B operands are
            f16/bf16/iu8/iu4/fp8 only), so P and dS must be narrowed before
            they can feed the matrix cores. bf16 goes through the bitwise
            truncation `fmha.bf16_trunc_pack_v8` documents.
            """
            if const_expr(dtype_str == "bf16"):
                return fmha.bf16_trunc_pack_v8(f32_vals, elem_dtype)
            return Vec.from_elements(
                [fx.Float32(f32_vals[j]).to(elem_dtype) for j in range_constexpr(8)], elem_dtype
            ).ir_value()

        if const_expr(CAUSAL):
            wl_i32, wr_i32 = fmha.resolve_window(window_left, window_right, seqlen_q_i32, seqlen_k_i32)

        # ==================================================================
        # Role 1: dK and dV. Own KV_TILE key rows, walk the query axis.
        # ==================================================================
        def dkdv_role():
            start_k = fx.Index(pid) * fx.Index(KV_TILE)
            start_k_i32 = pid * fx.Int32(KV_TILE)
            alive = start_k_i32 < seqlen_k_i32

            # This wave's 16 key rows, and where they sit in the LDS K/V tile.
            kv_row_local = wave_id * fx.Index(WMMA_N) + lane16
            kv_row_i32 = start_k_i32 + fx.Int32(kv_row_local)

            # K and V are loop-invariant here, so they are staged once. Their
            # LDS region is disjoint from the Q/dO tiles the loop rewrites, so
            # the per-iteration barriers do not disturb them.
            fmha.stage(a_k_ap, lds_buf, fetch_k(start_k), load_row_in_batch, load_col_base, fx.Index(KV_TILE))
            fmha.stage(a_v_ap, lds_buf, fetch_v(start_k), load_row_in_batch, load_col_base, fx.Index(KV_TILE))
            gpu.barrier()

            def q_body(start_q_i, args, MASKED):
                """One Q_STEP block of queries against this program's key rows.

                `MASKED` is a Python bool resolved at trace time: the full
                region emits no mask at all, which is the whole point of
                splitting the range into regions rather than testing per tile.
                """
                _a = listify(args, 2 * ND)
                dk_accs = _a[:ND]
                dv_accs = _a[ND:]

                start_q_v = fx.Index(start_q_i)
                # Both orientations of Q and dO. See the operand-layout note in
                # the module docstring for why both are needed; the transposed
                # copies come from `global_load_tr_b128`, which delivers a
                # 16x16 block already in operand order.
                fmha.stage(a_q_ap, lds_buf, fetch_q(start_q_v), load_row_in_batch, load_col_base, fx.Index(Q_STEP))
                fmha.stage(a_do_ap, lds_buf, fetch_do(start_q_v), load_row_in_batch, load_col_base, fx.Index(Q_STEP))
                fmha.publish_transposed(
                    a_qt_ap, qt_tr, lds_buf, fmha.read_transposed(a_qt_ap, qt_tr, fetch_q_tr(start_q_v))
                )
                # `qt_tr` for dO^T as well, not a tiling of its own: the two
                # tiles have identical geometry (head_dim x Q_STEP), and the
                # object holds only that geometry plus the lane offsets.
                fmha.publish_transposed(
                    a_dot_ap, qt_tr, lds_buf, fmha.read_transposed(a_dot_ap, qt_tr, fetch_do_tr(start_q_v))
                )
                gpu.barrier()

                # S[q][kv] and dP[q][kv]. Both contract over the head axis, so
                # both take the row-major tiles; the result element `e` is
                # query `klane*8 + e` at key `lane16`.
                s_acc = fx.as_ir_value(c_zero_v8f32)
                dp_acc = fx.as_ir_value(c_zero_v8f32)
                for ks in range_constexpr(ND):
                    col = fx.Index(ks * WMMA_K) + klane * WMMA_LANE_K
                    s_acc = fmha.wmma_acc(
                        a_q_ap.from_lds(lds_buf, lane16, col),
                        a_k_ap.from_lds(lds_buf, kv_row_local, col),
                        s_acc,
                    )
                    dp_acc = fmha.wmma_acc(
                        a_do_ap.from_lds(lds_buf, lane16, col),
                        a_v_ap.from_lds(lds_buf, kv_row_local, col),
                        dp_acc,
                    )

                p_vals = []
                ds_vals = []
                for e in range_constexpr(8):
                    q_g = fx.Int32(start_q_i) + fx.Int32(klane * fx.Index(WMMA_LANE_K)) + fx.Int32(e)
                    lse = load_row_f32(l_ptr, q_g)
                    dlt = load_row_f32(dlt_ptr, q_g)
                    # Scale before the subtract, so the exponent is a plain
                    # difference rather than the FMA form AOTriton flags in
                    # ROCm/aotriton#54. LSE arrives natural-log, hence log2e.
                    s = fastmath.mul(fx.Float32(Vec(s_acc)[e]), sm_log2e)
                    if const_expr(BIAS_TYPE):
                        # This half holds one kv column and eight q rows, so
                        # the eight bias entries are `stride_b_seq_q` apart and each
                        # is its own load -- the standalone dK/dV kernel pays
                        # exactly this. Clamped, since a dead row still issues
                        # the load and `dead` below is what removes it.
                        _bq = fx.Index(q_g) if (q_g < seqlen_q_i32) else fx.Index(0)
                        _bo = _b_head + (_q_row_off_v + _bq) * fx.Index(stride_b_seq_q) + fx.Index(kv_row_i32)
                        _bv = fx.Float32(fx.as_dsl_value(load_bias_1(_bo)).to(fx.Float32))
                        s = fastmath.add(s, fastmath.mul(_bv, fx.Float32(_LOG2E)))
                    s = fastmath.sub(s, fastmath.mul(fx.Float32(lse), fx.Float32(_LOG2E)))
                    if const_expr(MASKED):
                        dead = q_g >= seqlen_q_i32
                        if const_expr(CAUSAL):
                            # A key is visible to a query iff
                            # `q - w_left <= k <= q + w_right`. Signed
                            # throughout: `q - w_left` is negative for every
                            # row when the left edge is unbounded, which is how
                            # plain causal reaches this path.
                            dead = dead | (kv_row_i32 > q_g + wr_i32)
                            dead = dead | (kv_row_i32 < q_g - wl_i32)
                        s = fx.Float32(c_neg_inf if dead else fx.Float32(s))
                    p = rocdl.exp2(f32_ty, fx.as_ir_value(s))
                    p_vals.append(p)
                    ds_vals.append(
                        fastmath.mul(
                            fx.Float32(p),
                            fastmath.sub(fx.Float32(Vec(dp_acc)[e]), fx.Float32(dlt)),
                        )
                    )

                # P and dS are already in operand layout: their row index is
                # the query axis, which is exactly what dV and dK contract
                # over. No rearrangement, only a narrowing to 16 bits.
                p_pack = pack_v8(p_vals)
                ds_pack = pack_v8(ds_vals)
                for db in range_constexpr(ND):
                    d_row = fx.Index(db * WMMA_N) + lane16
                    t_col = klane * WMMA_LANE_K
                    dv_accs[db] = fmha.wmma_acc(a_dot_ap.from_lds(lds_buf, d_row, t_col), p_pack, dv_accs[db])
                    dk_accs[db] = fmha.wmma_acc(a_qt_ap.from_lds(lds_buf, d_row, t_col), ds_pack, dk_accs[db])

                # Every wave must be done reading the Q/dO tiles before the
                # next iteration overwrites them.
                gpu.barrier()
                return dk_accs + dv_accs

            init = [fx.as_ir_value(c_zero_v8f32) for _ in range_constexpr(2 * ND)]
            if const_expr(CAUSAL):
                # The roles of the two axes swap relative to the forward pass:
                # here a *key* block is fixed and the *query* blocks are cut
                # into runs, so the window bounds swap with them. AOTriton does
                # the same thing by passing `window_right, window_left` in that
                # order to `calculate_intervals`.
                reg = fmha.decompose_causal_regions(
                    start_k, seqlen_k_i32, seqlen_q_i32, wr_i32, wl_i32, KV_TILE, Q_STEP, alive
                )
            else:
                reg = plain_regions(seqlen_q_i32, Q_STEP, alive)
            step_i32 = fx.Int32(Q_STEP)
            n_left, n_full, n_right = reg.n_left, reg.n_full, reg.n_right
            left_col0, full_col0, right_col0 = reg.left_col0, reg.full_col0, reg.right_col0

            results = init
            for _col, iargs in range(fx.Index(full_col0), fx.Index(full_col0 + n_full * step_i32), Q_STEP, init=init):
                results = yield q_body(_col, iargs, False)

            def masked_col(i_idx):
                """Query-block origin for masked iteration `i`: the left run,
                then the right one. Discontinuous at the seam."""
                _i = fx.Int32(i_idx)
                return left_col0 + _i * step_i32 if _i < n_left else right_col0 + (_i - n_left) * step_i32

            for _mi, iargs in range(fx.Index(0), fx.Index(n_left + n_right), 1, init=results):
                results = yield q_body(fx.Index(masked_col(_mi)), iargs, True)

            # ---- Epilogue ----
            # Accumulator element `e` is head-dim `db*16 + klane*8 + e` at key
            # row `lane16`, so each store is 8 contiguous columns of one row.
            # dK carries the softmax scale; dV does not (AOTriton scales dk and
            # dq by sm_scale and leaves dv alone).
            dk_ap = fmha.Aperture(qk_cols)
            dv_ap = fmha.Aperture(vo_cols)
            scale_vec = Vec.from_elements([sm_scale], fx.Float32).broadcast_to(8).ir_value()
            row_in_tile = wave_id * fx.Index(WMMA_N) + lane16

            def write_dk(row, col, val):
                store_global_v8(dk_ptr, dk_tbase(start_k), dk_toff(row, col), val)

            def write_dv(row, col, val):
                store_global_v8(dv_ptr, dv_tbase(start_k), dv_toff(row, col), val)

            if kv_row_i32 < seqlen_k_i32:
                for db in range_constexpr(ND):
                    col = fx.Index(db * WMMA_N) + klane * WMMA_LANE_K
                    dk_v = Vec(fastmath.mul(results[db], scale_vec)).to(elem_dtype).ir_value()
                    dv_v = Vec(results[ND + db]).to(elem_dtype).ir_value()
                    fmha.write_v8(dk_ap, write_dk, row_in_tile, col, dk_v)
                    fmha.write_v8(dv_ap, write_dv, row_in_tile, col, dv_v)

        # ==================================================================
        # Role 2: dQ. Own Q_TILE query rows, walk the key axis.
        # ==================================================================
        def dq_role():
            q_blk_raw = pid - num_kv_blocks
            if const_expr(LPT_TILE_ORDER):
                # Longest-processing-time-first. Under causal a query block's
                # cost grows with its index and grid.x issues in increasing
                # order, so without this the most expensive blocks land in the
                # tail. Only the dQ half is reversed -- the dK/dV half's cost
                # already decreases with its index. `num_q_blocks`, not this
                # sequence's block count: the reversal must be a permutation of
                # the *grid*, whose extent the host sized from Max_seqlen_q.
                q_blk = num_q_blocks - fx.Int32(1) - q_blk_raw
            else:
                q_blk = q_blk_raw
            start_q_i32 = q_blk * fx.Int32(Q_TILE)
            start_q = fx.Index(start_q_i32)
            alive = start_q_i32 < seqlen_q_i32

            # This wave's 16 query rows. Clamped for addressing, gated for use.
            q_row_i32 = start_q_i32 + fx.Int32(wave_id * fx.Index(WMMA_M) + lane16)
            q_row_in_tile = wave_id * fx.Index(WMMA_M) + lane16
            q_rows_ax = fmha.MaskedAxis(seqlen_q_v)
            q_ap = fmha.Aperture(qk_cols, rows=q_rows_ax)
            do_ap = fmha.Aperture(vo_cols, rows=q_rows_ax)

            # **The tile base must be clamped, not just the row within it.**
            # A dead program still addresses `row_off + start_q` rows in, which
            # for a stacked layout runs past the whole allocation rather than
            # merely past this sequence.
            q_start_safe = fx.Index(start_q if alive else fx.Index(0))
            q_tile_base = q_tbase(q_start_safe)
            do_tile_base = do_tbase(q_start_safe)

            def fetch_q_reg(row, col):
                return load_global_v8(q_ptr, q_tile_base, q_toff(row, col))

            def fetch_do_reg(row, col):
                return load_global_v8(do_ptr, do_tile_base, do_toff(row, col))

            # Q and dO stay resident in registers across the whole key loop:
            # they are this wave's own 16 rows and are read once. That is
            # `head_dim / 4` VGPRs each, on top of the `head_dim / 2` the dQ
            # accumulator costs.
            row_ok, row_safe = q_rows_ax.gate(fx.Index(q_row_i32), q_row_in_tile)
            q_packs = []
            do_packs = []
            for ks in range_constexpr(ND):
                col = fx.Index(ks * WMMA_K) + klane * WMMA_LANE_K
                q_packs.append(q_ap.read_v8(fetch_q_reg, row_safe, col, row_ok))
                do_packs.append(do_ap.read_v8(fetch_do_reg, row_safe, col, row_ok))

            lse_lane = load_row_f32(l_ptr, q_row_i32)
            dlt_lane = load_row_f32(dlt_ptr, q_row_i32)
            lse_log2e = fastmath.mul(fx.Float32(lse_lane), fx.Float32(_LOG2E))

            def kv_body(start_k_i, args, MASKED):
                dq_accs = listify(args, ND)
                start_k_v = fx.Index(start_k_i)

                fmha.stage(b_k_ap, lds_buf, fetch_k(start_k_v), load_row_in_batch, load_col_base, fx.Index(KV_STEP))
                fmha.stage(b_v_ap, lds_buf, fetch_v(start_k_v), load_row_in_batch, load_col_base, fx.Index(KV_STEP))
                fmha.publish_transposed(
                    b_kt_ap, kt_tr, lds_buf, fmha.read_transposed(b_kt_ap, kt_tr, fetch_k_tr(start_k_v))
                )
                gpu.barrier()

                # S^T[kv][q] and dP^T[kv][q]: element `e` is key
                # `klane*8 + e` at query `lane16`. This is the forward
                # kernel's GEMM1 orientation, and it is forced -- dQ contracts
                # over the key axis, so the scores must carry the key axis as
                # their row index.
                st_acc = fx.as_ir_value(c_zero_v8f32)
                dpt_acc = fx.as_ir_value(c_zero_v8f32)
                for ks in range_constexpr(ND):
                    col = fx.Index(ks * WMMA_K) + klane * WMMA_LANE_K
                    st_acc = fmha.wmma_acc(b_k_ap.from_lds(lds_buf, lane16, col), q_packs[ks], st_acc)
                    dpt_acc = fmha.wmma_acc(b_v_ap.from_lds(lds_buf, lane16, col), do_packs[ks], dpt_acc)

                if const_expr(BIAS_TYPE):
                    # The mirror of the other half: one q row and eight
                    # *adjacent* kv columns, so one v8 covers all eight and the
                    # load hoists out of the element loop.
                    _bq = fx.Index(q_row_i32) if (q_row_i32 < seqlen_q_i32) else fx.Index(0)
                    _bo = (
                        _b_head
                        + (_q_row_off_v + _bq) * fx.Index(stride_b_seq_q)
                        + fx.Index(start_k_i)
                        + klane * fx.Index(WMMA_LANE_K)
                    )
                    _bvec = load_bias_8(_bo)

                ds_vals = []
                for e in range_constexpr(8):
                    kv_g = fx.Int32(start_k_i) + fx.Int32(klane * fx.Index(WMMA_LANE_K)) + fx.Int32(e)
                    s = fastmath.mul(fx.Float32(Vec(st_acc)[e]), sm_log2e)
                    if const_expr(BIAS_TYPE):
                        _bv = fx.Float32(Vec(_bvec)[e].to(fx.Float32))
                        s = fastmath.add(s, fastmath.mul(_bv, fx.Float32(_LOG2E)))
                    s = fastmath.sub(s, lse_log2e)
                    if const_expr(MASKED):
                        dead = kv_g >= seqlen_k_i32
                        if const_expr(CAUSAL):
                            dead = dead | (kv_g > q_row_i32 + wr_i32)
                            dead = dead | (kv_g < q_row_i32 - wl_i32)
                        s = fx.Float32(c_neg_inf if dead else fx.Float32(s))
                    p = rocdl.exp2(f32_ty, fx.as_ir_value(s))
                    ds_vals.append(
                        fastmath.mul(
                            fx.Float32(p),
                            fastmath.sub(fx.Float32(Vec(dpt_acc)[e]), fx.Float32(dlt_lane)),
                        )
                    )

                ds_pack = pack_v8(ds_vals)
                for db in range_constexpr(ND):
                    d_row = fx.Index(db * WMMA_N) + lane16
                    dq_accs[db] = fmha.wmma_acc(
                        b_kt_ap.from_lds(lds_buf, d_row, klane * WMMA_LANE_K), ds_pack, dq_accs[db]
                    )
                gpu.barrier()
                return dq_accs

            init = [fx.as_ir_value(c_zero_v8f32) for _ in range_constexpr(ND)]
            if const_expr(CAUSAL):
                reg = fmha.decompose_causal_regions(
                    start_q, seqlen_q_i32, seqlen_k_i32, wl_i32, wr_i32, Q_TILE, KV_STEP, alive
                )
            else:
                reg = plain_regions(seqlen_k_i32, KV_STEP, alive)
            step_i32 = fx.Int32(KV_STEP)
            n_left, n_full, n_right = reg.n_left, reg.n_full, reg.n_right
            left_col0, full_col0, right_col0 = reg.left_col0, reg.full_col0, reg.right_col0

            results = init
            for _col, iargs in range(fx.Index(full_col0), fx.Index(full_col0 + n_full * step_i32), KV_STEP, init=init):
                results = yield kv_body(_col, iargs, False)
            results = listify(results, ND)

            def masked_col(i_idx):
                _i = fx.Int32(i_idx)
                return left_col0 + _i * step_i32 if _i < n_left else right_col0 + (_i - n_left) * step_i32

            for _mi, iargs in range(fx.Index(0), fx.Index(n_left + n_right), 1, init=results):
                results = yield kv_body(fx.Index(masked_col(_mi)), iargs, True)
            results = listify(results, ND)

            dq_ap = fmha.Aperture(qk_cols)
            scale_vec = Vec.from_elements([sm_scale], fx.Float32).broadcast_to(8).ir_value()

            def write_dq(row, col, val):
                store_global_v8(dq_ptr, dq_tbase(q_start_safe), dq_toff(row, col), val)

            if row_ok:
                for db in range_constexpr(ND):
                    col = fx.Index(db * WMMA_N) + klane * WMMA_LANE_K
                    dq_v = Vec(fastmath.mul(results[db], scale_vec)).to(elem_dtype).ir_value()
                    fmha.write_v8(dq_ap, write_dq, q_row_in_tile, col, dq_v)

        # The pid split. Workgroup-uniform, so the barriers inside each role
        # are safe: every thread of a workgroup takes the same arm.
        #
        # Written as a call per arm rather than inline because
        # `ast_rewriter._collect_assigned_vars` treats a name assigned inside a
        # dynamic `if` as carried state that the `scf.if` must yield -- and
        # neither of these bodies has anything to hand back. A call assigns
        # nothing at this level, so nothing is collected.
        if pid < num_kv_blocks:
            dkdv_role()
        else:
            dq_role()

    @flyc.jit
    def launch_bwd_fuse(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        Bias: fx.Pointer,
        DO: fx.Pointer,
        DK: fx.Pointer,
        DV: fx.Pointer,
        DQ: fx.Pointer,
        LSE: fx.Pointer,
        Delta: fx.Pointer,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        batch_size: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        num_kv_blocks: fx.Int32,
        num_q_blocks: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
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
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()

        # Grid x is the concatenation of the two roles' program spaces. Both
        # extents key on Max_seqlen: under varlen there is no single sequence
        # length, so every sequence gets the longest one's worth of programs
        # and the short ones exit with empty region counts.

        launcher = bwd_fuse_kernel(
            Q,
            K,
            V,
            Bias,
            DO,
            DK,
            DV,
            DQ,
            LSE,
            Delta,
            seqinfo_q0,
            seqinfo_q1,
            seqinfo_k0,
            seqinfo_k1,
            varlen_bits,
            batch_size,
            num_seqlens,
            max_seqlen_q,
            max_seqlen_k,
            num_kv_blocks,
            num_q_blocks,
            window_left,
            window_right,
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
            stride_dq_batch,
            stride_dq_head,
            stride_dq_seq,
            stride_b_batch,
            stride_b_head,
            stride_b_seq_q,
        )

        if const_expr(WAVES_PER_EU is not None):
            _wpe = int(WAVES_PER_EU)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.WAVES_PER_EU"] = ir.IntegerAttr.get(T.i32, _wpe)
        if const_expr(FLAT_WORK_GROUP_SIZE is not None):
            _fwgs = int(FLAT_WORK_GROUP_SIZE)
            if const_expr(_fwgs >= 1):
                flat_wg_attr = ir.StringAttr.get(f"{_fwgs},{_fwgs}")
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.FLAT_WORK_GROUP_SIZE"] = flat_wg_attr

        passthrough_entries = []
        if const_expr(SCHED_STRATEGY):
            passthrough_entries.append(
                ir.ArrayAttr.get([ir.StringAttr.get("amdgpu-sched-strategy"), ir.StringAttr.get(SCHED_STRATEGY)])
            )
        if const_expr(DENORMALS_ARE_ZERO):
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("denormal-fp-math-f32"),
                        ir.StringAttr.get("preserve-sign,preserve-sign"),
                    ]
                )
            )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

        launcher.launch(
            grid=(
                fx.Index(num_kv_blocks) + fx.Index(num_q_blocks),
                fx.Index(num_head_q),
                fx.Index(num_seqlens if num_seqlens != fx.Int32(0) else batch_size),
            ),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    launch_bwd_fuse.compile_hints = {
        "FAST_FP_MATH": FAST_FP_MATH,
        "UNSAFE_FP_MATH": UNSAFE_FP_MATH,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    def _launch(
        Q,
        K,
        V,
        DO,
        DK,
        DV,
        DQ,
        L,
        Delta,
        batch_size,
        seqlen_q,
        seqlen_k=None,
        num_seqlens=0,
        scale=None,
        stream=None,
        window=None,
        varlen=None,
        bias=None,
    ):
        """Dispatch one fused backward pass.

        `DK` and `DV` carry **num_head_q** heads, not num_head_k: this kernel
        gives GQA one program per query head and leaves the reduction to the
        caller. `L` and `Delta` are compact f32 with the layout VarlenBits
        names.
        """
        seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
        st = []
        for t, name in ((Q, "Q"), (K, "K"), (V, "V"), (DO, "DO"), (DK, "DK"), (DV, "DV"), (DQ, "DQ")):
            st.extend(abi.strides_of(t, name))
        nhq, nhk = Q.shape[1], K.shape[1]
        hqk, hvo = Q.shape[3], V.shape[3]
        if nhq % nhk:
            raise ValueError(f"num_heads_q ({nhq}) must be divisible by num_heads_k ({nhk})")
        if DK.shape[1] != nhq or DV.shape[1] != nhq:
            raise ValueError(
                f"DK/DV must carry num_head_q ({nhq}) heads -- GQA is reduced by the "
                f"caller, not in the kernel. Got {DK.shape[1]} and {DV.shape[1]}"
            )
        # This kernel has no separate device-side causal type; like dkdv and dq
        # it collapses every masking mode onto gSWA, so the first argument --
        # which `resolve_window` only tests against 0 -- is that derivation.
        _wl, _wr = abi.resolve_window(0 if HOST_CAUSAL_TYPE == 0 else 3, HOST_CAUSAL_TYPE, window, seqlen_q, seqlen_k)
        _vb, _sq0, _sq1, _sk0, _sk1, _mq, _mk = abi.varlen_args(
            False, varlen, seqlen_q, seqlen_k, Q, batch_size, num_seqlens
        )
        _scale = float(scale) if scale is not None else float(sm_scale)
        # `with_db=False`: dB is the standalone dQ kernel's, so the DB slot is
        # discarded rather than being a kernarg this one does not have.
        _bp, _, _sb0, _sb1, _sb2 = abi.bias_args(BIAS_TYPE, False, bias, None, Q)
        abi.run_compiled(
            _COMPILED,
            launch_bwd_fuse,
            *[abi.ptr_arg(t) for t in (Q, K, V)],
            _bp,
            *[abi.ptr_arg(t) for t in (DO, DK, DV, DQ, L, Delta)],
            _sq0,
            _sq1,
            _sk0,
            _sk1,
            _vb,
            int(batch_size),
            int(num_seqlens),
            _mq,
            _mk,
            (_mk + KV_TILE - 1) // KV_TILE,
            (_mq + Q_TILE - 1) // Q_TILE,
            _wl,
            _wr,
            nhq,
            nhk,
            hqk,
            hvo,
            _scale,
            *st,
            _sb0,
            _sb1,
            _sb2,
            stream if stream is not None else fx.Stream(None),
        )

    return _launch
