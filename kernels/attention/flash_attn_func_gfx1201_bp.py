# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Flash Attention for gfx1201, scheduled with **binding prefetch**.

Variant of ``flash_attn_func_gfx1201.py`` that changes only how K/V tiles are
staged. See ``gfx1201_fmha.md`` for the terminology.

gfx1201 has no direct global->LDS copy (RDNA3 dropped it and RDNA4 did not
bring it back), so a KV tile must transit VGPRs. The only available form of
latency hiding is therefore a *binding* prefetch: issue the global loads early,
hold the values in registers, and consume them later. Prefetch distance is
bounded by the register budget rather than by LDS.

The baseline kernel prefetches V this way but loads K at distance 0 -- its
``coop_load_k`` emits ``global_load; s_wait_loadcnt 0x0; ds_store`` with no work
in between, so the load latency is fully exposed. Here **both** K and V ride the
loop in registers at distance 1:

    prologue: load K[0], V[0] -> registers
    iteration i (carrying K[i], V[i] in registers):
        store K[i] -> LDS ; barrier ; issue load K[i+1]   <- flies over GEMM1
        GEMM1 (S = K @ Q^T) ; online softmax
        store V[i] -> LDS ; barrier ; issue load V[i+1]   <- flies over GEMM2
        GEMM2 (O += V^T @ P)
        yield K[i+1], V[i+1]

Barrier count per iteration is unchanged (2). The added cost is one more live
register set (K), which is the tradeoff this variant exists to measure.

Correctness of the barrier placement: the barrier after ``store K[i]`` separates
GEMM1's K reads in iteration i-1 from the K writes in iteration i, and the
barrier after ``store V[i]`` separates GEMM2's V reads in iteration i-1 from the
V writes in iteration i.

V is staged **transposed** in LDS (``V^T[d][kv]``) and filled with
``global_load_tr_b128``, whose hardware 16x16 transpose delivers each lane the
8 kv-elements it needs contiguously. GEMM2 therefore reads one vector per
operand instead of 8 strided scalar loads, and the LDS store stays contiguous
rather than becoming a 16-way scatter. Worth +2.7% at N>=4096.

Status: correctness-oriented; see ``gfx1201_fmha.md`` for measured component
costs and for two approaches that were tried and rejected (bypassing LDS for V,
double-buffering K).

Supported configs (enforced): head_dim 64 or 128, BLOCK_M=128, BLOCK_N=32,
f16/bf16, causal and non-causal.

WMMA 16x16x16 register layout (wave32):
  - A/B operand: v8bf16 per lane (lane16 = row/col, klane*8 = K-offset)
  - C/D result: v8f32 per lane, element si = C[klane*8+si][lane16]

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads, head_dim).
Grid:   (batch * num_q_tiles * num_heads,)
Block:  (256,) -- 8 waves x 32 threads/wave.

Requires: head_dim % 32 == 0, head_dim >= 64.
"""

import math as host_math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
)
from flydsl.expr import math as fmath
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import ArithValue, _to_raw as _raw

KERNEL_NAME = "flash_attn_func_gfx1201_bp_kernel"
_LOG2E = host_math.log2(host_math.e)

# `dtype_to_elem_type` and `_run_compiled` are inlined copies of
# `kernels.common.kernels_common.dtype_to_elem_type` and
# `kernels.common.tensor_shim._run_compiled`. They are duplicated on purpose:
# this directory is a self-contained prototype that must run with the cwd set
# to it and no PYTHONPATH, which puts `kernels.*` out of reach. Fold them back
# into the shared modules if this graduates out of prototype status.


def dtype_to_elem_type(dtype_str: str):
    """Map a dtype string to its FlyDSL numeric type."""
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f32', 'f16', or 'bf16')")


def _run_compiled(exe, *args):
    """First call: ``flyc.compile(exe, *args)`` compiles **and** executes the kernel.
    Subsequent calls: fast dispatch via the cached ``CompiledFunction``.
    """
    cf = getattr(exe, "_cf", None)
    if cf is None:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    else:
        cf(*args)


def _llvm_value(value):
    """Unwrap FlyDSL scalar/vector wrappers for LLVM pointer load ops."""
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _pointer_to_llvm_ptr(ptr) -> ir.Value:
    """Convert a FlyDSL pointer argument to the LLVM pointer used by raw loads."""
    ptr_i64 = arith.index_cast(T.i64, fx.ptrtoint(ptr))
    return _llvm.IntToPtrOp(_llvm_ptr_ty(), ptr_i64).result


def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return _llvm.LoadOp(result_type, _llvm_value(ptr)).result


def _pointer_store(value: ir.Value, ptr: ir.Value):
    return _llvm.StoreOp(_llvm_value(value), _llvm_value(ptr))


def vo_chunks(head_dim, block_n, qk_shards, pad=4):
    """V staging passes needed to keep the *padded* K+V tile inside 64 KiB.

    Sharding V/O across waves means every wave's slice is live at once, so the
    full head_dim of V^T would have to be resident -- 69888 B at head_dim 512,
    over the cap. Staging V in `nc` passes makes only head_dim/nc columns
    resident, which restores the padding and with it conflict-free LDS. Costs
    one extra barrier pair per extra pass. Returns 1 whenever one pass fits.
    """
    for nc in (1, 2, 4, 8):
        if head_dim % nc:
            continue
        cols = head_dim // nc
        if cols % (qk_shards * 16):        # each wave needs whole 16-col chunks
            continue
        if block_n * (head_dim + pad) * 2 + cols * (block_n + pad) * 2 <= 65536:
            return nc
    return 1


# Head-dimension sharding policy, exported so the interface can agree on
# BLOCK_M (which it needs for seq_len padding) without building the kernel.
# One shard per 128 head-dim columns; head_dim <= 128 stays unsharded.
_BP_TARGET_WAVES = 8
_BP_ROWS_PER_WAVE = 16


def bp_qk_shards(head_dim):
    """Waves cooperating on one Q row-tile at this head_dim."""
    return max(1, head_dim // 128)


# Q row-tiles per workgroup, where the default of TARGET_WAVES/shards is not
# the fastest. head_dim 224 has THREADS_PER_ROW_LOAD=14, whose cooperative-load
# geometry at BLOCK_M=128 spills 76 registers (53.4 TFLOPS) against 11 at
# BLOCK_M=64 (69.5). Same awkward width that spills 101 on the baseline there.
_BP_Q_TILES_BY_HEAD_DIM = {224: 4}


def bp_q_tiles(head_dim, block_n=32, shards=None):
    """Q row-tiles per workgroup: TARGET_WAVES traded against the shard count.

    The V transpose tiling no longer has to divide evenly across the waves --
    tail tiles are guarded at the LDS store -- so this is otherwise free.
    """
    shards = bp_qk_shards(head_dim) if shards is None else shards
    return _BP_Q_TILES_BY_HEAD_DIM.get(head_dim, max(1, _BP_TARGET_WAVES // shards))


def bp_block_m(head_dim):
    """BLOCK_M for this head_dim: Q row-tiles are traded for shards."""
    return _BP_ROWS_PER_WAVE * bp_q_tiles(head_dim)


def build_flash_attn_func_bp_module_primary(
    num_heads,
    head_dim,
    causal=True,
    dtype_str="bf16",
    sm_scale=None,
    waves_per_eu=2,
    flat_work_group_size=None,
    block_m=None,
    block_n=None,
    qk_shards=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    path_tag="auto",
):
    """Build gfx1201 flash_attn_func (BN=32 + rocdl.exp2 + pipelined GEMM2 + overlapped V load)."""

    # ---- WMMA / wave32 constants ----
    WARP_SIZE = 32
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    K_SUB_N = 32
    ROWS_PER_WAVE = WMMA_M

    BLOCK_M = block_m if block_m is not None else 128
    BLOCK_N = block_n if block_n is not None else 32

    assert (
        BLOCK_N % K_SUB_N == 0
    ), f"BLOCK_N ({BLOCK_N}) must be a multiple of K_SUB_N ({K_SUB_N})"
    assert (
        BLOCK_M % ROWS_PER_WAVE == 0
    ), f"BLOCK_M ({BLOCK_M}) must be a multiple of {ROWS_PER_WAVE}"

    N_SUB_TILES = BLOCK_N // K_SUB_N
    NUM_S_ACCS = N_SUB_TILES * 2
    NUM_S_VALS = NUM_S_ACCS * 8

    # Stage 1 of the binding-prefetch variant targets correctness on a
    # deliberately narrow config set; widen only with matching tests.
    if head_dim % 16 or not (16 <= head_dim <= 512):
        raise ValueError(f"binding-prefetch variant needs 16 <= head_dim <= 512, %16==0, got {head_dim}")
    if BLOCK_N != 32:
        raise ValueError(f"binding-prefetch variant supports BLOCK_N=32 only, got {BLOCK_N}")
    if causal and NUM_S_VALS != 16:
        # The causal mask below is unrolled into 16 explicitly named scalars.
        raise ValueError(f"causal masking assumes NUM_S_VALS == 16, got {NUM_S_VALS}")

    # Head-dimension sharding. QK_SHARDS waves cooperate on one Q row-tile,
    # each reducing over its own head_dim slice in GEMM1 and owning the matching
    # V/O column slice in GEMM2. QK_SHARDS == 1 is the unsharded kernel: every
    # sharded construct below is behind `const_expr(QK_SHARDS > 1)`, so this
    # path must trace to identical IR. See plan_gfx1201_large_hdim.md.
    QK_SHARDS = qk_shards if qk_shards is not None else bp_qk_shards(head_dim)
    QK_SLICE = head_dim // QK_SHARDS          # head-dim columns per wave in GEMM1

    # V/O columns are staged in VO_CHUNKS passes. Within a pass the resident
    # window is VO_CHUNK_COLS wide and contiguous, and wave s owns VO_SLICE of
    # it -- partition (B): re-partitioning per pass keeps the LDS window
    # contiguous, so no global->local d remap is needed.
    VO_CHUNKS = vo_chunks(head_dim, BLOCK_N, QK_SHARDS)
    VO_CHUNK_COLS = head_dim // VO_CHUNKS     # V columns resident per pass
    VO_SLICE = VO_CHUNK_COLS // QK_SHARDS     # V/O columns per wave per pass
    if head_dim % QK_SHARDS or QK_SLICE % WMMA_K:
        # K_STEPS_QK = QK_SLICE // WMMA_K truncates otherwise, silently dropping
        # part of the reduction: head_dim 224 with 4 shards gives a 56-wide
        # slice, of which only 48 would be reduced (measured rel err 0.97).
        raise ValueError(
            f"head_dim {head_dim} with {QK_SHARDS} shards gives a {QK_SLICE}-wide "
            f"slice, which must be a multiple of WMMA_K={WMMA_K}"
        )
    if VO_SLICE % WMMA_N:
        raise ValueError(
            f"V/O slice {VO_SLICE} must be a multiple of WMMA_N={WMMA_N}"
        )

    # Keep the workgroup at TARGET_WAVES by trading Q row-tiles for shards, so
    # BLOCK_M shrinks as QK_SHARDS grows. QK_SHARDS=1 gives BLOCK_M=128 as before.
    Q_TILES_PER_BLOCK = bp_q_tiles(head_dim, BLOCK_N, QK_SHARDS)
    BLOCK_M = ROWS_PER_WAVE * Q_TILES_PER_BLOCK
    NUM_WAVES = Q_TILES_PER_BLOCK * QK_SHARDS
    # The V TR tiling need not divide evenly across the waves: tail tiles are
    # guarded at the LDS store. Requiring divisibility used to force head_dim
    # 160 down to 4 waves, which cost it 89.1 -> 70.0 TFLOPS.
    _V_TR_TILES = (VO_CHUNK_COLS // WMMA_N) * (BLOCK_N // WMMA_K)
    V_TR_LOADS = (_V_TR_TILES + NUM_WAVES - 1) // NUM_WAVES
    V_TR_NEEDS_GUARD = V_TR_LOADS * NUM_WAVES != _V_TR_TILES
    if flat_work_group_size is None:
        flat_work_group_size = NUM_WAVES * WARP_SIZE
    BLOCK_SIZE = flat_work_group_size

    BLOCK_N_OUT = BLOCK_N

    NUM_PREFETCH_K = 1
    NUM_PREFETCH_V = 1

    # LLVM's amdgpu-sched-strategy function attribute; "" leaves the default
    # GCN scheduler in place. See the passthrough block in the launch wrapper
    # for what this buys. Measured at BATCH=2 H=12 N=4096 d=128 f16:
    # causal 85.6 -> 88.5 TFLOPS, non-causal 91.4 -> 91.9.
    SCHED_STRATEGY = os.environ.get("FMHA_SCHED_STRATEGY", "max-memory-clause")

    # global_load_tr_b128 transposes an 8x8 tile of 16-bit elements across each
    # group of 8 lanes, so one wave-wide TR load produces a 16(d) x 16(kv) block
    # already in WMMA-operand layout. Split those blocks over the waves.
    V_TR_D_BLOCKS = VO_CHUNK_COLS // WMMA_N

    K_STEP_QK = WMMA_K
    K_STEPS_QK = QK_SLICE // K_STEP_QK        # GEMM1 K-steps for this wave's slice
    WMMA_LANE_K = 8

    D_CHUNK = WMMA_N
    D_CHUNKS = VO_SLICE // D_CHUNK            # accs per wave per chunk
    O_ACCS = VO_CHUNKS * D_CHUNKS             # accs live across the KV loop

    PV_K_STEP = WMMA_K
    PV_K_STEPS = K_SUB_N // PV_K_STEP

    # BLOCK_M is ROWS_PER_WAVE * Q_TILES_PER_BLOCK by construction; with
    # QK_SHARDS > 1 NUM_WAVES exceeds the Q-tile count, so the old
    # `BLOCK_M % NUM_WAVES` invariant no longer applies.
    assert BLOCK_M % ROWS_PER_WAVE == 0
    assert head_dim % WMMA_K == 0  # WMMA_K, not 32: see the head_dim guard above
    assert head_dim >= WMMA_K
    assert dtype_str in ("f16", "bf16")

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim
    CAUSAL = causal
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    # LDS layout -- K uses padding instead of XOR swizzle; V row-major with padding
    # Padding costs BLOCK_N*4*2 B on K and head_dim*4*2 B on V, which at
    # head_dim 512 pushes K+V to 69888 B, over the 64 KiB workgroup cap. Drop it
    # there: 32768 + 32768 = 65536 lands exactly at the limit. Bank conflicts
    # return; an XOR swizzle would avoid both, and is the follow-up.
    _LDS_PAD = 4  # chunking bounds the V window, so padding always fits
    K_STRIDE = HEAD_DIM + _LDS_PAD  # padding to reduce bank conflicts (no swizzle)
    # V is staged TRANSPOSED: V^T[d][kv], so GEMM2 reads 8 consecutive kv for a
    # fixed d as one contiguous vector instead of 8 strided scalar loads.
    # +4 makes the row stride 36 elems = 72 B: 18 dwords, so lane16*18 mod 32
    # hits 16 distinct banks (conflict-free) while staying 8-byte aligned.
    VT_STRIDE = BLOCK_N + _LDS_PAD

    ENABLE_LDS_VEC16 = os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1") == "1"
    VEC_WIDTH = 16 if ENABLE_LDS_VEC16 else 8
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    # Cover BLOCK_N rows with ceil() batches, not floor(). Flooring silently
    # dropped rows whenever ROWS_PER_BATCH_LOAD neither reached BLOCK_N nor
    # divided it: head_dim 160/192/224 give 25/21/18, so BLOCK_N // that == 1
    # and only 25/21/18 of the 32 KV rows reached LDS. The rest was stale LDS,
    # which surfaced as NaN. Same defect as the baseline kernel's; it was
    # unreachable here while bp was gated to head_dim 64/128.
    NUM_BATCHES_KV = (BLOCK_N + ROWS_PER_BATCH_LOAD - 1) // ROWS_PER_BATCH_LOAD
    KV_NEEDS_GUARD = NUM_BATCHES_KV * ROWS_PER_BATCH_LOAD != BLOCK_N

    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_V_TILE_SIZE = VO_CHUNK_COLS * VT_STRIDE
    LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * LDS_K_TILE_SIZE
    LDS_V_BASE = LDS_K_TOTAL_SIZE
    LDS_V_TOTAL_SIZE = NUM_PREFETCH_V * LDS_V_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_SIZE

    # The cross-shard S reduction aliases the V region rather than allocating
    # its own: V is written to LDS only *after* softmax, so between the
    # post-K-store barrier and that write the V tile holds the previous
    # iteration's data, already consumed by the previous GEMM2.
    RED_F32_PER_WAVE = NUM_S_VALS * WARP_SIZE
    RED_F32_TOTAL = NUM_WAVES * RED_F32_PER_WAVE
    # Alias the V region when it is large enough (free); otherwise append a
    # dedicated buffer. The V window shrinks with VO_CHUNKS, and at small
    # head_dim it is simply narrow, so the alias is not always available.
    RED_ALIASES_V = QK_SHARDS == 1 or RED_F32_TOTAL * 4 <= LDS_V_TOTAL_SIZE * 2
    if not RED_ALIASES_V:
        LDS_KV_TOTAL_SIZE += (RED_F32_TOTAL * 4 + 1) // 2  # in elem_dtype units

    # FlyDSL's `dtype_to_elem_type` returns a Numeric class, which is what the
    # Vector API (`Vec.make_type`, `.to(...)`) and `fx.Array` require. aiter's
    # same-named helper returns a raw MLIR `ir.Type` instead; the places that
    # need that form (LLVM GEPs) derive it via `.ir_type` inside the kernel.
    elem_numeric_cls = dtype_to_elem_type(dtype_str)

    @fx.struct
    class SharedStorage:
        kv: fx.Array[elem_numeric_cls, LDS_KV_TOTAL_SIZE, 16]

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_func_bp_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        seq_len: fx.Int32,
    ):
        elem_type = elem_numeric_cls.ir_type
        elem_dtype = elem_numeric_cls

        def _to_global_ptr_i64(ptr):
            return arith.index_cast(T.i64, fx.ptrtoint(ptr))

        def _global_load_tr_v8(base_i64, base64, off32):
            """One global_load_tr_b128: an 8x8 16-bit transpose per lane-group.

            Lane g_i supplies an address; the 8 contiguous elements there become
            column i of the group's output, so lane g_j receives
            [M_0[j] .. M_7[j]]. Verified empirically on gfx1201.
            """
            # Split as elsewhere: the (batch, head, tile) origin in 64 bits,
            # the intra-tile part in 32. Feeding LLVM `uniform_i64 +
            # zext(i32 divergent)` is what lets SelectGlobalSAddr keep the base
            # in SGPRs instead of forcing a 64-bit VGPR address pair.
            base_bytes = arith.index_cast(T.i64, _raw(fx.Index(base64) * 2))
            off_bytes = arith.extui(
                T.i64, arith.index_cast(T.i32, _raw(fx.Index(off32) * 2))
            )
            addr = arith.addi(arith.addi(base_i64, base_bytes), off_bytes)
            p = _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), addr).result
            return rocdl.global_load_tr_b128(v8f16_type, p)

        q_ptr = _pointer_to_llvm_ptr(Q)
        k_ptr = _pointer_to_llvm_ptr(K)
        v_ptr = _pointer_to_llvm_ptr(V)
        v_ptr_i64 = _to_global_ptr_i64(V)
        o_ptr = _pointer_to_llvm_ptr(O)
        fm_fast = arith.FastMathFlags.fast

        # Local fast-math arithmetic helpers — preserve fastmath flag while using
        # the lowercase op names that accept _raw() unwrapping (PR #462 pattern).
        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

        v8f32_type = Vec.make_type(8, fx.Float32)
        v8f16_type = Vec.make_type(8, elem_dtype)
        vxf16_type = Vec.make_type(VEC_WIDTH, elem_dtype)
        v4f16_type = Vec.make_type(4, elem_dtype)

        # ---- Honest-alignment LDS access ----
        # K/V rows are K_STRIDE * 2 == 264 bytes apart, so LDS addresses here are
        # only guaranteed 8-byte aligned. `fly.ptr_load` / `fly.ptr_store` emit no
        # alignment attribute (see PtrLoadOpLowering in FlyToROCDL.cpp), so LLVM
        # falls back to the vector type's ABI alignment -- 16B for v8f16, 32B for
        # v16f16. That over-promise is what makes the backend select
        # ds_load_b128 / ds_store_b128 on addresses that are not actually 16-byte
        # aligned: 2.2x slower here (39 -> 92 TFLOPS) and undefined behaviour
        # besides. Splitting into 8-byte (v4f16) accesses carries a truthful
        # `align 8` and folds back into ds_load2_b64 / ds_store2_b64, which is
        # what the older memref/`Vec.load` formulation of this kernel emitted.
        def _lds_load_v8(lds_idx):
            lo = fx.ptr_load(lds_kv + fx.Int32(lds_idx), result_type=v4f16_type)
            hi = fx.ptr_load(lds_kv + fx.Int32(lds_idx + 4), result_type=v4f16_type)
            return Vec(lo).shuffle(Vec(hi), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()

        def _lds_store_vx(vec, lds_idx):
            v = Vec(vec)
            for _i in range_constexpr(VEC_WIDTH // 4):
                part = v.shuffle(v, [_i * 4, _i * 4 + 1, _i * 4 + 2, _i * 4 + 3])
                fx.ptr_store(part, lds_kv + fx.Int32(lds_idx + _i * 4))

        def wmma_acc(a_v8, b_v8, c_v8):
            if const_expr(dtype_str == "bf16"):
                a_i16 = Vec(a_v8).bitcast(fx.Int16)
                b_i16 = Vec(b_v8).bitcast(fx.Int16)
                return rocdl.wmma_f32_16x16x16_bf16(
                    v8f32_type, _raw(a_i16), _raw(b_i16), c_v8
                ).result
            return rocdl.wmma_f32_16x16x16_f16(v8f32_type, a_v8, b_v8, c_v8).result

        seq_len_v = fx.Index(seq_len)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_kv = lds.kv.ptr

        # f32 view of the V LDS region, for the cross-shard S reduction. The
        # kv array is elem_dtype (16-bit), so go through an addrspace(3) LLVM
        # pointer: ptrtoint on a shared pointer yields the 32-bit LDS offset.
        _lds_byte_base = _raw(fx.ptrtoint(lds_kv))
        _RED_BYTE0 = (LDS_V_BASE if RED_ALIASES_V else LDS_KV_TOTAL_SIZE
                      - (RED_F32_TOTAL * 4 + 1) // 2) * 2

        def _red_addr(i_f32):
            off = fx.Int32(_RED_BYTE0) + fx.Int32(i_f32) * fx.Int32(4)
            addr = arith.addi(_lds_byte_base, _raw(off))
            return _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<3>"), addr).result

        def _red_store(i_f32, val):
            _llvm.StoreOp(_raw(val), _red_addr(i_f32))

        def _red_load(i_f32):
            return _llvm.LoadOp(ir.F32Type.get(), _red_addr(i_f32)).result

        block_id = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)

        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16
        klane = lane // 16

        # (q_tile, shard) decomposition of the wave index. At QK_SHARDS == 1
        # this is q_tile == wave_id and shard == 0, i.e. the original mapping.
        q_tile_in_block = wave_id // QK_SHARDS
        shard_id = wave_id % QK_SHARDS
        wave_q_offset = q_tile_in_block * ROWS_PER_WAVE

        # Column origins of this wave's slices. Both are 0 at QK_SHARDS == 1.
        shard_qk_off = shard_id * fx.Index(QK_SLICE)   # into Q/K head_dim
        shard_vo_off = shard_id * fx.Index(VO_SLICE)   # into V/O head_dim

        head_idx = block_id % NUM_HEADS
        batch_q_tile_id = block_id // NUM_HEADS
        num_q_tiles = (seq_len_v + BLOCK_M - 1) // BLOCK_M
        q_tile_idx = batch_q_tile_id % num_q_tiles
        batch_idx = batch_q_tile_id // num_q_tiles
        q_start = q_tile_idx * BLOCK_M

        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        def global_idx(token_idx, col):
            token = batch_idx * seq_len_v + token_idx
            return token * STRIDE_TOKEN + head_idx * HEAD_DIM + col

        # Binding prefetch always runs one tile ahead, so the final iteration
        # addresses a KV tile past the end of the sequence, and the unguarded
        # cooperative load addresses rows past BLOCK_N. Clamp the row so those
        # reads stay in bounds; the values are never consumed.
        seq_last = seq_len_v - fx.Index(1)

        # ---- Address split: 64-bit uniform base + 32-bit divergent offset ----
        #
        # The full linear element index is
        #     ((batch * seq_len) + token) * nheads * head_dim + head * head_dim + d
        # i.e. it spans all of B, S, H and D, and overflows i32 at 2G elements
        # (2 GB at f16, which real shapes reach). Only the *intra-tile* part is
        # safely 32-bit: it is bounded by
        #     max(BLOCK_M, BLOCK_N) * nheads * head_dim + head_dim
        # because the row index is relative to the tile.
        #
        # So the batch/head/tile origin stays in 64 bits, and it is uniform
        # across the wave -- which is also exactly the shape LLVM's
        # SelectGlobalSAddr folds into an SGPR base plus a 32-bit VGPR offset.
        _bh_base = batch_idx * seq_len_v * STRIDE_TOKEN + head_idx * HEAD_DIM

        def tile_base(tile_start):
            """Uniform 64-bit element base for (batch, head, tile_start)."""
            return _bh_base + tile_start * STRIDE_TOKEN

        def tile_off(row_in_tile, col):
            """Divergent 32-bit element offset inside the tile."""
            return row_in_tile * STRIDE_TOKEN + col

        def kv_addr(tile_start, row_in_tile, col):
            """(uniform base, divergent offset) for a KV row, clamped in bounds.

            tile_start is clamped first because the distance-1 prefetch can
            address a tile past the end; the in-tile row is then clamped, and
            since that can only fire when tile_start is within BLOCK_N of the
            end, the clamped row stays below BLOCK_N.
            """
            ts = fx.Index(
                ArithValue(tile_start < seq_len_v).select(tile_start, seq_last)
            )
            in_range = (ts + row_in_tile) < seq_len_v
            row = fx.Index(ArithValue(in_range).select(row_in_tile, seq_last - ts))
            return tile_base(ts), tile_off(row, col)

        def _split_ptr(ptr, base64, off32):
            """ptr + base64 (uniform, 64-bit) + off32 (divergent, 32-bit)."""
            p = buffer_ops.get_element_ptr(
                ptr, fx.Int64(base64), elem_type=elem_type
            )
            return buffer_ops.get_element_ptr(
                p, fx.Int32(off32), elem_type=elem_type
            )

        def _load_global_half_vec(ptr, base64, off32, vec_type):
            return _pointer_load(vec_type, _split_ptr(ptr, base64, off32))

        def _store_global_half(ptr, base64, off32, val):
            _pointer_store(val, _split_ptr(ptr, base64, off32))

        def load_global_f16xN(base_ptr, base64, off32):
            return _load_global_half_vec(base_ptr, base64, off32, vxf16_type)

        def load_global_v8f16(base_ptr, base64, off32):
            return _load_global_half_vec(base_ptr, base64, off32, v8f16_type)

        def _bitcast_i32(value):
            return fx.Int32(ArithValue(value).bitcast(fx.Int32.ir_type))

        def _pack_bf16_pair(lo, hi, shift, mask):
            lo_i32 = _bitcast_i32(lo)
            hi_i32 = _bitcast_i32(hi)
            return (hi_i32 & mask) | lo_i32.shrui(shift)

        def bf16_trunc_pack_v8(f32_vals):
            """Pack 8 f32 values into v8bf16 via bitwise truncation (upper 16 bits).

            On P precision, before anyone tries to raise it here:

            **There is no way to keep P in f32 through GEMM2 on gfx1201.** RDNA4
            WMMA has no F32xF32 form (ISA manual Table 41); A/B operands are
            f16/bf16/iu8/iu4/fp8 only. LLVM does define
            `v_wmma_f32_16x16x4_f32`, but it is real-ized under
            `VOP3P_Real_WMMA_gfx1250` -- gfx1250 only, not gfx12/gfx1201. The
            AOTriton idiom `acc += tl.dot(p, v.to(p.type.element_ty))` works on
            CDNA because that has `v_mfma_f32_16x16x4f32`; it has no gfx1201
            equivalent. Doing PV in f32 here would mean dropping to VALU FMA and
            giving up the matrix cores for GEMM2.

            Note also that V is *not* downcast: it reaches GEMM2 at the input
            tensor's native 16-bit width, so only P loses precision.

            Truncation is round-toward-zero. Measured against an fp64 reference
            (`accuracy_probe.py`, B=1 H=4 N=1024 d=128): no output bias (O sums
            P*V and V is zero-mean, so the one-sided P error cancels), but the
            RMS error is 1.6x torch SDPA's at bf16 (4.43e-3 vs 2.78e-3). f16 is
            already at exact parity. Switching to round-to-nearest-even --
            `x += 0x7FFF + ((x >> 16) & 1)` before the shift -- closes that gap
            exactly (2.79e-3) but costs 2-3% on bp and 2.7-5.4% on m32, so it is
            deliberately not done. Kept as truncation by decision, not oversight.
            """
            _c16 = fx.Int32(16)
            _cmask = fx.Int32(0xFFFF0000)
            pairs = []
            for j in range_constexpr(4):
                pairs.append(
                    _pack_bf16_pair(f32_vals[j * 2], f32_vals[j * 2 + 1], _c16, _cmask)
                )
            return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()

        def k_buf_base(buf_id):
            if const_expr(isinstance(buf_id, int)):
                return fx.Index(buf_id * LDS_K_TILE_SIZE)
            return buf_id * fx.Index(LDS_K_TILE_SIZE)

        def v_buf_base(buf_id):
            return fx.Index(LDS_V_BASE + buf_id * LDS_V_TILE_SIZE)

        def coop_load_k_global(tile_start):
            """Issue this thread's K global loads; results stay in registers."""
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                b64, o32 = kv_addr(
                    tile_start, load_row_in_batch + row_offset, load_col_base
                )
                vecs.append(load_global_f16xN(k_ptr, b64, o32))
            return vecs

        def coop_store_k_lds(vecs, buf_id=0):
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = (
                        load_row_in_batch + fx.Index(row_offset) < fx.Index(BLOCK_N)
                    )
                    if row_valid:
                        lds_row = load_row_in_batch + row_offset
                        lds_idx = k_base + lds_row * K_STRIDE + load_col_base
                        _lds_store_vx(vecs[batch], lds_idx)
                else:
                    lds_row = load_row_in_batch + row_offset
                    lds_idx = k_base + lds_row * K_STRIDE + load_col_base
                    _lds_store_vx(vecs[batch], lds_idx)

        def _v_store_transposed(v_base, l, vec):
            """Lane holds V[kv0+klane*8 .. +7][d]; contiguous in V^T[d][kv]."""
            tile = wave_id + fx.Index(l * NUM_WAVES)
            d = (tile % V_TR_D_BLOCKS) * WMMA_N + lane16
            kv = (tile // V_TR_D_BLOCKS) * WMMA_K + klane * WMMA_LANE_K
            lds_idx = v_base + d * VT_STRIDE + kv
            v = Vec(vec)
            for h in range_constexpr(2):  # 2x v4f16: honest 8-byte alignment
                part = v.shuffle(v, [h * 4, h * 4 + 1, h * 4 + 2, h * 4 + 3])
                fx.ptr_store(part, lds_kv + fx.Int32(lds_idx + h * 4))

        # Address each lane must supply so the hardware transpose lands the
        # right 16(d) x 16(kv) block in WMMA-operand order (see the derivation
        # in _global_load_tr_v8): within a group of 8 lanes the lane index picks
        # the kv row, and the group index picks the 8-wide d half.
        _tr_kv_off = (lane // 16) * WMMA_LANE_K + (lane % 8)
        _tr_d_off = ((lane // 8) % 2) * WMMA_LANE_K

        def coop_load_v_global(tile_start, chunk=0):
            """V columns [chunk*VO_CHUNK_COLS, +VO_CHUNK_COLS) for this KV tile."""
            vecs = []
            for l in range_constexpr(V_TR_LOADS):
                tile = wave_id + fx.Index(l * NUM_WAVES)
                d_base = (tile % V_TR_D_BLOCKS) * WMMA_N
                kv_base = (tile // V_TR_D_BLOCKS) * WMMA_K
                col = d_base + _tr_d_off
                if const_expr(chunk):
                    col = fx.Index(chunk * VO_CHUNK_COLS) + col
                b64, o32 = kv_addr(tile_start, kv_base + _tr_kv_off, col)
                vecs.append(_global_load_tr_v8(v_ptr_i64, b64, o32))
            return vecs

        def coop_store_v_lds(vecs, buf_id=0):
            # When V_TR_LOADS * NUM_WAVES overshoots the tile count the last
            # step has tail waves with no tile. Their global load still ran --
            # kv_global_idx clamped the row, so it read in bounds -- and is
            # simply not published here.
            v_base = v_buf_base(buf_id)
            for l in range_constexpr(V_TR_LOADS):
                if const_expr(V_TR_NEEDS_GUARD and (l + 1) * NUM_WAVES > _V_TR_TILES):
                    tile_ok = wave_id + fx.Index(l * NUM_WAVES) < fx.Index(_V_TR_TILES)
                    if tile_ok:
                        _v_store_transposed(v_base, l, vecs[l])
                else:
                    _v_store_transposed(v_base, l, vecs[l])

        # ---- Q preload ----
        q_row = q_start + wave_q_offset + lane16
        q_row_i32 = fx.Int32(q_row)
        # Use explicit signed-less-than predicate to match baseline ISA
        # (`v_cmp_gt_i64_e64`). fx.Index defaults to unsigned which would lower
        # to `v_cmp_gt_u64_e64` and cause an ISA hash drift even though both
        # variants are semantically equivalent for non-negative offsets.
        q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, _raw(q_row), _raw(seq_len_v))
        # Intra-tile Q row, clamped to 0 when the padded tile runs past the
        # sequence. Bounded by BLOCK_M, so the 32-bit offset stays small.
        q_row_in_tile = wave_q_offset + lane16
        q_row_in_tile_safe = fx.Index(
            ArithValue(q_in_bounds).select(q_row_in_tile, fx.Index(0))
        )
        q_tile_base = tile_base(q_start)
        c_zero_v8f16 = Vec.filled(8, 0.0, elem_dtype).ir_value()
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = shard_qk_off + fx.Index(ks * K_STEP_QK) + klane * WMMA_LANE_K
            raw = load_global_v8f16(
                q_ptr, q_tile_base, tile_off(q_row_in_tile_safe, q_col)
            )
            q_b_packs.append(ArithValue(q_in_bounds).select(raw, c_zero_v8f16))

        # ---- Constants ----
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_sm_scale_log2e = fx.Float32(sm_scale * _LOG2E)
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)
        width_i32 = fx.Int32(WARP_SIZE)
        shuf_16_i32 = fx.Int32(16)

        def reduction_peer(v_f32):
            return fx.Float32(v_f32).shuffle_xor(shuf_16_i32, width_i32)

        _q_end = q_start + BLOCK_M
        if const_expr(CAUSAL):
            kv_upper = fx.Index(
                ArithValue(_q_end < seq_len_v).select(_q_end, seq_len_v)
            )
        else:
            kv_upper = seq_len_v

        # ---- Binding prefetch prologue: tile 0's K and V into registers ----
        _k_vecs_init = coop_load_k_global(fx.Index(0))
        _v_vecs_init = coop_load_v_global(fx.Index(0))

        init_args = [_raw(c_neg_inf), _raw(c_zero_f)]
        for _ in range_constexpr(O_ACCS):
            init_args.append(_raw(c_zero_v8f32))
        # Both K and V tiles ride the loop in registers (prefetch distance 1).
        for batch in range_constexpr(NUM_BATCHES_KV):
            init_args.append(_k_vecs_init[batch])
        for batch in range_constexpr(V_TR_LOADS):
            init_args.append(_v_vecs_init[batch])

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(
            0, kv_upper, BLOCK_N_OUT, init=init_args
        ):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(O_ACCS)]
            _k_vecs_cur = [
                inner_iter_args[2 + O_ACCS + b]
                for b in range_constexpr(NUM_BATCHES_KV)
            ]
            _v_vecs_cur = [
                inner_iter_args[2 + O_ACCS + NUM_BATCHES_KV + b]
                for b in range_constexpr(V_TR_LOADS)
            ]

            next_kv_start = kv_block_start + fx.Index(BLOCK_N_OUT)

            # This tile's K is already in registers (loaded last iteration):
            # publish it, then immediately issue the *next* tile's K load so it
            # is in flight across GEMM1 + softmax rather than being waited on
            # right here. This is the whole point of the variant.
            coop_store_k_lds(_k_vecs_cur, 0)
            gpu.barrier()
            _k_vecs_next = coop_load_k_global(next_kv_start)
            k_base = k_buf_base(0)

            # ==== GEMM1: S = K @ Q^T (no swizzle, padding-based) ====
            s_accs = [_raw(c_zero_v8f32) for _ in range(NUM_S_ACCS)]

            for ks in range_constexpr(K_STEPS_QK):
                k_col = shard_qk_off + fx.Index(ks * K_STEP_QK) + klane * WMMA_LANE_K

                for st_idx in range_constexpr(N_SUB_TILES):
                    st_base_row = st_idx * K_SUB_N

                    k_row_a = lane16 + fx.Index(st_base_row)
                    k_lds_a = k_base + k_row_a * K_STRIDE + k_col
                    k_pack_a = _lds_load_v8(k_lds_a)

                    k_row_b = lane16 + fx.Index(st_base_row + 16)
                    k_lds_b = k_base + k_row_b * K_STRIDE + k_col
                    k_pack_b = _lds_load_v8(k_lds_b)

                    acc_idx_a = st_idx * 2
                    acc_idx_b = st_idx * 2 + 1
                    s_accs[acc_idx_a] = wmma_acc(
                        k_pack_a, q_b_packs[ks], s_accs[acc_idx_a]
                    )
                    s_accs[acc_idx_b] = wmma_acc(
                        k_pack_b, q_b_packs[ks], s_accs[acc_idx_b]
                    )

            # ==== Cross-shard S reduction ====
            # Each shard-wave holds a partial sum over its own head_dim slice;
            # the full S is their sum. Explicit partials, not ds_add_f32:
            # measured 54 vs 1055 WMMA-equivalents, see
            # kernels/microbench/lds_reduce.py.
            if const_expr(QK_SHARDS > 1):
                s_flat = []
                for st in range_constexpr(NUM_S_ACCS):
                    for r in range_constexpr(8):
                        s_flat.append(_raw(Vec(s_accs[st])[r]))

                own = wave_id * fx.Index(RED_F32_PER_WAVE)
                for e in range_constexpr(NUM_S_VALS):
                    _red_store(own + fx.Index(e * WARP_SIZE) + lane, s_flat[e])
                gpu.barrier()

                base_group = q_tile_in_block * fx.Index(QK_SHARDS * RED_F32_PER_WAVE)
                for e in range_constexpr(NUM_S_VALS):
                    acc = s_flat[e]
                    for k in range_constexpr(QK_SHARDS - 1):
                        peer = base_group + (
                            (shard_id + fx.Index(k + 1)) % fx.Index(QK_SHARDS)
                        ) * fx.Index(RED_F32_PER_WAVE)
                        acc = _fadd(
                            acc, _red_load(peer + fx.Index(e * WARP_SIZE) + lane)
                        )
                    s_flat[e] = acc
                gpu.barrier()

                s_accs = [
                    Vec.from_elements(
                        [fx.Float32(s_flat[st * 8 + r]) for r in range_constexpr(8)],
                        fx.Float32,
                    ).ir_value()
                    for st in range_constexpr(NUM_S_ACCS)
                ]

            # ==== Online softmax ====
            s_raw = []
            for st in range_constexpr(NUM_S_ACCS):
                for r in range_constexpr(8):
                    s_raw.append(Vec(s_accs[st])[r])

            if const_expr(CAUSAL):
                kv_start_i32 = fx.Int32(kv_block_start)
                klane_i32 = fx.Int32(klane)
                q_start_i32 = fx.Int32(q_start)
                max_kv_col_i32 = kv_start_i32 + fx.Int32(BLOCK_N - 1)
                tile_needs_mask = max_kv_col_i32 > q_start_i32

                # SSA-style restructure (PR #462 pattern, lines 700-870):
                # FlyDSL's `if` rewriter requires each loop-carried/conditional
                # state variable to be a single MLIR Value, not a list. Unfold
                # `s_raw[0..NUM_S_VALS-1]` into NUM_S_VALS named scalars, then
                # reassign each one inside the `if tile_needs_mask:` branch.
                # NUM_S_VALS == NUM_S_ACCS * 8 == 16 for BLOCK_N=32.
                s_v0 = s_raw[0]
                s_v1 = s_raw[1]
                s_v2 = s_raw[2]
                s_v3 = s_raw[3]
                s_v4 = s_raw[4]
                s_v5 = s_raw[5]
                s_v6 = s_raw[6]
                s_v7 = s_raw[7]
                s_v8 = s_raw[8]
                s_v9 = s_raw[9]
                s_v10 = s_raw[10]
                s_v11 = s_raw[11]
                s_v12 = s_raw[12]
                s_v13 = s_raw[13]
                s_v14 = s_raw[14]
                s_v15 = s_raw[15]
                if tile_needs_mask:
                    klane_off_i32 = klane_i32 * fx.Int32(8)
                    # st=0
                    _b0 = kv_start_i32 + fx.Int32(0) + klane_off_i32
                    s_v0 = ArithValue(_b0 > q_row_i32).select(c_neg_inf, s_v0)
                    _b1 = kv_start_i32 + fx.Int32(1) + klane_off_i32
                    s_v1 = ArithValue(_b1 > q_row_i32).select(c_neg_inf, s_v1)
                    _b2 = kv_start_i32 + fx.Int32(2) + klane_off_i32
                    s_v2 = ArithValue(_b2 > q_row_i32).select(c_neg_inf, s_v2)
                    _b3 = kv_start_i32 + fx.Int32(3) + klane_off_i32
                    s_v3 = ArithValue(_b3 > q_row_i32).select(c_neg_inf, s_v3)
                    _b4 = kv_start_i32 + fx.Int32(4) + klane_off_i32
                    s_v4 = ArithValue(_b4 > q_row_i32).select(c_neg_inf, s_v4)
                    _b5 = kv_start_i32 + fx.Int32(5) + klane_off_i32
                    s_v5 = ArithValue(_b5 > q_row_i32).select(c_neg_inf, s_v5)
                    _b6 = kv_start_i32 + fx.Int32(6) + klane_off_i32
                    s_v6 = ArithValue(_b6 > q_row_i32).select(c_neg_inf, s_v6)
                    _b7 = kv_start_i32 + fx.Int32(7) + klane_off_i32
                    s_v7 = ArithValue(_b7 > q_row_i32).select(c_neg_inf, s_v7)
                    # st=1 (st_base=16)
                    _b8 = kv_start_i32 + fx.Int32(16) + klane_off_i32
                    s_v8 = ArithValue(_b8 > q_row_i32).select(c_neg_inf, s_v8)
                    _b9 = kv_start_i32 + fx.Int32(17) + klane_off_i32
                    s_v9 = ArithValue(_b9 > q_row_i32).select(c_neg_inf, s_v9)
                    _b10 = kv_start_i32 + fx.Int32(18) + klane_off_i32
                    s_v10 = ArithValue(_b10 > q_row_i32).select(c_neg_inf, s_v10)
                    _b11 = kv_start_i32 + fx.Int32(19) + klane_off_i32
                    s_v11 = ArithValue(_b11 > q_row_i32).select(c_neg_inf, s_v11)
                    _b12 = kv_start_i32 + fx.Int32(20) + klane_off_i32
                    s_v12 = ArithValue(_b12 > q_row_i32).select(c_neg_inf, s_v12)
                    _b13 = kv_start_i32 + fx.Int32(21) + klane_off_i32
                    s_v13 = ArithValue(_b13 > q_row_i32).select(c_neg_inf, s_v13)
                    _b14 = kv_start_i32 + fx.Int32(22) + klane_off_i32
                    s_v14 = ArithValue(_b14 > q_row_i32).select(c_neg_inf, s_v14)
                    _b15 = kv_start_i32 + fx.Int32(23) + klane_off_i32
                    s_v15 = ArithValue(_b15 > q_row_i32).select(c_neg_inf, s_v15)
                s_raw = [
                    s_v0,
                    s_v1,
                    s_v2,
                    s_v3,
                    s_v4,
                    s_v5,
                    s_v6,
                    s_v7,
                    s_v8,
                    s_v9,
                    s_v10,
                    s_v11,
                    s_v12,
                    s_v13,
                    s_v14,
                    s_v15,
                ]

            local_max = s_raw[0]
            for r in range_constexpr(NUM_S_VALS - 1):
                local_max = _fmax(local_max, s_raw[r + 1])
            peer_max = reduction_peer(local_max)
            row_max = _fmax(local_max, peer_max)
            m_new_raw = _fmax(m_running, row_max)

            # ---- Opt2: rocdl.exp2 ----
            diff_m_raw = _fsub(m_running, m_new_raw)
            diff_m_scaled = _fmul(diff_m_raw, c_sm_scale_log2e)
            corr = rocdl.exp2(ir.F32Type.get(), _raw(diff_m_scaled))

            scaled_max = _fmul(c_sm_scale_log2e, m_new_raw)
            neg_scaled_max = _fsub(c_zero_f, scaled_max)

            p_vals = []
            local_sum = _raw(c_zero_f)
            for r in range_constexpr(NUM_S_VALS):
                diff = fmath.fma(s_raw[r], _raw(c_sm_scale_log2e), neg_scaled_max)
                p = rocdl.exp2(ir.F32Type.get(), _raw(diff))
                p_vals.append(p)
                local_sum = _fadd(local_sum, p)

            peer_sum = reduction_peer(local_sum)
            tile_sum = _fadd(local_sum, peer_sum)
            l_corr = _fmul(corr, l_running)
            l_new = _fadd(l_corr, tile_sum)

            corr_vec = Vec.from_elements([corr], fx.Float32).broadcast_to(8).ir_value()
            for dc in range_constexpr(O_ACCS):
                o_accs[dc] = _fmul(o_accs[dc], corr_vec)

            # V staging is chunked when VO_CHUNKS > 1: only VO_CHUNK_COLS
            # columns are resident at a time, which is what lets the LDS tile
            # keep its padding. The prefetch runs one step ahead of the
            # flattened (iteration, chunk) sequence, so exactly one chunk of V
            # rides the loop in registers.

            # ==== Build P packs ====
            p_packs_all = []
            for st_idx in range_constexpr(N_SUB_TILES):
                p_packs_st = []
                for pks in range_constexpr(PV_K_STEPS):
                    acc_idx = st_idx * 2 + pks
                    p_base = acc_idx * 8
                    p_slice = [p_vals[p_base + j] for j in range(8)]

                    if const_expr(dtype_str == "bf16"):
                        p_packs_st.append(bf16_trunc_pack_v8(p_slice))
                    else:
                        elem_list = []
                        for j in range_constexpr(8):
                            elem_list.append(fx.Float32(p_slice[j]).to(elem_dtype))
                        p_packs_st.append(
                            Vec.from_elements(elem_list, elem_dtype).ir_value()
                        )
                p_packs_all.append(p_packs_st)

            def _gemm2_chunk(_vc):
                """GEMM2 over the V window currently resident in LDS."""
                # ==== GEMM2: O += V^T @ P (software pipelined, row-major V) ====
                # Opt3: Prefetch next V pack while current WMMA executes
                v_base = v_buf_base(0)

                def _load_v_rowmajor(st_kv_base_val, pks_val, dc_val):
                    # V^T[d][kv]: the 8 kv values this lane needs are contiguous, so
                    # this is one vector read instead of 8 strided scalar loads.
                    d_pos = shard_vo_off + fx.Index(dc_val * D_CHUNK) + lane16
                    kv0 = fx.Index(st_kv_base_val + pks_val * PV_K_STEP) + klane * WMMA_LANE_K
                    return _lds_load_v8(v_base + d_pos * VT_STRIDE + kv0)

                # Software pipeline: preload first V pack
                cur_v_packs = []
                for st_idx in range_constexpr(N_SUB_TILES):
                    cur_v_packs.append(_load_v_rowmajor(st_idx * K_SUB_N, 0, 0))

                for pks in range_constexpr(PV_K_STEPS):
                    for dc in range_constexpr(D_CHUNKS):
                        next_dc = dc + 1
                        next_pks = pks
                        if const_expr(next_dc >= D_CHUNKS):
                            next_dc = 0
                            next_pks = pks + 1
                        has_next = const_expr(next_pks < PV_K_STEPS)

                        # Prefetch next V while current WMMA runs
                        next_v_packs = []
                        if const_expr(has_next):
                            for st_idx in range_constexpr(N_SUB_TILES):
                                next_v_packs.append(
                                    _load_v_rowmajor(st_idx * K_SUB_N, next_pks, next_dc)
                                )

                        for st_idx in range_constexpr(N_SUB_TILES):
                            o_accs[_vc * D_CHUNKS + dc] = wmma_acc(
                                cur_v_packs[st_idx],
                                p_packs_all[st_idx][pks],
                                o_accs[_vc * D_CHUNKS + dc],
                            )

                        if const_expr(has_next):
                            cur_v_packs = next_v_packs



            if const_expr(VO_CHUNKS == 1):
                # Unchunked: keep the original shape exactly. Wrapping this in a
                # 1-trip chunk loop cost 4 VGPRs (190 -> 194), which crosses an
                # allocation-granularity boundary and drops occupancy 8 -> 7
                # waves/SIMD, measured at -4% on head_dim 128.
                coop_store_v_lds(_v_vecs_cur, 0)
                gpu.barrier()
                _v_vecs_next = coop_load_v_global(next_kv_start, 0)
                _gemm2_chunk(0)
            else:
                _v_hold = _v_vecs_cur
                for vchunk in range_constexpr(VO_CHUNKS):
                    coop_store_v_lds(_v_hold, 0)
                    gpu.barrier()
                    if const_expr(vchunk + 1 < VO_CHUNKS):
                        _v_hold = coop_load_v_global(kv_block_start, vchunk + 1)
                    else:
                        _v_vecs_next = coop_load_v_global(next_kv_start, 0)
                    _gemm2_chunk(vchunk)
                    if const_expr(vchunk + 1 < VO_CHUNKS):
                        # all waves must finish reading this window before the
                        # next chunk overwrites it
                        gpu.barrier()

            m_running = m_new_raw
            l_running = l_new

            _yield_args = [m_running, l_running] + o_accs
            for batch in range_constexpr(NUM_BATCHES_KV):
                _yield_args.append(_k_vecs_next[batch])
            for batch in range_constexpr(V_TR_LOADS):
                _yield_args.append(_v_vecs_next[batch])
            loop_results = yield _yield_args

        # ---- Normalize and store O ----
        l_final = loop_results[1]
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(O_ACCS)]

        inv_l = arith.divf(_raw(c_one_f), _raw(l_final), fastmath=fm_fast)
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(8).ir_value()

        if q_in_bounds:
            for _oi in range_constexpr(O_ACCS):
                vc, dc = _oi // D_CHUNKS, _oi % D_CHUNKS
                o_norm_vec = _fmul(o_finals[_oi], inv_l_vec)
                o_trunc = Vec(o_norm_vec).to(elem_dtype).ir_value()
                d_col = shard_vo_off + fx.Index(dc * D_CHUNK) + klane * 8
                if const_expr(vc):
                    d_col = fx.Index(vc * VO_CHUNK_COLS) + d_col
                _store_global_half(
                    o_ptr, q_tile_base, tile_off(q_row_in_tile, d_col), o_trunc
                )

    @flyc.jit
    def launch_flash_attn_bp(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()

        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M
        grid_x = bs_idx * num_q_tiles * NUM_HEADS

        launcher = flash_attn_func_bp_kernel(Q, K, V, O, seq_len)

        if const_expr(waves_per_eu is not None):
            _wpe = int(waves_per_eu)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32, _wpe
                        )
        if const_expr(flat_work_group_size is not None):
            _fwgs = int(flat_work_group_size)
            if const_expr(_fwgs >= 1):
                flat_wg_attr = ir.StringAttr.get(f"{_fwgs},{_fwgs}")
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.flat_work_group_size"] = flat_wg_attr

        passthrough_entries = []
        # The default GCN scheduler sinks every LDS load next to its consuming
        # WMMA and funnels them all through one VGPR quad, so SIInsertWaitcnts
        # emits `s_wait_dscnt 0x0` between each load and use and the GEMMs run
        # with no LDS latency hiding. max-ilp / max-memory-clause trade VGPRs
        # for keeping several loads in flight.
        if const_expr(SCHED_STRATEGY):
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("amdgpu-sched-strategy"),
                        ir.StringAttr.get(SCHED_STRATEGY),
                    ]
                )
            )
        if const_expr(daz):
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("denormal-fp-math-f32"),
                        ir.StringAttr.get("preserve-sign,preserve-sign"),
                    ]
                )
            )
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("no-nans-fp-math"),
                        ir.StringAttr.get("true"),
                    ]
                )
            )
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("unsafe-fp-math"),
                        ir.StringAttr.get("true"),
                    ]
                )
            )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

        launcher.launch(grid=(grid_x, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

    _fmha_compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    def _ptr_arg(t):
        if hasattr(t, "data_ptr"):
            type_name = type(t).__name__
            module_name = type(t).__module__
            ptr = (
                0
                if type_name == "FakeTensor" or "fake_tensor" in module_name
                else t.data_ptr()
            )
            return flyc.from_c_void_p(fx.Uint8, ptr)
        return t

    def _wrap_qkvo(args, kwargs):
        args = list(args)
        for idx in range(min(4, len(args))):
            args[idx] = _ptr_arg(args[idx])
        for name in ("Q", "K", "V", "O"):
            if name in kwargs:
                kwargs[name] = _ptr_arg(kwargs[name])
        return tuple(args), kwargs

    launch_flash_attn_bp.compile_hints = dict(_fmha_compile_hints)

    def _launch(*args, **kwargs):
        args, kwargs = _wrap_qkvo(args, kwargs)
        stream = kwargs.pop("stream", fx.Stream(None))
        _run_compiled(launch_flash_attn_bp, *args, stream)

    def _compile(Q, K, V, O, batch_size, seq_len, stream=None):  # noqa: E741
        return flyc.compile(
            launch_flash_attn_bp,
            _ptr_arg(Q),
            _ptr_arg(K),
            _ptr_arg(V),
            _ptr_arg(O),
            batch_size,
            seq_len,
            fx.Stream(stream),
        )

    _launch.compile = _compile
    return _launch


build_flash_attn_func_bp_module = build_flash_attn_func_bp_module_primary
