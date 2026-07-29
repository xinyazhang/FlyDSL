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
    if head_dim not in (64, 128):
        raise ValueError(f"binding-prefetch variant supports head_dim 64 or 128, got {head_dim}")
    if (BLOCK_M, BLOCK_N) != (128, 32):
        raise ValueError(
            f"binding-prefetch variant supports BLOCK_M=128, BLOCK_N=32 only, got {(BLOCK_M, BLOCK_N)}"
        )
    if causal and NUM_S_VALS != 16:
        # The causal mask below is unrolled into 16 explicitly named scalars.
        raise ValueError(f"causal masking assumes NUM_S_VALS == 16, got {NUM_S_VALS}")

    # Head-dimension sharding. QK_SHARDS waves cooperate on one Q row-tile,
    # each reducing over its own head_dim slice in GEMM1 and owning the matching
    # V/O column slice in GEMM2. QK_SHARDS == 1 is the unsharded kernel: every
    # sharded construct below is behind `const_expr(QK_SHARDS > 1)`, so this
    # path must trace to identical IR. See plan_gfx1201_large_hdim.md.
    QK_SHARDS = qk_shards if qk_shards is not None else 1
    QK_SLICE = head_dim // QK_SHARDS          # head-dim columns per wave in GEMM1
    VO_SLICE = head_dim // QK_SHARDS          # V/O columns owned per wave in GEMM2
    assert head_dim % QK_SHARDS == 0

    # Keep the workgroup at TARGET_WAVES by trading Q row-tiles for shards, so
    # BLOCK_M shrinks as QK_SHARDS grows. QK_SHARDS=1 gives BLOCK_M=128 as before.
    TARGET_WAVES = 8
    Q_TILES_PER_BLOCK = max(1, TARGET_WAVES // QK_SHARDS)
    BLOCK_M = ROWS_PER_WAVE * Q_TILES_PER_BLOCK
    NUM_WAVES = Q_TILES_PER_BLOCK * QK_SHARDS
    _V_TR_TILES = (head_dim // WMMA_N) * (BLOCK_N // WMMA_K)
    if _V_TR_TILES % NUM_WAVES:
        raise ValueError(f"V transpose tiles ({_V_TR_TILES}) must divide across {NUM_WAVES} waves")
    V_TR_LOADS = _V_TR_TILES // NUM_WAVES
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
    V_TR_D_BLOCKS = head_dim // WMMA_N

    K_STEP_QK = WMMA_K
    K_STEPS_QK = QK_SLICE // K_STEP_QK        # GEMM1 K-steps for this wave's slice
    WMMA_LANE_K = 8

    D_CHUNK = WMMA_N
    D_CHUNKS = VO_SLICE // D_CHUNK            # o_accs count for this wave's slice

    PV_K_STEP = WMMA_K
    PV_K_STEPS = K_SUB_N // PV_K_STEP

    assert BLOCK_M % NUM_WAVES == 0
    assert head_dim % 32 == 0
    assert head_dim >= 64
    assert dtype_str in ("f16", "bf16")

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim
    CAUSAL = causal
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    # LDS layout -- K uses padding instead of XOR swizzle; V row-major with padding
    K_STRIDE = HEAD_DIM + 4  # padding to reduce bank conflicts (no swizzle)
    # V is staged TRANSPOSED: V^T[d][kv], so GEMM2 reads 8 consecutive kv for a
    # fixed d as one contiguous vector instead of 8 strided scalar loads.
    # +4 makes the row stride 36 elems = 72 B: 18 dwords, so lane16*18 mod 32
    # hits 16 distinct banks (conflict-free) while staying 8-byte aligned.
    VT_STRIDE = BLOCK_N + 4

    ENABLE_LDS_VEC16 = os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1") == "1"
    VEC_WIDTH = 16 if ENABLE_LDS_VEC16 else 8
    THREADS_PER_ROW_LOAD = HEAD_DIM // VEC_WIDTH
    ROWS_PER_BATCH_LOAD = BLOCK_SIZE // THREADS_PER_ROW_LOAD

    if ROWS_PER_BATCH_LOAD >= BLOCK_N:
        NUM_BATCHES_KV = 1
        KV_NEEDS_GUARD = ROWS_PER_BATCH_LOAD > BLOCK_N
    else:
        NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_LOAD
        KV_NEEDS_GUARD = False

    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_V_TILE_SIZE = HEAD_DIM * VT_STRIDE
    LDS_K_TOTAL_SIZE = NUM_PREFETCH_K * LDS_K_TILE_SIZE
    LDS_V_BASE = LDS_K_TOTAL_SIZE
    LDS_V_TOTAL_SIZE = NUM_PREFETCH_V * LDS_V_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_SIZE

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

        def _global_load_tr_v8(base_i64, elem_idx):
            """One global_load_tr_b128: an 8x8 16-bit transpose per lane-group.

            Lane g_i supplies an address; the 8 contiguous elements there become
            column i of the group's output, so lane g_j receives
            [M_0[j] .. M_7[j]]. Verified empirically on gfx1201.
            """
            byte_off = arith.index_cast(T.i64, _raw(fx.Index(elem_idx) * 2))
            addr = arith.addi(base_i64, byte_off)
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

        def kv_global_idx(token_idx, col):
            row = fx.Index(ArithValue(token_idx < seq_len_v).select(token_idx, seq_last))
            return global_idx(row, col)

        def _load_global_half_vec(ptr, base_idx, vec_type):
            gep = buffer_ops.get_element_ptr(
                ptr, fx.Int64(base_idx), elem_type=elem_type
            )
            return _pointer_load(vec_type, gep)

        def _store_global_half(ptr, base_idx, val):
            gep = buffer_ops.get_element_ptr(
                ptr, fx.Int64(base_idx), elem_type=elem_type
            )
            _pointer_store(val, gep)

        def load_global_f16xN(base_ptr, base_idx):
            return _load_global_half_vec(base_ptr, base_idx, vxf16_type)

        def load_global_v8f16(base_ptr, base_idx):
            return _load_global_half_vec(base_ptr, base_idx, v8f16_type)

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
                row_idx = tile_start + load_row_in_batch + row_offset
                g_idx = kv_global_idx(row_idx, load_col_base)
                vecs.append(load_global_f16xN(k_ptr, g_idx))
            return vecs

        def coop_store_k_lds(vecs, buf_id=0):
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = load_row_in_batch < fx.Index(BLOCK_N)
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

        def coop_load_v_global(tile_start):
            vecs = []
            for l in range_constexpr(V_TR_LOADS):
                tile = wave_id + fx.Index(l * NUM_WAVES)
                d_base = (tile % V_TR_D_BLOCKS) * WMMA_N
                kv_base = (tile // V_TR_D_BLOCKS) * WMMA_K
                row = tile_start + kv_base + _tr_kv_off
                col = d_base + _tr_d_off
                vecs.append(_global_load_tr_v8(v_ptr_i64, kv_global_idx(row, col)))
            return vecs

        def coop_store_v_lds(vecs, buf_id=0):
            # The TR tiling covers the V tile exactly (V_TR_LOADS blocks per wave over
            # NUM_WAVES waves), so unlike the K path there is no partial row to
            # guard against.
            v_base = v_buf_base(buf_id)
            for l in range_constexpr(V_TR_LOADS):
                _v_store_transposed(v_base, l, vecs[l])

        # ---- Q preload ----
        q_row = q_start + wave_q_offset + lane16
        q_row_i32 = fx.Int32(q_row)
        # Use explicit signed-less-than predicate to match baseline ISA
        # (`v_cmp_gt_i64_e64`). fx.Index defaults to unsigned which would lower
        # to `v_cmp_gt_u64_e64` and cause an ISA hash drift even though both
        # variants are semantically equivalent for non-negative offsets.
        q_in_bounds = arith.cmpi(arith.CmpIPredicate.slt, _raw(q_row), _raw(seq_len_v))
        q_row_safe = fx.Index(ArithValue(q_in_bounds).select(q_row, fx.Index(0)))
        c_zero_v8f16 = Vec.filled(8, 0.0, elem_dtype).ir_value()
        q_b_packs = []
        for ks in range_constexpr(K_STEPS_QK):
            q_col = shard_qk_off + fx.Index(ks * K_STEP_QK) + klane * WMMA_LANE_K
            g_idx = global_idx(q_row_safe, q_col)
            raw = load_global_v8f16(q_ptr, g_idx)
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
        for _ in range_constexpr(D_CHUNKS):
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
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(D_CHUNKS)]
            _k_vecs_cur = [
                inner_iter_args[2 + D_CHUNKS + b]
                for b in range_constexpr(NUM_BATCHES_KV)
            ]
            _v_vecs_cur = [
                inner_iter_args[2 + D_CHUNKS + NUM_BATCHES_KV + b]
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
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = _fmul(o_accs[dc], corr_vec)

            # Same for V: publish the registers held since last iteration,
            # then issue the next tile's V load to fly across GEMM2.
            coop_store_v_lds(_v_vecs_cur, 0)
            gpu.barrier()
            _v_vecs_next = coop_load_v_global(next_kv_start)

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
                        o_accs[dc] = wmma_acc(
                            cur_v_packs[st_idx], p_packs_all[st_idx][pks], o_accs[dc]
                        )

                    if const_expr(has_next):
                        cur_v_packs = next_v_packs

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
        o_finals = [loop_results[2 + dc] for dc in range_constexpr(D_CHUNKS)]

        inv_l = arith.divf(_raw(c_one_f), _raw(l_final), fastmath=fm_fast)
        inv_l_vec = Vec.from_elements([inv_l], fx.Float32).broadcast_to(8).ir_value()

        if q_in_bounds:
            for dc in range_constexpr(D_CHUNKS):
                o_norm_vec = _fmul(o_finals[dc], inv_l_vec)
                o_trunc = Vec(o_norm_vec).to(elem_dtype).ir_value()
                d_col = shard_vo_off + fx.Index(dc * D_CHUNK) + klane * 8
                o_global = global_idx(q_row, d_col)
                _store_global_half(o_ptr, o_global, o_trunc)

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
