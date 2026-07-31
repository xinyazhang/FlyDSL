# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Flash Attention for gfx1201 -- **aiw**, the all-in-one kernel.

``aiw`` = "all-in-one". The ``w`` stands in for the ``o`` of "one" on purpose:
``aio`` reads as "async IO" and ``ai1`` reads as "Artificial Intelligence One",
so neither was usable. Please do not "fix" the spelling.

This module unifies three previously separate gfx1201 kernels --
``flash_attn_func_gfx1201.py`` (baseline), ``..._bp.py`` (binding prefetch) and
``..._m32.py`` (two Q row-tiles per wave) -- which were never three designs.
They were one design at three points in a knob space, plus drift. Each knob
below is a ``const_expr`` switch resolved at trace time, so a given build emits
exactly one variant's code with no runtime branching.

    knob              baseline      bp             m32
    ---------------------------------------------------------------
    K_PREFETCH_DIST   0             1              1
    V_LDS_LAYOUT      "row"         "transposed"   "transposed"
    Q_ROW_TILES       1             1              2
    QK_SHARDS         1             bp_qk_shards() 1
    VO_CHUNKS         1             vo_chunks()    1
    VO_WIDTH          slice of D    head_dim       head_dim

The three originals are kept on disk unchanged and serve as correctness
oracles: for every knob setting that reproduces one of them, aiw must match it
bitwise wherever the floating-point reduction order is unchanged (see
``test_flash_attn_func_gfx1201_aiw.py``). ISA-level divergence from the
originals is expected and accepted -- aiw uses the 64-bit-base + 32-bit-offset
addressing scheme everywhere, which only ``bp`` had.

--- The knobs -------------------------------------------------------------

``K_PREFETCH_DIST`` -- 0 or 1. gfx1201 has no direct global->LDS copy, so a KV
tile must transit VGPRs and the only latency hiding available is a *binding*
prefetch: issue the global load early, hold it in registers, consume it later.
At distance 0 the load and the LDS store are adjacent, so the latency is fully
exposed (``global_load; s_wait_loadcnt 0x0; ds_store``). At distance 1 both K
and V ride the loop in registers:

    prologue: load K[0], V[0] -> registers
    iteration i (carrying K[i], V[i]):
        store K[i] -> LDS ; barrier ; issue load K[i+1]   <- flies over GEMM1
        GEMM1 (S = K @ Q^T) ; online softmax
        store V[i] -> LDS ; barrier ; issue load V[i+1]   <- flies over GEMM2
        GEMM2 (O += V^T @ P)

Barrier count is unchanged (2/iteration); the cost is one more live register
set. Distance 1 wins from head_dim 48 up; below that the tiles are small enough
that the extra pressure buys nothing.

``V_LDS_LAYOUT`` -- ``"row"`` stages V as V[kv][d]; GEMM2 then needs 8 strided
scalar LDS reads per operand. ``"transposed"`` stages V^T[d][kv] filled with
``global_load_tr_b128``, whose hardware 16x16 transpose delivers each lane its
8 kv-elements contiguously, so GEMM2 reads one vector per operand and the LDS
store stays contiguous instead of becoming a 16-way scatter. Worth +2.7% at
N >= 4096.

``Q_ROW_TILES`` -- Q row-tiles owned by each wave. At 2, one K or V operand
feeds two WMMAs instead of one (halving LDS reads per FLOP) and BLOCK_M
doubles, halving the grid and with it K/V global traffic. Costs
``o_accs + q_b_packs + s_accs`` VGPRs, which scales with head_dim: +64 at
head_dim 64 (fine), +112 at 128 (hits the 256-VGPR cap and spills, -27%).

``QK_SHARDS`` -- waves cooperating on one Q row-tile, each reducing over its
own head_dim slice in GEMM1 and owning the matching V/O column slice in GEMM2.
Their partial S values are summed through LDS. Lets large head_dim spread its
register cost across waves.

``VO_CHUNKS`` -- V staging passes, so only ``head_dim/VO_CHUNKS`` columns are
LDS-resident at a time. Keeps the padded K+V tile inside the 64 KiB workgroup
cap at large head_dim, at one extra barrier pair per extra pass.

``VO_WIDTH`` / ``D_OFFSET`` -- V/O column *window*, distinct from ``VO_SLICE``
(which is the per-wave share of a window). Attention is column-separable in V:
``O[:, s] = P @ V[:, s]`` and P does not depend on V, so the V/O width can be a
slice of the QK width. This is what keeps ``o_accs`` (VO_WIDTH/2 VGPRs) and the
V LDS tile in budget above head_dim 256, at the cost of repeating GEMM1 and the
K traffic per window.

--- Register layout -------------------------------------------------------

WMMA 16x16x16, wave32:
  - A/B operand: v8f16 per lane (lane16 = row/col, klane*8 = K-offset)
  - C/D result:  v8f32 per lane, element si = C[klane*8+si][lane16]

Layout: Q/K/V/O are 1D flattened from BSHD (batch, seq_len, num_heads, head_dim).
Grid:   (batch * num_q_tiles * num_heads,)
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

KERNEL_NAME = "flash_attn_func_gfx1201_aiw_kernel"
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
_SHARDS_BY_HEAD_DIM = {224: 2}
_Q_TILES_BY_HEAD_DIM = {48: 8, 64: 16, 80: 16, 96: 16, 128: 16, 160: 16,
                        192: 8, 224: 8, 256: 4, 384: 4, 512: 2}

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


def build_flash_attn_func_aiw_module_primary(
    num_heads,
    head_dim,
    head_dim_v=None,
    d_offset=0,
    causal=True,
    dtype_str="bf16",
    sm_scale=None,
    waves_per_eu=2,
    flat_work_group_size=None,
    block_m=None,
    block_n=None,
    # ---- knobs; None means "use the tuning policy above" ----
    k_prefetch_dist=None,
    v_prefetch_dist=1,
    v_lds_layout=None,
    strides_constexpr=False,
    safe_softmax=True,
    padded_head=False,
    q_row_tiles=1,
    shards=None,
    unsafe_fp_math=True,
    fast_fp_math=True,
    daz=True,
    path_tag="auto",
):
    """Build the unified gfx1201 flash-attention kernel.

    See the module docstring for what each knob selects and which of the three
    original kernels a given setting reproduces.
    """

    # ---- WMMA / wave32 constants ----
    WARP_SIZE = 32
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16
    K_SUB_N = 32
    WMMA_LANE_K = 8

    if head_dim % 16 or not (16 <= head_dim <= 512):
        raise ValueError(f"aiw needs 16 <= head_dim <= 512 and head_dim % 16 == 0, got {head_dim}")
    if dtype_str not in ("f16", "bf16"):
        raise ValueError(f"aiw supports f16/bf16, got {dtype_str!r}")

    # ---- Knob resolution ----
    K_PREFETCH_DIST = (
        default_prefetch_dist(head_dim) if k_prefetch_dist is None else k_prefetch_dist
    )
    if K_PREFETCH_DIST not in (0, 1):
        raise ValueError(f"k_prefetch_dist must be 0 or 1, got {K_PREFETCH_DIST}")
    # K and V prefetch distances are INDEPENDENT. The baseline kernel is
    # (K=0, V=1) -- its "pre-issue first V global load before loop" carries V in
    # registers exactly as bp does, and only K is staged at distance 0. Folding
    # the two into one knob produces a (K=0, V=0) schedule that exists in none
    # of the originals and costs 9.6% at head_dim 32 non-causal.
    V_PREFETCH_DIST = v_prefetch_dist
    if V_PREFETCH_DIST not in (0, 1):
        raise ValueError(f"v_prefetch_dist must be 0 or 1, got {V_PREFETCH_DIST}")
    V_LDS_LAYOUT = (
        ("transposed" if K_PREFETCH_DIST else "row")
        if v_lds_layout is None
        else v_lds_layout
    )
    if V_LDS_LAYOUT not in ("row", "transposed"):
        raise ValueError(f"v_lds_layout must be 'row' or 'transposed', got {V_LDS_LAYOUT!r}")
    V_TRANSPOSED = V_LDS_LAYOUT == "transposed"
    Q_ROW_TILES = q_row_tiles

    ROWS_PER_WAVE = WMMA_M * Q_ROW_TILES

    BLOCK_N = block_n if block_n is not None else default_block_n(
        head_dim, causal, K_PREFETCH_DIST
    )
    if BLOCK_N % K_SUB_N:
        raise ValueError(f"BLOCK_N ({BLOCK_N}) must be a multiple of K_SUB_N ({K_SUB_N})")

    N_SUB_TILES = BLOCK_N // K_SUB_N
    NUM_S_ACCS = N_SUB_TILES * 2
    NUM_S_VALS = NUM_S_ACCS * 8

    if causal and NUM_S_VALS != 16:
        # The causal mask below is unrolled into 16 explicitly named scalars and
        # rebinds s_raw to a 16-element list; a wider BLOCK_N would overrun it
        # with an IndexError at trace time. Fail clearly instead. (Dies with the
        # interval-decomposition work.)
        raise ValueError(
            f"causal masking requires BLOCK_N == {K_SUB_N} (NUM_S_VALS == 16), "
            f"got BLOCK_N={BLOCK_N} (NUM_S_VALS={NUM_S_VALS})"
        )

    # V/O column *window*: the slice of the output width this build computes.
    # Distinct from VO_SLICE (a wave's share of a window) and from VO_CHUNK_COLS
    # (a staging pass's share of a window).
    VO_WIDTH = head_dim if head_dim_v is None else head_dim_v
    D_OFFSET = d_offset
    if VO_WIDTH % 16 or not (0 < VO_WIDTH <= head_dim):
        raise ValueError(f"head_dim_v must be a positive multiple of 16 and <= head_dim, got {VO_WIDTH}")
    if D_OFFSET % 16 or D_OFFSET + VO_WIDTH > head_dim:
        raise ValueError(f"d_offset {D_OFFSET} + head_dim_v {VO_WIDTH} must fit in head_dim {head_dim}")

    # Head-dimension sharding. QK_SHARDS waves cooperate on one Q row-tile,
    # each reducing over its own head_dim slice in GEMM1 and owning the matching
    # V/O column slice in GEMM2. QK_SHARDS == 1 is the unsharded kernel: every
    # sharded construct below is behind `const_expr(QK_SHARDS > 1)`.
    # Resolved against the V/O *window*, not head_dim, so a narrow window does
    # not inherit a shard count it cannot divide.
    if shards is not None:
        QK_SHARDS = shards
    elif K_PREFETCH_DIST == 0 or Q_ROW_TILES > 1:
        QK_SHARDS = 1
    else:
        QK_SHARDS = resolve_shards(head_dim, VO_WIDTH, BLOCK_N)
    QK_SLICE = head_dim // QK_SHARDS      # head-dim columns per wave in GEMM1

    VO_CHUNKS = vo_chunks(VO_WIDTH, BLOCK_N, QK_SHARDS) if V_TRANSPOSED else 1
    VO_CHUNK_COLS = VO_WIDTH // VO_CHUNKS   # V columns resident per pass
    VO_SLICE = VO_CHUNK_COLS // QK_SHARDS   # V/O columns per wave per pass

    if head_dim % QK_SHARDS or QK_SLICE % WMMA_K:
        # K_STEPS_QK = QK_SLICE // WMMA_K truncates otherwise, silently dropping
        # part of the reduction: head_dim 224 with 4 shards gives a 56-wide
        # slice, of which only 48 would be reduced (measured rel err 0.97).
        raise ValueError(
            f"head_dim {head_dim} with {QK_SHARDS} shards gives a {QK_SLICE}-wide "
            f"slice, which must be a multiple of WMMA_K={WMMA_K}"
        )
    if VO_SLICE % WMMA_N:
        raise ValueError(f"V/O slice {VO_SLICE} must be a multiple of WMMA_N={WMMA_N}")

    # ---- Validity predicate over the knob space ----
    # These combinations are not implemented rather than not expressible. Fail
    # at build time; do not emit a kernel that silently computes the wrong
    # thing.
    if not V_TRANSPOSED and QK_SHARDS > 1:
        raise ValueError("v_lds_layout='row' does not implement cross-shard reduction; use 'transposed'")
    if not V_TRANSPOSED and VO_CHUNKS > 1:
        raise ValueError("v_lds_layout='row' does not implement chunked V staging; use 'transposed'")
    if VO_CHUNKS > 1 and not V_PREFETCH_DIST:
        raise ValueError("chunked V staging requires v_prefetch_dist=1")
    if Q_ROW_TILES > 1 and QK_SHARDS > 1:
        raise ValueError("q_row_tiles > 1 with qk_shards > 1 is untested; pick one")
    if Q_ROW_TILES not in (1, 2):
        raise ValueError(f"q_row_tiles must be 1 or 2, got {Q_ROW_TILES}")

    # ---- Workgroup geometry ----
    if K_PREFETCH_DIST == 0:
        BLOCK_M = block_m if block_m is not None else default_block_m(head_dim, 0)
        if BLOCK_M % ROWS_PER_WAVE:
            raise ValueError(f"BLOCK_M ({BLOCK_M}) must be a multiple of {ROWS_PER_WAVE}")
        Q_TILES_PER_BLOCK = BLOCK_M // ROWS_PER_WAVE
        NUM_WAVES = Q_TILES_PER_BLOCK
    else:
        # Keep the workgroup at TARGET_WAVES by trading Q row-tiles for shards,
        # so BLOCK_M shrinks as QK_SHARDS grows.
        #
        # Divide by Q_ROW_TILES so BLOCK_M is *invariant* to that knob and only
        # the wave count changes: at Q_ROW_TILES=2 the same rows are covered by
        # half as many waves, each doing twice the work, which is the whole
        # point (one K/V operand feeds two WMMAs). Without the division a
        # Q_ROW_TILES=2 build would silently double BLOCK_M as well, doubling
        # per-wave register pressure on top of the knob's own cost.
        #
        # Note this is invisible to a bitwise output comparison -- each Q row's
        # arithmetic is identical however rows are grouped into blocks -- so
        # only the benchmark catches it.
        Q_TILES_PER_BLOCK = max(
            1, q_tiles_per_block(head_dim, QK_SHARDS) // Q_ROW_TILES
        )
        BLOCK_M = ROWS_PER_WAVE * Q_TILES_PER_BLOCK
        NUM_WAVES = Q_TILES_PER_BLOCK * QK_SHARDS

    if flat_work_group_size is None:
        flat_work_group_size = NUM_WAVES * WARP_SIZE
    BLOCK_SIZE = flat_work_group_size

    BLOCK_N_OUT = BLOCK_N

    # LLVM's amdgpu-sched-strategy function attribute; "" leaves the default
    # GCN scheduler in place. See the passthrough block in the launch wrapper.
    # Measured at BATCH=2 H=12 N=4096 d=128 f16 -- distance 1: causal
    # 85.6 -> 88.5 TFLOPS, non-causal 91.4 -> 91.9. Distance 0: causal
    # 69.8 -> 79.2, but non-causal 89.4 -> 88.6, so only causal wants it there.
    _DEFAULT_SCHED = "max-memory-clause" if (K_PREFETCH_DIST or causal) else ""
    SCHED_STRATEGY = os.environ.get("FMHA_SCHED_STRATEGY", _DEFAULT_SCHED)

    K_STEP_QK = WMMA_K
    K_STEPS_QK = QK_SLICE // K_STEP_QK      # GEMM1 K-steps for this wave's slice

    D_CHUNK = WMMA_N
    D_CHUNKS = VO_SLICE // D_CHUNK          # accs per wave per chunk
    O_ACCS = VO_CHUNKS * D_CHUNKS           # accs live across the KV loop, per Q tile

    PV_K_STEP = WMMA_K
    PV_K_STEPS = K_SUB_N // PV_K_STEP

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(head_dim)

    NUM_HEADS = num_heads
    HEAD_DIM = head_dim
    CAUSAL = causal
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    # Strides and sm_scale are runtime arguments, not folded constants: an AOT
    # kernel cannot bake them in, since a fixed set of binaries has to cover
    # every shape.
    #
    # Measured price (B=1 H=8 N=4096 f16, interleaved 3-rep A/B over the full
    # head_dim ladder x causal): **median ratio 0.996**, worst 0.967 (head_dim
    # 16 causal), best 1.041 (head_dim 192 causal). Several configs come out
    # *faster* and the spread is symmetric about 1.0, so this is the board's
    # noise floor rather than a measurable cost. Registers: +0 to +4 VGPRs,
    # +22 SGPRs, no new spills at any head_dim. Output is bitwise identical to
    # the folded form, sm_scale included.
    #
    # Each tensor carries its own triple, and they are not interchangeable: K
    # and V reach the kernel exactly as the caller allocated them (`mha_fwd_aot`
    # passes them through untouched), and under MQA/GQA they carry Num_head_k
    # rather than Num_head_q. Going from one shared triple to four cost +18
    # SGPRs and zero VGPRs -- strides are uniform scalars, so they only change
    # which value an address multiplies.
    #
    # `strides_constexpr=True` keeps them folded. It is retained only as an A/B
    # arm for future phases -- if addressing ever becomes expensive we want to
    # be able to measure against the folded form -- and is not a shipping
    # configuration.
    #
    # Naming is numeric (`stride_q0/q1/q2`), not by axis letter. The suffixed forms
    # inherited from the maths (`stride_qz/qh/qm`) read badly and have caused
    # real mix-ups during AOTriton's kernel development. Axis 3 is `D`, which is
    # contiguous by contract, so it is never passed.
    #
    # Longest-processing-time-first dispatch for causal. Under causal masking
    # a workgroup's cost grows with its q_tile (tile 0 walks one KV block, tile
    # N-1 walks N), and grid.y is dispatched in increasing order -- so the
    # cheapest blocks go first and the most expensive land in the tail, which
    # is the worst possible order. Reversing the index puts the expensive
    # blocks first and leaves only cheap ones to fill the tail.
    #
    # Measured B=1 H=8 N=4096 f16 causal, TFLOPS forward -> reversed:
    #   head_dim  16   31.8 -> 35.7  (+12%)
    #             32   51.2 -> 59.5  (+16%)
    #             64   67.1 -> 77.8  (+16%)
    #            128   74.8 -> 87.0  (+16%)
    #            256   68.8 -> 72.8  (+6%)
    #            512   43.9 -> 46.6  (+6%)
    #
    # Non-causal is untouched: every tile costs the same there, so the reversal
    # would be pure arithmetic for no gain and is not emitted.
    #
    # This is orthogonal to the *axis* order (see the grid comment below) --
    # that one decides whether a scheduling group has uniform duration, this
    # one decides the order the groups are issued in. Both matter, and neither
    # of the pre-unification kernels had either.
    _REVERSE_Q_TILES = os.environ.get("FMHA_REVERSE_Q_TILES", "1") == "1"
    STRIDES_CONSTEXPR = strides_constexpr

    # Two softmax corrections, both from AOTriton's hard-won list. Kept behind
    # one knob so the pre-unification kernels remain usable as bitwise oracles
    # (they have neither fix), and so the cost can be measured.
    #
    # (a) `m_i` initialises to -3.40282e+38, not -inf. If a tile is entirely
    #     masked its row max is -inf, and with an -inf init the rescale becomes
    #     exp2(-inf - -inf) = exp2(NaN) = NaN. A finite floor makes it
    #     exp2(0) = 1 and the masked probabilities exp2(-inf - m) = 0, which is
    #     the right answer. The *mask* fill stays -inf; only the init changes.
    #
    #     Unreachable as the kernel stands -- the first KV tile always contains
    #     kv = 0 <= q_row, so it is never fully masked. It becomes reachable
    #     with **bias**, which may hold any non-NaN value including -inf and is
    #     routinely used as an attention mask: a bias covering the whole first
    #     tile drives its row max to -inf. The regression test therefore belongs
    #     with the bias feature, not here.
    #
    # (b) The QK scale is applied to the scores **before** the row max, and
    #     `m_i` is therefore kept in the scaled domain. The kernel previously
    #     kept `m_i` unscaled and folded the scale into the exponent as
    #     `exp2(fma(s, qk_scale, -qk_scale * m))`, which is exactly the FMA
    #     pattern AOTriton flags in ROCm/aotriton#54 as producing numerical
    #     errors on large inputs. Costs NUM_S_VALS multiplies per tile and
    #     saves the FMA; on a loop this LDS-latency-bound that is close to free.
    #     It is also the convention LSE needs, since logsumexp is
    #     m + log2(l) in the scaled domain.
    SAFE_SOFTMAX = safe_softmax

    # `head_dim` is BLOCK_DMODEL: the compile-time tile width, drawn from the
    # ladder. The *real* extents are the runtime `hdim_qk` / `hdim_vo`
    # arguments, and PADDED_HEAD says whether they differ from the tile.
    # AOTriton derives exactly this (`attn_fwd.cc`):
    #     hdim_rounded = round_value(max(hdim_qk, hdim_vo), ladder)
    #     PADDED_HEAD  = (hdim_rounded != hdim_qk || hdim_rounded != hdim_vo)
    # With PADDED_HEAD false the two are equal to the tile and no masking is
    # emitted at all -- that is the common case and the one the ladder measures.
    PADDED_HEAD = padded_head
    BLOCK_DMODEL = head_dim

    # ---- LDS layout ----
    # K is padded rather than XOR-swizzled (a swizzle was implemented and
    # measured a net loss; see sdpa_lore_gfx1201.md). Chunking bounds the V
    # window, so the padding always fits.
    _LDS_PAD = 4
    K_STRIDE = HEAD_DIM + _LDS_PAD
    # Transposed V: V^T[d][kv]. +4 makes the row stride 36 elems = 72 B, i.e.
    # 18 dwords, so lane16*18 mod 32 hits 16 distinct banks (conflict-free)
    # while staying 8-byte aligned.
    VT_STRIDE = BLOCK_N + _LDS_PAD
    V_STRIDE = VO_WIDTH + _LDS_PAD          # row-major V

    # 32-byte (16-element) loads need the D-axis pitch to be a multiple of 16
    # elements. The contract only guarantees 8 (16 bytes), and it is only
    # BLOCK_DMODEL -- itself always a multiple of 16 -- that makes the wider
    # load safe when hdim == the tile. Under PADDED_HEAD the real extent can
    # end on any 8-boundary, so a 16-element load can run past the allocation
    # at the tensor tail. Drop to 8 there.
    ENABLE_LDS_VEC16 = (
        os.getenv("FLYDSL_FLASH_ATTN_FUNC_ENABLE_LDS_VEC16", "1") == "1"
        and not PADDED_HEAD
    )
    VEC_WIDTH = 16 if ENABLE_LDS_VEC16 else 8

    def _load_geom(width):
        """Cooperative-load geometry for a row of `width` elements."""
        tpr = width // VEC_WIDTH
        rpb = BLOCK_SIZE // tpr
        nb = (BLOCK_N + rpb - 1) // rpb
        return tpr, rpb, nb, nb * rpb != BLOCK_N

    # Cover BLOCK_N rows with ceil() batches, not floor(). Flooring silently
    # dropped rows whenever ROWS_PER_BATCH_LOAD neither reached BLOCK_N nor
    # divided it: head_dim 160/192/224 give 25/21/18, so BLOCK_N // that == 1
    # and only 25/21/18 of the 32 KV rows reached LDS. The rest was stale LDS,
    # which surfaced as NaN.
    THREADS_PER_ROW_LOAD, ROWS_PER_BATCH_LOAD, NUM_BATCHES_KV, KV_NEEDS_GUARD = _load_geom(HEAD_DIM)

    # global_load_tr_b128 transposes an 8x8 tile of 16-bit elements across each
    # group of 8 lanes, so one wave-wide TR load produces a 16(d) x 16(kv) block
    # already in WMMA-operand layout. Split those blocks over the waves.
    V_TR_D_BLOCKS = VO_CHUNK_COLS // WMMA_N
    _V_TR_TILES = V_TR_D_BLOCKS * (BLOCK_N // WMMA_K)
    # The V TR tiling need not divide evenly across the waves: tail tiles are
    # guarded at the LDS store. Requiring divisibility used to force head_dim
    # 160 down to 4 waves, which cost it 89.1 -> 70.0 TFLOPS.
    V_TR_LOADS = (_V_TR_TILES + NUM_WAVES - 1) // NUM_WAVES
    V_TR_NEEDS_GUARD = V_TR_LOADS * NUM_WAVES != _V_TR_TILES

    V_TPR_LOAD, V_ROWS_PER_BATCH, NUM_BATCHES_V, V_NEEDS_GUARD = _load_geom(VO_CHUNK_COLS)

    # How many register-resident V vectors ride the loop, under either layout.
    V_LOADS = V_TR_LOADS if V_TRANSPOSED else NUM_BATCHES_V

    LDS_K_TILE_SIZE = BLOCK_N * K_STRIDE
    LDS_V_TILE_SIZE = (
        VO_CHUNK_COLS * VT_STRIDE if V_TRANSPOSED else BLOCK_N * V_STRIDE
    )
    LDS_K_TOTAL_SIZE = LDS_K_TILE_SIZE
    LDS_V_BASE = LDS_K_TOTAL_SIZE
    LDS_V_TOTAL_SIZE = LDS_V_TILE_SIZE
    LDS_KV_TOTAL_SIZE = LDS_K_TOTAL_SIZE + LDS_V_TOTAL_SIZE

    # The cross-shard S reduction aliases the V region rather than allocating
    # its own: V is written to LDS only *after* softmax, so between the
    # post-K-store barrier and that write the V tile holds the previous
    # iteration's data, already consumed by the previous GEMM2.
    RED_F32_PER_WAVE = NUM_S_VALS * WARP_SIZE
    RED_F32_TOTAL = NUM_WAVES * RED_F32_PER_WAVE
    RED_ALIASES_V = QK_SHARDS == 1 or RED_F32_TOTAL * 4 <= LDS_V_TOTAL_SIZE * 2
    if not RED_ALIASES_V:
        LDS_KV_TOTAL_SIZE += (RED_F32_TOTAL * 4 + 1) // 2  # in elem_dtype units

    # FlyDSL's `dtype_to_elem_type` returns a Numeric class, which is what the
    # Vector API (`Vec.make_type`, `.to(...)`) and `fx.Array` require.
    elem_numeric_cls = dtype_to_elem_type(dtype_str)

    @fx.struct
    class SharedStorage:
        kv: fx.Array[elem_numeric_cls, LDS_KV_TOTAL_SIZE, 16]

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def flash_attn_func_aiw_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        seq_len: fx.Int32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        stride_q0: fx.Int64,
        stride_q1: fx.Int64,
        stride_q2: fx.Int64,
        stride_k0: fx.Int64,
        stride_k1: fx.Int64,
        stride_k2: fx.Int64,
        stride_v0: fx.Int64,
        stride_v1: fx.Int64,
        stride_v2: fx.Int64,
        stride_o0: fx.Int64,
        stride_o1: fx.Int64,
        stride_o2: fx.Int64,
        sm_scale_arg: fx.Float32,
    ):
        elem_type = elem_numeric_cls.ir_type
        elem_dtype = elem_numeric_cls

        def _to_global_ptr_i64(ptr):
            return arith.index_cast(T.i64, fx.ptrtoint(ptr))

        q_ptr = _pointer_to_llvm_ptr(Q)
        k_ptr = _pointer_to_llvm_ptr(K)
        v_ptr = _pointer_to_llvm_ptr(V)
        v_ptr_i64 = _to_global_ptr_i64(V)
        o_ptr = _pointer_to_llvm_ptr(O)
        fm_fast = arith.FastMathFlags.fast

        # Local fast-math arithmetic helpers -- preserve the fastmath flag while
        # using the lowercase op names that accept _raw() unwrapping.
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

        def _global_load_tr_v8(base_i64, base64, off32):
            """One global_load_tr_b128: an 8x8 16-bit transpose per lane-group.

            Lane g_i supplies an address; the 8 contiguous elements there become
            column i of the group's output, so lane g_j receives
            [M_0[j] .. M_7[j]]. Verified empirically on gfx1201.
            """
            # Split as elsewhere: the (batch, head, tile) origin in 64 bits, the
            # intra-tile part in 32. Feeding LLVM `uniform_i64 + zext(i32
            # divergent)` is what lets SelectGlobalSAddr keep the base in SGPRs
            # instead of forcing a 64-bit VGPR address pair.
            base_bytes = arith.index_cast(T.i64, _raw(fx.Index(base64) * 2))
            off_bytes = arith.extui(
                T.i64, arith.index_cast(T.i32, _raw(fx.Index(off32) * 2))
            )
            addr = arith.addi(arith.addi(base_i64, base_bytes), off_bytes)
            p = _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), addr).result
            return rocdl.global_load_tr_b128(v8f16_type, p)

        # ---- Honest-alignment LDS access ----
        # K/V rows are K_STRIDE * 2 bytes apart, so LDS addresses here are only
        # guaranteed 8-byte aligned. `fly.ptr_load` / `fly.ptr_store` emit no
        # alignment attribute, so LLVM falls back to the vector type's ABI
        # alignment -- 16 B for v8f16, 32 B for v16f16. That over-promise is what
        # makes the backend select ds_load_b128 / ds_store_b128 on addresses
        # that are not actually 16-byte aligned: 2.2x slower here (39 -> 92
        # TFLOPS) and undefined behaviour besides. Splitting into 8-byte (v4f16)
        # accesses carries a truthful `align 8` and folds back into
        # ds_load2_b64 / ds_store2_b64.
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

        # f32 view of the V LDS region, for the cross-shard S reduction. The kv
        # array is elem_dtype (16-bit), so go through an addrspace(3) LLVM
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

        tid = fx.Index(gpu.thread_idx.x)

        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane16 = lane % 16
        klane = lane // 16

        # (q_tile, shard) decomposition of the wave index. At QK_SHARDS == 1
        # this is q_tile == wave_id and shard == 0, i.e. the unsharded mapping.
        q_tile_in_block = wave_id // QK_SHARDS
        shard_id = wave_id % QK_SHARDS
        wave_q_offset = q_tile_in_block * ROWS_PER_WAVE

        # Column origins of this wave's slices. Both are 0 at QK_SHARDS == 1.
        shard_qk_off = shard_id * fx.Index(QK_SLICE)   # into Q/K head_dim
        shard_vo_off = shard_id * fx.Index(VO_SLICE)   # into the V/O window

        # 3D grid: (head_q, q_tile, batch). A flat grid would need two integer
        # divisions here to recover these, and with num_heads runtime neither
        # would fold away.
        #
        # **The axis order is load-bearing for causal.** Under causal masking a
        # workgroup's cost grows with its q_tile -- tile 0 walks one KV block,
        # tile N-1 walks N. The x axis dispatches fastest, so putting q_tile
        # there spreads durations 1..N across every scheduling group, while
        # putting head there gives each group a uniform duration. Measured at
        # B=1 H=8 N=4096 f16 causal, q_tile-fastest against head-fastest:
        # head_dim 16 0.587, 32 0.612, 64 0.715, 128 0.769. Non-causal is
        # indifferent (all within 1%), which is what identifies the cause.
        #
        # Note AOTriton uses dim3{S,H,B} -- q_tile fastest -- for NUM_XCDS == 1.
        # That is not a contradiction: it also forces PERSISTENT_TYPE = 2 for
        # every causal functional, which replaces the grid with a work-stealing
        # loop and makes the axis order irrelevant. Porting its grid_calculator
        # verbatim without persistent-dynamic would reintroduce the regression
        # above. Revisit this ordering when persistent-dynamic lands.
        head_q = fx.Index(gpu.block_idx.x)
        if const_expr(CAUSAL and _REVERSE_Q_TILES):
            # Longest-processing-time-first: under causal, cost grows with
            # q_tile, so dispatching the expensive tiles first leaves only
            # cheap ones to fill the tail.
            _ntiles = (seq_len_v + (BLOCK_M - 1)) // BLOCK_M
            q_tile_idx = _ntiles - fx.Index(1) - fx.Index(gpu.block_idx.y)
        else:
            q_tile_idx = fx.Index(gpu.block_idx.y)
        batch_idx = fx.Index(gpu.block_idx.z)
        q_start = q_tile_idx * BLOCK_M

        # MQA/GQA: Num_head_q / Num_head_k query heads share each KV head.
        # The ratio is uniform and computed once, so the scalar divide is
        # immaterial; the per-head division below is by that ratio.
        head_k = head_q // (fx.Index(num_head_q) // fx.Index(num_head_k))

        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        v_row_in_batch = tid // V_TPR_LOAD
        v_col_base = (tid % V_TPR_LOAD) * VEC_WIDTH

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
        # Strides, either folded or taken from arguments. The *body* below is
        # identical either way -- FlyDSL's arithmetic accepts a Python int and
        # an fx value interchangeably -- so only this binding differs. That is
        # the whole reason a single kernel source can serve both the JIT and AOT
        # paths (see D2 in sdpa-close-gap-plan1.md).
        if const_expr(STRIDES_CONSTEXPR):
            # Axis 0 (batch) still depends on the runtime seq_len, so it is
            # never a compile-time constant even here; only 1 and 2 fold.
            _st = (seq_len_v * STRIDE_TOKEN, STRIDE_TOKEN, HEAD_DIM)
            q_st = k_st = v_st = o_st = _st
            sm_log2e = fx.Float32(sm_scale * _LOG2E)
        else:
            q_st = (fx.Index(stride_q0), fx.Index(stride_q1), fx.Index(stride_q2))
            k_st = (fx.Index(stride_k0), fx.Index(stride_k1), fx.Index(stride_k2))
            v_st = (fx.Index(stride_v0), fx.Index(stride_v1), fx.Index(stride_v2))
            o_st = (fx.Index(stride_o0), fx.Index(stride_o1), fx.Index(stride_o2))
            sm_log2e = _fmul(sm_scale_arg, fx.Float32(_LOG2E))

        # Q/K/V/O each get their own address pair. They genuinely differ: K and
        # V are whatever the caller allocated (`mha_fwd_aot` passes them through
        # untouched), and under MQA/GQA they carry Num_head_k rather than
        # Num_head_q, so their head stride differs from Q's by construction.
        # Assuming one shared layout is not a simplification, it is wrong.
        def _addr_pair(st, head):
            s_batch, s_seq, s_head = st
            bh = batch_idx * s_batch + head * s_head

            def tbase(tile_start):
                """Uniform 64-bit element base for (batch, head, tile_start)."""
                return bh + tile_start * s_seq

            def toff(row_in_tile, col):
                """Divergent 32-bit element offset inside the tile."""
                return row_in_tile * s_seq + col

            return tbase, toff

        # Q and O are indexed by the query head; K and V by the KV head they
        # share. At num_head_q == num_head_k these coincide.
        q_tbase, q_toff = _addr_pair(q_st, head_q)
        k_tbase, k_toff = _addr_pair(k_st, head_k)
        v_tbase, v_toff = _addr_pair(v_st, head_k)
        o_tbase, o_toff = _addr_pair(o_st, head_q)

        def _kv_addr(tbase, toff, tile_start, row_in_tile, col):
            """(uniform base, divergent offset) for a KV row, clamped in bounds.

            At K_PREFETCH_DIST == 1 the loop runs one tile ahead, so the final
            iteration addresses a tile past the end of the sequence; the
            unguarded cooperative load also addresses rows past BLOCK_N. Clamp
            tile_start first, then the in-tile row -- since the row clamp can
            only fire when tile_start is within BLOCK_N of the end, the clamped
            row stays below BLOCK_N. The values are never consumed.

            With both distances 0 there is no over-read: the interface pads
            seq_len to a multiple of BLOCK_M and BLOCK_N divides BLOCK_M, so
            tile_start + BLOCK_N <= seq_len always. Skip the clamp there; it is
            pure VALU.
            """
            if const_expr(
                K_PREFETCH_DIST == 0
                and V_PREFETCH_DIST == 0
                and not KV_NEEDS_GUARD
            ):
                return tbase(tile_start), toff(row_in_tile, col)
            ts = fx.Index(
                ArithValue(tile_start < seq_len_v).select(tile_start, seq_last)
            )
            in_range = (ts + row_in_tile) < seq_len_v
            row = fx.Index(ArithValue(in_range).select(row_in_tile, seq_last - ts))
            return tbase(ts), toff(row, col)

        def k_addr(tile_start, row_in_tile, col):
            return _kv_addr(k_tbase, k_toff, tile_start, row_in_tile, col)

        def v_addr(tile_start, row_in_tile, col):
            return _kv_addr(v_tbase, v_toff, tile_start, row_in_tile, col)

        # ---- PADDED_HEAD column handling ----
        #
        # Exactly one rule: an element is valid iff its column < hdim. It covers
        # both invalid regions without the kernel ever knowing the pitch:
        #
        #   [hdim, ceil8(hdim))   pad inside the allocation. Safe to load (the
        #                         chunk containing hdim ends at ceil8(hdim),
        #                         which is <= pitch by the contract), but the
        #                         contents are not guaranteed zero.
        #   [ceil8(hdim), tile)   past the row entirely. In BSHD these bytes
        #                         belong to head h+1, and at the last head of
        #                         the last token they are past the allocation.
        #                         Must not be addressed.
        #
        # So: a chunk whose *start* is >= hdim is redirected to column 0 (always
        # valid) and then masked away wholesale; a chunk that straddles is
        # loaded as-is and masked per element. Both fall out of the same two
        # operations, which is why there is no case analysis below.
        _hdim_qk_i = fx.Index(hdim_qk)
        _hdim_vo_i = fx.Index(hdim_vo)

        def _col_safe(col, hdim_i):
            """Redirect a wholly-invalid chunk to column 0 so the load is safe."""
            if const_expr(not PADDED_HEAD):
                return col
            return fx.Index(ArithValue(col < hdim_i).select(col, fx.Index(0)))

        def _col_mask(col, hdim_i, width):
            """i1 vector, element j set iff column `col + j` holds real data.

            Built from a loop-invariant column, so for the cooperative loads
            this is hoisted out of the KV loop entirely and costs one vector
            select per load inside it.
            """
            return Vec.from_elements(
                [(col + fx.Index(j)) < hdim_i for j in range_constexpr(width)],
                fx.Boolean,
            )

        def _apply_col_mask(vec, col, hdim_i, width):
            if const_expr(not PADDED_HEAD):
                return vec
            zeros = Vec.filled(width, 0.0, elem_dtype)
            return _col_mask(col, hdim_i, width).select(Vec(vec), zeros).ir_value()

        def _split_ptr(ptr, base64, off32):
            """ptr + base64 (uniform, 64-bit) + off32 (divergent, 32-bit)."""
            p = buffer_ops.get_element_ptr(ptr, fx.Int64(base64), elem_type=elem_type)
            return buffer_ops.get_element_ptr(p, fx.Int32(off32), elem_type=elem_type)

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
            exactly (2.79e-3) but costs 2-3% at distance 1 and 2.7-5.4% at
            Q_ROW_TILES=2, so it is deliberately not done. Truncation by
            decision, not oversight.
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

        # ---- K staging ----

        def coop_load_k_global(tile_start):
            """Issue this thread's K global loads; results stay in registers."""
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                b64, o32 = k_addr(
                    tile_start, load_row_in_batch + row_offset,
                    _col_safe(load_col_base, _hdim_qk_i),
                )
                vecs.append(
                    _apply_col_mask(
                        load_global_f16xN(k_ptr, b64, o32),
                        load_col_base, _hdim_qk_i, VEC_WIDTH,
                    )
                )
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

        def coop_load_store_k(tile_start, buf_id=0):
            """Distance-0 K staging: load and store inside a single guard.

            The guard has to cover the *load*, not just the store. When
            ROWS_PER_BATCH_LOAD overshoots BLOCK_N some cooperative-load lanes
            have no row -- at head_dim 32 with BLOCK_N 64 that is exactly half
            of them -- and issuing their (clamped, redundant) global loads
            anyway measured -9.6% against the baseline kernel. Distance 1
            cannot do this: there the loaded value is loop-carried, so it has
            to exist unconditionally.
            """
            k_base = k_buf_base(buf_id)
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                lds_row = load_row_in_batch + row_offset
                if const_expr(KV_NEEDS_GUARD):
                    row_valid = lds_row < fx.Index(BLOCK_N)
                    if row_valid:
                        b64, o32 = k_addr(
                            tile_start, lds_row, _col_safe(load_col_base, _hdim_qk_i)
                        )
                        _lds_store_vx(
                            _apply_col_mask(
                                load_global_f16xN(k_ptr, b64, o32),
                                load_col_base, _hdim_qk_i, VEC_WIDTH,
                            ),
                            k_base + lds_row * K_STRIDE + load_col_base,
                        )
                else:
                    b64, o32 = k_addr(
                        tile_start, lds_row, _col_safe(load_col_base, _hdim_qk_i)
                    )
                    _lds_store_vx(
                        _apply_col_mask(
                            load_global_f16xN(k_ptr, b64, o32),
                            load_col_base, _hdim_qk_i, VEC_WIDTH,
                        ),
                        k_base + lds_row * K_STRIDE + load_col_base,
                    )

        # ---- V staging ----
        #
        # Two layouts. Transposed stages V^T[d][kv] via global_load_tr_b128 so
        # GEMM2 reads one contiguous vector per operand; row-major stages
        # V[kv][d] and GEMM2 gathers 8 strided scalars.

        # Address each lane must supply so the hardware transpose lands the
        # right 16(d) x 16(kv) block in WMMA-operand order (see the derivation
        # in _global_load_tr_v8): within a group of 8 lanes the lane index picks
        # the kv row, and the group index picks the 8-wide d half.
        _tr_kv_off = (lane // 16) * WMMA_LANE_K + (lane % 8)
        _tr_d_off = ((lane // 8) % 2) * WMMA_LANE_K

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

        def _v_store_row_major(v_base, lds_row, vec):
            _lds_store_vx(vec, v_base + lds_row * V_STRIDE + v_col_base)

        def coop_load_v_global(tile_start, chunk=0):
            """V columns [chunk*VO_CHUNK_COLS, +VO_CHUNK_COLS) of this KV tile.

            Columns are relative to the V/O window, so D_OFFSET is added here
            and only here -- LDS indices stay window-relative.
            """
            vecs = []
            if const_expr(V_TRANSPOSED):
                for l in range_constexpr(V_TR_LOADS):
                    tile = wave_id + fx.Index(l * NUM_WAVES)
                    d_base = (tile % V_TR_D_BLOCKS) * WMMA_N
                    kv_base = (tile // V_TR_D_BLOCKS) * WMMA_K
                    col = d_base + _tr_d_off
                    if const_expr(chunk):
                        col = fx.Index(chunk * VO_CHUNK_COLS) + col
                    if const_expr(D_OFFSET):
                        col = fx.Index(D_OFFSET) + col
                    b64, o32 = v_addr(
                        tile_start, kv_base + _tr_kv_off,
                        _col_safe(col, _hdim_vo_i),
                    )
                    vecs.append(_global_load_tr_v8(v_ptr_i64, b64, o32))
            else:
                for batch in range_constexpr(NUM_BATCHES_V):
                    row_offset = batch * V_ROWS_PER_BATCH
                    col = v_col_base
                    if const_expr(D_OFFSET):
                        col = fx.Index(D_OFFSET) + col
                    b64, o32 = v_addr(
                        tile_start, v_row_in_batch + row_offset,
                        _col_safe(col, _hdim_vo_i),
                    )
                    vecs.append(load_global_f16xN(v_ptr, b64, o32))
            return vecs

        def coop_store_v_lds(vecs, buf_id=0):
            v_base = v_buf_base(buf_id)
            if const_expr(V_TRANSPOSED):
                # When V_TR_LOADS * NUM_WAVES overshoots the tile count the last
                # step has tail waves with no tile. Their global load still ran
                # -- kv_addr clamped the row, so it read in bounds -- and is
                # simply not published here.
                for l in range_constexpr(V_TR_LOADS):
                    if const_expr(V_TR_NEEDS_GUARD and (l + 1) * NUM_WAVES > _V_TR_TILES):
                        tile_ok = wave_id + fx.Index(l * NUM_WAVES) < fx.Index(_V_TR_TILES)
                        if tile_ok:
                            _v_store_transposed(v_base, l, vecs[l])
                    else:
                        _v_store_transposed(v_base, l, vecs[l])
            else:
                for batch in range_constexpr(NUM_BATCHES_V):
                    row_offset = batch * V_ROWS_PER_BATCH
                    if const_expr(V_NEEDS_GUARD):
                        row_valid = (
                            v_row_in_batch + fx.Index(row_offset) < fx.Index(BLOCK_N)
                        )
                        if row_valid:
                            _v_store_row_major(
                                v_base, v_row_in_batch + row_offset, vecs[batch]
                            )
                    else:
                        _v_store_row_major(
                            v_base, v_row_in_batch + row_offset, vecs[batch]
                        )

        # ---- Q preload ----
        # One row-tile per Q_ROW_TILES; at 1 this is the single-tile mapping.
        q_rows = [
            q_start + wave_q_offset + fx.Index(qt * WMMA_M) + lane16
            for qt in range_constexpr(Q_ROW_TILES)
        ]
        q_row_i32s = [fx.Int32(r) for r in q_rows]
        # Intra-tile Q rows, bounded by BLOCK_M so the 32-bit offset stays small.
        q_rows_in_tile = [
            wave_q_offset + fx.Index(qt * WMMA_M) + lane16
            for qt in range_constexpr(Q_ROW_TILES)
        ]
        q_tile_base = q_tbase(q_start)
        c_zero_v8f16 = Vec.filled(8, 0.0, elem_dtype).ir_value()

        q_in_bounds_all = []
        q_b_packs_all = []
        for qt in range_constexpr(Q_ROW_TILES):
            # Explicit signed-less-than predicate: fx.Index defaults to unsigned,
            # which lowers to v_cmp_gt_u64_e64 instead of the signed form.
            _in = arith.cmpi(
                arith.CmpIPredicate.slt, _raw(q_rows[qt]), _raw(seq_len_v)
            )
            _safe = fx.Index(
                ArithValue(_in).select(q_rows_in_tile[qt], fx.Index(0))
            )
            _packs = []
            for ks in range_constexpr(K_STEPS_QK):
                q_col = shard_qk_off + fx.Index(ks * K_STEP_QK) + klane * WMMA_LANE_K
                raw = load_global_v8f16(
                    q_ptr, q_tile_base,
                    q_toff(_safe, _col_safe(q_col, _hdim_qk_i)),
                )
                raw = _apply_col_mask(raw, q_col, _hdim_qk_i, 8)
                _packs.append(ArithValue(_in).select(raw, c_zero_v8f16))
            q_in_bounds_all.append(_in)
            q_b_packs_all.append(_packs)

        # ---- Constants ----
        # Mask fill: genuinely -inf, so exp2(-inf - m) is exactly 0.
        c_neg_inf = fx.Float32(float("-inf"))
        # m_i floor: finite, so an all-masked tile cannot produce -inf - -inf.
        c_m_init = fx.Float32(-3.40282e38 if SAFE_SOFTMAX else float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_sm_scale_log2e = sm_log2e
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

        # ---- Prologue: at distance 1, tile 0's K / V go to registers ----
        if const_expr(K_PREFETCH_DIST):
            _k_vecs_init = coop_load_k_global(fx.Index(0))
        if const_expr(V_PREFETCH_DIST):
            _v_vecs_init = coop_load_v_global(fx.Index(0))

        # Loop-carried state layout:
        #   [0 .. 2*Q_ROW_TILES)             m/l per Q row-tile, interleaved
        #   [_ML .. _ML + Q_ROW_TILES*O_ACCS) O accumulators per Q row-tile
        #   [_OFF ..)                         K vectors (distance 1 only), then V
        _ML = 2 * Q_ROW_TILES
        _OFF = _ML + Q_ROW_TILES * O_ACCS
        _KOFF = _OFF
        _VOFF = _OFF + (NUM_BATCHES_KV if K_PREFETCH_DIST else 0)

        init_args = []
        for _ in range_constexpr(Q_ROW_TILES):
            init_args.append(_raw(c_m_init))
            init_args.append(_raw(c_zero_f))
        for _ in range_constexpr(Q_ROW_TILES * O_ACCS):
            init_args.append(_raw(c_zero_v8f32))
        if const_expr(K_PREFETCH_DIST):
            for batch in range_constexpr(NUM_BATCHES_KV):
                init_args.append(_k_vecs_init[batch])
        if const_expr(V_PREFETCH_DIST):
            for batch in range_constexpr(V_LOADS):
                init_args.append(_v_vecs_init[batch])

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(
            0, kv_upper, BLOCK_N_OUT, init=init_args
        ):
            m_run = [inner_iter_args[2 * qt] for qt in range_constexpr(Q_ROW_TILES)]
            l_run = [inner_iter_args[2 * qt + 1] for qt in range_constexpr(Q_ROW_TILES)]
            o_accs_all = [
                [
                    inner_iter_args[_ML + qt * O_ACCS + i]
                    for i in range_constexpr(O_ACCS)
                ]
                for qt in range_constexpr(Q_ROW_TILES)
            ]
            if const_expr(K_PREFETCH_DIST):
                _k_vecs_cur = [
                    inner_iter_args[_KOFF + b] for b in range_constexpr(NUM_BATCHES_KV)
                ]
            if const_expr(V_PREFETCH_DIST):
                _v_vecs_cur = [
                    inner_iter_args[_VOFF + b] for b in range_constexpr(V_LOADS)
                ]

            next_kv_start = kv_block_start + fx.Index(BLOCK_N_OUT)

            # At distance 1 this tile's K is already in registers: publish it,
            # then immediately issue the *next* tile's K load so it is in flight
            # across GEMM1 + softmax rather than being waited on right here.
            # At distance 0 the load and store are adjacent and the latency is
            # exposed -- that is the whole difference between the two schedules.
            if const_expr(K_PREFETCH_DIST):
                coop_store_k_lds(_k_vecs_cur, 0)
                gpu.barrier()
                _k_vecs_next = coop_load_k_global(next_kv_start)
            else:
                coop_load_store_k(kv_block_start, 0)
                gpu.barrier()
            k_base = k_buf_base(0)

            # ==== GEMM1: S = K @ Q^T ====
            # At Q_ROW_TILES > 1 each K pack feeds every row-tile's S
            # accumulators: one LDS read serves Q_ROW_TILES WMMAs. That reuse is
            # the point of the knob.
            s_accs_all = [
                [_raw(c_zero_v8f32) for _ in range(NUM_S_ACCS)]
                for _ in range_constexpr(Q_ROW_TILES)
            ]

            for ks in range_constexpr(K_STEPS_QK):
                k_col = shard_qk_off + fx.Index(ks * K_STEP_QK) + klane * WMMA_LANE_K

                for st_idx in range_constexpr(N_SUB_TILES):
                    st_base_row = st_idx * K_SUB_N

                    k_row_a = lane16 + fx.Index(st_base_row)
                    k_pack_a = _lds_load_v8(k_base + k_row_a * K_STRIDE + k_col)

                    k_row_b = lane16 + fx.Index(st_base_row + 16)
                    k_pack_b = _lds_load_v8(k_base + k_row_b * K_STRIDE + k_col)

                    acc_idx_a = st_idx * 2
                    acc_idx_b = st_idx * 2 + 1
                    for qt in range_constexpr(Q_ROW_TILES):
                        s_accs_all[qt][acc_idx_a] = wmma_acc(
                            k_pack_a, q_b_packs_all[qt][ks], s_accs_all[qt][acc_idx_a]
                        )
                        s_accs_all[qt][acc_idx_b] = wmma_acc(
                            k_pack_b, q_b_packs_all[qt][ks], s_accs_all[qt][acc_idx_b]
                        )

            # ==== Cross-shard S reduction ====
            # Each shard-wave holds a partial sum over its own head_dim slice;
            # the full S is their sum. Explicit partials, not ds_add_f32:
            # measured 54 vs 1055 WMMA-equivalents, see
            # kernels/microbench/lds_reduce.py.
            if const_expr(QK_SHARDS > 1):
                for qt in range_constexpr(Q_ROW_TILES):
                    s_flat = []
                    for st in range_constexpr(NUM_S_ACCS):
                        for r in range_constexpr(8):
                            s_flat.append(_raw(Vec(s_accs_all[qt][st])[r]))

                    own = wave_id * fx.Index(RED_F32_PER_WAVE)
                    for e in range_constexpr(NUM_S_VALS):
                        _red_store(own + fx.Index(e * WARP_SIZE) + lane, s_flat[e])
                    gpu.barrier()

                    base_group = q_tile_in_block * fx.Index(
                        QK_SHARDS * RED_F32_PER_WAVE
                    )
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

                    s_accs_all[qt] = [
                        Vec.from_elements(
                            [
                                fx.Float32(s_flat[st * 8 + r])
                                for r in range_constexpr(8)
                            ],
                            fx.Float32,
                        ).ir_value()
                        for st in range_constexpr(NUM_S_ACCS)
                    ]

            # ==== Online softmax, per Q row-tile ====
            # Each row-tile keeps its own running max/sum and its own O
            # accumulators.
            m_new_all, l_new_all, p_vals_all = [], [], []
            for qt in range_constexpr(Q_ROW_TILES):
                s_accs = s_accs_all[qt]
                q_row_i32 = q_row_i32s[qt]
                m_running = m_run[qt]
                l_running = l_run[qt]

                s_raw = []
                for st in range_constexpr(NUM_S_ACCS):
                    for r in range_constexpr(8):
                        s_raw.append(Vec(s_accs[st])[r])
                if const_expr(SAFE_SOFTMAX):
                    # Scale here, before the row max, so m_i lives in the
                    # scaled domain and the exponent is a plain subtract.
                    s_raw = [_fmul(v, c_sm_scale_log2e) for v in s_raw]

                if const_expr(CAUSAL):
                    kv_start_i32 = fx.Int32(kv_block_start)
                    klane_i32 = fx.Int32(klane)
                    q_start_i32 = fx.Int32(q_start)
                    max_kv_col_i32 = kv_start_i32 + fx.Int32(BLOCK_N - 1)
                    tile_needs_mask = max_kv_col_i32 > q_start_i32

                    # FlyDSL's `if` rewriter requires each conditional state
                    # variable to be a single MLIR Value, not a list. Unfold
                    # s_raw into NUM_S_VALS named scalars, reassign each inside
                    # the branch, then rebuild the list. NUM_S_VALS == 16 for
                    # BLOCK_N == 32, which is why causal is gated to that.
                    # (This whole block dies with the interval decomposition.)
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

                if const_expr(SAFE_SOFTMAX):
                    # m is already scaled, so no scale appears in either
                    # exponent -- this is the whole point of the change.
                    corr = rocdl.exp2(
                        ir.F32Type.get(), _raw(_fsub(m_running, m_new_raw))
                    )
                    neg_m = _fsub(c_zero_f, m_new_raw)
                else:
                    diff_m_raw = _fsub(m_running, m_new_raw)
                    diff_m_scaled = _fmul(diff_m_raw, c_sm_scale_log2e)
                    corr = rocdl.exp2(ir.F32Type.get(), _raw(diff_m_scaled))
                    scaled_max = _fmul(c_sm_scale_log2e, m_new_raw)
                    neg_m = _fsub(c_zero_f, scaled_max)

                p_vals = []
                local_sum = _raw(c_zero_f)
                for r in range_constexpr(NUM_S_VALS):
                    if const_expr(SAFE_SOFTMAX):
                        diff = _fadd(s_raw[r], neg_m)
                    else:
                        diff = fmath.fma(
                            s_raw[r], _raw(c_sm_scale_log2e), neg_m
                        )
                    p = rocdl.exp2(ir.F32Type.get(), _raw(diff))
                    p_vals.append(p)
                    local_sum = _fadd(local_sum, p)

                peer_sum = reduction_peer(local_sum)
                tile_sum = _fadd(local_sum, peer_sum)
                l_corr = _fmul(corr, l_running)
                l_new = _fadd(l_corr, tile_sum)

                corr_vec = (
                    Vec.from_elements([corr], fx.Float32).broadcast_to(8).ir_value()
                )
                for dc in range_constexpr(O_ACCS):
                    o_accs_all[qt][dc] = _fmul(o_accs_all[qt][dc], corr_vec)

                m_new_all.append(m_new_raw)
                l_new_all.append(l_new)
                p_vals_all.append(p_vals)

            # ==== Build P packs, per Q row-tile ====
            p_packs_all_qt = []
            for qt in range_constexpr(Q_ROW_TILES):
                p_vals = p_vals_all[qt]
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
                p_packs_all_qt.append(p_packs_all)

            def _gemm2_chunk(_vc):
                """GEMM2 over the V window currently resident in LDS."""
                v_base = v_buf_base(0)

                def _load_v(st_kv_base_val, pks_val, dc_val):
                    if const_expr(V_TRANSPOSED):
                        # V^T[d][kv]: the 8 kv values this lane needs are
                        # contiguous, so this is one vector read instead of 8
                        # strided scalar loads.
                        d_pos = shard_vo_off + fx.Index(dc_val * D_CHUNK) + lane16
                        kv0 = (
                            fx.Index(st_kv_base_val + pks_val * PV_K_STEP)
                            + klane * WMMA_LANE_K
                        )
                        return _lds_load_v8(v_base + d_pos * VT_STRIDE + kv0)
                    d_pos = fx.Index(dc_val * D_CHUNK) + lane16
                    v_elems = []
                    for k_sub in range_constexpr(8):
                        kv_row = (
                            fx.Index(st_kv_base_val + pks_val * PV_K_STEP)
                            + klane * WMMA_LANE_K
                            + fx.Index(k_sub)
                        )
                        v_elems.append(
                            fx.ptr_load(lds_kv + fx.Int32(v_base + kv_row * V_STRIDE + d_pos))
                        )
                    return Vec.from_elements(v_elems, elem_dtype).ir_value()

                # Software pipeline: preload the first V pack, then prefetch the
                # next one while the current WMMA runs.
                cur_v_packs = []
                for st_idx in range_constexpr(N_SUB_TILES):
                    cur_v_packs.append(_load_v(st_idx * K_SUB_N, 0, 0))

                for pks in range_constexpr(PV_K_STEPS):
                    for dc in range_constexpr(D_CHUNKS):
                        next_dc = dc + 1
                        next_pks = pks
                        if const_expr(next_dc >= D_CHUNKS):
                            next_dc = 0
                            next_pks = pks + 1
                        has_next = const_expr(next_pks < PV_K_STEPS)

                        next_v_packs = []
                        if const_expr(has_next):
                            for st_idx in range_constexpr(N_SUB_TILES):
                                next_v_packs.append(
                                    _load_v(st_idx * K_SUB_N, next_pks, next_dc)
                                )

                        # One V operand, Q_ROW_TILES WMMAs: this is what halves
                        # the V LDS reads per FLOP at Q_ROW_TILES > 1.
                        for st_idx in range_constexpr(N_SUB_TILES):
                            for qt in range_constexpr(Q_ROW_TILES):
                                o_accs_all[qt][_vc * D_CHUNKS + dc] = wmma_acc(
                                    cur_v_packs[st_idx],
                                    p_packs_all_qt[qt][st_idx][pks],
                                    o_accs_all[qt][_vc * D_CHUNKS + dc],
                                )

                        if const_expr(has_next):
                            cur_v_packs = next_v_packs

            if const_expr(VO_CHUNKS == 1):
                # Unchunked: keep the original shape exactly. Wrapping this in a
                # 1-trip chunk loop cost 4 VGPRs (190 -> 194), which crosses an
                # allocation-granularity boundary and drops occupancy 8 -> 7
                # waves/SIMD, measured at -4% on head_dim 128.
                if const_expr(V_PREFETCH_DIST):
                    coop_store_v_lds(_v_vecs_cur, 0)
                    gpu.barrier()
                    # Issue the next tile's V here, not after GEMM2: this way
                    # the load flies over GEMM2 instead of only over the loop
                    # back-edge. (The baseline kernel issues it after GEMM2;
                    # bp's placement is strictly better and is what aiw uses
                    # for both schedules.)
                    _v_vecs_next = coop_load_v_global(next_kv_start, 0)
                else:
                    coop_store_v_lds(coop_load_v_global(kv_block_start, 0), 0)
                    gpu.barrier()
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
                        # All waves must finish reading this window before the
                        # next chunk overwrites it.
                        gpu.barrier()

            _yield_args = []
            for qt in range_constexpr(Q_ROW_TILES):
                _yield_args.append(m_new_all[qt])
                _yield_args.append(l_new_all[qt])
            for qt in range_constexpr(Q_ROW_TILES):
                for i in range_constexpr(O_ACCS):
                    _yield_args.append(o_accs_all[qt][i])
            if const_expr(K_PREFETCH_DIST):
                for batch in range_constexpr(NUM_BATCHES_KV):
                    _yield_args.append(_k_vecs_next[batch])
            if const_expr(V_PREFETCH_DIST):
                for batch in range_constexpr(V_LOADS):
                    _yield_args.append(_v_vecs_next[batch])
            loop_results = yield _yield_args

        # ---- Normalize and store O ----
        for qt in range_constexpr(Q_ROW_TILES):
            l_final = loop_results[2 * qt + 1]
            inv_l = arith.divf(_raw(c_one_f), _raw(l_final), fastmath=fm_fast)
            inv_l_vec = (
                Vec.from_elements([inv_l], fx.Float32).broadcast_to(8).ir_value()
            )
            if q_in_bounds_all[qt]:
                for _oi in range_constexpr(O_ACCS):
                    vc, dc = _oi // D_CHUNKS, _oi % D_CHUNKS
                    o_norm_vec = _fmul(loop_results[_ML + qt * O_ACCS + _oi], inv_l_vec)
                    o_trunc = Vec(o_norm_vec).to(elem_dtype).ir_value()
                    d_col = shard_vo_off + fx.Index(dc * D_CHUNK) + klane * 8
                    if const_expr(vc):
                        d_col = fx.Index(vc * VO_CHUNK_COLS) + d_col
                    if const_expr(D_OFFSET):
                        d_col = fx.Index(D_OFFSET) + d_col
                    # The store is 8 columns wide, so under PADDED_HEAD its
                    # last chunk can straddle hdim_vo. Only whole-chunk
                    # skipping is available for a store, and that is exact
                    # here: O is the caller's tensor, whose D pitch is a
                    # 16-byte multiple, so columns in [hdim_vo,
                    # ceil8(hdim_vo)) lie inside O's own allocation. They
                    # receive computed-but-unused values, mirroring the pad
                    # region of the inputs, which the caller slices off.
                    def _emit_o_store():
                        _store_global_half(
                            o_ptr,
                            o_tbase(q_start),
                            o_toff(q_rows_in_tile[qt], d_col),
                            o_trunc,
                        )

                    if const_expr(PADDED_HEAD):
                        if d_col < _hdim_vo_i:
                            _emit_o_store()
                    else:
                        _emit_o_store()

    @flyc.jit
    def launch_flash_attn_aiw(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        batch_size: fx.Int32,
        seq_len: fx.Int32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        stride_q0: fx.Int64,
        stride_q1: fx.Int64,
        stride_q2: fx.Int64,
        stride_k0: fx.Int64,
        stride_k1: fx.Int64,
        stride_k2: fx.Int64,
        stride_v0: fx.Int64,
        stride_v1: fx.Int64,
        stride_v2: fx.Int64,
        stride_o0: fx.Int64,
        stride_o1: fx.Int64,
        stride_o2: fx.Int64,
        sm_scale_v: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()

        bs_idx = fx.Index(batch_size)
        sl_idx = fx.Index(seq_len)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M

        # Strides come from the caller, read off the real tensors -- never
        # derived here from num_heads/head_dim. The shape does not determine
        # the layout (plan1 section 0), and K/V need not share Q's layout at
        # all. Axis 3 is D, contiguous by contract, so it is not passed.
        # Always forwarded: with STRIDES_CONSTEXPR the kernel simply does not
        # read them, which is what keeps the two arms directly comparable.
        launcher = flash_attn_func_aiw_kernel(
            Q, K, V, O, seq_len,
            num_head_q, num_head_k,
            hdim_qk, hdim_vo,
            stride_q0, stride_q1, stride_q2,
            stride_k0, stride_k1, stride_k2,
            stride_v0, stride_v1, stride_v2,
            stride_o0, stride_o1, stride_o2,
            sm_scale_v,
        )

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
        # with no LDS latency hiding. max-memory-clause trades VGPRs for keeping
        # several loads in flight.
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
                    [ir.StringAttr.get("no-nans-fp-math"), ir.StringAttr.get("true")]
                )
            )
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [ir.StringAttr.get("unsafe-fp-math"), ir.StringAttr.get("true")]
                )
            )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)

        launcher.launch(
            grid=(fx.Index(num_head_q), num_q_tiles, bs_idx),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

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

    def _strides_of(t, name):
        """The three leading strides of a rank-4 tensor, in elements.

        Read from the tensor, never derived from its shape: the shape says
        BHSD/BSHD but the memory can be any `xxxD` permutation, and the two are
        not related (see sdpa-close-gap-plan1.md section 0).
        """
        if not hasattr(t, "stride"):
            raise TypeError(
                f"{name} must be a rank-4 tensor so its strides can be read, "
                f"got {type(t).__name__}"
            )
        if t.dim() != 4:
            raise ValueError(f"{name} must be rank 4, got shape {tuple(t.shape)}")
        if t.stride(3) != 1:
            raise ValueError(
                f"{name} must have a contiguous last dimension, got "
                f"stride(3)={t.stride(3)}"
            )
        return t.stride(0), t.stride(1), t.stride(2)

    def _resolve_scale(Q, scale):
        """Default sm_scale from the tensor's *real* head dim, not the tile.

        The builder can only default to `1/sqrt(BLOCK_DMODEL)`, since it has no
        idea what `hdim_qk` will be -- and under PADDED_HEAD that is the wrong
        number. Deriving it here from `Q.shape[3]` is right in both cases and
        identical to the builder default whenever hdim == the tile.
        """
        if scale is not None:
            return float(scale)
        if PADDED_HEAD and hasattr(Q, "shape"):
            return 1.0 / host_math.sqrt(Q.shape[3])
        return float(sm_scale)

    def _prep(Q, K, V, O):  # noqa: E741
        """Pointers, head counts and the twelve strides, in launch order.

        Deliberately does **not** flatten: `t.reshape(-1)` materialises a copy
        for any non-contiguous tensor, which would silently defeat the whole
        point of reading strides. `data_ptr()` is the tensor's base either way.
        """
        st = []
        for t, name in ((Q, "Q"), (K, "K"), (V, "V"), (O, "O")):
            st.extend(_strides_of(t, name))
        # BSHD: axis 2 is the head axis. Read rather than assumed -- under
        # MQA/GQA K and V carry fewer heads than Q.
        nhq, nhk = Q.shape[2], K.shape[2]
        hqk, hvo = Q.shape[3], V.shape[3]
        if V.shape[2] != nhk:
            raise ValueError(f"K and V must share num_heads, got {nhk} and {V.shape[2]}")
        if O.shape[2] != nhq:
            raise ValueError(f"O and Q must share num_heads, got {O.shape[2]} and {nhq}")
        if nhq % nhk:
            raise ValueError(
                f"num_heads_q ({nhq}) must be divisible by num_heads_k ({nhk})"
            )
        return [_ptr_arg(t) for t in (Q, K, V, O)], (nhq, nhk, hqk, hvo), st

    launch_flash_attn_aiw.compile_hints = dict(_fmha_compile_hints)

    def _launch(Q, K, V, O, batch_size, seq_len, scale=None, stream=None):  # noqa: E741
        ptrs, meta, st = _prep(Q, K, V, O)
        _run_compiled(
            launch_flash_attn_aiw,
            *ptrs,
            batch_size,
            seq_len,
            *meta,
            *st,
            _resolve_scale(Q, scale),
            stream if stream is not None else fx.Stream(None),
        )

    def _compile(Q, K, V, O, batch_size, seq_len, scale=None, stream=None):  # noqa: E741
        ptrs, meta, st = _prep(Q, K, V, O)
        return flyc.compile(
            launch_flash_attn_aiw,
            *ptrs,
            batch_size,
            seq_len,
            *meta,
            *st,
            _resolve_scale(Q, scale),
            fx.Stream(stream),
        )

    _launch.compile = _compile
    return _launch


build_flash_attn_func_aiw_module = build_flash_attn_func_aiw_module_primary
