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

from torch import float32 as torch_f32

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
from flydsl.expr.typing import T, Vector as Vec
from philox import Philox, dropout_threshold
from flydsl.expr.utils.arith import ArithValue, _to_raw as _raw

from gfx1201_standalone import wmma_ops

from dataclasses import fields

from fmha_tuning_gfx1201 import (  # noqa: F401
    FmhaInputMetadata,
    FmhaKnobs,
    resolve_knobs,
    default_block_m,
    default_block_n,
    default_prefetch_dist,
    q_tiles_per_block,
    qk_shards,
    resolve_shards,
    vo_chunks,
)

KERNEL_NAME = "flash_attn_func_gfx1201_aiw_kernel"
_LOG2E = host_math.log2(host_math.e)
_LN2 = 0.6931471824645996  # matches AOTriton's literal exactly

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


# Causal alignment as a *window* sentinel, AOTriton's `WindowValue`. These
# occupy two values a real bound never takes, so `Window_left`/`Window_right`
# stay plain signed integers rather than gaining a discriminant.
_WINDOW_TOPLEFT = -2147483647    # 0x80000001
_WINDOW_BOTRIGHT = -2147483646   # 0x80000002



def build_flash_attn_func_aiw_module_primary(meta, knobs):
    """Build the unified gfx1201 flash-attention kernel.

    Takes the two objects rather than 26 loose parameters, split on *who
    decides*: `problem` is what the caller asked for, `schedule` is what the
    tuning policy answered. Build `schedule` with
    `fmha_tuning_gfx1201.resolve_knobs(problem)` -- every field must be
    resolved, since nothing here falls back to a policy any more.

    `build_flash_attn_func_aiw_module` below is the keyword-argument wrapper
    for callers that only want to name a problem.

    See the module docstring for what each knob selects and which of the three
    original kernels a given setting reproduces.
    """
    # Unpacked to locals so the body below reads unchanged. Anything not
    # resolved is a caller error, not a default to be invented here.
    num_heads = meta.num_heads
    causal = meta.causal
    dtype_str = meta.dtype_str
    sm_scale = meta.sm_scale
    causal_type = meta.causal_type
    bias = meta.bias
    dropout = meta.dropout
    philox_width = meta.philox_width

    WAVES_PER_EU = knobs.waves_per_eu
    FLAT_WORK_GROUP_SIZE = knobs.flat_work_group_size
    BLOCK_M_KNOB = knobs.block_m
    BLOCK_N_KNOB = knobs.block_n
    BLOCK_DMODEL = knobs.block_dmodel
    BLOCK_DMODEL_V = knobs.block_dmodel_v
    D_OFFSET = knobs.d_offset
    K_PREFETCH_DIST_KNOB = knobs.k_prefetch_dist
    V_PREFETCH_DIST = knobs.v_prefetch_dist
    V_LDS_LAYOUT = knobs.v_lds_layout
    STRIDES_CONSTEXPR = knobs.strides_constexpr
    PADDED_HEAD = knobs.padded_head
    Q_ROW_TILES = knobs.q_row_tiles
    SHARDS = knobs.shards
    UNSAFE_FP_MATH = knobs.unsafe_fp_math
    FAST_FP_MATH = knobs.fast_fp_math
    DENORMALS_ARE_ZERO = knobs.denormals_are_zero
    SCHED_STRATEGY_KNOB = knobs.sched_strategy
    LPT_TILE_ORDER_KNOB = knobs.lpt_tile_order
    UNSAFE_NO_KV_CLAMP = knobs.unsafe_no_kv_clamp
    KV_ADDR_HOIST = knobs.kv_addr_hoist
    FP_MODE = knobs.fp_mode
    PATH_TAG = knobs.path_tag
    """Build the unified gfx1201 flash-attention kernel.

    See the module docstring for what each knob selects and which of the three
    original kernels a given setting reproduces.
    """

    # ---- WMMA / wave32 constants ----
    #
    # The first four are the hardware: RDNA4 runs wave32 and its WMMA is
    # 16x16x16. The last two are *derived* from those, not free parameters, and
    # are spelled out here because both read as magic at the use site.
    WARP_SIZE = 32
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 16

    # KV columns spanned by one *pair* of S accumulators -- `2 * WMMA_N`.
    #
    # The pair, rather than the single accumulator, is the unit because it is
    # where the two GEMMs meet. GEMM1 emits `NUM_S_ACCS = N_SUB_TILES * 2`
    # accumulators of `WMMA_N` columns each, and GEMM2 consumes the same span
    # as `PV_K_STEPS = K_SUB_N // WMMA_K` steps of K. Both equal 2, so a
    # sub-tile is exactly one GEMM1 output pair and one GEMM2 input pair.
    #
    # The mapping this implies is open-coded wherever accumulators are indexed
    # by column: accumulator `st` starts at KV column
    # `(st // 2) * K_SUB_N + (st % 2) * WMMA_N`.
    K_SUB_N = 2 * WMMA_N

    # K elements each lane holds of a WMMA A/B operand.
    #
    # A wave32 WMMA spreads `WMMA_M` rows over 32 lanes, so two lanes share
    # each row and split the K extent between them: `WMMA_K / (WARP_SIZE /
    # WMMA_M)` = 8. That is why the operand is a v8f16 and why `klane` (the
    # half-wave index, `lane // 16`) appears multiplied by this everywhere a
    # column is computed -- half-wave 0 covers K 0..7, half-wave 1 covers 8..15.
    # See the operand-layout note in the module docstring.
    WMMA_LANE_K = WMMA_K // (WARP_SIZE // WMMA_M)


    # ---- Knob resolution ----
    K_PREFETCH_DIST = K_PREFETCH_DIST_KNOB
    # K and V prefetch distances are INDEPENDENT. The baseline kernel is
    # (K=0, V=1) -- its "pre-issue first V global load before loop" carries V in
    # registers exactly as bp does, and only K is staged at distance 0. Folding
    # the two into one knob produces a (K=0, V=0) schedule that exists in none
    # of the originals and costs 9.6% at BLOCK_DMODEL 32 non-causal.
    V_LDS_LAYOUT = (
        ("transposed" if K_PREFETCH_DIST else "row")
        if V_LDS_LAYOUT is None
        else V_LDS_LAYOUT
    )
    V_TRANSPOSED = V_LDS_LAYOUT == "transposed"

    ROWS_PER_WAVE = WMMA_M * Q_ROW_TILES

    BLOCK_N = BLOCK_N_KNOB

    # `_sdiv_rd` in the kernel is an arithmetic shift, which is only a
    # floor-division when BLOCK_N is a power of two.
    _BLOCK_N_LOG2 = BLOCK_N.bit_length() - 1

    N_SUB_TILES = BLOCK_N // K_SUB_N
    NUM_S_ACCS = N_SUB_TILES * 2
    NUM_S_VALS = NUM_S_ACCS * 8


    # V/O column *window*: the slice of the output width this build computes.
    # Distinct from VO_SLICE (a wave's share of a window) and from VO_CHUNK_COLS
    # (a staging pass's share of a window).
    VO_WIDTH = BLOCK_DMODEL_V

    # Head-dimension sharding. QK_SHARDS waves cooperate on one Q row-tile,
    # each reducing over its own BLOCK_DMODEL slice in GEMM1 and owning the matching
    # V/O column slice in GEMM2. QK_SHARDS == 1 is the unsharded kernel: every
    # sharded construct below is behind `const_expr(QK_SHARDS > 1)`.
    # Resolved against the V/O *window*, not BLOCK_DMODEL, so a narrow window does
    # not inherit a shard count it cannot divide.
    if SHARDS is not None:
        QK_SHARDS = SHARDS
    elif K_PREFETCH_DIST == 0 or Q_ROW_TILES > 1:
        QK_SHARDS = 1
    else:
        QK_SHARDS = resolve_shards(BLOCK_DMODEL, VO_WIDTH, BLOCK_N)
    QK_SLICE = BLOCK_DMODEL // QK_SHARDS      # head-dim columns per wave in GEMM1

    VO_CHUNKS = vo_chunks(VO_WIDTH, BLOCK_N, QK_SHARDS) if V_TRANSPOSED else 1
    VO_CHUNK_COLS = VO_WIDTH // VO_CHUNKS   # V columns resident per pass
    VO_SLICE = VO_CHUNK_COLS // QK_SHARDS   # V/O columns per wave per pass


    # ---- Validity predicate over the knob space ----
    #
    # Assertions, not ValueError: every name below was resolved by
    # `fmha_tuning_gfx1201.resolve_knobs`, so a violation is that module
    # contradicting itself, not a caller mistake. Caller input is validated
    # with ValueError in `plan()` and in the launcher.
    #
    # Grouped rather than scattered through the derivation so the buildable
    # subset of the knob space can be read in one place.

    # Shapes and enumerations.
    assert BLOCK_DMODEL % 16 == 0 and 16 <= BLOCK_DMODEL <= 512, (
        f"aiw needs 16 <= BLOCK_DMODEL <= 512 and BLOCK_DMODEL % 16 == 0, got {BLOCK_DMODEL}"
    )
    assert dtype_str in ("f16", "bf16"), f"aiw supports f16/bf16, got {dtype_str!r}"
    assert K_PREFETCH_DIST in (0, 1), f"K_PREFETCH_DIST must be 0 or 1, got {K_PREFETCH_DIST}"
    assert V_PREFETCH_DIST in (0, 1), f"V_PREFETCH_DIST must be 0 or 1, got {V_PREFETCH_DIST}"
    assert V_LDS_LAYOUT in ("row", "transposed"), (
        f"V_LDS_LAYOUT must be 'row' or 'transposed', got {V_LDS_LAYOUT!r}"
    )
    assert Q_ROW_TILES in (1, 2), f"Q_ROW_TILES must be 1 or 2, got {Q_ROW_TILES}"

    # BLOCK_N. The power of two is load-bearing: `_sdiv_rd` in the kernel is an
    # arithmetic shift, which is only a floor-division when BLOCK_N is one.
    assert BLOCK_N % K_SUB_N == 0, f"BLOCK_N ({BLOCK_N}) must be a multiple of K_SUB_N ({K_SUB_N})"
    assert BLOCK_N & (BLOCK_N - 1) == 0, f"BLOCK_N ({BLOCK_N}) must be a power of two"

    # The V/output window, and how it divides.
    assert VO_WIDTH % 16 == 0 and 0 < VO_WIDTH <= BLOCK_DMODEL, (
        f"BLOCK_DMODEL_V must be a positive multiple of 16 and <= BLOCK_DMODEL, got {VO_WIDTH}"
    )
    assert D_OFFSET % 16 == 0 and D_OFFSET + VO_WIDTH <= BLOCK_DMODEL, (
        f"D_OFFSET {D_OFFSET} + BLOCK_DMODEL_V {VO_WIDTH} must fit in BLOCK_DMODEL {BLOCK_DMODEL}"
    )
    assert VO_SLICE % WMMA_N == 0, f"V/O slice {VO_SLICE} must be a multiple of WMMA_N={WMMA_N}"

    # Sharding. A slice not a multiple of WMMA_K would silently drop part of the
    # reduction: BLOCK_DMODEL 224 with 4 shards gives a 56-wide slice, of which
    # only 48 would be reduced (measured rel err 0.97).
    assert BLOCK_DMODEL % QK_SHARDS == 0 and QK_SLICE % WMMA_K == 0, (
        f"BLOCK_DMODEL {BLOCK_DMODEL} with {QK_SHARDS} SHARDS gives a {QK_SLICE}-wide "
        f"slice, which must be a multiple of WMMA_K={WMMA_K}"
    )

    # Combinations the kernel does not implement. Written as conditionals
    # rather than negated conjunctions -- `not (not V_TRANSPOSED and ...)` is
    # a sentence nobody can read twice the same way.
    if not V_TRANSPOSED:
        assert QK_SHARDS == 1, (
            "V_LDS_LAYOUT='row' does not implement cross-shard reduction; use 'transposed'"
        )
        assert VO_CHUNKS == 1, (
            "V_LDS_LAYOUT='row' does not implement chunked V staging; use 'transposed'"
        )
    if VO_CHUNKS > 1:
        assert V_PREFETCH_DIST, "chunked V staging requires V_PREFETCH_DIST=1"
    if Q_ROW_TILES > 1:
        assert QK_SHARDS == 1, "Q_ROW_TILES > 1 with qk_shards > 1 is untested; pick one"
    if causal:
        # The causal mask indexes s_accs as a flat 16, which an unrolled loop
        # over a longer list walks off the end of -- an IndexError at trace
        # time rather than a wrong answer. (Dies with the interval work.)
        assert NUM_S_VALS == 16, (
            f"causal masking requires BLOCK_N == {K_SUB_N} (NUM_S_VALS == 16), "
            f"got BLOCK_N={BLOCK_N} (NUM_S_VALS={NUM_S_VALS})"
        )
    # These combinations are not implemented rather than not expressible. Fail
    # at build time; do not emit a kernel that silently computes the wrong
    # thing.

    # ---- Workgroup geometry ----
    if K_PREFETCH_DIST == 0:
        BLOCK_M = BLOCK_M_KNOB
        # Stays here rather than in the section above: BLOCK_M only exists on
        # this branch, and on the other one it is derived rather than given.
        assert BLOCK_M % ROWS_PER_WAVE == 0, (
            f"BLOCK_M ({BLOCK_M}) must be a multiple of {ROWS_PER_WAVE}"
        )
        Q_TILES_PER_BLOCK = BLOCK_M // ROWS_PER_WAVE
        NUM_WAVES = Q_TILES_PER_BLOCK
    else:
        # Keep the workgroup at TARGET_WAVES by trading Q row-tiles for SHARDS,
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
            1, q_tiles_per_block(BLOCK_DMODEL, QK_SHARDS) // Q_ROW_TILES
        )
        BLOCK_M = ROWS_PER_WAVE * Q_TILES_PER_BLOCK
        NUM_WAVES = Q_TILES_PER_BLOCK * QK_SHARDS

    if FLAT_WORK_GROUP_SIZE is None:
        FLAT_WORK_GROUP_SIZE = NUM_WAVES * WARP_SIZE
    BLOCK_SIZE = FLAT_WORK_GROUP_SIZE

    BLOCK_N_OUT = BLOCK_N

    # LLVM's amdgpu-sched-strategy function attribute; "" leaves the default
    # GCN scheduler in place. See the passthrough block in the launch wrapper.
    # Measured at BATCH=2 H=12 N=4096 d=128 f16 -- distance 1: causal
    # 85.6 -> 88.5 TFLOPS, non-causal 91.4 -> 91.9. Distance 0: causal
    # 69.8 -> 79.2, but non-causal 89.4 -> 88.6, so only causal wants it there.
    # `None` means the policy below; pass `""` for the stock GCN scheduler.
    SCHED_STRATEGY = (
        ("max-memory-clause" if (K_PREFETCH_DIST or causal) else "")
        if SCHED_STRATEGY_KNOB is None
        else SCHED_STRATEGY_KNOB
    )

    K_STEP_QK = WMMA_K
    K_STEPS_QK = QK_SLICE // K_STEP_QK      # GEMM1 K-steps for this wave's slice

    D_CHUNK = WMMA_N
    D_CHUNKS = VO_SLICE // D_CHUNK          # accs per wave per chunk
    O_ACCS = VO_CHUNKS * D_CHUNKS           # accs live across the KV loop, per Q tile

    PV_K_STEP = WMMA_K
    PV_K_STEPS = K_SUB_N // PV_K_STEP

    if sm_scale is None:
        sm_scale = 1.0 / host_math.sqrt(BLOCK_DMODEL)

    NUM_HEADS = num_heads
    HEAD_DIM = BLOCK_DMODEL
    CAUSAL = causal
    STRIDE_TOKEN = NUM_HEADS * HEAD_DIM

    # Strides and sm_scale are runtime arguments, not folded constants: an AOT
    # kernel cannot bake them in, since a fixed set of binaries has to cover
    # every shape.
    #
    # Measured price (B=1 H=8 N=4096 f16, interleaved 3-rep A/B over the full
    # BLOCK_DMODEL ladder x causal): **median ratio 0.996**, worst 0.967 (BLOCK_DMODEL
    # 16 causal), best 1.041 (BLOCK_DMODEL 192 causal). Several configs come out
    # *faster* and the spread is symmetric about 1.0, so this is the board's
    # noise floor rather than a measurable cost. Registers: +0 to +4 VGPRs,
    # +22 SGPRs, no new spills at any BLOCK_DMODEL. Output is bitwise identical to
    # the folded form, sm_scale included.
    #
    # Each tensor carries its own triple, and they are not interchangeable: K
    # and V reach the kernel exactly as the caller allocated them (`mha_fwd_aot`
    # passes them through untouched), and under MQA/GQA they carry Num_head_k
    # rather than Num_head_q. Going from one shared triple to four cost +18
    # SGPRs and zero VGPRs -- strides are uniform scalars, so they only change
    # which value an address multiplies.
    #
    # `STRIDES_CONSTEXPR=True` keeps them folded. It is retained only as an A/B
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
    #   BLOCK_DMODEL  16   31.8 -> 35.7  (+12%)
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
    # Longest-processing-time-first dispatch of the Q tiles. Named for what it
    # is for rather than what it does: under causal masking a tile's cost grows
    # with its index, so issuing the expensive ones first leaves only cheap
    # tiles to fill the tail. With uniform cost -- every non-causal tile -- the
    # reversal is a permutation with no load-balancing content, so `and CAUSAL`
    # is part of the knob's *definition* and is resolved here rather than
    # re-tested at the use site, where it read as an arbitrary restriction on
    # an unrelated flag.
    _LPT_TILE_ORDER = CAUSAL and LPT_TILE_ORDER_KNOB
    # Measurement-only: drops the KV row clamp. UNSAFE in general -- it is
    # what buffer bounds checking would replace -- but valid for a benchmark
    # where seq_len is an exact multiple of BLOCK_M.
    _NO_KV_CLAMP = UNSAFE_NO_KV_CLAMP
    # Floating-point latitude granted to the compiler.
    #
    # "noninf" (default) is `fast` minus `ninf`, and drops the function-level
    # `unsafe-fp-math` / `no-nans-fp-math` attributes. `denormal-fp-math-f32`
    # (DAZ) is kept in every mode -- it is about denormals, not infinities, and
    # it is where the actual win is.
    #
    # Why: `ninf` lets the compiler assume no operand is infinite, so an -inf
    # flowing through a fast-math op may simply be folded away. That is not
    # hypothetical -- it silently deleted the KV tail mask (see the comment
    # there), and it will do the same to a bias tensor, where a boolean
    # attention mask cast to float is exactly a matrix of -inf.
    #
    # Cost of giving it up, measured with DAZ held constant (B=1 H=8 N=4096
    # f16, BLOCK_DMODEL 16/64/128/256/512 x causal): **within noise everywhere** --
    # 91.5 vs 91.9 TFLOPS at BLOCK_DMODEL 128 non-causal, 45.5 vs 45.4 at 512. The
    # permission bought nothing and cost a silent miscompile.
    #
    # `nnan` is retained. NaN can only arise here from -inf minus -inf, which
    # the m_i floor rules out, and the API contract excludes
    # NaN inputs.
    #
    # "fast" restores the old behaviour for A/B; "safe" additionally drops
    # `nnan` (~0.6%).
    _FP_MODE = FP_MODE

    # Two softmax corrections, unconditional. Both come from AOTriton's
    # hard-won list and there is no reason to keep the un-corrected form
    # reachable -- a knob selecting known-wrong numerics is a liability, not a
    # feature.
    #
    # (a) `m_i` initialises to -3.40282e+38, not -inf. If a tile is entirely
    #     masked its row max is -inf, and with an -inf init the rescale becomes
    #     exp2(-inf - -inf) = exp2(NaN) = NaN. A finite floor makes it
    #     exp2(0) = 1 and the masked probabilities exp2(-inf - m) = 0, which is
    #     the right answer. The *mask* fill stays -inf; only the init changes.
    #
    # (b) The QK scale is applied to the scores **before** the row max, so
    #     `m_i` lives in the scaled domain and the exponent is a plain
    #     subtract. The alternative -- `exp2(fma(s, qk_scale, -qk_scale*m))` --
    #     is exactly the FMA pattern AOTriton flags in ROCm/aotriton#54, and it
    #     measurably loses accuracy at large input magnitudes: at BLOCK_DMODEL 128
    #     causal the corrected form is *exact* against an fp64 reference from
    #     magnitude 300 up, where the FMA form sits at 4-7e-4.

    # `BLOCK_DMODEL` is BLOCK_DMODEL: the compile-time tile width, drawn from the
    # ladder. The *real* extents are the runtime `hdim_qk` / `hdim_vo`
    # arguments, and PADDED_HEAD says whether they differ from the tile.
    # AOTriton derives exactly this (`attn_fwd.cc`):
    #     hdim_rounded = round_value(max(hdim_qk, hdim_vo), ladder)
    #     PADDED_HEAD  = (hdim_rounded != hdim_qk || hdim_rounded != hdim_vo)
    # With PADDED_HEAD false the two are equal to the tile and no masking is
    # emitted at all -- that is the common case and the one the ladder measures.

    # Causal masking. `causal_type` is the *caller's* vocabulary, AOTriton's
    # CAUSAL_TYPE: 0 = none, 1 = top-left aligned, 2 = bottom-right aligned,
    # 3 = an explicit sliding window. The two alignments differ only in where
    # the diagonal sits, and they coincide when seqlen_q == seqlen_k -- which
    # is why a single `causal` bool sufficed for a long time.
    #
    #   top-left      key j is visible to query i iff  j <= i
    #   bottom-right  ...                        iff  j <= i + (seqlen_k - seqlen_q)
    #
    # 3 = generalized sliding window: the test becomes a two-sided band,
    # `i - window_left <= j <= i + window_right`, with both bounds signed
    # runtime arguments. It subsumes 1 and 2 exactly -- (seqlen_q, 0) and
    # (seqlen_q, seqlen_k - seqlen_q) respectively -- which is why AOTriton
    # ships only {0, 3} and resolves 1/2 on the host. Types 1 and 2 survive
    # here only until that equivalence is nailed down by a test; see
    # sdpa-gswa-plan.md.
    #
    # PyTorch's is_causal=True is top-left; see
    # https://github.com/pytorch/pytorch/issues/108108 for the debate about
    # changing that default.
    if causal_type is None:
        HOST_CAUSAL_TYPE = 1 if causal else 0
    else:
        HOST_CAUSAL_TYPE = causal_type
    if HOST_CAUSAL_TYPE not in (0, 1, 2, 3):
        raise ValueError(f"causal_type must be 0, 1, 2 or 3, got {HOST_CAUSAL_TYPE}")
    if bool(HOST_CAUSAL_TYPE) != bool(causal):
        raise ValueError(
            f"causal={causal} disagrees with causal_type={HOST_CAUSAL_TYPE}"
        )
    # **The kernel only ever sees 0 or 3.** 1 and 2 are host-side conveniences
    # that resolve to a window before dispatch, which is what AOTriton ships
    # (`@ati.scalar('CAUSAL_TYPE', options=[0, 3])`). Keeping them in the
    # kernel would leave two ways to express one diagonal, free to drift apart
    # under maintenance. The window path reproduces both *bitwise*, which is
    # what licensed removing them; see sdpa-gswa-plan.md section 0.
    CAUSAL_TYPE = 0 if HOST_CAUSAL_TYPE == 0 else 3

    # Bias tensor, AOTriton's BIAS_TYPE. 0 = none, 1 = a (B, H, Sq, Sk) matrix
    # added to the scores before the softmax. A build axis, so BIAS_TYPE == 0
    # emits nothing at all -- the loop is latency-bound and a bias that costs
    # anything when unused would be paid by every caller who does not want it.
    BIAS_TYPE = 1 if bias else 0
    if BIAS_TYPE and CAUSAL_TYPE:
        # Undefined, not unimplemented. Causal is an attention mask with a
        # fixed pattern; bias *is* an attention mask supplied directly, since
        # a large negative or -inf entry is how callers spell "do not attend
        # here". Asking for both asks which wins where they disagree, and
        # there is no answer -- the same thing has been specified twice in two
        # vocabularies with no rule for reconciling them.
        #
        # AOTriton disables the functional; PyTorch's math backend raises
        # "Explicit attn_mask should not be set when is_causal=True" and its
        # flash backend has no kernel for the pair. See sdpa-bias-plan.md 3.2.
        raise ValueError(
            "bias and causal masking are mutually exclusive: bias already is "
            "an attention mask, so combining it with a causal one has no "
            "defined meaning. Fold the causal pattern into the bias tensor, "
            "or drop the bias"
        )

    # Dropout, AOTriton's ENABLE_DROPOUT. A build axis, so a build without it
    # emits no PRNG at all -- the loop is latency-bound and a caller who does
    # not want dropout should not pay for the option.
    #
    # The PRNG itself lives in `philox.py`: it is not attention, the backward
    # pass and the debug mask kernel need the identical stream, and this file
    # is long enough. What stays here is the *offset scheme* -- which element
    # gets which offset -- because that is layout-specific.
    ENABLE_DROPOUT = bool(dropout)
    PHILOX = Philox.for_arch() if philox_width is None else Philox(width=philox_width)
    RN_PER_OFFSET = PHILOX.randoms_per_offset

    # Mask KV columns past seqlen_k. Required whenever seqlen need not divide
    # BLOCK_N -- i.e. always, now that the interface no longer pads. The guard
    # inside is dynamic, so interior tiles cost one scalar compare.
    NEEDS_KV_TAIL_MASK = True

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

    # Cooperative-load vector width, in elements. 8 == 16 bytes.
    #
    # Fixed at 8, which is exactly what the alignment contract guarantees: the
    # D-axis pitch is a multiple of 16 bytes, nothing more. A 16-element
    # (32-byte) load needs 32-byte alignment, and there is no way to establish
    # it -- the row address is `base + row * stride_seq`, and `stride_seq` need
    # only be a multiple of the pitch. A tensor whose pitch is an odd multiple
    # of 8 elements (say a 16-wide head sliced out of a 24-wide allocation)
    # puts every odd row on a 16-byte boundary, and the wider load is then
    # undefined behaviour. This is the same over-promised-alignment failure
    # documented on `_lds_load_v8` for LDS, where it cost 2.2x.
    #
    # It is also not a win. Measured 8 against 16 (B=1 H=8 N=4096 f16, TFLOPS):
    #
    #   BLOCK_DMODEL   non-causal        causal
    #       16    37.4 -> 40.2   28.2 -> 35.7
    #       32    64.0 -> 61.4   47.0 -> 47.2
    #       64    81.0 -> 81.4   74.3 -> 70.2
    #      128    92.3 -> 92.1   83.9 -> 81.9
    #      192    97.7 -> 94.5   74.0 -> 76.5
    #      256    89.1 -> 90.1   72.6 -> 73.7
    #      512    45.7 -> 55.1   43.1 -> 51.6
    #
    # Median +0.8%, best +27% (BLOCK_DMODEL 16 causal), worst -5.5%. So the wider
    # load was buying nothing on average while carrying an alignment hazard
    # that no test would catch -- it would fault or corrupt only on layouts we
    # do not currently generate. Removed rather than tuned: tuning an unsound
    # knob just spreads the hazard across more configs.
    VEC_WIDTH = 8

    def _load_geom(width):
        """Cooperative-load geometry for a row of `width` elements."""
        tpr = width // VEC_WIDTH
        rpb = BLOCK_SIZE // tpr
        nb = (BLOCK_N + rpb - 1) // rpb
        return tpr, rpb, nb, nb * rpb != BLOCK_N

    # Cover BLOCK_N rows with ceil() batches, not floor(). Flooring silently
    # dropped rows whenever ROWS_PER_BATCH_LOAD neither reached BLOCK_N nor
    # divided it: BLOCK_DMODEL 160/192/224 give 25/21/18, so BLOCK_N // that == 1
    # and only 25/21/18 of the 32 KV rows reached LDS. The rest was stale LDS,
    # which surfaced as NaN.
    THREADS_PER_ROW_LOAD, ROWS_PER_BATCH_LOAD, NUM_BATCHES_KV, KV_NEEDS_GUARD = _load_geom(HEAD_DIM)

    # global_load_tr_b128 transposes an 8x8 tile of 16-bit elements across each
    # group of 8 lanes, so one wave-wide TR load produces a 16(d) x 16(kv) block
    # already in WMMA-operand layout. Split those blocks over the waves.
    V_TR_D_BLOCKS = VO_CHUNK_COLS // WMMA_N
    _V_TR_TILES = V_TR_D_BLOCKS * (BLOCK_N // WMMA_K)
    # The V TR tiling need not divide evenly across the waves: tail tiles are
    # guarded at the LDS store. Requiring divisibility used to force BLOCK_DMODEL
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
        L: fx.Pointer,
        Bias: fx.Pointer,
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
        philox_seed: fx.Int64,
        philox_offset_base: fx.Int64,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
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
        stride_b0: fx.Int64,
        stride_b1: fx.Int64,
        stride_b2: fx.Int64,
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
        # Fast-math set for the softmax arithmetic. `fast` includes `ninf`,
        # which is what silently deleted the KV tail mask (see the comment at
        # that mask). FMHA_FP_MODE selects a narrower set for measurement.
        _F = arith.FastMathFlags
        if const_expr(_FP_MODE == "fast"):
            fm_fast = _F.fast
        elif const_expr(_FP_MODE == "noninf"):
            fm_fast = _F.reassoc | _F.nnan | _F.nsz | _F.arcp | _F.contract | _F.afn
        else:  # "safe": also drop nnan
            fm_fast = _F.reassoc | _F.nsz | _F.arcp | _F.contract | _F.afn

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
            # Split as elsewhere: the (batch, head, tile) origin and the
            # intra-tile part, added in 64 bits. Feeding LLVM
            # `uniform_i64 + divergent` is what lets SelectGlobalSAddr keep the
            # base in SGPRs instead of forcing a 64-bit VGPR address pair.
            #
            # The divergent half is *not* narrowed to i32 on the way. It
            # carries `row_in_tile * s_seq`, and a view's sequence stride is
            # bounded by the tensor it was taken from rather than by the shape
            # here -- eight heads sliced out of a 1 GiB (1, 64, 16384, 512) f16
            # tensor give `s_seq = 8388608`, whose 256th row is exactly 2**31.
            base_bytes = arith.index_cast(T.i64, _raw(fx.Index(base64) * 2))
            off_bytes = arith.index_cast(T.i64, _raw(fx.Index(off32) * 2))
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
            # Dispatch is on the operand element type, in `wmma_ops`, rather
            # than on `dtype_str` here -- see that module for why.
            return wmma_ops.wmma_f32_16x16x16(a_v8, b_v8, c_v8, v8f32_type)

        def _scmp_i32(pred, a, b):
            """A *signed* integer compare, with both operands forced to i32.

            The coercion is the point, not the predicate. `fx.Int32` is
            declared `signed=True` and its `<`/`>` overloads already emit
            `slt`/`sgt` -- an earlier version of this docstring claimed
            otherwise and was wrong. What is genuinely unsafe is comparing an
            `fx.Index`, which is `signed=False` and 64-bit, so *its* overloads
            emit `ult`/`ugt` and a negative window bound compares as something
            enormous.

            Wrapping both sides in `fx.Int32` first makes the signedness a
            property of this call rather than of whatever the caller happened
            to be holding. Once the sequence-space quantities are `fx.Int32`
            throughout there is nothing left to coerce and this helper goes
            away -- see `sdpa-readability-plan.md` P3.3.
            """
            return ArithValue(
                arith.cmpi(pred, _raw(fx.Int32(a)), _raw(fx.Int32(b)))
            )

        def _ssel_i32(pred, a, b):
            return fx.Int32(ArithValue(pred).select(fx.Int32(a), fx.Int32(b)))

        def _smin_i32(a, b):
            return _ssel_i32(_scmp_i32(arith.CmpIPredicate.slt, a, b), a, b)

        def _smax_i32(a, b):
            return _ssel_i32(_scmp_i32(arith.CmpIPredicate.sgt, a, b), a, b)

        def _sdiv_rd(x):
            """floor(x / BLOCK_N), signed -- plan section 2.4 rule 4.

            BLOCK_N is a power of two, so an arithmetic right shift *is* the
            round-toward-negative-infinity division. `arith.divsi` truncates
            toward zero instead, which is wrong for every window that reaches
            past the start of the sequence: at BLOCK_N 32, floor(-1/32) is -1
            but truncation gives 0, and the left run would then start one tile
            too late and silently drop live columns.
            """
            return fx.Int32(
                ArithValue(
                    arith.shrsi(_raw(fx.Int32(x)), _raw(fx.Int32(_BLOCK_N_LOG2)))
                )
            )

        # ---- Varlen prologue: VarlenBits -> six scalars ----
        #
        # The *only* place the layout is examined. Everything downstream reads
        # the scalars and cannot tell which of the configurations in
        # sdpa-varlen-plan.md section 2 it is running under -- which is that
        # plan's objective 3, and why there is no `if varlen_mode` in the body.
        #
        # `z` is uniform across the workgroup, so every load here is scalar:
        # at most four, once, into SGPRs. They do not touch the VGPR budget.
        #
        # Real branches, not selects. A select-based decode would issue the
        # loads unconditionally and fault on the null `seqinfo` pointers that
        # the dense case passes; the branch also keeps `VarlenBits == 0` free
        # rather than merely correct.
        _z_i32 = fx.Int32(gpu.block_idx.z)

        # The `seqinfo` arguments arrive as untyped byte pointers, so type
        # them once and index in elements. `fx.recast_iter` + `fx.ptr_load` is
        # the DSL idiom (see `kernels/moe/moe_a8w4_mxscale_gfx1250.py` and
        # `kernels/gemm/mxfp4_preshuffle.py`, which build the same type).
        #
        # The shorthand `fx.recast_iter(fx.Int32, ptr)` does *not* work here:
        # it inherits the source pointer's alignment, and a kernel argument
        # arrives as `u8` with alignment 1, so it raises "alignment must be a
        # positive multiple of element byte size (4), got 1". The type has to
        # be spelled out.
        _i32_gptr = fx.PointerType.get(
            elem_ty=fx.Int32.ir_type,
            address_space=fx.AddressSpace.Global,
            alignment=4,
        )

        def _seqinfo_at(ptr, idx_i32):
            return fx.Int32(
                fx.ptr_load(fx.recast_iter(_i32_gptr, ptr) + fx.Int64(idx_i32))
            )

        def _decode_side(bits_shift, max_seqlen, s0, s1):
            """One side of VarlenBits. Called twice, identically.

            Returns (seqlen, row_off, batch_index, tokens). See section 3.1
            of sdpa-varlen-plan.md; the axes are STACKED (bit 0), LENGTH (bits
            2:1) and POSITION (bits 4:3).

            `tokens` is the LSE token pitch, and only the Q call's is used. It
            is derived rather than passed because the logsumexp tensor -- alone
            among the tensors here -- is always compact, so its strides are a
            function of the bits and would otherwise be a second source of
            truth for one fact (plan section 4.2).
            """
            _b = fx.Int32(varlen_bits) >> fx.Int32(bits_shift)
            _stacked = _b & fx.Int32(1)
            _lenmode = (_b >> fx.Int32(1)) & fx.Int32(3)
            _posmode = (_b >> fx.Int32(3)) & fx.Int32(3)

            # LENGTH. `_s0_z` is the cumulative *position* that REUSE then
            # reuses -- the whole point of that mode is that this load has
            # already happened (plan section 1.2).
            _seqlen = fx.Int32(max_seqlen)
            _s0_z = fx.Int32(0)
            if _lenmode == fx.Int32(1):          # CUMULATIVE
                _s0_z = _seqinfo_at(s0, _z_i32)
                _seqlen = _seqinfo_at(s0, _z_i32 + fx.Int32(1)) - _s0_z
            elif _lenmode == fx.Int32(2):        # INDIVIDUAL
                _seqlen = _seqinfo_at(s0, _z_i32)

            # POSITION.
            _row_off = _ssel_i32(
                _scmp_i32(arith.CmpIPredicate.ne, _stacked, fx.Int32(0)),
                _z_i32 * fx.Int32(max_seqlen), fx.Int32(0),
            )
            if _posmode == fx.Int32(1):          # REUSE: already in a register
                _row_off = _s0_z
            elif _posmode == fx.Int32(2):        # ARRAY
                _row_off = _seqinfo_at(s1, _z_i32)

            _batch = _ssel_i32(
                _scmp_i32(arith.CmpIPredicate.ne, _stacked, fx.Int32(0)),
                fx.Int32(0), _z_i32,
            )

            # LSE token pitch. Batched layouts pad every row-group to
            # Max_seqlen; stacked ones run to the batch total, which lives in
            # slot [N] of whichever array supplies positions. That `[N]` read
            # is the prefix-sum assumption of plan section 9.4, asserted host
            # side.
            _tokens = fx.Int32(max_seqlen)
            if _stacked != fx.Int32(0):
                _tokens = fx.Int32(num_seqlens) * fx.Int32(max_seqlen)
                if _posmode == fx.Int32(1):
                    _tokens = _seqinfo_at(s0, fx.Int32(num_seqlens))
                elif _posmode == fx.Int32(2):
                    _tokens = _seqinfo_at(s1, fx.Int32(num_seqlens))
            return _seqlen, _row_off, _batch, _tokens

        _seqlen_q_i32, _q_row_off, _q_batch, _lse_tokens = _decode_side(
            0, max_seqlen_q, seqinfo_q0, seqinfo_q1
        )
        _seqlen_k_i32, _k_row_off, _k_batch, _ = _decode_side(
            8, max_seqlen_k, seqinfo_k0, seqinfo_k1
        )

        seqlen_q_v = fx.Index(_seqlen_q_i32)
        seqlen_k_v = fx.Index(_seqlen_k_i32)

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
        shard_qk_off = shard_id * fx.Index(QK_SLICE)   # into Q/K BLOCK_DMODEL
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
        # BLOCK_DMODEL 16 0.587, 32 0.612, 64 0.715, 128 0.769. Non-causal is
        # indifferent (all within 1%), which is what identifies the cause.
        #
        # Note AOTriton uses dim3{S,H,B} -- q_tile fastest -- for NUM_XCDS == 1.
        # That is not a contradiction: it also forces PERSISTENT_TYPE = 2 for
        # every causal functional, which replaces the grid with a work-stealing
        # loop and makes the axis order irrelevant. Porting its grid_calculator
        # verbatim without persistent-dynamic would reintroduce the regression
        # above. Revisit this ordering when persistent-dynamic lands.
        head_q = fx.Index(gpu.block_idx.x)
        if const_expr(_LPT_TILE_ORDER):
            # Max_seqlen_q, not this sequence's length: the reversal has to
            # be a permutation of the *grid*, whose y extent the host sized
            # from Max_seqlen_q.
            _ntiles = (fx.Index(max_seqlen_q) + (BLOCK_M - 1)) // BLOCK_M
            q_tile_idx = _ntiles - fx.Index(1) - fx.Index(gpu.block_idx.y)
        else:
            q_tile_idx = fx.Index(gpu.block_idx.y)
        start_q = q_tile_idx * BLOCK_M

        # Does this workgroup own any real Q row? Under varlen the grid's Q
        # extent is sized from Max_seqlen_q, so whole workgroups land past the
        # end of a shorter sequence and this is false for them.
        _alive = ArithValue(
            arith.cmpi(arith.CmpIPredicate.slt, _raw(start_q), _raw(seqlen_q_v))
        )

        # **The Q base must be clamped, not just the row within the tile.**
        # `q_tbase(start_q)` folds start_q into the 64-bit base, and the
        # in-bounds guard below only clamps the row *inside* the tile -- so a
        # dead workgroup still addresses `row_off + start_q` rows in, which
        # for a packed tensor runs past the end of the whole allocation, not
        # merely past this sequence. Dense never reached it: there the grid is
        # exactly ceil(seqlen_q / BLOCK_M), so start_q < seqlen_q always.
        #
        # Faults for real -- a 1.3 MB overshoot on a 16-sequence batch hits an
        # unmapped page. It is a *read*, and one whose result is discarded, so
        # smaller overshoots land inside the allocation and are silently
        # harmless, which is exactly why the varlen tests did not catch it.
        _q_start_addr = fx.Index(
            ArithValue(_alive).select(start_q, fx.Index(0))
        )

        # MQA/GQA: Num_head_q / Num_head_k query heads share each KV head.
        # The ratio is uniform and computed once, so the scalar divide is
        # immaterial; the per-head division below is by that ratio.
        head_k = head_q // (fx.Index(num_head_q) // fx.Index(num_head_k))

        load_row_in_batch = tid // THREADS_PER_ROW_LOAD
        load_lane_in_row = tid % THREADS_PER_ROW_LOAD
        load_col_base = load_lane_in_row * VEC_WIDTH

        v_row_in_batch = tid // V_TPR_LOAD
        v_col_base = (tid % V_TPR_LOAD) * VEC_WIDTH

        # `max(seqlen_k - 1, 0)`. `fx.Index` is **unsigned**, so a bare
        # `seqlen_k - 1` wraps to 2**64-1 at seqlen_k == 0 and the KV clamp
        # below then pins every address to that row -- the fault lands at
        # 0xfffffffff000, which is that value truncated to the virtual address
        # width.
        #
        # Now unreachable: the only caller that could arrive here with
        # seqlen_k == 0 was the prologue prefetch, and that is skipped when
        # there are no KV tiles. Kept because the hazard is an unsigned wrap,
        # which produces a plausible address rather than an obvious failure,
        # and it costs one scalar op to pin.
        _slast_i32 = _smax_i32(_seqlen_k_i32 - fx.Int32(1), fx.Int32(0))
        seq_last = fx.Index(_slast_i32)

        # ---- Address split: 64-bit uniform base + 32-bit divergent offset ----
        #
        # The full linear element index is
        #     ((batch * seq_len) + token) * nheads * BLOCK_DMODEL + head * BLOCK_DMODEL + d
        # i.e. it spans all of B, S, H and D, and overflows i32 at 2G elements
        # (2 GB at f16, which real shapes reach). Only the *intra-tile* part is
        # safely 32-bit: it is bounded by
        #     max(BLOCK_M, BLOCK_N) * nheads * BLOCK_DMODEL + BLOCK_DMODEL
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
            # Diagnostic arm only, and valid solely when seqlen_q == seqlen_k.
            # Dense-only: it derives the layout from the shape, which varlen
            # invalidates outright. The host rejects the combination.
            _stq = (fx.Index(max_seqlen_q) * STRIDE_TOKEN, STRIDE_TOKEN, HEAD_DIM)
            _stk = (fx.Index(max_seqlen_k) * STRIDE_TOKEN, STRIDE_TOKEN, HEAD_DIM)
            q_st = o_st = _stq
            k_st = v_st = _stk
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
        def _addr_pair(st, head, batch_index, row_off):
            # `row_off` is the varlen row offset, and it belongs in the
            # **64-bit base** rather than the 32-bit per-lane offset: on a
            # packed tensor it is a whole-batch quantity and overflows 32 bits
            # at realistic token counts (sdpa-varlen-plan.md section 5).
            s_batch, s_seq, s_head = st
            bh = batch_index * s_batch + head * s_head + row_off * s_seq

            def tbase(seq_start):
                """Uniform 64-bit element base for (batch, head, seq_start).

                `seq_start` is a position on whichever sequence axis this
                tensor is indexed by -- rows for Q/O, KV columns for K/V --
                since `_addr_pair` builds one of these per tensor.
                """
                return bh + seq_start * s_seq

            def toff(row_in_tile, col):
                """Divergent 64-bit element offset inside the tile.

                64-bit because `row_in_tile * s_seq` genuinely does not fit in
                32: nothing requires the caller's tensor to be compact, and a
                view keeps its source's strides -- slicing `(1, 64, 16384,
                512)` f16, a 1 GiB tensor, down to eight heads leaves
                `s_seq = 8388608`, and 256 rows of that is exactly 2**31.

                It is worth keeping separate from `tbase` for callers outside
                the KV loop, where it is loop-invariant and LICM pays the
                64-bit width once. Inside the loop it is `kv_off` that decides
                whether that stays true.
                """
                return row_in_tile * s_seq + col

            def kv_off(ts, row_in_tile, col):
                """`toff` for a KV row, with the out-of-range row folded in.

                Two forms of the same value, and the whole difference is
                whether `row_in_tile * s_seq` stays loop-invariant.

                Recomputed (KV_ADDR_HOIST off) clamps the row first, so `row`
                depends on `ts`, which moves every KV iteration: the 64-bit
                multiply is loop-carried and re-emitted per load per
                iteration. At BLOCK_DMODEL 192 that is 14 `v_mul_lo_u32` and
                21 `v_add_co_u32` in the loop body, against 3 and 11 for the
                pre-64-bit kernel.

                Hoisted selects between two whole offsets instead, so both
                arms are loop-invariant per-lane values and the one uniform
                term is factored out of the select: the loop pays two adds and
                the select, and the multiply leaves it entirely. What it costs
                is one more 64-bit value live per cooperative load, which is
                why this is a knob and not simply the better form -- see
                `_KV_ADDR_HOIST_HEAD_DIMS` in the tuning module for where each
                one wins.

                The hoisted out-of-range arm sends the lane to its own column
                in row `ts`, the tile's first row, rather than to the last row
                of the sequence: any in-bounds address will do, since the
                value is discarded, and this one shares `ts * s_seq` with the
                in-range arm. `col` and not the literal `0`, which is equally
                in bounds and needs no register of its own -- the 0 arm holds
                one value fewer live and still spills *more*, 272 bytes of
                scratch against 44 at BLOCK_DMODEL 192, for 0.863 against
                1.172 on the same baseline. Re-measure before changing it.

                Each form states the bounds predicate its own way, and that is
                deliberate rather than untidy. `row_in_tile < seqlen_k - ts`
                puts the whole uniform half on one side, so it is one compare
                against an SGPR instead of a divergent 64-bit add and compare
                -- but only the hoisted form is free to use it, because the
                recomputed one needs `seq_last - ts` for its clamp anyway and
                because keeping it verbatim is what makes a knob-off build
                bitwise identical to the kernel before this knob existed.
                """
                if const_expr(not KV_ADDR_HOIST):
                    in_range = (ts + row_in_tile) < seqlen_k_v
                    row = fx.Index(
                        ArithValue(in_range).select(row_in_tile, seq_last - ts)
                    )
                    return toff(row, col)
                # `ts < seqlen_k` always -- it is either start_k, which the
                # caller's branch tested, or seq_last -- so this cannot wrap.
                in_range = row_in_tile < (seqlen_k_v - ts)
                return fx.Index(
                    ArithValue(in_range).select(toff(row_in_tile, col), col)
                )

            return tbase, toff, kv_off

        # Q and O are indexed by the query head; K and V by the KV head they
        # share. At num_head_q == num_head_k these coincide.
        _q_batch_v = fx.Index(_q_batch)
        _k_batch_v = fx.Index(_k_batch)
        _q_row_off_v = fx.Index(_q_row_off)
        _k_row_off_v = fx.Index(_k_row_off)
        # Bias is (B, H, Sq, Sk): the last axis is the KV column, so unlike
        # Q/K/V/O its "row" stride is stride_b2 and the contiguous axis is the
        # one the KV tile walks. Indexed with the *same* (batch_index,
        # q_row_off) the varlen decode produced, so it inherits every layout
        # for free rather than needing its own -- sdpa-bias-plan.md 3.
        if const_expr(BIAS_TYPE):
            _b_ptr = _pointer_to_llvm_ptr(Bias)
            _b_base = (
                _q_batch_v * fx.Index(stride_b0)
                + head_q * fx.Index(stride_b1)
                + _q_row_off_v * fx.Index(stride_b2)
            )

        if const_expr(ENABLE_DROPOUT):
            # The offset scheme, and the *only* dropout-specific arithmetic in
            # this file -- everything else is `philox.py`.
            #
            #   stride = cdiv(Max_seqlen_k, RN)
            #   base   = philox_offset_base + off_zh * Max_seqlen_q * stride
            #   offset(m, n) = base + m * stride + n // RN
            #
            # `BLOCK_M` and `BLOCK_N` appear nowhere: `m` and `n` are global
            # element coordinates, so the mask does not move when the kernel is
            # re-tuned. That is the reproducibility contract of
            # sdpa-dropout-plan.md §3, and it is invisible in any test that
            # uses one tile size.
            #
            # The offset scheme itself is `Philox.grid_plane`/`grid_offset`,
            # shared with the debug mask kernel -- see the comment there. This
            # kernel supplies only which plane a workgroup is on.
            _off_zh = fx.Int32(_z_i32) * fx.Int32(num_head_q) + fx.Int32(head_q)
            _ph_base, _ph_stride = PHILOX.grid_plane(
                philox_offset_base, _off_zh, max_seqlen_q, max_seqlen_k
            )

        # Q and O are addressed once each, outside the KV loop, so they have no
        # use for the clamped form.
        q_tbase, q_toff, _ = _addr_pair(q_st, head_q, _q_batch_v, _q_row_off_v)
        k_tbase, k_toff, k_kv_off = _addr_pair(k_st, head_k, _k_batch_v, _k_row_off_v)
        v_tbase, v_toff, v_kv_off = _addr_pair(v_st, head_k, _k_batch_v, _k_row_off_v)
        o_tbase, o_toff, _ = _addr_pair(o_st, head_q, _q_batch_v, _q_row_off_v)

        def _kv_addr(tbase, toff, kv_off, start_k, row_in_tile, col):
            """(uniform base, divergent offset) for a KV row, clamped in bounds.

            At K_PREFETCH_DIST == 1 the loop runs one tile ahead, so the final
            iteration addresses a tile past the end of the sequence; the
            unguarded cooperative load also addresses rows past BLOCK_N. Clamp
            start_k first, then send any row still past the end to the last
            row of the sequence. The values are never consumed; the clamp
            exists only so the address stays inside the allocation.

            With both distances 0 there is no over-read: the interface pads
            seq_len to a multiple of BLOCK_M and BLOCK_N divides BLOCK_M, so
            start_k + BLOCK_N <= seq_len always. Skip the clamp there; it is
            pure VALU.
            """
            if const_expr(
                _NO_KV_CLAMP
                or (
                    K_PREFETCH_DIST == 0
                    and V_PREFETCH_DIST == 0
                    and not KV_NEEDS_GUARD
                )
            ):
                return tbase(start_k), toff(row_in_tile, col)
            ts = fx.Index(
                ArithValue(start_k < seqlen_k_v).select(start_k, seq_last)
            )
            return tbase(ts), kv_off(ts, row_in_tile, col)

        def k_addr(start_k, row_in_tile, col):
            return _kv_addr(k_tbase, k_toff, k_kv_off, start_k, row_in_tile, col)

        def v_addr(start_k, row_in_tile, col):
            return _kv_addr(v_tbase, v_toff, v_kv_off, start_k, row_in_tile, col)

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

        def _split_ptr(ptr, base64, off):
            """ptr + base64 (uniform) + off (divergent). Both 64-bit.

            `off` used to be truncated to i32 on the reasoning that a
            within-tile offset is small. It is not: it carries
            `row_in_tile * s_seq`, and a non-compact input's sequence stride is
            bounded by the tensor its view was taken from, not by the shape the
            kernel sees. Slicing eight heads out of a 1 GiB
            (1, 64, 16384, 512) f16 tensor gives `s_seq = 8388608`, and 256
            rows of that is exactly 2**31 -- so the truncation wrapped and the
            kernel read another allocation.

            `s_seq` is already `fx.Index`, so the product was always computed
            in 64 bits; only this cast threw the high half away.
            """
            p = buffer_ops.get_element_ptr(ptr, fx.Int64(base64), elem_type=elem_type)
            return buffer_ops.get_element_ptr(p, fx.Int64(off), elem_type=elem_type)

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

        def coop_load_k_global(start_k):
            """Issue this thread's K global loads; results stay in registers."""
            vecs = []
            for batch in range_constexpr(NUM_BATCHES_KV):
                row_offset = batch * ROWS_PER_BATCH_LOAD
                b64, o32 = k_addr(
                    start_k, load_row_in_batch + row_offset,
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

        def coop_load_store_k(start_k, buf_id=0):
            """Distance-0 K staging: load and store inside a single guard.

            The guard has to cover the *load*, not just the store. When
            ROWS_PER_BATCH_LOAD overshoots BLOCK_N some cooperative-load lanes
            have no row -- at BLOCK_DMODEL 32 with BLOCK_N 64 that is exactly half
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
                            start_k, lds_row, _col_safe(load_col_base, _hdim_qk_i)
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
                        start_k, lds_row, _col_safe(load_col_base, _hdim_qk_i)
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

        def coop_load_v_global(start_k, chunk=0):
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
                        start_k, kv_base + _tr_kv_off,
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
                        start_k, v_row_in_batch + row_offset,
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
            start_q + wave_q_offset + fx.Index(qt * WMMA_M) + lane16
            for qt in range_constexpr(Q_ROW_TILES)
        ]
        q_row_i32s = [fx.Int32(r) for r in q_rows]
        # Intra-tile Q rows, bounded by BLOCK_M so the 32-bit offset stays small.
        q_rows_in_tile = [
            wave_q_offset + fx.Index(qt * WMMA_M) + lane16
            for qt in range_constexpr(Q_ROW_TILES)
        ]
        q_tile_base = q_tbase(_q_start_addr)
        c_zero_v8f16 = Vec.filled(8, 0.0, elem_dtype).ir_value()

        q_in_bounds_all = []
        q_b_packs_all = []
        for qt in range_constexpr(Q_ROW_TILES):
            # Explicit signed-less-than predicate: fx.Index defaults to unsigned,
            # which lowers to v_cmp_gt_u64_e64 instead of the signed form.
            _in = arith.cmpi(
                arith.CmpIPredicate.slt, _raw(q_rows[qt]), _raw(seqlen_q_v)
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
        # Finite floor, so an all-masked row cannot produce -inf - -inf.
        c_m_init = fx.Float32(-3.40282e38)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_sm_scale_log2e = sm_log2e
        c_zero_v8f32 = Vec.filled(8, 0.0, fx.Float32)
        width_i32 = fx.Int32(WARP_SIZE)
        shuf_16_i32 = fx.Int32(16)

        def reduction_peer(v_f32):
            return fx.Float32(v_f32).shuffle_xor(shuf_16_i32, width_i32)

        # Right edge of the visible band: key j is visible to query i only if
        # j <= i + _diag. This *is* `window_right` -- 0 for top-left,
        # seqlen_k - seqlen_q for bottom-right, and an explicit argument under
        # gSWA. It is signed: with seqlen_q > seqlen_k it goes negative, and
        # whole leading Q rows then see no keys at all.
        #
        # Everything derived from a window stays fx.Int32 with explicit signed
        # predicates, per sdpa-gswa-plan.md section 2.4. fx.Index is unsigned,
        # so a negative window reaching it silently becomes enormous.
        if const_expr(CAUSAL):
            # ---- parse_window: resolve the causal sentinels per sequence ----
            #
            # `Window_left`/`Window_right` may carry 0x80000001 (top-left) or
            # 0x80000002 (bottom-right) instead of a literal bound, and the
            # kernel resolves them against *this sequence's* lengths.
            #
            # That is the whole reason the sentinels exist. Resolving on the
            # host works only when there is one length to resolve against:
            # under varlen, bottom-right needs `seqlen_k[z] - seqlen_q[z]`,
            # and the batch-wide `Max_seqlen_k - Max_seqlen_q` is a different
            # number for every sequence whose difference is not the maximum.
            #
            # Both sentinels give an unbounded left edge -- no row reaches
            # further back than the start of its own sequence -- so they
            # differ only in the right one. Matches AOTriton's `parse_window`.
            _wl_i32 = fx.Int32(window_left)
            _wr_i32 = fx.Int32(window_right)
            _is_tl_l = _scmp_i32(
                arith.CmpIPredicate.eq, _wl_i32, fx.Int32(_WINDOW_TOPLEFT)
            )
            _is_br_l = _scmp_i32(
                arith.CmpIPredicate.eq, _wl_i32, fx.Int32(_WINDOW_BOTRIGHT)
            )
            _wl_i32 = _ssel_i32(
                ArithValue(arith.ori(_raw(_is_tl_l), _raw(_is_br_l))),
                _seqlen_q_i32, _wl_i32,
            )
            _wr_i32 = _ssel_i32(
                _scmp_i32(arith.CmpIPredicate.eq, _wr_i32,
                          fx.Int32(_WINDOW_TOPLEFT)),
                fx.Int32(0), _wr_i32,
            )
            _wr_i32 = _ssel_i32(
                _scmp_i32(arith.CmpIPredicate.eq, fx.Int32(window_right),
                          fx.Int32(_WINDOW_BOTRIGHT)),
                _seqlen_k_i32 - _seqlen_q_i32, _wr_i32,
            )

        # ---- Split the KV range into full and masked regions ----
        #
        # Emitting the regions as separate loops means the masks exist only in
        # the masked one -- `MASK_STEPS` in AOTriton's terms, with the split
        # structural rather than a per-tile branch.
        #
        # **How many regions there are depends on the arm.** Non-causal has
        # two, `[full][tail-masked]`, because the only thing that can cut a
        # tile is running past seqlen_k. Causal has *three*: a left window
        # kills columns at the start of the range as well, so masked tiles are
        # a prefix as well as a suffix and tile 0 is not automatically live.
        # A negative `window_left` is the sharpest case -- it pushes the whole
        # band right of the diagonal, so the leading masked run can span
        # several tiles rather than clipping one.
        #
        # Do not carry the two-region intuition into the causal branch; the
        # "every earlier tile is fully live" shortcut is true only above.
        #
        # This is what pays back the unconditional mask P1e had to use: a
        # dynamic `scf.if` guard inside one loop measured much worse than no
        # guard at all, because the region boundary blocks scheduling in a
        # latency-bound loop. A static split has no such boundary.
        if const_expr(CAUSAL):
            # ---- gSWA: three regions over one contiguous block range ----
            #
            # A left window can kill columns in any tile, including tile 0, so
            # the masked tiles are a prefix as well as a suffix:
            #
            #     [ left-masked ][ full ][ right-masked ]
            #
            # The three are *contiguous and non-overlapping by construction*
            # here, because they are derived by cutting one visited range
            # rather than intersected as three independent intervals. That
            # collapses two of the three special cases in sdpa-gswa-plan.md
            # section 2.2: a window narrower than a block leaves the full
            # region empty, which is detected once and turns the other two
            # into a single masked run, and an irregular seqlen_q needs no
            # special handling because `_q_hi` already bounds the rows.
            #
            # Bounds for the rows this Q block actually owns, [start_q, q_hi):
            #   a column c is live for row i iff  i - w_left <= c <= i + w_right
            # so over the whole block the live columns span
            #   [ start_q - w_left, (q_hi - 1) + w_right ]
            # and a tile is *fully* live iff every one of its columns is live
            # for every row -- worst case the largest row on the left and the
            # smallest on the right.
            _q_start_i32 = fx.Int32(start_q)
            _q_hi_i32 = _smin_i32(
                _q_start_i32 + fx.Int32(BLOCK_M), _seqlen_q_i32
            )
            _q_last_i32 = _q_hi_i32 - fx.Int32(1)

            # Blocks that exist at all, and the last block that is *whole*.
            # Splitting these two is section 2.2 case 3: a ragged seqlen_k
            # leaves a partial final tile, which must be masked rather than
            # counted as full. The old code spelled the same thing
            # `_full_seq`.
            _blk_last = _sdiv_rd(_seqlen_k_i32 - fx.Int32(1))
            _blk_last_whole = _sdiv_rd(_seqlen_k_i32) - fx.Int32(1)

            # The visited range: outside it every column is dead for every row
            # in this Q block, so those tiles are not walked at all.
            _v_lo = _smax_i32(_sdiv_rd(_q_start_i32 - _wl_i32), fx.Int32(0))
            _v_hi = _smin_i32(_blk_last, _sdiv_rd(_q_last_i32 + _wr_i32))
            # Empty work, not skipped work. Varlen sizes the grid from
            # Max_seqlen_q, so a short sequence gets workgroups whose rows are
            # all past its end; this kernel is one single-exit trace and
            # cannot `return` out of them (plan section 6.1). Inverting the
            # visited range drives every region count to zero, and the
            # existing row-bound guards already suppress both stores.
            _v_hi = _ssel_i32(_alive, _v_hi, _v_lo - fx.Int32(1))

            # First tile clear of the left edge: ceil(((q_hi-1) - w_left)/BN).
            # First tile touched by the right edge: floor((start_q+w_right+1)/BN).
            #
            # Both are *exact*, not the conservative-by-one form the plan
            # sketched: when the boundary column falls exactly on a tile edge
            # that tile is still wholly live, and rounding it into the masked
            # run would send a tile through the other loop body. That is
            # invisible to a tolerance test but not to the bitwise one -- and
            # the pre-gSWA causal split is exact here too, so anything else
            # would stop reproducing it.
            _l_first_full = _sdiv_rd(
                _q_last_i32 - _wl_i32 + fx.Int32(BLOCK_N - 1)
            )
            _r_first_mask = _sdiv_rd(_q_start_i32 + _wr_i32 + fx.Int32(1))

            _fb_lo = _smax_i32(_l_first_full, _v_lo)
            _fb_hi = _smin_i32(
                _smin_i32(_r_first_mask - fx.Int32(1), _blk_last_whole), _v_hi
            )
            _fb_empty = _scmp_i32(arith.CmpIPredicate.sgt, _fb_lo, _fb_hi)

            # Cut [_v_lo, _v_hi] at the full region. With no full region the
            # whole range becomes one masked run, which is section 2.2 case 2
            # (the window is narrower than a block) falling out for free.
            _lb_hi = _ssel_i32(_fb_empty, _v_hi, _fb_lo - fx.Int32(1))
            _rb_lo = _ssel_i32(_fb_empty, _v_hi + fx.Int32(1), _fb_hi + fx.Int32(1))
            _n_l = _smax_i32(_lb_hi - _v_lo + fx.Int32(1), fx.Int32(0))
            _n_f = _smax_i32(_fb_hi - _fb_lo + fx.Int32(1), fx.Int32(0))
            _n_r = _smax_i32(_v_hi - _rb_lo + fx.Int32(1), fx.Int32(0))

            _BN_I32 = fx.Int32(BLOCK_N)
            _l_col0 = _v_lo * _BN_I32
            _r_col0 = _rb_lo * _BN_I32
            _f_col0 = _fb_lo * _BN_I32
            # First tile of the masked run, which is also what the full loop's
            # last prefetch must fetch: the two loops are adjacent only when
            # the left run is empty.
            # Clamped: with a window that admits no key at all every run is
            # empty and `_rb_lo` sits below zero, and this value still reaches
            # the prologue's address computation.
            _m_col0 = _smax_i32(
                _ssel_i32(
                    _scmp_i32(arith.CmpIPredicate.sgt, _n_l, fx.Int32(0)),
                    _l_col0, _r_col0,
                ),
                fx.Int32(0),
            )
            _n_masked = fx.Index(_n_l + _n_r)
        else:
            # No mask beyond the KV tail: one full region, then one partial
            # tile. `kv_upper` is rounded *up* to a whole BLOCK_N because the
            # loop steps by BLOCK_N, and a bound that is not a multiple of it
            # drops the final partial tile entirely -- at seqlen 40 with
            # BLOCK_N 32 the kernel attended to only the first 32 keys, which
            # is wrong for every row rather than just the tail.
            _full_end = (seqlen_k_v // fx.Index(BLOCK_N)) * fx.Index(BLOCK_N)
            kv_upper = fx.Index(
                ((seqlen_k_v + fx.Index(BLOCK_N - 1)) // fx.Index(BLOCK_N))
                * fx.Index(BLOCK_N)
            )
            # Same empty-work clamp as the causal arm above.
            _full_end = fx.Index(
                ArithValue(_alive).select(_full_end, fx.Index(0))
            )
            kv_upper = fx.Index(
                ArithValue(_alive).select(kv_upper, fx.Index(0))
            )

        # Tiles this workgroup will actually walk. Zero for a sequence with no
        # keys and for every workgroup the varlen grid dispatches past the end
        # of a short one.
        if const_expr(CAUSAL):
            _kv_tiles_i32 = _n_f + _n_l + _n_r
        else:
            _kv_tiles_i32 = fx.Int32(kv_upper)

        # ---- Prologue: at distance 1, the first tile's K / V go to registers ----
        #
        # "First" is tile 0 for everything except gSWA, where a left window
        # can make the first *visited* tile any block -- loading tile 0 there
        # feeds the first iteration the wrong K and V. That was worth a
        # relative error of 1.3 on a shape whose interval arithmetic was
        # already correct, which is the failure mode plan section 5 predicted:
        # a prefetch bug survives every check that only looks at the loop
        # bounds.
        if const_expr(CAUSAL):
            _first_col = fx.Index(
                _smax_i32(
                    _ssel_i32(
                        _scmp_i32(arith.CmpIPredicate.sgt, _n_f, fx.Int32(0)),
                        _f_col0, _m_col0,
                    ),
                    fx.Int32(0),
                )
            )
        else:
            _first_col = fx.Index(0)
        # **Issued only if a KV tile will actually be walked.**
        #
        # The prefetch runs before any loop bound is consulted, so a workgroup
        # with nothing to do -- a sequence with no keys, or one of the dead
        # workgroups varlen's Max_seqlen_q-sized grid dispatches past the end
        # of a short sequence -- would otherwise still issue a K and a V tile
        # load and throw the result away.
        #
        # Clamping the address instead would make that load land somewhere
        # harmless, which fixes the symptom: the load should not happen. And
        # at seqlen_k == 0 there is no harmless address to clamp *to* -- row 0
        # of an empty sequence is one past its end.
        #
        # A 0-or-1-trip `range(..., init=...)` is how this kernel already
        # expresses predicated state: FlyDSL's dynamic `if` merges named
        # scalars only and rejects the list of vectors a prefetch produces,
        # while a loop carries exactly that list. The trip count is uniform
        # across the workgroup, so no divergence is introduced.
        _pf_n = fx.Index(
            _ssel_i32(
                _scmp_i32(arith.CmpIPredicate.sgt, _kv_tiles_i32, fx.Int32(0)),
                fx.Int32(1), fx.Int32(0),
            )
        )
        _pf_init = []
        if const_expr(K_PREFETCH_DIST):
            for _ in range_constexpr(NUM_BATCHES_KV):
                _pf_init.append(Vec.filled(VEC_WIDTH, 0.0, elem_dtype).ir_value())
        if const_expr(V_PREFETCH_DIST):
            for _ in range_constexpr(V_LOADS):
                _pf_init.append(Vec.filled(VEC_WIDTH, 0.0, elem_dtype).ir_value())
        _pf = _pf_init
        if const_expr(K_PREFETCH_DIST or V_PREFETCH_DIST):
            for _pfi, _pf_args in range(fx.Index(0), _pf_n, 1, init=_pf_init):
                _y = []
                if const_expr(K_PREFETCH_DIST):
                    _y.extend(coop_load_k_global(_first_col))
                if const_expr(V_PREFETCH_DIST):
                    _y.extend(coop_load_v_global(_first_col))
                _pf = yield _y
            # `scf_yield_` returns a bare value, not a list, when the loop
            # carries exactly one -- which happens at BLOCK_DMODEL 16, where the K
            # prefetch is off and V is a single load. Indexing that bare value
            # extracts a vector *element*, so the next use sees an f16 scalar
            # where it wants a vector, and the failure surfaces far away in
            # the LDS store.
            if const_expr(len(_pf_init) == 1):
                _pf = [_pf]
        if const_expr(K_PREFETCH_DIST):
            _k_vecs_init = [_pf[_i] for _i in range_constexpr(NUM_BATCHES_KV)]
        if const_expr(V_PREFETCH_DIST):
            _off = NUM_BATCHES_KV if K_PREFETCH_DIST else 0
            _v_vecs_init = [_pf[_off + _i] for _i in range_constexpr(V_LOADS)]

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
        def _kv_body(kv_block_start, inner_iter_args, _MASK_STEPS,
                     next_kv_start=None):
            """One KV tile. `_MASK_STEPS` is a Python bool resolved at trace
            time, so the masked and unmasked regions emit different code.

            `next_kv_start` is the tile the distance-1 prefetch should fetch.
            It defaults to the following tile, which is right whenever the
            region being walked is contiguous. gSWA's masked loop walks two
            disjoint runs, so it passes the piecewise successor explicitly --
            getting that wrong fetches the wrong tile and is invisible to a
            correctness test whenever the value is overwritten before use."""
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

            if const_expr(next_kv_start is None):
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
            # Each shard-wave holds a partial sum over its own BLOCK_DMODEL slice;
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

                # ---- KV tail mask: columns >= seqlen_k are not real keys ----
                #
                # seqlen is never padded, so the final KV tile of a ragged
                # sequence covers columns past the end. `kv_addr` clamps those
                # *addresses* so the loads stay in bounds, which means they read
                # a duplicate real row -- safe, but the scores are garbage and
                # would enter the softmax. Mask them to -inf here.
                #
                # Done on the eight-wide accumulators rather than the unpacked
                # scalars: NUM_S_ACCS is at most 8 (BLOCK_N 128), so the
                # branch's live set stays small, and one vector select replaces
                # eight scalar ones.
                #
                # Guarded, so only the tail tile pays. Interior tiles cost a
                # single scalar compare. (This is `MASK_STEPS` with a dynamic
                # guard instead of a structural one; P2's interval
                # decomposition replaces it with the structural form.)
                s_raw = []
                for st in range_constexpr(NUM_S_ACCS):
                    for r in range_constexpr(8):
                        s_raw.append(Vec(s_accs[st])[r])

                # Scale before the row max, so m_i lives in the scaled
                # domain and the exponent is a plain subtract.
                s_raw = [_fmul(v, c_sm_scale_log2e) for v in s_raw]

                if const_expr(BIAS_TYPE):
                    # ---- Bias, after the scale and before the mask ----
                    #
                    # After the scale because m_i and the exponent live in the
                    # base-2 scaled domain, so a bias in natural units has to
                    # be multiplied by log2(e) first -- which is what
                    # AOTriton's `qk += bias * 1.44269504089` is doing.
                    #
                    # Before the mask so a column past seqlen_k stays -inf
                    # rather than becoming -inf + bias. Those columns are not
                    # keys the caller hid; they do not exist, and neither do
                    # their bias entries.
                    #
                    # Not a gather. Element i of the flattened accumulators is
                    # KV column (i//16)*32 + ((i//8)%2)*16 + klane*8 + i%8, so
                    # within each group of eight only i%8 varies: those eight
                    # are eight *contiguous* columns starting at klane*8, and
                    # one v8 load covers them -- the same shape as the K and V
                    # loads.
                    _b_row = _b_base + q_rows[qt] * fx.Index(stride_b2)
                    for _st in range_constexpr(NUM_S_ACCS):
                        _c0 = (_st // 2) * 32 + (_st % 2) * 16
                        _bv = load_global_v8f16(
                            _b_ptr,
                            _b_row + fx.Index(kv_block_start) + fx.Index(_c0)
                            + klane * fx.Index(8),
                            fx.Index(0),
                        )
                        for _r in range_constexpr(8):
                            _bs = fx.Float32(Vec(_bv)[_r].to(fx.Float32))
                            s_raw[_st * 8 + _r] = _fadd(
                                s_raw[_st * 8 + _r],
                                _fmul(_bs, fx.Float32(_LOG2E)),
                            )

                if const_expr(_MASK_STEPS):
                    # ---- Masked region: KV tail and causal, fused ----
                    #
                    # Only tiles in the masked region reach this. Full tiles
                    # are emitted by the other region with no mask at all,
                    # which is the entire point of the split: masking used to
                    # be paid on every tile.
                    #
                    # Element i of the flattened accumulators is KV column
                    #   (i//16)*32 + ((i//8)%2)*16 + klane*8 + i%8
                    # -- the GEMM1 unroll walks (sub-tile, half) pairs, and
                    # within a 16-row WMMA block a lane holds rows
                    # klane*8 + si.
                    #
                    # Two conditions, one select: a column is dead if it is
                    # past seqlen_k, or (causal) beyond this row's diagonal.
                    #
                    # **Must run after the sm_scale multiply.** The kernel is
                    # built with nnan but *not* ninf precisely so an -inf can
                    # survive arithmetic; even so, keeping the mask after the
                    # scale avoids relying on that. Masking before it silently
                    # did nothing when ninf was still enabled.
                    #
                    # No `scf.if` guard: a runtime "does this tile need it"
                    # branch measured far worse than just doing the selects
                    # (BLOCK_DMODEL 192 non-causal 78.0 against 98.3 TFLOPS), the
                    # region boundary blocking scheduling in a latency-bound
                    # loop. The region split gives the same benefit statically.
                    _kv_i32 = fx.Int32(kv_block_start)
                    _klane_off = fx.Int32(klane) * fx.Int32(8)
                    _seq_i32 = fx.Int32(seqlen_k_v)
                    for _i in range_constexpr(NUM_S_VALS):
                        _col = (
                            _kv_i32
                            + fx.Int32((_i // 16) * 32 + ((_i // 8) % 2) * 16 + _i % 8)
                            + _klane_off
                        )
                        _dead = _col >= _seq_i32
                        if const_expr(CAUSAL):
                            # Both edges of the band. Signed throughout:
                            # q_row - w_left is negative for every row when
                            # w_left is "unbounded" (== seqlen_q), which is how
                            # plain causal maps onto this path with the left
                            # term inert.
                            _dead = _dead | ArithValue(
                                arith.cmpi(
                                    arith.CmpIPredicate.sgt,
                                    _raw(_col),
                                    _raw(q_row_i32 + _wr_i32),
                                )
                            )
                            _dead = _dead | ArithValue(
                                arith.cmpi(
                                    arith.CmpIPredicate.slt,
                                    _raw(_col),
                                    _raw(q_row_i32 - _wl_i32),
                                )
                            )
                        s_raw[_i] = ArithValue(_dead).select(c_neg_inf, s_raw[_i])

                local_max = s_raw[0]
                for r in range_constexpr(NUM_S_VALS - 1):
                    local_max = _fmax(local_max, s_raw[r + 1])
                peer_max = reduction_peer(local_max)
                row_max = _fmax(local_max, peer_max)
                m_new_raw = _fmax(m_running, row_max)

                # m is already scaled, so no scale appears in either exponent.
                corr = rocdl.exp2(
                    ir.F32Type.get(), _raw(_fsub(m_running, m_new_raw))
                )
                neg_m = _fsub(c_zero_f, m_new_raw)

                p_vals = []
                local_sum = _raw(c_zero_f)
                for r in range_constexpr(NUM_S_VALS):
                    diff = _fadd(s_raw[r], neg_m)
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

                if const_expr(ENABLE_DROPOUT):
                    # **After `l_new`, before the O accumulation.** The softmax
                    # denominator must be the *undropped* sum, or the result
                    # stops being an expectation of the undropped attention and
                    # the logsumexp the backward pass reads is wrong. Reversing
                    # these two lines produces plausible output that is wrong by
                    # a per-row factor, and no shape check notices
                    # (sdpa-dropout-plan.md §6).
                    #
                    # A group of eight consecutive elements is eight contiguous
                    # KV columns (§2 of the bias plan has the same identity), so
                    # each group is one span of the stream.
                    for _st in range_constexpr(NUM_S_ACCS):
                        _c0 = (_st // 2) * 32 + (_st % 2) * 16
                        _bcol = (
                            fx.Int64(kv_block_start)
                            + fx.Int64(_c0)
                            + fx.Int64(fx.Int32(klane) * fx.Int32(8))
                        )
                        _first = PHILOX.grid_offset(
                            _ph_base, _ph_stride, q_rows[qt], _bcol
                        )
                        _keep = PHILOX.keep_span(
                            philox_seed, _first, 8, idropout_p
                        )
                        for _r in range_constexpr(8):
                            _i = _st * 8 + _r
                            p_vals[_i] = _raw(
                                _keep[_r].select(
                                    fx.Float32(p_vals[_i]), fx.Float32(0.0)
                                )
                            )

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
                # waves/SIMD, measured at -4% on BLOCK_DMODEL 128.
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
            return _yield_args

        loop_results = init_args
        if const_expr(CAUSAL):
            # Still **two** emitted bodies, not three. The body already costs
            # 63 VGPRs at BLOCK_DMODEL 128 and spills BLOCK_DMODEL 192 at two copies
            # (plan1 sections 6.2, 2.6), so the two masked runs share one loop
            # walked over a piecewise index -- one select per masked iteration,
            # paid only in the masked region.
            #
            # Order is full, then left-masked, then right-masked. Running the
            # full region *first* is what made a causal-equivalent window
            # bit-identical to the dedicated causal path that used to live
            # here: with an unbounded left window the left run is empty and
            # the order collapses to full-then-tail, exactly the pre-gSWA
            # split. Online softmax is order-independent mathematically but
            # not in floating point, so walking the masked runs first would
            # have been just as correct and would have lost that property --
            # which is the property that licensed deleting CAUSAL_TYPE 1/2.
            for kv_block_start, inner_iter_args in range(
                fx.Index(_f_col0), fx.Index(_f_col0 + _n_f * _BN_I32),
                BLOCK_N_OUT, init=init_args,
            ):
                # The successor of the last full tile is the first masked one,
                # which is only the adjacent tile when the left run is empty.
                _nxt = fx.Index(
                    _ssel_i32(
                        _scmp_i32(
                            arith.CmpIPredicate.slt,
                            fx.Int32(kv_block_start) + _BN_I32,
                            _f_col0 + _n_f * _BN_I32,
                        ),
                        fx.Int32(kv_block_start) + _BN_I32,
                        _m_col0,
                    )
                )
                loop_results = yield _kv_body(
                    kv_block_start, inner_iter_args, False, next_kv_start=_nxt
                )

            def _masked_col(i_idx):
                """Tile column for masked iteration i: the left run, then the
                right one. Discontinuous at the seam, which is exactly why the
                prefetch has to go through this same map."""
                _i = fx.Int32(i_idx)
                return _ssel_i32(
                    _scmp_i32(arith.CmpIPredicate.slt, _i, _n_l),
                    _l_col0 + _i * _BN_I32,
                    _r_col0 + (_i - _n_l) * _BN_I32,
                )

            for _mi, inner_iter_args in range(
                fx.Index(0), _n_masked, 1, init=loop_results
            ):
                loop_results = yield _kv_body(
                    fx.Index(_masked_col(_mi)),
                    inner_iter_args,
                    True,
                    next_kv_start=fx.Index(_masked_col(fx.Int32(_mi) + fx.Int32(1))),
                )
        else:
            # Region 1: tiles that are wholly live -- no masking emitted at all.
            for kv_block_start, inner_iter_args in range(
                fx.Index(0), _full_end, BLOCK_N_OUT, init=init_args
            ):
                loop_results = yield _kv_body(kv_block_start, inner_iter_args, False)

            # Region 2: the tail, where columns can be past seqlen_k or past the
            # causal diagonal.
            for kv_block_start, inner_iter_args in range(
                _full_end, kv_upper, BLOCK_N_OUT, init=loop_results
            ):
                loop_results = yield _kv_body(kv_block_start, inner_iter_args, True)

        # ---- logsumexp ----
        # LSE = (m + log2(l)) * ln2, with m in the base-2 scaled domain -- which
        # is exactly the convention the scaled-m softmax establishes, so nothing extra
        # is needed there. `rocdl.log` is v_log_f32, i.e. base 2.
        #
        # One value per (batch, head, q_row). A lane's m/l belong to its own
        # q_row (lane16), replicated across the klane halves by the shuffle_xor
        # reduction and across SHARDS by the cross-shard reduction, so exactly
        # one lane per row must store: klane 0 of shard 0.
        #
        # Layout is AOTriton's single branch-free formula
        #   offset = (b * H + h) * S + s
        # with (b=batch, s=0, S=Max_seqlen_q) giving (B*H, S). Varlen will pass
        # (b=0, s=cu_seqlens_q_start, S=total) for (H, TotalS) without changing
        # anything here -- that is the point of computing the base on the host.
        # Conditions are combined with arith.andi, not Python `and`/`not`:
        # those call __bool__ on the MLIR value and are resolved at trace time,
        # which silently folded this whole block away on the first attempt.
        _f32_ty = ir.F32Type.get()
        _l_valid = arith.cmpi(
            arith.CmpIPredicate.ne, _raw(fx.Int64(fx.ptrtoint(L))), _raw(fx.Int64(0))
        )
        _lse_writer = arith.andi(
            _l_valid,
            arith.andi(
                _raw(klane == fx.Index(0)), _raw(shard_id == fx.Index(0))
            ),
        )
        # Everything -- the log2, the scale, the address -- lives inside the
        # guard. Hoisting it out cost 8% at BLOCK_DMODEL 256 non-causal even though
        # the store itself was still predicated: the values stay live across
        # the epilogue and lengthen it for every wave, including the ones that
        # never store and the whole kernel when L is null.
        for qt in range_constexpr(Q_ROW_TILES):
            _do_store = arith.andi(_lse_writer, _raw(q_in_bounds_all[qt]))
            if _do_store:
                _m = loop_results[2 * qt]
                _l = loop_results[2 * qt + 1]
                _lse = _fmul(
                    _fadd(_m, rocdl.log(_f32_ty, _raw(_l))), fx.Float32(_LN2)
                )
                # A row with no live keys gets +inf, not -inf: the backward
                # pass subtracts LSE from qk, so +inf makes exp(qk - inf)
                # zero for exactly the rows that must contribute nothing.
                # l is bit-exact 0 there, so test the bit pattern; integer
                # compares lower predictably.
                _lse = ArithValue(
                    arith.cmpi(
                        arith.CmpIPredicate.ne,
                        _raw(_bitcast_i32(fx.Float32(_l))),
                        _raw(fx.Int32(0)),
                    )
                ).select(fx.Float32(_lse), fx.Float32(float("inf")))
                # LSE_LAYOUT, VarlenBits bits 17:16. The *inputs* are Q's
                # decode -- batch, row offset, length -- so the two layouts
                # are the same indices arranged two ways, not two features
                # (sdpa-varlen-plan.md section 3.2).
                #
                #   _HT  (H, T)   AOTriton's, and the default: T contiguous
                #   _TH  (T, H)   Transformer Engine's:         H contiguous
                _row = _q_row_off_v + q_rows[qt]
                _tok = fx.Index(_lse_tokens)
                _nhq = fx.Index(num_head_q)
                _lse_off_ht = (_q_batch_v * _nhq + head_q) * _tok + _row
                # Compact in both layouts, so the head pitch is exactly
                # num_head_q and the token pitch is the decode's `tokens`.
                _lse_off_th = (_q_batch_v * _tok + _row) * _nhq + head_q
                _is_th = ArithValue(
                    arith.cmpi(
                        arith.CmpIPredicate.ne,
                        _raw((fx.Int32(varlen_bits) >> fx.Int32(16)) & fx.Int32(3)),
                        _raw(fx.Int32(0)),
                    )
                )
                _lse_off = fx.Index(
                    _is_th.select(fx.Index(_lse_off_th), fx.Index(_lse_off_ht))
                )
                _pointer_store(
                    _lse,
                    buffer_ops.get_element_ptr(
                        _pointer_to_llvm_ptr(L),
                        fx.Int64(_lse_off),
                        elem_type=_f32_ty,
                    ),
                )

        # ---- Normalize and store O ----
        for qt in range_constexpr(Q_ROW_TILES):
            l_final = loop_results[2 * qt + 1]
            # A row can legitimately see *no* keys: bottom-right causal with
            # seqlen_q > seqlen_k leaves the leading seqlen_q - seqlen_k rows
            # fully masked, and bias or a sliding window will do the same.
            # Then l is exactly 0, 1/l is +inf, and o_acc * inf is NaN even
            # though o_acc is exactly 0.
            #
            # Clamp rather than branch: for a live row l >= 1 always, since
            # the running max contributes exp2(0) = 1, so this is a no-op
            # there.
            _l_safe = _fmax(l_final, fx.Float32(1e-30))
            inv_l = arith.divf(_raw(c_one_f), _raw(_l_safe), fastmath=fm_fast)
            if const_expr(ENABLE_DROPOUT):
                # `1/(1-p)` folds into the existing `1/l` rather than becoming
                # a per-element multiply: it is uniform across the tile, and
                # the dropped entries are already zero, so scaling the whole
                # accumulator is equivalent and costs one scalar.
                #
                # `l` is deliberately *not* scaled -- it is the undropped sum,
                # and the logsumexp written below is the undropped one, which
                # is what the backward pass needs.
                inv_l = _raw(_fmul(fx.Float32(inv_l), dropout_scale))
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
                            o_tbase(start_q),
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
        L: fx.Pointer,
        Bias: fx.Pointer,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        batch_size: fx.Int32,
        varlen_bits: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        philox_seed: fx.Int64,
        philox_offset_base: fx.Int64,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
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
        stride_b0: fx.Int64,
        stride_b1: fx.Int64,
        stride_b2: fx.Int64,
        sm_scale_v: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()

        bs_idx = fx.Index(batch_size)
        # Grid Q extent keys on Max_seqlen_q: under varlen there is no single
        # seqlen_q, so every sequence gets the longest one's worth of
        # workgroups and the short ones exit empty (plan section 6).
        sl_idx = fx.Index(max_seqlen_q)
        num_q_tiles = (sl_idx + BLOCK_M - 1) // BLOCK_M

        # Strides come from the caller, read off the real tensors -- never
        # derived here from num_heads/BLOCK_DMODEL. The shape does not determine
        # the layout (plan1 section 0), and K/V need not share Q's layout at
        # all. Axis 3 is D, contiguous by contract, so it is not passed.
        # Always forwarded: with STRIDES_CONSTEXPR the kernel simply does not
        # read them, which is what keeps the two arms directly comparable.
        launcher = flash_attn_func_aiw_kernel(
            Q, K, V, O, L, Bias,
            seqinfo_q0, seqinfo_q1, seqinfo_k0, seqinfo_k1,
            varlen_bits, batch_size, max_seqlen_q, max_seqlen_k,
            window_left, window_right,
            philox_seed, philox_offset_base, idropout_p, dropout_scale,
            num_head_q, num_head_k,
            hdim_qk, hdim_vo,
            stride_q0, stride_q1, stride_q2,
            stride_k0, stride_k1, stride_k2,
            stride_v0, stride_v1, stride_v2,
            stride_o0, stride_o1, stride_o2,
            stride_b0, stride_b1, stride_b2,
            sm_scale_v,
        )

        if const_expr(WAVES_PER_EU is not None):
            _wpe = int(WAVES_PER_EU)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.WAVES_PER_EU"] = ir.IntegerAttr.get(
                            T.i32, _wpe
                        )
        if const_expr(FLAT_WORK_GROUP_SIZE is not None):
            _fwgs = int(FLAT_WORK_GROUP_SIZE)
            if const_expr(_fwgs >= 1):
                flat_wg_attr = ir.StringAttr.get(f"{_fwgs},{_fwgs}")
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.FLAT_WORK_GROUP_SIZE"] = flat_wg_attr

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
        if const_expr(DENORMALS_ARE_ZERO):
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("denormal-fp-math-f32"),
                        ir.StringAttr.get("preserve-sign,preserve-sign"),
                    ]
                )
            )
            if const_expr(_FP_MODE == "fast"):
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
        "FAST_FP_MATH": FAST_FP_MATH,
        "UNSAFE_FP_MATH": UNSAFE_FP_MATH,
        "llvm_options": {"enable-post-misched": False, "lsr-drop-solution": True},
    }

    _NULL_PTR = flyc.from_c_void_p(fx.Uint8, 0)

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

    def _lse_args(lse, seq_len, varlen, num_head_q):
        """The logsumexp pointer, and a check that its layout matches the bits.

        Returns only a pointer: unlike Q/K/V/O this tensor is always compact,
        so the kernel derives both pitches from `LSE_LAYOUT`, `num_head_q` and
        the token count (sdpa-varlen-plan.md section 4.2). Inferring strides
        here as well would give one fact two sources.

        What the host can do instead -- and could not while it was inferring --
        is verify the caller's tensor actually has the declared layout.
        """
        if lse is None:
            return _NULL_PTR
        if lse.dtype != torch_f32:
            raise ValueError(f"logsumexp must be float32, got {lse.dtype}")
        if lse.dim() != 2:
            raise ValueError(f"logsumexp must be rank 2, got shape {tuple(lse.shape)}")
        if not lse.is_contiguous():
            raise ValueError(
                "logsumexp must be contiguous: the kernel derives its pitches "
                "from VarlenBits rather than reading strides"
            )
        _layout = 0 if varlen is None else (int(varlen["bits"]) >> 16) & 3
        if _layout == 0:
            # The token pitch the kernel will derive. Only checkable when the
            # caller supplies it: under a stacked layout it lives in
            # `seqinfo[N]` on the device, and reading that back would cost a
            # sync for a validation.
            want_last = int(seq_len) if varlen is None else varlen.get("lse_tokens")
            if want_last is not None and lse.shape[1] != int(want_last):
                raise ValueError(
                    f"VARLEN_LSE_LAYOUT_HT wants (*, {int(want_last)}), got "
                    f"{tuple(lse.shape)}"
                )
        else:
            if lse.shape[1] != num_head_q:
                raise ValueError(
                    f"VARLEN_LSE_LAYOUT_TH wants (*, {num_head_q}), got "
                    f"{tuple(lse.shape)}"
                )
        return _ptr_arg(lse)

    # Causal alignment is expressed as a *sentinel* window, resolved in the
    # kernel against each sequence's own lengths (`parse_window` in the
    # prologue). The host does not resolve it.
    #
    # It used to: `_CAUSAL_WINDOW` mapped causal_type 1/2 to literal bounds
    # using the single (seqlen_q, seqlen_k) pair passed to the launcher. That
    # is correct only when there is one such pair. Under varlen it silently
    # gave bottom-right the batch-wide `Max_seqlen_k - Max_seqlen_q` for every
    # sequence, which is wrong for all but the longest -- and invisible to any
    # test whose sequences share a uniform k-q difference, since then the two
    # numbers coincide.
    #
    # Resolving in one place removes the class of bug rather than the
    # instance.
    _CAUSAL_SENTINEL = {
        1: _WINDOW_TOPLEFT,      # j <= i
        2: _WINDOW_BOTRIGHT,     # j <= i + (seqlen_k - seqlen_q)
    }

    # ---- VarlenBits, sdpa-varlen-plan.md section 2 ----
    #
    # One byte per side, decoded by the same kernel-side function twice, plus
    # the LSE layout in byte 2. `0` is BHSD / MAX / IMPLIED on both sides with
    # an (H, T) logsumexp -- the conventional dense case, and the default.
    VARLEN_STACKED = 1
    VARLEN_LENGTH_MAX = 0 << 1
    VARLEN_LENGTH_CUMULATIVE = 1 << 1
    VARLEN_LENGTH_INDIVIDUAL = 2 << 1
    VARLEN_POSITION_IMPLIED = 0 << 3
    VARLEN_POSITION_REUSE = 1 << 3
    VARLEN_POSITION_ARRAY = 2 << 3
    # `_HT` is AOTriton's and this kernel's default: shape (H, T), T
    # contiguous. `_TH` is Transformer Engine's (T, H).
    VARLEN_LSE_LAYOUT_HT = 0 << 16
    VARLEN_LSE_LAYOUT_TH = 1 << 16

    def varlen_bits(q_side=0, k_side=0, lse_layout=VARLEN_LSE_LAYOUT_HT):
        """Assemble VarlenBits from per-side bytes."""
        for name, b in (("q_side", q_side), ("k_side", k_side)):
            if not 0 <= b <= 0xFF:
                raise ValueError(f"{name} must fit in a byte, got {b:#x}")
            if (b >> 3) & 3 == 1 and (b >> 1) & 3 != 1:
                # REUSE takes a *position* out of the length array, which is
                # only a position when the lengths are cumulative.
                raise ValueError(
                    f"{name}={b:#04x}: POSITION=REUSE requires "
                    f"LENGTH=CUMULATIVE (plan section 1, axis C)"
                )
            if (b >> 1) & 3 == 3 or (b >> 3) & 3 == 3:
                raise ValueError(f"{name}={b:#04x} uses a reserved code")
        return q_side | (k_side << 8) | lse_layout

    VARLEN_DENSE = 0
    VARLEN_COMPACT_SIDE = (
        VARLEN_STACKED | VARLEN_LENGTH_CUMULATIVE | VARLEN_POSITION_REUSE
    )                                                              # 0x0B
    VARLEN_PADDED_SIDE = VARLEN_LENGTH_CUMULATIVE | VARLEN_POSITION_IMPLIED
                                                                   # 0x02

    VARLEN_STRIDED_SIDE = (
        VARLEN_STACKED | VARLEN_LENGTH_CUMULATIVE | VARLEN_POSITION_ARRAY
    )                                                              # 0x13
    VARLEN_SEQUSED_PACKED_SIDE = (
        VARLEN_STACKED | VARLEN_LENGTH_INDIVIDUAL | VARLEN_POSITION_ARRAY
    )                                                              # 0x15
    VARLEN_SEQUSED_CACHE_SIDE = (
        VARLEN_LENGTH_INDIVIDUAL | VARLEN_POSITION_IMPLIED
    )                                                              # 0x04

    def varlen_compact(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       lse_tokens=None, lse_layout=VARLEN_LSE_LAYOUT_HT):
        """Classical packed varlen: 1THD tensors, `cu_seqlens` for both roles.

        `seqinfo_?1` is deliberately **not** passed: `POSITION = REUSE` takes
        the position out of the cumulative length value already loaded, so
        this configuration reads no position array at all (plan section 1.2).
        """
        return dict(
            bits=varlen_bits(VARLEN_COMPACT_SIDE, VARLEN_COMPACT_SIDE, lse_layout),
            seqinfo_q0=cu_seqlens_q, seqinfo_q1=None,
            seqinfo_k0=cu_seqlens_k, seqinfo_k1=None,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            lse_tokens=lse_tokens,
        )

    def varlen_padded(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                      lse_tokens=None, lse_layout=VARLEN_LSE_LAYOUT_HT):
        """BHSD tensors whose sequences are short: lengths only, no positions."""
        return dict(
            bits=varlen_bits(VARLEN_PADDED_SIDE, VARLEN_PADDED_SIDE, lse_layout),
            seqinfo_q0=cu_seqlens_q, seqinfo_q1=None,
            seqinfo_k0=cu_seqlens_k, seqinfo_k1=None,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            lse_tokens=lse_tokens,
        )

    def varlen_strided(cu_seqlens_q, cu_seqlens_k, seq_strides_q, seq_strides_k,
                       max_seqlen_q, max_seqlen_k,
                       lse_tokens=None, lse_layout=VARLEN_LSE_LAYOUT_HT):
        """Packed tensors with padding *between* sequences (TE's layout).

        Differs from `varlen_compact` in one thing only: positions come from a
        second array instead of being reused from the length array. That is
        the whole of AOTriton's `StridedVarlen`, and the reason the two roles
        must never be swapped -- `seq_strides` differences are *padded*
        extents, not lengths.
        """
        return dict(
            bits=varlen_bits(VARLEN_STRIDED_SIDE, VARLEN_STRIDED_SIDE, lse_layout),
            seqinfo_q0=cu_seqlens_q, seqinfo_q1=seq_strides_q,
            seqinfo_k0=cu_seqlens_k, seqinfo_k1=seq_strides_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            lse_tokens=lse_tokens,
        )

    def varlen_seqused_k(cu_seqlens_q, cu_seqlens_k, seqused_k,
                         max_seqlen_q, max_seqlen_k, k_is_cache=False,
                         lse_tokens=None, lse_layout=VARLEN_LSE_LAYOUT_HT):
        """Packed Q against a KV cache with per-sequence *used* lengths.

        `torch.nn.attention.varlen`'s `seqused_k`, and the configuration no
        `VarlenType` can express: the K side takes its **length** from an
        individual array and its **position** from a cumulative one, so the
        two axes read different tensors.

        `k_is_cache=True` is the rectangular variant -- a BHSD cache with no
        `cu_seqlens_k` at all, where the position is implied by the batch
        index.
        """
        k_side = (VARLEN_SEQUSED_CACHE_SIDE if k_is_cache
                  else VARLEN_SEQUSED_PACKED_SIDE)
        return dict(
            bits=varlen_bits(VARLEN_COMPACT_SIDE, k_side, lse_layout),
            seqinfo_q0=cu_seqlens_q, seqinfo_q1=None,
            seqinfo_k0=seqused_k,
            seqinfo_k1=None if k_is_cache else cu_seqlens_k,
            max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            lse_tokens=lse_tokens,
        )

    def _resolve_window(window, seqlen_q, seqlen_k):
        """(window_left, window_right), signed, as the kernel wants them.

        Non-causal still forwards a pair so both arms share one ABI and stay
        directly comparable -- the same reason the strides are always passed
        even under STRIDES_CONSTEXPR.

        Causal alignments forward a *sentinel*, not a bound: the kernel
        resolves it per sequence, which is the only correct thing to do once
        there is more than one sequence.
        """
        if CAUSAL_TYPE == 0:
            if window is not None:
                # Silently dropping it would return dense attention that is
                # the right shape, finite, and wrong -- and a window is only
                # ever passed by a caller who believes it is being applied.
                # The non-causal arm has no left-masked region to apply one
                # with, so this is a build-time choice, not a runtime one.
                raise ValueError(
                    "window= requires a causal build; this one has "
                    "causal=False. Pass causal=True, causal_type=3 for "
                    "generalized sliding-window attention"
                )
            return 0, 0
        if HOST_CAUSAL_TYPE in _CAUSAL_SENTINEL:
            if window is not None:
                raise ValueError(
                    f"causal_type={HOST_CAUSAL_TYPE} already fixes the window; "
                    "pass causal_type=3 to choose one"
                )
            _s = _CAUSAL_SENTINEL[HOST_CAUSAL_TYPE]
            wl, wr = _s, _s
        else:
            if window is None:
                raise ValueError(
                    "causal_type=3 is generalized sliding-window attention and "
                    "requires window=(left, right); "
                    f"pass ({seqlen_q}, 0) for top-left causal or "
                    f"({seqlen_q}, {seqlen_k - seqlen_q}) for bottom-right"
                )
            wl, wr = window
        return int(wl), int(wr)

    def _dropout_args(dropout_p, seed, offset_base):
        """(seed, offset_base, threshold, 1/(1-p)) in launch order.

        The threshold and the scale are computed here, once per call, rather
        than per element in the kernel -- `philox.dropout_threshold` turns the
        probability into an i32 the raw random can be compared against, so the
        hot path never converts a random to a float.
        """
        if not ENABLE_DROPOUT:
            return 0, 0, 0, 1.0
        if dropout_p is None:
            raise ValueError("this build has dropout=True and requires dropout_p=")
        p = float(dropout_p)
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout_p must be in [0, 1), got {p}")
        return (int(seed), int(offset_base), dropout_threshold(p), 1.0 / (1.0 - p))

    def _bias_args(bias):
        """(pointer, stride_b0, stride_b1, stride_b2) for a (B, H, Sq, Sk) bias.

        The last axis is the KV column and must be contiguous, exactly as the
        D axis is for Q/K/V/O -- the kernel loads eight adjacent columns per
        accumulator group in one v8.
        """
        if not BIAS_TYPE:
            return _NULL_PTR, 0, 0, 0
        if bias is None:
            raise ValueError("this build has BIAS_TYPE=1 and requires bias=")
        if bias.dim() != 4:
            raise ValueError(f"bias must be rank 4 (B, H, Sq, Sk), got {tuple(bias.shape)}")
        if bias.stride(3) != 1:
            raise ValueError(
                f"bias must have a contiguous last (Sk) dimension, got "
                f"stride(3)={bias.stride(3)}"
            )
        return (_ptr_arg(bias), bias.stride(0), bias.stride(1), bias.stride(2))

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

    def _varlen_args(varlen, seqlen_q, seqlen_k):
        """(bits, q0, q1, k0, k1, max_q, max_k) in launch order.

        `varlen` is None for the dense case, else a dict with `bits` and
        whichever `seqinfo_*` tensors that configuration reads. Unread slots
        stay **null**, which is safe because the kernel's decode branches
        rather than selects -- see the prologue.
        """
        if varlen is None:
            return (0, _NULL_PTR, _NULL_PTR, _NULL_PTR, _NULL_PTR,
                    int(seqlen_q), int(seqlen_k))
        if STRIDES_CONSTEXPR:
            raise ValueError(
                "STRIDES_CONSTEXPR derives the layout from the shape, which "
                "varlen invalidates; it is a dense-only diagnostic arm"
            )
        # No implemented-subset gate: every encodable side byte now decodes,
        # since the decoder is one function covering all three axis values.
        # `varlen_bits` rejects the combinations that are not *meaningful*
        # (reserved codes, REUSE without cumulative lengths).
        bits = int(varlen["bits"])
        got = tuple(
            _ptr_arg(varlen[k]) if varlen.get(k) is not None else _NULL_PTR
            for k in ("seqinfo_q0", "seqinfo_q1", "seqinfo_k0", "seqinfo_k1")
        )
        return (bits,) + got + (int(varlen["max_seqlen_q"]),
                                int(varlen["max_seqlen_k"]))

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

    def _launch(Q, K, V, O, batch_size, seqlen_q, seqlen_k=None, scale=None, stream=None, lse=None, window=None, varlen=None, bias=None, dropout_p=None, philox_seed=0, philox_offset=0):  # noqa: E741
        seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
        ptrs, meta, st = _prep(Q, K, V, O)
        _lse_p = _lse_args(lse, seqlen_q, varlen, meta[0])
        _wl, _wr = _resolve_window(window, seqlen_q, seqlen_k)
        _vb, _sq0, _sq1, _sk0, _sk1, _mq, _mk = _varlen_args(
            varlen, seqlen_q, seqlen_k
        )
        _bp, _sb0, _sb1, _sb2 = _bias_args(bias)
        _ps, _po, _ip, _dsc = _dropout_args(dropout_p, philox_seed, philox_offset)
        _run_compiled(
            launch_flash_attn_aiw,
            *ptrs,
            _lse_p,
            _bp,
            _sq0, _sq1, _sk0, _sk1,
            batch_size,
            _vb,
            _mq,
            _mk,
            _wl,
            _wr,
            _ps, _po, _ip, _dsc,
            *meta,
            *st,
            _sb0, _sb1, _sb2,
            _resolve_scale(Q, scale),
            stream if stream is not None else fx.Stream(None),
        )

    def _compile(Q, K, V, O, batch_size, seqlen_q, seqlen_k=None, scale=None, stream=None, lse=None, window=None, varlen=None, bias=None, dropout_p=None, philox_seed=0, philox_offset=0):  # noqa: E741
        seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
        ptrs, meta, st = _prep(Q, K, V, O)
        _lse_p = _lse_args(lse, seqlen_q, varlen, meta[0])
        _wl, _wr = _resolve_window(window, seqlen_q, seqlen_k)
        _vb, _sq0, _sq1, _sk0, _sk1, _mq, _mk = _varlen_args(
            varlen, seqlen_q, seqlen_k
        )
        _bp, _sb0, _sb1, _sb2 = _bias_args(bias)
        _ps, _po, _ip, _dsc = _dropout_args(dropout_p, philox_seed, philox_offset)
        return flyc.compile(
            launch_flash_attn_aiw,
            *ptrs,
            _lse_p,
            _bp,
            _sq0, _sq1, _sk0, _sk1,
            batch_size,
            _vb,
            _mq,
            _mk,
            _wl,
            _wr,
            _ps, _po, _ip, _dsc,
            *meta,
            *st,
            _sb0, _sb1, _sb2,
            _resolve_scale(Q, scale),
            fx.Stream(stream),
        )

    _launch.compile = _compile
    # Attached rather than module-level: the validation closes over
    # STRIDES_CONSTEXPR and the shipped-configuration set.
    _launch.varlen_bits = varlen_bits
    _launch.varlen_compact = varlen_compact
    _launch.varlen_padded = varlen_padded
    _launch.varlen_strided = varlen_strided
    _launch.varlen_seqused_k = varlen_seqused_k
    return _launch

def build_flash_attn_func_aiw_module(**kwargs):
    """Keyword front end: name a problem, get the policy's schedule.

    Any `FmhaKnobs` field may be passed as a keyword to pin it; the rest are
    resolved by `resolve_knobs`. This is what keeps "the tuning module is the
    only producer of a schedule" true even for callers who never mention one.
    """
    meta_fields = {f.name for f in fields(FmhaInputMetadata)}
    knob_fields = {f.name for f in fields(FmhaKnobs)}
    unknown = set(kwargs) - meta_fields - knob_fields
    if unknown:
        raise TypeError(f"unknown build parameter(s): {sorted(unknown)}")
    # `resolve_knobs`, not `plan`: this front end takes a *compiled tile width*
    # and must keep rejecting anything off the ladder. Rounding a real head_dim
    # up is the interface's job, because only it also arranges the runtime
    # extent and the padded_head contract that make the rounding safe.
    meta = FmhaInputMetadata(**{k: v for k, v in kwargs.items() if k in meta_fields})
    overrides = FmhaKnobs(**{k: v for k, v in kwargs.items() if k in knob_fields})
    return build_flash_attn_func_aiw_module_primary(meta, resolve_knobs(meta, overrides))
