# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Hardware detail for the gfx1201 (RDNA4) attention kernels.

Two boundaries, and the arch suffix is the important one.

**Not `kernels/common/`**: what is here either encodes an attention ABI
decision or has no meaning outside one. The general-purpose siblings live next
door -- `kernels/common/mem_ops.py` for pointer and global load/store,
`kernels/common/utils.py` for the scalar integer helpers.

**Not shared with gfx950 or gfx1250 either.** `flash_attn_utils.py` is the
gfx950 equivalent and this module deliberately does not import from it or
extend it. The hardware differs enough that the *algorithms* differ: gfx950
schedules around MFMA/VALU co-execution, which RDNA4 has no equivalent of, and
its dualwave traits/context machinery exists to serve that. Merging the two
would mean one abstraction serving two schedules that agree on almost nothing.
When gfx1250 FMHA arrives it gets its own `fmha_common_gfx1250.py` for the
same reason.


How to hand a helper *object* to kernel code
--------------------------------------------

A helper that carries state -- `FastMath`, `MaskedAxis` -- cannot simply be
built and assigned in the kernel body. **No variable of any kind may hold a
non-MLIR Python object that is live across a dynamic `if`.** The AST rewriter
turns such an `if` into an `scf.if` and collects every variable live across it
as loop-carried state, which must be MLIR-backed. A Python object is not, and
the failure is late, config-dependent and unobvious:

    fastmath = FastMath(FP_MODE)         # assigned in the body
    -> TypeError: state variable 'fastmath' is FastMath, not an MLIR Value

    qk_cols = MaskedAxis(hdim, ...)      # assigned in the body, read in a
    -> UnboundLocalError: cannot access   #   nested fn under `if row_valid:`
       local variable 'qk_cols'

Passing the object as a *parameter* does not help; it is still a variable live
across the branch. That was measured, not assumed: threading it through
`coop_load_store_k(start_k, cols, ...)` compiles at BLOCK_DMODEL 16 non-causal
and fails at 16 causal with `state variable 'cols' is MaskedAxis, not an MLIR
Value`.

Two patterns work, and which one applies depends on what the object needs:

1. **Build it on the host** when everything it captures is `const_expr`. This
   is `FastMath(FP_MODE)`, constructed in the builder beside the knob unpacking
   and captured by the traced code as a plain constant.

2. **Build it in a factory function** when it needs traced values. This is
   `MaskedAxis`, whose extent is `hdim_qk` -- a kernel *argument*, so there is
   nothing to capture on the host:

       def qk_cols():
           return fmha.MaskedAxis(_hdim_qk_i, active=PADDED_HEAD, ...)

       ... qk_cols().safe(load_col_base) ...

   The call re-creates the object inside whatever branch reads it, so nothing
   is ever live across the `scf.if`. It costs nothing -- trace-time Python that
   emits no IR, and every gate config is bitwise identical to the variable form
   where the variable form compiles at all.

The `()` is the tell that a helper is in category 2. Prefer 1 when the object
allows it; it reads better and constructs once.
"""

from dataclasses import dataclass

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import scf as _scf
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import ArithValue, _to_raw
from gfx1201_standalone import buffer_ops, utils as common_utils

__all__ = ["llvm_ptr_ty", "pointer_to_llvm_ptr", "lds_load_v8", "lds_store_vx", "global_load_tr_v8",
           "bitcast_i32", "pack_bf16_pair", "bf16_trunc_pack_v8",
           "FastMath", "MaskedAxis",
           "lds_f32_ptr", "lds_f32_store", "lds_f32_load",
           "reduce_s_across_shards",
           "cond_load", "seqinfo_addr", "decode_addressing", "lse_token_pitch",
           "WINDOW_TOPLEFT", "WINDOW_BOTRIGHT", "resolve_window",
           "CausalRegions", "decompose_causal_regions", "make_addr_pair"]


def llvm_ptr_ty() -> ir.Type:
    return ir.Type.parse("!llvm.ptr")


def pointer_to_llvm_ptr(ptr) -> ir.Value:
    """A `!fly.ptr` kernel argument as an LLVM pointer, for raw load/store.

    Not `mem_ops.get_llvm_ptr`, which takes the same idea from the other end:
    it calls `extract_aligned_pointer_as_index`, which requires a memref and
    rejects a `!fly.ptr` outright. `fx.ptrtoint` is the reverse -- it reads
    `ptr.address_space` and so requires the pointer. Neither op accepts both,
    so this is not a duplicate of that helper but the other half of the pair.

    **Why the attention kernels take pointers and not tensors.** Every other
    kernel in the tree declares `fx.Tensor`, which would reach `get_llvm_ptr`
    directly and make this function unnecessary. It was measured on the
    gfx1201 SDPA kernel and rejected: each `fx.Tensor` argument adds a 40-byte
    `by_value` memref descriptor *interleaved immediately after its pointer*,
    growing the kernarg segment from 268 to 428 bytes and shifting the offset
    of every argument after the first. AOTriton dispatches that hsaco
    directly rather than through the Python wrapper, so the kernarg layout is
    an ABI, and rearranging it to delete one four-line helper is a bad trade.

    The switch would not even remove the helper: `L`, `Bias` and the four
    `seqinfo_*` arguments are optional, signalled by a null pointer the host
    builds with `flyc.from_c_void_p(..., 0)`, and there is no tensor to pass
    for a thing that is not there. They would stay pointers and keep needing
    this.
    """
    ptr_i64 = arith.index_cast(T.i64, fx.ptrtoint(ptr))
    return _llvm.IntToPtrOp(llvm_ptr_ty(), ptr_i64).result


# --------------------------------------------------------------------------
# LDS access, split for honest alignment
# --------------------------------------------------------------------------
#
# The two functions below exist only to avoid over-promising alignment, which
# is a property of RDNA's LDS instruction selection rather than of attention,
# and is why they live in the arch module.


def lds_load_v8(lds_ptr, lds_idx, v4_type):
    """Load 8 half-precision elements from LDS as two honest 8-byte accesses.

    **Not one v8 load.** K/V rows are `K_STRIDE * 2` bytes apart, so these
    addresses are only guaranteed 8-byte aligned. `fly.ptr_load` emits no
    alignment attribute, so LLVM falls back to the vector type's ABI
    alignment -- 16 B for v8f16, 32 B for v16f16 -- and that over-promise makes
    the backend select `ds_load_b128` on addresses that are not 16-byte
    aligned. Measured 2.2x slower (92 -> 39 TFLOPS), and undefined behaviour
    besides. Two v4f16 accesses carry a truthful `align 8` and fold back into
    `ds_load2_b64`.
    """
    lo = fx.ptr_load(lds_ptr + fx.Int32(lds_idx), result_type=v4_type)
    hi = fx.ptr_load(lds_ptr + fx.Int32(lds_idx + 4), result_type=v4_type)
    return Vec(lo).shuffle(Vec(hi), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()


def lds_store_vx(lds_ptr, vec, lds_idx, vec_width):
    """Store `vec_width` half elements to LDS in 8-byte pieces. See `lds_load_v8`."""
    v = Vec(vec)
    for _i in range_constexpr(vec_width // 4):
        part = v.shuffle(v, [_i * 4, _i * 4 + 1, _i * 4 + 2, _i * 4 + 3])
        fx.ptr_store(part, lds_ptr + fx.Int32(lds_idx + _i * 4))


def global_load_tr_v8(base_i64, base64, off32, v8_type):
    """One `global_load_tr_b128`: an 8x8 16-bit transpose per lane-group.

    Lane g_i supplies an address; the 8 contiguous elements there become
    column i of the group's output, so lane g_j receives [M_0[j] .. M_7[j]].
    Verified empirically on gfx1201. The instruction is RDNA-only, which is
    what puts this in the arch module.

    Address is split the same way the rest of the kernel splits one: the
    (batch, head, tile) origin and the intra-tile part, added in 64 bits.
    Feeding LLVM `uniform_i64 + divergent` is what lets `SelectGlobalSAddr`
    keep the base in SGPRs instead of forcing a 64-bit VGPR address pair.

    The divergent half is deliberately *not* narrowed to i32 on the way. It
    carries `row_in_tile * stride_seq`, and a view's sequence stride is bounded
    by the tensor it was taken from rather than by the shape here -- eight
    heads sliced out of a 1 GiB (1, 64, 16384, 512) f16 tensor give
    `stride_seq = 8388608`, whose 256th row is exactly 2**31. Narrowing it
    wrapped and read another allocation.
    """
    base_bytes = arith.index_cast(T.i64, _to_raw(fx.Index(base64) * 2))
    off_bytes = arith.index_cast(T.i64, _to_raw(fx.Index(off32) * 2))
    addr = arith.addi(arith.addi(base_i64, base_bytes), off_bytes)
    p = _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), addr).result
    return rocdl.global_load_tr_b128(v8_type, p)


def bitcast_i32(value):
    return fx.Int32(ArithValue(value).bitcast(fx.Int32.ir_type))

def pack_bf16_pair(lo, hi, shift, mask):
    lo_i32 = bitcast_i32(lo)
    hi_i32 = bitcast_i32(hi)
    return (hi_i32 & mask) | lo_i32.shrui(shift)

def bf16_trunc_pack_v8(f32_vals, elem_dtype):
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
            pack_bf16_pair(f32_vals[j * 2], f32_vals[j * 2 + 1], _c16, _cmask)
        )
    return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()


class FastMath:
    """The softmax's float ops, with one `arith.FastMathFlags` set bound once.

    A class rather than four free functions taking the flags, because the flag
    set is the thing worth making visible. It is a *knob* -- `fp_mode` selects
    between "fast", "noninf" and "safe" -- and the choice is load-bearing:
    "fast" includes `ninf`, which once silently deleted the KV tail mask,
    because `exp2(-inf - m)` folds to something the flag says cannot happen.

    Takes the mode rather than a flag set, so the mapping from knob to flags
    lives here and not at each kernel that uses it. Construct it on the **host**
    side of the builder, not in the kernel body: `fp_mode` is `const_expr`, so
    this is a plain Python object the traced code captures, and assigning it
    inside the body makes the AST rewriter treat it as `scf` loop/if state
    ("state variable 'fastmath' is FastMath, not an MLIR Value").
    """

    __slots__ = ("flags",)

    def __init__(self, fp_mode: str):
        _F = arith.FastMathFlags
        if fp_mode == "fast":
            self.flags = _F.fast
        elif fp_mode == "noninf":
            self.flags = _F.reassoc | _F.nnan | _F.nsz | _F.arcp | _F.contract | _F.afn
        elif fp_mode == "safe":  # as "noninf", also dropping nnan
            self.flags = _F.reassoc | _F.nsz | _F.arcp | _F.contract | _F.afn
        else:
            raise ValueError(f"unknown fp_mode {fp_mode!r}; expected fast/noninf/safe")

    def div(self, a, b):
        return arith.divf(_to_raw(a), _to_raw(b), fastmath=self.flags)

    def add(self, a, b):
        return arith.addf(_to_raw(a), _to_raw(b), fastmath=self.flags)

    def sub(self, a, b):
        return arith.subf(_to_raw(a), _to_raw(b), fastmath=self.flags)

    def mul(self, a, b):
        return arith.mulf(_to_raw(a), _to_raw(b), fastmath=self.flags)

    def max(self, a, b):
        return arith.MaxNumFOp(_to_raw(a), _to_raw(b), fastmath=self.flags).result


class MaskedAxis:
    """One axis of a tile whose index can run past the real extent.

    Out-of-range indices always need two things, and keeping them together is
    the point: an address that is *safe to issue*, and a way to *discard what
    it returns*. Issuing the access unconditionally and throwing the value away
    beats branching around it, but only once the address has been redirected
    somewhere legal -- element 0 of the axis, which always exists.

    **One class covers rows and columns**, because the apparent difference
    between them is not about the axes. An access reads `width` contiguous
    elements *along one axis*; for that axis the extent boundary can fall
    inside the access, so validity is per element. For every other axis the
    index is a single scalar and the whole access stands or falls together.
    In this kernel the vector runs along the column axis, which is why columns
    look "per element" and rows "whole" -- but that is a property of the
    access, not of the axis, and `valid(idx)` is exactly `mask(idx, 1)`.

    `active=False` compiles the masking away, for an axis whose extent is known
    to be a multiple of the access width.
    """

    __slots__ = ("extent", "active", "elem_dtype")

    def __init__(self, extent, active=True, elem_dtype=None):
        self.extent = extent
        self.active = active
        # Only `discard` needs this, so the row axis leaves it unset. It is a
        # property of the tensor and every access along an axis shares it.
        #
        # `width` is deliberately *not* bound here. It belongs to the access,
        # not the axis, and the two accesses along the QK column axis agree
        # only by coincidence: the cooperative loads are `VEC_WIDTH` wide while
        # the Q preload is 8 wide because `load_global_v8f16` is tied to the
        # WMMA operand shape. Both are 8 today from unrelated definitions, and
        # binding one would make the Q preload silently follow `VEC_WIDTH`.
        self.elem_dtype = elem_dtype

    def _bound(self):
        """The extent as a traced value.

        Resolved lazily so the object can be built on the host when the extent
        is a `const_expr` int, which is what the kernel body requires: an
        object assigned inside the body becomes a local of the recompiled
        function and does not survive the AST rewriter's scoping.
        """
        return fx.Index(self.extent) if isinstance(self.extent, int) else self.extent

    def valid(self, idx):
        """i1: is this index inside the extent?

        A signed compare, on every axis. `fx.Index` is *unsigned*, so a plain
        `idx < extent` emits `v_cmp_lt_u64`; the answer agrees here because
        every index and extent is non-negative, but the two are not the same
        code and mixing them per axis was a difference with no reason behind
        it. Signed throughout, matching the rest of the file.
        """
        return arith.cmpi(arith.CmpIPredicate.slt, _to_raw(idx), _to_raw(self._bound()))

    def mask(self, idx, width):
        """i1 vector, element j set iff `idx + j` is inside the extent.

        Built from a loop-invariant index at every current caller, so it hoists
        out of the KV loop and costs one vector select per access inside it.
        """
        return Vec.from_elements(
            [ArithValue(self.valid(idx + fx.Index(j))) for j in range_constexpr(width)],
            fx.Boolean,
        )

    def safe(self, idx, addressed=None):
        """`addressed` if `idx` is inside the extent, else 0.

        `addressed` defaults to `idx`, which is the column case. Rows need the
        two to differ: the bound is on the *absolute* row, `start_q + ...`,
        while the address is built from the row's offset *within the tile*, so
        the tested and the redirected quantity are not the same value.
        """
        if addressed is None:
            addressed = idx
        if not self.active:
            return addressed
        return fx.Index(ArithValue(self.valid(idx)).select(addressed, fx.Index(0)))

    def discard(self, vec, idx, width):
        """Zero the elements of `vec` whose index is past the extent."""
        if not self.active:
            return vec
        zeros = Vec.filled(width, 0.0, self.elem_dtype)
        return self.mask(idx, width).select(Vec(vec), zeros).ir_value()


# --------------------------------------------------------------------------
# f32 scratch aliased over the 16-bit LDS tile
# --------------------------------------------------------------------------
#
# The cross-shard S reduction needs f32 scratch, and the KV tile it borrows
# space from is `elem_dtype` (16-bit). There is no retyped view of a shared
# pointer, so the address is built by hand: `ptrtoint` on a shared pointer
# yields the 32-bit LDS offset, and the f32 element index is scaled into it.
#
# Plain functions taking both halves of the base, rather than an object: these
# are read inside the reduction's dynamic branches, and an object could not be
# live there (see "How to hand a helper object to kernel code" above). Both
# arguments are safe to hold in kernel-body variables -- one is an MLIR value,
# the other a `const_expr` int.


def lds_f32_ptr(lds_byte_base, byte0, index):
    """`!llvm.ptr<3>` at f32 element `index` of the scratch starting at `byte0`."""
    off = fx.Int32(byte0) + fx.Int32(index) * fx.Int32(4)
    addr = arith.addi(lds_byte_base, _to_raw(off))
    return _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<3>"), addr).result


def lds_f32_store(lds_byte_base, byte0, index, value):
    _llvm.StoreOp(_to_raw(value), lds_f32_ptr(lds_byte_base, byte0, index))


def lds_f32_load(lds_byte_base, byte0, index):
    return _llvm.LoadOp(ir.F32Type.get(), lds_f32_ptr(lds_byte_base, byte0, index)).result


def reduce_s_across_shards(
    s_accs,
    *,
    lds_byte_base,
    byte0,
    wave_id,
    lane,
    shard_id,
    q_tile_in_block,
    num_shards,
    f32_per_wave,
    warp_size,
    fastmath,
):
    """Sum one Q row-tile's S accumulators across the QK shards, through LDS.

    Each shard-wave holds a partial sum over its own slice of BLOCK_DMODEL;
    the full S is their sum. Returns the reduced accumulators, same shape in.

    **Explicit partials, not `ds_add_f32`.** The atomic form measured 1055
    WMMA-equivalents against 54 for this, because every lane contends on the
    same address -- see `kernels/microbench/lds_reduce.py`. So each wave writes
    its own partials to a private slot, and then every wave reads the others'
    and adds them locally: two barriers and no contention.

    Called only when `num_shards > 1`, which on the current ladder is
    BLOCK_DMODEL 384 (2 shards) and 512 (4). Every other width reduces nothing
    and never reaches here -- worth knowing when gating a change to it, since
    the usual 128 build does not execute a line of this.
    """
    s_flat = [_to_raw(Vec(a)[r]) for a in s_accs for r in range_constexpr(8)]

    own = wave_id * fx.Index(f32_per_wave)
    for e in range_constexpr(len(s_flat)):
        lds_f32_store(
            lds_byte_base, byte0, own + fx.Index(e * warp_size) + lane, s_flat[e]
        )
    gpu.barrier()

    base_group = q_tile_in_block * fx.Index(num_shards * f32_per_wave)
    for e in range_constexpr(len(s_flat)):
        acc = s_flat[e]
        for k in range_constexpr(num_shards - 1):
            peer = base_group + (
                (shard_id + fx.Index(k + 1)) % fx.Index(num_shards)
            ) * fx.Index(f32_per_wave)
            acc = fastmath.add(
                acc,
                lds_f32_load(
                    lds_byte_base, byte0, peer + fx.Index(e * warp_size) + lane
                ),
            )
        s_flat[e] = acc
    gpu.barrier()

    return [
        Vec.from_elements(
            [fx.Float32(s_flat[st * 8 + r]) for r in range_constexpr(8)], fx.Float32
        ).ir_value()
        for st in range_constexpr(len(s_accs))
    ]


# --------------------------------------------------------------------------
# Varlen prologue: VarlenBits -> per-sequence addressing
# --------------------------------------------------------------------------


def cond_load(cond, addr, default):
    """Load i32 from `addr` when `cond`, else `default`. The load is skipped.

    A real `scf.if`, not a select, and that is the point: the sequence-info
    pointers are **null** whenever their mode is off, so a select -- which
    evaluates both arms -- would fault. Inside the region the load is never
    issued; verified against a null pointer.

    Built as an explicit `IfOp` rather than Python's `if`, which is what lets
    this live in a module at all: the rewrite from `if` to `scf.if` is lexical
    per `@flyc.kernel` function, but an `IfOp` written out needs no rewriting.

    `addr` is computed by the caller and may be derived from a null pointer --
    address arithmetic touches no memory.
    """
    if_op = _scf.IfOp(_to_raw(cond), results_=[T.i32], has_else=True)
    with ir.InsertionPoint(if_op.then_block):
        _scf.YieldOp([_llvm.LoadOp(T.i32, addr).result])
    with ir.InsertionPoint(if_op.else_block):
        _scf.YieldOp([_to_raw(default)])
    return fx.Int32(if_op.results[0])


def seqinfo_addr(ptr, index):
    """`&ptr[index]` for an i32 sequence-info array. No memory is touched."""
    return buffer_ops.get_element_ptr(
        pointer_to_llvm_ptr(ptr), fx.Int64(index), elem_type=T.i32
    )


def decode_addressing(varlen_bits, bits_shift, max_seqlen, s0, s1, z, num_seqlens):
    """One side of VarlenBits: where this workgroup's sequence lives.

    Returns `(seqlen, row_off, batch)` -- how long this sequence is, which row
    it starts at, and which batch index to use. Called once for Q and once for
    K. See section 3.1 of `sdpa-varlen-plan.md`; the axes are STACKED (bit 0),
    LENGTH (bits 2:1) and POSITION (bits 4:3).

    The LSE token pitch is *not* here, though it decodes from the same bits: it
    describes the logsumexp output rather than where Q or K live, and only the
    Q side needs it. See `lse_token_pitch`.

    Every load goes through `cond_load`, so the shape is flat -- fetch what
    each mode might need, then select. `s0[z]` serves both length modes and,
    under REUSE, the position too, which is why three loads cover five modes.
    """
    bits = fx.Int32(varlen_bits) >> fx.Int32(bits_shift)
    stacked = (bits & fx.Int32(1)) != fx.Int32(0)
    lenmode = (bits >> fx.Int32(1)) & fx.Int32(3)
    posmode = (bits >> fx.Int32(3)) & fx.Int32(3)

    cumulative = lenmode == fx.Int32(1)
    individual = lenmode == fx.Int32(2)
    reuse = posmode == fx.Int32(1)      # position already read as `cur`
    array = posmode == fx.Int32(2)      # position from its own array
    zero = fx.Int32(0)

    cur = cond_load(lenmode != zero, seqinfo_addr(s0, z), zero)
    nxt = cond_load(cumulative, seqinfo_addr(s0, z + fx.Int32(1)), zero)
    pos = cond_load(array, seqinfo_addr(s1, z), zero)

    seqlen = common_utils.ssel(
        cumulative, nxt - cur,
        common_utils.ssel(individual, cur, fx.Int32(max_seqlen)),
    )
    row_off = common_utils.ssel(
        array, pos,
        common_utils.ssel(
            reuse, cur,
            common_utils.ssel(stacked, z * fx.Int32(max_seqlen), zero),
        ),
    )
    batch = common_utils.ssel(stacked, zero, z)
    return seqlen, row_off, batch


def lse_token_pitch(varlen_bits, bits_shift, max_seqlen, s0, s1, num_seqlens):
    """Row pitch of the logsumexp output, in tokens. Q side only.

    Batched layouts pad every row-group to `max_seqlen`; stacked ones run to
    the batch total, which lives in slot [N] of whichever array supplies
    positions -- the prefix-sum assumption of plan section 9.4, asserted host
    side.

    Derived from the bits rather than passed because the logsumexp tensor,
    alone among the tensors here, is always compact: its strides are a function
    of the bits, and passing them would be a second source of truth for one
    fact (plan section 4.2).
    """
    bits = fx.Int32(varlen_bits) >> fx.Int32(bits_shift)
    stacked = (bits & fx.Int32(1)) != fx.Int32(0)
    posmode = (bits >> fx.Int32(3)) & fx.Int32(3)
    reuse = posmode == fx.Int32(1)
    array = posmode == fx.Int32(2)
    zero = fx.Int32(0)

    total_s0 = cond_load(
        arith.andi(_to_raw(stacked), _to_raw(reuse)),
        seqinfo_addr(s0, num_seqlens), zero,
    )
    total_s1 = cond_load(
        arith.andi(_to_raw(stacked), _to_raw(array)),
        seqinfo_addr(s1, num_seqlens), zero,
    )
    return common_utils.ssel(
        stacked,
        common_utils.ssel(
            reuse, total_s0,
            common_utils.ssel(
                array, total_s1, fx.Int32(num_seqlens) * fx.Int32(max_seqlen)
            ),
        ),
        fx.Int32(max_seqlen),
    )


# --------------------------------------------------------------------------
# Sliding-window attention: resolving the window, and cutting the KV range
# --------------------------------------------------------------------------

WINDOW_TOPLEFT = -2147483647   # 0x80000001
WINDOW_BOTRIGHT = -2147483646  # 0x80000002


def resolve_window(window_left, window_right, seqlen_q, seqlen_k):
    """`(window_left, window_right)` with the causal sentinels resolved.

    `Window_left` / `Window_right` may carry `WINDOW_TOPLEFT` or
    `WINDOW_BOTRIGHT` instead of a literal bound, and they are resolved
    against *this sequence's* lengths rather than on the host. That is the
    whole reason the sentinels exist: host resolution works only when there is
    one length to resolve against, and under varlen bottom-right needs
    `seqlen_k[z] - seqlen_q[z]`, which differs per sequence. Matches
    AOTriton's `parse_window`.

    Both sentinels give an unbounded left edge -- no row reaches further back
    than the start of its own sequence -- so they differ only in the right one.

    **Everything derived from a window stays i32.** Window bounds go negative;
    that is what a sentinel and a leading masked region are. `fx.Int32` is
    signed, so `<`/`>` emit `slt`/`sgt`, while `fx.Index` is unsigned and
    64-bit -- widening any of these even once makes the same comparison
    unsigned and a negative bound comes out enormous.
    """
    left = fx.Int32(window_left)
    right = fx.Int32(window_right)
    left_is_sentinel = arith.ori(
        _to_raw(left == fx.Int32(WINDOW_TOPLEFT)),
        _to_raw(left == fx.Int32(WINDOW_BOTRIGHT)),
    )
    left = common_utils.ssel(ArithValue(left_is_sentinel), seqlen_q, left)
    right = common_utils.ssel(right == fx.Int32(WINDOW_TOPLEFT), fx.Int32(0), right)
    right = common_utils.ssel(
        fx.Int32(window_right) == fx.Int32(WINDOW_BOTRIGHT),
        seqlen_k - seqlen_q,
        right,
    )
    return left, right


@dataclass(frozen=True, slots=True)
class CausalRegions:
    """The three contiguous KV block runs a causal/windowed Q block walks.

    Every field is a traced `fx.Int32`, not a Python int -- these are values
    the kernel computes per workgroup. Signed, because `right_col0` goes
    negative when the window admits no key at all; see
    `decompose_causal_regions`.

    A dataclass rather than a `NamedTuple` to match `Philox` next door, and
    because no caller destructures it positionally -- the kernel reads the
    seven fields by name. Being a Python object it is subject to the usual
    rule: do not let one live across a dynamic `if` (see "How to hand a helper
    object to kernel code"). The kernel unpacks it immediately, which is why
    it is safe here.
    """

    n_left: fx.Int32      # masked tiles before the full run
    n_full: fx.Int32      # tiles with no mask at all
    n_right: fx.Int32     # masked tiles after it
    left_col0: fx.Int32   # first KV column of each run
    full_col0: fx.Int32
    right_col0: fx.Int32
    masked_col0: fx.Int32  # first column of the masked run, whichever side


def decompose_causal_regions(
    start_q, q_len, k_len, window_left, window_right, block_m, block_n, alive
):
    """Cut this Q block's visited KV range into `[masked][full][masked]`.

    **Three regions, not two.** A left window kills columns at the *start* of
    the range as well as the end, so masked tiles are a prefix as well as a
    suffix and tile 0 is not automatically live. A negative `window_left` is
    the sharpest case: it pushes the whole band right of the diagonal, so the
    leading masked run can span several tiles rather than clipping one. Do not
    carry the non-causal two-region intuition in here.

    The three are contiguous and non-overlapping *by construction*, because
    they are derived by cutting one visited range rather than intersected as
    three independent intervals. That collapses two of the three special cases
    in `sdpa-gswa-plan.md` section 2.2: a window narrower than a block leaves
    the full region empty, which is detected once and turns the other two into
    a single masked run, and an irregular `seqlen_q` needs no special handling
    because `q_hi` already bounds the rows.

    A column c is live for row i iff `i - window_left <= c <= i + window_right`,
    so over the block the live columns span
    `[start_q - window_left, (q_hi - 1) + window_right]`, and a tile is *fully*
    live iff every one of its columns is live for every row -- worst case the
    largest row on the left and the smallest on the right.

    `alive` is false for a workgroup whose rows all sit past `q_len`, which the
    varlen grid dispatches because its Q extent is sized from `Max_seqlen_q`.
    The kernel is one single-exit trace and cannot return out of those (plan
    section 6.1), so the visited range is *inverted* instead and every region
    count falls to zero. Dropping this makes those workgroups walk real tiles.

    Everything here is i32 and signed, deliberately: `left_col0` and friends
    go negative when the window admits no key at all. See `resolve_window`.
    """
    one = fx.Int32(1)
    zero = fx.Int32(0)
    bn = fx.Int32(block_n)

    q_start = fx.Int32(start_q)
    q_hi = common_utils.smin(q_start + fx.Int32(block_m), q_len)
    q_last = q_hi - one

    # Blocks that exist at all, and the last block that is *whole*. Splitting
    # these is section 2.2 case 3: a ragged seqlen_k leaves a partial final
    # tile, which must be masked rather than counted as full.
    blk_last = common_utils.sdiv_rd_pow2(k_len - one, block_n)
    blk_last_whole = common_utils.sdiv_rd_pow2(k_len, block_n) - one

    # The visited range: outside it every column is dead for every row in this
    # Q block, so those tiles are not walked at all.
    v_lo = common_utils.smax(
        common_utils.sdiv_rd_pow2(q_start - window_left, block_n), zero
    )
    v_hi = common_utils.smin(
        blk_last, common_utils.sdiv_rd_pow2(q_last + window_right, block_n)
    )
    v_hi = common_utils.ssel(alive, v_hi, v_lo - one)

    # Rounded *up* on the left: a block is fully live only once its first
    # column clears the leftmost row's window. Rounding down would send a
    # partly-masked tile through the unmasked loop body -- invisible to a
    # tolerance test, not to the bitwise one.
    l_first_full = common_utils.sdiv_rd_pow2(
        q_last - window_left + fx.Int32(block_n - 1), block_n
    )
    r_first_mask = common_utils.sdiv_rd_pow2(q_start + window_right + one, block_n)

    fb_lo = common_utils.smax(l_first_full, v_lo)
    fb_hi = common_utils.smin(
        common_utils.smin(r_first_mask - one, blk_last_whole), v_hi
    )
    fb_empty = fb_lo > fb_hi

    # Cut [v_lo, v_hi] at the full region. With no full region the whole range
    # becomes one masked run -- section 2.2 case 2, the window narrower than a
    # block, falling out for free.
    lb_hi = common_utils.ssel(fb_empty, v_hi, fb_lo - one)
    rb_lo = common_utils.ssel(fb_empty, v_hi + one, fb_hi + one)

    n_left = common_utils.smax(lb_hi - v_lo + one, zero)
    n_full = common_utils.smax(fb_hi - fb_lo + one, zero)
    n_right = common_utils.smax(v_hi - rb_lo + one, zero)

    left_col0 = v_lo * bn
    right_col0 = rb_lo * bn
    full_col0 = fb_lo * bn
    # First tile of the masked run, which is also what the full loop's last
    # prefetch must fetch: the two loops are adjacent only when the left run is
    # empty. Clamped, because with a window admitting no key at all every run
    # is empty and `rb_lo` sits below zero -- and this value still reaches the
    # prologue's address computation.
    masked_col0 = common_utils.smax(
        common_utils.ssel(n_left > zero, left_col0, right_col0), zero
    )
    return CausalRegions(
        n_left, n_full, n_right, left_col0, full_col0, right_col0, masked_col0
    )


def make_addr_pair(
    strides, head, batch_index, row_off, *, seqlen_k, seq_last, hoist, clamp
):
    """Address builders for one tensor: `(tbase, toff, kv_addr)`.

    Q, K, V and O each get their own. They genuinely differ: K and V are
    whatever the caller allocated, and under MQA/GQA they carry `num_head_k`
    rather than `num_head_q`, so their head stride differs from Q's by
    construction. Assuming one shared layout is not a simplification, it is
    wrong.

    `hoist` and `clamp` are `const_expr` -- they select which code is emitted,
    not which branch runs.
    """
    # `row_off` is the varlen row offset, and it belongs in the
    # **64-bit base** rather than the 32-bit per-lane offset: on a
    # packed tensor it is a whole-batch quantity and overflows 32 bits
    # at realistic token counts (sdpa-varlen-plan.md section 5).
    s_batch, s_seq, s_head = strides
    bh = batch_index * s_batch + head * s_head + row_off * s_seq

    def tbase(seq_start):
        """Uniform 64-bit element base for (batch, head, seq_start).

        `seq_start` is a position on whichever sequence axis this
        tensor is indexed by -- rows for Q/O, KV columns for K/V --
        since `make_addr_pair` builds one of these per tensor.
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
        if not hoist:
            in_range = (ts + row_in_tile) < seqlen_k
            row = fx.Index(
                ArithValue(in_range).select(row_in_tile, seq_last - ts)
            )
            return toff(row, col)
        # `ts < seqlen_k` always -- it is either start_k, which the
        # caller's branch tested, or seq_last -- so this cannot wrap.
        in_range = row_in_tile < (seqlen_k - ts)
        return fx.Index(
            ArithValue(in_range).select(toff(row_in_tile, col), col)
        )

    def kv_addr(start_k, row_in_tile, col):
        """(uniform base, divergent offset) for a KV row, clamped in bounds.

        At K_PREFETCH_DIST == 1 the loop runs one tile ahead, so the final
        iteration addresses a tile past the end of the sequence; the unguarded
        cooperative load also addresses rows past BLOCK_N. Clamp start_k
        first, then send any row still past the end to the last row of the
        sequence. The values are never consumed; the clamp exists only so the
        address stays inside the allocation.

        With both prefetch distances 0 and no load guard there is no over-read
        -- BLOCK_N divides BLOCK_M and the tail is masked -- so `clamp` is
        false and this is pure VALU saved.
        """
        if not clamp:
            return tbase(start_k), toff(row_in_tile, col)
        ts = fx.Index(
            ArithValue(start_k < seqlen_k).select(start_k, seq_last)
        )
        return tbase(ts), kv_off(ts, row_in_tile, col)

    return tbase, toff, kv_addr
