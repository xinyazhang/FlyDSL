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

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import ArithValue, _to_raw

__all__ = ["llvm_ptr_ty", "pointer_to_llvm_ptr", "lds_load_v8", "lds_store_vx", "global_load_tr_v8",
           "bitcast_i32", "pack_bf16_pair", "bf16_trunc_pack_v8",
           "FastMath", "MaskedAxis",
           "lds_f32_ptr", "lds_f32_store", "lds_f32_load",
           "reduce_s_across_shards"]


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
