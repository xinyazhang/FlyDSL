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
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith, range_constexpr
from flydsl.expr.typing import T, Vector as Vec

__all__ = ["llvm_ptr_ty", "pointer_to_llvm_ptr", "lds_load_v8", "lds_store_vx"]


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
