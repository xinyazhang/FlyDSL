# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Helpers shared by the FlyDSL attention kernels.

Attention-specific, and deliberately not in `kernels/common/`: what is here
either encodes an attention ABI decision or has no meaning outside one. The
general-purpose siblings live next door -- `kernels/common/mem_ops.py` for
pointer and global load/store, `kernels/common/utils.py` for the scalar
integer helpers.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith
from flydsl.expr.typing import T

__all__ = ["llvm_ptr_ty", "pointer_to_llvm_ptr"]


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
