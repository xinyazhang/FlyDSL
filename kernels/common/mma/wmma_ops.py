# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""WMMA accumulate helpers, dispatched on the operand type.

The RDNA WMMA intrinsics are one entry point per input dtype
(`rocdl.wmma_f32_16x16x16_f16`, `..._bf16`, ...), and the bf16 one takes its
operands as `i16` vectors rather than `bf16` vectors. A caller that wants to be
generic over dtype therefore has to branch and bitcast, and every caller that
does so open-codes the same two lines.

**This dispatches on the operands, not on a build-time dtype string.** That
distinction is the reason this is worth a module: a helper keyed on a
`dtype_str` in the caller's scope can only be called from a place that has one,
which rules out shared code -- notably a backward kernel that receives vectors
and has no build flag to consult. The element type is already carried by the
value; reading it there is both more general and impossible to get out of sync
with the data.

Naming follows the neighbouring `mfma_epilogues.py` / `mfma_preshuffle_pipeline.py`.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import rocdl
from flydsl.expr.typing import Vector
from flydsl.expr.utils.arith import _to_raw as _raw

__all__ = ["wmma_f32_16x16x16", "vector_elem_type"]


def vector_elem_type(value) -> ir.Type:
    """Element type of a vector-typed value, raw or DSL-wrapped.

    `ir.VectorType.isinstance` does not exist in this binding; Python
    `isinstance` against the downcast class is what works.
    """
    ty = _raw(value).type
    if not isinstance(ty, ir.VectorType):
        raise TypeError(f"expected a vector value, got {ty}")
    return ir.VectorType(ty).element_type


def wmma_f32_16x16x16(a, b, acc, acc_type: ir.Type | None = None):
    """One 16x16x16 WMMA into an f32 accumulator: `acc += a @ b`.

    `a` and `b` are 8-lane f16 or bf16 vectors and must agree; `acc` is an
    8-lane f32 vector. `acc_type` defaults to `acc`'s own type and only needs
    passing when `acc` is a raw value whose type the caller wants to override.

    bf16 operands are bitcast to `i16` because that is the ABI the intrinsic
    takes -- a reinterpretation, not a conversion, so it costs nothing and
    changes no bits.
    """
    a_ty, b_ty = vector_elem_type(a), vector_elem_type(b)
    if a_ty != b_ty:
        raise TypeError(f"WMMA operands must share an element type, got {a_ty} and {b_ty}")
    res_ty = acc_type if acc_type is not None else _raw(acc).type

    if isinstance(a_ty, ir.BF16Type):
        a16 = _raw(Vector(_raw(a)).bitcast(fx.Int16))
        b16 = _raw(Vector(_raw(b)).bitcast(fx.Int16))
        return rocdl.wmma_f32_16x16x16_bf16(res_ty, a16, b16, _raw(acc)).result
    if isinstance(a_ty, ir.F16Type):
        return rocdl.wmma_f32_16x16x16_f16(res_ty, _raw(a), _raw(b), _raw(acc)).result
    raise TypeError(f"wmma_f32_16x16x16 supports f16 and bf16 operands, got {a_ty}")
