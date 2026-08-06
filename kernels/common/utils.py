# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import ArithValue, _to_raw

# Pointer/global-load helpers now live in mem_ops; re-exported here for back-compat.
from kernels.common.mem_ops import extract_global_ptr as extract_global_ptr
from kernels.common.mem_ops import global_load as global_load
from kernels.common.mem_ops import global_load_i32 as global_load_i32
from kernels.common.mem_ops import global_load_i64x2 as global_load_i64x2
from kernels.common.mem_ops import global_ptr_from_addr as global_ptr_from_addr


def rcp_f32(value):
    return rocdl.rcp(T.f32, value)


def exp2_amdgcn_scalar(scalar_value):
    raw = (
        arith.unwrap(scalar_value)
        if hasattr(scalar_value, "ir_value") or hasattr(scalar_value, "type")
        else scalar_value
    )
    f32_ty = ir.F32Type.get()
    return llvm.call_intrinsic(f32_ty, "llvm.amdgcn.exp2.f32", [raw], [], [])


def exp2_f32_fast(value):
    from flydsl._mlir.dialects import vector as _vector_dialect

    raw = arith.unwrap(value) if hasattr(value, "ir_value") or hasattr(value, "type") else value
    ty = raw.type
    if isinstance(ty, ir.VectorType):
        vec = fx.Vector(raw)
        elems = [exp2_amdgcn_scalar(vec[i]) for i in range(ty.shape[0])]
        return _vector_dialect.from_elements(ty, elems)
    return exp2_amdgcn_scalar(raw)


def cdiv(numer: int, denom: int) -> int:
    return (numer + denom - 1) // denom


# Alias: several kernels historically spelled this ``ceildiv``.
ceildiv = cdiv


def align_up(value: int, align: int) -> int:
    """Round *value* up to the next multiple of *align* (static ints)."""
    return ((int(value) + int(align) - 1) // int(align)) * int(align)


def pow2_shift(value: int) -> int:
    assert value > 0 and (value & (value - 1)) == 0
    return value.bit_length() - 1


def is_pow2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def udiv_pow2(value, divisor: int):
    return value >> fx.Int32(pow2_shift(divisor))


def urem_pow2(value, divisor: int):
    return value & fx.Int32(divisor - 1)


def ssel(pred, a, b):
    """``pred ? a : b`` as an ``fx.Int32``.

    Thin, and the thinness is the point: ``ArithValue.select`` returns a raw
    MLIR value with no arithmetic overloads, so a caller writing
    ``ssel(c, x, y) + fx.Int32(1)`` needs the result typed. Without this the
    wrap gets open-coded at every call site, which is how the gfx1201 SDPA
    kernel and ``pa_metadata.py`` each ended up with their own copy.

    Operands are *not* coerced. Pass i32; passing an ``fx.Index`` gets you a
    64-bit unsigned select, which is a silent bug wherever the value can be
    negative.
    """
    return fx.Int32(ArithValue(pred).select(a, b))


def smin(a, b):
    """Signed minimum of two i32 values.

    Not `arith.minsi`-backed on purpose: this spells out the select so it
    composes with `ssel`'s typing rule, and so both operands stay visibly i32
    at the call site. Same operand contract as `ssel`.
    """
    return ssel((a < b), a, b)


def smax(a, b):
    """Signed maximum of two i32 values. See `smin`."""
    return ssel((a > b), a, b)


def sdiv_rd_pow2(value, divisor: int):
    """``floor(value / divisor)`` for a *signed* i32 and a power-of-two divisor.

    The signed counterpart to `udiv_pow2`, and the distinction is not
    cosmetic. An arithmetic right shift rounds toward negative infinity;
    `arith.divsi` truncates toward zero. They agree on non-negative input and
    differ on every negative one -- `floor(-1/32)` is -1, truncation gives 0.

    That case is reachable: it is how a sliding window whose left edge reaches
    past the start of the sequence gets turned into a tile index. Truncating
    there starts the left run one tile late and silently drops live columns.

    Written as an explicit `arith.shrsi` rather than `value >> n` so the
    signedness is visible at the definition, which is the entire reason this
    exists next to `udiv_pow2`.
    """
    assert is_pow2(divisor), f"sdiv_rd_pow2 needs a power-of-two divisor, got {divisor}"
    return fx.Int32(
        ArithValue(
            arith.shrsi(_to_raw(fx.Int32(value)), _to_raw(fx.Int32(pow2_shift(divisor))))
        )
    )


def udiv_const(value, divisor: int):
    if const_expr(is_pow2(divisor)):
        return udiv_pow2(value, divisor)
    return value // fx.Int32(divisor)


def urem_const(value, divisor: int):
    if const_expr(is_pow2(divisor)):
        return urem_pow2(value, divisor)
    return value % fx.Int32(divisor)


def unflatten_k(k_flat, qkhe_loop: int = 2):
    n = qkhe_loop * 2
    return [[k_flat[td * n + j] for j in range(n)] for td in range(len(k_flat) // n)]
