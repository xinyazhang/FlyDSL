# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Philox-4x{32,64}-N counter-based PRNG for FlyDSL kernels.

A counter-based PRNG is a pure function of ``(key, counter)`` with no state and
no sequence: every element derives its own random from its own coordinates,
with no ordering between them. That is what a GPU needs, and it is why the
backward pass can regenerate a dropout mask from ``(seed, offset)`` alone
rather than storing it -- a 4K x 4K mask is 268 MB per head.

**This module is deliberately arch-agnostic and layout-agnostic.** It maps
``(seed, offset) -> N randoms`` and nothing else. How a caller assigns offsets
to elements is the caller's business; there are no wave-size assumptions, no
target intrinsics, and no attention-specific code. It is written to the
standard a FlyDSL library needs because that is where it is going.

Matches Triton's ``triton.language.random`` bit for bit -- see
``test_philox.py``, which checks against both a CPU reference and Triton
itself. Callers compare dropout against seeded ``torch`` runs, so the stream is
part of the contract, not an implementation detail.

Two widths
----------
``PHILOX_WIDTH`` selects 32- or 64-bit lanes. They are *different PRNGs* and
produce different streams -- as any two PRNGs do -- so the width is part of
whatever contract a caller builds on top:

    width  lanes    randoms per call   state
    32     u32      4 x u32            6 registers
    64     u64      4 x u64 = 8 x u32  12 registers

Which is faster is a property of the target's integer ALU, so it is a
parameter here and a per-arch default in the caller. On gfx1201 the 32-bit
variant needs one ``v_mul_hi_u32`` per high product while the 64-bit one needs
a multi-instruction sequence; see ``kernels/microbench/philox_bench.py``.

Both widths take a **64-bit seed and a 64-bit offset**. That does not require
64-bit arithmetic: at width 32 the seed fills the two key words and the offset
the low two counter words. Splitting a 64-bit value costs nothing -- it lives
in two consecutive 32-bit registers, so the halves are addressable directly.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir.dialects import arith
from flydsl.expr.utils.arith import ArithValue, _to_raw as _raw

__all__ = [
    "PHILOX_WIDTHS",
    "DEFAULT_ROUNDS",
    "default_width",
    "randoms_per_offset",
    "philox_4x",
    "philox_u32",
]

PHILOX_WIDTHS = (32, 64)
DEFAULT_ROUNDS = 10

# Per-arch lane width. **Changing an entry changes every stream derived from
# it**, and therefore every dropout mask -- unlike a tuning table, where a
# re-sweep is invisible. Version it and record why, rather than re-measuring
# and overwriting.
#
# gfx1201, measured by `kernels/microbench/philox_bench.py`:
#
#   width  u32/call   G randoms/s   VGPRs (one call)
#   32     4          284.5         8
#   64     8           65.7         26
#
# 32-bit is **4.3x** the throughput per random and costs 18 fewer registers.
# Both margins are wider than the instruction counts predicted (~2x): RDNA4
# has `v_mul_hi_u32` as a single instruction, while the 64-bit variant needs
# the *high* half of a 64x64 product, whose expansion is worse than the low
# half that was counted. The register gap is the six-vs-twelve state words
# plus the eight-vs-four live outputs.
#
# So the 64-bit variant's one advantage -- eight randoms per call, matching an
# eight-column accumulator group exactly -- does not come close to paying for
# itself here. It is kept because that trade will look different on a target
# with a native 64-bit multiplier.
_WIDTH_BY_ARCH = {
    "gfx1201": 32,
}
_FALLBACK_WIDTH = 32


def default_width(arch: str | None = None) -> int:
    """Lane width for `arch`, defaulting to 32 where nothing was measured.

    32 is the conservative default: it is the cheaper variant on every target
    without a native 64-bit integer multiplier, which is most of them.
    """
    if arch is None:
        from flydsl.runtime.device import get_rocm_arch

        arch = get_rocm_arch()
    base = str(arch).split(":")[0]
    return _WIDTH_BY_ARCH.get(base, _FALLBACK_WIDTH)

# Weyl-sequence key increments and round multipliers. The 32-bit values are the
# original Philox-4x32 constants (golden ratio / sqrt(3) fractions); the 64-bit
# ones are their 64-bit counterparts. Both match Triton's table.
_CONSTS = {
    32: dict(KEY_A=0x9E3779B9, KEY_B=0xBB67AE85,
             MUL_A=0xD2511F53, MUL_B=0xCD9E8D57),
    64: dict(KEY_A=0x9E3779B97F4A7C15, KEY_B=0xBB67AE8584CAA73B,
             MUL_A=0xD2E7470EE14C6C93, MUL_B=0xCA5A826395121157),
}


def randoms_per_offset(width: int) -> int:
    """u32 values one ``philox_4x`` call yields. Part of a caller's contract.

    It appears in offset arithmetic (``n // RN``), so changing it changes every
    derived stream -- which is why callers pin it rather than infer it.
    """
    _check_width(width)
    return 4 if width == 32 else 8


def _check_width(width: int) -> None:
    if width not in PHILOX_WIDTHS:
        raise ValueError(f"PHILOX_WIDTH must be one of {PHILOX_WIDTHS}, got {width}")


def _ty(width: int):
    return fx.Int32 if width == 32 else fx.Int64


def _const(width: int, name: str):
    return _ty(width)(_CONSTS[width][name])


def _mul_lo(a, b, width: int):
    return _ty(width)(ArithValue(arith.muli(_raw(a), _raw(b))))


def _mul_hi(a, b, width: int):
    """High half of an unsigned product.

    `arith.mului_extended` yields both halves in one op, which lowers to
    `v_mul_hi_u32` on RDNA4 rather than a shift of a widened product. At width
    64 the backend expands it; that expansion is exactly what makes the 64-bit
    variant expensive, and is the thing the microbenchmark prices.
    """
    op = arith.MulUIExtendedOp(_raw(a), _raw(b))
    return _ty(width)(ArithValue(op.high))


def _xor(a, b, width: int):
    return _ty(width)(ArithValue(arith.xori(_raw(a), _raw(b))))


def _add(a, b, width: int):
    return _ty(width)(ArithValue(arith.addi(_raw(a), _raw(b))))


def philox_4x(c0, c1, c2, c3, k0, k1, width: int = 32, n_rounds: int = DEFAULT_ROUNDS):
    """`n_rounds` Philox rounds over counter `(c0..c3)` and key `(k0, k1)`.

    Returns the four counter words, which are the random output. Operands must
    already be `fx.Int32` at width 32 or `fx.Int64` at width 64.

    The round is Triton's verbatim, and the ordering matters: `c1` and `c3` are
    computed from the *pre-round* `_c0`/`_c2`, so the temporaries are not an
    optimisation to remove.
    """
    _check_width(width)
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")
    A, B = _const(width, "MUL_A"), _const(width, "MUL_B")
    KA, KB = _const(width, "KEY_A"), _const(width, "KEY_B")
    for _ in range(n_rounds):
        _c0, _c2 = c0, c2
        c0 = _xor(_xor(_mul_hi(B, _c2, width), c1, width), k0, width)
        c2 = _xor(_xor(_mul_hi(A, _c0, width), c3, width), k1, width)
        c1 = _mul_lo(B, _c2, width)
        c3 = _mul_lo(A, _c0, width)
        k0 = _add(k0, KA, width)
        k1 = _add(k1, KB, width)
    return c0, c1, c2, c3


def philox_u32(seed, offset, width: int = 32, n_rounds: int = DEFAULT_ROUNDS):
    """`randoms_per_offset(width)` uniform u32 values from a 64-bit `(seed, offset)`.

    `seed` and `offset` are `fx.Int64`. This is the entry point callers want:
    it handles the width-dependent packing so they do not have to.

    At width 32 the seed fills the key and the offset the low counter words,
    which is how a 64-bit seed and offset survive 32-bit arithmetic. The
    splitting is free -- a 64-bit value occupies two consecutive 32-bit
    registers, so `trunc` and `shr 32; trunc` lower to register naming.

    At width 64 the whole seed and offset go in one word each, and the four
    u64 outputs are unpacked into eight u32 **low half first**, matching
    Triton's `join(hi, lo)` ordering.
    """
    _check_width(width)
    seed64, off64 = fx.Int64(seed), fx.Int64(offset)
    if width == 32:
        lo, hi = _split64(off64)
        klo, khi = _split64(seed64)
        zero = fx.Int32(0)
        return list(philox_4x(lo, hi, zero, zero, klo, khi, 32, n_rounds))
    words = philox_4x(off64, fx.Int64(0), fx.Int64(0), fx.Int64(0),
                      seed64, fx.Int64(0), 64, n_rounds)
    out = []
    for w in words:
        w_lo, w_hi = _split64(w)
        out.extend((w_lo, w_hi))
    return out


def _split64(v):
    """(low 32, high 32) of a 64-bit value, as `fx.Int32`.

    Free on GPU targets: a 64-bit value is a pair of 32-bit registers, so this
    lowers to register naming rather than a shift and a mask.
    """
    v = fx.Int64(v)
    lo = fx.Int32(ArithValue(arith.trunci(fx.Int32.ir_type, _raw(v))))
    hi = fx.Int32(
        ArithValue(
            arith.trunci(
                fx.Int32.ir_type,
                _raw(ArithValue(arith.shrui(_raw(v), _raw(fx.Int64(32))))),
            )
        )
    )
    return lo, hi
