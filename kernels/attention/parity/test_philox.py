# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for `philox.py`, with no attention in the picture.

Three references, because they fail differently:

- **Random123's published known-answer vectors** for Philox-4x32-10. These
  pin the *algorithm* against its own specification, and they are the only
  check here that does not depend on anyone's reading of anyone's source.
- **A CPU model** in plain numpy, which extends that to the packing this
  module does (64-bit seed and offset into key and counter words) and to
  width 64, where no public vectors are to hand.
- **Triton itself**, which is the contract: callers compare dropout against
  seeded `torch` runs, so "is it a good PRNG" is not the question -- "is it
  *this* PRNG" is.

The three overlap deliberately. The CPU model alone cannot catch a *misreading*
of Triton, since it is written from the same reading; the KAT vectors cannot
catch a wrong packing, since they test the raw round function. Together they
leave only a narrow gap, and §`test_matches_triton` closes it where Triton can
be built.
"""

import numpy as np
import pytest
import torch
from philox import DEFAULT_ROUNDS, PHILOX_WIDTHS, philox_u32, randoms_per_offset

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

pytestmark = pytest.mark.parametrize("width", PHILOX_WIDTHS, ids=["u32", "u64"])

_ARCH_OK = torch.cuda.is_available()
_SKIP = pytest.mark.skipif(not _ARCH_OK, reason="requires a GPU")


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------

_C = {
    32: dict(KEY_A=0x9E3779B9, KEY_B=0xBB67AE85, MUL_A=0xD2511F53, MUL_B=0xCD9E8D57, MASK=(1 << 32) - 1, BITS=32),
    64: dict(
        KEY_A=0x9E3779B97F4A7C15,
        KEY_B=0xBB67AE8584CAA73B,
        MUL_A=0xD2E7470EE14C6C93,
        MUL_B=0xCA5A826395121157,
        MASK=(1 << 64) - 1,
        BITS=64,
    ),
}


def _ref_u32(seed, offset, width, n_rounds=DEFAULT_ROUNDS):
    """Philox in Python ints, returning the u32 values the module should give."""
    c = _C[width]
    M, BITS = c["MASK"], c["BITS"]
    if width == 32:
        c0, c1, c2, c3 = offset & M, (offset >> 32) & M, 0, 0
        k0, k1 = seed & M, (seed >> 32) & M
    else:
        c0, c1, c2, c3 = offset & M, 0, 0, 0
        k0, k1 = seed & M, 0
    A, B = c["MUL_A"], c["MUL_B"]
    for _ in range(n_rounds):
        _c0, _c2 = c0, c2
        c0 = (((B * _c2) >> BITS) ^ c1 ^ k0) & M
        c2 = (((A * _c0) >> BITS) ^ c3 ^ k1) & M
        c1 = (B * _c2) & M
        c3 = (A * _c0) & M
        k0 = (k0 + c["KEY_A"]) & M
        k1 = (k1 + c["KEY_B"]) & M
    words = [c0, c1, c2, c3]
    if width == 32:
        return words
    out = []
    for w in words:
        out.extend((w & 0xFFFFFFFF, (w >> 32) & 0xFFFFFFFF))
    return out


# ---------------------------------------------------------------------------
# Device harness
# ---------------------------------------------------------------------------


def _run_device(seeds, offsets, width, n_rounds=DEFAULT_ROUNDS):
    """One thread per (seed, offset) pair; returns an (n, RN) uint32 array."""
    rn = randoms_per_offset(width)
    n = len(seeds)

    @flyc.kernel(known_block_size=[64, 1, 1])
    def k(SEED: fx.Pointer, OFF: fx.Pointer, OUT: fx.Pointer):
        i64p = fx.PointerType.get(elem_ty=fx.Int64.ir_type, address_space=fx.AddressSpace.Global, alignment=8)
        i32p = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        z = fx.Int32(fx.Index(gpu.block_idx.x))
        seed = fx.ptr_load(fx.recast_iter(i64p, SEED) + fx.Int64(z))
        off = fx.ptr_load(fx.recast_iter(i64p, OFF) + fx.Int64(z))
        vals = philox_u32(seed, off, width, n_rounds)
        out = fx.recast_iter(i32p, OUT)
        # `range_constexpr`, not `range`: inside a traced kernel the latter is
        # FlyDSL's runtime loop and its induction variable cannot index a
        # Python list.
        for j in range_constexpr(rn):
            fx.ptr_store(fx.Int32(vals[j]), out + fx.Int64(z * fx.Int32(rn) + fx.Int32(j)))

    @flyc.jit
    def launch(SEED: fx.Pointer, OFF: fx.Pointer, OUT: fx.Pointer, stream: fx.Stream = fx.Stream(None)):
        k(SEED, OFF, OUT).launch(grid=(fx.Index(n), 1, 1), block=(64, 1, 1), stream=stream)

    ts = torch.tensor(seeds, dtype=torch.int64, device="cuda")
    to = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    out = torch.zeros(n * rn, dtype=torch.int32, device="cuda")
    p = lambda t: flyc.from_c_void_p(fx.Uint8, t.data_ptr())  # noqa: E731
    exe = flyc.compile(launch, p(ts), p(to), p(out), fx.Stream(None))
    exe(p(ts), p(to), p(out), fx.Stream(None))
    torch.cuda.synchronize()
    return out.cpu().numpy().astype(np.uint32).reshape(n, rn)


_CASES = [
    (0, 0),
    (1, 0),
    (0, 1),
    (12345, 67890),
    (0xDEADBEEF, 0xCAFEBABE),
    # Above 2**32 on each side independently. A build that truncates either to
    # 32 bits still produces a perfectly random-looking stream, so nothing but
    # an exact comparison catches it.
    (1 << 32, 7),
    (7, 1 << 32),
    ((1 << 40) + 12345, (1 << 35) + 999),
    ((1 << 63) - 1, (1 << 63) - 1),
]


@_SKIP
def test_matches_cpu_reference(width):
    """Bit-exact against a Python model of the algorithm."""
    seeds = [s for s, _ in _CASES]
    offs = [o for _, o in _CASES]
    got = _run_device(seeds, offs, width)
    for i, (s, o) in enumerate(_CASES):
        want = _ref_u32(s, o, width)
        assert list(got[i]) == want, (
            f"width={width} seed={s:#x} offset={o:#x}\n  got  {list(got[i])}\n" f"  want {want}"
        )


@_SKIP
def test_seed_and_offset_are_really_64_bit(width):
    """Truncating either to 32 bits must change the stream.

    This is the row the rest of the suite cannot cover: a 32-bit truncation is
    self-consistent, uniform, and passes every distributional check. Only
    comparing two values that differ *above* bit 32 detects it.
    """
    pairs = [(1 << 32, 5), (0, 5), (5, 1 << 32), (5, 0)]
    got = _run_device([s for s, _ in pairs], [o for _, o in pairs], width)
    assert list(got[0]) != list(got[1]), "seed above 2**32 was truncated"
    assert list(got[2]) != list(got[3]), "offset above 2**32 was truncated"


@_SKIP
def test_rounds_are_applied(width):
    """`n_rounds` must reach the loop rather than being ignored."""
    a = _run_device([12345], [678], width, n_rounds=DEFAULT_ROUNDS)
    b = _run_device([12345], [678], width, n_rounds=DEFAULT_ROUNDS - 3)
    assert list(a[0]) != list(b[0])
    assert list(b[0]) == _ref_u32(12345, 678, width, DEFAULT_ROUNDS - 3)


# Random123 `kat_vectors`, philox4x32 at 10 rounds. Independent of Triton and
# of this module: they come from the algorithm's reference implementation.
_KAT_4X32_10 = [
    ([0, 0, 0, 0], [0, 0], [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]),
    ([0xFFFFFFFF] * 4, [0xFFFFFFFF] * 2, [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD]),
    (
        [0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344],
        [0xA4093822, 0x299F31D0],
        [0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1],
    ),
]


@_SKIP
def test_matches_random123_known_answers(width):
    """The round function against the algorithm's own published vectors.

    Runs on the device through `philox_4x` directly, bypassing the seed/offset
    packing so that only the round function is under test -- which is exactly
    what the published vectors specify. Width 64 has no public vectors, so it
    checks the 32-bit core the module shares.
    """
    if width == 64:
        pytest.skip("Random123 publishes no 4x64-10 vectors in this form")
    from philox import philox_4x

    n = len(_KAT_4X32_10)

    @flyc.kernel(known_block_size=[64, 1, 1])
    def k(IN: fx.Pointer, OUT: fx.Pointer):
        i32p = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        z = fx.Int32(fx.Index(gpu.block_idx.x))
        src = fx.recast_iter(i32p, IN)
        w = [fx.Int32(fx.ptr_load(src + fx.Int64(z * fx.Int32(6) + fx.Int32(j)))) for j in range_constexpr(6)]
        r = philox_4x(w[0], w[1], w[2], w[3], w[4], w[5], 32, DEFAULT_ROUNDS)
        out = fx.recast_iter(i32p, OUT)
        for j in range_constexpr(4):
            fx.ptr_store(fx.Int32(r[j]), out + fx.Int64(z * fx.Int32(4) + fx.Int32(j)))

    @flyc.jit
    def launch(IN: fx.Pointer, OUT: fx.Pointer, stream: fx.Stream = fx.Stream(None)):
        k(IN, OUT).launch(grid=(fx.Index(n), 1, 1), block=(64, 1, 1), stream=stream)

    flat = []
    for ctr, key, _ in _KAT_4X32_10:
        flat.extend(ctr + key)
    ti = torch.tensor(np.array(flat, dtype=np.uint32).astype(np.int32), dtype=torch.int32, device="cuda")
    out = torch.zeros(n * 4, dtype=torch.int32, device="cuda")
    p = lambda t: flyc.from_c_void_p(fx.Uint8, t.data_ptr())  # noqa: E731
    exe = flyc.compile(launch, p(ti), p(out), fx.Stream(None))
    exe(p(ti), p(out), fx.Stream(None))
    torch.cuda.synchronize()
    got = out.cpu().numpy().astype(np.uint32).reshape(n, 4)
    for i, (ctr, key, want) in enumerate(_KAT_4X32_10):
        assert list(got[i]) == want, f"KAT {i}: got {[hex(x) for x in got[i]]} want {[hex(x) for x in want]}"


@_SKIP
def test_matches_triton(width):
    """The contract: our stream is *Triton's* stream, not merely a Philox.

    Skipped at width 64 -- `tl.randint4x` returns uint32 on this Triton build
    (see the static_assert in AOTriton's dropout.py, which pins the same
    behaviour), so there is no 64-bit lane path to compare against. The CPU
    reference still covers that width.
    """
    if width == 64:
        pytest.skip("tl.randint4x has no 64-bit lane variant to compare against")
    triton = pytest.importorskip("triton")
    tl = pytest.importorskip("triton.language")
    import os
    import sysconfig

    if not os.path.isfile(os.path.join(sysconfig.get_paths()["include"], "Python.h")):
        # Triton's HIP backend compiles a utility module at import time.
        # Without the dev headers it cannot, which is an environment gap
        # rather than a result. The Random123 vectors above cover the
        # algorithm and the CPU model covers the packing; what is lost is
        # only the check against a *misreading* of Triton's source.
        pytest.skip("python dev headers absent; triton cannot build its HIP utils")

    @triton.jit
    def ref_kernel(SEED, OFF, OUT, N: tl.constexpr):
        i = tl.program_id(0)
        seed = tl.load(SEED + i)
        off = tl.load(OFF + i)
        r0, r1, r2, r3 = tl.randint4x(seed, off)
        tl.store(OUT + i * 4 + 0, r0.to(tl.int32, bitcast=True))
        tl.store(OUT + i * 4 + 1, r1.to(tl.int32, bitcast=True))
        tl.store(OUT + i * 4 + 2, r2.to(tl.int32, bitcast=True))
        tl.store(OUT + i * 4 + 3, r3.to(tl.int32, bitcast=True))

    seeds = [s for s, _ in _CASES]
    offs = [o for _, o in _CASES]
    ts = torch.tensor(seeds, dtype=torch.int64, device="cuda")
    to = torch.tensor(offs, dtype=torch.int64, device="cuda")
    ref = torch.zeros(len(seeds) * 4, dtype=torch.int32, device="cuda")
    ref_kernel[(len(seeds),)](ts, to, ref, 4)
    torch.cuda.synchronize()
    want = ref.cpu().numpy().astype(np.uint32).reshape(len(seeds), 4)
    got = _run_device(seeds, offs, width)
    for i, (s, o) in enumerate(_CASES):
        assert list(got[i]) == list(want[i]), (
            f"diverged from Triton at seed={s:#x} offset={o:#x}\n" f"  ours   {list(got[i])}\n  triton {list(want[i])}"
        )


@_SKIP
def test_configured_object_matches_free_functions(width):
    """`Philox` must be a wrapper, not a second implementation.

    It exists so that width and round count travel together -- forward,
    backward and the mask kernel have to agree bit for bit -- which is only
    worth anything if it produces the same stream as the functions it wraps.
    """
    from philox import Philox

    rng = Philox(width=width)
    assert rng.randoms_per_offset == randoms_per_offset(width)
    seeds = [s for s, _ in _CASES]
    offs = [o for _, o in _CASES]
    direct = _run_device(seeds, offs, width)
    for i, (s, o) in enumerate(_CASES):
        assert list(direct[i]) == _ref_u32(s, o, width), "free function drifted"


def test_configured_object_validates(width):
    """Bad configurations fail at construction, not at trace time."""
    from philox import Philox

    with pytest.raises(ValueError, match="PHILOX_WIDTH"):
        Philox(width=48)
    with pytest.raises(ValueError, match="n_rounds"):
        Philox(width=width, n_rounds=0)


# ---------------------------------------------------------------------------
# Dropout helpers
# ---------------------------------------------------------------------------


def test_dropout_threshold_keeps_the_right_fraction(width):
    """`p` in, an i32 threshold out, keeping `1 - p` of a uniform u32 stream."""
    from philox import dropout_threshold

    for p in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        t = dropout_threshold(p)
        kept = (2**31 - 1 - t) / 2**32
        assert abs(kept - (1.0 - p)) < 1e-3, f"p={p} keeps {kept:.4f}"
    assert dropout_threshold(0.0) < -(2**31) + 2, "p=0 must keep everything"
    assert dropout_threshold(1.0) > 2**31 - 2, "p=1 must keep nothing"
    with pytest.raises(ValueError):
        dropout_threshold(1.5)


@_SKIP
def test_keep_mask_compares_signed(width):
    """The trap this helper exists to close.

    `fx.Int32`'s `>` overload is **unsigned**, and `dropout_threshold(p)` is
    negative for every `p < 0.5` -- the common case. Comparing unsigned there
    keeps *everything*, which is a dropout layer that silently does nothing and
    passes any test that only checks the output is finite.

    Driven with hand-picked values rather than randoms so the expected answer
    is known exactly.
    """
    from philox import dropout_threshold, keep_mask

    thr = dropout_threshold(0.25)  # negative
    assert thr < 0
    # as i32: -2**31 (lowest) must drop, +2**31-1 (highest) must keep
    vals_i32 = [-(2**31), thr - 1, thr, thr + 1, 2**31 - 1]
    want = [v > thr for v in vals_i32]
    assert want == [False, False, False, True, True], "test's own model is wrong"

    n = len(vals_i32)

    @flyc.kernel(known_block_size=[64, 1, 1])
    def k(IN: fx.Pointer, OUT: fx.Pointer, threshold: fx.Int32):
        i32p = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        src = fx.recast_iter(i32p, IN)
        dst = fx.recast_iter(i32p, OUT)
        z = fx.Int32(fx.Index(gpu.block_idx.x))
        v = fx.Int32(fx.ptr_load(src + fx.Int64(z)))
        keep = keep_mask([v], threshold)[0]
        fx.ptr_store(fx.Int32(keep.select(fx.Int32(1), fx.Int32(0))), dst + fx.Int64(z))

    @flyc.jit
    def launch(IN: fx.Pointer, OUT: fx.Pointer, threshold: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        k(IN, OUT, threshold).launch(grid=(fx.Index(n), 1, 1), block=(64, 1, 1), stream=stream)

    ti = torch.tensor(vals_i32, dtype=torch.int32, device="cuda")
    out = torch.zeros(n, dtype=torch.int32, device="cuda")
    p_ = lambda t: flyc.from_c_void_p(fx.Uint8, t.data_ptr())  # noqa: E731
    exe = flyc.compile(launch, p_(ti), p_(out), thr, fx.Stream(None))
    exe(p_(ti), p_(out), thr, fx.Stream(None))
    torch.cuda.synchronize()
    got = [bool(x) for x in out.cpu().tolist()]
    assert got == want, f"signed compare failed: got {got}, want {want}"


@_SKIP
def test_span_is_the_stream_read_consecutively(width):
    """`span_u32` must equal the individual calls it stands for.

    The blocked form exists for the tile case; it is only useful if it is the
    same stream. A span that advanced an internal counter would agree here and
    diverge at the wrap -- which is why it takes an absolute offset.
    """
    from philox import Philox

    rng = Philox(width=width)
    rn = rng.randoms_per_offset
    seed, off0 = 0xFEED_FACE_1234, (1 << 33) + 17
    want = []
    for k in range(3):
        want.extend(_ref_u32(seed, off0 + k, width))
    got = _run_span(seed, off0, 3 * rn, width)
    assert list(got) == want


def _run_span(seed, first_offset, count, width):
    from philox import Philox

    rng = Philox(width=width)

    @flyc.kernel(known_block_size=[64, 1, 1])
    def k(OUT: fx.Pointer, s: fx.Int64, o: fx.Int64):
        i32p = fx.PointerType.get(elem_ty=fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=4)
        vals = rng.span_u32(s, o, count)
        dst = fx.recast_iter(i32p, OUT)
        for j in range_constexpr(count):
            fx.ptr_store(fx.Int32(vals[j]), dst + fx.Int64(j))

    @flyc.jit
    def launch(OUT: fx.Pointer, s: fx.Int64, o: fx.Int64, stream: fx.Stream = fx.Stream(None)):
        k(OUT, s, o).launch(grid=(fx.Index(1), 1, 1), block=(64, 1, 1), stream=stream)

    out = torch.zeros(count, dtype=torch.int32, device="cuda")
    p_ = lambda t: flyc.from_c_void_p(fx.Uint8, t.data_ptr())  # noqa: E731
    exe = flyc.compile(launch, p_(out), seed, first_offset, fx.Stream(None))
    exe(p_(out), seed, first_offset, fx.Stream(None))
    torch.cuda.synchronize()
    return out.cpu().numpy().astype(np.uint32)


def test_span_rejects_a_partial_call(width):
    """A count that is not a whole number of calls is a caller error."""
    from philox import Philox

    rng = Philox(width=width)
    with pytest.raises(ValueError, match="multiple of randoms_per_offset"):
        rng.span_u32(0, 0, rng.randoms_per_offset + 1)


@_SKIP
def test_distribution_is_plausible(width):
    """Weak by construction, kept because it catches a stuck word.

    An exact-match suite passes if every word is identical to a reference that
    is *also* wrong in the same way. This does not prove randomness; it
    notices a word that never changes.
    """
    n = 4096
    got = _run_device(list(range(n)), [0] * n, width).astype(np.float64) / 2**32
    assert 0.45 < got.mean() < 0.55, f"mean {got.mean():.4f}"
    for j in range(got.shape[1]):
        assert got[:, j].std() > 0.2, f"word {j} looks stuck"
