# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Philox 32- vs 64-bit lanes: randoms/second and register cost.

`PHILOX_WIDTH` is a per-arch default rather than a decision, because which
variant wins is a property of the target's integer ALU. This is what sets the
entry (`sdpa-dropout-plan.md` §4).

Two numbers, and the second is why this is a benchmark rather than an
instruction count:

- **randoms/second**, which the ALU cost dominates. The 32-bit variant needs
  one `v_mul_hi_u32` per high product; the 64-bit one needs a multi-instruction
  expansion.
- **VGPRs**, which decides whether the *caller* spills. Philox state is
  `c0..c3` plus `k0, k1`: six registers at 32-bit lanes, twelve at 64-bit,
  since each `u64` is a register pair. In a kernel already spilling at
  head_dim 192+ that difference is the whole question, and a benchmark with
  the register file to itself will not show it -- hence the attention-side
  check in the plan's step 3.

The 64-bit variant's case is call count, not arithmetic: it yields eight u32
per call against four, which is exactly the eight contiguous columns one
accumulator group covers. So the fair comparison is **per u32 produced**, and
that is what the ratio column reports.

    export ROCM_PATH=$(rocm-sdk path --root)
    python3 kernels/microbench/philox_bench.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "attention"))

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from flydsl.expr import gpu, range_constexpr  # noqa: E402

from bench_shim import do_bench  # noqa: E402
from philox import DEFAULT_ROUNDS, PHILOX_WIDTHS, Philox  # noqa: E402

BLOCKS = 4096
THREADS = 256
CALLS_PER_THREAD = 64


def build(width, n_rounds=DEFAULT_ROUNDS, calls=CALLS_PER_THREAD):
    """Sum `calls` calls' worth of randoms so nothing is dead code."""
    rng = Philox(width=width, n_rounds=n_rounds)
    rn = rng.randoms_per_offset

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k(OUT: fx.Pointer, seed: fx.Int64, base: fx.Int64):
        i32p = fx.PointerType.get(elem_ty=fx.Int32.ir_type,
                                  address_space=fx.AddressSpace.Global, alignment=4)
        tid = fx.Int64(fx.Int32(fx.Index(gpu.thread_idx.x)))
        blk = fx.Int64(fx.Int32(fx.Index(gpu.block_idx.x)))
        acc = fx.Int32(0)
        # Distinct offsets per call, as a real caller has: consecutive
        # elements walk consecutive offsets.
        for c in range_constexpr(calls):
            off = base + blk * fx.Int64(THREADS * calls) \
                + tid * fx.Int64(calls) + fx.Int64(c)
            vals = rng.u32(seed, off)
            for j in range_constexpr(rn):
                acc = acc ^ fx.Int32(vals[j])
        out = fx.recast_iter(i32p, OUT)
        fx.ptr_store(acc, out + fx.Int64(fx.Int32(blk) * fx.Int32(THREADS)
                                         + fx.Int32(fx.Index(gpu.thread_idx.x))))

    @flyc.jit
    def launch(OUT: fx.Pointer, seed: fx.Int64, base: fx.Int64,
               stream: fx.Stream = fx.Stream(None)):
        k(OUT, seed, base).launch(grid=(fx.Index(BLOCKS), 1, 1),
                                  block=(THREADS, 1, 1), stream=stream)

    return launch


def vgpr_count(width, calls=1):
    """VGPRs from the emitted ISA at a *small* unroll, or None.

    Measured at `calls=1` deliberately. The throughput run unrolls 64 calls to
    saturate the pipeline, which also saturates the register file and makes
    both widths report the 256 cap -- a benchmark artifact, not a property.
    One call shows the state cost: `c0..c3` plus `k0, k1`, which is six
    registers at 32-bit lanes and twelve at 64-bit.
    """
    import glob
    import re
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=d,
                   FLYDSL_RUNTIME_ENABLE_CACHE="0")
        subprocess.run([sys.executable, __file__, "--dump-only", str(width), str(calls)],
                       env=env, capture_output=True, timeout=600)
        for f in glob.glob(os.path.join(d, "*", "21_final_isa.s")):
            m = re.search(r"\.vgpr_count:\s*(\d+)", open(f).read())
            if m:
                return int(m.group(1))
    return None


def run_one(width):
    launch = build(width)
    out = torch.zeros(BLOCKS * THREADS, dtype=torch.int32, device="cuda")
    p = flyc.from_c_void_p(fx.Uint8, out.data_ptr())
    args = (p, 0x1234_5678_9ABC_DEF0, 0)
    exe = flyc.compile(launch, *args, fx.Stream(None))

    def fn():
        exe(*args, fx.Stream(None))

    fn()
    torch.cuda.synchronize()
    ms = do_bench(fn, warmup=25, rep=100, return_mode="median")
    n = BLOCKS * THREADS * CALLS_PER_THREAD * Philox(width=width).randoms_per_offset
    return ms, n / ms * 1e3 / 1e9      # G randoms/s


def main():
    if "--dump-only" in sys.argv:
        w, calls = int(sys.argv[-2]), int(sys.argv[-1])
        launch = build(w, calls=calls)
        out = torch.zeros(BLOCKS * THREADS, dtype=torch.int32, device="cuda")
        p = flyc.from_c_void_p(fx.Uint8, out.data_ptr())
        flyc.compile(launch, p, 1, 0, fx.Stream(None))
        return 0

    print(f"Philox, {BLOCKS} blocks x {THREADS} threads x {CALLS_PER_THREAD} calls, "
          f"{DEFAULT_ROUNDS} rounds\n")
    print(f"{'width':>6} {'u32/call':>9} {'ms':>8} {'G rand/s':>10} {'VGPR@1call':>11}")
    res = {}
    for w in PHILOX_WIDTHS:
        ms, grs = run_one(w)
        v = vgpr_count(w)
        res[w] = (grs, v)
        print(f"{w:>6} {Philox(width=w).randoms_per_offset:>9} {ms:>8.3f} {grs:>10.2f} "
              f"{(v if v is not None else '?'):>11}")
    a, b = res[32][0], res[64][0]
    print(f"\n32-bit is {a / b:.2f}x the throughput of 64-bit per random")
    print(f"VGPR delta (64 - 32): "
          f"{res[64][1] - res[32][1] if None not in (res[32][1], res[64][1]) else '?'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
