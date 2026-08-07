# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Sustained dense WMMA throughput -- the matrix-pipeline ceiling for a part.

Measures nothing but the WMMA issue rate: operands live in registers,
accumulators are loop-carried, and the loop body contains no memory traffic. Use
it to express a real kernel's throughput as a fraction of what the hardware can
actually reach, rather than against a spec sheet.

``nacc`` independent accumulator chains are advanced per iteration. With
``nacc=1`` consecutive WMMAs are serially dependent, so the result is the
*latency* bound; raising ``nacc`` until throughput plateaus reveals the pipeline
depth and the sustained peak.

Self-checking: with all-ones operands every WMMA adds exactly ``K`` per output
element, so the final accumulator must equal ``K * iters * nacc``. That catches
a chain being dead-code eliminated or a loop being collapsed -- failure modes
that would otherwise show up as an implausibly high TFLOPS number.

Run directly (no PYTHONPATH needed -- this module imports only flydsl/torch)::

    export ROCM_PATH=$(rocm-sdk path --root)   # only for a pip-installed ROCm
    python3 kernels/microbench/wmma_peak.py

Measured on Radeon AI PRO R9700 (gfx1201, 64 CU), 2026-07: ~205 TFLOPS for both
f16 and bf16, stable across grid 256-2048 and iters 500-8000.
"""

import argparse

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

WARP_SIZE = 32
BLOCK = 256
WAVES_PER_WG = BLOCK // WARP_SIZE

# v_wmma_f32_16x16x16_* computes a full 16x16x16 matmul per wave instruction.
WMMA_M = WMMA_N = WMMA_K = 16
FLOP_PER_WMMA = 2 * WMMA_M * WMMA_N * WMMA_K

_ELEM = {"f16": fx.Float16, "bf16": fx.BFloat16}


def build_wmma_peak(nacc: int, dtype_str: str = "f16"):
    """Build a launcher running ``nacc`` independent WMMA chains for ``iters`` steps."""
    if dtype_str not in _ELEM:
        raise ValueError(f"dtype must be one of {sorted(_ELEM)}, got {dtype_str!r}")
    elem = _ELEM[dtype_str]

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def wmma_peak_kernel(OUT: fx.Pointer, iters: fx.Int32):
        v8f32 = Vec.make_type(8, fx.Float32)
        a = Vec.filled(8, 1.0, elem).ir_value()
        b = Vec.filled(8, 1.0, elem).ir_value()
        zero = Vec.filled(8, 0.0, fx.Float32).ir_value()

        def wmma(x, y, c):
            if const_expr(dtype_str == "bf16"):
                xi = Vec(x).bitcast(fx.Int16).ir_value()
                yi = Vec(y).bitcast(fx.Int16).ir_value()
                return rocdl.wmma_f32_16x16x16_bf16(v8f32, xi, yi, c).result
            return rocdl.wmma_f32_16x16x16_f16(v8f32, x, y, c).result

        def as_list(x):
            # A single loop-carried value is handed back bare, not as a 1-list.
            return list(x) if isinstance(x, (list, tuple)) else [x]

        init = [zero for _ in range_constexpr(nacc)]
        res = init
        for _i, carried in range(0, fx.Index(iters), 1, init=init):
            cur = as_list(carried)
            out = [wmma(a, b, cur[j]) for j in range_constexpr(nacc)]
            res = yield out

        res = as_list(res)

        # Consume every chain so none is dead-code eliminated, and so the stored
        # value certifies that all of them ran for the full iteration count.
        total = Vec(res[0])[0]
        for j in range_constexpr(nacc - 1):
            total = total + Vec(res[j + 1])[0]
        idx = fx.Index(gpu.block_idx.x) * BLOCK + fx.Index(gpu.thread_idx.x)
        fx.ptr_store(total, OUT + fx.Int32(idx))

    @flyc.jit
    def launch(
        OUT: fx.Pointer,
        iters: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        wmma_peak_kernel(OUT, iters).launch(grid=(fx.Index(grid), 1, 1), block=(BLOCK, 1, 1), stream=stream)

    def run(out, iters, grid, stream=None):
        ptr = flyc.from_c_void_p(fx.Float32, out.data_ptr())
        cf = getattr(run, "_cf", None)
        if cf is None:
            run._cf = flyc.compile(launch, ptr, iters, grid, fx.Stream(stream))
        else:
            cf(ptr, iters, grid, fx.Stream(stream))

    return run


def measure(nacc: int, iters: int = 2000, grid: int = 1024, dtype_str: str = "f16", reps: int = 5):
    """Return (tflops, checked_ok). Uses the fastest of ``reps`` timed runs."""
    run = build_wmma_peak(nacc, dtype_str)
    out = torch.zeros(grid * BLOCK, dtype=torch.float32, device="cuda")
    raw_stream = torch.cuda.current_stream().cuda_stream

    run(out, iters, grid, raw_stream)
    torch.cuda.synchronize()
    expect = float(WMMA_K * iters * nacc)
    got = out[0].item()
    checked_ok = abs(got - expect) < max(1.0, 1e-3 * expect)

    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run(out, iters, grid, raw_stream)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    flops = grid * WAVES_PER_WG * iters * nacc * FLOP_PER_WMMA
    return flops / min(times) * 1e-9, checked_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=2000, help="loop trip count per wave")
    ap.add_argument("--grid", type=int, default=1024, help="workgroups (256 threads each)")
    ap.add_argument("--reps", type=int, default=5, help="timed repetitions; the fastest is reported")
    ap.add_argument("--nacc", type=int, nargs="+", default=[1, 2, 4, 8, 16], help="accumulator chains to sweep")
    ap.add_argument("--dtype", nargs="+", default=["f16", "bf16"], choices=sorted(_ELEM))
    args = ap.parse_args()

    props = torch.cuda.get_device_properties(0)
    cus = props.multi_processor_count * 2  # ROCm reports WGPs on RDNA; 1 WGP = 2 CUs
    print(f"{props.name} | {props.gcnArchName} | {props.multi_processor_count} WGP ({cus} CU)")
    print(f"grid={args.grid} wg x {BLOCK} thr, iters={args.iters}, best of {args.reps}\n")
    print(f"{'dtype':>6} {'chains':>7} {'TFLOPS':>9}  check")
    for dtype_str in args.dtype:
        for nacc in args.nacc:
            tflops, ok = measure(nacc, args.iters, args.grid, dtype_str, args.reps)
            print(f"{dtype_str:>6} {nacc:>7} {tflops:>9.1f}  {'ok' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
