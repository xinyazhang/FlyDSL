# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Cost of reducing a per-wave partial tile across waves through LDS.

Step 1 decision gate for the wide-head_dim FMHA design in
``kernels/attention/plan_gfx1201_large_hdim.md``. That design shards the head
dimension across ``SHARDS`` waves in GEMM1, which means each wave produces a
*partial* S tile that must be summed across the shard group before softmax. The
whole design only pays if that reduction costs materially less than the WMMA
work it saves.

One reduction is a 16 x 32 f32 tile (16 f32 per lane, 2048 B per Q-tile),
performed once per Q-tile per KV tile. Variants:

``none``    no reduction; the arithmetic only, to isolate the LDS cost.
``expl``    each wave writes its partial, then reads the other SHARDS-1 and
            sums. O(N^2) traffic, fixed summation order, deterministic.
``atomic``  shard 0 plain-stores (which also initialises the slot), the rest
            ``ds_add_f32`` onto it, then all read the total. O(N) traffic, but
            an atomic add is a bank-level read-modify-write and the summation
            order is not deterministic.
``wmma``    ``--wmma-count`` WMMAs per iteration instead of a reduction, to
            price the reduction in units of the work it is meant to displace.

The reduced tile is carried into the next iteration's partial, so no variant
can be folded away; ``--check`` verifies the arithmetic end to end.

Run directly (imports only flydsl/torch)::

    export ROCM_PATH=$(rocm-sdk path --root)
    python3 kernels/microbench/lds_reduce.py
"""

import argparse

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, llvm as _llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

WARP_SIZE = 32
SHARDS = 4          # waves cooperating on one Q-tile's head-dim reduction
Q_TILES = 2         # Q-tiles per workgroup
WAVES = SHARDS * Q_TILES
BLOCK = WAVES * WARP_SIZE

TILE_F32 = 16       # f32 per lane in one 16x32 S tile (16*32/32 lanes)
SLOTS = Q_TILES * SHARDS
LDS_F32 = SLOTS * TILE_F32 * WARP_SIZE

WMMA_M = WMMA_N = WMMA_K = 16
VARIANTS = ("none", "expl", "atomic", "wmma")



def build(variant: str, wmma_count: int = 32):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Float32, LDS_F32, 16]

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def reduce_kernel(OUT: fx.Pointer, iters: fx.Int32):
        v8f32 = Vec.make_type(8, fx.Float32)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        buf = lds.buf.ptr

        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        q_tile = wave // SHARDS
        shard = wave % SHARDS

        # Slot layout [slot][elem][lane]: consecutive lanes hit consecutive
        # banks, so a whole wave's access to one elem is conflict-free.
        def slot_base(slot):
            return slot * (TILE_F32 * WARP_SIZE)

        own_slot = q_tile * SHARDS + shard
        acc_slot = q_tile * SHARDS  # atomic variant accumulates into shard 0's

        def addr(base, e):
            return base + e * WARP_SIZE + lane

        # ptrtoint on a shared pointer already yields i32 (LDS is 32-bit addressed)
        lds_base_i32 = _raw(fx.ptrtoint(buf))

        def lds_ptr(idx):
            """LDS f32 element pointer as an addrspace(3) llvm.ptr.

            fly.ptr is not an LLVM pointer, so go through ptrtoint/inttoptr the
            way the attention kernels do for global. LDS addresses are 32-bit.
            """
            byte = arith.muli(_raw(fx.Int32(idx)), _raw(fx.Int32(4)))
            addr = arith.addi(lds_base_i32, byte)
            return _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<3>"), addr).result

        def atomic_add(idx, val):
            _llvm.AtomicRMWOp(
                _llvm.AtomicBinOp.fadd,
                lds_ptr(idx),
                _raw(val),
                _llvm.AtomicOrdering.monotonic,
            )

        # Runtime-derived seed: keeps the partial data-dependent so nothing in
        # the loop can be constant-folded or hoisted.
        seed = fx.Int32(gpu.thread_idx.x).to(fx.Float32) * fx.Float32(1e-3)
        init = [_raw(seed) for _ in range_constexpr(TILE_F32)]

        a = Vec.filled(8, 1.0, fx.Float16).ir_value()
        b = Vec.filled(8, 1.0, fx.Float16).ir_value()

        res = init
        for _i, carried in range(0, fx.Index(iters), 1, init=init):
            p = list(carried)

            if const_expr(variant == "none"):
                # Same shape of VALU work as `expl`'s summation, no LDS.
                out = [p[e] + p[e] + p[e] + p[e] for e in range_constexpr(TILE_F32)]

            elif const_expr(variant == "wmma"):
                c = Vec.filled(8, 0.0, fx.Float32).ir_value()
                for _k in range_constexpr(wmma_count):
                    c = rocdl.wmma_f32_16x16x16_f16(v8f32, a, b, c).result
                lead = Vec(c)[0]
                out = [p[e] + lead for e in range_constexpr(TILE_F32)]

            elif const_expr(variant == "expl"):
                base = slot_base(own_slot)
                for e in range_constexpr(TILE_F32):
                    fx.ptr_store(fx.Float32(p[e]), buf + fx.Int32(addr(base, e)))
                gpu.barrier()
                out = []
                for e in range_constexpr(TILE_F32):
                    s = p[e]
                    for k in range_constexpr(SHARDS - 1):
                        peer = slot_base(q_tile * SHARDS + (shard + k + 1) % SHARDS)
                        s = s + fx.ptr_load(
                            buf + fx.Int32(addr(peer, e)), result_type=ir.F32Type.get()
                        )
                    out.append(s)
                gpu.barrier()

            else:  # atomic
                base = slot_base(acc_slot)
                # shard 0 stores (initialising the slot), the rest add onto it.
                if shard == fx.Index(0):
                    for e in range_constexpr(TILE_F32):
                        fx.ptr_store(fx.Float32(p[e]), buf + fx.Int32(addr(base, e)))
                gpu.barrier()
                if shard != fx.Index(0):
                    for e in range_constexpr(TILE_F32):
                        atomic_add(addr(base, e), p[e])
                gpu.barrier()
                out = [
                    fx.ptr_load(buf + fx.Int32(addr(base, e)), result_type=ir.F32Type.get())
                    for e in range_constexpr(TILE_F32)
                ]
                gpu.barrier()

            res = yield [_raw(o) for o in out]

        res = list(res)
        total = fx.Float32(res[0])
        for e in range_constexpr(TILE_F32 - 1):
            total = total + fx.Float32(res[e + 1])
        idx = fx.Index(gpu.block_idx.x) * BLOCK + tid
        fx.ptr_store(total, OUT + fx.Int32(idx))

    @flyc.jit
    def launch(
        OUT: fx.Pointer,
        iters: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        reduce_kernel(OUT, iters).launch(
            grid=(fx.Index(grid), 1, 1), block=(BLOCK, 1, 1), stream=stream
        )

    def run(out, iters, grid, stream=None):
        ptr = flyc.from_c_void_p(fx.Float32, out.data_ptr())
        cf = getattr(run, "_cf", None)
        if cf is None:
            run._cf = flyc.compile(launch, ptr, iters, grid, fx.Stream(stream))
        else:
            cf(ptr, iters, grid, fx.Stream(stream))

    return run


def measure(variant, iters, grid, reps, wmma_count=32):
    """Return ns per reduction (per Q-tile), using the fastest of ``reps`` runs."""
    run = build(variant, wmma_count)
    out = torch.zeros(grid * BLOCK, dtype=torch.float32, device="cuda")
    raw = torch.cuda.current_stream().cuda_stream
    run(out, iters, grid, raw)
    torch.cuda.synchronize()

    times = []
    for _ in range(reps):
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record()
        run(out, iters, grid, raw)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    ms = min(times)
    reductions = grid * Q_TILES * iters      # one per Q-tile per iteration
    return ms * 1e6 / reductions, out[0].item()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--grid", type=int, default=512, help=f"workgroups ({BLOCK} thr each)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--wmma-count", type=int, default=32,
                    help="WMMAs per iteration for the 'wmma' yardstick")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} | {p.gcnArchName} | {p.multi_processor_count} WGP")
    print(f"{SHARDS} shards x {Q_TILES} q-tiles = {WAVES} waves, {TILE_F32} f32/lane per tile")
    print(f"grid={args.grid}, iters={args.iters}, best of {args.reps}\n")
    print(f"{'variant':>8} {'ns/reduction':>13} {'vs none':>9}   note")
    base = None
    for v in args.variants:
        ns, chk = measure(v, args.iters, args.grid, args.reps, args.wmma_count)
        if v == "none":
            base = ns
        delta = f"{ns - base:+.1f}" if base is not None and v != "none" else "-"
        note = f"{args.wmma_count} WMMA/iter" if v == "wmma" else ""
        print(f"{v:>8} {ns:>13.2f} {delta:>9}   {note}")


if __name__ == "__main__":
    main()
