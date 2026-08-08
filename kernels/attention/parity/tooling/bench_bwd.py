#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Full backward-operator throughput, on AOTriton's terms.

    python3 bench_bwd.py [--heads 48] [--batch 4] [--causal] [--layout bhsd]

**The FLOP count is AOTriton's**, from `modules/flash/kernel/
performance_backward.py`, so the numbers are directly comparable to theirs::

    flops_per_matmul = 2 * B * H * N * N * D
    total_flops      = 2 * flops_per_matmul     # QK^T and PV
    if causal:  total_flops *= 0.5
    total_flops *= 2.5                          # 2.0 backward + 0.5 recompute

i.e. `10 * B*H*N^2*D`, halved for causal.

**What "full operator" includes here, and the caveat that matters.** AOTriton
times `o.backward(do)`, which runs its own fused preprocess kernel for
`delta = rowsum(dO . O)` and then the backward kernel(s). Our `delta` is still
a torch pass -- `(dO.float() * O.float()).sum(-1)`, two extra BHSD reads per
call -- so it is timed separately and reported as its own column. The
`total` column is the honest operator number; the `kernel` column is what the
kernels alone would give if `delta` were fused, which is the fair target to
compare against a future preprocess kernel.

Three providers:

    split   dkdv + dq, two launches
    fuse    one launch, both roles
    torch   `torch.autograd.grad` on `F.scaled_dot_product_attention`

`torch` is the anchor. It is measured on the same tensors, in the same shapes,
with the same clock, so the ratio is meaningful even where the absolute
numbers move with the board.
"""

import argparse
import os

import torch
import torch.nn.functional as F

import _bootstrap  # noqa: F401  (puts parity/ on sys.path)
from bench_shim import do_bench
from qkv import make_qkv

from flash_attn_func_gfx1201_interface import flydsl_flash_attn_func_gfx1201
from fmha_bwd_dkdv_gfx1201_interface import flydsl_flash_attn_bwd_dkdv_gfx1201
from fmha_bwd_dq_gfx1201_interface import flydsl_bwd_dq_gfx1201
from fmha_bwd_fuse_gfx1201_interface import flydsl_fmha_bwd_fuse_gfx1201


def bwd_flops(batch, heads, n, d, causal):
    """AOTriton's count: 10 * B*H*N^2*D, halved for causal."""
    total = 2.0 * (2.0 * batch * heads * n * n * d)
    if causal:
        total *= 0.5
    return total * 2.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=48)
    p.add_argument("--head-dims", default="64,128")
    p.add_argument("--n-ctx", default="1024,2048,4096")
    p.add_argument("--causal", action="store_true")
    p.add_argument("--layout", default=os.environ.get("FLYDSL_BENCH_LAYOUT", "bhsd"))
    a = p.parse_args()

    print(f"batch={a.batch} heads={a.heads} causal={a.causal} layout={a.layout} f16")
    print(f"{'D':>4} {'N':>6} {'provider':>8} {'delta':>8} {'kernel':>8} {'total':>8} "
          f"{'TFLOPS':>8} {'vs torch':>8}")

    for d in (int(x) for x in a.head_dims.split(",")):
        for n in (int(x) for x in a.n_ctx.split(",")):
            q, k, v = make_qkv(a.batch, a.heads, n, d, layout=a.layout, seed=0)
            do = make_qkv(a.batch, a.heads, n, d, layout=a.layout, seed=1, n=1)
            o, lse = flydsl_flash_attn_func_gfx1201(q, k, v, causal=a.causal, return_lse=True)
            lse2 = lse.reshape(a.batch * a.heads, n).contiguous()
            flops = bwd_flops(a.batch, a.heads, n, d, a.causal)

            ms_delta = do_bench(lambda: (do.float() * o.float()).sum(-1))
            delta = (do.float() * o.float()).sum(-1).reshape(a.batch * a.heads, n).contiguous()

            qg, kg, vg = (t.detach().requires_grad_(True) for t in (q, k, v))
            ref = F.scaled_dot_product_attention(qg, kg, vg, is_causal=a.causal)
            ms_torch = do_bench(lambda: torch.autograd.grad(ref, (qg, kg, vg), do, retain_graph=True))
            tf_torch = flops / ms_torch * 1e-9

            rows = []
            try:
                ms_dkdv = do_bench(lambda: flydsl_flash_attn_bwd_dkdv_gfx1201(
                    q, k, v, do, o, lse2, causal=a.causal, delta=delta))
                ms_dq = do_bench(lambda: flydsl_bwd_dq_gfx1201(
                    q, k, v, o, do, lse2, causal=a.causal, delta=delta))
                rows.append(("split", ms_dkdv + ms_dq))
            except Exception as e:
                rows.append(("split", f"n/a ({type(e).__name__})"))
            try:
                ms_fuse = do_bench(lambda: flydsl_fmha_bwd_fuse_gfx1201(
                    q, k, v, o, do, lse2, causal=a.causal, delta=delta))
                rows.append(("fuse", ms_fuse))
            except Exception as e:
                rows.append(("fuse", f"n/a ({type(e).__name__})"))

            print(f"{d:>4} {n:>6} {'torch':>8} {'':>8} {'':>8} {ms_torch:8.2f} "
                  f"{tf_torch:8.1f} {1.0:8.2f}")
            for name, ms in rows:
                if isinstance(ms, str):
                    print(f"{d:>4} {n:>6} {name:>8} {'':>8} {ms:>26}")
                    continue
                tot = ms + ms_delta
                print(f"{d:>4} {n:>6} {name:>8} {ms_delta:8.2f} {ms:8.2f} {tot:8.2f} "
                      f"{flops / tot * 1e-9:8.1f} {tf_torch and (ms_torch / tot):8.2f}")


if __name__ == "__main__":
    main()
