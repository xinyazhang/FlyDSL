# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Interleaved A/B of the unified kernel against the legacy ones.

The board drifts ~5% over minutes, which is larger than most effects worth
chasing here, so A and B are measured alternately within each rep and the
median of per-rep *ratios* is reported -- never two separate sweeps compared
after the fact. See sdpa_lore_gfx1201.md for why this discipline is not
optional.

Kept (rather than deleted with the rest of the aiw bring-up scratch) because
every feature phase in sdpa-close-gap-plan*.md has to re-run this: each one
adds runtime values or live state to a loop that is already latency-bound and
spilling at head_dim 512.

    export ROCM_PATH=$(rocm-sdk path --root)
    cd kernels/attention && python3 bench_aiw_ab.py
"""

import sys

import torch

from bench_shim import do_bench
from flash_attn_func_gfx1201_interface import flydsl_flash_attn_func_gfx1201

BATCH, H, N = 1, 8, 4096
REPS = 3
LADDER = [16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 384, 512]


def tflops(ms, d, causal):
    fl = 2.0 * BATCH * H * N * N * d * 2 * (0.5 if causal else 1.0)
    return fl / ms * 1e-9


def measure(q, k, v, causal, **kw):
    def fn():
        return flydsl_flash_attn_func_gfx1201(q, k, v, causal=causal, **kw)

    fn()
    torch.cuda.synchronize()
    return do_bench(fn, warmup=50, rep=200, return_mode="median")


def main():
    dtype = torch.float16
    print(f"B={BATCH} H={H} N={N} f16   aiw vs legacy, {REPS} interleaved reps\n")
    print(f"{'hdim':>5} {'causal':>6} {'legacy':>8} {'aiw':>8} {'ratio':>7}")
    worst = (1e9, None)
    for d in LADDER:
        for causal in (False, True):
            torch.manual_seed(0)
            q, k, v = (
                torch.randn((BATCH, N, H, d), dtype=dtype, device="cuda")
                for _ in range(3)
            )
            # Legacy route: baseline below 48, bp at/above it (what the
            # interface used to pick).
            leg = "legacy" if d < 48 else "legacy_bp"
            ratios, a_ms, b_ms = [], [], []
            for _ in range(REPS):
                ml = measure(q, k, v, causal, variant=leg)
                ma = measure(q, k, v, causal)
                a_ms.append(ml)
                b_ms.append(ma)
                ratios.append(ml / ma)
            ratios.sort()
            a_ms.sort()
            b_ms.sort()
            r = ratios[len(ratios) // 2]
            tl = tflops(a_ms[len(a_ms) // 2], d, causal)
            ta = tflops(b_ms[len(b_ms) // 2], d, causal)
            flag = "" if r > 0.97 else "   <-- regression"
            print(f"{d:5} {int(causal):6} {tl:8.1f} {ta:8.1f} {r:7.3f}{flag}")
            if r < worst[0]:
                worst = (r, (d, causal))
    print(f"\nworst ratio {worst[0]:.3f} at hdim={worst[1][0]} causal={int(worst[1][1])}")


if __name__ == "__main__":
    sys.exit(main())
