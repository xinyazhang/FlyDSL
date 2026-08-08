# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Interleaved ladder benchmark for the gfx1201 attention kernel.

The board drifts ~5% over minutes, which is larger than most effects worth
chasing here, so A and B are measured alternately within each rep and the
median of per-rep *ratios* is reported -- never two separate sweeps compared
after the fact. See sdpa_lore_gfx1201.md for why this discipline is not
optional.

Originally an A/B against the three pre-unification kernels; those are retired
(N2), so it now measures the one kernel across the ladder. The output is meant
to be diffed **across revisions** -- stash, run, unstash, run -- which is how
every feature phase in sdpa-close-gap-plan*.md checks that it has not
regressed. Each phase adds runtime values or live state to a loop that is
already latency-bound and spilling at head_dim 512.

Read `sdpa_lore_gfx1201.md` before trusting a small delta: head_dim 32
non-causal is bimodal at ~58 and ~63 TFLOPS independent of the code, so any
delta there below ~10% needs at least three alternating runs.

    export ROCM_PATH=$(rocm-sdk path --root)
    cd kernels/attention/parity && python3 bench_aiw_ab.py
"""

import os
import sys

import _bootstrap  # noqa: F401  (puts parity/ on sys.path)
import torch
from bench_shim import do_bench
from flash_attn_func_gfx1201_interface import flydsl_flash_attn_func_gfx1201
from qkv import make_qkv

# BHSD shape always; layout is a knob. See `qkv.py`.
LAYOUT = os.environ.get("FLYDSL_BENCH_LAYOUT", "bhsd")

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
    print(f"B={BATCH} H={H} N={N} f16   {REPS} reps, median\n")
    print(f"{'hdim':>5} {'causal':>6} {'TFLOPS':>8} {'ms':>9}")
    for d in LADDER:
        for causal in (False, True):
            torch.manual_seed(0)
            q, k, v = make_qkv(BATCH, H, N, d, dtype=dtype, layout=LAYOUT)
            ms = sorted(measure(q, k, v, causal) for _ in range(REPS))
            m = ms[len(ms) // 2]
            print(f"{d:5} {int(causal):6} {tflops(m, d, causal):8.1f} {m:9.4f}")


if __name__ == "__main__":
    sys.exit(main())
