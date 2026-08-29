# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""TFLOPS across the head_dim ladder at one sequence length, all three kernels.

The existing harnesses sweep the wrong axis for this question: `bench_fmha.py`
fixes head_dim and sweeps N_CTX, and `bench_bwd.py` covers only the backward.
This one fixes the sequence length and walks the ladder, which is the shape of
an "did that change cost anything" check.

Defaults are B=1, H=8, seqlen 8192. At BLOCK_M 128 that is 64 Q tiles, so
512 workgroups over this board's 64 CUs -- saturated with room to spare, and
the same (B, H) the tuning tables in `fmha_tuning_*.py` were measured at, so
the numbers are comparable to what is recorded there.

    export ROCM_PATH=$(rocm-sdk path --root)
    cd kernels/attention/parity
    python3 tooling/bench_hdim_ladder.py
    python3 tooling/bench_hdim_ladder.py --causal --head-dims 64,128,256

**Read the numbers against the drift, not against each other.** This board
moves up to 15% between whole-script runs, so a single arm is worth little; two
arms interleaved inside one process are worth something. `--repeat` reports the
spread so you can see which you are looking at.
"""

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LADDER = (16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 384, 512)


def _time_ms(fn, warmup, iters):
    """Median of `iters` timed launches, in ms. Median, not mean: a stray
    context switch skews a mean and this board produces them."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=8192)
    ap.add_argument("--head-dims", default=",".join(str(d) for d in LADDER))
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--dtype", default="f16", choices=("f16", "bf16"))
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=1, help="repeat the whole sweep; reports the spread")
    a = ap.parse_args()

    import torch
    from flash_attn_func_gfx1201_interface import flydsl_flash_attn_func_gfx1201
    from fmha_bwd_dkdv_gfx1201_interface import flydsl_flash_attn_bwd_dkdv_gfx1201
    from fmha_bwd_dq_gfx1201_interface import flydsl_bwd_dq_gfx1201

    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16
    B, H, S = a.batch, a.heads, a.seqlen
    dims = [int(x) for x in a.head_dims.split(",")]

    print(f"B={B} H={H} seqlen={S} causal={a.causal} {a.dtype}  warmup={a.warmup} iters={a.iters}")
    print(f"{'hdim':>5} {'fwd':>9} {'dkdv':>9} {'dq':>9}   (TFLOPS)")
    print("-" * 46)

    rows = {}
    for rep in range(a.repeat):
        for d in dims:
            g = torch.Generator(device="cuda").manual_seed(0)

            def t():
                return torch.randn(B, H, S, d, dtype=dtype, device="cuda", generator=g)

            q, k, v, do = t(), t(), t(), t()
            # 2 GEMMs of (S x S x d) in the forward; the backward does 5 of the
            # same shape (dV, dP, dS^T Q, dQ, plus recomputing S).
            base = 2.0 * B * H * S * S * d
            half = 0.5 if a.causal else 1.0
            o, lse = flydsl_flash_attn_func_gfx1201(q, k, v, causal=a.causal, return_lse=True)
            delta = (do.float() * o.float()).sum(-1)

            fwd = _time_ms(lambda: flydsl_flash_attn_func_gfx1201(q, k, v, causal=a.causal), a.warmup, a.iters)
            kv = _time_ms(
                lambda: flydsl_flash_attn_bwd_dkdv_gfx1201(q, k, v, do, o, lse, causal=a.causal, delta=delta),
                a.warmup,
                a.iters,
            )
            dq = _time_ms(
                lambda: flydsl_bwd_dq_gfx1201(q, k, v, o, do, lse, causal=a.causal, delta=delta),
                a.warmup,
                a.iters,
            )
            tf = lambda ms, n: n * base * half / (ms * 1e-3) / 1e12  # noqa: E731
            rows.setdefault(d, []).append((tf(fwd, 2), tf(kv, 3), tf(dq, 3)))
            print(f"{d:>5} {tf(fwd, 2):>9.1f} {tf(kv, 3):>9.1f} {tf(dq, 3):>9.1f}", flush=True)
            # No `del`: rebinding on the next iteration frees the previous
            # generation, and two generations of the widest rung is under a
            # gigabyte. An explicit `del` here reads as a use-after-delete to
            # ruff, because the timing lambdas close over these names.
        if a.repeat > 1:
            print(f"--- end of repeat {rep + 1}/{a.repeat} ---", flush=True)

    if a.repeat > 1:
        print(f"\n{'hdim':>5} {'fwd spread':>12} {'dkdv spread':>13} {'dq spread':>11}")
        for d, vals in rows.items():
            sp = [f"{min(x[i] for x in vals):.1f}-{max(x[i] for x in vals):.1f}" for i in range(3)]
            print(f"{d:>5} {sp[0]:>12} {sp[1]:>13} {sp[2]:>11}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
