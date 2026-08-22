"""Backward throughput against the forward, one shape, one accounting.

The two backward kernels were first benchmarked separately, at different shapes
and with different FLOP counts, which makes their numbers incomparable with
each other and with the forward. This measures all three at **one** shape and
reports three things that answer different questions:

- **nominal TFLOP/s per kernel** -- its own GEMM count, so it says how well that
  kernel issues MFMAs. dK/dV does four GEMMs (`8*B*H*Sq*Sk*d`), dQ three
  (`6*...`), the forward two (`4*...`).
- **effective backward TFLOP/s** -- the *mathematically necessary* five GEMMs,
  `10*B*H*Sq*Sk*d`, over the two kernels' combined time. The split recomputes S
  and dP, so it issues seven GEMMs to deliver five; this is the number that
  prices that choice.
- **bwd / fwd wall-clock**, which is what a training step feels. The usual
  expectation for flash attention is around 2.5x.

    python3 bench_bwd_gfx950.py [hdim ...]
"""

import sys
import time

import _bootstrap  # noqa: F401  (puts parity/ on sys.path)
import torch
from flash_attn_func_gfx950 import build_flash_attn_func_gfx950_module as build_fwd
from fmha_bwd_dkdv_gfx950 import build_fmha_bwd_dkdv_gfx950_module as build_dkdv
from fmha_bwd_dq_gfx950 import build_fmha_bwd_dq_gfx950_module as build_dq

DT = torch.bfloat16
B, H, S = 2, 8, 4096
HDIMS = [64, 128, 192, 256, 384, 512]


def timed(call, warmup=10, rep=20, rounds=3):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(rep):
            call()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / rep)
    return best


def cfg(knobs, *names):
    return " ".join(f"{n}={getattr(knobs, n, '?')}" for n in names)


def main(hdims):
    print(f"B={B} H={H} S={S} bf16 non-causal, GPU {torch.cuda.current_device()}\n")
    hdr = f"{'hdim':>5} {'fwd TF':>8} {'dkdv TF':>8} {'dq TF':>8} | {'bwd eff TF':>10} {'bwd/fwd':>8}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for d in hdims:
        q, k, v, do = (torch.randn(B, H, S, d, device="cuda", dtype=DT) for _ in range(4))
        o = torch.empty_like(q)
        dq, dk, dv = (torch.empty_like(q) for _ in range(3))
        lse = torch.zeros(B * H, S, device="cuda", dtype=torch.float32)
        delta = torch.zeros(B * H, S, device="cuda", dtype=torch.float32)
        scale = d**-0.5

        f_fwd = build_fwd(num_heads=H, head_dim=d, causal=False, dtype_str="bf16", num_kv_heads=H, return_lse=True)
        f_kv = build_dkdv(num_heads=H, head_dim=d, num_kv_heads=H)
        f_dq = build_dq(num_heads=H, head_dim=d, num_kv_heads=H)

        # Bound as defaults, not captured: the `del` below frees the tensors,
        # and a closure over names that get deleted is an F821 to ruff.
        def c_fwd(q=q, k=k, v=v, o=o, lse=lse):
            f_fwd(q, k, v, o, B, S, seqlen_k=S, scale=scale, lse=lse)

        def c_kv(q=q, k=k, v=v, do=do, dk=dk, dv=dv, lse=lse, delta=delta):
            f_kv(q, k, v, do, dk, dv, lse, delta, B, S, seqlen_k=S, scale=scale)

        def c_dq(q=q, k=k, v=v, do=do, dq=dq, lse=lse, delta=delta):
            f_dq(q, k, v, do, dq, lse, delta, B, S, seqlen_k=S, scale=scale)

        c_fwd()
        c_kv()
        c_dq()
        torch.cuda.synchronize()
        t_f, t_kv, t_dq = timed(c_fwd), timed(c_kv), timed(c_dq)

        base = B * H * S * S * d * 1e-12
        tf_f, tf_kv, tf_dq = 4 * base / t_f, 8 * base / t_kv, 6 * base / t_dq
        eff = 10 * base / (t_kv + t_dq)
        print(f"{d:>5} {tf_f:8.0f} {tf_kv:8.0f} {tf_dq:8.0f} | {eff:10.0f} {(t_kv + t_dq) / t_f:8.2f}x")
        rows.append((d, tf_f, tf_kv, tf_dq, eff))
        del q, k, v, do, o, dq, dk, dv, lse, delta
        torch.cuda.empty_cache()

    print("\nchosen configuration per rung (does it diverge with width, as the forward's does?)")
    print(f"{'hdim':>5}  {'dkdv':<52} {'dq':<40}")
    from fmha_tuning_bwd_dkdv_gfx950 import BwdDkDvInputMetadata as KVM
    from fmha_tuning_bwd_dkdv_gfx950 import bwd_dkdv_knobs
    from fmha_tuning_bwd_dq_gfx950 import bwd_dq_knobs
    from fmha_tuning_gfx950 import FmhaInputMetadata as QM

    for d, *_ in rows:
        kv = bwd_dkdv_knobs("gfx950").resolve(KVM(num_heads=H, num_kv_heads=H, head_dim=d))
        dqk = bwd_dq_knobs("gfx950").resolve(QM(num_heads=H, num_kv_heads=H, head_dim=d, causal=False))
        print(
            f"{d:>5}  {cfg(kv, 'num_waves', 'waves_per_eu', 'head_dim_granule', 'dkv_shards', 'block_kv'):<52}"
            f" {cfg(dqk, 'num_waves', 'waves_per_eu', 'block_n'):<40}"
        )


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or HDIMS)
