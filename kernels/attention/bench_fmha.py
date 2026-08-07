#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Forward FMHA benchmark for the FlyDSL gfx1201 (RDNA4) flash-attention kernel.

Sweeps ``N_CTX`` for a fixed (BATCH, N_HEADS, D_HEAD) and reports TFLOPS (or
milliseconds) per sequence length, printing a table and writing one CSV per
configuration.

Overridable via environment variables: ``BATCH``, ``N_HEADS``, ``D_HEAD``,
``USE_TFLOPS`` (``1`` reports TFLOPS, ``0`` reports ms).

Run with::

    export ROCM_PATH=$(rocm-sdk path --root)   # only needed for pip-installed ROCm
    cd FlyDSL/kernels/attention && python3 bench_fmha.py

The cwd is assumed to be this directory. These four files (``bench_fmha.py``,
``bench_shim.py``, ``flash_attn_func_gfx1201_interface.py``,
``flash_attn_func_gfx1201.py``) are a self-contained prototype: they import each
other by bare module name and depend on nothing from the wider repository, so
the directory can be dropped into another container as-is. No ``PYTHONPATH``,
no ``sys.path`` manipulation.

``ROCM_PATH`` must point at a ROCm tree containing ``llvm/bin/ld.lld`` and
``amdgcn/bitcode``: MLIR's ROCDL target locates the linker and device bitcode
through it at serialization time, and defaults to ``/opt/rocm``, which does not
exist in a pip-only ROCm install. Without it the JIT fails with a bare
``lld invocation failed``. Shared-library loading is handled by the wheels'
runpath, so ``LD_LIBRARY_PATH`` does not need to be set.

One CSV is written per configuration into the current directory, i.e. next to
this file when run as above.

Timing/reporting comes from :mod:`bench_shim`, a triton-free reimplementation of
the ``triton.testing`` API this harness was written against.
"""

import os

import torch
from bench_shim import Benchmark, do_bench, perf_report
from flash_attn_func_gfx1201_interface import flydsl_flash_attn_func_gfx1201

USE_TFLOPS = bool(int(os.getenv("USE_TFLOPS", default="1")))
print(f"{USE_TFLOPS=}")

BATCH, N_CTX, N_HEADS, D_HEAD = 2, 32768, 12, 128

X_VALS = [1024, 2048, 4096, 8192, 16384, N_CTX]

D_HEAD = int(os.getenv("D_HEAD", default=D_HEAD))
BATCH = int(os.getenv("BATCH", default=BATCH))
N_HEADS = int(os.getenv("N_HEADS", default=N_HEADS))
ALL_CAUSALS = [False]
print(f"{ALL_CAUSALS=}")

configs = []
for mode in ["fwd"]:
    for causal in ALL_CAUSALS:
        configs.append(
            Benchmark(
                x_names=["N_CTX"],
                x_vals=list(X_VALS),
                line_arg="provider",
                line_vals=["flydsl"],
                line_names=["FlyDSL"],
                styles=[("red", "-"), ("blue", "-")],
                ylabel="TFLOPS" if USE_TFLOPS else "ms",
                plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "D_HEAD": D_HEAD,
                    "dtype": torch.float16,
                    "mode": mode,
                    "causal": causal,
                },
            )
        )


@perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, mode, provider, dtype=torch.float16, device="cuda"):
    print(f"{N_CTX=}")
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    # Bwd pass only supports causal=True right now
    if mode == "bwd":
        assert False, f"Dont support {mode=}"
    if provider == "flydsl":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flydsl_flash_attn_func_gfx1201(q, k, v, causal=causal)  # noqa: E731
        ms = do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    if USE_TFLOPS:
        return total_flops / ms * 1e-9
    return ms


if __name__ == "__main__":
    bench_flash_attention.run(save_path=".", print_data=True)
