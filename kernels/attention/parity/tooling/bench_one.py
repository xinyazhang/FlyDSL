"""Measure a single (variant, causal) config in an isolated process.

Usage: python3 bench_one.py <baseline|bp|m32> <0|1> [f16|bf16]
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

VARIANTS = {
    "baseline": (128, {}),
    "bp": (128, {"use_binding_prefetch": True}),
    "m32": (64, {"variant": "m32"}),
}
name, causal = sys.argv[1], bool(int(sys.argv[2]))
dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[sys.argv[3] if len(sys.argv) > 3 else "f16"]
d, kw = VARIANTS[name]
BATCH, H, N = 2, 12, 4096
torch.manual_seed(0)
q, k, v = make_qkv(BATCH, H, N, d, dtype=dtype, layout=LAYOUT)
fn = lambda: flydsl_flash_attn_func_gfx1201(q, k, v, causal=causal, **kw)  # noqa: E731
fn()
torch.cuda.synchronize()
ms = do_bench(fn, warmup=200, rep=1000, return_mode="median")
fl = 2.0 * BATCH * H * N * N * d * 2 * (0.5 if causal else 1.0)
print(f"{name:>8} causal={int(causal)} {sys.argv[3] if len(sys.argv) > 3 else 'f16':>4}  {fl / ms * 1e-9:7.1f} TFLOPS")
