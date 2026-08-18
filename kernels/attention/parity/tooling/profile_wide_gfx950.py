# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Single-launch driver for profiling one parity build under rocprofv3.

    rocprofv3 --pmc <counters> -- python3 tooling/profile_wide_gfx950.py 512
    rocprofv3 --att --att-target-cu 1 -- python3 tooling/profile_wide_gfx950.py 512

Deliberately **one launch after a warm-up**, because both tools attribute per
dispatch: ATT fills its buffer from whichever dispatches it sees, and a
benchmark loop would either overflow it or mix the JIT's first-call behaviour
into the counters. The warm-up call is what forces the compile, so the measured
dispatch is the steady-state one.

The shape is small on purpose (`B=1 H=8`, one KV pass) -- ATT traces a single
CU, so a grid that saturates the device buys nothing but decode time.
"""

import sys

import torch
from flash_attn_func_gfx950 import build_flash_attn_func_gfx950_module as build


def main():
    head_dim = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    seqlen = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    b, h = 1, 8

    torch.manual_seed(0)
    q, k, v = (torch.randn(b, h, seqlen, head_dim, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    o = torch.empty_like(q)
    fn = build(num_heads=h, head_dim=head_dim, causal=False, dtype_str="bf16")

    fn(q, k, v, o, b, seqlen)  # warm-up: compiles, and is not the traced dispatch
    torch.cuda.synchronize()

    fn(q, k, v, o, b, seqlen)  # the one that gets profiled
    torch.cuda.synchronize()
    print(f"profiled head_dim={head_dim} seqlen={seqlen}", file=sys.stderr)


if __name__ == "__main__":
    main()
