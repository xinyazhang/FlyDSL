#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Dump a compilation stage for one forward configuration, from a given checkout.

    python3 dump_isa.py <repo-root> <head_dim> <causal> <out> [options]

    --dtype f16|bf16     operand dtype (default f16)
    --pitch N            pad the D allocation to N and slice, for PADDED_HEAD
    --layout bhsd|bshd   memory layout; the shape is BHSD either way
    --fp-mode MODE       fp_mode knob, if not the tuning default
    --stage NAME         a filename fragment; default `21_final_isa.s`.
                         `00_origin.mlir` is the pre-lowering IR.
    --heads H --seq S --batch B

**Why this replaced four scripts.** There were separate `dump_isa`,
`dump_isa_pad`, `dump_isa_bf16` and `dump_mlir` copies, and they drifted: after
the ABI moved to BHSD all four still built `(1, 512, 8, D)`, so they were
dumping num_heads=512/seq_len=8 while their callers labelled them 8/512. The
comparisons stayed valid -- both sides of a diff used the same config -- but
nobody could tell from the output which config that was. Four copies of a
config builder is four chances for one of them to be the stale one.

Runs the build in a child process so `FLYDSL_DUMP_IR` applies to a fresh
interpreter, and so a crash in the JIT does not take the caller with it.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile


def _child(a):
    sys.path.insert(0, os.path.join(a.root, "kernels", "attention", "parity"))
    sys.path.insert(0, os.path.join(a.root, "kernels", "attention", "parity", "tooling"))
    import torch
    from flash_attn_func_gfx1201_interface import flydsl_flash_attn_func_gfx1201 as F
    from fmha_tuning_gfx1201 import FmhaKnobs
    from qkv import make_qkv

    dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[a.dtype]
    q, k, v = make_qkv(a.batch, a.heads, a.seq, a.head_dim, dtype=dtype, layout=a.layout, seed=0, pitch=a.pitch)
    knobs = FmhaKnobs(fp_mode=a.fp_mode) if a.fp_mode else None
    F(q, k, v, causal=bool(a.causal), knobs=knobs)
    torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("head_dim", type=int)
    p.add_argument("causal", type=int)
    p.add_argument("out")
    p.add_argument("--dtype", default="f16", choices=("f16", "bf16"))
    p.add_argument("--pitch", type=int, default=None)
    p.add_argument("--layout", default="bhsd", choices=("bhsd", "bshd"))
    p.add_argument("--fp-mode", dest="fp_mode", default=None)
    p.add_argument("--stage", default="21_final_isa.s")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--child", action="store_true")
    a = p.parse_args()

    if a.child:
        _child(a)
        return 0

    # The dump tree is ~25MB and every early `return 1` below used to leak one.
    # See `check_exec_hazard_gfx950.build_isa` for what that cost in practice.
    d = tempfile.mkdtemp(prefix="dump_isa_")
    try:
        env = dict(os.environ, FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=d, FLYDSL_RUNTIME_ENABLE_CACHE="0")
        r = subprocess.run(
            [sys.executable, __file__, *sys.argv[1:], "--child"], env=env, capture_output=True, timeout=1800
        )
        if r.returncode != 0:
            print(r.stderr.decode()[-3000:], file=sys.stderr)
            return 1
        got = sorted(glob.glob(os.path.join(d, "*", f"*{a.stage}*")))
        if not got:
            print(f"no stage matching {a.stage!r} in {d}", file=sys.stderr)
            return 1
        shutil.copy(got[0], a.out)
        print(a.out)
        return 0
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
