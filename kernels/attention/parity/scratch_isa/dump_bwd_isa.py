#!/usr/bin/env python3
"""Scratch: dump final ISA for one backward kernel at one config.

Modelled on tooling/dump_isa.py (same FLYDSL_DUMP_IR child-process trick), but
for the three backward entry points, which dump_isa.py does not cover.

    python3 dump_bwd_isa.py <repo-root> <dkdv|dq|fuse> <head_dim> <causal> <outdir> [--dtype f16|bf16]
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = "/home/xinyazha/dockerhome/meff/FlyDSL"


def _child(a):
    sys.path.insert(0, os.path.join(a.root, "kernels", "attention", "parity"))
    sys.path.insert(0, os.path.join(a.root, "kernels", "attention", "parity", "tooling"))
    import torch
    from flash_attn_func_gfx1201_interface import flydsl_flash_attn_func_gfx1201
    from qkv import make_qkv

    dtype = {"f16": torch.float16, "bf16": torch.bfloat16}[a.dtype]
    q, k, v = make_qkv(a.batch, a.heads, a.seq, a.head_dim, dtype=dtype, layout=a.layout, seed=0)
    do = make_qkv(a.batch, a.heads, a.seq, a.head_dim, dtype=dtype, layout=a.layout, seed=1, n=1)
    o, lse = flydsl_flash_attn_func_gfx1201(q, k, v, causal=bool(a.causal), return_lse=True)
    lse2 = lse.reshape(a.batch * a.heads, a.seq).contiguous()
    delta = (do.float() * o.float()).sum(-1).reshape(a.batch * a.heads, a.seq).contiguous()
    torch.cuda.synchronize()

    # Mark: everything dumped after this point belongs to the backward kernel.
    open(os.environ["FLYDSL_DUMP_DIR"] + "/.mark", "w").close()

    if a.kernel == "dkdv":
        from fmha_bwd_dkdv_gfx1201_interface import flydsl_flash_attn_bwd_dkdv_gfx1201 as K

        K(q, k, v, do, o, lse2, causal=bool(a.causal), delta=delta)
    elif a.kernel == "dq":
        from fmha_bwd_dq_gfx1201_interface import flydsl_bwd_dq_gfx1201 as K

        K(q, k, v, o, do, lse2, causal=bool(a.causal), delta=delta)
    else:
        from fmha_bwd_fuse_gfx1201_interface import flydsl_fmha_bwd_fuse_gfx1201 as K

        K(q, k, v, o, do, lse2, causal=bool(a.causal), delta=delta)
    torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("kernel", choices=("dkdv", "dq", "fuse"))
    p.add_argument("head_dim", type=int)
    p.add_argument("causal", type=int)
    p.add_argument("outdir")
    p.add_argument("--dtype", default="f16", choices=("f16", "bf16"))
    p.add_argument("--layout", default="bhsd")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--child", action="store_true")
    a = p.parse_args()

    if a.child:
        _child(a)
        return 0

    d = tempfile.mkdtemp()
    env = dict(os.environ, FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=d, FLYDSL_RUNTIME_ENABLE_CACHE="0")
    r = subprocess.run([sys.executable, __file__, *sys.argv[1:], "--child"], env=env, capture_output=True, timeout=3600)
    if r.returncode != 0:
        print(r.stderr.decode()[-4000:], file=sys.stderr)
        return 1
    mark = os.path.getmtime(os.path.join(d, ".mark"))
    got = sorted(glob.glob(os.path.join(d, "*", "*21_final_isa.s*")))
    os.makedirs(a.outdir, exist_ok=True)
    n = 0
    for f in got:
        if os.path.getmtime(f) < mark:
            continue  # forward kernel, dumped before the mark
        tag = f"{a.kernel}_hd{a.head_dim}_{'c' if a.causal else 'nc'}_{a.dtype}"
        dst = os.path.join(a.outdir, f"{tag}__{os.path.basename(os.path.dirname(f))}.s")
        shutil.copy(f, dst)
        print(dst)
        n += 1
    if n == 0:
        print(f"no post-mark stage in {d}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
