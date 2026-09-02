#!/usr/bin/env python
"""Compile a list of Triton configurations (AOTriton `Bare.compile` format or a
plain signature list) and report lost definitions + whether any is LIVE.

usage:
    sweep.py --shard <Bare.compile> --out <dir> [--stride N] [--limit N] [-j N]
    sweep.py --module k.py --kernel name --sigs sigs.txt --out <dir>
"""

import argparse
import os
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def parse_shard(path):
    """AOTriton Bare.compile: tag;python;out.hsaco;src.py;kernel;NW;NS;WPEU;arch;signature"""
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        f = line.split(";")
        if len(f) < 10:
            continue
        out.append(
            dict(
                src=f[3],
                kernel=f[4],
                warps=int(f[5]),
                stages=int(f[6]),
                wpeu=int(f[7]),
                arch=f[8],
                sig=";".join(f[9:]).strip(),
            )
        )
    return out


def one(job):
    idx, cfg, outdir = job
    from triton_repro import compile_one

    stem = f"{cfg['kernel']}_{idx:05d}"
    try:
        p = compile_one(
            cfg["src"], cfg["kernel"], cfg["sig"], cfg["arch"], cfg["warps"], cfg["stages"], cfg["wpeu"], outdir, stem
        )
        return idx, str(p), None
    except Exception:
        return idx, None, traceback.format_exc(limit=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("-j", type=int, default=os.cpu_count() // 2)
    a = ap.parse_args()

    cfgs = parse_shard(a.shard)
    sel = list(enumerate(cfgs))[a.offset :: a.stride]
    if a.limit:
        sel = sel[: a.limit]
    print(f"{len(cfgs)} configs, compiling {len(sel)}", flush=True)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    ok, fail = [], 0
    with ProcessPoolExecutor(max_workers=a.j) as ex:
        for idx, p, err in ex.map(one, [(i, c, a.out) for i, c in sel]):
            if p:
                ok.append(p)
            else:
                fail += 1
    print(f"compiled {len(ok)}, failed {fail}", flush=True)
    if not ok:
        return
    env = dict(os.environ)
    r = subprocess.run(
        [sys.executable, str(HERE / "harmscan.py")] + ok, capture_output=True, text=True, env=env, cwd=str(HERE)
    )
    print(r.stdout)
    print(r.stderr[-2000:] if r.returncode else "", file=sys.stderr)


if __name__ == "__main__":
    main()
