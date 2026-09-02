#!/usr/bin/env python
"""Run every .ll through `amdclang -mllvm -print-after=greedy` and classify it
with mirscan.

Categories (see REPORT-triton-gfx950-lost-sgpr-copy.md):
  cat2  a spill slot is stored from a value with undefined lanes, but no restore
        reads those lanes           -> partial-live-tuple spill, benign
  cat3  a restore reads a lane that no store into the slot ever defined
        -> lost definition consumed = miscompile

usage: mirsweep.py <dir-with-.ll> [-j N] [--func NAME]
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import mircfg  # noqa: E402
import mirscan  # noqa: E402

BIN = os.environ.get("LLVM_BIN", "")
CLANG = os.path.join(BIN, "amdclang") if BIN else "amdclang"


def one(path):
    p = Path(path)
    cmd = [
        CLANG,
        "-x",
        "ir",
        "-S",
        "--target=amdgcn-amd-amdhsa",
        "-mcpu=gfx950",
        "-O3",
        str(p),
        "-o",
        "/dev/null",
        "-mllvm",
        "-print-after=greedy",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    txt = r.stderr + r.stdout
    if "IR Dump After" not in txt:
        return path, None, None, (r.stderr or "")[:200]
    res, partial = [], {}
    for sec in mirscan.sections(txt):
        _f, pa, _si = mirscan.analyse_section(sec)
        partial.update(pa)
        res += mircfg.analyse(sec)
    return path, len(res), len(partial), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("-j", type=int, default=max(1, os.cpu_count() // 4))
    a = ap.parse_args()
    files = sorted(str(p) for p in Path(a.dir).glob("*.ll"))
    print(f"{len(files)} modules", flush=True)
    n2 = n3 = err = 0
    hits = []
    with ProcessPoolExecutor(max_workers=a.j) as ex:
        for path, c3, c2, e in ex.map(one, files, chunksize=4):
            if e is not None:
                err += 1
                continue
            if c2:
                n2 += 1
            if c3:
                n3 += 1
                hits.append((path, c3))
    print(f"cat2 (partial-undef spill, not read back): {n2}")
    print(f"cat3 (undefined lane CONSUMED -- miscompile): {n3}")
    print(f"errors: {err}")
    for h, c in hits[:40]:
        print(f"  {h}  ({c} hits)")


if __name__ == "__main__":
    main()
