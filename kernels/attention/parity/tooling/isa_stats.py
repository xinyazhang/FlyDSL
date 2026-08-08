"""ISA stats keyed on the *KV loop*, identified by containing WMMA.

Written because a size-based "largest backward-branch region" heuristic picked
different loops in two builds of the same kernel and produced a fictitious
20-instruction regression.
"""

import collections
import re
import sys


def _loops(lines):
    lab = {}
    for i, ln in enumerate(lines):
        m = re.match(r"^(\.LBB[\w\.]+):", ln.strip())
        if m:
            lab[m.group(1)] = i
    for i, ln in enumerate(lines):
        m = re.search(r"\bs_(cbranch\w*|branch)\s+(\.LBB[\w\.]+)", ln)
        if m and m.group(2) in lab and lab[m.group(2)] < i:
            yield lab[m.group(2)], i


def kv_loop(path):
    """Instruction list of the smallest loop that contains a WMMA."""
    lines = [ln.rstrip() for ln in open(path)]
    best = None
    for a, b in _loops(lines):
        body = [x for x in lines[a : b + 1] if re.match(r"^\s+[a-z]", x)]
        if any(re.match(r"^\s+v_wmma", x) for x in body):
            if best is None or len(body) < len(best):
                best = body
    if best is None:
        raise SystemExit(f"{path}: no WMMA loop found")
    return best


def hist(body):
    c = collections.Counter()
    for ln in body:
        c[re.match(r"^\s+([a-z][\w.]*)", ln).group(1)] += 1
    return c


def summary(path):
    t = open(path).read()
    g = lambda k: int(re.search(rf"\.{k}:\s*(\d+)", t).group(1))
    body = kv_loop(path)
    return dict(
        vgpr=g("vgpr_count"),
        sgpr=g("sgpr_count"),
        scratch=g("private_segment_fixed_size"),
        total=sum(1 for ln in t.split("\n") if re.match(r"^\s+[a-z]", ln)),
        loop=len(body),
        wmma=sum(1 for ln in body if re.match(r"^\s+v_wmma", ln)),
    )


if __name__ == "__main__":
    a, b = sys.argv[1], sys.argv[2]
    sa, sb = summary(a), summary(b)
    print(f"  {'':10s} {'vgpr':>5} {'sgpr':>5} {'scratch':>8} {'total':>6} {'loop':>5} {'wmma':>5}")
    for tag, s in (("base", sa), ("head", sb)):
        print(
            f"  {tag:10s} {s['vgpr']:5d} {s['sgpr']:5d} {s['scratch']:8d} "
            f"{s['total']:6d} {s['loop']:5d} {s['wmma']:5d}"
        )
    ha, hb = hist(kv_loop(a)), hist(kv_loop(b))
    d = [(k, ha[k], hb[k]) for k in sorted(set(ha) | set(hb)) if ha[k] != hb[k]]
    print("  KV-loop opcode deltas:", d if d else "NONE (register naming only)")
