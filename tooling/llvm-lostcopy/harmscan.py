#!/usr/bin/env python
"""Middle predicate between lostdef.py (mechanism) and spillscan.py (descriptor base).

lostdef.py flags `v_writelane_b32 vN, sX, L` where sX has no reaching definition.
That alone is *not* a miscompile: LLVM may legally spill a partially-live tuple,
leaving the dead subregister undefined in the slot.  It only becomes a miscompile
if the lane is reloaded and the reloaded value is then *live* -- i.e. read before
it is redefined, on some path.

This scans for exactly that, with a proper CFG liveness for the reload's
destination register, and classifies every lost definition as LIVE or DEAD.

usage: LLVM_BIN=... harmscan.py <objects...>
"""
import re
import sys
from pathlib import Path

import lostdef as L
import undefscan as U

READLANE = re.compile(r's(\d+),\s*v(\d+),\s*(\d+)\s*$')


def liveness(insns, blocks, succs):
    """Classic backward liveness over SGPR numbers."""
    n = len(blocks)
    use, dfn = [set() for _ in range(n)], [set() for _ in range(n)]
    for b, (s, e) in enumerate(blocks):
        for i in range(s, e):
            ins = insns[i]
            for r in U.uses_of(ins):
                if r not in dfn[b]:
                    use[b].add(r)
            dfn[b] |= U.defs_of(ins)
    live_in = [set() for _ in range(n)]
    live_out = [set() for _ in range(n)]
    changed = True
    while changed:
        changed = False
        for b in range(n - 1, -1, -1):
            out = set()
            for s in succs[b]:
                out |= live_in[s]
            inn = use[b] | (out - dfn[b])
            if out != live_out[b] or inn != live_in[b]:
                live_out[b], live_in[b] = out, inn
                changed = True
    return live_in, live_out


def live_at(insns, blocks, succs, live_out, idx, reg):
    """Is `reg` live immediately after instruction index `idx`?"""
    for b, (s, e) in enumerate(blocks):
        if s <= idx < e:
            break
    else:
        return False
    for j in range(idx + 1, e):
        ins = insns[j]
        if reg in U.uses_of(ins):
            return True
        if reg in U.defs_of(ins):
            return False
    return reg in live_out[b]


def scan(path):
    bad = L.scan(path)
    if not bad:
        return []
    insns = U.disassemble(path)
    blocks, succs = U.build_blocks(insns)
    _, live_out = liveness(insns, blocks, succs)
    out = []
    for pc, ops in bad:
        m = L.W.search(ops)
        v, s, lane = int(m.group(1)), int(m.group(2)), int(m.group(3))
        reloads = []
        for i, ins in enumerate(insns):
            if not ins['op'].startswith('v_readlane_b32'):
                continue
            mm = READLANE.search(ins['ops'].strip())
            if not mm or int(mm.group(2)) != v or int(mm.group(3)) != lane:
                continue
            dst = int(mm.group(1))
            reloads.append((ins['pc'], dst, live_at(insns, blocks, succs, live_out, i, dst)))
        out.append((pc, v, s, lane, reloads))
    return out


def main():
    n_obj = n_live = 0
    for p in sys.argv[1:]:
        res = scan(p)
        if not res:
            continue
        n_obj += 1
        live_here = False
        print(f'### {Path(p).name}')
        for pc, v, s, lane, reloads in res:
            print(f'  0x{pc:08X}  v_writelane_b32 v{v}, s{s}, {lane}   (s{s} undefined)')
            if not reloads:
                print('      no reload of that lane')
            for rpc, dst, live in reloads:
                tag = 'LIVE  <<< consumed' if live else 'dead'
                live_here |= live
                print(f'      reload 0x{rpc:08X} -> s{dst}: {tag}')
        if live_here:
            n_live += 1
    print(f'\n{n_live} of {n_obj} flagged objects have a LIVE lost definition')


if __name__ == '__main__':
    main()
