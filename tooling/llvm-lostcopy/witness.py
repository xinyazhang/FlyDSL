#!/usr/bin/env python
"""Print a concrete CFG path from a partial (lane-incomplete) spill store to a
restore that reads a lane the store never defined.

This is the hand-verification step for a mircfg.py hit: it names the basic
blocks and the branch out of each one, so the claim "there is a path on which
lane N is never written" can be checked by reading the dump.

usage: witness.py <dump> <slot> [--section N]
"""
import argparse
import re
from collections import deque
from pathlib import Path

import mircfg as C
import mirscan as M


def run(path, slot):
    for sec in M.sections(Path(path).read_text()):
        if 'SI_SPILL' not in sec:
            continue
        blocks, succ, defs, uses = C.parse(sec)
        if not blocks:
            continue
        order = [b for b, _ in blocks]
        body = {b: ls for b, ls in blocks}
        w = 0
        ev = {}
        for b in order:
            lst = []
            for i, ln in enumerate(body[b]):
                st = C.STACK.search(ln)
                if not st or int(st.group(1)) != slot:
                    continue
                m = C.SAVE.search(ln)
                if m:
                    ww = M.CLASS_W.get(m.group(1), 0)
                    w = max(w, ww)
                    d = set()
                    for name in defs.get(int(m.group(2)), set()):
                        d |= M.lanes(name, ww)
                    lst.append((i, 'save', int(m.group(2)), d, ln.strip()))
                    continue
                if C.SAVE_PHYS.search(ln):
                    ww = M.CLASS_W.get(C.SAVE_PHYS.search(ln).group(1), 0)
                    w = max(w, ww)
                    lst.append((i, 'save', None, set(range(ww)), ln.strip()))
                    continue
                m = C.RESTORE.search(ln)
                if m:
                    ww = M.CLASS_W.get(m.group(2), 0)
                    w = max(w, ww)
                    rd = set()
                    for name in uses.get(int(m.group(1)), set()):
                        if name:
                            rd |= M.lanes(name, ww)
                    lst.append((i, 'restore', int(m.group(1)), rd, ln.strip()))
            ev[b] = lst

        # candidate partial stores and reading restores
        partial = [(b, e) for b in order for e in ev[b]
                   if e[1] == 'save' and e[3] != set(range(w))]
        readers = [(b, e) for b in order for e in ev[b]
                   if e[1] == 'restore' and e[3]]
        if not partial or not readers:
            continue

        for pb, pe in partial:
            missing_lane_set = set(range(w)) - pe[3]
            for rb, re_ in readers:
                need = re_[3] & missing_lane_set
                if not need:
                    continue
                p = bfs(pb, pe[0], rb, re_[0], order, succ, ev, slot, w)
                if p:
                    print(f'slot %stack.{slot}  lanes {sorted(need)} never written on this path')
                    print(f'  store  bb.{pb}: {pe[4][:130]}')
                    print(f'          defines lanes {sorted(pe[3])}')
                    print(f'  path   {" -> ".join("bb.%d" % x for x in p)}')
                    print(f'  restore bb.{rb}: {re_[4][:130]}')
                    print(f'          reads lanes {sorted(re_[3])}')
                    return True
    return False


def bfs(sb, si, rb, ri, order, succ, ev, slot, w):
    """Path from (sb,si) to (rb,ri) with no full store to `slot` in between."""
    FULL = set(range(w))

    def full_after(b, lo, hi):
        for i, kind, reg, lanes, _ln in ev.get(b, []):
            if kind == 'save' and lanes == FULL and lo < i < hi:
                return True
        return False

    if sb == rb and si < ri and not full_after(sb, si, ri):
        return [sb]
    if full_after(sb, si, 10 ** 9):
        start_ok = False
    else:
        start_ok = True
    if not start_ok:
        return None
    prev = {sb: None}
    q = deque([sb])
    while q:
        b = q.popleft()
        for s in succ.get(b, []):
            if s in prev:
                continue
            if s == rb:
                if not full_after(s, -1, ri):
                    prev[s] = b
                    out = []
                    x = s
                    while x is not None:
                        out.append(x)
                        x = prev[x]
                    return out[::-1]
                continue
            if full_after(s, -1, 10 ** 9):
                continue
            prev[s] = b
            q.append(s)
    return None


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dump')
    ap.add_argument('slot', type=int)
    a = ap.parse_args()
    if not run(a.dump, a.slot):
        print('no witness path found')
