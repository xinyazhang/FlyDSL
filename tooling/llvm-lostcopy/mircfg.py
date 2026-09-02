#!/usr/bin/env python
"""Reachability-aware version of mirscan.

mirscan.py compares, per spill slot, the union of undefined lanes over *all*
stores against the lanes each restore reads.  That over-approximates: a store
that only happens after a restore (or on a disjoint path) is still counted, and
that produces false positives.

This does the real thing -- a must-analysis over the MachineIR CFG:

    state(slot) = set of lanes known-defined in the slot
    entry:      {}                       (slot uninitialised)
    merge:      intersection over predecessors
    SI_SPILL_*_SAVE  %v, slot            state[slot] := lanes-defined(%v)
    SI_SPILL_*_RESTORE %r, slot          report lanes-read(%r) - state[slot]

A non-empty result is a subregister that is read out of a spill slot on a path
where nothing ever wrote it: a lost definition that is consumed.

usage: mircfg.py <print-after-greedy dump> [...]
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import mirscan as M

BB = re.compile(r'^(?:(\d+)B\t)?bb\.(\d+)')
SUCC = re.compile(r'^\s*successors:(.*)')
SAVE = re.compile(r'SI_SPILL_(S\d+)_SAVE\s+%(\d+)[:,]')
SAVE_PHYS = re.compile(r'SI_SPILL_(S\d+)_SAVE\s+\$\w+')
RESTORE = re.compile(r'%(\d+):\S*\s*=\s*SI_SPILL_(S\d+)_RESTORE')
STACK = re.compile(r'%stack\.(\d+)')


def parse(sec):
    """-> (blocks: list of (bbid, [lines]), succ: {bbid: [bbid]}, defs, uses)"""
    blocks, succ = [], {}
    cur = None
    defs, uses = defaultdict(set), defaultdict(set)
    for ln in sec.splitlines():
        body = ln.split(' ; ')[0]
        m = BB.match(body.strip()) if not body.startswith('\t') else None
        m = BB.match(body) or (BB.match(body.strip()) if body.strip().startswith('bb.') else None)
        if m and ('bb.' in body.split(':')[0]):
            cur = int(m.group(2))
            blocks.append((cur, []))
            succ[cur] = []
            continue
        s = SUCC.match(body)
        if s and cur is not None:
            succ[cur] = [int(x) for x in re.findall(r'%bb\.(\d+)', s.group(1))]
            continue
        if cur is not None:
            blocks[-1][1].append(body)
        if '=' in body:
            lhs, _, rhs = body.partition('=')
        else:
            lhs, rhs = '', body
        for mm in M.VREG.finditer(lhs):
            defs[int(mm.group(1))].add(mm.group(2))
        for mm in M.VREG.finditer(rhs):
            uses[int(mm.group(1))].add(mm.group(2))
    return blocks, succ, defs, uses


def analyse(sec):
    blocks, succ, defs, uses = parse(sec)
    if not blocks:
        return []
    order = [b for b, _ in blocks]
    body = {b: ls for b, ls in blocks}
    preds = defaultdict(list)
    for b in order:
        for s in succ.get(b, []):
            preds[s].append(b)

    def lanes_of(reg, w, table):
        s = set()
        for name in table.get(reg, set()):
            s |= M.lanes(name, w)
        return s

    # collect slot events per block
    events = defaultdict(list)   # bb -> [(kind, slot, reg, width)]
    slots = set()
    for b in order:
        for ln in body[b]:
            st = STACK.search(ln)
            if not st:
                continue
            m = SAVE.search(ln)
            if m:
                w = M.CLASS_W.get(m.group(1), 0)
                events[b].append(('save', int(st.group(1)), int(m.group(2)), w))
                slots.add(int(st.group(1)))
                continue
            m = SAVE_PHYS.search(ln)
            if m:
                # a physical register source is fully defined by construction
                w = M.CLASS_W.get(m.group(1), 0)
                events[b].append(('save', int(st.group(1)), None, w))
                slots.add(int(st.group(1)))
                continue
            m = RESTORE.search(ln)
            if m:
                w = M.CLASS_W.get(m.group(2), 0)
                events[b].append(('restore', int(st.group(1)), int(m.group(1)), w))
                slots.add(int(st.group(1)))

    findings = []
    for slot in sorted(slots):
        # width for this slot
        w = 0
        for b in order:
            for k, s, r, ww in events[b]:
                if s == slot and ww:
                    w = max(w, ww)
        if not w:
            continue
        FULL = frozenset(range(w))
        TOP = FULL          # optimistic init for the fixpoint
        state_in = {b: (FULL if b != order[0] else frozenset()) for b in order}

        def transfer(b, inn):
            cur = set(inn)
            for k, s, r, ww in events[b]:
                if s != slot:
                    continue
                if k == 'save':
                    cur = set(range(w)) if r is None else lanes_of(r, w, defs)
            return frozenset(cur)

        for _ in range(len(order) + 4):
            changed = False
            for b in order:
                if b == order[0]:
                    continue
                ps = preds.get(b, [])
                if not ps:
                    new = frozenset()
                else:
                    new = frozenset.intersection(
                        *[transfer(p, state_in[p]) for p in ps])
                if new != state_in[b]:
                    state_in[b] = new
                    changed = True
            if not changed:
                break

        for b in order:
            cur = set(state_in[b])
            for k, s, r, ww in events[b]:
                if s != slot:
                    continue
                if k == 'save':
                    cur = set(range(w)) if r is None else lanes_of(r, w, defs)
                else:
                    read = set()
                    whole = False
                    for name in uses.get(r, set()):
                        if name is None:
                            whole = True
                        else:
                            read |= M.lanes(name, w)
                    miss = read - cur
                    if miss:
                        findings.append(dict(slot=slot, width=w, bb=b, restore=r,
                                             read=sorted(read), have=sorted(cur),
                                             missing=sorted(miss), whole=whole))
    return findings


def main():
    tot = 0
    for p in sys.argv[1:]:
        res = []
        for sec in M.sections(Path(p).read_text()):
            res += analyse(sec)
        if not res:
            continue
        tot += 1
        print(f'### {Path(p).name}')
        for f in res[:10]:
            print(f"  %stack.{f['slot']} (S{f['width']*32}) restore %{f['restore']} in bb.{f['bb']}: "
                  f"reads {f['read']}, defined-on-all-paths {f['have']} -> MISSING {f['missing']}")
    print(f'\n{tot} of {len(sys.argv)-1} dumps consume a spill lane with no reaching store')


if __name__ == '__main__':
    main()
