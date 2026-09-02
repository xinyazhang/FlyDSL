#!/usr/bin/env python
"""MachineIR-level check, independent of every disassembly heuristic.

Reads a `-print-after=greedy` dump and, per SGPR spill slot, compares

    the lanes actually *defined* on each value stored into the slot
    against
    the lanes *read* out of each value restored from that slot.

A lane that is read but that some store into the slot never defined is a
subregister with no definition anywhere in the function -- the shape reported in
REPORT-gfx950-splitkit-lost-subrange.md §1/§2.

Nothing here depends on preinit_count, on the CFG llvm-objdump implies, or on a
reaching-definition approximation: "%N is spilled whole but only %N.sub1 is ever
defined" is a syntactic fact of the dump.

usage: mirscan.py [-v] <print-after-greedy dump> [...]
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

SUB = re.compile(r'sub(\d+)')
VREG = re.compile(r'%(\d+)(?:\.(sub[0-9_a-z]+))?')
SPILL_SAVE = re.compile(r'SI_SPILL_(S\d+)_SAVE\s+%(\d+)[:,]')
SPILL_RESTORE = re.compile(r'%(\d+):\S*\s*=\s*SI_SPILL_(S\d+)_RESTORE')
STACK = re.compile(r'%stack\.(\d+)')
CLASS_W = {'S32': 1, 'S64': 2, 'S96': 3, 'S128': 4, 'S160': 5, 'S192': 6,
           'S224': 7, 'S256': 8, 'S288': 9, 'S320': 10, 'S352': 11, 'S384': 12,
           'S512': 16, 'S1024': 32}


def lanes(subname, width):
    if not subname:
        return set(range(width))
    return {int(x) for x in SUB.findall(subname)}


def sections(text):
    parts = re.split(r'^# \*\*\* IR Dump After .*$', text, flags=re.M)
    return [p for p in parts if p.strip()]


def analyse_section(sec):
    defs = defaultdict(set)     # vreg -> set of raw subreg names (None = whole)
    uses = defaultdict(set)
    saves = []                  # (slot, vreg, width)
    restores = []
    for ln in sec.splitlines():
        body = ln.split(' ; ')[0]
        m = SPILL_SAVE.search(body)
        st = STACK.search(body)
        if m and st:
            saves.append((int(st.group(1)), int(m.group(2)), CLASS_W.get(m.group(1), 0)))
        m = SPILL_RESTORE.search(body)
        if m and st:
            restores.append((int(st.group(1)), int(m.group(1)), CLASS_W.get(m.group(2), 0)))
        if '=' in body:
            lhs, _, rhs = body.partition('=')
        else:
            lhs, rhs = '', body
        for m in VREG.finditer(lhs):
            defs[int(m.group(1))].add(m.group(2))
        for m in VREG.finditer(rhs):
            uses[int(m.group(1))].add(m.group(2))

    def expand(raw, w):
        s = set()
        for name in raw:
            s |= lanes(name, w)
        return s

    # undefined lanes per slot: union over every store into the slot
    undef_by_slot = defaultdict(set)
    save_info = defaultdict(list)
    for slot, reg, w in saves:
        if not w:
            continue
        d = expand(defs.get(reg, set()), w)
        u = set(range(w)) - d
        save_info[slot].append((reg, sorted(d), sorted(u)))
        undef_by_slot[slot] |= u

    partial = {s: v for s, v in undef_by_slot.items() if v}
    findings = []
    for slot, reg, w in restores:
        if not w or not undef_by_slot.get(slot):
            continue
        raw = uses.get(reg, set())
        sub_reads = set()
        whole = False
        for name in raw:
            if name is None:
                whole = True
            else:
                sub_reads |= lanes(name, w)
        hit = sub_reads & undef_by_slot[slot]
        if hit or (whole and undef_by_slot[slot]):
            findings.append(dict(slot=slot, restore=reg, width=w,
                                 undef=sorted(undef_by_slot[slot]),
                                 sub_reads=sorted(sub_reads), whole=whole,
                                 hit=sorted(hit), saves=save_info[slot]))
    return findings, partial, save_info


def main():
    argv = sys.argv[1:]
    verbose = '-v' in argv
    argv = [a for a in argv if a != '-v']
    total = 0
    for p in argv:
        res, partial, save_info = [], {}, {}
        for sec in sections(Path(p).read_text()):
            r, pa, si = analyse_section(sec)
            res += r
            partial.update(pa)
            save_info.update(si)
        strong = [f for f in res if f['hit']]
        if not res and not partial:
            continue
        total += 1 if strong else 0
        print(f'### {Path(p).name}   {len(strong)} subregister-read hits, '
              f'{len(res) - len(strong)} whole-register-read only, '
              f'{len(partial)} slots stored with undefined lanes')
        if verbose and not res:
            for slot, u in partial.items():
                print(f'  %stack.{slot}: stored with UNDEFINED lanes {sorted(u)} '
                      f'(never read back) {save_info.get(slot)}')
        for f in (strong if not verbose else res)[:12]:
            print(f"  %stack.{f['slot']} (S{f['width'] * 32}):")
            for reg, d, u in f['saves']:
                print(f"      store %{reg}: defines lanes {d}, UNDEFINED {u}")
            print(f"      restore %{f['restore']}: reads subregs {f['sub_reads']}"
                  f"{' + whole-register use' if f['whole'] else ''}"
                  f"  -> consumes undefined {f['hit']}")
    print(f'\n{total} of {len(argv)} dumps read a spilled subregister that no store defined')


if __name__ == '__main__':
    main()
