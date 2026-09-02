#!/usr/bin/env python
"""Narrow predicate for Triton-compiled gfx950 code.

spillscan.py asks whether a lost SGPR definition reaches the *base* of a buffer
resource that is then dereferenced.  Triton's AMD backend does not emit buffer
descriptors for these loads -- it emits `global_load`/`global_store` with a
64-bit SGPR base (saddr) plus a VGPR offset -- so spillscan structurally cannot
fire on Triton output, and a 0 from it means nothing there.

This is the same conjunction against the operand Triton actually uses:

    lost definition -> spilled to a lane -> reloaded -> used as the saddr of a
    global_* access.

It is strictly worse than the buffer case: a buffer descriptor at least clamps
the offset against num_records, whereas global_* has no bounds check at all.

Operand order (gfx9 flat-global encoding as llvm-objdump prints it):
    global_load_dwordx4 vdst, vaddr, saddr
    global_store_dwordx4 vaddr, vdata, saddr
so the SGPR pair, when present, is the last operand.

usage: LLVM_BIN=... globalscan.py <objects...>
"""
import re
import sys
from pathlib import Path

import spillscan as S
import undefscan as U

PAIR = re.compile(r's\[(\d+):(\d+)\]\s*$')


def saddr_of(ins):
    """The 64-bit SGPR base of a global_* instruction, or empty."""
    parts = [p.strip() for p in ins['ops'].split(',')]
    if not parts:
        return set()
    m = PAIR.match(parts[-1])
    if not m:
        return set()
    lo, hi = int(m.group(1)), int(m.group(2))
    return {lo, hi} if hi - lo == 1 else set()


def scan(path):
    insns = U.disassemble(path)
    if not insns:
        return []
    blocks, succs = U.build_blocks(insns)
    preds = [[] for _ in blocks]
    for b, out in enumerate(succs):
        for s in out:
            preds[s].append(b)
    inn = S.must_defined(insns, blocks, preds, U.preinit_count(path))

    state = [(set(), {}) for _ in blocks]
    for _ in range(len(blocks) + 2):
        changed = False
        for b in range(len(blocks)):
            if b == 0 or not preds[b]:
                new = (set(), {})
            else:
                outs = [S.run_block(insns, blocks[p], inn[p], state[p]) for p in preds[b]]
                merged = {}
                for _t, lns in outs:
                    for k, v in lns.items():
                        merged[k] = merged.get(k, False) or v
                new = (set().union(*(t for t, _l in outs)), merged)
            if new != state[b]:
                state[b], changed = new, True
        if not changed:
            break

    findings = []
    for b, (s, e) in enumerate(blocks):
        tainted, lanes = set(state[b][0]), dict(state[b][1])
        live = set(inn[b])
        for i in range(s, e):
            ins = insns[i]
            if ins['op'].startswith('global_'):
                base = saddr_of(ins)
                if base & tainted:
                    findings.append((ins['pc'], ins['op'], ins['ops'], sorted(base & tainted)))
            tainted, lanes = S.run_block(insns, (i, i + 1), live, (tainted, lanes))
            live |= U.defs_of(ins)
    return findings


def main(paths):
    flagged = 0
    for p in paths:
        try:
            f = scan(p)
        except Exception as exc:  # noqa: BLE001
            print(f'### {Path(p).name}  SCAN ERROR {exc}')
            continue
        if f:
            flagged += 1
            print(f'### {Path(p).name}  ({len(f)} accesses)')
            for pc, op, ops, bad in f[:4]:
                print(f'    0x{pc:08X}  {op} {ops}   saddr from lost def: {bad}')
    print(f'\n{flagged} of {len(paths)} objects address a global_* access '
          f'with a base built from a lost definition')


if __name__ == '__main__':
    main(sys.argv[1:])
