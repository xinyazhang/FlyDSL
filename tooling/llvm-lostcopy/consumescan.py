#!/usr/bin/env python
"""Strictest ISA-level predicate: a lost definition that is *computed with*.

harmscan.py asks whether the reload of a lost lane is live.  That is still too
weak, because "live" includes being immediately written back to another spill
lane: LLVM shuffles a partially-undefined tuple around without ever using the
undefined half, and every real use recomputes it (e.g. `s_ashr_i32 s3, s2, 31`
regenerates a sign extension from the defined half).  Such an object carries the
lost definition but never observes it.

This propagates taint from the lost lane through copies and spill lanes and
reports the first consumer that is *not* a copy/spill -- an arithmetic or memory
instruction that actually reads the undefined value.

usage: LLVM_BIN=... consumescan.py <objects...>
"""
import re
import sys
from pathlib import Path

import lostdef as L
import spillscan as S
import undefscan as U

# instructions that only move a value around without observing it
MOVERS = ('v_writelane_b32', 'v_readlane_b32', 's_mov_b32', 's_mov_b64')

# VOP3B-style instructions whose *second* operand is a second destination
# (carry-out / scale-out), not a source.  undefscan.uses_of() treats every
# operand after the first as a source, which flags these spuriously.
TWO_DST = ('v_mad_u64_u32', 'v_mad_i64_i32', 'v_add_co_u32', 'v_sub_co_u32',
           'v_subrev_co_u32', 'v_div_scale_f32', 'v_div_scale_f64')


def uses_of(ins):
    """undefscan.uses_of, minus the second destination of VOP3B forms."""
    if not ins['op'].startswith(TWO_DST):
        return U.uses_of(ins)
    parts = [p.strip() for p in ins['ops'].split(',')]
    out = set()
    for s in parts[2:]:
        out |= U.regs(s)
    return out


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
            op = ins['op']
            if not op.startswith(MOVERS):
                bad = uses_of(ins) & tainted
                if bad:
                    findings.append((ins['pc'], op, ins['ops'], sorted(bad)))
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
            print(f'### {Path(p).name}  ({len(f)} consumers)')
            for pc, op, ops, bad in f[:6]:
                print(f'    0x{pc:08X}  {op} {ops}   reads lost: {bad}')
    print(f'\n{flagged} of {len(paths)} objects compute with a lost definition')


if __name__ == '__main__':
    main(sys.argv[1:])
