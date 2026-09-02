"""Find kernels where a lost SGPR definition reaches a buffer descriptor that is used.

Neither half of this is a defect predicate on its own:

  * Reading an undefined SGPR is legal and common -- LLVM emits such reads on paths
    where the value is dead.  Scanning for them flags a third of the tree.
  * Propagating taint from every undefined read saturates for the same reason: the
    dead-path undefs feed selects that feed everything.

Their conjunction is sharp.  A spill is the compiler asserting the value is live and
will be reloaded, so `v_writelane_b32 vN, sX, L` with no reaching definition of sX is
a lost definition rather than a dead-path undef; and a descriptor *base* built from
one is unsurvivable, because buffer bounds clamping only constrains the offset.

That conjunction is the proven shape of the gfx950 dkdv miscompile: the
`s_mov_b64 s[60:61], s[80:81]` copy carrying stride_do_head went missing from the
chain out of an s_load_dwordx16, the undefined pair was spilled to lanes 4/5 of v254,
reloaded as the DO row-slab head-stride multiplier, and the resulting base was
dereferenced by a buffer_load.

Taint is seeded only at such spills and only flows through SGPRs and constant-index
VGPR spill lanes -- the kernels set private_segment_fixed_size to 0, so lanes are the
whole of the spill storage.
"""
import re
import sys
from pathlib import Path

import undefscan as U            # must sit in the same directory

WRITELANE = re.compile(r'v(\d+),\s*s(\d+),\s*(\d+)')
READLANE = re.compile(r's(\d+),\s*v(\d+),\s*(\d+)')


def must_defined(insns, blocks, preds, preinit):
    gen = [set().union(*(U.defs_of(insns[i]) for i in range(s, e))) if e > s else set()
           for s, e in blocks]
    entry = frozenset(range(preinit))
    inn = [entry if b == 0 else U.ALL for b in range(len(blocks))]
    for _ in range(len(blocks) + 2):
        changed = False
        for b in range(1, len(blocks)):
            new = U.ALL if not preds[b] else U.ALL.intersection(
                *(inn[p] | gen[p] for p in preds[b]))
            if new != inn[b]:
                inn[b], changed = new, True
        if not changed:
            break
    return inn


def run_block(insns, span, defined, state):
    """Advance taint through one block. `state` is (tainted sgprs, tainted lanes)."""
    tainted, lanes = set(state[0]), dict(state[1])
    live = set(defined)
    for i in range(*span):
        ins = insns[i]
        op, ops = ins['op'], ins['ops']
        m = WRITELANE.search(ops) if op.startswith('v_writelane_b32') else None
        if m:
            src = int(m.group(2))
            lost = src not in live                      # a lost definition, not dead undef
            lanes[(int(m.group(1)), int(m.group(3)))] = lost or src in tainted
            live.add(src)                               # do not re-report downstream
            continue
        m = READLANE.search(ops) if op.startswith('v_readlane_b32') else None
        if m:
            dst = int(m.group(1))
            live.add(dst)
            if lanes.get((int(m.group(2)), int(m.group(3))), False):
                tainted.add(dst)
            else:
                tainted.discard(dst)
            continue
        used = U.uses_of(ins)
        dirty = bool(used & tainted)
        for r in U.defs_of(ins):
            live.add(r)
            tainted.add(r) if dirty else tainted.discard(r)
    return tainted, lanes


def scan(path):
    insns = U.disassemble(path)
    if not insns:
        return []
    blocks, succs = U.build_blocks(insns)
    preds = [[] for _ in blocks]
    for b, out in enumerate(succs):
        for s in out:
            preds[s].append(b)
    inn = must_defined(insns, blocks, preds, U.preinit_count(path))

    state = [(set(), {}) for _ in blocks]
    for _ in range(len(blocks) + 2):
        changed = False
        for b in range(len(blocks)):
            if b == 0 or not preds[b]:
                new = (set(), {})
            else:
                outs = [run_block(insns, blocks[p], inn[p], state[p]) for p in preds[b]]
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
            if ins['op'].startswith('buffer_'):
                rsrc = U.srsrc_of(ins)
                base = {min(rsrc), min(rsrc) + 1} if rsrc else set()
                if base & tainted:
                    findings.append((ins['pc'], ins['op'], ins['ops'], sorted(base & tainted)))
            tainted, lanes = run_block(insns, (i, i + 1), live, (tainted, lanes))
            live |= U.defs_of(ins)
    return findings


def main(paths):
    flagged = 0
    for p in paths:
        try:
            f = scan(p)
        except Exception as exc:                     # noqa: BLE001
            print(f'### {Path(p).name}  SCAN ERROR {exc}')
            continue
        if f:
            flagged += 1
            print(f'### {Path(p).name}  ({len(f)} accesses)')
            for pc, op, ops, bad in f[:4]:
                print(f'    0x{pc:08X}  {op} {ops}   base from lost def: {bad}')
    print(f'\n{flagged} of {len(paths)} objects dereference a descriptor built from a lost definition')


if __name__ == '__main__':
    main(sys.argv[1:])
