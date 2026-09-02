"""Standalone check of the handoff's proposed assertion:
   no v_writelane_b32 may name a source SGPR with no reaching definition.
Independent of spillscan's taint/descriptor conjunction."""
import re, sys
from pathlib import Path
import undefscan as U
import spillscan as S

W = re.compile(r'v(\d+),\s*s(\d+),\s*(\d+)')

def scan(path):
    insns = U.disassemble(path)
    blocks, succs = U.build_blocks(insns)
    preds = [[] for _ in blocks]
    for b, out in enumerate(succs):
        for s in out:
            preds[s].append(b)
    inn = S.must_defined(insns, blocks, preds, U.preinit_count(path))
    bad = []
    for b, (s, e) in enumerate(blocks):
        live = set(inn[b])
        for i in range(s, e):
            ins = insns[i]
            if ins['op'].startswith('v_writelane_b32'):
                m = W.search(ins['ops'])
                if m and int(m.group(2)) not in live:
                    bad.append((ins['pc'], ins['ops']))
            live |= U.defs_of(ins)
    return bad

def main():
  for p in sys.argv[1:]:
    b = scan(p)
    print(f"{Path(p).name}: {len(b)} writelane(s) with no reaching definition")
    for pc, ops in b[:8]:
        print(f"    0x{pc:08X}  v_writelane_b32 {ops}")

if __name__ == "__main__":
    main()
