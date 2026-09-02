#!/usr/bin/env python
"""Wrap an `mlir-translate --import-llvm` result in a gpu.module carrying an
#rocdl.target, so it can be fed to gpu-module-to-binary -- i.e. so a Triton
.ll can be compiled by the *pinned* LLVM inside the flydsl wheel.

usage: wrap_gpu_module.py <imported.mlir> <original.ll> <out.mlir> [chip]
"""

import re
import sys
from pathlib import Path

imp, ll, out = sys.argv[1], sys.argv[2], sys.argv[3]
chip = sys.argv[4] if len(sys.argv) > 4 else "gfx950"

dl = ""
for line in Path(ll).read_text().splitlines():
    m = re.match(r'target datalayout = "(.*)"', line)
    if m:
        dl = m.group(1)
        break

lines = Path(imp).read_text().splitlines()
i = next(k for k, ln in enumerate(lines) if ln.startswith("module"))
head = lines[:i]
body = lines[i + 1 :]
# drop the module's closing brace (last line that is exactly "}")
j = len(body) - 1
while body[j].strip() != "}":
    j -= 1
body = body[:j] + body[j + 1 :]

wrapped = (
    head
    + [
        "module {",
        f'  gpu.module @kernels [#rocdl.target<chip = "{chip}">] ' f'attributes {{llvm.data_layout = "{dl}"}} {{',
    ]
    + body
    + ["  }", "}"]
)
Path(out).write_text("\n".join(wrapped) + "\n")
print("wrote", out)
