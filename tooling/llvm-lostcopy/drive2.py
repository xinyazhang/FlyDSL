#!/usr/bin/env python
"""drive.py with a configurable library path plus MachineIR capture.

Compiles the FlyDSL ROCDL-dialect reproducer module with the *pinned* LLVM
(24.0.0git @ e2a39f504fee) that ships inside the flydsl wheel, optionally
turning on -print-after=<pass> through the same cl::opt mechanism the driver
uses.  MachineIR goes to stderr; redirect it.

usage: drive2.py <in.mlir> <out.s> [name=value ...]
e.g.   drive2.py mod.mlir out.s print-after=greedy filter-print-funcs=flyc_bwd_dkdv
"""
import ctypes
import os
import sys
from pathlib import Path

import flydsl  # noqa: F401  (locates the wheel)

LIB = os.environ.get(
    "FLYDSL_CAPI",
    str(Path(flydsl.__file__).parent / "_mlir/_mlir_libs/libFlyPythonCAPI.so.24.0git"),
)
from flydsl._mlir import ir  # noqa: E402
from flydsl._mlir.passmanager import PassManager  # noqa: E402

lib = ctypes.CDLL(LIB, mode=ctypes.RTLD_GLOBAL)
lib.flydslSetLLVMOptionStr.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
lib.flydslSetLLVMOptionStr.restype = ctypes.c_int
lib.flydslSetLLVMOptionBool.argtypes = [ctypes.c_char_p, ctypes.c_bool, ctypes.c_void_p]
lib.flydslSetLLVMOptionBool.restype = ctypes.c_int


def setopt(name, value):
    rc = lib.flydslSetLLVMOptionStr(name.encode(), str(value).encode(), None)
    if rc == 2:
        rc = lib.flydslSetLLVMOptionBool(name.encode(), str(value).lower() in ("1", "true", "on"), None)
    if rc:
        raise SystemExit(f"FAIL setting {name}={value}: rc={rc}")


def unescape(s):
    out = bytearray()
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            nx = s[i + 1]
            if nx in '\\"':
                out.append(ord(nx))
                i += 2
            elif nx == "n":
                out.append(10)
                i += 2
            elif nx == "t":
                out.append(9)
                i += 2
            else:
                out.append(int(s[i + 1 : i + 3], 16))
                i += 3
        else:
            out.append(ord(c))
            i += 1
    return bytes(out)


inp, outp = sys.argv[1], sys.argv[2]
for kv in sys.argv[3:]:
    k, _, v = kv.partition("=")
    setopt(k, v)
with ir.Context():
    m = ir.Module.parse(Path(inp).read_text())
    PassManager.parse(
        'builtin.module(gpu-module-to-binary{format=isa opts="--amdgpu-waves-per-eu=1"})'
    ).run(m.operation)
    txt = str(m.operation)
key = 'assembly = "'
i = txt.index(key) + len(key)
j = i
while True:
    j = txt.index('"', j)
    bs = 0
    k = j - 1
    while txt[k] == "\\":
        bs += 1
        k -= 1
    if bs % 2 == 0:
        break
    j += 1
Path(outp).write_bytes(unescape(txt[i:j]))
print("OK", outp, file=sys.stderr)
