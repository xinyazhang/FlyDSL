"""Compile the repro MLIR with arbitrary LLVM cl::opt settings.
usage: drive.py <in.mlir> <out> <fmt: isa|fatbin> [name=value ...]
"""
import ctypes, re, sys, os
LIB = "/home/xinyazha/.venvs/gfx950-7.14/lib/python3.13/site-packages/flydsl/_mlir/_mlir_libs/libFlyPythonCAPI.so.24.0git"
from flydsl._mlir import ir
from flydsl._mlir.passmanager import PassManager
lib = ctypes.CDLL(LIB, mode=ctypes.RTLD_GLOBAL)
lib.flydslSetLLVMOptionStr.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p]
lib.flydslSetLLVMOptionStr.restype = ctypes.c_int
lib.flydslSetLLVMOptionBool.argtypes = [ctypes.c_char_p, ctypes.c_bool, ctypes.c_void_p]
lib.flydslSetLLVMOptionBool.restype = ctypes.c_int

def setopt(name, value):
    rc = lib.flydslSetLLVMOptionStr(name.encode(), str(value).encode(), None)
    if rc == 2:
        rc = lib.flydslSetLLVMOptionBool(name.encode(), str(value).lower() in ("1","true","on"), None)
    if rc:
        raise SystemExit(f"FAIL setting {name}={value}: rc={rc}")

def unescape(s):
    out = bytearray(); i = 0
    while i < len(s):
        c = s[i]
        if c == '\\':
            nx = s[i+1]
            if nx in '\\"': out.append(ord(nx)); i += 2
            elif nx == 'n': out.append(10); i += 2
            elif nx == 't': out.append(9); i += 2
            else: out.append(int(s[i+1:i+3], 16)); i += 3
        else:
            out.append(ord(c)); i += 1
    return bytes(out)

inp, outp, fmt = sys.argv[1], sys.argv[2], sys.argv[3]
for kv in sys.argv[4:]:
    k, _, v = kv.partition("=")
    setopt(k, v)
with ir.Context():
    m = ir.Module.parse(open(inp).read())
    PassManager.parse(
        'builtin.module(gpu-module-to-binary{format=%s opts="--amdgpu-waves-per-eu=1"})' % fmt
    ).run(m.operation)
    txt = str(m.operation)
key = 'assembly = "' if fmt == "isa" else 'bin = "'
i = txt.index(key) + len(key)
j = i
while True:
    j = txt.index('"', j)
    bs = 0; k = j - 1
    while txt[k] == '\\': bs += 1; k -= 1
    if bs % 2 == 0: break
    j += 1
open(outp, "wb").write(unescape(txt[i:j]))
print("OK", outp)
