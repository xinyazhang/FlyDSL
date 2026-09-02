#!/usr/bin/env python
"""Compile-only Triton driver for the gfx950 lost-SGPR-definition hunt.

No GPU is touched: the target is passed explicitly to triton.compile(), so the
HIP driver is never asked for a device.

usage:
    triton_repro.py <kernel-module.py> <kernel-name> <outdir> \
        --sig "*fp16:16, ... , 128, 512, False" [--warps 4] [--stages 1] [--wpeu 2]

Writes <outdir>/<name>.hsaco, .ll (llir), .amdgcn (asm) and .json.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-lostcopy-cache")


def constexpr(s):
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    if s == "True":
        return True
    if s == "False":
        return False
    return None


def compile_one(mod_path, kernel_name, sig_str, arch, warps, stages, wpeu, outdir, stem):
    import triton
    from triton.backends.compiler import GPUTarget

    mod_path = Path(mod_path)
    sys.path.insert(0, str(mod_path.parent))
    spec = importlib.util.spec_from_file_location(mod_path.stem, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_path.stem] = mod
    spec.loader.exec_module(mod)
    kernel = getattr(mod, kernel_name)

    signature = [s.strip() for s in sig_str.split(",")]
    hints = {(i,): constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    constants = {kernel.arg_names[i]: constexpr(s) for i, s in enumerate(signature)}
    constants = {k: v for k, v in constants.items() if v is not None}
    for key, value in hints.items():
        if value == 1:
            constants[kernel.arg_names[key[0]]] = value
    signature = {kernel.arg_names[i]: s.split(":")[0] for i, s in enumerate(signature)}
    for key in constants:
        signature[key] = "constexpr"
    attrs = {k: [("tt.divisibility", 16)] for k, v in hints.items() if v == 16}
    attrs.update({k: [("tt.divisibility", 8)] for k, v in hints.items() if v == 8})

    src = triton.compiler.ASTSource(fn=kernel, constexprs=constants, signature=signature, attrs=attrs)
    target = GPUTarget("hip", arch, 64 if arch.startswith("gfx9") else 32)
    backend = triton.compiler.make_backend(target)
    opts = backend.parse_options({"num_warps": warps, "num_stages": stages, "waves_per_eu": wpeu})
    cc = triton.compile(src, target=target, options=opts.__dict__)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{stem}.hsaco").write_bytes(cc.kernel)
    for key, ext in (("llir", "ll"), ("amdgcn", "amdgcn"), ("ttgir", "ttgir"), ("ttir", "ttir")):
        if key in cc.asm:
            (outdir / f"{stem}.{ext}").write_text(cc.asm[key])
    md = cc.metadata._asdict()
    md.pop("target", None)
    (outdir / f"{stem}.json").write_text(json.dumps(md, indent=2, default=str))
    return outdir / f"{stem}.hsaco"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("kernel")
    ap.add_argument("outdir")
    ap.add_argument("--sig", required=True)
    ap.add_argument("--arch", default="gfx950")
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--stages", type=int, default=1)
    ap.add_argument("--wpeu", type=int, default=2)
    ap.add_argument("--stem", default=None)
    a = ap.parse_args()
    stem = a.stem or a.kernel
    p = compile_one(a.module, a.kernel, a.sig, a.arch, a.warps, a.stages, a.wpeu, a.outdir, stem)
    print("wrote", p)


if __name__ == "__main__":
    main()
