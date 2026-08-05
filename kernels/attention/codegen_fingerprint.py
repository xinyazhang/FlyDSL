"""Codegen fingerprint: VGPRs, scratch, instruction count -- no benchmarking.

Cheap enough to run on every commit. It does not measure performance; it
decides *whether performance can have changed*. An unchanged fingerprint means
the emitted code is identical, so no A/B is needed. A changed one means run it.
"""
import glob, hashlib, json, os, re, subprocess, sys, tempfile
HERE = "/home/xinyazha/dockerhome/meff/FlyDSL/kernels/attention"
sys.path.insert(0, HERE)
CONFIGS = [(64, False), (128, True), (256, True)]


def child(hd, causal):
    import torch
    from flash_attn_func_gfx1201_interface import flydsl_flash_attn_func_gfx1201 as F
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 512, 8, hd, dtype=torch.float16, device="cuda")
               for _ in range(3))
    F(q, k, v, causal=causal)
    torch.cuda.synchronize()
    return 0


def fingerprint(hd, causal):
    with tempfile.TemporaryDirectory() as d:
        env = dict(os.environ, FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=d,
                   FLYDSL_RUNTIME_ENABLE_CACHE="0")
        r = subprocess.run([sys.executable, __file__, "--child", str(hd), str(int(causal))],
                           env=env, capture_output=True, timeout=1800)
        if r.returncode != 0:
            return {"error": r.stderr.decode()[-200:]}
        for f in glob.glob(os.path.join(d, "*", "21_final_isa.s")):
            t = open(f).read()
            body = "\n".join(l for l in t.split("\n")
                             if l.strip() and not l.strip().startswith((".", ";", "//")))
            v = re.search(r"\.vgpr_count:\s*(\d+)", t)
            sc = re.search(r"\.private_segment_fixed_size:\s*(\d+)", t)
            return {"vgpr": int(v.group(1)) if v else None,
                    "scratch": int(sc.group(1)) if sc else 0,
                    "insts": len(body.split("\n")),
                    "isa_sha": hashlib.sha256(body.encode()).hexdigest()[:12]}
    return {"error": "no isa"}


if __name__ == "__main__":
    if "--child" in sys.argv:
        sys.exit(child(int(sys.argv[-2]), bool(int(sys.argv[-1]))))
    out = {f"hd{hd}_{'c' if c else 'f'}": fingerprint(hd, c) for hd, c in CONFIGS}
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        json.dump(out, open(sys.argv[1], "w"), indent=1)
