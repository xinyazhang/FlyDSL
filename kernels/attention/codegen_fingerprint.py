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
    q, k, v = (torch.randn(1, 8, 512, hd, dtype=torch.float16, device="cuda")
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


def main() -> int:
    """`--expect <baseline.json>` makes this a `git bisect run` predicate.

    Exit 0 when the emitted code matches the baseline, 1 when it differs. It is
    a sound predicate precisely because it is deterministic -- unlike a
    benchmark, which at this kernel's noise floor flips good/bad near the
    threshold and converges on the wrong commit (plan section 0.5).
    """
    out = {f"hd{hd}_{'c' if c else 'f'}": fingerprint(hd, c) for hd, c in CONFIGS}

    if "--expect" in sys.argv:
        base = json.load(open(sys.argv[sys.argv.index("--expect") + 1]))
        diffs = [k for k in base if base[k] != out.get(k)]
        for k in diffs:
            b, a = base[k], out.get(k, {})
            print(f"{k}: vgpr {b.get('vgpr')}->{a.get('vgpr')} "
                  f"scratch {b.get('scratch')}->{a.get('scratch')} "
                  f"insts {b.get('insts')}->{a.get('insts')} "
                  f"sha {b.get('isa_sha')}->{a.get('isa_sha')}")
        print("UNCHANGED" if not diffs else f"CHANGED ({len(diffs)}/{len(base)})")
        return 1 if diffs else 0

    print(json.dumps(out, indent=1))
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        json.dump(out, open(args[0], "w"), indent=1)
    return 0


if __name__ == "__main__":
    if "--child" in sys.argv:
        sys.exit(child(int(sys.argv[-2]), bool(int(sys.argv[-1]))))
    sys.exit(main())
