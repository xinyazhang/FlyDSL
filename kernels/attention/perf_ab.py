# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Interleaved A/B of two git revisions, sharded over the available GPUs.

Two problems this exists to solve, both learned the hard way on this kernel
(`sdpa_lore_gfx1201.md`):

- **Sequential before/after runs cannot resolve a few percent.** Running the
  whole sweep on revision A and then the whole sweep on revision B puts minutes
  between the two measurements of any given point, and the board drifts over
  minutes. Measured noise floor of that method: ~5%, with outliers to +19%, on
  points whose kernel selection was *provably identical*.
- **You cannot interleave two git revisions inside one process.** The kernel
  module and its siblings are imported by name, so two checkouts cannot coexist
  in one interpreter without cross-contaminating each other's imports.

So: one long-lived worker process per (revision, GPU), each holding its own
checkout on `sys.path`, and the parent alternates requests between them. The
gap between the two measurements of a point is then ~1 second rather than
~4 minutes, and no process is reloaded between alternations.

Sharding is by head_dim across GPUs, which is embarrassingly parallel -- the
configs share nothing.

    # tier 2.9: fast screen, N=4096 non-causal, major head_dims
    python3 perf_ab.py --base <rev> [--head <rev>]

    # tier 3: the full ladder, both causal modes
    python3 perf_ab.py --base <rev> --full

`--base <rev>` with no `--head` compares the working tree against `<rev>`.
Passing the *same* revision for both is the harness self-test: the ratio it
reports is then its own resolution, and anything it cannot separate from 1.0 is
noise it cannot see.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

# Tier 2.9: the fast screen. One sequence length, one masking mode, the head
# dims that production actually uses plus the two that are structurally
# interesting -- 80 is the only width taking row_subtiles=2, 192 is the one that
# spills. Causal coverage is deliberately absent; that is what tier 3 is for.
FAST_HEAD_DIMS = (64, 80, 128, 192, 256)
FAST_N = 4096
FAST_CAUSAL = (False,)

LADDER = (16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 384, 512)
FULL_N = (1024, 4096)
FULL_CAUSAL = (False, True)

B, H, REPS = 1, 8, 5


# --------------------------------------------------------------------------
# worker: runs inside a checkout, one config at a time, on one GPU
# --------------------------------------------------------------------------


def _worker() -> int:
    """Read `hd causal N` lines on stdin, print `tflops` per line."""
    sys.path.insert(0, os.path.join(os.environ["FLYDSL_AB_ROOT"], "kernels", "attention"))
    import torch
    from bench_shim import do_bench
    from flash_attn_func_gfx1201_interface import flydsl_flash_attn_func_gfx1201 as F

    cache: dict = {}
    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line or line == "QUIT":
            break
        hd, causal_s, n = line.split()
        hd, n, causal = int(hd), int(n), causal_s == "1"
        key = (hd, causal, n)
        if key not in cache:
            torch.manual_seed(0)
            cache[key] = tuple(torch.randn(B, H, n, hd, dtype=torch.float16, device="cuda") for _ in range(3))
            q, k, v = cache[key]
            F(q, k, v, causal=causal)  # warm the JIT before timing
            torch.cuda.synchronize()
        q, k, v = cache[key]
        ms = do_bench(lambda: F(q, k, v, causal=causal), warmup=25, rep=100, return_mode="median")
        fl = 2.0 * B * H * n * n * hd * 2 * (0.5 if causal else 1.0)
        print(f"{fl / ms * 1e-9:.4f}", flush=True)
    return 0


class Worker:
    """A checkout pinned to a GPU, kept alive across configs."""

    def __init__(self, root: str, gpu: int):
        env = dict(os.environ, FLYDSL_AB_ROOT=root, HIP_VISIBLE_DEVICES=str(gpu), CUDA_VISIBLE_DEVICES=str(gpu))
        self.p = subprocess.Popen(
            [sys.executable, os.path.join(_HERE, "perf_ab.py"), "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        line = self.p.stdout.readline().strip()
        if line != "READY":
            raise RuntimeError(f"worker in {root} on gpu {gpu} failed: {line!r}")

    def measure(self, hd: int, causal: bool, n: int) -> float:
        self.p.stdin.write(f"{hd} {int(causal)} {n}\n")
        self.p.stdin.flush()
        return float(self.p.stdout.readline().strip())

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


# --------------------------------------------------------------------------
# parent
# --------------------------------------------------------------------------


def _checkout(rev: str, into: str) -> str:
    """A detached worktree at `rev`, or the live tree when `rev` is None."""
    if rev is None:
        return _REPO
    subprocess.run(["git", "-C", _REPO, "worktree", "add", "--detach", into, rev], check=True, capture_output=True)
    return into


def _shard(configs, gpu, base_root, head_root, results):
    a = Worker(base_root, gpu)
    b = Worker(head_root, gpu)
    try:
        for hd, causal, n in configs:
            got_a, got_b = [], []
            for _ in range(REPS):
                # Alternate, so the two sides see the same board state.
                got_a.append(a.measure(hd, causal, n))
                got_b.append(b.measure(hd, causal, n))
            results[(hd, causal, n)] = (max(got_a), max(got_b))
    finally:
        a.close()
        b.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--base", help="baseline revision")
    ap.add_argument("--head", default=None, help="revision to compare (default: working tree)")
    ap.add_argument("--full", action="store_true", help="tier 3: the whole ladder")
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.worker:
        return _worker()
    if not args.base:
        ap.error("--base is required")

    import torch

    ngpu = args.gpus or torch.cuda.device_count()

    if args.full:
        configs = [(hd, c, n) for hd in LADDER for c in FULL_CAUSAL for n in FULL_N]
        tier = "3 (full ladder)"
    else:
        configs = [(hd, c, FAST_N) for hd in FAST_HEAD_DIMS for c in FAST_CAUSAL]
        tier = "2.9 (fast screen)"

    with tempfile.TemporaryDirectory() as tmp:
        base_root = _checkout(args.base, os.path.join(tmp, "base"))
        head_root = _checkout(args.head, os.path.join(tmp, "head"))
        try:
            # Round-robin so each GPU gets a similar mix of cheap and dear.
            shards = [configs[i::ngpu] for i in range(ngpu)]
            results: dict = {}
            with ThreadPoolExecutor(max_workers=ngpu) as ex:
                list(
                    ex.map(
                        lambda i: _shard(shards[i], i, base_root, head_root, results),
                        [i for i in range(ngpu) if shards[i]],
                    )
                )
        finally:
            for root in (base_root, head_root):
                if root != _REPO:
                    subprocess.run(["git", "-C", _REPO, "worktree", "remove", "--force", root], capture_output=True)

    print(
        f"\ntier {tier}   base={args.base}   head={args.head or 'working tree'}   "
        f"{ngpu} GPUs   {len(configs)} configs\n"
    )
    print(f"{'hd':>5} {'causal':>7} {'N':>6} {'base':>9} {'head':>9} {'ratio':>7}")
    worst = (None, 1e9)
    for key in sorted(results):
        hd, causal, n = key
        a, b = results[key]
        r = b / a
        flag = "  <--" if r < 0.97 else ""
        print(f"{hd:>5} {str(causal):>7} {n:>6} {a:>9.2f} {b:>9.2f} {r:>7.3f}{flag}")
        if r < worst[1]:
            worst = (key, r)
    print(f"\nworst: {worst[0]} at {worst[1]:.3f}")
    if args.json:
        json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in results.items()}, open(args.json, "w"), indent=1)
    return 0 if worst[1] >= 0.97 else 1


if __name__ == "__main__":
    sys.exit(main())
