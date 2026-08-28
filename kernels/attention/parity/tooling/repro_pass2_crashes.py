# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""The twelve `flyc_pass2.out` shapes whose worker aborted, one process each.

Not tolerance failures: xdist reports the in-flight test as FAILED when the
node goes down, and every one of these sits immediately after a `node down:
Not properly terminated`. The abort itself lands in `_common_test.lmax`, whose
`.item()` is the first device sync after the launch -- which is where an
asynchronous HIP fault surfaces.

Run one shape per process so a fault names itself::

    export ROCM_PATH=$(rocm-sdk path --root)
    cd kernels/attention/parity
    python3 tooling/repro_pass2_crashes.py --list
    python3 tooling/repro_pass2_crashes.py 0

`--all` forks a child per shape and reports each one's exit status, which is
what distinguishes an abort from a wrong number.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (label, batch, heads, seqlen_q, seqlen_k, head_dim, causal, dropout_p, bias, dtype)
#
# Decoded from the pytest ids, which order parametrize marks bottom-decorator
# first: `...-CausalOff-64-8-hdim224-5-3` is seqlen_k=64, seqlen_q=8, D_HEAD
# 224, N_HEADS 5, BATCH 3.
CASES = [
    ("regular hd224 sk64 sq8", 3, 5, 8, 64, 224, False, 0.0, False, "f16"),
    ("regular hd224 sk128 sq8 causal", 3, 5, 8, 128, 224, True, 0.0, False, "bf16"),
    ("regular hd32 sk512 sq8 causal drop", 3, 5, 8, 512, 32, True, 0.5, False, "bf16"),
    ("irregular hd216 sk71 sq37 causal drop", 3, 5, 37, 71, 216, True, 0.5, False, "bf16"),
    ("irregular hd128 sk71 sq257 bias", 3, 5, 257, 71, 128, False, 0.0, True, "f16"),
    ("irregular hd248 sk13 sq11 bias drop", 3, 5, 11, 13, 248, False, 0.5, True, "f16"),
    ("irregular hd184 sk31 sq11 drop", 3, 5, 11, 31, 184, False, 0.5, False, "f16"),
]


def _run(case):
    import torch
    from fmha_bwd_dkdv_gfx1201_interface import flydsl_flash_attn_bwd_dkdv_gfx1201
    from fmha_bwd_dq_gfx1201_interface import flydsl_bwd_dq_gfx1201

    label, b, h, sq, sk, d, causal, p, bias, dt = case
    dtype = torch.bfloat16 if dt == "bf16" else torch.float16
    g = torch.Generator(device="cuda").manual_seed(0)

    def t(s):
        return torch.randn(b, h, s, d, dtype=dtype, device="cuda", generator=g)

    q, k, v, do = t(sq), t(sk), t(sk), t(sq)
    # `p` and `bias` are carried in the table but not driven here. Neither
    # reaches the JIT interfaces -- dropout needs the forward's Philox seed and
    # offset to line up, and bias is a builder-level argument the interfaces do
    # not forward -- and a fault does not need either to be the *right* value.
    # Five of the seven shapes are bias-free, so if nothing below aborts those
    # two are the next axes to add rather than a clean bill.

    print(f"[repro] {label}: B={b} H={h} sq={sq} sk={sk} d={d} causal={causal} p={p} bias={bias} {dt}")
    # `o`/`lse` from torch, not from our forward: its JIT interface refuses
    # `seq_q != seq_k`, which every shape here has. That restriction is the
    # interface's and not the kernel's -- AOTriton dispatches the hsaco
    # directly -- and these two are inputs to the backward either way.
    s = (q.float() @ k.float().transpose(-1, -2)) * (1.0 / d**0.5)
    if causal:
        i = torch.arange(sq, device=q.device)[:, None]
        j = torch.arange(sk, device=q.device)[None, :]
        s = s.masked_fill(j > i, float("-inf"))
    live = torch.isfinite(s).any(-1, keepdim=True)
    pmat = torch.where(live, torch.softmax(s, -1), torch.zeros_like(s))
    o = (pmat @ v.float()).to(dtype)
    lse = torch.where(live.squeeze(-1), torch.logsumexp(s, -1), torch.full(s.shape[:-1], float("inf"), device=q.device))
    # dropout only on this half: `ENABLE_DROPOUT` is a compiled axis, so it is
    # a different binary, and five of the seven shapes carry p=0.5. The dQ
    # interface has no dropout argument to pass.
    dk, dv = flydsl_flash_attn_bwd_dkdv_gfx1201(
        q, k, v, do, o, lse, causal=causal, dropout_p=p or None, philox_seed=0x1234, philox_offset2=0
    )
    torch.cuda.synchronize()
    print(f"[repro]   dkdv ok, finite={bool(torch.isfinite(dk).all() and torch.isfinite(dv).all())}")
    dq = flydsl_bwd_dq_gfx1201(q, k, v, o, do, lse, causal=causal)
    torch.cuda.synchronize()
    print(f"[repro]   dq ok, finite={bool(torch.isfinite(dq).all())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index", nargs="?", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.list:
        for i, c in enumerate(CASES):
            print(i, c[0])
        return 0
    if a.all:
        bad = []
        for i, c in enumerate(CASES):
            r = subprocess.run([sys.executable, os.path.abspath(__file__), str(i)])
            print(f"[repro] case {i} ({c[0]}) exit={r.returncode}")
            if r.returncode != 0:
                bad.append((i, c[0], r.returncode))
        print(f"[repro] {len(bad)} of {len(CASES)} did not exit cleanly: {bad}")
        return 1 if bad else 0
    _run(CASES[a.index if a.index is not None else 0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
