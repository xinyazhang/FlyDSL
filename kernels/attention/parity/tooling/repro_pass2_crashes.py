# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Poisoned-margin OOB probe for the `flyc_pass2.out` shapes whose worker aborted.

Those failures are not tolerance failures: xdist reports the in-flight test as
FAILED when the node goes down, and each sits immediately after a `node down:
Not properly terminated`. Seven of them reproduce byte-identically across two
runs, so they are deterministic and config-specific rather than contention.

**Every tensor gets a margin, and every margin is poisoned**, which turns both
directions of an out-of-bounds access into a positive signal rather than into
plausible-looking numbers:

    inputs   allocated `[..., 2*MARGIN + D]`, filled with NaN, real data
             written through the view `[..., MARGIN : MARGIN + D]`. The margin
             is on *both* sides, so a read that walks backwards off a row --
             a negative column, or a row-index underflow -- is caught as well
             as one that runs past the end. Any OOB read reaches the output as
             a NaN.

    outputs  allocated the same way and filled with -inf *everywhere*,
             including the live view. Three distinct verdicts fall out:
               - margin no longer -inf   -> OOB WRITE
               - live view holds NaN     -> OOB READ (poison propagated)
               - live view still -inf    -> position never written

    row f32  `lse` and `delta` must be contiguous rank-2, so they cannot take
             a per-row margin. They get whole guard *planes* above and below
             instead, which catches a row index that overruns the tensor.

**The build is forced to `fp_mode="safe"`.** The shipped default is "noninf",
whose flag set is `reassoc|nnan|nsz|arcp|contract|afn` -- `nnan` licenses the
compiler to assume no NaN exists, which is exactly the assumption this probe
needs removed. `unsafe_fp_math` and `fast_fp_math` are cleared for the same
reason. Addresses do not depend on float flags, so an OOB access found here is
an OOB access in the shipped build; only its *visibility* changes.

`--aot` builds the way AOTriton does -- one compiled tile for both head-dim
axes, `padded_head` set explicitly -- rather than the way `plan` does. For a
head_dim already on the ladder the two agree.

**What this cannot see.** Column overruns are caught completely: every offset
outside `[MARGIN, MARGIN + D)` of a row is poison, in both directions. A *row*
overrun is only caught when it clears the tensor's last row, because a row
index one too large lands on the next row's real data -- the same
"plausible data inside the right allocation" hole the suite's varlen tests
exist for. A clean run therefore rules out an OOB column, not an OOB row.

Run one shape per process so a fault names itself::

    export ROCM_PATH=$(rocm-sdk path --root)
    cd kernels/attention/parity
    python3 tooling/repro_pass2_crashes.py --list
    python3 tooling/repro_pass2_crashes.py 0
    python3 tooling/repro_pass2_crashes.py --all

Pair it with `PYTORCH_NO_HIP_MEMORY_CACHING=1` to remove the caching
allocator's slack, so an OOB access that currently lands in a neighbouring
block faults at a printed address instead.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Elements of poison on each side of every row. A multiple of 8 keeps each row
# start 16-byte aligned, which is the pitch guarantee the copy atoms rely on --
# 8 f16 elements is exactly one cooperative-load vector.
MARGIN = 8

# (label, batch, heads, seqlen_q, seqlen_k, head_dim, causal, dropout_p, bias, dtype)
#
# Decoded from the pytest ids, which order parametrize marks bottom-decorator
# first: `...-CausalOff-64-8-hdim224-5-3` is seqlen_k=64, seqlen_q=8, D_HEAD
# 224, N_HEADS 5, BATCH 3. The first five crashed in *both* runs.
CASES = [
    ("both-runs  regular hd224 sk128 sq8 causal", 3, 5, 8, 128, 224, True, 0.0, False, "bf16"),
    ("both-runs  regular hd224 sk128 sq8", 3, 5, 8, 128, 224, False, 0.0, False, "f16"),
    ("both-runs  regular hd224 sk64 sq8", 3, 5, 8, 64, 224, False, 0.0, False, "f16"),
    ("both-runs  regular hd32 sk512 sq8 causal drop", 3, 5, 8, 512, 32, True, 0.5, False, "bf16"),
    ("both-runs  irregular hd216 sk71 sq37 causal drop", 3, 5, 37, 71, 216, True, 0.5, False, "bf16"),
    ("pass1-only regular hd96 sk128 sq8", 3, 5, 8, 128, 96, False, 0.0, False, "f16"),
    ("pass1-only irregular hd72 sk13 sq11", 3, 5, 11, 13, 72, False, 0.0, False, "f16"),
    ("pass1-only irregular hd152 sk13 sq11", 3, 5, 11, 13, 152, False, 0.0, False, "f16"),
    ("pass1-only irregular hd216 sk32 sq16", 3, 5, 16, 32, 216, False, 0.0, False, "f16"),
    ("control    regular hd224 sk128 sq128 causal", 3, 5, 128, 128, 224, True, 0.0, False, "f16"),
    ("control    regular hd216 sk128 sq8 causal", 3, 5, 8, 128, 216, True, 0.0, False, "bf16"),
]


def _poisoned_in(torch, b, h, s, d, dtype, gen):
    """Random `(b, h, s, d)` inside a NaN margin on both sides of every row."""
    buf = torch.full((b, h, s, d + 2 * MARGIN), float("nan"), dtype=dtype, device="cuda")
    view = buf[..., MARGIN : MARGIN + d]
    view.copy_(torch.rand(b, h, s, d, dtype=dtype, device="cuda", generator=gen))
    return buf, view


def _poisoned_out(torch, b, h, s, d, dtype):
    """`(b, h, s, d)` of -inf inside a -inf margin. Everything starts poisoned."""
    buf = torch.full((b, h, s, d + 2 * MARGIN), float("-inf"), dtype=dtype, device="cuda")
    return buf, buf[..., MARGIN : MARGIN + d]


def _verdict(torch, name, buf, view, report):
    """Three checks on one output: OOB write, OOB read, and never-written."""
    intact = torch.isneginf(buf).clone()
    intact[..., MARGIN : MARGIN + view.shape[-1]] = True  # only the margin is under test here
    n_write = int((~intact).sum())
    n_nan = int(torch.isnan(view).sum())
    n_unwritten = int(torch.isneginf(view).sum())
    if n_write:
        report.append(f"OOB WRITE   {name}: {n_write} margin element(s) overwritten")
    if n_nan:
        report.append(f"OOB READ    {name}: {n_nan}/{view.numel()} output element(s) are NaN")
    if n_unwritten:
        report.append(f"UNWRITTEN   {name}: {n_unwritten}/{view.numel()} output element(s) still -inf")
    if not (n_write or n_nan or n_unwritten):
        report.append(f"clean       {name}")


def _run(case, aot):
    import torch
    from fmha_bwd_dkdv_gfx1201_interface import build_bwd_dkdv_module
    from fmha_bwd_dq_gfx1201_kernel import build_bwd_dq_module
    from fmha_tuning_bwd_dkdv_gfx1201 import BwdDkDvKnobs, BwdDkDvMetadata
    from fmha_tuning_bwd_dkdv_gfx1201 import _round_to_ladder as dkdv_tile
    from fmha_tuning_bwd_dkdv_gfx1201 import resolve_knobs as dkdv_knobs
    from fmha_tuning_bwd_dq_gfx1201 import _round_to_ladder as dq_tile

    label, b, h, sq, sk, d, causal, p, bias, dt = case
    assert d % 8 == 0, "the margin scheme assumes an 8-element-aligned head_dim"
    dtype = torch.bfloat16 if dt == "bf16" else torch.float16
    g = torch.Generator(device="cuda").manual_seed(0)
    ctype = 1 if causal else 0

    qb, q = _poisoned_in(torch, b, h, sq, d, dtype, g)
    kb, k = _poisoned_in(torch, b, h, sk, d, dtype, g)
    vb, v = _poisoned_in(torch, b, h, sk, d, dtype, g)
    dob, do = _poisoned_in(torch, b, h, sq, d, dtype, g)

    print(f"[repro] {label}")
    print(f"[repro]   B={b} H={h} sq={sq} sk={sk} d={d} causal={causal} p={p} bias={bias} {dt} margin={MARGIN}")

    # o/lse/delta from torch: the forward's JIT interface refuses seq_q !=
    # seq_k, which every shape here has, and these are inputs to the backward
    # either way. Computed on the *clean* views, so the reference is NaN-free.
    s = (q.float() @ k.float().transpose(-1, -2)) * (1.0 / d**0.5)
    if causal:
        i = torch.arange(sq, device=q.device)[:, None]
        j = torch.arange(sk, device=q.device)[None, :]
        s = s.masked_fill(j > i, float("-inf"))
    alive = torch.isfinite(s).any(-1, keepdim=True)
    pmat = torch.where(alive, torch.softmax(s, -1), torch.zeros_like(s))
    o = (pmat @ v.float()).to(dtype)
    lse2 = torch.where(
        alive.squeeze(-1), torch.logsumexp(s, -1), torch.full(s.shape[:-1], float("inf"), device=q.device)
    )
    delta2 = (do.float() * o.float()).sum(-1)

    # `lse`/`delta` must be contiguous rank-2, so they take guard *planes*
    # rather than a per-row margin: a row index that overruns the tensor lands
    # in one, and row slicing keeps the slice contiguous.
    def _row_tensor(src):
        pad = torch.full((b * h + 2, sq), float("nan"), dtype=torch.float32, device="cuda")
        pad[1 : 1 + b * h].copy_(src.reshape(b * h, sq))
        return pad, pad[1 : 1 + b * h]

    lse_buf, lse = _row_tensor(lse2)
    delta_buf, delta = _row_tensor(delta2)

    report = []

    # ---- dK / dV -----------------------------------------------------------
    dkb, dk = _poisoned_out(torch, b, h, sk, d, dtype)
    dvb, dv = _poisoned_out(torch, b, h, sk, d, dtype)
    tile = dkdv_tile(d) if aot else d
    meta = BwdDkDvMetadata(
        num_heads=h,
        head_dim=tile if aot else d,
        head_dim_v=tile if aot else d,
        causal=causal,
        causal_type=ctype or None,
        dtype_str=dt,
        dropout=p > 0.0,
    )
    over = BwdDkDvKnobs(fp_mode="safe", unsafe_fp_math=False, fast_fp_math=False)
    if aot:
        over = over.merge(BwdDkDvKnobs(block_dmodel=tile, block_dmodel_v=tile, padded_head=tile != d))
    exe = build_bwd_dkdv_module(meta, dkdv_knobs(meta, over))
    exe(
        q, k, v, do, dk, dv, lse, delta, b, sq,
        seqlen_k=sk,
        window=None,
        dropout_p=p or None,
        philox_seed=0x1234,
        philox_offset2=0,
    )  # fmt: skip
    torch.cuda.synchronize()
    _verdict(torch, "dK", dkb, dk, report)
    _verdict(torch, "dV", dvb, dv, report)

    # ---- dQ ----------------------------------------------------------------
    dqb, dq = _poisoned_out(torch, b, h, sq, d, dtype)
    qtile = dq_tile(d)
    exe_dq = build_bwd_dq_module(
        num_heads=h,
        head_dim=qtile if aot else d,
        head_dim_v=qtile if aot else d,
        causal=causal,
        causal_type=ctype or None,
        dtype_str=dt,
        fp_mode="safe",
        unsafe_fp_math=False,
        fast_fp_math=False,
        **({"block_dmodel": qtile, "padded_head": qtile != d} if aot else {}),
    )
    exe_dq(q, k, v, do, dq, lse, delta, b, sq, sk)
    torch.cuda.synchronize()
    _verdict(torch, "dQ", dqb, dq, report)

    # ---- inputs and row tensors must come back untouched --------------------
    for nm, bufr, viewr in (("Q", qb, q), ("K", kb, k), ("V", vb, v), ("DO", dob, do)):
        n = int((~torch.isnan(bufr[..., :MARGIN])).sum()) + int((~torch.isnan(bufr[..., MARGIN + d :])).sum())
        if n:
            report.append(f"OOB WRITE   {nm} (input!): {n} margin element(s) overwritten")
    for nm, bufr in (("lse", lse_buf), ("delta", delta_buf)):
        n = int((~torch.isnan(bufr[0])).sum()) + int((~torch.isnan(bufr[-1])).sum())
        if n:
            report.append(f"OOB WRITE   {nm} guard plane: {n} element(s) overwritten")

    for line in report:
        print(f"[repro]   {line}")
    return any(line.startswith(("OOB", "UNWRITTEN")) for line in report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index", nargs="?", type=int)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--aot", action="store_true", help="build AOTriton's way: one tile, explicit padded_head")
    a = ap.parse_args()
    if a.list:
        for i, c in enumerate(CASES):
            print(i, c[0])
        return 0
    if a.all:
        bad, dirty = [], []
        for i, c in enumerate(CASES):
            argv = [sys.executable, os.path.abspath(__file__), str(i)] + (["--aot"] if a.aot else [])
            r = subprocess.run(argv)
            print(f"[repro] case {i} ({c[0]}) exit={r.returncode}")
            if r.returncode == 2:
                dirty.append((i, c[0]))
            elif r.returncode != 0:
                bad.append((i, c[0], r.returncode))
        print(f"[repro] {len(dirty)} case(s) reported an OOB/unwritten finding: {dirty}")
        print(f"[repro] {len(bad)} case(s) crashed or errored: {bad}")
        return 1 if (bad or dirty) else 0
    return 2 if _run(CASES[a.index if a.index is not None else 0], a.aot) else 0


if __name__ == "__main__":
    sys.exit(main())
