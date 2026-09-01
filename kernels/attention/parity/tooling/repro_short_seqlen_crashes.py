# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Short-sequence sweep for the bias-free gfx1201 aborts.

**Resolved, and not in this kernel.** The bias-free aborts this file was built
to chase turned out to be the *reference* computation, not the kernels: rerun
with `AOTRITON_TORCH_ONLY_USE_CPU=1`, which moves torch's fp32 reference off
the GPU, all twelve pass (flyc_pass6, 29 targeted tests, 100%, zero
node-downs). Every clean result this probe reported was therefore correct.

That is worth keeping rather than deleting the file over, for two reasons. The
poison found real bugs when there were real bugs to find -- the forward's
unclamped bias row and column, both confirmed by HIP faults naming our kernels
-- so a clean run here is now calibrated evidence rather than an absence of
evidence. And the shapes below are the ones a future regression would land on
first.

Note what a clean probe cannot tell you on its own: "the symptom moved when the
reference moved" is consistent with a masked kernel bug as well as with a
reference bug. What rules the first out is that the two aborts we *did* get
fault logs for both named our kernels by name, and the twelve never did.

`repro_pass2_crashes.py` probes a fixed list of shapes copied out of a log.
This one sweeps, because the bias-free crashes are *hard to hit*: the same id
aborts in one run and passes in the next, and the surviving three all sit at
`seqlen_q = 8`::

    test_regular_bwd[...-CausalOn-128-8-hdim224]
    test_regular_bwd[...-CausalOff-64-8-hdim224]
    test_regular_bwd[...-CausalOn-512-8-hdim32]

A workgroup covers BLOCK_M query rows whatever `seqlen_q` is -- 256 of them at
head_dim 224 against a `seqlen_q` of 8, so 248 lanes out of 256 are dead and
every one of them still executes each per-row access. That is exactly the
precondition of the bias-row bug this suite already fixed in the forward, and
the reason to sweep short extents on *both* axes rather than probe one shape.

**Margins are large on purpose.** The earlier probe used 8 elements on the
column axis, which an overrun jumps clean over: a dead group can start a whole
BLOCK_N past the extent, land beyond the poison, and read real data with no
signal -- a false negative dressed as a result. Here the column margin is
`COL_MARGIN` elements (4096, i.e. 8 KiB at 16-bit) and the row guard is
`ROW_GUARD` rows (512, twice the largest BLOCK_M), so any overrun this kernel
family can express lands *inside* the poison rather than past it.

Poison, and what each verdict means:

    inputs   NaN everywhere outside the live view. An OOB read reaches the
             output as a NaN.
    outputs  -inf everywhere including the live view:
               margin no longer -inf -> OOB WRITE
               live view holds NaN   -> OOB READ
               live view still -inf  -> never written
    row f32  `lse`/`delta` must stay contiguous rank-2, so they get whole guard
             planes above and below instead of per-row margins.

`storage_flip` is an axis, matching AOTriton's harness: it allocates
`(B, S, H, D)` and transposes to `(B, H, S, D)`, so the head stride is smaller
than the sequence stride. Two of the three failing ids differ only in that bit.

Builds force `fp_mode="safe"`. The shipped default is "noninf", whose flag set
includes `nnan` -- the compiler is told no NaN exists, which is the assumption
the poison needs removed. Addresses do not depend on float flags, so a finding
here is a finding in the shipped build.

One process per *build* (head_dim, causal, dtype), sweeping every seqlen pair
inside it, so the module is compiled once and a crash names the pair it died
on::

    export ROCM_PATH=$(rocm-sdk path --root)
    cd kernels/attention/parity
    python3 tooling/repro_short_seqlen_crashes.py --list
    python3 tooling/repro_short_seqlen_crashes.py 0
    PYTORCH_NO_HIP_MEMORY_CACHING=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
        python3 tooling/repro_short_seqlen_crashes.py --all

`PYTORCH_NO_HIP_MEMORY_CACHING=1` matters more here than anywhere: it removes
the caching allocator's slack, which is the only reason these aborts are
intermittent in the first place.
"""

import argparse
import itertools
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Elements of poison each side of every row, and rows of poison past the
# sequence extent. Both deliberately larger than any overrun the kernels can
# express: BLOCK_DMODEL tops out at 512 and BLOCK_M at 256.
COL_MARGIN = 4096
ROW_GUARD = 512

# Head dims to build. The four that crash bias-free (224, 216, 96, 32) plus 88
# and 64 as neighbours -- 224/216/96/64 carry BLOCK_M 256, the largest, and so
# the most dead lanes at a short seqlen_q.
HEAD_DIMS = (408, 224, 216, 96, 88, 64, 48, 32)
DTYPES = ("bf16", "f16")

# Short on both axes. seqlen_q dominates the suspicion, but seqlen_k is swept
# too: it decides the KV tile count and the ragged tail, and 1/2/4 are below a
# single WMMA tile in a way no AOTriton shape reaches.
SEQ_Q = (1, 2, 4, 7, 8, 15, 16, 37)
SEQ_K = (1, 4, 8, 15, 16, 64, 128, 512)

# The pairs AOTriton actually aborts on, verbatim, because the cross product
# above does not contain them and a sweep that misses the known failures is not
# evidence of anything. Two lessons are encoded here:
#
#   * a *short seqlen_k against a long seqlen_q* is its own case. The sweep was
#     built on "short seqlen_q with a big BLOCK_M", and `4-2048` / `16-2048`
#     are the mirror image -- one KV tile against 16 Q tiles.
#   * every one of these carries dropout 0.5 except the hdim224 family, which
#     carries 0.0. The split is exact across the twelve, so dropout is an axis
#     and not a detail.
#
# `(seqlen_q, seqlen_k, head_dim, causal, dropout)`.
OBSERVED = (
    (2048, 4, 32, True, 0.5),
    (4, 579, 32, True, 0.5),
    (2048, 16, 48, True, 0.5),
    (17, 13, 408, False, 0.5),
    (37, 71, 216, False, 0.5),
    (8, 64, 224, False, 0.0),
    (8, 128, 224, False, 0.0),
    (8, 128, 224, True, 0.0),
)


def _configs():
    return [(hd, causal, dt) for hd in HEAD_DIMS for causal in (False, True) for dt in DTYPES]


def _alloc(torch, b, h, s, d, dtype, flip, margin, fill):
    """`(buf, view, live)` -- a poisoned buffer, its clean view, and the index
    tuple naming the live region inside `buf`.

    `flip` allocates `(B, S, H, D)` and transposes, which is what AOTriton's
    harness does: the head stride ends up smaller than the sequence stride, and
    a kernel that has the two confused only fails in that layout.
    """
    if flip:
        shape = (b, s + ROW_GUARD, h, d + 2 * margin)
        live = (slice(None), slice(0, s), slice(None), slice(margin, margin + d))
    else:
        shape = (b, h, s + ROW_GUARD, d + 2 * margin)
        live = (slice(None), slice(None), slice(0, s), slice(margin, margin + d))
    buf = torch.full(shape, fill, dtype=dtype, device="cuda")
    view = buf[live]
    if flip:
        view = view.transpose(1, 2)
    return buf, view, live


def _poisoned_in(torch, b, h, s, d, dtype, gen, flip):
    buf, view, live = _alloc(torch, b, h, s, d, dtype, flip, COL_MARGIN, float("nan"))
    view.copy_(torch.rand(b, h, s, d, dtype=dtype, device="cuda", generator=gen))
    return buf, view, live


def _poisoned_out(torch, b, h, s, d, dtype, flip):
    return _alloc(torch, b, h, s, d, dtype, flip, COL_MARGIN, float("-inf"))


def _check_out(torch, name, buf, view, live, report):
    """OOB write, OOB read, and never-written -- three verdicts on one output."""
    intact = torch.isneginf(buf).clone()
    intact[live] = True
    n_write = int((~intact).sum())
    n_nan = int(torch.isnan(view).sum())
    n_unwritten = int(torch.isneginf(view).sum())
    if n_write:
        report.append(f"OOB WRITE {name}: {n_write} poison element(s) overwritten")
    if n_nan:
        report.append(f"OOB READ  {name}: {n_nan}/{view.numel()} output element(s) NaN")
    if n_unwritten:
        report.append(f"UNWRITTEN {name}: {n_unwritten}/{view.numel()} still -inf")


def _check_in(torch, name, buf, live, report):
    """An input's poison must come back untouched."""
    poisoned = torch.isnan(buf).clone()
    poisoned[live] = True
    n = int((~poisoned).sum())
    if n:
        report.append(f"OOB WRITE {name} (input!): {n} poison element(s) overwritten")


def _run_config(cfg, batch, heads, only_pair, dropout):
    import torch
    from flash_attn_func_gfx1201_aiw import build_flash_attn_func_aiw_module
    from fmha_bwd_dkdv_gfx1201_interface import build_bwd_dkdv_module
    from fmha_bwd_dq_gfx1201_kernel import build_bwd_dq_module
    from fmha_tuning_bwd_dkdv_gfx1201 import BwdDkDvKnobs, BwdDkDvMetadata
    from fmha_tuning_bwd_dkdv_gfx1201 import _round_to_ladder as dkdv_tile
    from fmha_tuning_bwd_dkdv_gfx1201 import resolve_knobs as dkdv_knobs
    from fmha_tuning_bwd_dq_gfx1201 import _round_to_ladder as dq_tile
    from fmha_tuning_gfx1201 import _round_to_ladder as fwd_tile

    d, causal, dt = cfg
    dtype = torch.bfloat16 if dt == "bf16" else torch.float16
    ctype = 1 if causal else 0
    b, h = batch, heads

    # Every builder wants the compiled tile, not the real extent: BLOCK_DMODEL
    # must be a multiple of 16 and 216 is not. The real head_dim rides as a
    # runtime argument, which `padded_head` is what records.
    ft, qt, kt = fwd_tile(d), dq_tile(d), dkdv_tile(d)
    # Poison is not free: at the widest pair here a single buffer is
    # `(B, H, sk+ROW_GUARD, d+2*COL_MARGIN)`, and eight of them are live at
    # once. Printed so an OOM is a number rather than a surprise; dial
    # --col-margin / --row-guard down if it does not fit.
    _big = max(SEQ_K) + ROW_GUARD
    _bytes = b * h * _big * (d + 2 * COL_MARGIN) * 2
    print(
        f"[cfg] head_dim={d} causal={causal} {dt}  tiles fwd/dq/dkdv={ft}/{qt}/{kt}  "
        f"col_margin={COL_MARGIN} row_guard={ROW_GUARD}  peak ~{8 * _bytes / 2**30:.1f} GiB",
        flush=True,
    )

    fwd = build_flash_attn_func_aiw_module(
        num_heads=h, head_dim=ft, block_dmodel=ft, padded_head=ft != d,
        causal=causal, causal_type=ctype or None, dtype_str=dt,
        fp_mode="safe", unsafe_fp_math=False, fast_fp_math=False,
    )  # fmt: skip
    kmeta = BwdDkDvMetadata(
        num_heads=h, head_dim=kt, head_dim_v=kt, causal=causal,
        causal_type=ctype or None, dtype_str=dt, dropout=dropout > 0.0,
    )  # fmt: skip
    dkdv = build_bwd_dkdv_module(
        kmeta,
        dkdv_knobs(
            kmeta,
            BwdDkDvKnobs(
                fp_mode="safe", unsafe_fp_math=False, fast_fp_math=False,
                block_dmodel=kt, block_dmodel_v=kt, padded_head=kt != d,
            ),  # fmt: skip
        ),
    )
    dqk = build_bwd_dq_module(
        num_heads=h, head_dim=qt, head_dim_v=qt, causal=causal,
        causal_type=ctype or None, dtype_str=dt, block_dmodel=qt, padded_head=qt != d,
        fp_mode="safe", unsafe_fp_math=False, fast_fp_math=False,
    )  # fmt: skip

    pairs = [only_pair] if only_pair else list(itertools.product(SEQ_Q, SEQ_K))
    findings = []
    for sq, sk in pairs:
        for flip in (False, True):
            # Printed *before* the launch and flushed, so a hard abort still
            # names the pair it died on -- which is the whole point of running
            # the sweep in one process per build.
            print(f"[run] sq={sq:<4} sk={sk:<5} flip={str(flip):<5}", end=" ", flush=True)
            report = _run_one(torch, fwd, dkdv, dqk, b, h, sq, sk, d, causal, dtype, flip, dropout)
            if report:
                print("FINDING", flush=True)
                for line in report:
                    print(f"        {line}", flush=True)
                findings.append((sq, sk, flip, report))
            else:
                print("clean", flush=True)
    return findings


def _run_one(torch, fwd, dkdv, dqk, b, h, sq, sk, d, causal, dtype, flip, dropout):
    g = torch.Generator(device="cuda").manual_seed(0)
    qb, q, ql = _poisoned_in(torch, b, h, sq, d, dtype, g, flip)
    kb, k, kl = _poisoned_in(torch, b, h, sk, d, dtype, g, flip)
    vb, v, vl = _poisoned_in(torch, b, h, sk, d, dtype, g, flip)
    dob, do, dol = _poisoned_in(torch, b, h, sq, d, dtype, g, flip)

    # Reference o/lse/delta from torch on the clean views. The forward's own
    # output is checked but thrown away, so a forward bug cannot mask a
    # backward one by poisoning its inputs.
    s = (q.float() @ k.float().transpose(-1, -2)) * (1.0 / d**0.5)
    if causal:
        i = torch.arange(sq, device=q.device)[:, None]
        j = torch.arange(sk, device=q.device)[None, :]
        s = s.masked_fill(j > i, float("-inf"))
    alive = torch.isfinite(s).any(-1, keepdim=True)
    pmat = torch.where(alive, torch.softmax(s, -1), torch.zeros_like(s))
    o_ref = (pmat @ v.float()).to(dtype)
    lse2 = torch.where(
        alive.squeeze(-1), torch.logsumexp(s, -1), torch.full(s.shape[:-1], float("inf"), device=q.device)
    )
    delta2 = (do.float() * o_ref.float()).sum(-1)

    def _row_tensor(src):
        pad = torch.full((b * h + 2, sq), float("nan"), dtype=torch.float32, device="cuda")
        pad[1 : 1 + b * h].copy_(src.reshape(b * h, sq))
        return pad, pad[1 : 1 + b * h]

    lse_buf, lse = _row_tensor(lse2)
    delta_buf, delta = _row_tensor(delta2)

    report = []

    ob, o_k, ol = _poisoned_out(torch, b, h, sq, d, dtype, flip)
    lse_k = torch.full((b * h, sq), float("-inf"), dtype=torch.float32, device="cuda")
    fwd(q, k, v, o_k, b, sq, seqlen_k=sk, lse=lse_k)
    torch.cuda.synchronize()
    _check_out(torch, "O", ob, o_k, ol, report)

    dkb, dk, dkl = _poisoned_out(torch, b, h, sk, d, dtype, flip)
    dvb, dv, dvl = _poisoned_out(torch, b, h, sk, d, dtype, flip)
    dkdv(
        q, k, v, do, dk, dv, lse, delta, b, sq,
        seqlen_k=sk, dropout_p=dropout or None, philox_seed=0x1234, philox_offset2=0,
    )  # fmt: skip
    torch.cuda.synchronize()
    _check_out(torch, "dK", dkb, dk, dkl, report)
    _check_out(torch, "dV", dvb, dv, dvl, report)

    dqb, dq, dql = _poisoned_out(torch, b, h, sq, d, dtype, flip)
    dqk(q, k, v, do, dq, lse, delta, b, sq, sk)
    torch.cuda.synchronize()
    _check_out(torch, "dQ", dqb, dq, dql, report)

    for nm, buf, live in (("Q", qb, ql), ("K", kb, kl), ("V", vb, vl), ("DO", dob, dol)):
        _check_in(torch, nm, buf, live, report)
    for nm, buf in (("lse", lse_buf), ("delta", delta_buf)):
        n = int((~torch.isnan(buf[0])).sum()) + int((~torch.isnan(buf[-1])).sum())
        if n:
            report.append(f"OOB WRITE {nm} guard plane: {n} element(s) overwritten")
    return report


def main():
    global COL_MARGIN, ROW_GUARD
    ap = argparse.ArgumentParser()
    ap.add_argument("index", nargs="?", type=int, help="config index; see --list")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--heads", type=int, default=5)
    ap.add_argument("--pair", type=str, default=None, help="one 'sq,sk' instead of the sweep")
    ap.add_argument("--observed", action="store_true", help="only the shapes AOTriton aborts on")
    ap.add_argument("--only", type=str, default=None, help="'hd,causal,dtype' -- build exactly this one")
    ap.add_argument("--col-margin", type=int, default=COL_MARGIN, help="poison elements each side of a row")
    ap.add_argument("--row-guard", type=int, default=ROW_GUARD, help="poison rows past the sequence extent")
    ap.add_argument("--dropout", type=float, default=0.0, help="dK/dV dropout probability (a build axis)")
    a = ap.parse_args()
    COL_MARGIN, ROW_GUARD = a.col_margin, a.row_guard
    if a.observed:
        # One process per distinct build, each running only its own pairs.
        bad, dirty = [], []
        builds = sorted({(hd, causal, dt, drop) for _, _, hd, causal, drop in OBSERVED for dt in ("bf16", "f16")})
        for hd, causal, dt, drop in builds:
            pairs = [(sq, sk) for sq, sk, h, c, d in OBSERVED if (h, c, d) == (hd, causal, drop)]
            for sq, sk in pairs:
                argv = [sys.executable, os.path.abspath(__file__), "0",
                        "--batch", str(a.batch), "--heads", str(a.heads),
                        "--col-margin", str(a.col_margin), "--row-guard", str(a.row_guard),
                        "--dropout", str(drop), "--pair", f"{sq},{sk}",
                        "--only", f"{hd},{causal},{dt}"]  # fmt: skip
                r = subprocess.run(argv)
                tag = f"hd{hd} sq{sq} sk{sk} causal={causal} {dt} p={drop}"
                print(f"[obs] {tag} exit={r.returncode}", flush=True)
                if r.returncode == 2:
                    dirty.append(tag)
                elif r.returncode != 0:
                    bad.append((tag, r.returncode))
        print(f"[done] {len(dirty)} finding(s): {dirty}")
        print(f"[done] {len(bad)} crash(es): {bad}")
        return 1 if (bad or dirty) else 0

    cfgs = _configs()
    if a.list:
        for i, c in enumerate(cfgs):
            print(i, f"head_dim={c[0]} causal={c[1]} {c[2]}")
        return 0
    if a.all:
        bad, dirty = [], []
        for i, c in enumerate(cfgs):
            argv = [sys.executable, os.path.abspath(__file__), str(i),
                    "--batch", str(a.batch), "--heads", str(a.heads),
                    "--col-margin", str(a.col_margin), "--row-guard", str(a.row_guard),
                    "--dropout", str(a.dropout)]  # fmt: skip
            if a.pair:
                argv += ["--pair", a.pair]
            r = subprocess.run(argv)
            tag = f"head_dim={c[0]} causal={c[1]} {c[2]}"
            print(f"[cfg] {i} ({tag}) exit={r.returncode}", flush=True)
            if r.returncode == 2:
                dirty.append((i, tag))
            elif r.returncode != 0:
                bad.append((i, tag, r.returncode))
        print(f"[done] {len(dirty)} config(s) reported findings: {dirty}")
        print(f"[done] {len(bad)} config(s) crashed: {bad}")
        return 1 if (bad or dirty) else 0

    pair = tuple(int(x) for x in a.pair.split(",")) if a.pair else None
    if a.only:
        _hd, _c, _dt = a.only.split(",")
        cfg = (int(_hd), _c == "True", _dt)
    else:
        cfg = cfgs[a.index or 0]
    found = _run_config(cfg, a.batch, a.heads, pair, a.dropout)
    return 2 if found else 0


if __name__ == "__main__":
    sys.exit(main())
