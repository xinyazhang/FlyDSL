#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""B4: is any `ds_read_b64_tr_b16` reachable with EXEC narrowed?

CDNA4 §11.4 requires **EXEC all 1s** across the LDS transpose reads. A read
inside a divergent region is *undefined*, and the failure mode is this
codebase's speciality: finite, wrong, no diagnostic. Through B3.5 the hazard
could not fire, because neither backward kernel contained a single `scf.if`.
**B4 is the phase that changes that** -- causal and window masking introduce
real conditional regions, and the transpose reads are the load-bearing path in
both MFMA families.

The invariant the kernel is written to hold is stronger than "no read inside an
`if`": every mask predicate is **wave-uniform**, built from the wave's own KV
row range and the tile index, so the branch is *scalar* and EXEC is never
narrowed anywhere in the kernel. That is checkable, and it is what this script
checks.

Two levels of report, so a future legitimate EXEC write does not turn this into
a test nobody can satisfy:

- **`exec_writes`** -- how many instructions write EXEC at all. The kernels
  currently emit zero, at every configuration below. A nonzero count is not by
  itself a bug; it means the second check starts doing work.
- **`unsafe_reads`** -- transpose reads reached while EXEC may be narrowed,
  by a conservative linear scan. This is the one that must be zero.

The scan is linear rather than a real CFG walk, and deliberately conservative:
a narrowing write puts it in the "narrowed" state until an explicit restore, so
control flow that rejoins without one is *reported* rather than missed. A false
positive here costs an investigation; a false negative costs a wrong gradient.

**Varlen (B5) adds a second conditional region** -- the `active` guard around
the whole body, for a workgroup whose KV block is past this sequence's keys --
and it is the one that would matter most if it were divergent, because the
transpose reads are inside it rather than beside it. Its predicate is
workgroup-uniform, so it is a scalar branch too; that is what these
configurations check.

**GQA (B7) adds the first nested runtime region.** The group loop wraps the
tile loop, so there are now two loop-carried `scf.for`s one inside the other,
and the accumulators live across both. Nested control flow is where a compiler
is most likely to materialise a mask rather than a scalar branch, so the
`kvh=1` arms below are the ones that would catch it. Bias, by contrast, is
per-element arithmetic with no branch of its own -- it is here because it is
new, not because it is suspicious.

**Dropout (B6) is the one feature that is genuinely per-lane**, and so the one
most likely to make the compiler reach for `v_cmpx`: `keep` is a different bit
in every lane and the kernel selects on it. It is written as `select` rather
than as control flow precisely so that it stays a VALU predicate, and these
configurations are what confirms the compiler agreed.

**A tooling bug this script had through B5, recorded because it is the failure
mode a scanner is prone to:** `--varlen` was parsed by the child and accepted
by `check()` but never placed in the child's argv, so every "varlen" arm
silently compiled a *dense* kernel and reported it clean. The scan was honest
about what it scanned and wrong about what that was. A checker that can only
report "clean" needs its own negative control -- here the `tr_reads == 0`
guard, which is why the arms were at least known to be looking at a kernel.

    python3 check_exec_hazard_gfx950.py            # every configuration
    python3 check_exec_hazard_gfx950.py 128 --rows 16 --causal 1 --varlen 1
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARITY = os.path.dirname(_HERE)

# Instructions that can *narrow* EXEC. `saveexec` forms both narrow and stash
# the old mask; `v_cmpx` writes EXEC directly.
_NARROW = re.compile(
    r"^\s*(s_and_saveexec_b64|s_or_saveexec_b64|s_xor_saveexec_b64|s_andn2_saveexec_b64"
    r"|s_nand_saveexec_b64|s_nor_saveexec_b64|s_xnor_saveexec_b64|s_and_b64\s+exec"
    r"|s_andn2_b64\s+exec|v_cmpx_\w+)\b"
)
# Explicit restores: a move or an or into EXEC.
_RESTORE = re.compile(r"^\s*(s_mov_b64\s+exec|s_or_b64\s+exec)\b")
_ANY_EXEC_WRITE = re.compile(r"^\s*\w+\s+exec\b|^\s*(s_\w*saveexec\w*|v_cmpx_\w+)\b")
_TR_READ = re.compile(r"^\s*ds_read_b64_tr_b16\b")

_CHILD = r"""
import argparse, sys
sys.path.insert(0, {parity!r})
p = argparse.ArgumentParser()
p.add_argument("head_dim", type=int)
p.add_argument("--rows", type=int, default=32)
p.add_argument("--causal", type=int, default=0)
p.add_argument("--window", type=int, default=0)
p.add_argument("--varlen", type=int, default=0)
p.add_argument("--dropout", type=int, default=0)
p.add_argument("--bias", type=int, default=0)
p.add_argument("--kvheads", type=int, default=0)
p.add_argument("--dt", default="bf16")
a = p.parse_args()
import torch
import fmha_common_gfx1201 as fmha
from fmha_bwd_dkdv_gfx950 import build_fmha_bwd_dkdv_gfx950_module as build
d, B, H, S = a.head_dim, 1, 4, 256
HK = a.kvheads or H
_DT = torch.bfloat16 if a.dt == "bf16" else torch.float16
q = torch.randn(B, H, S, d, device="cuda", dtype=_DT)
do = torch.randn_like(q)
k = torch.randn(B, HK, S, d, device="cuda", dtype=_DT)
v = torch.randn_like(k)
dk, dv = torch.empty_like(k), torch.empty_like(v)
lse = torch.randn(B * H, S, device="cuda", dtype=torch.float32)
delta = torch.randn_like(lse)
kw = dict(mfma_rows=a.rows, dkv_shards=1, num_waves=4, block_kv=a.rows * 4, block_q=64,
          head_dim_granule=64 if d % 64 == 0 else 32)
fn = build(num_heads=H, head_dim=d, num_kv_heads=HK, causal=bool(a.causal), window=bool(a.window),
           varlen=bool(a.varlen), dropout=bool(a.dropout), bias=bool(a.bias),
           dtype_str=a.dt, **kw)
ka = dict(seqlen_k=S, scale=1.0 / d ** 0.5)
if a.window:
    ka["window"] = (fmha.WINDOW_BOTRIGHT, fmha.WINDOW_BOTRIGHT)
if a.dropout:
    u64 = lambda x: torch.tensor([x], device="cuda", dtype=torch.int64)  # noqa: E731
    ka.update(dropout_p=0.25, philox_seed=u64(1234), philox_offset1=u64(0), philox_offset2=0)
if a.bias:
    ka["bias"] = torch.randn(B, H, S, S, device="cuda", dtype=_DT)
if a.varlen:
    import fmha_abi_gfx1201 as _abi
    cu = torch.tensor([0, S], device="cuda", dtype=torch.int32)
    ka["varlen"] = _abi.varlen_compact(cu, cu, S, S, lse_tokens=S)
    ka["num_seqlens"] = 1
fn(q, k, v, do, dk, dv, lse, delta, B, S, **ka)
torch.cuda.synchronize()
"""


def build_isa(head_dim, rows, causal, window, varlen=0, dropout=0, bias=0, kvheads=0, dt="bf16"):
    """Compile one configuration in a child process and return its final ISA.

    The dump directory is removed on the way out. A `FLYDSL_DUMP_IR` tree is
    ~25MB per configuration and this function is called once per row of a
    sweep, so leaving them behind is not a tidiness question: an earlier run of
    this script left 152 of them holding 1.7GB and filled the disk underneath
    an unrelated test run. The `finally` matters more than the `rmtree` -- the
    calls that leaked were the ones that raised.
    """
    out = tempfile.mkdtemp(prefix="exec_hazard_")
    env = dict(os.environ)
    env.update(FLYDSL_RUNTIME_ENABLE_CACHE="0", FLYDSL_DUMP_IR="1", FLYDSL_DUMP_DIR=out)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_CHILD.format(parity=_PARITY))
        script = f.name
    try:
        subprocess.run(
            [
                sys.executable,
                script,
                str(head_dim),
                "--rows",
                str(rows),
                "--causal",
                str(int(causal)),
                "--window",
                str(int(window)),
                "--varlen",
                str(int(varlen)),
                "--dropout",
                str(int(dropout)),
                "--bias",
                str(int(bias)),
                "--kvheads",
                str(int(kvheads)),
                "--dt",
                str(dt),
            ],
            env=env,
            capture_output=True,
            timeout=900,
            check=True,
        )
        path = os.path.join(out, "fmha_bwd_dkdv_gfx950_kernel_0", "21_final_isa.s")
        with open(path) as fh:
            return fh.read()
    finally:
        os.unlink(script)
        shutil.rmtree(out, ignore_errors=True)


def scan(isa):
    """`(exec_writes, tr_reads, unsafe_reads)` for one ISA listing."""
    narrowed = False
    exec_writes = tr_reads = unsafe = 0
    for line in isa.split("\n"):
        if _RESTORE.match(line):
            narrowed = False
            exec_writes += 1
            continue
        if _NARROW.match(line):
            narrowed = True
            exec_writes += 1
            continue
        if _ANY_EXEC_WRITE.match(line):
            # Unrecognised write to EXEC. Assume the worst.
            narrowed = True
            exec_writes += 1
            continue
        if _TR_READ.match(line):
            tr_reads += 1
            unsafe += int(narrowed)
    return exec_writes, tr_reads, unsafe


def check(head_dim, rows, causal, window, varlen=0, dropout=0, bias=0, kvheads=0, dt="bf16"):
    isa = build_isa(head_dim, rows, causal, window, varlen, dropout, bias, kvheads, dt)
    exec_writes, tr_reads, unsafe = scan(isa)
    tag = (
        f"d{head_dim:<4} rows{rows:<3} causal={int(causal)} window={int(window)} "
        f"varlen={int(varlen)} dropout={int(dropout)} bias={int(bias)} kvh={kvheads or '-'} {dt}"
    )
    print(f"  {tag}  tr_reads {tr_reads:>4}  exec_writes {exec_writes:>3}  unsafe {unsafe:>3}")
    if tr_reads == 0:
        raise SystemExit(f"{tag}: no transpose read in the ISA at all -- the scan is not looking at the kernel")
    return unsafe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("head_dim", type=int, nargs="?")
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--causal", type=int, default=0)
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--varlen", type=int, default=0)
    p.add_argument("--dropout", type=int, default=0)
    p.add_argument("--bias", type=int, default=0)
    p.add_argument("--kvheads", type=int, default=0)
    a = p.parse_args()
    if a.head_dim is not None:
        raise SystemExit(
            1 if check(a.head_dim, a.rows, a.causal, a.window, a.varlen, a.dropout, a.bias, a.kvheads) else 0
        )
    bad = 0
    for head_dim in (64, 128):
        for rows in (32, 16):
            for causal, window, varlen, dropout, bias, kvh in (
                (0, 0, 0, 0, 0, 0),
                (1, 0, 0, 0, 0, 0),
                (1, 1, 0, 0, 0, 0),
                (0, 0, 1, 0, 0, 0),
                (1, 0, 1, 0, 0, 0),
                (0, 0, 0, 1, 0, 0),
                (1, 0, 0, 1, 0, 0),
                (0, 0, 1, 1, 0, 0),
                # B7. Bias is per-element arithmetic with no branch of its own;
                # GQA adds the group loop, which is the first *nested* runtime
                # region in this kernel and therefore the first chance for the
                # compiler to reach for a mask on the outer one.
                (0, 0, 0, 0, 1, 0),
                (0, 0, 1, 0, 1, 0),
                (0, 0, 0, 1, 1, 0),
                (0, 0, 0, 0, 0, 1),
                (1, 0, 0, 0, 0, 1),
                (0, 0, 1, 0, 0, 1),
                (0, 0, 0, 1, 0, 1),
                (1, 0, 1, 1, 0, 1),
            ):
                bad += check(head_dim, rows, causal, window, varlen, dropout, bias, kvh)
            # B8. f16 changes the output pack and the MFMA opcode, so "no EXEC
            # write anywhere" is a claim about a different instruction stream.
            # One arm per family rather than the full cross product: the dtype
            # does not touch control flow, so this is a check that it did not
            # start to, not a matrix.
            for causal, window, varlen, dropout, bias, kvh in (
                (0, 0, 0, 0, 0, 0),
                (1, 0, 1, 1, 0, 1),
                (0, 0, 0, 0, 1, 0),
            ):
                bad += check(head_dim, rows, causal, window, varlen, dropout, bias, kvh, "f16")
    print("EXEC HAZARD: clean" if not bad else f"EXEC HAZARD: {bad} transpose reads under a narrowed EXEC")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
