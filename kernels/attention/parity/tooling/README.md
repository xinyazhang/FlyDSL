# `parity/tooling`

Development harness for the gfx1201 attention kernels: benchmarks, the codegen
gate, the accuracy probe, and the ISA/MLIR dumper. Nothing here ships — the
kernels, interfaces, tuning modules and tests live one level up.

Run from this directory, with `ROCM_PATH` exported:

```bash
export ROCM_PATH=$(rocm-sdk path --root)
cd kernels/attention/parity/tooling
```

`_bootstrap.py` puts `parity/` on `sys.path`; import it before any kernel
import. `gfx1201_standalone.py` (one level up) does the same job for
`kernels/common`.

## Shape is the ABI; layout is a knob

`qkv.make_qkv(...)` builds **BHSD-shaped** tensors in either memory layout:

| layout | allocation | a head's tokens are |
|---|---|---|
| `bhsd` (default) | contiguous `(B, H, S, D)` | `D` apart |
| `bshd` | `(B, S, H, D)`, transposed to BHSD shape | `H*D` apart |

Both satisfy the kernel's only layout constraint, `stride(3) == 1`, and both
are worth measuring — the access patterns genuinely differ. Set
`FLYDSL_AB_LAYOUT` for `perf_ab.py` or `FLYDSL_BENCH_LAYOUT` for the benches.

**This exists because every script used to hardcode the shape and imply the
layout from whichever `torch.randn` argument order someone typed.** Three of
them kept building BSHD-shaped tensors for months after the ABI moved to BHSD.
The comparisons stayed valid — both sides of a diff used the same wrong config
— but nobody could tell from the output which config that was.

## The scripts

| script | what |
|---|---|
| `perf_ab.py` | interleaved A/B of two git revisions, sharded over GPUs. **Run `--base X --head X` first**: the ratio it reports for a revision against itself is its own resolution, and anything inside that is not a measurement. |
| `codegen_fingerprint.py` | VGPR, scratch, instruction count and ISA hash at three configs. Deterministic, so it is the sound predicate for `git bisect run` — the benchmark is not. |
| `dump_isa.py` | one compilation stage for one config. `--dtype`, `--pitch`, `--layout`, `--fp-mode`, `--stage`. Replaces four drifting copies. |
| `isa_stats.py` | per-loop ISA statistics, keyed on the KV loop by *containing WMMA* — never by region size, which picked different loops in two builds and invented a regression. |
| `accuracy_probe.py` | forward error against an fp64 reference, beside torch SDPA. |
| `bench_fmha.py`, `bench_one.py`, `bench_aiw_ab.py` | throughput sweeps. `bench_shim.py` is a triton-free `do_bench`. |
| `sdpa_efficiency_gfx1201.py` | the roofline model. |

## Two rules learned the hard way

**Re-dump the baseline from the actual parent commit.** A stale reference
inverted a conclusion twice in this kernel's history.

**A gate is only as good as its harness.** Three separate bugs in this
directory silently weakened one rather than failing it: a hardcoded absolute
path in `codegen_fingerprint`, dump scripts pointing at the pre-`parity/`
location, and those same scripts building the pre-BHSD shape. All three were
invisible because the wrong thing still produced a number. That is why these
scripts are committed rather than living in `/tmp`.
