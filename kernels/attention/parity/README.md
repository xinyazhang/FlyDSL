# `kernels/attention/parity`

The gfx1201 / RDNA4 flash-attention family. **Parity** because the goal is
functional equivalence with AOTriton's `attn_fwd` — same masking modes, same
varlen layouts, same BHSD ABI — so that this kernel can replace the Triton one
there rather than sit beside it.

Separate from `kernels/attention/` for two reasons. The functional surface is
large enough to be its own thing (causal and both bottom-right variants,
sliding window, GQA/MQA, varlen in five layouts, bias, dropout, logsumexp).
And these files are a **self-contained prototype run in place** — bare module
imports, cwd must be this directory — rather than a package imported from
elsewhere.

## Running it

```bash
export ROCM_PATH=$(rocm-sdk path --root)     # else the JIT dies with "lld invocation failed"
cd kernels/attention/parity
python3 -m pytest test_flash_attn_func_gfx1201_aiw.py -q     # 298
python3 -m pytest test_flash_attn_func_gfx1201.py -q         # 94, via the public API
```

`gfx1201_standalone.py` is what makes the bare imports work: it puts the
repository root on `sys.path` so `kernels/common/*` resolves to the working
tree. Nothing here is collected by `scripts/run_tests.sh`.

## The files

| what | where |
|---|---|
| kernel builder | `flash_attn_func_gfx1201_aiw.py` |
| arch helpers (LDS, WMMA, apertures, varlen decode) | `fmha_common_gfx1201.py` |
| tuning policy — which knobs, for which shape | `fmha_tuning_gfx1201.py` |
| public API | `flash_attn_func_gfx1201_interface.py` |
| PRNG, and the debug dropout-mask kernel | `philox.py`, `dropout_mask_gfx1201.py` |
| correctness | `test_flash_attn_func_gfx1201{,_aiw}.py`, `test_dropout_mask_gfx1201.py`, `test_philox.py` |
| performance | `perf_ab.py` (interleaved A/B), `bench_fmha.py`, `bench_one.py`, `bench_aiw_ab.py` |
| codegen gate | `codegen_fingerprint.py` — VGPR, scratch, instruction count, ISA hash |
| accuracy vs fp64 | `accuracy_probe.py` |

## The written record

`sdpa_lore_gfx1201.md` is the measured-facts file — read it before re-deriving
a number. `sdpa-*.md` are the plans, each carrying an outcome section saying
what actually happened and where it departed from the plan; the most recent are
`sdpa-common-preload-executive.md` (the `Aperture` refactor) and
`sdpa-fix-unstable.md` (the API-stability pass). `gfx1201_fmha.md` is the
architecture note.
