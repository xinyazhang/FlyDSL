# gfx1201 FMHA prototype

Notes for `bench_fmha.py`, `bench_shim.py`, `flash_attn_func_gfx1201.py`, and
`flash_attn_func_gfx1201_interface.py` — the dense flash-attention prototype for
gfx1201 (RDNA4, Navi48).

Everything below is either gfx1201-specific or still-settling vocabulary. The
arch-neutral half of the glossary should graduate into
[`docs/kernel_tuning_guide.md`](../../docs/kernel_tuning_guide.md) once the terms
stop moving; see [Graduation](#graduation).

## Running it

```bash
export ROCM_PATH=$(rocm-sdk path --root)   # only needed for pip-installed ROCm
cd kernels/attention && python3 bench_fmha.py
```

`ROCM_PATH` must point at a ROCm tree containing `llvm/bin/ld.lld` and
`amdgcn/bitcode`. MLIR's ROCDL target resolves the linker and device bitcode
through it at serialization time and defaults to `/opt/rocm`, which does not
exist in a pip-only ROCm install. Without it the JIT fails with a bare
`lld invocation failed` followed by a full IR dump that buries the message.
`LD_LIBRARY_PATH` does *not* need setting — the wheels' runpath covers
`libamdhip64`.

One CSV per configuration is written to the cwd.

Correctness tests live in `test_flash_attn_func_gfx1201.py` and check both
scheduling variants against PyTorch SDPA. They are intentionally **not** wired
into `scripts/run_tests.sh`; run them the same way:

```bash
cd kernels/attention && python3 -m pytest test_flash_attn_func_gfx1201.py -v
```

## Scheduling variants

| builder | file | staging |
|---|---|---|
| `build_flash_attn_func_module` | `flash_attn_func_gfx1201.py` | baseline: V prefetched in registers, K loaded at distance 0 |
| `build_flash_attn_func_bp_module` | `flash_attn_func_gfx1201_bp.py` | binding prefetch: K *and* V in registers at distance 1 |

Select with `flydsl_flash_attn_func_gfx1201(..., use_binding_prefetch=True)`.
The two are currently bit-identical in output and within noise on throughput;
the variant exists to be tuned. See the file's docstring for the schedule and
the open scheduling issue (a conservative `s_wait_loadcnt_dscnt 0x0` before the
barrier drains the prefetch a few instructions after it is issued).

## Why these four files break directory conventions

They import each other by **bare module name** and depend on nothing but
`flydsl`, `torch`, and stdlib, so the directory can be copied into another
container and run as-is. That is deliberate: the kernel is benchmarked against
other containers running different flydsl versions, and any extra path or
environment configuration makes the comparison harder to trust.

Consequences, all intentional:

- No `kernels.*` imports, so `import kernels.attention.bench_fmha` does **not**
  work. Use the cwd convention above.
- `dtype_to_elem_type` and `_run_compiled` are **duplicated** into
  `flash_attn_func_gfx1201.py` rather than imported from
  `kernels.common.{kernels_common,tensor_shim}`, which would be unreachable
  without the repository root on `sys.path`.

Fold both back into the shared modules if this graduates out of prototype status.

Note `dtype_to_elem_type` differs between projects: FlyDSL's returns a Numeric
class, aiter's returns a raw MLIR `ir.Type`. Code feeding
`buffer_ops.get_element_ptr(elem_type=...)` needs the latter, via `.ir_type`.

## gfx1201 hardware capabilities

Verified with `llvm-mc -arch=amdgcn -mcpu=gfx1201` from
`$(rocm-sdk path --root)/llvm/bin`, which is an exact oracle for instruction
legality and encoding — cheaper and more reliable than reading the ISA guide for
"does this instruction exist" questions.

**No direct global→LDS copy.** Every spelling of `global_load_lds_*` /
`buffer_load_* … lds` is rejected. RDNA2 had it; RDNA3 dropped it and it has not
returned:

| target | direct-to-LDS |
|---|---|
| gfx90a | `buffer_load_dword … lds` |
| gfx942, gfx950 | `global_load_lds_dword`, `buffer_load_dword … lds` |
| gfx1030 | `buffer_load_dword … lds` |
| gfx1100, **gfx1201**, gfx1250 | none (gfx1250 uses TDM instead) |

This is the central design constraint: K/V tiles must transit VGPRs on the way
to LDS, so pipeline depth is bounded by the register budget rather than by LDS
capacity. It also means the gfx950 dual-wave design does not port directly — it
is built on `rocdl.raw_ptr_buffer_load_lds` plus `s_waitcnt vmcnt(N)`.

**Split barriers, and richer than CDNA's.** Plain `s_barrier` does *not* exist
on gfx1201; `gpu.barrier()` lowers to `s_barrier_signal` / `s_barrier_wait`.
Also available: `s_setprio`, `s_barrier_signal m0` (barrier IDs, i.e. named
barriers), `s_barrier_signal_isfirst`, `s_get_barrier_state`, `s_barrier_leave`.
Named split barriers are a genuine enabler for producer/consumer schemes.

**LDS alignment.** `fly.ptr_load` / `fly.ptr_store` emit no alignment attribute
(`PtrLoadOpLowering` in `lib/Conversion/FlyToROCDL/FlyToROCDL.cpp`), so LLVM
assumes the vector type's ABI alignment — 16 B for `v8f16`, 32 B for `v16f16`.
With `K_STRIDE = HEAD_DIM + 4` the rows are 264 B apart and only 8-byte aligned,
so that promise is false: it selects `ds_load_b128` on misaligned addresses,
which is undefined behaviour and was measured 2.2x slower (39 → 92 TFLOPS).
`_lds_load_v8` / `_lds_store_vx` therefore split into 8-byte `v4f16` accesses,
which carry a truthful `align 8` and fold into `ds_load2_b64` / `ds_store2_b64`.

Any wide `fx.ptr_load`/`ptr_store` against a padded, non-power-of-2-stride LDS
tile has this problem; kernels migrated from `SmemAllocator` to
`SharedAllocator` are worth auditing. The upstream fix is an alignment override
on the op.

## Roofline and measured bottlenecks

Dense WMMA peak is **~205 TFLOPS** measured on this (overclocked) board with
[`kernels/microbench/wmma_peak.py`](../microbench/wmma_peak.py); AMD's published
figure is **191 TFLOPS** at reference clock. Use 191 as the denominator for
"% of theoretical" so numbers stay comparable off this machine. The kernel runs
at ~92 TFLOPS, i.e. **~48% of spec**.

**WMMA is a VALU instruction on RDNA4** — VOP3P encoded, documented in section
7.12 of the ISA doc's *Vector ALU Operations* chapter. There is no separate
matrix unit as on CDNA, so softmax VALU work contends directly with WMMA issue.
Do not reason about this kernel as if MFMA-style co-issue exists.

Component costs, from ablations measured **device-side** at N=4096 (each removes
one thing and is deliberately numerically incorrect):

| ablation | vs baseline |
|---|---|
| GEMM2 V-operand reads replaced by a constant | **-22%** (largest single item) |
| all barriers removed | -7.5% |
| O-rescale removed (64 `v_mul`) | -1.5% |
| global KV streaming removed (all cache hits) | +1.5% — **not DRAM bound** |
| all compute dead | -83% — the kernel is compute-dominated |

What has been tried against the V-operand cost:

- **V staged transposed in LDS** (`V^T[d][kv]`, filled via `global_load_tr_b128`
  so the store stays contiguous): **+2.7%**, landed. Turns 128 scalar
  `ds_load_u16` into 16 `ds_load_2addr_b64`.
- **V bypassing LDS entirely**, WMMA operand fetched per-wave straight from
  global with `global_load_tr_b128`: **-15%, rejected.** All 8 waves need the
  same V tile, so removing the LDS staging makes each wave re-fetch it — the
  redundancy and longer global latency cost more than the LDS reads saved.
  LDS staging is earning its keep as a sharing mechanism.
- **Double-buffered K** to drop a barrier: no gain (V still needs its own
  publish/protect pair, so the barrier count does not actually fall).

So the 22% is mostly *inherent* to getting V into WMMA-operand form, not to how
the reads are spelled. Reducing it further likely needs higher arithmetic
intensity — each of the 8 waves reads the whole K and V tile every iteration
because it owns only 16 of 128 Q rows.

### Measuring this kernel

Two traps, both of which produced confidently wrong conclusions:

1. **Never `sys.path.insert` a directory containing same-named modules.** A
   benchmark that inserted the repo's `kernels/attention` (to pick up
   `bench_shim`) silently imported the *repo* kernel instead of the variant
   under test, so six different ablations all benchmarked the same binary and
   agreed to within 1%. Copy the harness into the variant's directory instead.
2. **Cross-check against device time.** `rocprofv3 --stats --kernel-trace` gives
   per-kernel duration from the `top_kernels` table; it agrees with CUDA-event
   timing to ~4% when the harness is correct, and diverges wildly when it is not.

Run-to-run spread is a few percent, so use medians over repeats and treat
non-overlapping min/max ranges as the bar for believing a small win.

## Terminology

Four distinct concepts get conflated in attention-kernel discussions, and
"ping-pong" already means something specific in this repository. Names below
follow the CPU compiler and architecture literature.

| concept | name | scope |
|---|---|---|
| HW copy landing in LDS (`buffer_load … lds`), CDNA3/4 only | **non-binding prefetch** | data path |
| global → VGPR → LDS, issued δ iterations ahead (the only option on gfx1201) | **binding prefetch** | data path, intra-wave |
| two LDS buffers, one filled while the other feeds compute | **LDS ping-pong** / double-buffering | LDS allocation |
| two wave groups alternating GEMM and softmax phases under barriers | **dual-wave SWP** (`DUALWAVE_SWP`) | inter-wave |
| dedicated producer waves feeding consumer waves | **decoupled access-execute** (DAE) / warp specialization | inter-wave |

Rows 2, 4 and 5 are orthogonal and can be active simultaneously: a dual-wave
gfx1201 kernel would use binding prefetch *inside* each wave role.

Supporting terms:

- **Prefetch distance (δ)** — how many iterations ahead a load is issued.
  Classically `δ = ⌈memory latency / iteration time⌉` (Mowry, Lam & Gupta,
  ASPLOS-V 1992). Distinct from the existing `NUM_PREFETCH_K` / `NUM_PREFETCH_V`
  constants, which are buffer *depth*.
- **Software pipelining** / **modulo scheduling** — the loop transform that
  overlaps iterations (Rau & Glaeser; Lam).
- **Modulo variable expansion** — replicating the loop body so overlapping value
  lifetimes get distinct registers. The transform whose absence shows up as
  `global_load v[97:100]; s_wait_loadcnt 0x0; ds_store; global_load v[97:100]` —
  one register quad reused for two in-flight values, serializing loads that
  could have overlapped.
- **Memory-level parallelism (MLP)** — how many loads are in flight at once.
- **Loss of decoupling (LoD)** — when the access stream cannot run ahead of the
  execute stream, the failure mode any DAE/warp-specialized design must avoid.

Why *binding* prefetch: in the prefetching literature a **non-binding** prefetch
targets a cache and binds its value at reference time, while a **binding**
prefetch loads into a register and binds at prefetch time. The literature also
flags the exact cost we hit — binding prefetch "may require a lot of registers
to receive the load results". Since gfx1201 has no non-binding prefetch at all,
that one sentence states the whole constraint.

Avoid calling intra-wave load/ALU overlap "ping-pong": in FlashAttention-3 and
gfx950 usage that word means inter-warpgroup alternation, and in this repository
it already means LDS double-buffering.

References: [Software pipelining](https://en.wikipedia.org/wiki/Software_pipelining) ·
[Lam, *Software Pipelining for VLIW Machines*](https://suif.stanford.edu/papers/lam-sp.pdf) ·
[Rau, *Iterative Modulo Scheduling*](https://shiftleft.com/mirrors/www.hpl.hp.com/techreports/94/HPL-94-115.pdf) ·
[VanderWiel & Lilja, *Data Prefetch Mechanisms*](https://www.ece.lsu.edu/tca/papers/vanderwiel-00.pdf) ·
[Lee, Kim & Vuduc, *When Prefetching Works, When It Doesn't, and Why*](https://faculty.cc.gatech.edu/~hyesoon/lee_taco12.pdf) ·
Smith, *Decoupled Access/Execute Computer Architectures*, ISCA '82
([lecture notes](https://safari.ethz.ch/digitaltechnik/spring2022/lib/exe/fetch.php?media=onur-digitaldesign_comparch-2022-lecture19c-dae-beforelecture.pdf))

## Benchmarking

`bench_shim.py` is a triton-free reimplementation of `triton.testing`
(`do_bench`, `Benchmark`, `perf_report`). It preserves upstream's millisecond
*budget* semantics for `warmup`/`rep` and its cache flush, so numbers stay
comparable with triton-based runs; it emits the same CSV format and filenames,
but no plots (matplotlib is not a FlyDSL dependency).

`flydsl.autotune.do_bench` is **not** a substitute: its `warmup`/`rep` are raw
iteration counts, it returns the median rather than the mean, and it does not
flush caches between runs.

## Graduation

When the terminology settles, move the [Terminology](#terminology) section into
`docs/kernel_tuning_guide.md` (it is arch-neutral and applies equally to GEMM,
MoE and conv), leave the gfx1201-specific sections here, and update the pointer
in that guide's LDS double-buffering section.
