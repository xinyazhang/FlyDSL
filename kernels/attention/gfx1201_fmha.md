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
| `build_flash_attn_func_bp_module` | `flash_attn_func_gfx1201_bp.py` | binding prefetch: K *and* V in registers at distance 1; V staged transposed in LDS |
| `build_flash_attn_func_m32_module` | `flash_attn_func_gfx1201_m32.py` | **head_dim 64 only:** two Q row-tiles per wave (BLOCK_M=256), so one K/V operand feeds two WMMAs. **+14% at N=4096** |

Select with `use_binding_prefetch=True` or `variant="m32"`.
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

### Register budget curve

Price any change that adds live state against this before building it. Measured
by holding N extra `v8f32` accumulators live across the KV loop (head_dim 128):

| extra VGPRs | total | waves/SIMD | TFLOPS | cost |
|---|---|---|---|---|
| +0 | 150 | 9 | 92.1 | — |
| +64 | 222 | 6 | 89.8 | -2.5% |
| +96 | 247 | 5 | 76.0 | -17% |
| +112 | 256 (spills) | 5 | 67.3 | -27% |

So roughly **+64 VGPRs is affordable**; the cliff is between 6 and 5 waves/SIMD,
and spilling is catastrophic. Doubling a wave's Q row-tiles costs
`o_accs + q_b_packs + s_accs`, which scales with head_dim: +64 at head_dim 64
(fits, and `m32` takes it) but +112 at head_dim 128 (does not).

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

`bench_one.py` measures exactly one `(variant, causal)` config per process.
Prefer it over a multi-config sweep when A/B-ing a change, for the reason in
the next section.

### This board drifts ~5% between runs; always interleave A/B

Two arms of an A/B run back to back can differ by 5% from clock/thermal state
alone. A sweep that measures "before" as one whole script run and "after" as
another will attribute that drift to the change. A `max-memory-clause` A/B was
first measured this way at +4.8%; interleaving the arms and repeating three
times put the true effect at +0.4%, and re-running the original sweep
reproduced 96.3 then 91.2 TFLOPS for the *same binary*.

Interleave the arms (`for rep; for arm`) and repeat at least three times. Real
effects come out flat to ±0.1 TFLOPS across reps; anything that moves more than
that between reps of the same arm is drift, not signal.

## head_dim coverage

The baseline kernel handles any `head_dim` that is a multiple of `WMMA_K = 16`,
from 16 to 256. There is one kernel, not one per head_dim; tile sizes come from
`default_block_m()` in `flash_attn_func_gfx1201.py`.

Measured at B=1 H=8 N=4096 f16 non-causal, with per-wave VGPR use and spills:

| head_dim | VGPR | spills | TFLOPS |
|---|---|---|---|
| 64 | 97 | 0 | 85.0 |
| 128 | 148 | 0 | 93.0 |
| 160 | 237 | 0 | 80.8 |
| 192 | 256 | 24 | 67.2 |
| 224 | 256 | 64 (BM=64) | 50.9 |
| 256 | 256 | 36 | 67.2 |

Throughput falls off above 160 because the kernel starts spilling. Two per-wave
terms scale with head_dim and **neither depends on BLOCK_M or BLOCK_N**:

- `o_accs` = `head_dim/2` VGPRs -- a wave owns `WMMA_M`=16 Q rows x head_dim
  outputs, which is 16*head_dim/32 floats per lane.
- `q_b_packs` = `head_dim/4` VGPRs.

So tile size is a weak lever on spilling. It is not a null one, because it
changes the cooperative-load geometry -- head_dim 224 spills 101 registers at
BLOCK_M=128 (33.5 TFLOPS) versus 64 at BLOCK_M=64 (50.9), which is why 224 is
the one entry in `_BLOCK_M_BY_HEAD_DIM`. Everywhere else BLOCK_M=128 wins.

### head_dim above 256

`head_dim` 512 is rejected. Both limits are structural, and **smaller blocks
cannot fix either**:

- `o_accs` alone is 256 VGPRs, the entire register file, before Q packs,
  S accumulators or addressing.
- The K+V LDS tile is `32 * (516 + 516) * 2` = 66048 B, over the 64 KB
  workgroup limit. `BLOCK_N` is already at its floor of 32, because
  `BLOCK_N % K_SUB_N == 0` and `K_SUB_N` is 32.

The fix is to give the kernel a V/output width separate from its QK width.
Attention is column-separable in V -- `O[:, dslice] = P @ V[:, dslice]`, and P
does not depend on V at all -- so a `head_dim_v < head_dim` shrinks `o_accs`,
the V LDS tile and the output write, while GEMM1 keeps the full reduction.
head_dim 512 at a 128-wide V slice gives `o_accs` 64, LDS 41472 B, both
comfortable.

The cost is that GEMM1 and the K LDS traffic repeat once per slice: at 512 with
four slices that is 4x64 + 64 = 320 WMMA against an ideal 128, so ~2.5x the
matmul work. Avoiding the recompute means staging S in LDS and having each wave
consume a slice of it, which is a larger restructure. The same `head_dim_v`
generalisation is what MLA-style shapes (d_qk != d_v) need, so it is worth doing
properly rather than as a 512 special case.

## P precision: f32 P through GEMM2 is not possible on gfx1201

RDNA4 WMMA has no F32xF32 form (ISA manual Table 41) -- A/B operands are
f16/bf16/iu8/iu4/fp8 only, with an f32 *accumulator*. LLVM does define
`v_wmma_f32_16x16x4_f32`, but only under `VOP3P_Real_WMMA_gfx1250`; it is a
gfx1250 instruction and does not exist on gfx1201. The AOTriton idiom
`acc += tl.dot(p, v.to(p.type.element_ty))`, which keeps P in f32 and upcasts V
to match, relies on CDNA's `v_mfma_f32_16x16x4f32` and has no gfx1201
equivalent. Doing PV in f32 here means abandoning the matrix cores for GEMM2.

This is another case where gfx1250 and gfx1201 diverge despite adjacent arch
numbers -- check which target a WMMA form is real-ized for before assuming it.

Two facts worth knowing about the current numerics, measured against an fp64
reference by `accuracy_probe.py` (B=1 H=4 N=1024 d=128):

- **V is never downcast.** It reaches GEMM2 at the input tensor's native 16-bit
  width via LDS. Only P loses precision.
- **There is no output bias**, for either dtype. P's rounding error is one-sided
  for bf16 (truncation is round-toward-zero, and P > 0 always), but O sums P*V
  with zero-mean V, so it cancels and shows up as variance instead. Note the
  numerator and denominator do disagree by construction -- `l` sums exact f32 P
  while O accumulates rounded P -- so the error does not cancel in O/l.

| dtype | flydsl RMS | torch SDPA RMS | ratio |
|---|---|---|---|
| f16 | 3.51e-4 | 3.50e-4 | 1.00 |
| bf16 | 4.43e-3 | 2.78e-3 | 1.59 |

f16 is at exact parity. bf16 sits at 1.6x because `bf16_trunc_pack_v8` truncates
(RMS 0.577 ulp) rather than rounding to nearest-even (0.289 ulp). RTNE --
`x += 0x7FFF + ((x >> 16) & 1)` before the shift -- closes the gap exactly
(measured 2.79e-3) but costs 2-3% on bp and 2.7-5.4% on m32. **Kept as
truncation by decision, not oversight.**

## The GCN scheduler serializes LDS loads by default

At head_dim 128 the loop body was spending its time stalled on LDS, not on
VALU. The default scheduler sinks each `ds_load` next to the `v_wmma` that
consumes it and lets the register allocator funnel every one of them through a
single VGPR quad. That creates a WAR dependency on the preceding WMMA, so
`SIInsertWaitcnts` has to emit a full `s_wait_dscnt 0x0` between every load and
its use:

```
ds_load_2addr_b64 v[139:142], v121 offset0:16 offset1:17
s_wait_dscnt 0x0                                    <- full drain
v_wmma_f32_16x16x16_f16 v[105:112], v[139:142], ...
ds_load_2addr_b64 v[139:142], v128 offset1:1        <- same quad, must wait
s_wait_dscnt 0x0
```

33 of 35 waits in the loop were full drains. Setting LLVM's
`amdgpu-sched-strategy` function attribute (via the existing `passthrough`
block) to `max-memory-clause` spreads the loads across distinct quads and drops
full drains to 14, with the rest becoming `0x1`-`0x3`; VGPRs go 147 -> 157 with
no spills.

It is **not** a universal win -- measured at BATCH=2 H=12 N=4096 d=128 f16:

| variant | causal | default | max-memory-clause |
|---|---|---|---|
| baseline | no | 89.4 | 88.6 |
| baseline | yes | 69.8 | **79.2** |
| bp | no | 91.4 | 91.9 |
| bp | yes | 85.6 | **88.5** |
| m32 (d=64) | no | **95.5** | 90.6 |
| m32 (d=64) | yes | 84.2 | 83.9 |

So it is enabled unconditionally in `_bp`, only for `causal` in the baseline,
and not at all in `m32`. `max-ilp` was also tried and is a consistent loss
(-4.5%). Override with `FMHA_SCHED_STRATEGY=` (empty for the stock scheduler).

Two consequences worth remembering. Restructuring the *source* to batch the
loads does nothing -- the backend rescheduled a hand-batched GEMM1 to a
byte-identical schedule, so this is only reachable through the attribute. And
because the loop is latency-bound rather than VALU-bound, cutting VALU work
does not help: replacing the 16 `v_cvt_f16_f32` (`WriteFloatCvt`, 4 cycles
each) + 8 `v_pack_b32_f16` P-conversion with 8 `v_cvt_pk_rtz_f16_f32` removed a
verified 64 VALU cycles per iteration and changed throughput by 0.4%. That
saving is still available to bank if the kernel ever becomes VALU-bound; it was
reverted here because it buys nothing today and changes rounding to RTZ.

## Graduation

When the terminology settles, move the [Terminology](#terminology) section into
`docs/kernel_tuning_guide.md` (it is arch-neutral and applies equally to GEMM,
MoE and conv), leave the gfx1201-specific sections here, and update the pointer
in that guide's LDS double-buffering section.
