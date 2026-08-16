# Plan: `parity/flash_attn_func_gfx950.py` — AOTriton feature parity on the dualwave schedule

## Context

`kernels/attention/parity/` holds the gfx1201 flash-attention family, whose goal
is functional equivalence with AOTriton's Triton `attn_fwd` so it can *replace*
that kernel rather than sit beside it. `flash_attn_func_gfx1201_aiw.py` has
reached that parity: arbitrary strides, runtime `sm_scale`/`Num_head_q,k`/
`Hdim_qk,vo`, `PADDED_HEAD`, generalized sliding-window attention, five varlen
modes, bias, dropout, optional logsumexp. 57 test functions / 298 cases.

gfx950 has the opposite problem. `kernels/attention/flash_attn_gfx950.py` is a
hand-scheduled dual-wave software pipeline that reaches production speed, but
its surface is narrow: head_dim 64/128 only, one row stride per tensor, no
bias, no dropout, no windows, `sm_scale` and head counts baked at build time.

**The goal is gfx950 with aiw's feature surface at gfx950's speed.** Target:
900+ TFLOPS early (AOTriton's AOT Triton kernel reaches this on some configs),
**1100+ TFLOPS at every rung below head_dim 256**; 1300 TFLOPS is what the
Triton team's JIT kernel delivers and is the stretch.

### The one thing this plan gets right that the obvious approach gets wrong

Do **not** port aiw's structure. aiw's shape is a *consequence* of RDNA
constraints that do not exist here — gfx11/gfx12 WMMA cannot co-execute with
VALU, so every feature's VALU is directly additive to the critical path (hence
`K_PREFETCH_DIST`, `QK_SHARDS`, `VO_CHUNKS`, and plan1's risk table warning
about `s_waitcnt` bubbles), and RDNA has no direct global→LDS at all. On gfx950
MFMA and VALU co-execute, which is the entire premise of the dualwave schedule.
Copying aiw's structure imports RDNA's compromises and discards the schedule
that produces the throughput. **Features port across architectures; schedules
do not.**

### The gap is narrower than the feature list suggests

Reading `DualwaveSoftmaxHelper` (`kernels/attention/flash_attn_utils.py:3706`),
dualwave already has most of what gfx1201 spent a whole phase (P2) building:

| already satisfied | where |
|---|---|
| Per-tile mask gating — fully-live tiles cost one scalar compare and a not-taken branch, not per-element VALU. This is aiw's `MASK_STEPS`, natively. | `causal_mask_prologue_if_needed`, `seq_pad_mask_if_needed` |
| Mask written as a *relative diagonal*, `rel = q_row + delta − kv_start − lane_off`, already parameterized by `delta` — the top-left/bottom-right alignment shift | `_causal_mask_inplace` |
| Anti-FMA numerics: Q pre-scaled by `sm_scale·log2e` **before** the row max, which is the correction plan1 §P1 measured as worth up to 6.6e-4 max rel err | `q_loader.scale_all` |
| `−inf − −inf → NaN` floor on fully-masked rows | `floor_masked_max` |
| GQA/MQA, 3D grid, varlen (cu_seqlens), LSE, paged KV, split-K | `_make_dualwave_swp_traits` |

So the real work is host-side ABI generality, three genuinely new features
(bias, dropout, runtime `Hdim`), generalizing an existing mask from one bound
to two, and a second wave geometry for the upper ladder.

### The reference asm

`~/dockerhome/meff/aiter/hsa/gfx950/fmha_v3_fwd/*.co` — 17 hand-written kernels,
manifest in `fmha_fwd.csv`. Disassemble with:

```bash
~/.venvs/gfx950-7.14/lib/python3.13/site-packages/_rocm_sdk_devel/llvm/bin/llvm-objdump \
    -d --mcpu=gfx950 <kernel>.co
```

Resource metadata, which is the load-bearing part:

| kernel | tile M×N | waves | VGPR/lane | LDS | spill |
|---|---|---|---|---|---|
| `fwd_hd128_bf16{,_causal,_group}` | 256×64 | 8 (512 thr) | 256 | 160 KB | 0 |
| `fwd_hd192_hd128_bf16` | **128×128** | **4 (256 thr)** | **512** | 160 KB | 0 |
| `fwd_hd128_fp8` | 256×128 | 8 | 256 | 160 KB | 0 |
| `fwd_hd256_fp8` | 256×64 | 8 | 256 | 160 KB | 0 |

Three things this settles. `fwd_hd128_bf16` is dualwave's exact geometry, so
dualwave's baseline is the right one. **head_dim > 128 is solved by halving the
wave count to double the per-lane register file**, not by sharding across waves
the way aiw does — that is a second wave family, not a retune of the first. And
`mask` takes only 0 and 2 (no top-left), `mode` 0/1 is dense/group, so the
shipped surface matches AOTriton's `CAUSAL_TYPE {0, 3}` and includes varlen.

**No bias, no dropout, no sliding window exists in any of the 17.** For those
three there is no asm to read and we are on our own, as gfx1201 was.

---

## Decisions taken

| | |
|---|---|
| Compute core | Extend the dualwave schedule. Do not port aiw's structure. |
| Helper placement | **Subclass** the `Dualwave*` classes into `parity/`. `kernels/attention/flash_attn_utils.py` is imported by four production kernels and is not edited. |
| Shared scaffolding | Import `fmha_abi_gfx1201.py` (pure host-side ABI, no arch content) and the arch-neutral names of `fmha_common_gfx1201.py` **unedited**. There is no gfx1201 hardware on this host, so its 298 tests cannot re-validate a refactor. Rename/extract later. |
| head_dim coverage | Full 16..512. All three strategies below exist as compile-time knobs — the optimum is input-dependent. |
| Paged KV | A real feature (it changes the input surface — a `BlockTable`, page-indexed KV — and cannot be expressed through the dense API), but one AOTriton's `attn_fwd` does not have. Carried as a `const_expr` path so gfx950 does not lose it. **Not required to be tested.** |
| Split-K | **An optimization, not a feature** — see the note below. Carried for the same reason: not to close a gap, but so the port does not regress a capability gfx950 already ships. **Not required to be tested.** |
| fp8 | Out of scope. AOTriton `attn_fwd` is bf16/f16; `sdpa-feature-gap.md` puts INT8 out and leaves mxfp8 undesigned. |
| Sequencing | ABI first, then features. |

### Split-K is an optimization, not a feature gap

Worth stating outright, because "gfx1201 doesn't have it" reads like a gap and
is not one.

**The test that separates the two: does enabling it change the answer?** Bias,
dropout and windows all change *what is computed* — turn them off and a
different tensor comes out. Split-K computes exactly the same tensor (up to FP
reduction order) and only changes *how the work is distributed*, at the cost of
an fp32 workspace and a second launch. Same inputs, same outputs, same math.

Confirmed by reading, not assumed: `flash_attn_func_gfx1201_aiw.py` has **no
split-K of any kind**. Every `split` in it is one of three unrelated things —
the KV *region* split (full vs masked tiles, aiw:1250), the *address* split
(64-bit uniform base + 32-bit divergent offset, aiw:915), or prose. Nothing
splits the reduction axis, and neither does AOTriton's `attn_fwd`, which is why
split-K appears nowhere in `sdpa-feature-gap.md`.

What it actually is: gfx950's existing answer to the low-parallelism regime —
decode, where `seq_len_q` is 1 and the KV loop is the only long axis. AOTriton
answers the same problem with `PERSISTENT_DYNAMIC` instead. Two optimizations
for one problem; neither is on anybody's gap list. **So the reason to carry
split-K through the port is not to close a gap, but so the port does not
regress a capability gfx950 already ships.**

The other gfx1201 answers to that same problem are different in kind again, and
these *are* worth porting during P7:

- **Grid axis order.** `q_tile` on the fastest-dispatching x axis spreads
  durations 1..N across every scheduling group instead of giving each group a
  uniform one. Measured 0.587–0.769 against head-fastest, up to 1.7x;
  non-causal is indifferent, which is what identifies the cause (aiw:834-843).
  dualwave launches `(NUM_HEADS_Q, num_q_blocks, grid_z)` — head fastest, i.e.
  the arrangement gfx1201 measured as the slow one. **Re-measure this on
  gfx950; it may be free speed.**
- **`lpt_tile_order`** — longest-processing-time tile ordering, a knob. Default
  on for the gfx1201 forward; the backward ported it and measured it should
  stay off, so it is shape-dependent and belongs in the tuning table.
- **`PERSISTENT_TYPE` / persistent-dynamic** — AOTriton's approach, workgroups
  pulling tiles from a shared counter. Deferred on gfx1201 and never built;
  deferred here too.

Record this so a later reader does not mistake split-K's absence on gfx1201 for
a gap in that kernel, or its presence on gfx950 for a parity requirement.

---

## File map

New, all under `kernels/attention/parity/`:

| file | what |
|---|---|
| `flash_attn_func_gfx950.py` | **the deliverable** — builder, kernel body, `_args`/`_launch`/`_compile` |
| `fmha_dualwave_gfx950.py` | parity subclasses of `Dualwave{KernelContext,SoftmaxHelper,KvGmemToLdsLoader,QLoader,GemmHelper,StoreHelper}` |
| `fmha_common_gfx950.py` | curated re-export of the arch-neutral `fmha_common_gfx1201` names, plus gfx950-specific helpers |
| `fmha_tuning_gfx950.py` | `FmhaInputMetadata` / `FmhaKnobs` for gfx950 + the measured tables |
| `flash_attn_func_gfx950_interface.py` | public API, mirroring `flash_attn_func_gfx1201_interface.py` |
| `gfx950_standalone.py` | `sys.path` shim, mirroring `gfx1201_standalone.py` (which stays untouched) |
| `test_flash_attn_func_gfx950.py` | the ported suite |

Imported, never edited: `fmha_abi_gfx1201.py`, `fmha_common_gfx1201.py`,
`philox.py`, `kernels/attention/flash_attn_utils.py`.

Existing tooling reused as-is: `tooling/perf_ab.py` (interleaved A/B, median of
per-rep ratios), `tooling/isa_stats.py`, `tooling/dump_isa.py`,
`tooling/qkv.py`, `tooling/accuracy_probe.py`.

---

## Phases

Each phase lands a working, tested kernel. Partial coverage is deployable —
a functional we do not yet serve falls back, exactly as plan1 §0 describes.

### P−1 — Build environment

**Blocking prerequisite.** There is no `build-fly/` on this host and neither
python on `PATH` nor `~/.venvs/gfx950-7.14` can import `flydsl` or `torch`. The
8×gfx950 GPUs are visible to `rocm-smi`, so the hardware is here and only the
build is missing.

Run `scripts/build_llvm.sh -j64`, then `scripts/build.sh -j64`, then
`pip install -e .` — or use the ROCm container path in the `build-flydsl`
skill. Gate: `python3 -m pytest tests/kernels/ -k flash_attn` passes, and
`kernels/attention/flash_attn_gfx950.py` reproduces its benchmark number. That
number is the baseline every later phase is measured against, captured once,
here.

### P0 — ABI skeleton, behaviour unchanged

The AOTriton argument list wired to what the kernel already does. No new
features; this phase proves the ABI, the subclass seam and the harness.

- Per-tensor runtime strides — `stride_q0/q1/q2`, `stride_k*`, `stride_v*`,
  `stride_o*`, numerically named per `sdpa-feature-gap.md` §"Special
  Instruction When Porting". `stride_?3` is the contiguous D axis, stays 1.
  Replaces `stride_q_n` / `stride_kv_n`, which derive layout from shape and are
  simply wrong for a BHSD-shaped view over BSHD memory.
- A `strides_constexpr` knob folds them back to literals, so the fast path can
  be shown to emit unchanged code.
- Runtime `sm_scale`, `num_head_q`, `num_head_k`, `max_seqlen_q`, `max_seqlen_k`.
- Optional LSE behind an `L != nullptr` gate; rank-2 fp32, aiw's single
  branch-free offset formula (`fmha.lse_row_addressing`).
- `num_seqlens == 0` (dense) only. `window_left`/`window_right` accepted and
  validated but only `{none, causal}` honored.
- Paged and split-K `const_expr` paths carried through, compile-only.

*Verify.* Numerically identical to `kernels/attention/flash_attn_gfx950.py` on
every shape that kernel serves. `perf_ab.py` interleaved A/B against the P−1
baseline. `isa_stats.py`: with `strides_constexpr=True`, VGPR count, spill
count and LDS must be unchanged.

### P1 — runtime `Hdim`, three strategies behind one knob

`hdim_mode ∈ {"zero_fill", "runtime_qk_steps", "native"}`. The two GEMMs are
not symmetric here and the knob only affects the first:

- **`zero_fill`** (aiw's approach, the default). Loads skip whole 8-element
  chunks past `Hdim` and zero those *registers* — never per-element. The input
  tensor is not padded and never copied. The MFMA count stays tile-width-shaped;
  the zeroed columns contribute exactly 0 to the QK dot product.
- **`runtime_qk_steps`**. Additionally makes the QK reduction's MFMA K-step
  count `ceil(Hdim_qk/16)` at runtime. Legal because the S accumulator shape
  depends on BLOCK_M/BLOCK_N, not on D — nothing in the register layout moves.
  Recovers the wasted MFMA at the low rungs. Cost is an unrolled sequence
  becoming a dynamic-trip loop inside the compute clusters, which is where the
  8-cluster schedule is hand-balanced: **must be measured, not assumed**.
- **`native`**. Reserved for narrow LDS staging geometries below 64. Initially
  implemented for no rung; added only where P7's sweep justifies it.

`Hdim_vo` is masked at the O store and can never be shortened — `v_o` is
`D_CHUNKS` accumulators carried as `scf.for` init args
(`flash_attn_gfx950.py:292`), so the output width is compile-time by
construction. `Hdim_qk != Hdim_vo` falls out of this; oracle is
`fwd_hd192_hd128_bf16.co`.

Host-side obligation, alignment not padding: the D-axis pitch must be a
multiple of 8 elements so the chunk containing `head_dim` lands inside the
allocation. Reject a tightly-packed `(B,H,S,100)` tensor rather than copying
it, as `flash_attn_func_gfx1201_interface.py:172` does, and allocate O at the
padded pitch.

*Verify.* Port `test_padded_head_never_reads_the_pad` (poison the pad with NaN
and 1e4) and `test_padded_head_is_independent_of_pad_contents`. Ladder against
torch SDPA. A/B of the three `hdim_mode` values at every rung — this phase's
second output is that table.

### P2 — the upper ladder and the second wave family

| rung | family | tile M×N | waves | VGPR/lane | notes |
|---|---|---|---|---|---|
| 16, 32, 48 | A | 256×64 | 8 | 256 | via `PADDED_HEAD` in the 64 build |
| 64, 128 | A | 256×64 | 8 | 256 | dualwave today |
| 192, 256 | **B** | 128×128 | **4** | **512** | modelled on `fwd_hd192_hd128_bf16.co` |
| 384, 512 | B | 128×128 | 4 | 512 | V-column windows (`BLOCK_DMODEL_V` + `D_OFFSET`) |

Family B is the substantial item: a 4-wave CTA doubles the per-lane register
file, which changes the LDS tile geometry, the DMA thread mapping and the
stagger/barrier structure. Read `fwd_hd192_hd128_bf16.co` before designing it.

Tuning table in `fmha_tuning_gfx950.py`, keyed on `BLOCK_DMODEL` (plan1 §N3).

*Verify.* **1100+ TFLOPS at every rung below 256.** Correctness across the
ladder. `isa_stats.py` against the reference: 4 waves, 512 VGPR, 0 spill at
head_dim 192.

### P3 — generalized sliding window (gSWA)

The mask is already a relative diagonal parameterized by `delta`, so this adds
a second threshold comparison to `_causal_mask_inplace` rather than replacing
it. Negative `window_left`/`window_right` shift the relevant boundary the
opposite way, which is what makes both causal variants expressible as windows —
`CAUSAL_TYPE` collapses to `{0, 3}` exactly as AOTriton and the reference asm
ship it.

**The schedule is not restructured.** Window-derived tile bounds
`[t_begin, t_end)` replace `loop_lb` / `split_t_end`; the loop stays one loop at
two tiles per iteration and the per-tile `scf.if` gates stay as they are. Only
the bounds move. This is available precisely because dualwave's mask is already
a runtime gate, where gfx1201 had to split the loop body in two.

Host side: `abi.resolve_window`, `fmha.resolve_window`, and the
`WINDOW_TOPLEFT` / `WINDOW_BOTRIGHT` sentinels, all reused unedited.

*Verify.* **Bitwise** against P0/P1 causal for windows that reproduce causal —
plan1 records this as the sharpest oracle in the codebase, and it caught both a
prefetch bug and an interval off-by-one there. Separately, a *measurement* that
leading dead tiles are skipped: a fully dead tile is a bitwise no-op
(`corr = exp2(0) = 1`, `p = 0`), so correctness cannot prove work was avoided.

### P4 — varlen (`VarlenBits`)

The five modes of `sdpa-varlen-plan.md` §2 — one identically-decoded byte per
side plus the LSE layout — reusing `fmha.decode_addressing` and
`fmha.lse_token_pitch` unedited. Prologue-only: at most four scalar loads into
SGPRs, no VGPR cost. Compact LSE in varlen mode, matching FA.

Oracle: `mode=1` group kernels, `fwd_hd128_bf16{,_causal}_group.co`.

### P5 — bias

A third global stream in the compute clusters, where K and V already occupy the
DMA clusters. Each lane needs exactly the S elements it owns, and the MFMA
accumulator's lane→column mapping gives contiguous runs, so these are vector
loads with no LDS staging — the gfx950 analogue of `fmha.acc_elem_column`, whose
docstring explains why the eight elements of a group are eight *contiguous*
columns.

Carries the `m_i` floor regression test. A bias may legitimately be `-inf` over
a whole KV tile, driving a row max to `-inf`; this is the **first configuration
in which the `-inf − -inf → NaN` bug is reachable at all**, which is why the
test belongs here and not earlier. `floor_masked_max` already implements the
fix; this phase is where it becomes verifiable.

Fast-math note from plan1: `ninf` is not affordable once bias exists — it
silently deleted a KV tail mask on gfx1201. Audit `FastMath` / `fp_mode` here.

### P6 — dropout (philox)

`philox.py` reused unedited. The part that is easy to satisfy incorrectly is
the reproducibility contract: **the mask must not depend on the tiling**, which
constrains BLOCK_M/BLOCK_N from this phase onward. Write it down rather than
leaving it implicit in the offset arithmetic.

Pure VALU plus one compare per element — a candidate for the MFMA co-execution
shadow, which is the one place gfx950 is structurally cheaper than gfx1201 for
this feature. The risk is register pressure, not VALU throughput.

Port `test_dropout_mask_gfx1201.py` and `test_philox.py`.

### P7 — tuning sweep and final gates

Also the home for the two load-balance items from the split-K note above:
re-measure dualwave's grid axis order (it currently launches head-fastest,
which gfx1201 measured as the slow arrangement under causal), and port
`lpt_tile_order` as a knob.

Full re-sweep of `fmha_tuning_gfx950.py` after the register budget has settled.
Plan1 records the same lesson three times and it is worth pre-empting: **spill
count is not a proxy for speed** — head_dim 224 there was 27% faster unsharded
*despite* spilling, because quartering the workgroup count outweighed it. Every
entry rejected on spills or LDS earlier must be retried once the budget moves.

---

## Verification standard

Three gates, mirroring plan1 §5:

1. **Correctness** — bitwise against a preserved oracle wherever the FP
   reduction order is unchanged; tolerance against fp32 SDPA where it is not.
   The aiter `.co` kernels are a second numerical oracle for dense, causal and
   group modes at head_dim 128 and 192/128.
2. **Structure** — explicit assertions on wave count, tile shape, barrier count
   and prefetch distance. plan1 §2.2 found two performance bugs that every
   correctness gate passed.
3. **Performance** — `tooling/perf_ab.py`, interleaved A/B, median of per-rep
   *ratios*, reported with VGPR and spill counts. Never two sweeps compared
   after the fact; the board drifts ~5%.

Standing baseline captured at every phase boundary: `B=1 H=8 N=4096` bf16 and
f16, causal and non-causal, head_dim ∈ {16, 32, 48, 64, 80, 96, 128, 160, 192,
224, 256, 384, 512}, both BHSD and BSHD memory layouts via `tooling/qkv.py`.

---

## Risks and accepted debts

| risk | mitigation |
|---|---|
| Paged and split-K are carried but untested, so they rot silently | A compile-only smoke build of both in the test suite each phase, so they at least keep tracing |
| Register pressure: dualwave is already at 256 VGPR / 0 spill at head_dim 128, and every feature adds | `isa_stats.py` at every phase boundary; features are `const_expr`-gated so the shipped fast build carries none of them |
| Bias adds a third global stream to DMA clusters that are already balanced | Measured in P5 in isolation before it composes with dropout |
| No gfx1201 hardware here, so nothing shared can be re-validated | Zero edits to any gfx1201 file — the reason for the import-don't-refactor decision |
| "1100T at every rung below 256" measures something different at head_dim 16, which is softmax- and bandwidth-bound (4 MFMA against 17 `v_exp_f32` and 2 barriers per tile) | Report achieved bandwidth alongside TFLOPS at rungs ≤ 48; treat the TFLOPS target as applying to the MMA-bound rungs |
| `runtime_qk_steps` may disturb the hand-balanced cluster schedule | It is a knob, defaulted off; P1 ships the A/B that decides it per rung |

## Out of scope

fp8/INT8; mxfp8; persistent-dynamic scheduling; `NUM_XCDS`; fused
`RETURN_ENCODED_SOFTMAX`; backward passes.

---

# Outcome: P−1 — DONE

## Environment — no build was needed

The phase as planned called for `build_llvm.sh` + `build.sh`. **Neither was
run.** `flydsl==0.3.1` installs as a wheel with `flydsl._mlir` bundled, so the
whole ~40-minute LLVM build and the `nanobind`/`pybind11`/`cmake` dependency
chain fell away. Installed instead:

```
flydsl 0.3.1 · torch 2.12.0+rocm7.14.0 · amd-torch-device-gfx950 2.12.0+rocm7.14.0
numpy 2.5.2 · pytest 9.1.1 · triton 3.7.1+rocm7.14.0 · rocm-sdk 7.14.0
```

Two traps worth recording, both of which cost time:

- **`torch` alone is not enough.** Without `amd-torch-device-gfx950`, torch
  imports, enumerates all 8 GPUs and reports `gfx950` in
  `torch.cuda.get_arch_list()` — and then fails every kernel launch with
  `hipErrorInvalidImage`, including `tensor.zero_()`. The arch list describes
  the build config, not the code objects actually present.
- **`ROCM_PATH` must be exported** or the JIT dies at link with "lld invocation
  failed":
  ```bash
  export ROCM_PATH=$(rocm-sdk path --root)
  ```

Machine: 8× MI355X (`gfx950:sramecc+:xnack-`), 256 cores, ~3 TB RAM. The
container reaches github.com only — PyPI, files.pythonhosted.org and
download.pytorch.org are all proxy-filtered, so **every package has to be
installed by the user**; nothing can be fetched from inside.

*Gate met:* `pytest tests/kernels/test_flash_attn_fwd.py` → **33 passed** in
49 s. Trace → MLIR → ROCDL → HSACO → launch all work against the 0.3.1 wheel,
with no version skew against the working tree's kernels.

`aiter` is **not** installed, so `test_flash_attn_fwd.py --compare` (FlyDSL vs
aiter_ck vs aiter_asm) is unavailable. The `.co` files remain readable as an
ISA oracle; request `aiter` if the numerical/perf comparison is wanted.

## The baseline shape in this plan was wrong — corrected

`B=1 H=8 N=4096` was inherited from the gfx1201 plan and **does not saturate
gfx950**. At BLOCK_M=256 it dispatches `16 q_blocks × 8 heads × 1 batch = 128`
workgroups onto an MI355X's 256 CUs, so it measures occupancy rather than
kernel quality:

| shape (D=128 bf16 non-causal) | time | TFLOPS |
|---|---|---|
| B=1 H=8 N=4096 | 86.3 µs | **796** |
| B=4 H=8 N=4096 | 238.9 µs | **1151** |
| B=8 H=8 N=4096 | 480.0 µs | 1145 |
| B=2 H=16 N=8192 | 942.1 µs | 1167 |
| B=1 H=32 N=16384 | 3723.9 µs | 1181 |

A 1.45x spread with nothing but occupancy varying. **The standing baseline
becomes `B=4 H=8 N=4096`** (first saturating point); anything at B=1 is an
occupancy measurement and must not be used for phase A/B.

## Dualwave baseline — captured once, B=4 H=8 N=4096

| head_dim | mask | bf16 | fp16 |
|---|---|---|---|
| 64 | non-causal | 878.4 | 844.0 |
| 64 | causal | 571.9 | 554.2 |
| 128 | non-causal | **1152.7** | 1065.9 |
| 128 | causal | 789.6 | 753.3 |

TFLOPS; the harness already halves the FLOP count for causal, so these are
comparable across the mask column.

## Two findings that change the plan

**1. The 1100T target is currently met at exactly one point.** D=128
non-causal bf16 clears it at 1152.7; D=64 (878) and every causal config (789.6
at best) do not. That headroom is *pre-existing* — it has nothing to do with
the parity features, which have not been written yet. So the target is not
"hold the line through the port", it is "port without regressing, then close a
gap that is already open". Worth separating in the reporting so a later phase
is not blamed for a deficit it inherited.

**2. Causal runs at 68% of non-causal efficiency** (789.6 / 1152.7 at D=128;
571.9 / 878.4 = 65% at D=64), on FLOP counts that already account for the
halved work.

**The grid axis-order item is retracted — it does not transfer from gfx1201.**
I promoted it to P0 on the gfx1201 measurement and was wrong; reading
`_init_dualwave_grid_indices` (`flash_attn_utils.py:918-940`) settles it
against me twice over:

- gfx950 has **already measured the opposite sign**. The comment there reads:
  "Non-causal only: under a causal mask q-block i does work proportional to i,
  so making q_block the fast axis clusters unequal work and costs 7%
  (measured)." On gfx1201 that same change *won* by up to 1.7x. The machines
  differ in what dominates — MI355X has 8 XCDs, so the grid order is primarily
  an L2-locality lever (keep one head's q-blocks on one XCD and it re-streams
  K/V less), while single-die RDNA4 has no such structure and the order is
  purely about spreading unequal durations.
- The lever that *does* exist here, `XCD_SWIZZLE`, is already built and already
  wired into the **fp8** kernel (`flash_attn_fp8_gfx950.py:609-638`), but is
  never passed by the bf16 builder, so it is dead code on this path. Its
  dispatch gate is `num_q_blocks >= MIN_Q_BLOCKS_XCD_SWIZZLE` (64), i.e.
  seq_len >= 16384 at BLOCK_M=256 — **it would not engage at the baseline shape
  at all** (16 q-blocks).

So causal's 68% has a different cause on gfx950 and grid order is not it.
Leave the investigation in P7 where it started, scoped to long sequences, and
treat the causal deficit as pre-existing headroom rather than a P0 item.

*Lesson worth keeping:* a measured result from the gfx1201 plan is evidence
about gfx1201, not about gfx950. This plan quotes several of them; each needs
re-measuring here before it is acted on.

## Reproducing

```bash
export ROCM_PATH=$(rocm-sdk path --root)
cd /home/xinyazha/dockerhome/meff/FlyDSL
HIP_VISIBLE_DEVICES=0 python3 tests/kernels/test_flash_attn_fwd.py \
    --batch 4 --num_heads 8 --seq_len 4096 --head_dim 128 \
    --dtype bf16 --no-causal --warmup 20 --iters 50
```

---

# Outcome: P0 + P1 — DONE

Six new files, **no existing file modified** — `git status` shows only
additions, so the "import, never edit" constraint held end to end:

| file | lines | what |
|---|---|---|
| `flash_attn_func_gfx950.py` | ~740 | the kernel: ABI, pipeline body, launcher |
| `fmha_dualwave_gfx950.py` | ~300 | the five `Parity*` subclasses |
| `fmha_tuning_gfx950.py` | ~200 | ladder, `FmhaInputMetadata`/`FmhaKnobs`, `resolve_knobs` |
| `test_flash_attn_func_gfx950.py` | ~290 | 76 tests |
| `gfx950_standalone.py` | ~60 | `sys.path` shim |

*Gates.* 76/76 parity tests pass; 33/33 production tests still pass.

## P0: bit-exact, and free

An unpadded parity build is **bit-identical to `flash_attn_gfx950.py`** on all
four (causal × head_dim) combinations, which is the sharpest available gate --
it would fail on a reordering no tolerance would catch. Throughput at
B=4 H=8 N=4096, parity ÷ production: **0.999 / 1.016 / 0.999 / 1.011**. The
generalization is free because it is a change of variables, not new work: both
batch and head are workgroup-uniform, so they fold into the buffer descriptor's
base and the per-access arithmetic is unchanged.

Working now that production cannot express: arbitrary and **mixed** memory
layouts (Q as BHSD with K as BSHD), GQA/MQA at runtime head counts, runtime
`sm_scale`, per-tensor K and V strides.

## P1: the ladder, and one real bug

head_dim 16..128 all serve correctly through the {64, 128} ladder with
`PADDED_HEAD`, as does `hdim_qk != hdim_vo`.

**Masking Q alone is not enough, and the test that says so is the poison
test.** `QK^T = Σ Q[d]·K[d]`, so zeroing Q annihilates whatever K holds at the
padded columns -- for any *finite* pad. A pad holding NaN or Inf gives
`0 · NaN = NaN`, and a caller's D-axis padding is allocation slack whose
contents nothing constrains. Masking Q is nearly free (prologue, once) and
masking K is not (once per KV tile), which is exactly why it is tempting to
skip, and why the poison values are parametrized separately: with K unmasked,
`nan`/`inf` fail while `zero`/`big` pass.

*A near-miss worth recording:* the first poison test used head_dim 40 and
passed. It was vacuous -- `(40+7)//8*8 == 40`, so that tensor has no pad to
poison. head_dim 113 has a 7-element pad and failed immediately. **A padded-head
test whose head_dim is already 8-aligned tests nothing.**

The K mask needed the D-column of each register, which is not obvious because K
transits LDS in a swizzled layout. Derived rather than probed: matching the DMA
write against the register read term by term gives `w = lm % 8`, `d = ks // 4`,
`l = (lm // 8)·8 + ld + (ks % 4)·2`, and since `(lm // 8)·8` vanishes mod 8,

    D = (ld + (ks % 4)·2)·8 + (ks // 4)·64 + i  =  ks·16 + ld·8 + i

which is `_q_pack_col` exactly. **The swizzle permutes tokens across LDS lines
and leaves D in linear order**, so K masks with the same expression as Q. V
needs nothing -- its D columns are O's columns, suppressed at the store.

## Departures from the plan

- **The pipeline body is copied, not subclassed.** The plan said "subclass into
  parity/", and the *helpers* do subclass cleanly (five small overrides, none
  re-implementing a loop). The body could not: it is inline in
  `build_flash_attn_dualwave_swp_module`, and windows/bias/dropout each rewrite
  parts of it. A copy that diverges on purpose beats an import special-cased at
  six points.
- **LSE is still a build knob**, not the `L != nullptr` runtime gate the plan
  specified. `return_lse` gates code emission as it does in production; the
  null-pointer check is one `scf.if` in the epilogue and is outstanding.
- **`hdim_mode="runtime_qk_steps"` is plumbed but not implemented.** The knob
  resolves and is carried into the build; only `zero_fill` has a code path.
  Its A/B was to be P1's second output and is not yet measured.
- **LSE uses `traits.NUM_HEADS_Q`**, a constexpr, where the rest of the kernel
  now reads head counts at runtime. Consistent today because the builder and
  launcher take the same value, but it is a latent AOT hazard.

## Method note

`test_runtime_sm_scale[0.5]` failed on first run against an absolute
tolerance. It was not a bug: raising `sm_scale` sharpens the softmax and grows
`|O|` with it (0.24 at scale 0.05, 4.32 at 0.5), and **the production kernel
shows the same error to within 6% at every scale measured** -- which is what
identified it as bf16 output precision rather than anything the port did. The
fix was the metric, not the tolerance: normalise by `|ref|` when it exceeds 1,
which leaves every O(1) case bounded exactly as before.
