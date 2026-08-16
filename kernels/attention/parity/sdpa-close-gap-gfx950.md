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

### P2 evidence — family B is required, and now measured

Family B was scoped from the reference asm's resource metadata, which is
evidence about AMD's kernel, not ours. Measured directly by extending the
ladder to 192/256 on the existing 8-wave geometry and reading
`21_final_isa.s`:

| head_dim | VGPR | spills | LDS | correct? |
|---|---|---|---|---|
| 64 | 164 | 0 | 34 KB | yes |
| 128 | **248** | **0** | 68 KB | yes |
| 192 | 256 | **506** | 102 KB | no — NaN |
| 256 | 256 | **963** | 136 KB | no — NaN |

**The 8-wave geometry is saturated at head_dim 128** -- 248 of 256 VGPRs with
zero spills, i.e. the next rung has nowhere to go. LDS is *not* the binding
constraint (102 KB and 136 KB both fit the 160 KB budget); the register file
is. That is exactly why `fwd_hd192_hd128_bf16.co` runs 4 waves at 512
VGPRs/lane: halving the wave count doubles the per-lane register file, which is
the ~1.5-2x that 192 needs. **The asm's choice is now corroborated on our own
kernel rather than assumed from theirs.**

The NaN is a consequence, not a separate bug. It is non-deterministic at 506
spills and becomes deterministic when the DMAs are fully drained, and it is
*not* an addressing error: both the K and V LDS mappings were re-derived
against the DMA writes and cover the right tokens and D columns for any
`SMEM_D_RPT` (the K derivation is reproduced in `ParityKvLdsToVgprLoader`).
The most likely mechanism is `_anchor_v_o`, whose inline asm ties `D_CHUNKS`
`vector<16xf32>` operands -- 96 VGPRs at 192, 128 at 256 -- as simultaneously
resident, which is not satisfiable alongside 506 spills.

**One fix from this landed and is kept**: `DualwaveQLoader.load_all` assembles
Q through a fixed 8 -> 16 -> 32 tree and then takes at most *two* 32-packs, so
it silently serves only `K_STEPS_QK` 4 and 8 -- head_dim 64 and 128.
`ParityQLoader.load_all` replaces it with a left fold, which family B needs
regardless of wave count.

### P2 evidence — the granule was never 64 by necessity, and it costs 2x

`HEAD_DIM_GRANULE = 64` was inherited from the production kernel and treated as
a property of the algorithm. It is not: it is how many D elements one DMA issue
covers, and a wave moves a fixed 512 bf16 elements per issue (64 lanes x 8), so

    tokens per issue = 512 / granule
    lines            = BLOCK_N / tokens_per_issue
    issues per wave  = lines / NUM_WAVES        (positive integer)

Granule 64 with BLOCK_N 64 is one point in that space, not the only one.

**What it costs today.** Padding a small head into the 64-wide tile is
expensive rather than cheap, which settles the open question the plan carried
from P1. At B=4 H=8 N=4096 bf16 non-causal:

| head_dim | tile | µs | real TFLOPS | padded-equivalent |
|---|---|---|---|---|
| 16 | 64 | 203.2 | **169** | 676 |
| 32 | 64 | 205.2 | **335** | 670 |
| 48 | 64 | 215.2 | 479 | 639 |
| 64 | 64 | 171.6 | 801 | 801 |
| 128 | 128 | 253.6 | 1084 | 1084 |

head_dim 16, 32 and 48 all take *the same time as each other* and **more than
head_dim 64**, because they do the full 64-wide MFMA and pay the padded-head
masking on top. So these shapes are MFMA-bound, not bandwidth-bound -- the
opposite of what the plan assumed when it wrote that the target "does not
apply" to them.

**Granule 32 is viable; granule 16 is not.** Enumerating the space against the
rule above, the clean small-head family keeps everything about family A except
two numbers:

| family | tile width | waves | BLOCK_M | BLOCK_N | granule | issues/wave |
|---|---|---|---|---|---|---|
| S | <= 32 | 8 | 256 | **128** | **32** | 1 |
| A | <= 128 | 8 | 256 | 64 | 64 | 1 |
| B | > 128 | 4 | 128 | 128 | 64 | 4 |

BLOCK_N has to double because halving the granule doubles the tokens one issue
covers. Independently, gfx1201's tuning table reached BLOCK_N 128 for small
head_dim by measurement, on the grounds that a wider KV tile amortises the
per-tile softmax cost -- two architectures, same conclusion.

**Granule 16 is blocked by the instruction, not the staging.** The staging is
perfectly regular at 16 (BLOCK_N 256 over 8 waves is one issue per wave), but
the PV MFMA is `v_mfma_f32_32x32x16` and its output is 32 D columns wide, so
`D_CHUNKS = head_dim/32` would fall below 1. Serving head_dim 16 natively needs
`v_mfma_f32_16x16x16`, whose accumulator is v4f32 rather than v16f32 -- a
different register layout through every helper, i.e. a fourth family. Out of
scope. head_dim <= 16 therefore stays padded, but into a 32-wide tile: **2x
waste instead of 4x**.

Expected gain: head_dim 32 roughly doubles (335 -> ~670 real TFLOPS), head_dim
16 likewise (169 -> ~340).

**S and B share one blocker**, which is why they are one phase. Both need
`_make_dualwave_swp_traits` replaced by a parity-side constructor -- B because
it hardcodes 8/256/64, S because it derives the granule from a fixed 128-byte
row -- and both need BLOCK_N != 64, which doubles the score accumulators. One
piece of work unlocks both.

*Landed now:* the selection. `_with_wave_geometry` picks the family including
the granule, `staging_shape()` states the coherence rule a new family must
satisfy (rather than a table of expected numbers), and `_with_traits` refuses
any geometry it cannot actually build. `32` is deliberately **not** in
`LADDER_PLANNED`: that list is consulted only after `LADDER` misses, so it
would never be reached, while putting it in `LADDER` would route head_dim <= 32
to a tile that does not exist and break a path that works today, slowly.

### P2 evidence — the baseline shape was still wrong, and the knob sweep

**S=4096 saturates the grid but not the pipeline.** The P−1 outcome corrected
the baseline from B=1 to B=4 on occupancy grounds and stopped there. That was
not far enough: the dualwave epilogue drains three KV tiles, so at S=4096 there
are only 16 tiles per q-block and the drain is a large fraction of the work.

| shape (D=64 bf16 non-causal) | workgroups | vs 256 CUs | TFLOPS |
|---|---|---|---|
| B=4 H=8 S=4096 | 512 | 2.0x | 769 |
| B=2 H=8 S=8192 | 512 | 2.0x | 864 |
| B=2 H=16 S=8192 | 1024 | 4.0x | 919 |
| B=1 H=16 S=16384 | 1024 | 4.0x | **936** |
| B=2 H=16 S=16384 | 2048 | 8.0x | **937** |

Rows 1 and 2 have *identical* grids and differ by 12% on sequence length
alone, which is what identifies the pipeline rather than occupancy. Flat from
4x to 8x CUs. **The standing baseline becomes B=1 H=16 S=16384**; 4096 is a
tile-count measurement wearing an occupancy measurement's clothes.

**Knob sweep**, at that shape, bf16 non-causal, real TFLOPS with the ratio
against default:

| hdim | default | wpe=1 | wpe=4 | no-stagger | no-setprio | no-lazy |
|---|---|---|---|---|---|---|
| 16 | 197.8 | 0.996 | **0.277** | 0.909 | 1.014 | 0.890 |
| 32 | 386.7 | 1.002 | **0.282** | 0.916 | 1.006 | 0.889 |
| 48 | 563.1 | 1.002 | **0.289** | 0.922 | 1.015 | 0.895 |
| 64 | 934.2 | 1.002 | **0.213** | 0.895 | 0.984 | 0.849 |
| 80 | 506.8 | 1.001 | **0.097** | 0.879 | 1.020 | 0.859 |
| 96 | 601.3 | 0.998 | **0.096** | 0.888 | 1.011 | 0.866 |
| 128 | 1158.2 | 1.004 | **0.062** | 0.894 | 1.005 | 0.876 |

- **`waves_per_eu=4` is catastrophic** -- down to 0.06x at head_dim 128, i.e.
  16x slower. It caps the register budget the schedule is built around. Never.
- **`waves_per_eu=1` is neutral** everywhere (0.996-1.004). The kernel's own
  VGPR use already fixes occupancy, so this knob has nothing to say; 2 is kept
  only because it matches the production build.
- **`stagger` is worth ~10%** and **`lazy_rescale` ~12%**, consistently across
  the ladder. Both defaults confirmed.
- **`setprio` is a wash**, and marginally *negative* at six of seven rungs
  (up to 1.020 without it). **Not acted on:** the effect is 1-2% and this sweep
  is not interleaved, while the gfx1201 record puts board drift at ~5%. It is a
  candidate for an interleaved A/B, not a conclusion.

**The dominant lever is the ladder, not any of these knobs.** Every rung that
is not a compiled tile width runs at roughly its padding ratio: head_dim 16 is
0.21x of 64, and head_dim 80 is 0.44x of 128. No schedule knob moves more than
12%.

**This raises the value of granule 32 beyond small heads.** At granule 64 the
rungs can only be multiples of 64, so 80 and 96 both pad into 128. At granule
32 the ladder becomes 32/64/96/128, which makes 96 native and pads 80 into 96
instead of 128 -- from a 1.6x padding ratio to 1.2x. LDS at granule 32,
BLOCK_N 128, head_dim 96 is 102 KB, inside the budget. So family S is not a
small-head special case; it fixes head_dim 65-96 as well, where the measured
loss is 0.44x.

### P2 progress — the DMA blocker is gone; a V-read bug at D_CHUNKS > 4 is not

**A model, not a guess.** The LDS write/read mapping is now modelled offline
(`tooling/lds_model.py`): it builds the write map a geometry produces and
checks that the K read recovers the right `(token, D)` exactly once. It
reproduces family A -- the known-good production geometry -- and that is what
makes its verdicts on the others worth anything.

It immediately killed the family table this plan had written down:

| candidate | verdict |
|---|---|
| A: granule 64, BLOCK_N 64, 8 waves | **OK** |
| S: granule 32, BLOCK_N **128**, 8 waves | **fails** -- "never read: token 32" |
| S': granule 32, BLOCK_N **64**, 4 waves | **OK** (head_dim 32 and 96) |
| B: granule 64, BLOCK_N **128**, 4 waves | **fails** |
| B': granule 64, BLOCK_N **64**, 4 waves | **OK** (2 issues/wave) |

**BLOCK_N must stay 64.** A wave's K read covers exactly 64 tokens -- 32 lanes
by lo/hi packs -- so BLOCK_N 128 needs the doubled score accumulator. Dropping
to 4 waves reaches the same families with BLOCK_N 64 instead, which means
**no softmax helper changes at all**. That is a much smaller job than this plan
scoped, and it came from modelling rather than from building.

Corrected family table (BLOCK_N 64 throughout):

| family | tile width | waves | BLOCK_M | BLOCK_N | granule | issues/wave |
|---|---|---|---|---|---|---|
| S | off the 64 grid (32, 96, 160) | 4 | 128 | 64 | 32 | 1 |
| A | multiple of 64, <= 128 | 8 | 256 | 64 | 64 | 1 |
| B | multiple of 64, > 128 | 4 | 128 | 64 | 64 | 2 |

**Landed and verified: the DMA generalisation.** The production formula places
one tile line per wave per d-band, correct only when `SMEM_N_RPT == NUM_WAVES`.
`_ParityKvStaging` splits the flat DMA index into `(band, issue)`, and both
the line and the token collapse to the production form at one issue per wave.
Two independent confirmations:

- **The 4-wave geometry is correct at head_dim 128** (error 2.7e-3), where
  family A also works. So the wave-count change and the DMA rewrite are sound
  in isolation.
- **head_dim 192 at 4 waves uses 434 VGPRs with zero spills**, against 256 and
  506 spills at 8 waves. The register hypothesis this phase was built on is
  now confirmed on our own kernel, and it matches
  `fwd_hd192_hd128_bf16.co`'s 4-wave/512-VGPR shape.

**Still failing: head_dim > 128.** With `V` all ones -- where every output must
be exactly 1.0 regardless of the softmax -- the correct D columns at head_dim
192 are exactly `[32, 128)`, i.e. chunks 1..3 of 6; chunk 0 and chunks 4..5 are
wrong. At head_dim 256 no column is correct. Ruled out along the way:

- not the register pressure (zero spills),
- not the wave count (4 waves is correct at head_dim 128),
- not the DMA line coverage (every line is written; the model agrees),
- not a `s_waitcnt` race (forcing a full DMA drain does not change it, and the
  failure is deterministic).

**The V read is not it either -- measured, and it eliminated the leading
hypothesis.** Probed by setting `V[t][d] = d` (bf16 is exact to 256) under
causal, where `q_row 0` attends to a single key, so `P` is a delta and
`O[0][d]` *is* the V column the read fetched. At head_dim 192 the map is
perfect: **0 of 192 columns wrong**. So `_swizzled_v_dc_off` and the three
UNVERIFIED `V_LDS_TO_REG_*` strides are cleared.

**Two lessons about the probes themselves**, both of which cost time:

- **`V = ones` tests almost nothing.** With every V element 1.0, `O = sum_j
  P[i][j] = 1` for *any* softmax that normalises, so the probe is blind to
  wrong scores, wrong tokens, and any column permutation. The "columns 32..127
  are good" reading it produced was an artifact, and it pointed the
  investigation at the V read for two rounds.
- **A delta-`P` probe only inspects the row it isolates.** The perfect column
  map above is row 0. Re-running with `V[t][d] = d` at *every* token -- where
  `O[i][d] == d` must hold for every row -- shows **no row is correct**. The
  fault is in rows, and every probe up to that point had been reading columns.

Eliminated so far, each by measurement: register pressure (zero spills at 434
VGPRs), wave count (the 4-wave geometry is correct at head_dim 128), DMA line
coverage (model-verified, and every line is written), `s_waitcnt` races (a full
drain changes nothing; the failure is deterministic), the V LDS read (exact
column map), and all three schedule knobs (`stagger`, `lazy_rescale`,
`setprio` off individually and together).

What is left is whatever is specific to `D_CHUNKS = 6`, `K_STEPS_QK = 12`,
`SMEM_D_RPT = 3` in the multi-tile path. **The most concrete suspect is
`ParityQLoader.load_all`**: its left fold replaced a tree that only ever
produced 4 or 8 packs, so head_dim 192 is the *first configuration that
exercises it*, and the bitwise-vs-production oracle cannot reach it -- that
oracle only exists at head_dim 64 and 128, where the fold and the tree agree.
`scale_all` is the second: it materialises the whole of Q in f32, 96 VGPRs at
head_dim 192 against 64 at 128.

### P2 — the generated asm is not the problem

Checked against the CDNA4 ISA document (`amd-instinct-cdna4-instruction-set-architecture.pdf`,
§9.1.9 *Memory Buffer Load to LDS*, §11.4 *MFMA Transpose Load from LDS*) and
against the emitted `21_final_isa.s`. Three assumptions are now confirmed
rather than believed:

- **The LDS DMA address model is exactly right.** §9.1.9 gives
  `LDS_ADDR = LDSbase + M0 + inst_offset + TIDinWave * 4`, modified to
  `TIDinWave * 16` for 3- or 4-dword loads. So one `buffer_load_dwordx4 ... lds`
  writes 64 lanes x 16 B = 1024 B starting at M0 -- which is the
  `SMEM_LINEAR_WAVE = 512` element line the whole staging model is built on.
- **M0 is correctly sequenced.** 72 `s_mov_b32 m0` against 72
  `buffer_load ... lds` at head_dim 192, each immediately before its load. The
  hypothesis that several in-flight DMAs might share a stale M0 -- which would
  have explained everything -- is dead.
- **The instruction mix scales exactly as it should:**

  | | D=128 | D=192 | ratio |
  |---|---|---|---|
  | `v_mfma_*_32x32x16` | 192 | 288 | 1.5x |
  | `ds_read_b64_tr_b16` | 192 | 288 | 1.5x |
  | `buffer_load_dwordx4 ...lds` | 24 | 72 | 3.0x |
  | Q loads / O stores | 8 | 12 | 1.5x |
  | `s_barrier` | 25 | 25 | 1.0x |
  | `v_exp_f32` | 197 | 197 | 1.0x |
  | `scratch_*` | 0 | 0 | -- |

  1.5x for everything scaling with `K_STEPS_QK` (8->12) or `D_CHUNKS` (4->6),
  3x for the DMAs (`NUM_DMA` 2->6), 1x for the softmax and the barrier
  structure, and no spills. The instruction stream is the right shape.

**So this is not a codegen or ISA-usage bug.** A derived constant is producing
a wrong value inside a correctly-shaped kernel, which is why every structural
check has come back clean.

*One thing to pin before family B is called done.* The ISA document is
self-inconsistent about the M0 LDS offset width: §9.1.9's text says
`M0[17:0]` (18-bit, 256 KB) and its figure says `M0[15:0]` (16-bit, 64 KB).
The operative width is at least 17 bits, since head_dim 128 already addresses
up to 67 KB and works. But **head_dim 256 addresses up to 135 KB**, which
exceeds a 17-bit field -- so if the true width is 17 rather than 18 bits, 256
has a second, independent failure that fixing 192 would not touch.

### P2 — truncated kernels: every input to the GEMMs is correct at 192 and 256

`tooling/probe_kv_staging.py` builds **truncated kernels** -- the first one or
two pipeline steps and nothing else -- and writes the registers straight to
memory. Inferring a stage's correctness from the final O could not distinguish
"this stage is wrong" from "this stage is fine and a later one is wrong"; these
check each stage against the contract the next one relies on.

Result, at head_dim 192 **and 256**, against the working 128 as a control:

| probe | 128 | 192 | 256 |
|---|---|---|---|
| K stage, D map | 0 wrong | 0 | 0 |
| K stage, token map | 0 | 0 | 0 |
| V stage, D map | 0 | 0 | 0 |
| V stage, token map | 0 | 0 | 0 |
| Q load, D map | 0 | 0 | 0 |
| Q load, row map | 0 | 0 | 0 |
| QK GEMM reduction depth | exact | exact | exact |

The QK probe is the sharpest: with `Q = K = 1`, `S` must equal
`head_dim * sm_scale * log2(e)`, so its **value counts how many MFMA K-steps
actually accumulated**. Measured 64.1 / 128.2 / 192.3 / 256.5 against 64 / 128 /
192 / 256, with zero spread across lanes. The 12- and 16-step reductions are
whole.

**So every input to the GEMMs is correct, and the QK GEMM is correct.** What is
left is the softmax, the PV GEMM, the O accumulator and the store.

Two corrections the probes forced, both mine rather than the kernel's:

- **The V pack layout is not eight consecutive tokens.** A pack is two
  `ds_read_b64_tr_b16` results concatenated, the second read one granule away,
  so a lane's eight elements are **two groups of four, eight tokens apart**:
  lane 0 holds tokens {0,1,2,3, 8,9,10,11} and lane 32 holds
  {4,5,6,7, 12,13,14,15}, which together tile the step's 16 tokens once. The
  probe reported a uniform failure at *every* head_dim including the working
  ones, which is what identified the error as the expectation rather than the
  kernel.
- **A probe that fails on a known-good configuration is testing itself.** Both
  the V and Q "failures" were of this kind -- one a wrong layout formula, one a
  stale buffer size. Running every probe against head_dim 128 as a control is
  what kept them from being reported as findings.

The harness takes a `which=` selector and bypasses `LADDER` deliberately, so it
can build the rungs the ladder keeps unreachable -- a probe that cannot build
the broken thing is useless.

### gfx1201's safe softmax adopted — landed, and not the head_dim>128 bug

`ParitySoftmaxHelper.reduce_max` now seeds the reduction at `-3.0e38` instead
of `-inf`. This is the correction `sdpa-feature-gap.md` asks for
(`m_i = tl.full([BLOCK_M], -3.40282e+38)`, "do not use -inf"): a row whose
scores are all `-inf` gives `m_i = -inf`, and `exp2(-inf - -inf)` is NaN rather
than 0.

Two choices worth recording. It is **unconditional**, where the production
kernel floors the max only under `CAUSAL` -- seeding costs nothing, it swaps
one constant for another, and it removes the case for every masking mode at
once, including the windows and bias still to come. And it **seeds** rather
than flooring the result: a floor applied afterwards still lets an all-`-inf`
tile reach the subtract, whereas seeding means no lane ever holds `-inf` as a
max.

Safe by construction for existing behaviour, since `max(-3e38, x) == max(-inf,
x)` for any finite `x` -- confirmed by the bitwise-vs-production oracle still
passing.

**It does not fix head_dim 192/256**, which remain NaN. Worth having anyway.

### P2 — the PV probe did not discriminate

Added a `which="pv"` mode that runs prologue softmax plus one full `P*V` on
verified-correct operands and dumps `v_o`. With `Q = K = V = 1` the accumulator
should be `4 steps x 16 K = 64` exactly. Measured medians: 64.0 at head_dim 64,
**62.0 at 128**, 60.0 at 192, 62.0 at 256 -- uniform across every `dc` in each
case.

**head_dim 128 is a working configuration and also misses 64**, so the expected
value is wrong again rather than the kernel (`P` is not exactly 1.0 after
`cast_p`). The probe does not separate the cases and is left in place but not
relied on. That is the third expectation error in this investigation; the
control column is doing all the work.

### P2 — every stage is correct in isolation; the composition is not

The probe now covers every stage of the pipeline. At head_dim 192 **and 256**,
with the working 128 as a control throughout:

| stage | probe | 128 | 192 | 256 |
|---|---|---|---|---|
| Q load | D map, row map | clean | clean | clean |
| K stage, buffer 0 | D map, token map | clean | clean | clean |
| K stage, **buffer 1** | D map | clean | clean | clean |
| V stage, buffer 0 | D map, token map | clean | clean | clean |
| V stage, **buffer 1** | D map, token map | clean | clean | clean |
| QK GEMM | reduction depth | exact | exact | exact |
| softmax | `m_row`, `l_row` | exact | exact | exact |
| O store | (row, D) mapping | clean | clean | clean |

`m_row` matches the QK probe's value to the last digit with zero spread across
lanes, and `l_row` is exactly 64.00 -- the 64-token tile summed as `exp2(0) = 1`.
The O store was checked by setting `v_o[dc] := dc + 1` and requiring every
output column to come back as `D // 32 + 1`.

Buffer 1 was a real gap: every earlier probe staged into buffer 0 while the
pipeline alternates, so a wrong buffer-1 base would have been invisible until
the whole kernel ran. It is clean.

**So the fault is in the composition, not in any stage.** And it is not the
loop trip count either -- head_dim 192 fails identically at every sequence
length from 256 to 1024, i.e. from one loop iteration to seven, while head_dim
128 on the *same 4-wave geometry* is correct at all of them.

What that leaves, given head_dim 128 at 4 waves works and the only remaining
differences are `SMEM_D_RPT` 2 -> 3, `D_CHUNKS` 4 -> 6, `NUM_DMA` 4 -> 6 and
the LDS footprint:

- the interleaving of the two LDS buffers across the 8 pipeline clusters, where
  a K DMA for tile *j+1* is in flight while V for tile *j-1* is being read --
  the probes stage one tile and read it back, so they never exercise an
  overlap;
- something that only appears at the full register and LDS pressure of the
  assembled body, which the single-stage probes do not reproduce.

Both are properties of the pipeline rather than of a stage, which is consistent
with every stage passing.

Synchronisation is ruled out too: replacing every `_dualwave_sync_barrier` with
a full `s_waitcnt(0)` plus `s_barrier` -- strictly stronger than the
hand-tuned sync the pipeline relies on -- leaves head_dim 192 and 256 exactly
as wrong. The failure is deterministic and is not an ordering problem.

### P2 — the decoupling experiment cannot be run, and that decides the design

`SMEM_D_RPT` and `D_CHUNKS` are the two candidates left, and at granule 64 they
are **not independent**: `D_CHUNKS = head_dim/32` and `SMEM_D_RPT = head_dim/64`,
so `D_CHUNKS == 2 * SMEM_D_RPT` identically. Granule 32 would separate them --
head_dim 128 at granule 32 is `SMEM_D_RPT 4` with `D_CHUNKS 4` -- but granule 32
is itself broken, and in its *simplest* configuration:

| head_dim | granule | d_rpt | D_CHUNKS | result |
|---|---|---|---|---|
| 128 | 64 | 2 | 4 | 2.8e-3, correct |
| 128 | 32 | 4 | 4 | 0.90, wrong |
| 192 | 64 | 3 | 6 | NaN |
| **64** | **32** | **2** | **2** | **0.735, wrong** |

head_dim 64 at granule 32 has `d_rpt 2` and `D_CHUNKS 2` -- the same depths as
the working granule-64 build -- and is still wrong, which confirms the
`_check_helpers_support_geometry` guard was right to refuse it: the `% 8`,
`// 8` and `// 4` constants in `_k_lds_read_base_per_lane` and
`_swizzled_ks_offset` are `SMEM_N_RPT` and `granule // K_STEP_QK` at family A's
numbers and were never generalised. So granule 32 cannot serve as the control.

**This is what settles the architecture.** The way to hold `D_CHUNKS` at 4
while `SMEM_D_RPT` grows is a **V/O column window** -- computing O in
128-column slices while QK reduces over the full width -- and that is not a
diagnostic trick, it is what both references already do:

- `fwd_hd192_hd128_bf16.co`, the aiter asm for head_dim 192, is **hdim_qk 192
  with hdim_vo 128**. It never builds six O accumulators.
- aiw serves head_dim 384 and 512 the same way, through `BLOCK_DMODEL_V` and
  `D_OFFSET`, repeating GEMM1 per window.
- gfx1201's new head_dim 512 support (`fceea5a9`, `883adf03`) reaches the same
  place from the other side: *sharding* the contraction across waves so
  per-wave state falls to `0.75 * head_dim / C`, rather than scaling the tile.

All three keep per-wave accumulator state bounded instead of letting it grow
with head_dim. This port has been doing the opposite, and head_dim 192 is where
that stops working.

So the V/O column window moves from "a feature aiw has" to **the next piece of
work**, and it is dual-purpose: it is a required parity feature
(`hdim_qk != hdim_vo` is in the ABI), it is how the reference implementations
serve large head_dim, and it is the only way to get a `D_CHUNKS = 4` build at
`SMEM_D_RPT = 3` and so finally separate the two candidates.

`LADDER` stays `(64, 128)`: 192/256 must not be reachable while this is open.

**What family B still has to change**, none of it started:

- `_make_dualwave_swp_traits` hardcodes `num_waves=8`, `block_m=256`,
  `block_n=64`. It is in the production file, so parity needs its own traits
  constructor rather than an argument.
- `BLOCK_N` 64 -> 128 doubles the score accumulators (`v_s` is a pair of
  `v16f32` at BLOCK_N=64), which reaches every `DualwaveSoftmaxHelper` method
  and the causal mask's `_causal_pair_thresholds`.
- `BLOCK_M` 256 -> 128 with 4 waves keeps `rows_per_wave` at 32, so the MFMA
  row mapping survives -- the one piece that does.
- The stagger/barrier structure splits 8 waves into two groups of 4; at 4
  waves that is two groups of 2, and `stagger_i32 = wave_id_uni / 4` is wrong.

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

## Owed before the PR

Not a phase -- no new capability -- but a hard gate on filing. Ordering is
free; correctness of the result is not.

### R1 — Knobs and traits are one thing; make them one class — **DONE**

*Outcome at the end of this section.*


**The observation.** `FmhaKnobs` and the dualwave traits returned by
`make_parity_traits` both answer "how to compute it", at two points on the same
axis: knobs are the *partially resolved* form (`None` means "policy decides")
and traits are the *fully resolved* one. Keeping them as separate types with a
free function between them means every new feature knob has to be declared
twice and threaded through a converter, and it is why `varlen` and
`cross_seqlen` currently travel as keyword-only arguments beside the knob
object rather than inside it.

**The shape to reach.**

```python
knobs = fmha_knobs(arch, **kwargs)   # factory -> arch-specific Knobs subclass
cfg   = knobs.resolve(meta)          # fully-resolved, arch-specific config
```

`resolve` becomes a **method on the arch-specific knob class** and subsumes
both `resolve_knobs` and `make_parity_traits`: it takes the caller's
`FmhaInputMetadata` and returns the configuration the builder consumes. The
split that survives is the one that was always right -- `FmhaInputMetadata` is
*what to compute* and stays arch-neutral and shared; `Knobs` is *how*, and is
arch-specific behind a common base.

**What this deletes**, all of it currently in `fmha_tuning_gfx950.py`,
`fmha_dualwave_gfx950.py` and `flash_attn_func_gfx950.py`:

- the free function `resolve_knobs(meta, overrides)` -> `Knobs.resolve(meta)`
- `make_parity_traits(meta, knobs, *, varlen, cross_seqlen)` -> folded into
  `Gfx950Knobs.resolve`
- `FmhaPlan` and `plan()` -- a resolved knob object *is* the plan
- **`cross_seqlen`'s special treatment.** It becomes an ordinary field of the
  gfx950 knob class, which removes the keyword-only parameter on
  `build_..._primary`, the `kwargs.pop("cross_seqlen")` in the keyword front
  end, and the argument threaded through `make_parity_traits`. `varlen` goes
  the same way, which matters because P4 would otherwise add a third such
  passenger.

**The one risk, and it is about hardware, not design.** The API only unifies if
gfx1201 adopts it too -- `fmha_tuning_gfx1201.py` has its own `FmhaKnobs`,
`resolve_knobs`, `FmhaPlan` and `plan()`, and doing gfx950 alone yields the
shape without the unification. But there is no gfx1201 board on this host, so
its 298 tests cannot gate the change here. Two honest options, to pick when the
work starts:

1. Land the base class plus the gfx950 subclass now, and port gfx1201 on a
   machine that can run its suite. Leaves two idioms in the tree meanwhile.
2. Port both mechanically and have the gfx1201 suite run elsewhere before the
   PR merges. Faster to a single idiom, but the gfx1201 half ships unverified
   by whoever writes it.

**Do it before P4.** Nothing forces it earlier, but each remaining phase adds
knobs -- windows, the five varlen modes, bias, dropout -- and every one of them
declared against the current split is a field written twice and a converter
argument to thread. The cost of deferring grows with each phase; the cost of
doing it now is one refactor over ~200 lines with 76 tests already covering the
behaviour.

#### Outcome — done, and P2 is why it happened when it did

Landed ahead of P4, triggered by P2 rather than by the deadline: family B needs
its own traits constructor, and *that constructor is the knob-selection logic*.
Building it under the old split would have meant threading a second geometry
through `make_parity_traits` -- exactly the duplication R1 exists to delete.
The wave-geometry choice now lives in `Gfx950Knobs._wave_geometry`, and
`_build_traits` is the single place family B will slot into.

Shipped as specified: `fmha_knobs(arch, **overrides)` returns an arch-specific
subclass, `resolve(meta)` is a method that subsumes both `resolve_knobs` and
`make_parity_traits`, and the resolved object carries `.traits`. Deleted:
`resolve_knobs`, `make_parity_traits`, `FmhaPlan`, `plan()`, and
`cross_seqlen`'s three-site special treatment -- it and `varlen` are now
ordinary `Gfx950Knobs` fields. `FmhaInputMetadata` stayed arch-neutral.

Two things worth carrying forward:

- **The refactor found a dead field.** With the traits moved onto the knobs,
  `meta` became an *unused parameter* of the builder -- and the reason was
  that `meta.sm_scale` had never been read since P0. A build could declare a
  softmax scale and the kernel would silently use `1/sqrt(head_dim)` instead.
  Now honoured, with precedence per-call `scale` > `meta.sm_scale` > derived,
  and a test pinning both halves. Collapsing two representations into one is
  what made the gap visible: a parameter that is passed but never read is
  invisible while something else in the call also consumes it.
- **Family B fails loudly.** `_build_traits` raises if the geometry is not
  8/256/64 rather than falling back, because the fallback would build family
  A's kernel under family B's name -- and it would then be *benchmarked* as
  the new geometry. Tested.

gfx1201 has **not** adopted this; `fmha_tuning_gfx1201.py` keeps its own
`FmhaKnobs`/`resolve_knobs`/`plan`. That was option 1 of the two above, chosen
by default rather than deliberately -- there is still no board to run its 298
tests against. The API is unified in shape, not yet in fact.

86 parity tests pass; 33 production tests pass.

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
- **`cross_seqlen` travels beside the knobs rather than inside them**, as a
  keyword-only argument on `build_..._primary` and a `kwargs.pop` in the
  keyword front end. Owed to **R1**, which deletes the special case by making
  it an ordinary field of the arch-specific knob class.

## Method note

`test_runtime_sm_scale[0.5]` failed on first run against an absolute
tolerance. It was not a bug: raising `sm_scale` sharpens the softmax and grows
`|O|` with it (0.24 at scale 0.05, 4.32 at 0.5), and **the production kernel
shows the same error to within 6% at every scale measured** -- which is what
identified it as bf16 output precision rather than anything the port did. The
fix was the metric, not the tolerance: normalise by `|ref|` when it exceeds 1,
which leaves every O(1) case bounded exactly as before.
