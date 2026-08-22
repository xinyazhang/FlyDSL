# gfx950 FMHA lore: the hazards that bite when you widen the kernel

`flash_attn_gfx950.py` and its parity port were written for head_dim 64 and
128. Every attempt to serve another width has run into the same *class* of
problem: **the compiler does not know it must insert something, and the result
is a wrong-but-finite answer.** No NaN, no fault, no out-of-range address --
just numbers that are a few percent off, or occasionally garbage.

Two instances are documented below, both found the hard way. Read the
"recognising one" section first; it is the part that generalises.

---

## Recognising one

A hazard of this class has a signature that separates it from a logic bug:

- **A semantically-null change fixes it.** Enabling a mask that masks nothing,
  copying a value through an identity, adding an `s_nop`. If the answer moves,
  the arithmetic was never the problem.
- **The instruction counts are identical.** Same MFMA count, same DS reads,
  same everything -- only the register assignment differs. Check this before
  theorising.
- **It is deterministic.** Run-to-run identical. A race would not be.
- **It appears at one width.** Register allocation changes with `K_STEPS_QK` /
  `D_CHUNKS`, so a latent hazard surfaces at whichever shape happens to place
  two registers in conflict. The other widths are lucky, not correct.

### The discriminator ladder

Apply these in order. Which one fixes it tells you the class, and each is a few
minutes' work:

| probe | what it changes | if it fixes it |
|---|---|---|
| `rocdl.sched_barrier(0)` | machine-scheduler ordering | instruction motion, late |
| empty asm, `~{memory}` clobber | IR-level ordering; no registers | instruction motion, early (LICM/sink) |
| a source-level `s_nop` | scheduling **and** allocation | inconclusive -- see below |
| identity asm on the value (`"=v,0"`) | forces a register copy | register allocation |
| `amdgpu-snop-padding=1` (LLVM option) | a wait state before every instruction | **hardware wait state** |

`amdgpu-snop-padding` is the sharpest instrument here and the cheapest to try.
It is far too blunt to ship -- it costs ~3x at head_dim 128 -- but a one-line
answer to "is this a hazard at all?" is worth a lot.

**Neither of the first two can ever fix a wait-state hazard**, so a negative
from them rules out instruction motion and nothing else. `SCHED_BARRIER` emits
zero bytes and `SIInstrInfo::getNumWaitStates` returns 0 for meta instructions;
inline asm is *explicitly skipped* by both wait-state walkers
(`GCNHazardRecognizer.cpp:986` and `:1037`), whatever it clobbers.

**A source-level `s_nop` is not the same experiment as `amdgpu-snop-padding`.**
The padding option is post-RA, so it adds wait states and leaves registers
byte-identical -- that is what makes it a clean probe. An `s_nop` written in
the kernel is a side-effecting intrinsic present during scheduling, so it also
moves the allocator. If it fixes something, diff the nop-stripped streams
before believing you added timing.

Note the asymmetry: an **identity asm emits no instruction**. It pins
scheduling and liveness, and it changes which registers the allocator picks,
but it supplies *no* wait state. So "the anchor fixed it" means *registers*,
not *timing*. Do not conclude you have found a wait-state bug from that alone.

---

## Where the wait-state rules live

Once the ladder says "hardware wait state", stop guessing and read the source.
The rules are all in LLVM, they are gfx950-specific, and they are not in any
ISA doc at this resolution. A local checkout is at `~/dockerhome/meff/`.

**Numbers are deliberately not reproduced here.** They change -- the gfx950
requirements were raised twice in 2025 -- and a stale constant in a lore file
is worse than no constant. Look them up in *your* toolchain's tree, and check
the toolchain actually compiling the kernel, not upstream `main`.

### The three files

| file (`llvm/lib/Target/AMDGPU/`) | scope |
|---|---|
| `GCNHazardRecognizer.{h,cpp}` | everything that matters to us; post-RA |
| `AMDGPUHazardLatency.{h,cpp}` | gfx1250 co-execution only |
| `AMDGPUWaitSGPRHazards.{h,cpp}` | GFX12 only |

### Reading order

1. **`PreEmitNoopsCommon`** is the dispatch table -- one `check*Hazards` call
   per hazard class. Read it first: it tells you which classes *exist*, which
   is how you find out that the thing you suspect is not modelled at all.
2. Follow the checker for your class. For gfx950 MFMA the live ones are
   `checkMAIHazards` (which forwards to the `90A` variant), `checkMAIVALUHazards`,
   `checkVALUHazards`, `checkVMEMHazards`, `checkDPPHazards`,
   `checkPermlaneHazards`. Two are **dead on gfx950** and will waste your time:
   the `908` MAI variant (gfx908 AGPRs) and `checkMAILdStHazards` (returns 0 on
   gfx90a and later, at its first line).
3. The actual counts are in `GFX940_*`-named helper lambdas inside the MAI
   checkers, keyed on **number of passes** and an `IsGFX950` flag. Read the
   lambda, do not transcribe it.
4. `grep hasGFX950Insts() GCNHazardRecognizer.cpp` -- nine sites -- enumerates
   every gfx950 divergence in one shot.

### Get the pass count right first

This is the step that invalidated a day of my analysis. **The pass count is not
in the hazard recognizer.** It comes from the scheduling model:
`SIInstrInfo::isXDL` classifies XDL vs DGEMM, and `SISchedule.td` maps an
opcode regex to `WriteNPassMAI` under the per-subtarget model -- gfx950 has its
own. `v_mfma_f32_32x32x16_bf16` is **8-pass on gfx950**, and I spent a long
time checking it against 16-pass requirements, which are roughly double. Every
derived number was wrong and the scan reported violations that did not exist.

`VOP3PInstructions.td` (predicate `HasGFX950Insts`, flag `is_gfx940_xdl`) tells
you which MFMA opcodes exist on the target at all.

### Limits, and a real off-by-one

`getWaitStatesSince` walks backwards under a caller-supplied bound, and each
checker declares a local `MaxWaitStates`. Those bounds were **not raised** when
the gfx950 requirements went up in early 2025, so for 16-pass XDL the required
distance now exceeds what the walker can even measure. It is a genuine upstream
bug (the bound blames to 2021), it is still present at the head of `main`, and
it does **not** affect this kernel because we are 8-pass. It would affect a
16-pass MFMA kernel silently.

### The two padding options, and why one is a trap

Both are hidden `cl::opt`s declared at the top of `GCNHazardRecognizer.cpp`.

- `amdgpu-snop-padding` is applied unconditionally in `PreEmitNoops`. It is a
  **clean probe**: pre-emit, so it adds wait states without moving a single
  register.
- `amdgpu-mfma-padding-ratio` runs inside `checkMFMAPadding`, which is called
  from only two sites and **bails early** if the instruction is not an MFMA or
  if `getOccupancy() < 2`. A negative result from it may therefore be vacuous.
  Check the occupancy in your metadata before believing one.

### Finding the gaps

The productive query is the negative one: grep the recognizer for the opcode
you suspect. `DS_READ_B64_TR` returns nothing, which is how the current
leading hypothesis for Hazard 2 was formed. Similarly `buffer_load ... lds` is
recognised as LDS DMA (`isLdsDma`) but `createsVALUHazard` returns -1 for it,
so it is not modelled as a hazard source.

### Version skew is the point

The checkout surveyed here (`83e1178daa12`, 2026-08-15) is upstream `main` and
is **ahead of every ROCm 7.x**. That cuts both ways: a fix you find may not be
in your compiler. The concrete example is the gfx940-family MUBUF store-data
WAR hazard (see Hazard 2) -- fixed upstream in June 2026, `main` and
`release/23.x` only.

`git log --oneline llvm/lib/Target/AMDGPU/GCNHazardRecognizer.cpp` filtered for
gfx950 is a five-minute survey and tells you what your toolchain is missing.

---

## Hazard 1: inline asm hides memory ops from `SIInsertWaitcnts`

**Symptom.** Non-deterministic NaN at head_dim 192 and 256; head_dim 64 and 128
fine. No amount of `s_barrier` or `s_waitcnt` at the DSL level helped.

**Cause.** `_ds_read_tr16_b64_imm` emitted `ds_read_b64_tr_b16` as
`llvm.inline_asm`. `SIInsertWaitcnts` discovers outstanding LDS traffic by
scanning MIR for DS instructions; an inline asm is opaque to it, and a
`~{memory}` clobber is **not** an lgkm event. So the backend did not know a
read was in flight and inserted no `s_waitcnt lgkmcnt` before uses of the
result.

That was sound only while nothing read the destination before the kernel's own
cluster-boundary wait. Above head_dim 128 the kernel exceeds the 256
architectural-VGPR cap, the allocator starts using AGPRs, and it places
`v_accvgpr_write` copies of the destination *immediately after the read and
ahead of the wait* -- 22 unwaited uses at 192, 160 at 256, against 0 at 64 and
128.

**Rule.** *Never* emit a memory operation as inline asm. Use the ROCDL op
(`rocdl.ds_read_tr16_b64`, `rocdl.ds_read_tr8_b64`) so the backend can track
the dependency. The result type must be `vector<4xf16>` -- the intrinsic is
typed that way and `Cannot select`s on `vector<2xi32>`; bitcast afterwards.

**The sting in the tail.** Switching to the op cost 7-10%, and the cause was
*not* the waits LLVM then inserted. It was **five extra `s_waitcnt vmcnt(0)`**:
the backend treats `buffer_load ... lds` as an LDS-writing VMEM op and drains
*all* outstanding DMA before any DS read that may alias one in flight. The K
path's `ds_read_b128` provokes no such drain because it carries
`alias_scopes`/`noalias_scopes` from the `LDS_SCOPE_NAMES` scheme.
`ROCDL_LDS_Read_Tr_IntrOp` declares `requiresAliasAnalysis`, so tagging the V
reads the same way removes all five.

**Rule.** A ROCDL LDS read needs alias scopes, or you trade a correctness bug
for a 26%-of-runtime drain.

---

## Hazard 2: VALU -> MFMA wait states the compiler under-counts

**Symptom.** head_dim 96 computes a wrong answer -- ~9% of elements, scattered
in row and column, deterministic, ~2% low. Every other multiple of 32 from 32
to 256 is correct at the same granule, including 192, whose shape is exactly
twice 96's.

**What it is not**, all measured:

- not staging -- the probe reports 0/6144 wrong for K, V and Q on both buffers;
- not masking -- a *no-op* mask at an exact width is bit-identical to unpadded
  at 32/64/128/160/224, and differs only at 96, where it is correct;
- not the softmax -- `return_lse` against `torch.logsumexp` matches to 3e-4
  with zero deviating rows, so `l` and `m` are exactly right;
- not instruction motion -- neither `sched_barrier(0)` nor an IR `~{memory}`
  fence changes anything;
- not scheduling knobs -- `waves_per_eu`, `setprio`, `stagger` and
  `lazy_rescale` all leave the *identical* error.

**What it is.** `amdgpu-snop-padding=1` fixes it and leaves the register
assignment byte-identical, which makes it a hardware wait state. But every
*modelled* hazard is already satisfied: scanning the failing ISA against the
gfx950 MFMA requirements -- looked up as described above, and note that
`v_mfma_f32_32x32x16_bf16` is **8-pass** here, not 16 -- finds **zero
violations**.

Leading hypothesis: **`ds_read_b64_tr_b16` is not modelled by
`GCNHazardRecognizer` at all** (`DS_READ_B64_TR` appears nowhere in it), and
this kernel issues 144 of them at head_dim 96. Unproven.

Ruled out: the gfx940-family MUBUF/MTBUF store-data WAR hazard fixed upstream
in `62b7cf9623fc` (2026-06-01; `main` and `release/23.x` only, so absent from
every ROCm 7.x LLVM). A tempting match -- that commit says it was characterised
on gfx950 by Triton's fused-attention kernel -- but this kernel has six
`buffer_store_dwordx4` and none has a VALU overwriting its vdata inside the
window.

An earlier draft of this section blamed the pair below. That was wrong: the
*working* build contains the same pattern, and 2 wait states is exactly what
the VALU -> MFMA rule requires.

```
v_perm_b32 v119, v157, v158, s14      <- VALU writes v119 (P bf16 packing)
s_nop 1                                <- LLVM inserted 2 wait states
v_mfma_f32_32x32x16_bf16 v[48:63], v[220:223], v[116:119], v[48:63]
                                          SrcB spans v119
```

CDNA requires software-inserted wait states between a VALU write and an XDL
read of that register (ISA ch. 7.6, "Dependency Resolution: Required
Independent Instructions"). LLVM recognised the hazard and emitted `s_nop 1`;
this needs more. head_dim 96 is simply the width whose allocation puts a
freshly-packed P register inside an MFMA's SrcB range.

**Status: worked around, mechanism still unknown.** `ParityGemmHelper.qk`
emits `s_nop 1` after the QK burst. It costs +1.6% at head_dim 64 and +0.8% at 128 (both inside noise) and
buys head_dim 96 at 932 TF against 579 for the padded 128 tile.

The location came from bisection, not from identifying the pair: a wait state
after `qk`, `exp2` or `reduce_sum` each fix it; after `cast_p`,
`lazy_rescale_o`, `load_k`, `load_v`, `reduce_max`, `sub_m` or `pv_step_k` none
do. So the producer is at or before the QK MFMAs and the consumer is before the
P cast. **The exact instruction pair is still unidentified** -- scans of every
documented hazard class found no difference between a broken and a working
build -- so this is a correct fix with an incomplete explanation.

Two earlier attempts, for the record:

- Anchoring the K packs fixes it (579 -> 944 TF) but pins all `2*K_STEPS_QK`
  packs live at once -- 28 packs, 112 VGPRs at head_dim 224 -- and costs
  **-59% there**. It works by shifting registers, not by adding time.
- The shipped `s_nop` is **also** a register perturbation, not a wait state:
  nop-stripped, it gives the same 2817 instructions with 88 differing register
  assignments. It is a workaround of the same kind as the anchor, just a free
  one. The defect remains latent at other widths.
- Putting `s_nop 0` inside `_anchor_v_p`'s asm body **does not fix it**. The
  production anchor pins the *concatenated* P vector and then shuffles the
  packs out, so the `v_perm_b32` is emitted after the anchor and the nop lands
  in the wrong place.

head_dim 96 is a normal rung again.

---

## Method notes, earned the hard way

- **Always run a known-good control.** Three wrong conclusions in this work
  came from a probe with no control: "family B is broken" (the harness was
  broken), a bad V-layout expectation, and a wrong Q buffer size. Build the new
  configuration at head_dim 64 or 128 first, where the answer is known.
- **Always test several shapes.** "The wide body is correct at 96" came from
  one small shape; at `B=4 H=8 S=4096` it is wrong. Use at least
  `(1,8,512)`, `(2,8,1024)`, `(4,8,4096)` and one ragged case.
- **Structured inputs localise far better than random ones.** Uniform Q and K
  makes S constant and isolates the GEMMs; V held constant across tokens makes
  `O[r][d] = d` for *any* weights and isolates the normaliser. Both were
  decisive here where a random-input error map only said "9% of elements".
  Beware the converse: uniform inputs cannot detect a wrong cross-lane
  reduction or a token permutation, because every value is equal.
- **`FLYDSL_RUNTIME_ENABLE_CACHE=0`, always.** `flash_attn_utils.py` and the
  parity helpers are not part of the traced closure, so the JIT cache key does
  not see edits to them. This produced 8 phantom test failures in one sitting.
- **Cap builds at ~8 minutes.** A configuration whose register allocator runs
  away is telling you it does not fit; head_dim 512 at `d_stages` 4 or 8 never
  finished. Treat a slow build as a result, not an inconvenience.
- **Check which GPU you are on, and its temperature.** A shared board at 100 C
  junction reported 218 TF where a cool one reports 1101 -- and that nearly
  produced the conclusion that a single `s_nop` cost 80%. Any perf result that
  looks structurally impossible is a machine-state result until proven
  otherwise; `rocm-smi --showtemp` first, re-measure second.
- **Measure the tax before designing the tile that avoids it.** A head_dim off
  the rung grid ran 27-54% below its own rung, and the obvious reading -- the
  padded columns are wasted MFMA work -- was wrong. The giveaway was that the
  penalty barely moved with the amount of pad: 240-in-256 pads by 6% and was
  the *worst* case at 54%. Cost that does not scale with the thing you blame is
  not that thing. It was the per-K-step mask inside the KV loop, and deleting
  it outright (a one-line experiment, incorrect but decisive) restored every
  padded build to its rung's native rate exactly. Widening the ladder to a
  16-element grid would have been weeks of staging work for the ~7% that was
  actually arithmetic.
- **Never edit a source file while a run is in flight.** The AST rewriter
  reads the kernel source by line number at *trace* time, not at import, so
  inserting a docstring into a file a background pytest is still tracing shifts
  every line under it and fails 168 tests in 72 seconds with no coherent error.
  It looks exactly like a catastrophic regression. Two clean reruns said 259
  passed. Let the run finish, or edit a copy.
- **Do not hand-roll an LDS scaffold for a probe.** A minimal `@flyc.kernel`
  that does `fx.SharedAllocator().allocate(struct).peek()`, writes a bf16
  tensor into the array, barriers and reads back **segfaults the compiler** --
  exit 139, no diagnostic, before reaching any interesting instruction. It is
  not the transpose read, not `fx.Int16` (bf16 does it too) and not the `gpu`
  accessor; all three were bisected out and it still crashes. The working
  pattern is the one `tooling/probe_kv_staging.py` uses: build a real
  `ParityKernelContext` and let `init_lds` do the allocation. Copy that harness
  rather than starting from `SharedAllocator`, and a probe costs an hour
  instead of an afternoon.
- **Check `agpr_count` alongside spills.** On this kernel the allocator either
  uses the AGPR file for the O accumulator or abandons it entirely and spills
  to scratch, and the difference is 1.4-1.6x. Four waves keeps it; eight loses
  it.
- **Sweep `sm_scale`, not only the shape.** Several arithmetic choices are
  correct at `1/sqrt(head_dim)` and progressively wrong away from it, because
  they perturb the softmax *exponent* and the damage scales with `|S|`. The
  backward's dQ found one: folding `sm_scale * log2e` into Q and rounding the
  product back to bf16 -- which is what the forward does, and which the forward
  can afford because `O` is a normalised average -- measured an error ratio of
  1.29 at scale 0.05 and **10.9 at scale 1.0**. A single-scale test called it a
  passing kernel. The discriminator is that the ratio *climbs monotonically
  with the scale*; a genuine transpose or indexing bug does not.
- **A fudge factor is not a tuning knob.** The error-ratio gate
  (`err(ours, fp64) <= fudge * err(same-dtype-reference, fp64)`) prices bf16
  arithmetic order automatically, so a ratio that moves is a *result*. Raising
  the factor to make a rung pass discards the only signal the method produces.
- **`waves_per_eu` is a register budget wearing a scheduling hat, and on the
  backward it decided 1.8x.** The forward's advice above -- "four waves keeps
  the AGPR file, eight loses it" -- is really about how many registers a wave
  may *address*, and the wave count is only one of the two inputs. Measured on
  the dK/dV kernel at head_dim 128, which holds two accumulators rather than
  one: 8 waves gave 0 AGPRs / 118 spills / 444 TF, 4 waves gave 0 AGPRs / 126
  spills / **403** TF -- and the *same* 4-wave build with `waves_per_eu=1`
  gave 721 TF, with 2 waves reaching 788. So halving the wave count bought
  nothing on its own; the occupancy *hint* was still capping the budget at 256
  and the allocator was still spilling rather than reaching AGPRs. Sweep the
  pair, not the wave count, and read `agpr_count` as the discriminator: a build
  at 0 AGPRs with a nonzero spill count is not short of registers, it is
  forbidden from using half of them.
- **A policy default you did not choose can turn an inherited helper into a
  half-implementation.** The forward's `_with_d_axis_splits` switches
  `d_stages` on above head_dim 256. A backward kernel that subclasses those
  knobs and reaches those rungs inherits it -- and under `D_STAGES > 1` the
  shared `ParityGemmHelper.qk` becomes `qk_stage(..., stage=0)` while `pv`
  writes only the first stage's accumulator chunks. A body that never advances
  the stage then reduces over **half the head dim** and returns a finite,
  correctly-shaped wrong answer at exactly the widths that are hardest to
  eyeball. Caught on dQ by an LDS figure that was half what it should have
  been, and by nothing else -- the error sweep had run against an earlier
  override that happened to pin it to 1. Two rules follow: when you subclass a
  knob pipeline, **enumerate what its defaults do at the shapes you newly
  reach**, and refuse a feature the body does not implement *by name* rather
  than trusting a default to stay put.
- **`padded_head` is resolved, not passed, and resolving it wrong emits no mask
  at all.** `build(head_dim=128)` handed a V of width 64 resolves
  `padded_head=False` -- because it only ever saw one extent -- and the build
  then reduces `dP` over the caller's D-axis slack: finite, right shape, 0.70
  relative error. The host is the only place this is visible, so a non-padded
  build should *require* `hdim_qk == hdim_vo == BLOCK_DMODEL` at dispatch. This
  was found by a test helper making the mistake, which is the good case; a
  caller making it gets silently wrong gradients.
- **An `scf.for` with exactly one carried value hands it back unwrapped.**
  Indexing it then returns the first *element* of the vector rather than the
  vector, and the failure surfaces frames away ("Cannot cast type to
  VectorType" inside `_scale_o_accs`). It only appears where the loop carries a
  single value, so head_dim 32 (`D_CHUNKS == 1`) is the one rung that hits it
  and the forward never does -- it always carries `m_row` and `l_row` too.
  Normalise the loop results through a helper rather than indexing them
  directly.
- **Ask what a workgroup *re-reads*, not just what it computes.** The gfx950
  backward's dK/dV kernel keeps K/V resident and streams Q/dO, so every
  workgroup reads the *whole* of Q and dO for its head and the total traffic is
  `seqlen / BLOCK_KV` copies of that slab. `BLOCK_KV` is `32 * waves / shards`,
  which makes the wave count a **bandwidth** knob as well as a register one --
  worth 1.7x at head_dim 160 and 224 (408 -> 690, 486 -> 723) with the register
  allocation byte-identical, because 4 waves is still one per SIMD. Anything
  that divides `BLOCK_KV` (sharding, in that kernel) pays this back at the same
  time as it buys registers, and the two effects have to be priced together.
- **Sharing a slab across the fast grid axis is not a locality win here.** The
  obvious follow-on to the above -- put the KV block on the fast axis so the
  concurrently-issued workgroups all read the same Q -- is **12-15% slower at
  every width tried** (512: 230 TF against 260, 384: 260 against 283, 256: 390
  against 433). The eight XCDs have separate L2s, so a shared slab is
  duplicated across all of them instead of distinct work being spread over
  them. Head-fastest wins on both the forward and the backward, for opposite
  reasons; do not port the *reason*.
- **An MFMA shape's rate is FLOPs per *pass*, not FLOPs.** `SISchedule.td`'s
  per-subtarget model is the only place this is written down, and on gfx950
  `V_MFMA_.32_16X16X16` and `V_MFMA_.32_16X16X32` are both `Write4PassMAI`
  while `V_MFMA_.32_32X32X16` is `Write8PassMAI` -- so the narrow-K 16-row
  shape is **half rate** and the other two are equal. A 16-row family built on
  `16x16x16` measured 280 TFLOP/s where a 32-row one measured 713; rebuilding
  it on `16x16x32` took it to 605 on the same code. Look the pass count up
  before choosing a shape, in the model for *your* subtarget: this is the same
  file the lore already says decides whether an MFMA is 8-pass or 16-pass, and
  it decides the shape choice too.
- **A ratio of rates is not a ratio of times, and one of them flatters.** The
  backward's headline was reported as "bwd/fwd 1.35x-2.33x, inside the ~2.5x
  norm". The column was `fwd_TFLOPs / bwd_effective_TFLOPs`; the wall-clock
  ratio at the same rungs is **3.4x-6.1x**. The two differ by exactly the FLOP
  ratio the backward does more of -- ten GEMM-equivalents against the forward's
  four -- so a rate ratio *looks* like it clears a 2.5x wall-clock norm
  precisely because it has already divided by 2.5. Both numbers are meaningful
  and they answer different questions: the rate ratio says how well the
  backward uses the machine, the time ratio says what a training step pays.
  Label which one, and sanity-check that a claimed wall-clock ratio is
  computable from times alone.
- **A disassembly correlation is not a hardware constraint.** AITER's kernels
  that use `16x16x16` exclusively issue zero `ds_read_b64_tr_b16`, and the ones
  with hundreds of them use the wide-K shapes -- which reads as "the transpose
  does not serve a 16-row operand" and is false. It serves both 16-row shapes
  from a single staged orientation, measured end to end. The correlation was
  with AITER's own generation, not with the instruction. When a competitor's
  ISA is the only evidence, the probe is cheaper than the inference.
- **When an accumulator's row map and an operand's contraction map disagree,
  permute the operand that produced the accumulator.** A `16x16` accumulator
  holds four rows per lane and a `16x16x32` B operand wants eight contraction
  values, so `P` has to come from two accumulators -- and the halves land in
  different quarter waves, which looks like a `permlane`. It is not: *which*
  row lands on which accumulator row is set by the A operand's per-lane row
  index, so a different read address makes the two halves concatenate exactly.
  Free, loop-invariant, and it also keeps the accumulator's rows contiguous for
  the row-tensor loads. Check the producer's addressing before reaching for a
  cross-lane op.
- **A bitwise oracle that generates its own inputs is not an oracle.** The
  sentinel check -- a window build fed `WINDOW_BOTRIGHT` must reproduce plain
  causal *exactly* -- reported 130962 of 131072 elements differing, by up to
  2.28. Both builds were correct to 0.003 against fp64; the runner called
  `torch.randn` inside the per-build helper, so the two builds saw different
  Q/K/V. This is the fourth instance of "always run a known-good control" in
  this work and the first where the missing control was *the inputs* rather
  than a second configuration. A bitwise test must hoist its data above every
  build it compares, and the tell is that both sides pass their own tolerance
  check while failing equality.
- **A tile-cut timing ratio is bounded by occupancy before it is bounded by
  work, and the number alone tells you nothing.** Three measurements in this
  codebase read ~0.92x and meant three different things:

  | case | shape | ratio | truth |
  |---|---|---|---|
  | forward P3, wide body | narrow window vs unbounded, 512 WGs | 0.92x | cut **inert** -- the fix gave 8.04x |
  | backward dQ | dense vs causal, 256 WGs | 0.92x | cut **working** -- 0.69x at 1024 WGs |
  | backward dK/dV | dense vs causal, 16 WGs | 0.94x | cut **working** -- 1.84x at 2048 WGs |

  Two mechanisms, and they point opposite ways. **Too few workgroups** and
  every one is resident, so the clock is set by the *longest* -- the block at
  `kv_start = 0`, which walks every tile either way -- while the cut's whole
  saving is in the *average*. And **causal is only a factor of two even when it
  works**, because block `i` walks `2i + 2` tiles, where a narrow window is a
  factor of seven. So P3 nearly shipped an inert cut on a 0.92x reading and
  dK/dV nearly deleted a working one on a 0.94x reading, from the same number.

  The rule that survives all three: **state what the ratio would be if the cut
  worked, at the shape you are about to use, before you measure it.** If that
  number is near 1 the test cannot discriminate and the shape is wrong. Then
  pin the shape and the workgroup count in the assertion beside the bound, not
  just the ratio.
- **A feature's register cost lands on whichever rung was already at the cap.**
  Causal masking added ~19 live registers uniformly across the gfx950 dK/dV
  ladder. Nine rungs did not notice; head_dim 224 went from 486 VGPR / 0 spills
  to 512 / **53 spills**, and was the one rung where enabling causal bought no
  speedup at all (777 TFLOP/s against 791 dense, where its neighbours went to
  1400). One knob flip fixed it, at 1182. So a per-width tuning table has to be
  keyed on the **feature set** too, and a new feature wants a spill check at
  every rung rather than at the widest one.
- **Keep a mask predicate wave-uniform and the branch stays scalar.** CDNA4
  §11.4 requires EXEC all 1s across `ds_read_b64_tr_b16`, and the cheap way to
  guarantee it is not to audit `scf.if` placement but to build every predicate
  from `readfirstlane`-derived values -- the wave's own row range and the tile
  index. The gfx950 backward's causal and window builds then emit **zero
  EXEC-writing instructions of any kind**, at both MFMA families; the masked
  regions are `s_cbranch` and nothing else. That is a checkable invariant, and
  `tooling/check_exec_hazard_gfx950.py` scans the final ISA for it. Deriving
  the offset from `wave_id` rather than `wave_id_uni` is all it takes to lose.
- **Mask the value the outputs are linear in, not the scores.** The gfx950
  dK/dV backward masks `P` rather than `S`, because `dS = P * (dP - delta)` and
  `dV += P . dO` both inherit a zero from one select per element. The forward
  cannot -- its output depends on the softmax denominator -- but a backward
  can, and it is cheaper *and* keeps `-inf` out of arithmetic that runs under
  `fm_fast`'s `ninf` licence. plan1 records that licence deleting a KV tail
  mask on gfx1201.
