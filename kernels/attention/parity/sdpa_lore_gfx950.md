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
- **`LSE` is an output of the forward and an *input* to the backward, and it
  can legitimately be `-inf`.** A causal row with no live key -- every row
  below `seqlen_q - seqlen_kv` under bottom-right alignment, which is every
  `Sq > Sk` varlen sequence -- gives the forward `l_row == 0` and therefore
  `m_row*ln2 + log(l_row) == -inf`. Two consequences, and the second is the
  one that cost time:
  - In the kernel, `-inf` reaching the exponent is survivable only because the
    causal mask runs *after* the scale-and-subtract and overwrites the row.
    That rescue leans on `+inf` flowing through an FMA carrying
    `fastmath<fast>`, whose `ninf` says infinities are absent -- the same
    licence P5 recorded silently deleting a KV tail mask. Floor the LSE input
    once instead.
  - **In the reference.** `torch.softmax` of an all-`-inf` row is NaN, where
    the forward writes `O = 0` (`safe_l_inv` returns 0 when the denominator
    is). So `delta = rowsum(dO*O)` is 0, not NaN, and a host reference that
    skips `nan_to_num` feeds the kernel a NaN and then blames the kernel. The
    tell is that the NaN appears only in sequences with `Sq > Sk`.
- **Check what the *fix* costs before concluding what the *feature* costs.**
  Varlen on the gfx950 dK/dV backward looked like it cost 232 register spills
  and 3x the runtime at head_dim 224 -- the rung already at 486 of 512 VGPRs --
  and a second MFMA family looked like the answer. It was not the feature. The
  logsumexp row pitch stops being 1 under Transformer Engine's `(T, H)` layout,
  so the workaround was sixteen scalar loads in place of four `dwordx4`, and
  *that* was what filled the register file. Making the layout a build axis put
  the same rung back at 486 VGPRs, zero spills, and 720 TFLOP/s, and the
  override was deleted. The narrow rungs paid most, which was the clue: a
  row-tensor read is per tile and does not scale with the head dim, so a cost
  that grows as the head dim *shrinks* is not the head-dim-shaped feature you
  are looking at.
- **A predicted bug and a harness bug in the same place look identical.**
  P4 warns that Q stacked against a batched KV cache (`0x040B`) is the one
  varlen mode where the two sides' batch indices differ. That mode was also the
  only one to fail on first run -- and it was the *test*, which sliced the
  output as packed when dK/dV follow the K layout and are still batched. What
  separated them in a minute was asking which *other* modes should have failed:
  a shared batch index also breaks `varlen_padded`, and that one passed. When a
  failure lands exactly where the lore predicts, check the modes the prediction
  says should fail *with* it before believing it.
- **`V = I` turns the forward's output into a readout of its dropout mask.**
  With the identity as V, `O == P * keep / (1-p)` element for element, so
  `O != 0` recovers the exact mask the forward drew -- and an fp64 backward
  reference built from *that* is a real cross-kernel oracle. The alternative,
  reimplementing the PRNG host-side, is a second transcription of the thing
  under test: it can agree with the kernel and both be wrong. Needs
  `head_dim == seqlen_k`, which is cheap to arrange in a test.
- **A subclass that inherits the forward's `cast_p` inherits its dropout.**
  The forward masks `P` inside `cast_p` -- after the row sum, before the O
  accumulation. The backward must mask `dP` instead and leave `P` undropped,
  because `P` is defined by the *undropped* `lse` it reads. Inheriting the
  forward's `cast_p` applies the mask a second time, to a quantity that must
  not carry it. It happened to fail loudly here only because the forward's
  block reads a `tile_idx` the backward's call site does not pass; a signature
  that matched would have been a silently halved gradient. Override it and say
  why.
- **Low variance is not evidence that a timing measurement is valid.** A ratio
  gate in the dK/dV suite failed at 1.17x against a 1.25 bar, *stably*, over
  eight repeats -- tighter than the noise on a healthy machine. The GPU was
  running ~3.5x slow under a neighbouring job, and under contention the two
  builds compress toward a common floor, so the ratio moves even though nothing
  in the code did. Interleaving the previous commit against the current one
  settled it in one run: they measured identically to three decimal places. Any
  timing gate should assert an **absolute** floor first and say which of the two
  failures it is seeing, or it will one day cost somebody a day of bisecting.
- **A checker that can only report "clean" needs its own negative control.**
  `check_exec_hazard_gfx950.py` parsed `--varlen` in the child and accepted it
  in `check()`, but never placed it in the child's argv -- so for a whole phase
  every "varlen" arm compiled a *dense* kernel and reported it clean. The scan
  was honest about what it scanned and wrong about what that was. It was caught
  only when a later phase added arms and someone read the plumbing. The cheap
  guard that did survive is worth copying: the scan fails outright if the ISA
  contains zero of the instruction it is looking for, which at least proves it
  is looking at a kernel.
- **The accumulator's orientation decides how many randoms you waste.** Philox
  draws four values per call, and whether that is a 4x win or a 4x waste depends
  on which axis a lane's elements run along. A forward lane owns one q row and
  four adjacent k columns: one call covers all four. A dK/dV lane owns one KV
  column and many q rows -- the transpose -- so its elements are `row_stride`
  apart in the counter stream and it needs one call per element, using one
  result of four. Same PRNG, same mask, 4x the calls. What *is* hoistable is
  which of the four slots a lane wants, because a lane's column never changes.
- **A dropout mask must not depend on the tile geometry, and the way to keep
  that true is to never hand the tiling to the PRNG.** The mask is a function of
  absolute `(batch, head, row, column)` and the *maximum* sequence lengths.
  Nothing in `grid_plane`/`grid_offset` takes a `BLOCK_*`, so two builds of one
  problem at different tilings agree bit for bit -- which makes it a standing
  constraint on the tuner rather than a promise: `_GEOMETRY` may move and the
  mask may not follow it. Test it *within* one MFMA family; two families
  accumulate in different orders and differ in the last bits for reasons that
  have nothing to do with the mask.
- **`p = 0` must be bitwise identical to a build with no dropout compiled in,
  not merely close.** At `p = 0` the survivor scale is exactly 1.0 and every
  element is kept, so the arithmetic is unchanged and any difference is a bug.
  A tolerance would not notice `1/(1-p)` applied on the wrong side of a
  subtraction, which is the mistake the scale invites.
- **The survivor scale goes where a constant factor is free, and that is not
  where the forward puts it.** The forward folds `1/(1-p)` into the reciprocal
  of the row sum -- one multiply per output row -- because a row sum exists to
  fold into. In the backward the row sum is an *input*. `dV` takes the scale in
  its epilogue for free, one vector multiply per accumulator. `dS` cannot,
  because the scale multiplies only the `dP` term and `delta` is subtracted
  from it: pulling it out would need `delta/s`, a per-row divide on an input.
  It stays one f32 multiply per element, fused into the `select`. Folding it
  into `dO` before the GEMM instead would round `dO` through bf16 a second
  time.
- **Inheriting a helper that already does the feature adds it twice, and the
  backward does not renormalise it away.** B6 hit this with `cast_p`, which is
  where the *forward* applies dropout; B7 hit it again with
  `seq_pad_mask_if_needed`, which is where the parity forward adds the *bias*.
  Adding the bias a second time in the backward's body measured **3.3 relative
  error**. The reason it is worse here than it would be in the forward is
  structural: the forward renormalises by its row sum, so a uniform score shift
  cancels, while the backward computes `P = exp2(S - lse2)` against an `lse`
  it was *given* and a doubled shift multiplies `P` by `2^shift`. The
  discriminator that identified it was a **row-constant** bias: shift
  invariance says that must be a no-op on `P`, and it was not.
  Before subclassing a softmax helper for the backward, grep it for the feature
  you are about to add.
- **A helper that copies a context's `__dict__` does not see the context
  change.** `DualwaveKernelContext.__init__` does
  `self.__dict__.update(ctx.__dict__)` when handed another context, so every
  helper built from it holds a *snapshot*. That is invisible while the context
  is built once in a prologue, and it becomes a bug the moment anything
  re-points the context mid-kernel -- a GQA group loop rebinding the query
  side, say. The failure is finite and plausible: the stream loader keeps
  staging the first head's Q for the whole group, and the answer is wrong by
  exactly the group sum. The fix is to read the live `ctx_ref` at the call site
  for anything the loop rebinds, and to know which of a helper's attributes are
  snapshots and which are proxies.
- **A drain you added for a race you reasoned about needs both a control and a
  price.** A pipeline drain at a GQA group boundary turned out to be
  bit-identically inert on eight configurations, because the staging is
  partitioned by wave and one wave's DMAs to the same LDS address retire in
  issue order. It costs 0.2-0.4%. Keeping it is then a defensible trade -- the
  ordering property is an inference, not something read off a manual, and a
  wrong guess costs a silently zeroed tile -- but *only if the comment says the
  control found it inert*. An unmeasured guard reads as load-bearing to the
  next person and never gets removed.
- **An f32 reference compared against a bf16 output measures the dtype, not the
  algorithm.** Comparing an in-register f32 group sum against a host-side
  `.float().sum()` of bf16 partials appeared to show the f32 accumulation was
  no better -- because the reference was never rounded back to the output type.
  Round both arms to what the kernel would actually write and the ordering
  reverses. Any A/B between two accumulation strategies has to end in the same
  type or it is answering a different question.
- **A feature's measured cost at the policy's geometry can be mostly the
  geometry.** The bias input read 5.04x at head_dim 64 and 4.05x at 32 --
  numbers that look like the feature and are not. At head_dim 64 the rung does
  not spill at all; switching one existing knob took it to 1.41x, which is what
  the bytes-per-flop law predicts. The tell was that the cost split by **MFMA
  family** rather than by width: a per-element read costs the 32-row family
  double, because a lane there holds 32 accumulator elements to the 16-row
  family's 16. Before reporting a feature's cost, re-tune the rungs it made
  expensive; otherwise the number is a statement about the old tuning.
- **Two features can have the same per-element shape and completely different
  memory behaviour.** Dropout and the bias input both read one value per
  accumulator element with no per-lane vector available, because the dK/dV
  accumulator is the forward's transpose. That is where the similarity ends:
  philox does independent per-lane work and wastes three of every four
  randoms, while the bias's 32 lanes read 32 *consecutive* columns of one q row
  along the tensor's contiguous axis, so the wave gets a fully coalesced line
  even though the lane gets a scalar. "No vector per lane" is not the same
  claim as "uncoalesced", and the cost models differ by an order of magnitude.
- **To separate an occupancy effect from an arithmetic one, hold the feature
  fixed and change only the workgroup count.** A GQA fold at group 8 measured
  186 TF at `B=2` and 720 TF at `B=8` -- same group size, same code, 4x the
  workgroups, 3.9x the throughput. That single pair says the cost is the grid
  and not the fold, and it is worth more than any number of knob sweeps at the
  slow shape. A recorded "this configuration under-occupies the machine and
  that is inherent" beats a knob that moved 2%.
- **Probe an unfamiliar control-flow shape before designing around it.** Nested
  loop-carried `scf.for` had no precedent anywhere in this repo, and the one
  other kernel that needed a group loop had explicitly flattened instead --
  which reads like evidence that nesting does not work, and is not. A 20-line
  kernel summing over both loops answered it in ten minutes. The alternative
  was designing a flattened index scheme around a limitation that did not
  exist.
- **A build constant and the runtime argument that equals it are not
  interchangeable as a loop bound.** A GQA group loop bounded by the runtime
  `num_head_q // num_head_k` cannot be proven to run once, so every MHA caller
  pays for a loop; bounded by the trait, `[0, 1)` is promoted away and the MHA
  path is the pre-feature kernel plus a handful of instructions. The runtime
  values are still checked against the build, so nothing is assumed -- the
  check is what licenses using the constant.
- **Do not edit a kernel source file while a test run is in flight, not even a
  comment.** A 622-test combined run reported 351 failures, every test from
  #272 to the end; the identical command on the unedited tree passed 622/622.
  The cause was a five-line docstring insertion I made a few minutes into the
  run. `compiler/ast_rewriter.py` traces with `inspect.getsource(f)`, which
  re-reads the file through `linecache` -- invalidated on mtime -- and indexes
  it with `f.__code__.co_firstlineno` taken from the module imported *before*
  the edit. Inserting lines above a kernel therefore hands the tracer a window
  onto the wrong text, and every kernel traced after the save is misaligned.
  The tell is the shape of the failure: a contiguous block of failures
  beginning mid-run with everything before it green, and assertion failures
  rather than the HIP errors a device fault would produce. This is invisible to
  the disk-cache switch, since `FLYDSL_RUNTIME_ENABLE_CACHE=0` is exactly the
  setting that forces a re-trace. Suspect the harness before the kernel when a
  failure boundary is a wall-clock instant rather than a configuration.
- **Two element types of the same width are the perfect silent failure.** bf16
  and f16 are both two bytes, so every descriptor, stride, LDS offset, tile
  geometry and register count is identical and *nothing downstream can notice*
  a mismatch -- only the bit interpretation and the MFMA opcode differ. The
  consequence for testing is sharp: **feeding f16 tensors is not evidence that
  f16 was built.** If `dtype_str` fails to reach the traits, the kernel reads
  the operands as bf16, and the result is finite, wrong by a factor near
  2^112, and green against any tolerance. Two cheap gates fix it: assert on the
  emitted ISA (`v_mfma_f32_32x32x16_f16` present, `..._bf16` absent -- the
  *absence* half is what catches a dtype that reached one GEMM and not the
  rest), and assert the error moves in the direction only the new type can
  produce (f16 has three more mantissa bits, so ~8x smaller against fp64; a
  ratio near 1 means it silently ran bf16).
- **A `bitcast` after a conversion is a cast that cannot fail loudly.** The one
  dtype-specific site in the 16-row store emitted `cvt_pk_bf16_f32` and then
  bitcast the result to `elem_dtype`. Under bf16 that is correct; under f16 it
  reinterprets bf16 patterns as f16 and every value is silently wrong. Convert,
  do not reinterpret -- and when a helper is named for one type
  (`_bf16_trunc_pack_v8`) check whether it *branches* internally before
  assuming it is the hardcoded one. Two of the three sites in this kernel
  already handled both types; the name was the only thing that looked wrong.
- **`hash()` on a string is salted per process.** A cross-process bitwise
  comparison that seeds `torch.manual_seed(hash(name))` feeds the two builds
  *different random inputs*, and every case differs. Use `zlib.crc32` or an
  explicit counter. The tell was diagnostic and worth keeping: **46 of 46
  differed, including configurations the change could not reach** -- a real
  regression from a localised edit hits a subset, so "everything differs"
  points at the harness before it points at the kernel.
- **A comparison harness that produces zero cases passes its own diff.** The
  first bf16-regression run crashed inside the loop, wrote two empty files, and
  `diff` reported them identical -- which printed as "BITWISE IDENTICAL". Any
  script whose output is consumed by a comparison needs an assertion on the
  *number of results*, not just on their contents. Same failure family as a
  checker that only ever reports "clean".
- **Range constants chosen under one dtype do not carry to another.** The
  test harness substituted `1e30` for a dead row's `-inf` logsumexp, which is
  fine in bf16 and fp64 and does not exist in f16. That one failed loudly only
  because torch *raises* on an overflowing scalar conversion rather than
  saturating; the same constant inside kernel arithmetic would have become an
  infinity in silence. When adding a dtype, grep the harness for magnitudes as
  carefully as the kernel.
- **A range test whose safety margin is a random variable is not a range
  test.** An f16 overflow test picked an input scale a probe had measured
  clean, and a different seed's tail put the maximum over the line. Compute the
  quantity the test is claiming is in range -- here `max|dS|` in fp64 -- and
  assert *that* against the limit before asserting the kernel is finite. The
  test then states its own precondition instead of inheriting one from a run
  nobody will repeat.
- **Parametrise by dtype; do not add a parallel suite.** A second file drifts:
  the next feature gets added to one and not the other, which is how bf16
  became the only tested type in the first place. One module-level setting that
  every helper reads, driven by a fixture, means the ladder, both MFMA
  families, causal, windows, varlen, dropout, bias and GQA all gain the new
  type at once -- and so do tests that do not exist yet.
