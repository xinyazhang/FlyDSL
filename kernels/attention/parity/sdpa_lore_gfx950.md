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
- **Check `agpr_count` alongside spills.** On this kernel the allocator either
  uses the AGPR file for the O accumulator or abandons it entirely and spills
  to scratch, and the difference is 1.4-1.6x. Four waves keeps it; eight loses
  it.
