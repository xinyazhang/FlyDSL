# RegAllocGreedy/SplitKit emits a live-range split whose copy-in lane mask is a strict subset of the copy-out mask, leaving a subregister undefined

**Target:** `amdgcn-amd-amdhsa`, `gfx950` (wave64)
**LLVM:** 24.0.0git @ `e2a39f504fee836e4def9581bed817ecc327b9dc`
**Severity:** miscompile — silent wrong memory access. The undefined subregister
becomes the *base address* of a buffer descriptor that is then dereferenced.
**Reproducer:** deterministic, compile-only, no GPU. See §6.

---

## 1. Summary

`SplitEditor` splits a live range of an `sgpr_512` (a merged `S_LOAD_DWORDX16`
kernarg tuple). At the **copy-in** point it emits a bundle covering lanes
`sub0_sub1_sub2_sub3` and `sub6_sub7`. At a **copy-out** point later in the
function it emits a bundle covering `sub8..sub15` **and `sub4_sub5`**.

`sub4_sub5` is never defined anywhere in the function. The whole 512-bit
register is nevertheless spilled by `SI_SPILL_S512_SAVE`, reloaded, and the
undefined pair is copied out and used as an address operand.

This is not a benign dead-path `undef`: the value is reloaded and consumed. In
the final ISA it becomes the low half of a buffer resource base, and the
resulting `buffer_load` is issued against an address the kernel was never given.
`num_records` in that same descriptor is correct — it comes from a different,
clean chain — so hardware bounds clamping constrains the *offset* against a
valid bound while the *base* is garbage. The access is therefore "in bounds"
with respect to the wrong allocation: it either faults with
`hipErrorIllegalAddress` (observed) or silently reads and writes unrelated
memory.

Both `-verify-machineinstrs` and `-verify-regalloc` pass on the failing compile.

## 2. MachineIR evidence

From `-print-after-all`, in the section headed
`*** IR Dump After Greedy Register Allocator (greedy) ***`. Copy-in, in
`bb.31`:

```
9792B   %1270:sreg_64 = V_CMP_GT_I64_e64 1, %7924:sreg_64, implicit $exec
9808B   %1273:sreg_64 = S_AND_B64 $exec, %1270:sreg_64, implicit-def dead $scc
9824B   $vcc = COPY %1273:sreg_64
9828B   undef %8004.sub0_sub1_sub2_sub3:sgpr_512 = lr-split COPY %7996.sub0_sub1_sub2_sub3:sgpr_512 {
          internal %8004.sub6_sub7:sgpr_512 = lr-split COPY %7996.sub6_sub7:sgpr_512
9836B   }
        SI_SPILL_S512_SAVE %8004:sgpr_512, %stack.3, implicit $exec, implicit $sgpr32 :: (store (s512) into %stack.3, align 4, addrspace 5)
9852B   S_CBRANCH_VCCZ %bb.33, implicit $vcc
```

`%8004` occurs **exactly three times** in the entire function — the two lines of
that bundle and the spill. There is no definition of `%8004.sub4` or
`%8004.sub5`, and no full-register def.

Copy-out, ~106000 slots later:

```
115896B  %7901:sgpr_512 = SI_SPILL_S512_RESTORE %stack.3, ... (load (s512) from %stack.3, ...)
115992B  undef %7900.sub8_sub9_sub10_sub11_sub12_sub13_sub14_sub15:sgpr_512 = lr-split COPY %7901.sub8_..._sub15:sgpr_512 {
           internal %7900.sub4_sub5:sgpr_512 = lr-split COPY %7901.sub4_sub5:sgpr_512
116000B  }
         %2659:sreg_32 = S_MUL_I32    %7495:sreg_32, %7900.sub5:sgpr_512
116016B  %2661:sreg_32 = S_MUL_HI_U32 %7495:sreg_32, %7900.sub4:sgpr_512
116032B  %2662:sreg_32 = S_ADD_I32    %2661, %2659, implicit-def dead $scc
```

So the copy-out mask contains `sub4_sub5` and the copy-in mask does not. The
value is used in a 64-bit multiply that computes a byte offset into a tensor.

## 3. The same defect in the emitted ISA

Four `s_mov_b64` should stage `s[76:83]` into `s[56:63]`; the third is absent:

```asm
  24B4: s_load_dwordx16 s[76:91], s[42:43], 0xf8    ; kernargs, byte offset 248..311
  24D4: s_waitcnt       lgkmcnt(0)
  24D8: s_mov_b64       s[56:57], s[76:77]          ; ok
  24DC: s_mov_b64       s[58:59], s[78:79]          ; ok
                                                    ; <<< MISSING: s_mov_b64 s[60:61], s[80:81]
  24E0: s_mov_b64       s[62:63], s[82:83]          ; ok
  24E4: v_writelane_b32 v254, s56, 0
  ...
  2510: v_writelane_b32 v254, s60, 4                ; <<< no reaching definition
  2518: v_writelane_b32 v254, s61, 5                ; <<< no reaching definition
```

and ~7 KB of code later:

```asm
  3F68: v_readlane_b32  s16, v254, 4                ; reload of the never-defined s60
  3F70: v_readlane_b32  s17, v254, 5
  3F78: s_mov_b64       s[12:13], s[16:17]
  3F84: s_mul_i32       s8, s10, s13                ; head index * stride_do_head
  4024: s_lshl_b64      s[8:9], s[8:9], 1           ; * sizeof(fp16)
  4038: s_add_u32       s76, s8, s12                ; + base pointer lo  -> V# word0
  4048: s_and_b32       s77, s8, 0xffff             ; + base pointer hi  -> V# word1
  419C: s_mov_b32       s48, s76
  41A0: s_mov_b32       s49, s77
  41EC: buffer_load_dwordx4 v13, s[48:51], 0 offen lds   ; <<< dereferenced
```

The value carried by the dropped copy is a per-head row pitch (kernarg offset
264). Independently checked: what `s_load_dwordx16` places in `s[80:81]` is
never read before `s80` is redefined at `0x272C`, so the copy was not folded
into a redundant-move elimination — it was lost.

A sibling descriptor built from the same `s[10:11]` is dereferenced correctly
0x48 bytes earlier, which is what makes this a *lost definition* rather than a
wholesale addressing failure.

## 4. Where it is not

Each of these was measured across 216 built configurations of the affected
kernel, not argued:

| Hypothesis | Result |
|---|---|
| It is in the lane-spill lowering (`v_writelane`/`v_readlane`). | **No.** Forcing scratch spilling with `-amdgpu-spill-sgpr-to-vgpr=false` (which moves `private_segment_fixed_size` 0 → 508) flags an *identical* set of objects. The copy is lost before spill lowering picks a strategy. |
| It is scalar pressure; shorten the live ranges and it goes away. | **No.** Hoisting nine `i64` strides out of a loop moved spills down broadly (median 97 → 87; 180 of 216 builds improved) and the defect went 42 → 28 — but **18 previously-clean builds newly acquired it**, 14 of them while their spill count fell. Spill count is not the control variable. |
| It is one of the `-mllvm` flags in use (`-enable-post-misched=false`, `-lsr-drop-solution=true`). | **No.** Removing either or both reproduces. Sweeping them gives 28 flagged each, union 37, intersection 19 — 19 builds are flagged under every setting tried. |
| A newer LLVM fixes it. | **Not by rebasing.** See §7. |

The MachineIR in §2 is consistent with all four: the loss is already present in
greedy's own output, upstream of every one of those knobs.

## 5. Blast radius, and a stronger predicate than the one first used

Two detectors were run over **907** objects from one build tree:

| predicate | flagged |
|---|---|
| undefined at spill → reloaded → used as a buffer descriptor base → dereferenced | 42 |
| undefined at spill at all (no `v_writelane_b32` may name an SGPR with no reaching definition) | 160 |

Every hit of the first is a hit of the second.

| family | frontend | objects | lost definition | reaches a descriptor base |
|---|---|---|---|---|
| `flyc_bwd_dkdv` | MLIR/FlyDSL | 216 | **114** | **42** |
| `flyc_bwd_dq` | MLIR/FlyDSL | 216 | 21 | 0 |
| `flyc_attn_fwd` | MLIR/FlyDSL | 216 | 19 | 0 |
| `bwd_preprocess` | **Triton** | 72 | 6 | 0 |
| `bwd_preprocess_varlen` | Triton | 72 | 0 | 0 |
| `_gemm_afp4wfp4_preshuffle_kernel` | Triton | 112 | 0 | 0 |
| `debug_simulate_encoded_softmax` | Triton | 3 | 0 | 0 |

*(An earlier revision said 1054 objects, 166 flagged, and 12 for
`bwd_preprocess`. `pop.tsv` lists some families under **both** their bare and
their hashed names, so those rows were doubled: `bwd_preprocess` (72+72),
`bwd_preprocess_varlen` (72+72) and `debug_simulate_encoded_softmax` (3+3).
1054 − 72 − 72 − 3 = 907. The `flyc_*` rows are hashed-only and were never
affected, which is why **the headline 42 and 114 stand**. Correcting this took
two passes — the first deduped the two obvious families and missed the third —
so count the bare and hashed rows per family rather than assuming the pattern.)*

**This is not specific to one frontend**, and a companion report
(`REPORT-triton-gfx950-lost-sgpr-copy.md`) now reproduces the mechanism from a
~50-line Triton kernel on three LLVM builds, with the offending copy explicitly
tagged `lr-split`. But the two frontends are **not** in the same state, and the
difference matters more than the shared mechanism:

- 6 `bwd_preprocess` objects carry the *mechanism* — a split whose defined lanes
  are a strict subset of the register it then spills. In all six the lost lane is
  **dead or rematerialised before use**.
- `flyc_bwd_dkdv` is the only family measured to **consume** a lost definition,
  and the only one where it reaches a dereferenced address.

A sweep of **10,265** Triton modules (§6 of the companion) found 5,932 carrying
the mechanism, 131 where MachineIR says a restore reads an undefined lane, and
**zero** where the value survives to be computed with — in every one of the 131
a later pass regenerates the missing half (`s_ashr_i32 x, y, 31`). The exposure
difference is tuple width and rematerialisability: FlyDSL's 21 `i64` strides
merge into `sgpr_512`, while Triton's widest observed SGPR spill is `S256` and
its 64-bit indices are cheap to recompute.

### 5.1 The right verifier predicate is *not* the one first proposed

An earlier revision of this section proposed *"no SGPR spill may name a source
register with no reaching definition"* as a MachineVerifier check. **Measured:
that predicate has a large false-positive class and should not be used.** Of
2896 `bwd_kernel_dk_dv` configurations it flags 1859; a CFG-based analysis flags
**0** of the same set. Hand-traced in `bwd_kernel_dk_dv_02894`: both flagged
writelanes spill a vreg defined by `IMPLICIT_DEF` on the paths that dominate
them — a post-SSA register genuinely undef there. Spilling and reloading that is
legal, and a verifier check phrased this way would fire on every `IMPLICIT_DEF`
spill in the tree.

The predicate that discriminates is one level up, on spill slots rather than
registers:

> **No restore may read a spill-slot lane that no store into that slot defines
> on a path reaching it.**

That is a must-analysis over the MachineIR CFG — state is the set of
known-defined lanes in the slot, merge is intersection over predecessors, a save
overwrites, and a restore reports `lanes-read − state`. It fires on the failing
object here, naming exactly the documented site
(`%stack.3 restore %7901 in bb.37: reads [4,5,8..15], defined-on-all-paths
[0,1,2,3,6,7] → MISSING [4,5,8..15]`), and is silent on the clean sibling, on
the Triton reproducer, and on all 1859 `IMPLICIT_DEF` cases. `mircfg.py` beside
this file implements it and `witness.py` prints a concrete offending path, so a
hit can be checked by reading the dump rather than trusting the analysis.

## 6. Reproducer

Compile-only, no GPU, ~15 s. Inputs are a ROCDL-dialect MLIR module captured
immediately before `gpu-module-to-binary`, so everything exercised is the AMDGPU
backend.

```sh
gunzip -c 18_reconcile_unrealized_casts.mlir.gz > /tmp/mod.mlir
python repro.py /tmp/mod.mlir /tmp/out.s      # pip install flydsl==0.3.1 or 0.3.2
sha256sum /tmp/out.s                          # 1452f523…  — the missing s_mov_b64 is present
```

`20_llvm_ir.ll.gz` is the same module one step lower if you would rather drive
`llc`. To obtain the MachineIR in §2, set `print-after-all` through the same
`cl::opt` mechanism the driver uses (`drive.py` here does this via ctypes) —
**no `llc` build is required**, and assertions are enabled in the shipped
library.

The affected object and a clean sibling built from identical source with one
build flag changed are included; the detector should report `1 of 2` on them.

## 7. Not fixed on trunk, and why rebasing will not help

Verified by hand against **`upstream/main` at `810dc356352a` (2026-09-02)** —
pin the ref when you re-check this, because trunk moves and an unpinned
"unchanged since main" claim rots within days:

```
$ git diff --stat e2a39f504fee upstream/main -- \
      llvm/lib/CodeGen/SplitKit.cpp llvm/lib/CodeGen/SplitKit.h \
      llvm/lib/CodeGen/LiveInterval.cpp llvm/include/llvm/CodeGen/LiveInterval.h
 llvm/lib/CodeGen/SplitKit.cpp | 5 ++---
 1 file changed, 2 insertions(+), 3 deletions(-)

$ git log --oneline e2a39f504fee..upstream/main -- llvm/lib/CodeGen/SplitKit.cpp
 745e57793431 [SlotIndexes] Use analysis block numbers (NFC) (#215171)
```

**Exactly one commit has touched `SplitKit.cpp` since the pin, and it is
NFC** — a hoist of `getBlockNumbered` out of `splitLiveThroughBlock`.
`SplitKit.h`, `LiveInterval.cpp` and `LiveInterval.h` are byte-identical.
`buildCopy`, `buildSingleSubRegCopy`, `getLiveLaneMaskAt`, `addDeadDef`,
`defFromParent` and `transferValues` are therefore semantically unchanged across
the whole range, and no new MachineVerifier check for undefined subranges
landed.

The adjacent machinery has one change, also NFC:
`85e73b167487` ("CodeGen: Pass instruction to `LiveRangeEdit::useIsKill`",
#219466, 2026-08-28) touches `LiveRangeEdit.{h,cpp}` — 5 insertions, 5 deletions
— and only replaces a `MachineOperand` parameter with the instruction it was
used to recover. It is the sole commit in those two files since the pin.

Both implicated lines are still current on trunk: `SplitKit.cpp:452` is still
`if (PS == nullptr) continue;` and `:654` is still
`LaneBitmask LaneMask = getLiveLaneMaskAt(Edit->getParent(), UseIdx, MRI);`.

**Stated precisely:** this shows *the implicated code is unchanged*, which is
much stronger than "we could not reproduce on a different compiler". It does
**not** independently show that trunk reproduces — a different pass could feed
greedy a different live-range shape — but there is no fix in this code to find.

## 8. Suspect commit

`83dca924c250` — *"[CodeGen][SplitKit] Fix a crash in addDeadDef (#197014)"*,
Quentin Colombet, 2026-05-13, fixing issue #178867. It is an **ancestor of the
pin** and still current on `main`.

Two changes in it are the right shape for this defect, and its own commit
message describes the narrowing:

> *"original is the ancestor of the parent and may cover some lanes in the
> lanemasks that are not covered by the parent live-interval"*
> *"The fix consists in taking the lanes covered by the parent, not the
> original value, when creating the live-interval for the children."*

**(a) `defFromParent` narrowed its lane mask from the original interval to the
parent.** Before, `LaneMask` was accumulated over `OrigLI.subranges()` live at
`UseIdx`; after, it is `getLiveLaneMaskAt(Edit->getParent(), UseIdx, MRI)`. By
the commit's own statement the parent's mask is a subset of the original's, and
this mask feeds `Edit->rematerializeAt(...)` and the `buildCopy` path — i.e. it
is exactly the *copy-in* mask that came out too small in §2.

**(b) `addDeadDef` stopped asserting.** `getSubRangeForMask`, which ended in
`llvm_unreachable("SubRange for this mask not found")`, became
`findSubRangeForMask` returning `nullptr`, with the caller doing:

```cpp
const LiveInterval::SubRange *PS = findSubRangeForMask(S.LaneMask, Edit->getParent());
if (PS == nullptr)
  continue;                       // silently skips S.createDeadDef(...)
```

So a child subrange whose mask is absent from the parent no longer crashes — it
silently gets no dead def. That converts the crash of #178867 into a missing
definition, which is precisely the state observed.

The commit shipped with **no test**, for a stated reason:

> *"The crash was reported by a downstream user and they were not able to
> capture the issue with an upstream target."*

### 8.1 The narrowed mask *is* the copy-in mask — chain verified by reading

The connection is not circumstantial. `defFromParent` ends:

```cpp
LaneBitmask LaneMask = getLiveLaneMaskAt(Edit->getParent(), UseIdx, MRI);   // <- (a) changed this
...
SlotIndex Def;
if (LaneMask.none()) { /* IMPLICIT_DEF */ }
else { ++NumCopies; Def = buildCopy(Edit->getReg(), Reg, LaneMask, MBB, I, Late, RegIdx); }
```

so the mask the commit narrowed is passed **directly** to `buildCopy` as the set
of lanes to copy. `buildCopy` with a partial mask then does:

```cpp
if (!TRI.getCoveringSubRegIndexes(RC, LaneMask, SubIndexes))
  report_fatal_error("Impossible to implement partial COPY");
for (unsigned BestIdx : SubIndexes)
  Def = buildSingleSubRegCopy(FromReg, ToReg, MBB, InsertBefore, BestIdx, DestLI, Late, Def, Desc);
DestLI.refineSubRanges(Allocator, LaneMask, ...);
```

— one `COPY` per covering subregister index, bundled, which is exactly the shape
observed in §2 (`sub0_sub1_sub2_sub3` plus an `internal` `sub6_sub7`, and no
`sub4_sub5`). The closing `refineSubRanges` on the same narrowed mask is the
"refining of the subranges" the commit message names as the trigger.

So the defect is: **`getLiveLaneMaskAt(parent)` under-reports at the copy-in
`UseIdx` relative to the lanes a later copy-out asks for**, and every step from
there to the missing `s_mov_b64` is mechanical.

### 8.2 What a revert would show, and how to read each outcome

**Confidence that a revert changes this behaviour: high** — the changed value is
the copy-in mask itself. **Confidence that a revert is the right *fix*: low**,
which is a different question; see below.

Three outcomes, all informative:

- **The restored `llvm_unreachable("SubRange for this mask not found")` fires.**
  This module is then exactly the case `83dca924c250` was suppressing, and the
  true defect is further upstream: the parent interval genuinely does not cover
  lanes that are genuinely live. The commit converted a crash into silent
  memory corruption. Strongest outcome for an upstream report.
- **Clean code — all four `s_mov_b64` present, detector clean.** The commit
  over-narrowed and the `OrigLI`-derived mask was right here. Note a plain
  revert is still **not shippable**: it reinstates the crash of #178867. The
  real fix would widen the copy-in mask to cover what later copy-outs require,
  or make the parent's subranges cover what is live.
- **Still lost.** `defFromParent` is not the emitter for this bundle; move to
  `transferValues` / `extendPHIKillRanges`.

**Methodological warning for whoever runs it.** §4 established that this defect
*churns* under perturbation — 32 builds cleared and 18 newly infected from a
change that only moved live ranges. A revert also perturbs allocation, so **a
clean detector run alone does not confirm the hypothesis.** Check the mechanism:
does the bundle at `9828B` now include `sub4_sub5`, and does `%8004` acquire a
definition for it? Verify at the MachineIR level, not at the symptom.

## 9. Workaround

`-sgpr-regalloc=basic`. `SplitEditor` is owned only by `RegAllocGreedy`
(`RegAllocGreedy.h`), so this removes the mechanism rather than perturbing the
allocation — which matters, because perturbation does not work here (§4).

Verified in MachineIR on the failing compile:

| partial-lane `lr-split` copies, by register class | default | `-sgpr-regalloc=basic` |
|---|---|---|
| `sgpr_512` / `sgpr_128` / `sreg_64` | 22 / 8 / 2 | **0 / 0 / 0** |
| `vreg_*` / `av_*` (untouched, for contrast) | 96 / 15 / 90 / 12 | 96 / 15 / 90 / 12 |

and it holds under re-perturbation, which is the test the knob-level "fixes"
fail:

| perturbation | lost defs alone | + `-sgpr-regalloc=basic` |
|---|---|---|
| none | 4 | **0** |
| `-greedy-reverse-local-assignment` | 2 | **0** |
| `-greedy-regclass-priority-trumps-globalness` | 4 | **0** |
| `-amdgpu-spill-sgpr-to-vgpr=false` | 4 | **0** |
| `-split-spill-mode=size` | 4 | **0** |

Note the middle column: several of those perturbations reach *narrow-predicate*
clean while still carrying lost definitions. They move the damage rather than
removing it, which is why the slot-level predicate of §5.1 matters.

**Cost, measured statically and not yet benchmarked on hardware:** instructions
5129 → 6868, driven almost entirely by `v_readlane_b32` 149 → 1419, of which 109
land inside the MFMA span (0 in the default build). `sgpr_spill_count` 105 →
114; `vgpr_count` unchanged at 394. Basic RA rematerializes the strides from the
kernarg segment at each use — the same shape as the known-clean sibling object,
which is independent corroboration that the rematerializing shape is the safe
one. `-vgpr-regalloc=basic` on top adds 26 VGPRs for no gain; do not use it.

## 10. Allocator inventory, and workarounds that do *not* work

Four allocators are registered with the generic `RegisterRegAlloc` registry:
`fast`, `basic`, `greedy`, `pbqp` (`llvm/lib/CodeGen/RegAlloc*.cpp`). **AMDGPU's
`-sgpr-regalloc` accepts only three of them** — `AMDGPUTargetMachine.cpp`
registers `basic`, `greedy` and `fast` with its own `SGPRRegisterRegAlloc`
registry; `pbqp` is never registered there.

Everything below was **measured on the §6 reproducer**, not reasoned about:

| attempt | result |
|---|---|
| `-sgpr-regalloc=basic` | **Works.** 0 writelanes with no reaching definition. |
| `-sgpr-regalloc=pbqp` | **Silently ignored.** Output is byte-identical to the default build (SHA-256 `f81323c7…`), lost definition still present. The value does not reach the SGPR registry, and the compile does not fail — so this is a trap: it looks like a knob and is a no-op. |
| `-sgpr-regalloc=fast` | **Still flagged** (`v_writelane_b32 v251, s5, 3`). `RegAllocFast` has no `SplitEditor`, so either a second mechanism produces undefined spill sources or the detector has a false positive on this output. **Not established which** — and `fast` is the `-O0` allocator, so it is not a candidate workaround regardless. |
| `-enable-subreg-liveness=false` | **Crashes the compiler.** `RegisterPressure.cpp:68: void decreaseSetPressure(...): Assertion '(NewMask & ~PrevMask).none() && "Must not add bits"' failed.` This was the most attractive idea on paper — the faulty paths in `SplitKit.cpp` are gated on `LI.hasSubRanges()` (lines 428, 439), so removing subranges should remove the mechanism while keeping `greedy`. It is not usable: `GCNSubtarget::enableSubRegLiveness()` returns `true` unconditionally and the backend does not survive the override. |
| `-amdgpu-load-store-vectorizer=false` | **Still flagged.** (It is an IR-level pass and does not govern the `s_load_dwordx16` merge that forms the tuples.) |

Taken with §4's knob sweep, **`-sgpr-regalloc=basic` is the only setting measured
to remove the defect**, and it works because it removes `SplitEditor` from the
pipeline rather than perturbing its input.

## 11. "Would passing the 64-bit strides as pairs of 32-bit values avoid it?"

A natural question, since the tuples that get split are built from `i64`
kernargs. **Measured answer: no.** The defect is *"copy-in lane mask is a strict
subset of copy-out lane mask"*, and nothing in that is about lane width.

Two direct observations, both from the artifacts here:

**Splits already happen at single-dword granularity.** Lane masks on the
`lr-split COPY`s in the failing function, counted from greedy's output:

```
 10  sub6_sub7        5  sub2      3  sub10     2  sub4_sub5
  9  sub2_sub3        4  sub0      3  sub0_sub1_sub2_sub3
  5  sub0_sub1_sub2_sub3_sub4_sub5   1  sub6    1  sub14   ...
```

`sub0`, `sub2`, `sub6`, `sub10` and `sub14` are lone dwords. SplitKit is already
operating at 32-bit granularity in this very function; 64-bit lanes are not the
precondition.

That census is the load-bearing evidence here: it is read straight off the
`lr-split COPY` masks in greedy's own output, so it does not depend on any
detector.

**The Triton reproducer settles it independently, at 32-bit granularity on a
64-bit tuple.** Its split copies `sub1` of an `sreg_64` and spills `sub0_sub1`,
leaving `sub0` — a single 32-bit `i32` kernarg — undefined
(`REPORT-triton-gfx950-lost-sgpr-copy.md` §3). No `i64` is involved anywhere in
that kernel's lost value.

A supporting ISA observation, at the weaker predicate: in
`flyc_attn_fwd-c03d7148…`, `v251` lanes 4, 5 and 7 are written from
`s68`/`s69`/`s71`, and `s71` stands alone while `s70` at lane 6 has a reaching
definition. Treat this one as corroboration rather than proof — §5.1 establishes
that the writelane-level predicate cannot separate this defect from a legal
`IMPLICIT_DEF` spill, and this object was not re-checked with the CFG analysis.
The two items above do not depend on it.

**Worse, it is the same class of intervention that has already been measured to
churn rather than fix.** §4 records hoisting nine `i64` strides out of a loop:
32 builds cleared and **18 previously-clean builds newly acquired the defect**.
Narrowing 21 strides from `i64` to pairs of `u32` changes the liveness
granularity of exactly the values involved, so it would very likely change
*which* builds are affected without changing whether the defect can occur — and
it would do so while looking like a fix.

Two secondary points, reasoned rather than measured: the `S_LOAD_DWORDX16` merge
that forms the tuples is driven by contiguous kernarg offsets in
`SILoadStoreOptimizer`, not by source-level types, so two adjacent `u32`
kernargs merge into the same wide tuple as one `i64`; and 32-bit strides would
be a semantic narrowing of the ABI for a defect they do not address.

## 12. "Can splitting just be turned off inside greedy?"

**There is no such knob.** The split-related `cl::opt`s in `llvm/lib/CodeGen`
are all heuristics, and none of them gates `RAGreedy::trySplit`:

| option | what it does | tested |
|---|---|---|
| `split-spill-mode` | picks the spill mode *within* splitting | still flagged (§4) |
| `huge-size-for-split` | in `TargetRegisterInfo::shouldRegionSplitForVirtReg`, suppresses **region** splitting only for vregs that are *also* trivially rematerializable | `=0` and `=1` both still flagged, at the identical address |
| `split-threshold-for-reg-with-hint` | only applies to registers with an allocation hint | `=0` still flagged, identical address |
| `enable-split-loopiv-heuristic` | one loop-IV heuristic | n/a |

Greedy's stage machine (`RS_Split` / `RS_Split2` → `trySplit` →
`tryLocalSplit`/`tryRegionSplit`/`tryInstructionSplit`/`tryBlockSplit`) is not
guarded by any option. So "keep greedy, stop splitting" is not expressible.

The way to get an allocation with no live-range splitting is an allocator that
has no `SplitEditor`, which is what `-sgpr-regalloc=basic` is — and it is
*better* than a hypothetical global disable would be, because AMDGPU allocates
the two register banks separately: SGPRs move to `basic` while VGPRs keep
`greedy`. Measured, VGPR splitting is untouched (`vreg_*`/`av_*` partial-lane
split counts identical at 96/15/90/12).

The allocation cost, measured statically on the reproducer: `sgpr_spill_count`
105 → 114, instructions 5129 → 6868, `v_readlane_b32` 149 → 1419 (109 of them
inside the MFMA span), `vgpr_count` unchanged at 394. Basic RA rematerializes
the strides from the kernarg segment at each use rather than keeping them live —
the same shape as the known-clean sibling object.
## 13. Artifacts

All of these sit beside this file, so §6 runs without fetching anything.

**Reproducer inputs**

| file | what it is |
|---|---|
| `18_reconcile_unrealized_casts.mlir.gz` | the module immediately before `gpu-module-to-binary` — pure LLVM/GPU/ROCDL dialect with `#rocdl.target<chip="gfx950">` attached, so everything it exercises is the AMDGPU backend |
| `20_llvm_ir.ll.gz` | the same module one step lower, for driving `llc` directly without any of this project's tooling |

**Objects**

| file | what it is |
|---|---|
| `flyc_bwd_dkdv-2e43deb2….hsaco` | the affected object — `sgpr_count` 106, `sgpr_spill_count` 105, `vgpr_count` 394, `private_segment_fixed_size` 0 |
| `flyc_bwd_dkdv-be88cb68….hsaco` | a **clean sibling** from identical source with one build flag changed (`PADDED_HEAD=False`); it rematerialises the strides from the kernarg segment instead of staging and spilling them, and has 71 writelanes to the affected object's 105 |

The detectors should report `1 of 2` across that pair — check this before
trusting any scan, since neither predicate has been proved free of false
positives.

**Scripts**

| file | what it is |
|---|---|
| `repro.py` | ten lines: parse the module, run `gpu-module-to-binary{format=isa}`, write the result |
| `drive.py` | compiles the module under arbitrary LLVM `cl::opt` settings via ctypes — including `print-after-all`, which is how §2 was obtained without building `llc`. Passes a NULL old-value pointer, which is what avoids the unsound `static_cast` in the option setter |
| `spillscan.py` + `undefscan.py` | the narrow predicate (undefined → reloaded → descriptor base → dereferenced). Keep them side by side; `spillscan` imports `undefscan` |
| `lostdef.py` | the writelane-level predicate. **Superseded — see §5.1**: it cannot separate this defect from a legal `IMPLICIT_DEF` spill (1859 false positives in one family). Kept because §4's and §5's counts were produced with it |
| `mircfg.py` | the slot-level must-analysis of §5.1 — the predicate that discriminates |
| `witness.py` | prints a concrete CFG path for a `mircfg.py` hit, so a finding can be checked by reading the dump |

**Data**

`pop.tsv` — the scan behind §5, one row per object as
`lostdef-count`, `descriptor-base-count`, `name`. `flagged_old.txt` (42),
`flagged_final.txt` (28 after the stride hoist) and `flagged_{noln,nopm,nolsr}.txt`
(the knob sweep) are the flagged-hash sets behind §4; diff them to reproduce the
churn numbers without rebuilding anything.

Both detectors need `LLVM_BIN` pointing at a directory containing
`llvm-objdump` and `llvm-readelf`; any recent ROCm LLVM will do, since the
disassembler version does not matter here.


## 14. What triggers the subset condition, and can source code avoid it?

Both questions have the same answer at the bottom, so they are together.

### 14.1 The preconditions, in order

```cpp
LaneBitmask getLiveLaneMaskAt(const LiveInterval &LI, SlotIndex Idx, const MachineRegisterInfo &MRI) {
  if (!LI.hasSubRanges())
    return MRI.getMaxLaneMaskForVReg(LI.reg());     // full mask
  LaneBitmask LaneMask;
  for (const LiveInterval::SubRange &S : LI.subranges())
    if (S.liveAt(Idx)) LaneMask |= S.LaneMask;
  return LaneMask;
}
```

The copy-in mask is the union of the **parent's** subranges live at the
insertion index. Four things must hold:

1. **Subregister liveness is on.** `GCNSubtarget::enableSubRegLiveness()` returns
   `true` unconditionally, and overriding it crashes the backend (§12).
2. **The vreg is read at subregister granularity**, so it *has* subranges. A
   tuple always consumed whole takes the `!hasSubRanges()` path, gets the full
   mask, and cannot exhibit this. Many independent scalars merged by
   `SILoadStoreOptimizer` into one `S_LOAD_DWORDX16` and then read piecewise is
   precisely the shape that does have them.
3. **Pressure high enough that greedy splits** rather than assigns or evicts.
4. **At the copy-in index the parent's subranges under-report** relative to the
   lanes a later copy-out asks for. Per `83dca924c250`'s own message, children
   are built by inferring from the *original* live range and "may cover more
   lanes than the parent" — so a chain of splits can leave the parent's
   subrange structure coarser than what the child is later asked to produce.

Condition 2 is the one source code influences. Conditions 1, 3 and 4 are not
addressable from a kernel.

### 14.2 Source-level levers, and what is known about each

| lever | status |
|---|---|
| **Do not keep wide scalar tuples live across regions — let them be rematerialized at each use.** | **Directionally right, with corroboration.** The clean sibling object re-reads the strides from the kernarg segment instead of staging and spilling them (71 writelanes to the affected object's 105), and `-sgpr-regalloc=basic` independently produces the same shape. Not a guarantee: it makes splitting unnecessary rather than impossible. |
| Reduce scalar pressure generally | **Measured not to work** (§4). The stride hoist cleared 32 builds and newly infected 18, 14 of them while their spill count *fell*. |
| Split `i64` operands into `u32` halves | **Measured not to address the mechanism** (§11): losses at lone-dword granularity are directly observed. |
| Break kernarg contiguity so no `S_LOAD_DWORDX16` forms | **Untested, and fragile by construction** — it depends on an optimizer heuristic staying put, and nothing in the ABI expresses "do not merge these". |

### 14.3 Why a source-level workaround cannot be *verified*, even when it works

This is the part that matters more than the levers. Every item above changes the
shape of live ranges, and §4 measured what that does: the flagged set **churns**.
A change that clears the configurations you tested can silently infect
configurations you did not — that experiment cleared 32 and infected 18.

So "the detector is clean on our builds" does not establish that a source change
fixed anything; it establishes that it moved something. Verifying a source-level
workaround honestly means running the **general** predicate of §5 — not the
narrow one — over the *entire* build matrix, on every build, forever, because
the next unrelated change to register pressure can bring it back.

By contrast, `-sgpr-regalloc=basic` removes `SplitEditor` from the pipeline, so
there is nothing to churn. That asymmetry — a mitigation you can *reason* about
versus one you can only *sample* — is why the allocator switch is the
recommendation despite costing more code.
