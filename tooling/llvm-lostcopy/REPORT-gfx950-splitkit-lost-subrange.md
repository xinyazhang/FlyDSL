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

Two detectors were run over 1054 objects from one build tree:

| predicate | flagged |
|---|---|
| undefined at spill → reloaded → used as a buffer descriptor base → dereferenced | 42 |
| **undefined at spill at all** (no `v_writelane_b32` may name an SGPR with no reaching definition) | **166** |

Every hit of the first is a hit of the second. The extra 124 include kernels
that the narrow predicate calls clean. One was hand-verified: `v251` lanes 4, 5
and 7 are each written exactly once, from `s68`/`s69`/`s71`, whose defining
`s_load_dwordx8 s[64:71]` is at `0x2560` — *after* the spill at `0x2514` — and
those lanes are reloaded more than twenty times.

By kernel family, over the same 1054 objects:

| family | frontend | objects | lost definition | reaches a descriptor base |
|---|---|---|---|---|
| `flyc_bwd_dkdv` | MLIR/FlyDSL | 216 | **114** | **42** |
| `flyc_bwd_dq` | MLIR/FlyDSL | 216 | 21 | 0 |
| `flyc_attn_fwd` | MLIR/FlyDSL | 216 | 19 | 0 |
| `bwd_preprocess` | **Triton** | 144 | **12** | 0 |
| `bwd_preprocess_varlen` | **Triton** | 144 | 0 | 0 |
| `_gemm_afp4wfp4_preshuffle_kernel` | **Triton** | 112 | 0 | 0 |
| `debug_simulate_encoded_softmax` | Triton | 6 | 0 | 0 |

**This is not specific to one frontend.** Twelve Triton-compiled objects carry
the same defect. The bug is in the shared AMDGPU backend, below any language
front end; what varies is exposure. The one family that reaches a *dereferenced*
descriptor base is also the one with the most 64-bit kernarg strides (21 `i64`),
which is what produces the wide merged tuples the split operates on — so
frontends with lower scalar pressure are less likely to be *corrupted*, not less
likely to be *miscompiled*.

So the narrow number understates the defect: it counts only the objects where a
lost definition happens to reach an address computation. **The general predicate
is the right one, and it is cheap** — it is a reaching-definition question over
the spill's source operands, and it separates this defect from the benign
dead-path `undef` reads LLVM emits legitimately (a bare "reads an undefined
SGPR" scan flags about a third of any tree).

Suggested as a MachineVerifier check: *no SGPR spill — of any strategy — may
name a source register with no reaching definition.* That predicate would have
caught all 42 at compile time rather than at `hipErrorIllegalAddress` in a test
sweep months later.

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

Verified by hand against `upstream/main` at `83e1178daa12`, which is **5484
commits** (303 touching `llvm/lib/Target/AMDGPU/`) ahead of the pin:

```
$ git diff --stat e2a39f504fee upstream/main -- \
      llvm/lib/CodeGen/SplitKit.cpp llvm/lib/CodeGen/SplitKit.h \
      llvm/lib/CodeGen/LiveInterval.cpp llvm/include/llvm/CodeGen/LiveInterval.h
 llvm/lib/CodeGen/SplitKit.cpp | 5 ++---
 1 file changed, 2 insertions(+), 3 deletions(-)
```

and that single hunk is a pure NFC hoist of `getBlockNumbered` out of
`splitLiveThroughBlock`. `SplitKit.h`, `LiveInterval.cpp` and `LiveInterval.h`
are byte-identical. `buildCopy`, `buildSingleSubRegCopy`, `getLiveLaneMaskAt`,
`addDeadDef`, `defFromParent` and `transferValues` are therefore semantically
unchanged across the whole range, and no new MachineVerifier check for
undefined subranges landed.

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

**Confidence: moderate, and this is inferred rather than measured.** The
mechanism, the data path and the timing all fit, and the code is unchanged on
trunk. It has *not* been tested by reverting. The decisive experiment is small:
revert `83dca924c250` on top of the pin, rebuild, and re-run §6. If the restored
`llvm_unreachable` fires, the defect is exactly localized and the revert is a
55-line one-file candidate fix. If it does not fire and the copy is still lost,
the suspect is wrong and the search should move to `transferValues` /
`extendPHIKillRanges`.

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
removing it, which is why the general predicate in §5 matters.

**Cost, measured statically and not yet benchmarked on hardware:** instructions
5129 → 6868, driven almost entirely by `v_readlane_b32` 149 → 1419, of which 109
land inside the MFMA span (0 in the default build). `sgpr_spill_count` 105 →
114; `vgpr_count` unchanged at 394. Basic RA rematerializes the strides from the
kernarg segment at each use — the same shape as the known-clean sibling object,
which is independent corroboration that the rematerializing shape is the safe
one. `-vgpr-regalloc=basic` on top adds 26 VGPRs for no gain; do not use it.

## 10. Artifacts

`repro.py`, `drive.py` (compiles the module under arbitrary `cl::opt` settings
via ctypes, avoiding an unsound old-value read), `spillscan.py` + `undefscan.py`
(narrow predicate), `lostdef.py` (general predicate), `pop.tsv` (the 1054-object
scan), the affected and clean `.hsaco`, and the flagged-hash sets for the knob
sweeps.

## 11. Allocator inventory, and workarounds that do *not* work

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
