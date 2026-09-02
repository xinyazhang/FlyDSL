# Prior-art search for the gfx950 SplitKit lost-subrange miscompile

Companion to `REPORT-gfx950-splitkit-lost-subrange.md`. Question answered here:
**has this defect already been reported upstream, and if so where?**

Searched 2026-09-02.

**Bottom line: no upstream duplicate exists.** Every neighbouring report found is a
*crash* (`llvm_unreachable`) or a *verifier failure*, all with reproducers, all fixed
and all already contained in our pinned revision. Nobody has reported the silent
miscompile. Recommendation: **file a new issue** against `llvm/llvm-project`, and
link it from `llvm/llvm-project#178867` (not a reopen — see §5).

---

## 1. Access method — what worked

| endpoint | result |
|---|---|
| `https://api.github.com` (unauthenticated REST + search) | **Works.** Used for everything below. |
| `https://github.com` HTML | Works (verified: `issues/199337` returned 300 KB with the issue text inline). Kept as fallback, not needed. |
| `gh` CLI (`/usr/local/bin/gh`) | **Unusable.** It is a wrapper that prompts on `/dev/tty`, which does not exist in this sandbox; `gh auth status` dies with `/dev/tty: No such device or address`. |

Unauthenticated API limits that matter to the next person: **core 60/hr**, **search
10/min**, counted per source IP — the core budget was already at 15/60 when this
started and had to be waited out once. Search has its own budget and was never the
binding constraint. Sleeping 8 s between search calls was sufficient.

A local read-only checkout at `/home/xinyazha/dockerhome/meff/llvm-project`
(`upstream/main` fetched) was used to settle every "has it landed / is it in the pin"
question without spending API calls; that is much the faster route.

## 2. Queries run

Against `repo:llvm/llvm-project`, both open and closed, issues and PRs:

`SplitKit defFromParent` · `SplitKit subrange miscompile` · `"lr-split"` ·
`gfx950 miscompile` · `v_writelane undefined register` · `refineSubRanges` ·
`getLiveLaneMaskAt` · `"SubRange for this mask not found"` ·
`AMDGPU SGPR spill undef miscompile` · `greedy register allocator subrange lane mask miscompile` ·
`defFromParent` · `buffer descriptor base undefined AMDGPU` · `"sgpr-regalloc=basic"` ·
`is:issue subregister undefined after register allocation AMDGPU` ·
`hipErrorIllegalAddress miscompile register allocator` ·
`is:issue label:llvm:regalloc miscompile` ·
`is:issue label:miscompilation register allocator subregister` ·
`is:issue SplitEditor buildCopy` · `is:issue AMDGPU wrong code greedy register allocator spill` ·
`is:issue subrange lane mask wrong code` · `is:issue "partial COPY" lanemask regalloc` ·
`83dca924c250` · `is:issue "v_writelane_b32" wrong` ·
`is:issue label:llvm:regalloc created:>2026-05-13` ·
`is:issue label:backend:AMDGPU label:miscompilation created:>2025-06-01` ·
`is:issue "undef" spill reload register allocation incorrect value` ·
`is:issue missing COPY after register allocation subregister never defined` ·
`is:issue "-sgpr-regalloc=basic" workaround` · `is:issue Triton AMDGPU spill undefined SGPR`

Plus repo-wide (all of GitHub): `SplitKit lane mask miscompile undefined subregister`,
and `repo:ROCm/llvm-project` for `SplitKit subrange` and
`register allocator miscompile spill undefined`.

Additionally: the full **timeline** of `#178867` (reopen / cross-reference events), the
**issue comments** of `#178867` and `#197014`, and the **review comments** of `#197014`.

## 3. Is our defect already filed? — No

Nothing describes a *silent* lost definition. The searches converge on a small,
well-understood cluster of **crashes** at the same source file, which is a different
failure mode: they abort the compiler, they came with reproducers, and they are fixed.

| # | title | state | dates | same defect? |
|---|---|---|---|---|
| **178867** | `[CodeGen] Crash in SplitKit::addDeadDef when child interval has subrange not in parent` | closed / completed | filed 2026-01-30, closed 2026-05-13 | **No — it is the crash our suspect commit suppressed.** Same function, same lane-mask asymmetry, opposite symptom. This is the direct ancestor of our bug, not a duplicate of it. |
| **199337** | `[AMDGPU] UNREACHABLE "SubRange for this mask not found" in SplitKit` | closed / completed | filed 2026-05-23, closed 2026-06-04 | **No.** Different call site: `SplitKit.cpp:402` is `getSubRangeForMaskExact` (reached from `extendPHIKillRanges`), not the `getSubRangeForMask` at ~451 that `83dca924c250` touched. Fixed by `4a77ce7c6f51` / PR #201263, "Remove premature empty subrange elimination", removing a `removeEmptySubRanges` call in `rewriteAssigned`. **That fix is already an ancestor of our pin** — verified locally. gfx1100, fuzzer-found, crash-on-valid. |
| **170298** | `[RISCV] Crash at -O2: SubRange for this mask not found` | **open** | filed 2025-12-02 | **No.** Also `SplitKit.cpp:402` via `extendPHIKillRanges` → `tryBlockSplit`. Still a crash, RISC-V RVV. Note the incidental agreement with our §9: XChy comments *"It seems specific to the greedy register allocator, since `-riscv-rvv-regalloc=basic/fast` works well."* Same allocator, same "basic avoids it" property, different symptom and different call site. Worth citing as corroboration, not as a duplicate. |
| **197733** | `[AMDGPU] machine verifier "No live segment at def" on <7 x i32> shufflevector` | closed | 2026-05-14 → 2026-06-17 | No. Verifier failure in `RenameIndependentSubregs`, fixed by #204091. Our compile *passes* the verifier — that is the point of the report. |
| **166657** | `[AMDGPU] register spill instructions are generated inside control flow with exec=0` | closed | 2025-11-05 | No. `exec`-mask placement, not lane-mask coverage. |
| **178259** / **175745** | `[AMDGPU] Wrong code at -O2` / `at -Os` | open | 2026-01-27 / 2026-01-13 | No evidence of relation, and both predate `83dca924c250` (2026-05-13), so they cannot be caused by the suspect commit. Kept here only so the "we looked at the AMDGPU wrong-code list" claim is checkable. |

Repo-wide, the only hit for the *combination* of terms that describes our defect is
**our own downstream issue, `ROCm/FlyDSL#1087`** ("gfx950: a dropped `s_mov_b64` in an
SGPR spill-staging chain puts an undefined register in a buffer descriptor base", open,
filed 2026-09-02 by xinyazhang). That is us; it is not upstream and does not count as
prior art.

`ROCm/llvm-project#280` ("Fix `SubRange for this mask not found ... SplitKit.cpp:404`",
closed 2025-06-23) is a downstream cherry-pick of an upstream *crash* fix to unbreak a
QUDA build. Different symptom again.

### Confidence in the negative

Moderate-to-high, with one honest caveat. GitHub full-text search does index issue
bodies, and the distinctive strings (`SubRange for this mask not found`,
`getLiveLaneMaskAt`, `defFromParent`, `lr-split`) each returned small, coherent result
sets rather than noise — which is evidence the index is working, not that it is
complete. What *cannot* be excluded is a report phrased purely in end-user terms
("my HIP kernel returns garbage on MI355") with no compiler-internal vocabulary; the
`label:backend:AMDGPU label:miscompilation` sweep since 2025-06-01 (12 issues, all
inspected by title, none matching) is the mitigation for that, but it only catches
issues someone bothered to label.

## 4. What happened to #178867 — read in full

Filed 2026-01-30 by **kdupontbe** against a downstream RISC-V 64 core ("arc5rpx"),
labelled `llvm:codegen`, `crash`. Closed **completed** 2026-05-13 by `83dca924c250`.

**No reopen. No follow-up comment. No cross-reference to any miscompile.** The timeline
contains exactly one cross-reference, to PR #197014 itself. The last human comment is
2026-05-07, six days before the fix landed. This is the place a duplicate would live and
it is empty.

Three things in the thread are worth having in the upstream report:

**(a) The maintainer proposed the narrowing as a guess, from an AI-generated patch.**
qcolombet, 2026-05-02:

> "Instead of fixing `addDeadDef` (we could still do better :)), could we fix
> `defFromParent` to use the `Parent` lane masks instead of the original lane mask?
> That way we should still have the invariant that children have only lanes from their
> parent. **(That may not be entirely true but that would be better I think.)**"
>
> "Here is the AI generated idea (**didn't try it**)"

The parenthetical is the maintainer flagging, in advance, exactly the invariant our
report shows does not hold.

**(b) The only correctness argument on record is AI-generated, from the reporter, and
covers a single RISC-V function.** kdupontbe's final comment (2026-05-07) contains a
section headed *"Why this is not a miscompilation"*, whose argument is that
`splitSeparateComponents` had already legitimately removed the missing lane from the
parent because it "had a dead def ... but no use within that range". Preceding it:

> "I let the AI analyse the debug output with/without the patch to double check"

and the validation offered is:

> "I am working on a downstream llvm 21.1.8 based version. ... Running the test suites
> did not reveal any miscompiles or other issues related to the patch."

That reasoning may well be right for *their* case. Our §2 MachineIR is the counterexample
where it is not: the lane the copy-in omits is subsequently *copied out and used*, so it
is not a dead def with no reachable use.

**(c) The commit shipped with no test and knowingly so.** qcolombet, 2026-05-12, asking
for review:

> "@arsenm, @preames, would be nice to have another pair of eyes on this, but if you
> don't have time, that's fine, I'll just merge it. That's a once in a blue-moon type
> of bug that has been here since forever."

Merged 2026-05-13, one day later.

## 5. Did any reviewer anticipate this? — No

PR **#197014** has exactly **three** review comments, all on 2026-05-12, and none of them
touches the narrowed mask's effect on `buildCopy`:

- arsenm, `llvm/lib/CodeGen/SplitKit.cpp`: a bare `` ```suggestion LaneBitmask LaneMask;``` `` (style).
- arsenm, line 429: *"I'm never sure what the point of using `getMaxLaneMaskForVReg` is; can't you just always use `getAll`?"*
- qcolombet, replying: *"Supposedly it's more accurate as in `getAll` will just mask everything whereas this one only masks the lanes that are covered by the related register class. In practice, most of the time it doesn't matter when LI has no subrange"*

So: **no reviewer raised the risk**, and there is nothing to quote as corroboration
from review. The nearest thing to a contemporaneous warning is the author's own
"(That may not be entirely true...)" in §4(a) — which is better material anyway, since it
is the author of the fix, not a bystander.

Two nits for the upstream write-up, since PR review is a defensible place to be precise:
the merged form of the `addDeadDef` skip is qcolombet's two-line
`if (PS == nullptr) continue;`, **not** the three-tier fallback kdupontbe proposed in his
2026-03-25 comment. Our report describes the merged form correctly. And #197014 was
merged with no approving review recorded — only the two arsenm comments above.

## 6. Related-report sweep — nothing else in the neighbourhood

- **Undefined SGPR spilled and reloaded:** no upstream issue exists. Zero hits across
  four phrasings.
- **SplitKit subrange / lane-mask bugs:** the cluster in §3, all crashes.
- **gfx9xx / gfx950 miscompiles of this shape:** none. The `gfx950 miscompile` query
  returns nine items, all PRs for unrelated arithmetic/lowering fixes (FDOT2 subnormals,
  iDot4 chain walking, `isCanonicalized` sNaN, i128 alignment).

## 7. Cross-check of report §7 against the local tree

Confirmed independently against the local checkout. §7 needs no correction; the notes on
re-measuring below are about keeping it auditable, not about fixing it.

- `83dca924c250` **is** an ancestor of the pin `e2a39f504fee` (pin dated 2026-07-23).
- Every candidate "maybe it's already fixed" commit is **also already in the pin**:
  `4a77ce7c6f51` (#201263, the #199337 crash fix), `734d5690bc6e` (#204091,
  RenameIndependentSubregs duplicate subranges), `a896e12c5e70` (#211031, Rematerializer
  full-mask subranges), `b03d16c80faf` (#195023, RegisterCoalescer `pruneSubRegValues`).
  None of them is a fix for our defect and none is missing from our build.
- **`SplitKit.cpp` is unchanged since the pin apart from one NFC commit — §7 stands as
  written.** Measured against `upstream/main` at `810dc356352a` (2026-09-02):

  ```
  $ git log --oneline e2a39f504fee..upstream/main -- llvm/lib/CodeGen/SplitKit.cpp
  745e57793431 [SlotIndexes] Use analysis block numbers (NFC) (#215171)
  ```

  which is the NFC hoist §7 already names, and nothing else.
- **Still live on trunk at that ref:** `SplitKit.cpp:451-453` is still
  `findSubRangeForMask(...)` / `if (PS == nullptr) continue;`, and line 654 is still
  `LaneBitmask LaneMask = getLiveLaneMaskAt(Edit->getParent(), UseIdx, MRI);`.
- **Pin the ref when you re-measure.** `upstream/main` moved from `83e1178daa12` to
  `810dc356352a` between §7's measurement and this one, so an unpinned "unchanged since
  main" claim rots within days — quote the ref and its date, as §7 now does.
  One commit in the *adjacent* machinery landed after the pin and is worth naming so the
  scope of "unchanged" is auditable: `85e73b167487` *"CodeGen: Pass instruction to
  `LiveRangeEdit::useIsKill`"* (#219466, 2026-08-28) touches only
  `llvm/include/llvm/CodeGen/LiveRangeEdit.h` and `llvm/lib/CodeGen/LiveRangeEdit.cpp`.
  It is NFC — it replaces a `MachineOperand` parameter with the instruction that operand
  was used to recover — and it does **not** touch `SplitKit.cpp`.

## 8. Recommendation

**File a new issue** on `llvm/llvm-project`. Nothing existing can absorb this report.

Do **not** reopen #178867. It is a distinct, correctly-fixed crash; reopening it would
conflate the two and lose kdupontbe's reproducer. Instead:

1. File new, titled around the mechanism rather than the target — *"SplitKit/RegAllocGreedy:
   copy-in lane mask is a strict subset of copy-out lane mask, leaving a subregister
   undefined (miscompile)"*. Labels `llvm:regalloc`, `llvm:codegen`, `miscompilation`.
2. **Comment on #178867** with a one-paragraph pointer to the new issue, since that
   thread is where anyone investigating this code will land, and since our §8.2 outcome
   analysis is directly relevant to whether `83dca924c250` was the right shape of fix.
3. Cc the three people already in this code: **@qcolombet** (author of `83dca924c250`),
   **@arsenm** (only reviewer, AMDGPU owner), **@kdupontbe** (original reporter — worth
   asking whether their downstream test suites would flag our §5 general predicate).
4. Mention **#170298** in the new issue as an *open, unfixed* sibling: same pass, same
   `-regalloc=basic` escape hatch, still crashing on RISC-V. If the underlying invariant
   ("children have only lanes from their parent") is genuinely violated rather than
   merely unenforced, #170298 and our miscompile may share a root cause upstream of
   `defFromParent`. Do not assert that — offer it.
5. Lead with the two facts that make this filing worth a maintainer's time and that
   distinguish it from the crash reports: **`-verify-machineinstrs` and `-verify-regalloc`
   both pass**, and **12 of the affected objects come from Triton, not our frontend**
   (report §5), so this is not a downstream-DSL problem. The §5 MachineVerifier proposal
   — *no SGPR spill may name a source register with no reaching definition* — is the
   concrete ask, and it directly answers the "we could still do better :)" that
   qcolombet left on the table in #178867.
6. Carry over the caveat in report §8.2 verbatim. We have **not** run the revert. Claiming
   `83dca924c250` as the cause rather than the strongest-fitting hypothesis would be
   overclaiming, and the churn measured in §4 means a clean detector run after a revert
   would not settle it either.
