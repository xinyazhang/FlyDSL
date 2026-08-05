# gfx1201 SDPA readability plan

Response to `sdpa-readability.md`. That review raises ~25 items against
`flash_attn_func_gfx1201_aiw.py` (3296 lines; the interface beside it is 397).

**The items are not independent.** Most of them are symptoms of four root
causes, and fixing a root cause retires several review items at once rather
than one each. That is what this plan is organised around, because the
alternative -- walking the review top to bottom -- would touch the same code
repeatedly and gate each touch separately.

| root cause                                            | review items it generates                                                     |
| ----------------------------------------------------- | ----------------------------------------------------------------------------- |
| **A.** `fx.Index` used for signed sequence arithmetic | `seqlen_q_v`, `tile_start` as u64, `_scmp_i32`/`_ssel_i32`/`_smin`/`_smax`, `shard_qk_off` casting |
| **B.** One file is both build-time policy and kernel  | the five tuning constants, all four env vars, `_FP_MODE`/`fm_fast`, the 26-parameter signature |
| **C.** No module for code the backward pass will share | `composed_tensors.py`-style extraction, Q preload, gSWA regions, cross-shard reduction, the load/store helpers |
| **D.** Comments that state a *false* reason           | `_scmp_i32`'s docstring -- and this is the third instance, see §1.4            |

---

## 0. Status and work breakdown

### 0.1 Where this stands

| phase                      | state       | commit     |
| -------------------------- | ----------- | ---------- |
| P5 groundwork              | **landed**  | `49764d29` |
| P1 free corrections        | not started | --         |
| P2 policy extraction       | not started | --         |
| P3 type discipline         | not started | --         |
| P4 investigations          | not started | --         |
| P5 helper modules (rest)   | not started | --         |

`49764d29` landed the import shim (`gfx1201_standalone.py`) and moved the first
helper (`wmma_acc` -> `kernels/common/mma/wmma_ops.py`, generalised to dispatch
on the operand element type). It is the proof that the Phase 5 path works
end-to-end -- bitwise-identical across f16/bf16 x causal/non-causal -- and it
is deliberately *one* helper, ahead of the rest of Phase 5, because the rest
depends on Phases 3 and 4.

### 0.2 Work breakdown

Every row is one commit. `gate` is what must pass before it lands; the
protocol behind each gate name is §2. Effort: **S** under an hour, **M** a few
hours, **L** a day or more.

Every row runs the **tier-1 codegen fingerprint** (§2.3, 12 s) and nothing
more unless the fingerprint moves (§2.3.1). The `gate` column lists what is
required *beyond* tier 1, and the `fp?` column is whether the fingerprint is
*expected* to move -- "no" means an unchanged ISA is the pass condition and a
change is a defect, not a reason to benchmark.

Phase boundaries add a tier-2.9 fast screen (11 s) against the fixed pre-P1
baseline. Tier 3 runs once, after P5.5.

| id     | task                                                              | files                                          | gate                     | eff | risk | fp? | after   |
| ------ | ------------------------------------------------------------------ | ------------------------------------------------ | -------------------------- | --- | ---- | ---- | ------- |
| P1.1   | Document `K_SUB_N`, `WMMA_LANE_K`; audit every other bare constant | aiw                                            | none (comments)          | S   | none | no  | --      |
| P1.2   | Fix `_scmp_i32`'s false docstring (§1.4)                          | aiw                                            | none (comment)           | S   | none | no  | --      |
| ~~P1.3~~ | ~~Drop the `Vec` alias~~ **dropped** -- `Vec` is the majority spelling (17 files vs 9), see §1.1 | -- | -- | -- | -- | -- | -- |
| P1.4   | Delete `_ssel_i32`; use `cond.select(a, b)` (§1.2)                | aiw                                            | bitwise                  | S   | low  | yes | --      |
| P1.5   | Drop `fx.Index(<const>)` casts; sweep for siblings (§1.3)         | aiw                                            | bitwise                  | S   | none | no  | --      |
| P1.6   | Rename `_REVERSE_Q_TILES` -> `_LPT_TILE_ORDER` (§1.5)             | aiw                                            | bitwise                  | S   | none | no  | --      |
| P1.7   | Rename `tile_start` -> `start_k`, `q_start` -> `start_q` (21 + n sites) | aiw                                       | bitwise                  | S   | low  | no  | --      |
| P2.1   | Move 6 tuning functions/constants to the interface (§4.1)         | aiw, interface                                 | schedule-diff + bitwise  | M   | low  | no  | --      |
| P2.2   | Env vars -> build knobs, incl. `fp_mode` (§4.2)                   | aiw, interface                                 | schedule-diff + bitwise  | M   | med  | no  | P2.1    |
| P2.3   | Introduce `FmhaProblem` / `FmhaSchedule` dataclasses (§4.3)       | aiw, interface, tests                          | schedule-diff + bitwise  | M   | low  | no  | P2.2    |
| P2.4   | Make the interface the only producer of a `FmhaSchedule`          | aiw, interface                                 | schedule-diff            | S   | low  | no  | P2.3    |
| P3.1   | **Inventory**: classify all 100 `fx.Index` sites; publish the 64-bit list | plan (artifact)                        | review                   | M   | none | no  | --      |
| P3.2   | Narrow sequence-space quantities to `fx.Int32` (§5.2a)            | aiw                                            | bitwise + gSWA-90 + perf | L   | **high** | yes | P3.1, P2.3 |
| P3.3   | Delete `_scmp_i32`/`_smin_i32`/`_smax_i32`                        | aiw                                            | bitwise                  | S   | low  | yes | P3.2    |
| P4.1   | ~~`_pointer_to_llvm_ptr` still needed?~~ **answered** -- see §7.1 | --                                             | --                       | --  | --   | yes | done    |
| P4.2   | Can `fx.add_offset` replace LDS `ptrtoint`+`addi`?                | spike                                          | working example or a no  | S   | none | yes | --      |
| P4.3   | Can the layout API replace `coop_load/store_*`?                   | spike                                          | example + perf number    | M   | none | yes | --      |
| P5.1   | Duplication audit vs `common/*` and `flash_attn_utils` (§7.1)     | plan (artifact)                                | review                   | S   | none | no  | --      |
| P5.2   | Pointer + global load/store -> `common/mem_ops.py`                | aiw, common/mem_ops                            | bitwise (per helper) + perf | M   | med  | yes | P5.1, P4.1 |
| P5.3   | LDS load/store -> `common/mem_ops.py`                             | aiw, common/mem_ops                            | bitwise + perf           | M   | med  | yes | P5.2, P4.2 |
| P5.4   | Interval algebra + `div_rd` -> `common/utils.py`                  | aiw, common/utils                              | bitwise + tier 1         | S   | low  | yes | P3.3    |
| P5.5   | gSWA regions, Q preload, cross-shard reduction, LSE addressing, `_decode_side` -> `attention/fmha_common.py` | aiw, fmha_common | bitwise + perf           | L   | med  | yes | P3.3, P4.3 |

**Critical path:** `P3.1 -> P3.2 -> P3.3 -> P5.4/P5.5`. Everything in P1 and P4
is independent and can land at any time; P2 gates only on itself.

### 0.3 Commit granularity: one function per commit

**A commit refactors exactly one function, whatever its size.** Not one module,
not one "area", and explicitly not a bundle assembled to make the diffs
comparable. A 400-line change to one function is easier to reason about than a
40-line change spread over six unrelated ones, and only the first is a useful
bisect point.

The reason this is a rule rather than a preference is §2.3.2: the perf harness
resolves about 1% typically and 5% at the short-kernel configs. A regression
smaller than a wide commit's blast radius cannot be attributed by reading the
diff, so the commit boundary *is* the attribution mechanism. Wide commits do
not get harder to bisect -- they make bisect useless, because the answer it
returns is "one of these forty changes".

Three consequences worth stating, because each changes the task list:

**Thin wrappers still get their own commit.** `_split_ptr`,
`_load_global_half_vec`, `load_global_f16xN` are five lines each. Moving one and
leaving its caller pointing at the new location is a valid, complete, testable
change, so it is a commit. Tiny commits are the cheap case, not the wasteful
one -- tier 1 costs 12 s and most of these will not move the fingerprint at all.

**Code that is not yet a function needs two commits, not one.** The gSWA region
split (line 1821), the Q preload (1696) and the cross-shard S reduction (2121)
are *inline blocks*. Each becomes:

1. **extract in place** -- lift the block into a local function, same file, no
   import changes. This is the commit with the sharpest gate: pure code motion,
   bitwise-identical output, and the fingerprint should not move.
2. **move** -- relocate the now-existing function to its shared module.

Splitting these is not bureaucracy. Extraction is where a scoping mistake
happens (a captured variable that should have been a parameter), and relocation
is where an inlining change happens. They fail differently and should be
bisectable apart.

**Phase 3 has no function to refactor, so its unit is one *quantity*.** Narrowing
`seqlen_q_v` from `fx.Index` to `fx.Int32` is a change to a value's type that
propagates to every consumer; there is no single function that owns it. The
same principle applies -- one logical change per commit, regardless of how many
lines it touches -- with the quantity standing in for the function. P3.2 is
therefore one commit per quantity: `seqlen_q_v`/`seqlen_k_v`, then the tile and
row origins, then the shard offsets, then the window bounds, then whatever
P3.1's classification turns up.

### 0.4 The commit list, derived

The groups in §0.2 expand as follows. Each bullet is one commit.

| group | commits | one per                                                                                     |
| ----- | ------- | --------------------------------------------------------------------------------------------- |
| P3.2  | ~5      | quantity: `seqlen_?_v`; tile/row origins; shard offsets; window bounds; P3.1 remainder      |
| P3.3  | 4       | `_scmp_i32`, `_ssel_i32`, `_smin_i32`, `_smax_i32`                                          |
| P5.2  | 8       | `_pointer_to_llvm_ptr`, `_pointer_load`, `_pointer_store`, `_global_load_tr_v8`, `_split_ptr`, `_load_global_half_vec`, `_store_global_half`, `load_global_f16xN` + `load_global_v8f16` |
| P5.3  | 2       | `_lds_load_v8`, `_lds_store_vx`                                                             |
| P5.4  | 3       | `_sdiv_rd`, `_smin_i32`, `_smax_i32` (post-P3.3 survivors)                                  |
| P5.5  | ~12     | gSWA extract + move; Q preload extract + move; cross-shard extract + move; `_decode_side` split into 3; `_decode_side` move; `_seqinfo_at` move; LSE addressing extract + move |

That takes the plan from 21 tasks to roughly **50 commits**. That is the
intended outcome: at 12 s of tier-1 gate each, the whole sequence costs about
ten minutes of verification, and every one of them is a bisect point.

### 0.5 Bisecting a regression, when one appears

**Do not `git bisect run` the benchmark.** With a 5% floor at the short-kernel
configs, a noisy predicate flips good/bad near the threshold and binary search
converges on the wrong commit with total confidence. The repository's
`/bisect-perf-regression` skill drives a benchmark command, which is the right
tool for a large regression and the wrong one for a 3% drift.

Bisect the **fingerprint** instead. It is deterministic (verified), so it is a
sound predicate:

```
git bisect start <bad> <good>
git bisect run python3 kernels/attention/codegen_fingerprint.py --expect <baseline.json>
```

That localises the commit where the *emitted code* changed, in
`log2(n) x 12 s` -- under a minute across fifty commits. Then run tier 2.9 on
that one commit to confirm it is also where the *time* changed. Two-stage,
because the deterministic question and the noisy question deserve different
tools.

The failure mode to know about: if several commits change the fingerprint and
only one changes the timing, the fingerprint bisect returns the first of them.
That is why the `fp?` column exists -- the twelve tasks expected not to move it
narrow the search before it starts.

### 0.3 Definition of done, per phase

| phase | done when                                                                                          |
| ----- | ---------------------------------------------------------------------------------------------------- |
| P1    | no bare magic constant, no false comment, no local re-spelling of a house idiom                     |
| P2    | the AIW file contains no `os.environ`, no tuning table, and takes two frozen dataclasses             |
| P3    | `grep -c "fx.Index("` in the AIW file counts only addressing and grid ids, and the 64-bit list exists |
| P4    | each question has a working example or a recorded measurement saying no                              |
| P5    | no helper in the AIW file has a counterpart elsewhere in the tree                                    |

---

## 1. The review's questions, answered

Five items are questions rather than requests. They are answered here because
three of them change what the plan should do, and one of them turns out to be a
bug rather than a style point.

### 1.1 `Vec.make_type(8, fx.Float32)` -- is there a built-in for *the type*?

Three separate facts, and the useful one is the third:

- **No predefined alias exists.** There is no `fx.v8f32` or equivalent
  anywhere in `flydsl.expr`.
- **`fx.VectorAlias(fx.Float32, 8)` would mint one** -- it returns a named
  class `Float32x8` whose `ir_type` is lazily `Vector.make_type(8, Float32)`.
  It is used in **zero** places across `kernels/`, `tests/`, `examples/` and
  `python/`.
- **`Vector.make_type(n, dtype)` is the house convention**, under both
  spellings. Counted across `kernels/`: **17 files import `Vector as Vec`**
  (all four attention files that use vectors, plus most of gemm and moe) and
  **9 spell it `fx.Vector`**. The alias is the *majority*.

So the AIW file is already doing what most of the repo does, and the answer to
"shouldn't this be built in" is that the spelling *is* the built-in one.

**Action: none.** An earlier draft of this plan had a P1.3 to drop the `Vec`
alias, on the stated grounds that "every other kernel says
`fx.Vector.make_type`". That was wrong -- it generalised from a partial grep
that happened to surface only the minority spelling. Changing this file would
have moved it from the majority convention to the minority one and split the
attention directory 3-1, which is the opposite of the intent. Dropped, and
recorded here rather than silently, because it is the fourth false claim this
work has turned up and the first one that was in the plan itself.

**Action (Phase 5):** reconsider `VectorAlias` for the shared helper module,
where it earns its keep in a way it does not here. A helper that today takes a
raw `ir.Type` parameter (`_load_global_half_vec(..., vec_type)`) could take a
`Float16x8` *class* instead, which is both self-describing at the call site and
checkable. Being unused repo-wide is a reason to be careful, not a reason not
to -- but it makes this a proposal to try in one helper and look at, not a
blanket conversion.

### 1.2 `x = a if flag else b` -- "WTF, we still can't write this?"

The ternary itself is not fixable: Python's conditional expression calls
`__bool__` on the condition, which a value that does not exist until tracing
cannot answer. Every embedded DSL hits this.

But the review's actual point is readability, and there `_ssel_i32` is
indefensible -- it matches **nothing** in the codebase. The two established
spellings are:

```python
x = arith.select(cond, a, b)     # mfma_epilogues, splitk_hgemm, silu_and_mul_fq
x = cond.select(a, b)            # custom_all_reduce_kernel, moe/.../gemm1.py
```

`cond.select(a, b)` is the one that reads closest to a ternary and keeps the
operands in source order, and this file already uses it directly in the dropout
path (`_keep[_r].select(...)`). So the file is *inconsistent with itself*, not
just with the repo.

**Action:** delete `_ssel_i32`, use `cond.select(a, b)`. Under root cause A the
`fx.Int32(...)` coercion it also performs stops being necessary (§5), so this
is a delete rather than a replace.

### 1.3 `shard_qk_off = shard_id * fx.Index(QK_SLICE)` -- "requires explicit casting?"

**No.** Python ints coerce on both `fx.Index` and `fx.Int32` operands --
verified on device: `tid * 96` and `fx.Int32(tid) * 96` both compile and give
the right answer. The cast is noise.

**Action:** `shard_id * QK_SLICE`. Sweep for the same pattern elsewhere.

### 1.4 `_scmp_i32` exists to work around something that is not true

Its docstring says:

> The `<` and `>` overloads on fx.Int32 pick unsigned, which on a negative
> window silently compares against something enormous

**This is false.** `fx.Int32` is declared `signed=True`
(`expr/numeric.py:716`), and `_comparison_op` selects the signed predicate
table unless `signed is False`. Dumping the IR for a plain `v > thr` on two
`fx.Int32` values emits `arith.cmpi sgt`.

The helper is not *useless* -- it coerces both operands to `fx.Int32` first,
which does matter when one of them is an `fx.Index` (`signed=False`, so *its*
comparisons are genuinely unsigned). But that is a different justification, and
it points at root cause A rather than at the operators.

This is the **third** instance of a comment asserting a false reason, after the
`q_row_tiles` gate (`head_dim != 64` justified by an argument that only rules
out `> 64`) and `keep_mask`'s docstring (same wrong claim about `fx.Int32`,
fixed in c4221b10). All three read fine in isolation; all three were only
caught by checking the code against the claim.

**Action:** §2.5 makes "verify the stated reason" part of the review pass
rather than a thing done when someone happens to wonder.

### 1.5 `_REVERSE_Q_TILES` should be orthogonal to `CAUSAL` -- I disagree

The coupling is deliberate and the code says why:

> Longest-processing-time-first: under causal, cost grows with q_tile, so
> dispatching the expensive tiles first leaves only cheap ones to fill the tail.

Non-causal tiles all cost the same, so reversing them is a permutation with no
load-balancing content. The `and CAUSAL` is the knob's *definition*, not a
restriction on it.

What the review is actually reacting to is the **name**: `_REVERSE_Q_TILES`
describes a mechanism, so gating it on `CAUSAL` looks arbitrary. Named for its
purpose the gate reads as a tautology.

**Action:** rename to `_LPT_TILE_ORDER` (longest-processing-time-first) and
fold the causal condition into where it is *resolved*, not where it is used, so
the use site is a single flag. Behaviour unchanged. If someone wants it truly
orthogonal, that is a measurement (does reordering help non-causal at all?) and
is listed in §7 rather than assumed.

---

## 2. How this is gated

This kernel is latency-bound, and this session has already produced two
false-confident performance conclusions and three false comments. The
verification rules below are what the plan actually costs.

### 2.1 Pure refactors gate on **output-bitwise identity**

Renames, moves, dataclass packing and type changes must leave every output bit
unchanged. Integer index arithmetic is exact, so a correct `Index -> Int32`
change cannot move a single output bit even though it changes the emitted ISA.

The existing harness covers this: 429 tests across four suites, plus the 90
window x shape gSWA sweep, plus the dense/varlen bitwise comparisons.

### 2.2 What bitwise does *not* prove

Recorded in `sdpa_lore_gfx1201.md` and re-learned twice: bitwise identity says
nothing about whether work was *skipped*, and adding or removing a single FP op
breaks bitwise even when it is mathematically a no-op (`reassoc`/`contract`
re-associate differently). So:

- No phase here may add or remove an FP operation. If one seems necessary, it
  stops being a readability change and gets its own measurement.
- Phases that could change scheduling (§4, §6) need a perf gate as well.

### 2.3 How often performance is actually checked

Per-task gates alone are not enough, for two reasons that only show up when you
count them.

**Bitwise identity does not imply codegen identity.** It says the numbers came
out the same, not that the same instructions produced them. Deleting a helper
that performed a redundant coercion (P1.4, P3.3) or moving one across a module
boundary (P5.2 to P5.5) can leave every output bit untouched and still change
what the scheduler emits.

**And every gate compares against the previous commit.** Fifteen commits each
losing 0.7% all pass individually and compound to 10%. Nothing in a
per-task gate can see that, because at no point is any single step out of line.

So four tiers, cheapest first, and **a tier only runs if the one below it says
something changed**:

| tier | what                                                                | when                                          | measured cost |
| ---- | --------------------------------------------------------------------- | ----------------------------------------------- | --------------- |
| 1    | **codegen fingerprint** -- VGPR, scratch, instruction count, ISA SHA at 3 configs | **every commit**                | **12 s**      |
| 2    | interleaved A/B on the configs the change touches                   | only when tier 1 moves                        | ~11 s         |
| 2.9  | **fast screen** -- N=4096 non-causal, head_dim 64/80/128/192/256    | phase boundaries, and any task with a tier-1 move | **11 s**  |
| 3    | full ladder -- 13 head_dims x causal x N in {1024, 4096}            | **once, at the end of the plan**              | **44 s** (4 GPUs) |

### 2.3.1 If the ISA is unchanged, there is nothing to measure

**This is the rule that makes the plan cheap.** A codegen fingerprint that has
not moved is proof the emitted instruction stream is identical, and identical
instructions cannot run at different speeds. So tier 1 is not a screen that
*suggests* skipping the benchmark -- it *decides* it.

Phase 2 is the clearest case. Moving tuning constants to the interface,
replacing env vars with knobs, and packing 26 parameters into two dataclasses
are all host-side reorganisation: the same values reach the same builder and
the same kernel is emitted. The expected tier-1 result is byte-identical ISA at
all three configs, and if that holds, **P2.1 through P2.4 run no benchmark at
all**. If it does *not* hold, that is the interesting outcome -- a "cosmetic"
change altered codegen, which means it was not cosmetic, and it gets a tier 2.9
before anyone argues about why.

The same applies to most of Phase 1. P1.1 and P1.2 are comments; P1.7 is a
rename. Those cannot move the fingerprint, and if one does, the commit is wrong
rather than the tool.

### 2.3.2 The harness, and what it can actually resolve

`perf_ab.py` runs one long-lived worker process per (revision, GPU) with each
checkout on its own `sys.path`, and the parent alternates requests between
them. Two git revisions cannot be interleaved inside one interpreter -- the
kernel modules are imported by name and would cross-contaminate -- but they can
be interleaved at ~1 second granularity across two processes, which is what
matters. The alternative, running each revision's whole sweep in turn, puts
minutes between the two measurements of a point and is what produced the 19%
outliers in §11.5.

**Run the self-test before trusting a number.** `--base X --head X` compares a
revision against itself, so whatever ratio it reports is the harness's own
resolution. Measured here:

| tier | GPUs | wall  | worst self-test ratio | where                |
| ---- | ---- | ----- | ----------------------- | ---------------------- |
| 2.9  | 4    | 10.6 s | 0.952                  | head_dim 192          |
| 2.9  | 1    | 11.6 s | 0.993                  | head_dim 64           |
| 3    | 4    | 44.2 s | 0.977                  | head_dim 48, N=1024   |
| 3    | 1    | 69.8 s | 0.941                  | head_dim 48, N=4096   |

Two things fall out, and the second one is a trap avoided:

- **Four GPUs help tier 3 and not tier 2.9** (1.6x against 1.1x). At five
  configs the run is dominated by process startup, which parallelises anyway.
- **The noise is not caused by GPU concurrency, it tracks short kernels.** The
  worst point is head_dim 48 at N=1024 or head_dim 192 -- launch-overhead
  dominated configs -- and worst-case lands at 0.94 to 0.95 whether one GPU is
  busy or four. It would have been easy to read the first two rows alone and
  conclude "concurrency costs resolution"; the tier-3 rows say otherwise.

So: **a config's self-test ratio is its resolution floor, and a "regression"
smaller than that is not a measurement.** Judge each config against its own
floor rather than a single global threshold, and re-run rather than believe a
one-off at head_dim 48 or 192.

### 2.4 Perf gates are **interleaved, same-process A/B**

Sequential before/after runs of this kernel have a noise floor of about 5% with
outliers to +19% -- measured, on 48 points whose kernel selection was *provably
identical* (§11.5). Any phase claiming "no perf change" must show it with alternating
reps in one process, reporting whether the two sides' rep ranges separate.

Where the change is structural rather than numeric, prefer the stronger form:
enumerate the inputs and diff the resolved configuration. That is how the
`q_row_tiles` change established no-regression across 52 triples in a second,
and it is available to §4.

### 2.5 Every touched comment gets its stated reason checked

Not "is this comment clear" but "is this comment *true*". Three false ones in
one session, each surviving multiple readings. Concretely: any comment
asserting behaviour of the DSL, the hardware, or a bound gets a one-line check
(dump the IR, read the declaration, run the case) before it is left in place.

---

## 3. Phase 1 -- free corrections

No behaviour change, no codegen change possible for the comment-only items.
Do these first: they are what makes the file readable enough to do the rest.

| item                        | change                                                                              | review ref     |
| --------------------------- | ------------------------------------------------------------------------------------ | -------------- |
| `K_SUB_N = 32`              | document: it has **no comment at all** today                                        | Documentation  |
| `WMMA_LANE_K = 8`           | same                                                                                | Documentation  |
| `_scmp_i32` docstring       | replace the false claim with the real reason (coerces `Index` operands)              | §1.4           |
| `Vec` alias                 | drop; every other kernel says `fx.Vector.make_type`                                 | §1.1           |
| `_ssel_i32`                 | delete, use `cond.select(a, b)` -- the file already does elsewhere                  | §1.2           |
| `fx.Index(QK_SLICE)`        | delete the cast, sweep for siblings                                                  | §1.3           |
| `_REVERSE_Q_TILES`          | rename `_LPT_TILE_ORDER`, resolve the causal condition at definition                | §1.5           |
| `tile_start` (21 sites)     | rename per axis -- see below                                                        | Coding Style   |

**`tile_start` is the one worth spelling out.** The review is right that it is
ambiguous, and the ambiguity is real rather than aesthetic: the name is used for
a KV-column origin in the addressing helpers (`_kv_addr(tbase, toff,
tile_start, ...)`) and reads as if it could be a Q-row origin, which is what
`q_start` is two lines away. Rename to `start_k` (KV/N axis) and keep `q_start`
as `start_q`, so the pair is symmetric and the axis is in the name.

**Gate:** bitwise identity on the full suite. Renames cannot change codegen; if
bitwise fails, the rename was wrong.

**Cost:** small. **Risk:** near zero.

---

## 4. Phase 2 -- get build-time policy out of the kernel file

Root cause B. The review's framing is exactly right: *"The AIW file should only
care about the correctness and functionality."*

### 4.1 Tuning policy moves to the interface

| moves                     | at line |
| ------------------------- | ------- |
| `_PREFETCH_MIN_HEAD_DIM`  | 192     |
| `default_prefetch_dist`   | 302     |
| `q_tiles_per_block`       | 312     |
| `resolve_shards`          | 342     |
| `default_block_m`         | 363     |
| `default_block_n`         | 371     |

The interface already owns `_BP_MIN_HEAD_DIM`, `_Q_ROW_TILES_2_HEAD_DIMS` and
`_BLOCK_DMODEL_LADDER`, so this consolidates a policy layer that is currently
split across two files with no principle deciding which half lives where.

**Note the direction of the dependency.** The interface imports from the AIW
file today (`default_block_m as aiw_block_m`). After the move the AIW file must
not import them back, or the split is cosmetic. The build function takes
resolved values; `None`-means-policy defaults move to the interface.

### 4.2 Env vars become knobs

Four, all read at build time (lines 572, 643, 647, 672):
`FMHA_SCHED_STRATEGY`, `FMHA_REVERSE_Q_TILES`, `FMHA_UNSAFE_NO_KV_CLAMP`,
`FMHA_FP_MODE`. The review's objection is that they are invisible from the API,
which is right, and worse than that: they are part of the emitted code but *not*
part of the JIT cache key, so two callers in one process can silently share a
kernel built under a different setting.

**On `_FP_MODE` / `fm_fast` specifically** -- the review asks "shouldn't they be
computed at host and pass to the kernel instead?" They *are* computed at host
(build) time; `fm_fast` is a fastmath attribute on the emitted ops, so it cannot
be a runtime kernel argument. The half of the suggestion that applies is making
it a named build parameter (`fp_mode="noninf"`) rather than an env var.

### 4.3 The 26-parameter signature becomes a dataclass

`build_flash_attn_func_aiw_module_primary` takes 26 parameters. The review asks
for a dataclass, which is right, with one refinement: **two** of them, because
the parameters are not one kind of thing.

| dataclass       | holds                                                            | who sets it                        |
| --------------- | ---------------------------------------------------------------- | ---------------------------------- |
| `FmhaProblem`   | `num_heads`, `head_dim`, `head_dim_v`, `d_offset`, `causal`, `causal_type`, `dtype_str`, `sm_scale`, `padded_head`, `bias`, `dropout`, `philox_width` | the caller -- what to compute      |
| `FmhaSchedule`  | `block_m`, `block_n`, `k_prefetch_dist`, `v_prefetch_dist`, `v_lds_layout`, `q_row_tiles`, `shards`, `waves_per_eu`, `flat_work_group_size`, `sched_strategy`, `fp_mode`, `daz`, `unsafe_fp_math`, `fast_fp_math`, `strides_constexpr`, `lpt_tile_order`, `unsafe_no_kv_clamp` | the tuning policy -- how to compute it |

Splitting on that line is what makes §4.1 enforceable: the interface produces a
`FmhaSchedule`, and the AIW file never decides one. A single blob would let
policy drift back in unnoticed.

Both frozen, so a schedule cannot be mutated after it has been used as a cache
key.

**Gate:** the enumeration trick from §2.3 -- for every
`(head_dim, causal, dtype, flags)` combination the interface can produce, diff
the resolved `FmhaSchedule` before and after. Identical for all of them, or the
move changed policy. Then bitwise on the suite.

**Cost:** medium, mostly mechanical. **Risk:** low, but it touches every call
site, so do it in one commit rather than piecemeal.

---

## 5. Phase 3 -- type discipline

Root cause A, and the highest-value item in the review even though it is filed
under "possible bugs".

### 5.1 The problem

`fx.Index` is **64-bit and `signed=False`** on this target -- the DSL declares
`width=64, signed=False`, and the emitted LLVM sign-extends the 32-bit
workgroup id to `i64`. So every comparison on an `fx.Index` is `ult`/`ugt`.

The file then uses `fx.Index` for signed sequence quantities: `seqlen_q_v`,
`seqlen_k_v`, `q_start`, `tile_start`, `shard_qk_off`. That is why
`_scmp_i32`/`_ssel_i32`/`_smin_i32`/`_smax_i32` exist -- they coerce back to
`i32` so the compare is signed. The helpers are a workaround for the type
choice, not for the operators (§1.4).

It is also not hypothetical. The `seqlen_k == 0` fault earlier in this branch
was exactly this: `seq_last = seqlen_k_v - 1` underflowed to 2^64-1 and
addressed `0xfffffffff000`. The review's instinct that `seqlen_q/k` "always fits
i32 and we don't need to cast it to u64" is correct and the fix is worth more
than its line count.

### 5.2 The rule

**The goal is not to narrow everything to `i32`. It is to use 64 bits only
where 64 bits are needed, and to say so at the point of use.** Those are
different targets, and the second one is the one that survives contact with a
large tensor.

Two halves, and the second half is where the bugs are:

**(a) Sequence-space quantities are `i32`, because they fit.** Lengths, row and
column origins, tile indices, window bounds, shard offsets. `seqlen_q_v` is the
clearest case -- it is built as `fx.Index(_seqlen_q_i32)` from a value that was
already `i32`, widened for one comparison and used nowhere else. Everything
here is bounded by a sequence length or a head count and cannot approach 2^31.

**(b) Addressing still casts to 64 bits -- and the cast goes *before* the
arithmetic, not after.** This is the part that is easy to get wrong while
"cleaning up types", because both orders compile and only one is correct:

```python
fx.Int64(a * b)              # WRONG: multiplies in 32 bits, then widens
fx.Int64(a) * fx.Int64(b)    # right: widens, then multiplies
```

A tensor large enough to need the 64-bit product is exactly the tensor where
the first form has already overflowed. The file already gets this right in the
one place it was thought about -- the Philox plane base carries a comment
saying "`fx.Int64` first, then multiply", because `off_zh * Max_seqlen_q *
stride` reaches 2^32 at B*H = 256 with 8K sequences -- and the risk of this
phase is silently introducing the other order somewhere that has no such
comment.

So **P3.1 produces, as a reviewable artifact before any code changes, an
explicit list of which quantities are 64-bit and why.** Anything not on it is
`i32`; anything on it names the product that overflows. §11.2 is the first cut
-- the two known 64-bit-required quantities are the Philox plane base and
`max_seqlen_q * STRIDE_TOKEN`, and the measured bound on the shard offsets is
384, so those are settled and P3.1 only has to classify the remainder.

Two facts from §11.2 shape the effort. There are 100 `fx.Index(` sites, so the
audit is a morning rather than a week. And the widen-after-multiply hazard is
currently **absent** -- 7 `fx.Int64(` sites, one containing a multiply, bounded
by 24 -- so P3.2's job is to avoid introducing it, not to fix it.

Then delete `_scmp_i32` and `_ssel_i32` (§1.2, §1.4) and reduce
`_smin_i32`/`_smax_i32` to one-liners over the plain operators -- with the
operands already `i32`, the coercion those helpers existed to perform is gone.

### 5.3 Why this is the riskiest phase

It changes emitted integer arithmetic everywhere, including in address
computation, and 32-bit arithmetic that overflows where 64-bit did not would
fault or silently alias. The offsets that genuinely need 64 bits (the Philox
plane base, `off_zh * Max_seqlen_q * stride`, which reaches 2^32 at B*H=256 with
8K sequences) must be identified before, not after.

**Gate:** bitwise on the full suite *and* the 90-case gSWA window sweep
(`w_left` from -256 to 300 across five `(Lq, Lk)` pairs, several ragged), which
is the sharpest existing net for signed-boundary errors. Plus the varlen suite,
which is where a zero-length sequence would resurface. Plus an interleaved perf
A/B: narrower integers should not be slower, but "should not" is not a
measurement.

**Cost:** high. **Risk:** high -- highest in the plan. Do it as its own commit
with nothing else in it.

---

## 6. Phase 4 -- three API investigations

These decide the *shape* of Phase 5, so they run before it. Each is a question
with a measurable answer, not a commitment to change anything. The first is
already answered; the other two are short spikes (P4.2, P4.3).

| question                                                        | what is known now                                                                 | decides                              |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------ |
| ~~Is `_pointer_to_llvm_ptr` + `_pointer_load/store` still needed?~~ **Answered** | No, and they are not even the first copy -- verbatim duplicates live in `flash_attn_utils.py:523,527`, and `common/mem_ops.py` already provides `get_llvm_ptr`/`global_load`/`global_store`. See §7.1 | they are deleted, not moved |
| Can LDS pointer arithmetic replace `ptrtoint` + `addi`?         | `fx.add_offset(ptr, offset)` exists and shifts by *elements*; unverified for address space 3 | same                                 |
| Can the layout API replace `coop_load/store_*`?                 | `fx.rocdl.make_buffer_tensor` + `fx.copy_atom_call` is the documented path for tiled copy with/without transpose; these are hand-rolled | whether Phase 5 moves them or deletes them |

The third is the one the review flags as *"Check existing kernel for better
practice"*, and it is the largest potential simplification in the whole review
-- `coop_load` appears 11 times and `coop_store` 6. It is also the one most
likely to cost performance, since the hand-rolled versions exist in a kernel
that has been tuned against LDS latency. Treat a regression here as a reason not
to change it, and record the number either way.

**Gate:** each answered with a working example or a measured reason not to.

---

## 7. Phase 5 -- shared helper modules

Root cause C. Two surveys drive this section, and both changed what it says.

> **Groundwork landed (`49764d29`).** The decision below -- extend
> `kernels/common/` rather than create `helper_*` modules -- is settled, and
> the mechanics are in place: `gfx1201_standalone.py` makes the working tree's
> `kernels/common` importable from files run in place, and `wmma_acc` has moved
> to `kernels/common/mma/wmma_ops.py` and been generalised to dispatch on the
> operand element type. That is P5.0. The remaining moves (P5.2 to P5.5) still
> gate on Phases 3 and 4 -- see §8.

### 7.1 Survey: the repo already has helper modules, and the AIW file duplicates them

| module                                  | size    | holds                                                                    |
| --------------------------------------- | ------- | ------------------------------------------------------------------------ |
| `kernels/common/mem_ops.py`             | 7.4 KB  | `get_llvm_ptr`, `global_load`, `global_store`, `global_load_i32`, atomics |
| `kernels/common/utils.py`               | 2.7 KB  | `cdiv`, `align_up`, `pow2_shift`, `udiv_pow2`, `urem_pow2`, `exp2_f32_fast`, `rcp_f32` |
| `kernels/common/layout_utils.py`        | 5.9 KB  | `idx2crd`, `crd2idx`, `_div_pow2`, `_mod_pow2`                            |
| `kernels/common/mma/mfma_epilogues.py`  | 16.7 KB | MFMA epilogues                                                            |
| `kernels/common/buffer_ops.py`          | 24.8 KB | buffer resources and descriptors                                          |
| `kernels/attention/flash_attn_utils.py` | **5110 lines** | the gfx950 attention helpers: intrinsics, softmax pieces, traits/context/loader classes |
| `kernels/attention/pa_common.py`        | small   | paged-attention helpers                                                   |

**The AIW file re-implements things that exist in two of these.**
`_pointer_load` / `_pointer_store` are **verbatim identical** to
`flash_attn_utils.py:523,527` -- same body, differing only in whether the LLVM
dialect is imported as `llvm` or `_llvm`. And `common/mem_ops.py` already
provides `get_llvm_ptr` / `global_load` / `global_store`, which is what
`_pointer_to_llvm_ptr` plus those two amount to.

So the review's *"Are they still required in more recent FlyDSL?"* has a
sharper answer than expected: they are not required **and they are not even
the first copy**. Several other AIW helpers should be checked the same way
before being moved anywhere -- `bf16_trunc_pack_v8` against
`flash_attn_utils._bf16_trunc_pack_v8`, `_sdiv_rd` against
`common/utils.udiv_pow2`, `_fadd`/`_fsub`/`_fmul`/`_fmax` against
`flash_attn_utils._fadd`/`_fsub`/`_fmul`/`_fmax`.

`flash_attn_utils._fadd(a, b, fm_fast)` is worth singling out: it takes
`fm_fast` as a **parameter**, where the AIW file's version closes over it.
The parameter form is what a shared module needs, and it is also what §4.2's
`fp_mode` knob wants. Adopt the existing signature rather than inventing one.

### 7.2 Survey: what AOTriton actually shares between fwd and bwd

From `aotriton/modules/flash/kernel/`, by import:

| shared module            | used by                                | contents relevant to us                                            |
| ------------------------ | -------------------------------------- | ------------------------------------------------------------------ |
| `composed_tensors.py`    | fwd + all bwd                          | `composed_ptrs/load/store/advance/to`, `composed_mul_lhs`, `composed_dot_both`, `composed_dot_rhs`, `composed_mul_acc` |
| `masked_load_store.py`   | fwd + all bwd                          | `parse_window`, **`calculate_intervals`**, `closed_interval_isect`, `div_rd`, `load_fn`, `mload2d`, `mstore2d` |
| `dropout.py`             | fwd + bwd                              | `fast_dropout_mask`, `PHILOX_RN_PER_OFFSET`                        |
| *(leaked)* `fwd_kernel_inner._lse_offset` | bwd imports it **from the fwd module** | LSE addressing                                     |
| *(leaked)* `fwd_kernel.remap_xcd`         | `bwd_kernel_fuse` imports it from fwd  | grid remapping                                     |

Three things follow.

**`calculate_intervals` is our gSWA split, and AOTriton shares it.** It returns
`lb_lo, lb_hi, fb_lo, fb_hi, rb_lo, rb_hi` -- three intervals, left-masked /
full / right-masked -- which is exactly the structure our kernel computes, down
to the variable names (`_fb_lo`, `_fb_hi`, `_lb_hi`, `_rb_lo`). It sits in a
shared module because the backward pass walks the same regions. The review's
instinct here is confirmed by the reference implementation, not just plausible.
`parse_window` (sentinel resolution) and `div_rd` (our `_sdiv_rd`) travel with
it, along with the interval algebra our code open-codes via `_smin`/`_smax`.

**The two leaked imports are a warning, not a model.** AOTriton's bwd reaching
into `fwd_kernel_inner` for `_lse_offset` is what happens when a fwd/bwd-shared
routine is not extracted at the time it becomes shared. Our LSE addressing is
already more complicated than theirs -- it has the `VARLEN_LSE_LAYOUT_TH`/`_HT`
choice -- so it goes in the shared module from the start rather than being
imported out of the forward kernel later.

**Dropout is already done right.** `philox.py` is the analogue of their
`dropout.py`, and it is already shared, already generic over width, and already
has `grid_plane`/`grid_offset` as shared code rather than a shared formula.
Nothing to do; it is the template for the rest.

### 7.3 Naming, and where things go

The review asks for generic names -- `helper_memory.py`, `helper_wmma.py` --
on the grounds that this code is not SDPA-specific. Agreed on the reasoning.
On the spelling, one caution: **no `helper_*` module exists anywhere in the
repo.** The two live conventions are `*_common.py` (`pa_common.py`,
`moe_common.py`, `rmsnorm_common.py`, `gemm_common_gfx1250.py`) and `*_utils.py`
(`flash_attn_utils.py`, `layout_utils.py`, `dpp_utils.py`, `fp8_gemm_utils.py`).
Adding `helper_*` makes a third.

The stronger way to get what the review wants is **placement, not prefix**:
code that is not attention-specific goes in `kernels/common/`, which already
exists for exactly this and is imported by GEMM, MoE, comm and attention alike.
That is a stronger statement of "not SDPA-specific" than any filename.

| what                                                          | proposed home                            | why                                                        |
| -------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| pointer / global load-store (`_pointer_*`, `_global_load_tr_v8`) | **`kernels/common/mem_ops.py`** (extend)   | already has `get_llvm_ptr`/`global_load`/`global_store`; AIW duplicates it |
| LDS load/store (`_lds_load_v8`, `_lds_store_vx`)              | **`kernels/common/mem_ops.py`** (extend)   | same family, no LDS half exists yet                        |
| `wmma_acc`                                                    | **`kernels/common/mma/wmma_ops.py`** (new) | sits beside `mfma_epilogues.py`; WMMA is an arch feature, not an attention one |
| interval algebra, `div_rd`                                    | **`kernels/common/utils.py`** (extend)     | already holds `cdiv`/`align_up`/`pow2_shift`                |
| gSWA regions, `parse_window`, Q preload, cross-shard reduction, LSE addressing | **`kernels/attention/fmha_common.py`** (new) | attention-specific; matches `pa_common.py` next to it       |

If you prefer the `helper_` prefix I will use it -- but I would rather match
`pa_common.py`, which is the file sitting in the same directory doing the same
job for paged attention.

**Not `flash_attn_utils.py`.** It is 5110 lines and structured around the
gfx950 dualwave traits/context class hierarchy; adding gfx1201 gSWA to it would
couple two unrelated designs. Borrow its *signatures* (§7.1) without inheriting
its shape.

### 7.4 Two review items with specific fixes

- **`wmma_acc`'s `if const_expr(dtype_str == "bf16")` is odd.** Agreed -- it
  keys on a build-time *string* when the information is in the operands.
  Dispatch on the vector element type instead. That is what makes it callable
  from a module that does not have `dtype_str` in scope, which is the whole
  point of moving it. Same fix, same reason, for `_global_load_tr_v8` being
  generic over `elem_dtype`.
- **`_decode_side` (58 lines)** decomposes per axis -- length source, position
  source, batch and token counts -- each a small function over `VarlenBits`. It
  is the varlen decoder, so it is backward-shared and lands in
  `fmha_common.py`. On *"use @dataclass if flydsl supports"*: it returns four
  values today and a frozen dataclass of four `fx.Int32` fields is fine, with
  one constraint to check first -- the fields must stay plain SSA values and
  not be carried across an `scf` boundary, where FlyDSL wants an explicit list.

**Gate:** bitwise on the suite; interleaved perf A/B, since moving code across
a module boundary can change inlining. Where a helper is replaced by an
existing one rather than moved, that is a behaviour claim and needs the bitwise
gate specifically -- `common/mem_ops.global_load` takes a *byte* offset where
AIW's `_pointer_load` takes a typed pointer, and mixing those up is silent.

**Cost:** high. **Risk:** medium.

## 8. Ordering, and why this order

```
Phase 1  free corrections        (comments, names, casts)
Phase 2  policy out of the file  (constants, env vars, dataclasses)
Phase 3  type discipline         (Index -> Int32, delete the compare helpers)
Phase 4  three investigations    (ptr helpers, LDS add_offset, layout API)
Phase 5  shared helper module    (extraction for the backward pass)
```

Three ordering constraints, all real:

1. **Phase 3 before Phase 5.** The type work *deletes* `_scmp_i32` and friends.
   Extracting them into a shared module first would move code that is about to
   disappear, and would put the wrong helpers in the backward pass's path.
2. **Phase 4 before Phase 5.** If the layout API can replace `coop_load/store`,
   Phase 5 deletes them instead of moving them. Doing Phase 5 first risks
   carefully relocating code that should not exist -- and §7.1 shows that risk
   is already realised for `_pointer_load`/`_pointer_store`, which exist twice
   in the tree today. Phase 5 starts with a duplication audit against
   `common/mem_ops.py`, `common/utils.py` and `flash_attn_utils.py`, not with
   a move.
3. **Phase 2 before Phase 3.** The dataclass makes the 26-parameter signature
   tractable, and Phase 3 touches many of the same lines. Doing them in the
   other order means resolving the same conflicts twice.

Phase 1 is safe to land immediately and independently.

An earlier draft of this plan said Phases 2, 3 and 5 should each be a *single*
commit, on the grounds that bisecting a half-applied refactor is worse than the
refactor. That was wrong, and §0.3 replaces it. The objection conflates two
different intermediate states: a **broken** tree (does not compile, fails the
suite) genuinely is worse to bisect through, but an **inconsistent** one --
half the file narrowed to `i32`, half not -- compiles, passes, and is a
perfectly good bisect point. Requiring every commit to be independently valid
gets the safety; requiring them to be large does not.

---

## 9. Risks

| risk                                            | why it matters                                                          | mitigation                                          |
| ------------------------------------------------ | ------------------------------------------------------------------------ | --------------------------------------------------- |
| Phase 3 narrows an offset that needs 64 bits    | silent aliasing or a fault, and the aliasing case passes every test     | the §5.2(a) list -- enumerate the 64-bit quantities *first*, naming the product that overflows |
| Phase 3 widens *after* the multiply instead of before | `fx.Int64(a * b)` compiles and is wrong exactly on the large tensors that need it | §5.2(b); grep every new `fx.Int64(` for a multiply inside it |
| Existing tests cannot reach the overflow        | the suite's largest shape is far below B*H = 256 at 8K, so both orders pass | a host-side check on the resolved offsets at a synthetic large shape, since a real one will not fit in memory |
| A refactor changes scheduling in a latency-bound loop | a few percent, invisible to correctness gates                       | interleaved same-process A/B (§2.3), not sequential runs |
| Extraction across a module boundary blocks inlining | same, and harder to attribute after the fact                        | perf gate on Phase 5 specifically                   |
| Policy moves change a default silently          | a "readability" commit that also re-tunes the kernel                    | the enumeration gate in §4.3 -- diff every resolved schedule |
| The plan itself accretes false reasons          | three found this session, all surviving review                          | §2.5: check the claim, not the wording              |
| A helper is *replaced* by a near-equivalent, not an equivalent | `common/mem_ops.global_load` takes a byte offset; AIW's `_pointer_load` takes a typed pointer. Silent, and wrong by a factor of the element size | bitwise gate per replacement, not per phase        |
| `fmha_common.py` inherits `flash_attn_utils.py`'s shape | 5110 lines built around gfx950 dualwave traits/context classes; coupling two designs is worse than duplicating a few lines | borrow signatures, not structure (§7.3)            |

---

## 10. What this does not cover

- **Reformatting.** `black`/`ruff` are not installed in the working venv, so the
  CI style gate has not been run against any of this branch's commits. Worth
  fixing separately; not folded in here, because a formatting pass mixed with a
  refactor makes both unreviewable.
- **The kernel's algorithmic structure.** The softmax recurrence, the shard
  reduction and the epilogue are not restructured by any phase above. The review
  did not ask, and they are the parts where a readability change is most likely
  to cost performance.
- **`sdpa-readability.md` items already fixed.** The `_scmp_i32` docstring is
  listed in Phase 1 rather than treated as done -- `keep_mask`'s copy of the same
  false claim was fixed in c4221b10, but this one is still in the tree.

---

## 11. Measured facts (do not re-derive)

Everything here was checked on device or by reading the installed source
during the drafting of this plan. It is recorded so that no phase spends its
budget re-establishing it, and so that a claim can be challenged against a
number rather than against a memory.

### 11.1 DSL semantics

| claim                                                        | status | how checked                                              |
| -------------------------------------------------------------- | ------ | ---------------------------------------------------------- |
| `fx.Index` is 64-bit, `signed=False`                         | true   | `expr/numeric.py:832`; LLVM IR `sext i32 ... to i64`     |
| `fx.Int32` is `signed=True`; `>` emits `sgt`                 | true   | `expr/numeric.py:716`; dumped `arith.cmpi sgt`           |
| `fx.Int32`'s `<`/`>` are unsigned                            | **false** | the claim in `_scmp_i32`'s docstring; see §1.4        |
| Python ints coerce on `fx.Index` and `fx.Int32` operands     | true   | `tid * 96` and `fx.Int32(tid) * 96` both run correctly   |
| `shard_id * fx.Index(K)` and `shard_id * K` have the same type | true | both emit `arith.muli ... : index`                       |
| `range_constexpr` is renamed to builtin `range`              | true   | `ast_rewriter.py:1243`                                   |
| the loop rewrite is per-decorated-function                   | true   | `ast_rewriter.py:171` (`inspect.getsource(f)`)           |
| `ir.VectorType.isinstance` exists                            | **false** | use Python `isinstance` against the downcast class    |

### 11.2 Integer-width inventory

| quantity                                  | measured bound              | width needed |
| ------------------------------------------- | ----------------------------- | -------------- |
| `shard_qk_off = shard_id * QK_SLICE`      | **max 384** over the ladder  | i32 (5.6e6x headroom) |
| `shard_vo_off`                            | `< VO_CHUNK_COLS <= VO_WIDTH` | i32           |
| any head_dim column index                 | `< _MAX_HEAD_DIM = 512`      | i32           |
| Philox plane base `off_zh * Sq * stride`  | crosses 2^32 at B*H=256, 8K  | **i64**       |
| `max_seqlen_q * STRIDE_TOKEN` (line 1295) | grows with the tensor        | **i64**       |

Only 384 and 512 shard at all (`resolve_shards` returns 1 for every other
rung), so `shard_qk_off` is identically 0 for eleven of the thirteen compiled
widths.

**The widen-after-multiply hazard is currently absent.** There are 7
`fx.Int64(` sites in the AIW file and exactly one contains a multiply --
`fx.Int64(fx.Int32(klane) * fx.Int32(8))`, bounded by 24. P3.2's job is to keep
that count at zero-that-matter, not to fix an existing bug.

**Distribution of the 100 `fx.Index(` sites**, by a crude keyword pass (P3.1
replaces this with a real classification): ~25 addressing, ~10 sequence-space,
5 grid/thread ids, ~60 unclassified.

### 11.3 Duplication already in the tree

| AIW helper                       | already exists as                                              |
| ---------------------------------- | ---------------------------------------------------------------- |
| `_pointer_load` / `_pointer_store` | **verbatim** in `flash_attn_utils.py:523,527`                  |
| `_pointer_to_llvm_ptr`           | `common/mem_ops.get_llvm_ptr` (+ `global_load`/`global_store`) |
| `bf16_trunc_pack_v8`             | `flash_attn_utils._bf16_trunc_pack_v8` (check equivalence)     |
| `_fadd`/`_fsub`/`_fmul`/`_fmax`  | `flash_attn_utils._fadd`... -- and *they take `fm_fast` as a parameter* |
| `_sdiv_rd`                       | `common/utils.udiv_pow2` (unsigned; ours is signed -- check)   |
| `wmma_acc`                       | moved to `common/mma/wmma_ops.py` in `49764d29`                |

### 11.4 AOTriton's fwd/bwd shared surface

| their module           | their symbol                                    | our counterpart                        |
| ------------------------ | ------------------------------------------------- | ---------------------------------------- |
| `masked_load_store.py` | `calculate_intervals` -> `lb/fb/rb` x `lo/hi`   | the gSWA three-region split (P5.5)     |
| `masked_load_store.py` | `parse_window`                                  | `_resolve_window` + `_CAUSAL_SENTINEL` |
| `masked_load_store.py` | `div_rd`, `closed_interval_isect`               | `_sdiv_rd`, open-coded `_smin`/`_smax` |
| `composed_tensors.py`  | `composed_ptrs/load/store/advance`              | the coop load/store family (P4.3)      |
| `dropout.py`           | `fast_dropout_mask`, `PHILOX_RN_PER_OFFSET`     | `philox.py` -- **already shared**      |
| *(leaked into fwd)*    | `_lse_offset`, `remap_xcd`                      | LSE addressing -- extract *before* it leaks |

### 11.5 Measurement noise floor, for anyone reading a perf gate

Sequential before/after runs of this kernel have a noise floor of about **5%
with outliers to +19%** -- established on 48 points whose kernel selection was
provably identical. Same-process interleaved A/B separates cleanly at 1%. Any
perf gate in this plan means the second thing.

