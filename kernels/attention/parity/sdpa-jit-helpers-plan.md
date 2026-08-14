# Deduplicating the four kernels, and what `@flyc.jit` unblocks

## 0. The enabling fact

`fmha_common_gfx1201.py` opens with a section whose premise is:

> Nothing in this file is AST-rewritten. The rewrite from Python's `if` to
> `scf.if` is lexical per `@flyc.kernel` function, so a module-level helper gets
> a branch only by building the `scf.IfOp` itself.

Half wrong, and has been all along. The rewrite is lexical per **decorated**
function, and `@flyc.jit` decorates: `jit_function.py:22` runs the same
`ASTRewriter.transform` as `kernel_function.py:461`. `JitFunction.__call__`
opens with

```python
if ir.Context.current is not None:
    return self.func(*args, **kwargs)   # already tracing -> inline the body here
```

so the decorator means *"AST-rewrite and trace this"*, and the call site decides
the role: from Python a host launcher, from inside an open trace a device-side
inline. A module-level helper gets branching by asking for it.

**But that is not the main deduplication story**, and the survey below is
deliberately blunt about which parts it unblocks and which parts were merely
never done.

## 1. Survey

Method: every nested helper in the four kernels, anonymised (identifiers to `V`,
string constants to `S`) and compared pairwise by AST shape; plus a matching-run
scan over anonymised top-level statement sequences to catch straight-line
regions no helper wraps. Numbers are recoverable lines, i.e. total minus the one
surviving copy.

### 1a. Host-side helpers -- ~252 lines, and **never blocked by anything**

| helper | copies | lines | recoverable |
|---|---|---|---|
| `_launch` | 4 | 66 / 42 / 60 / 72 | ~168 |
| `_compile` | 2 | 65 / 42 | ~42 |
| `_prep` | 3 | 21 / 14 / 15 | ~29 |
| `_resolve_scale` | 3 | 13 / 7 / 6 | ~13 |
| `_row_tensor` / `_row_tensor_ptr` | 2 | 26 / 24 | (0.99 similar) |

`_resolve_scale` is **structurally identical** across three kernels (ratio 1.00
fwd/dq, 0.96 to dkdv). `_prep` is 0.82--0.89. `_launch` is 0.79--0.96 across all
four, and the four `launch_*` jit bodies are 0.83--0.96 similar to each other.

These are plain Python running before a launch. They want
`fmha_abi_gfx1201.py`, which already exists for exactly this purpose and whose
docstring already says so. **This is the largest single win in the survey and it
has nothing to do with `@flyc.jit`.** It was possible the whole time; it just was
not done.

Caveat on `_launch`/`_compile`: the similarity is real but so is the argument
list, which *is* the kernarg ABI and differs per kernel. What factors cleanly is
the middle -- validation, stride collection, window/varlen/dropout marshalling --
not the call itself. Expect to recover well under the 168 the diff suggests.

### 1b. Device-side straight-line prologue -- also never blocked

The top-level statement scan finds 28--75 shared lines per kernel *pair*, all in
the prologue (fwd L784--935, dq L380--500, dkdv L360--490, fuse L430--520):

| unit | copies | note |
|---|---|---|
| thread decomposition (`tid`, `wave_id`, `lane`, `lane16`, `klane`) | 4 | identical 5 lines |
| vector type triple + `wmma_acc` closure | 4 | `v8f32_type`, `v8f16_type`, `vxf16_type` |
| `pointer_to_llvm_ptr` run | 4 | same shape, different tensor sets |
| `_llvm_value` | 4 | ~12 recoverable |
| `_split_ptr` | 4 | ~15 recoverable; fwd/dq identical at 1.00 |
| `_to_global_ptr_i64` | 2 | ~2 |

All straight-line. No branch, no select, nothing that needed the rewrite. A
plain module-level function would have served since day one.

### 1c. Actually unblocked by `@flyc.jit` -- the smaller pile

Everything here needs a branch or a lazy ternary, so it genuinely could not live
in an undecorated module.

**Hand-built `_scf.IfOp` already in `fmha_common` (7 sites).** These are not
duplication -- they are the *workaround* for the missing rewrite, and they
collapse:

| helper | becomes |
|---|---|
| `cond_load` | a one-line ternary |
| `_load_u64_or_zero` | a one-line ternary |
| `_store_u64_if_nonnull` | `if ptr != 0: ...` |
| `philox_report` | nested `if`, two levels shallower |
| `_over_batches`, `write_v8`, `publish_transposed` | plain `if` |

`cond_load` is the headline. Its docstring spends nine lines arguing it must be
*"a real `scf.if`, not a select, and that is the point"* -- because the
sequence-info pointers are null when their mode is off and a select would
evaluate both arms and fault. The ternary lowers to
`scf_ifexp_dispatch(c, lambda: a, lambda: b)`: lazy arms, real `scf.if`. Exactly
the required semantics, in one line.

**`.select` / `ssel` in `fmha_common` (26 sites)** -- `decode_addressing` 6,
`decompose_causal_regions` 4, `lse_token_pitch` 3, `resolve_window` 3,
`make_addr_pair` 3, `lse_row_addressing` 2, `MaskedAxis` 2, `Aperture` 1, other
2. Readability only; the same change inside the kernels was bitwise-ISA-neutral.

**Cross-kernel duplicates that branch:**

| helper | copies | recoverable |
|---|---|---|
| `masked_col` / `_masked_col` | 4 (fwd, dq, fuse x2) | ~8 |
| `_alive` clamps (`x if alive else 0`) | 4+ | one-liners, probably not worth a helper |
| `_load_row_f32` / `load_global_f32` | 2, 0.79 similar | ~9 |
| `_pack_v8` / `pack_v8` | 2, 0.98 similar | ~11 |

**Honest total for 1c: ~30 lines of cross-kernel duplication, plus the seven
workaround helpers and 26 selects.** The value here is correctness and
legibility -- four copies of the discontinuous seam map is four chances to get
the seam wrong, and the seam *was* got wrong during bring-up -- not line count.

### 1d. The boundary, measured

Spikes under `/tmp/tmp/spike/`, on gfx1201, module-level `@flyc.jit` called from
a `@flyc.kernel`:

| pattern | result |
|---|---|
| closure argument called under a dynamic `if` | works |
| frozen-dataclass argument, method called in the *condition* | works |
| dataclass attribute *read* under the branch | works |
| dataclass **method call** under the branch | `TypeError: Cannot extract IR values from Axis(width=7)` |

The last row is the trap the file header documents. Its workaround still
applies: *call a free function so the base name is a module rather than the
object.* Every helper in 1c already obeys it -- their branch bodies call
`body(...)`, `write(...)`, `fx.ptr_load(...)`.

One new and favourable fact: the trap fires on names from an *enclosing scope*,
and inside a helper everything arrives as a **parameter**, already a local.
Moving branchy code out of a kernel into a decorated helper makes this class of
bug less likely, not more.

## 2. Plan

Ordered by value per unit of risk, which puts the `@flyc.jit` work *last* --
the big wins do not need it.

**Step 1 -- host helpers to `fmha_abi_gfx1201.py` (~150-250 lines).** Start with
`_resolve_scale` (identical x3), then `_prep`, then `_row_tensor`. Leave
`_launch`/`_compile` for a separate pass: factor the marshalling middle, keep
the per-kernel argument list where it is, since that list is the ABI. Pure
Python, no tracing, no `@flyc.jit`. Gate: full suite.

**Step 2 -- straight-line device prologue to `fmha_common` (~40 lines).** A
plain (undecorated) `fmha.wave_decomposition(tid)` returning the five indices,
and a `fmha.vector_types(elem_dtype)` returning the triple. `_llvm_value`,
`_split_ptr`, `_to_global_ptr_i64` alongside. Still no decorator needed. Gate:
full suite + bitwise ISA at hd 128/384 x both masking modes.

**Step 3 -- decorate, change nothing else.** Add `@flyc.jit` to the seven
workaround helpers and the select-heavy value helpers, bodies untouched. Proves
the decorator is inert before any body moves. Gate: as step 2.

**Step 4 -- collapse the workarounds.** Rewrite those seven bodies as
`if`/ternary. Delete the stale paragraphs from the file header and from
`cond_load`'s docstring; leaving the "must build the IfOp myself" rationale in
place would actively mislead. Gate: as step 2.

**Step 5 -- `ssel`/`.select` to ternaries (26 sites).** Mechanical. Watch
`MaskedAxis`/`Aperture`: they are methods, so `self` is a parameter and a
`self.foo()` under a branch trips 1d -- keep such calls in the condition or
hoist. Gate: as step 2.

**Step 6 -- the branchy cross-kernel duplicates.** `masked_col` first (4 copies,
and the seam is the part that has actually been wrong), then `_pack_v8` and
`_load_row_f32`. Gate: as step 2, plus the varlen and causal suites.

## 3. Risks

- **Silent eager ternary.** Add a ternary to an undecorated module function and
  Python evaluates it at trace time: `bool(cond)` on an `fx.Boolean` either
  raises or bakes in one arm, and every test exercising only that arm passes.
  This is why step 3 precedes step 5 and why the two must not share a commit.
- **Host-callable by accident.** A decorated device-only helper called with no
  open trace will try to *compile*. Unreachable today; worth a line in the
  header.
- **`test_signature_parity_gfx1201.py` asserts one `@flyc.jit` per module.** It
  scans only the four kernel modules, so decorating `fmha_common` is fine --
  but adding a nested jit helper *to a kernel module* trips it. Two-line fix.
- **Perf.** Expected nil -- one Python call and a context check at trace time,
  nothing at runtime, and ternaries if-convert. Expected, not known, until each
  step's ISA gate says so.
- **Prologue extraction is the one with real ISA risk.** Steps 1 and 3--5 move
  code that emits identical IR. Step 2 changes *where* values are materialised,
  which can reorder definitions and perturb the scheduler. Gate it hardest.

## 4. Not doing

`stage`, `publish`, `read_batches`, `read_transposed` and the rest of the
`Aperture` surface stay free functions taking the aperture as a parameter. That
shape is what keeps object method calls out of branch bodies, which 1d shows is
still the one hard constraint. Decorating them buys nothing -- their branching
is already delegated to `_over_batches` and `write_v8`.
