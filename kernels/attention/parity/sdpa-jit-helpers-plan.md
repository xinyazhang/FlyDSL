# Decorating the shared module: `@flyc.jit` on `fmha_common_gfx1201.py`

## 0. What changed, and why it matters

`fmha_common_gfx1201.py` opens with a section titled *"Why the branching helpers
live here"*, whose premise is:

> Nothing in this file is AST-rewritten. The rewrite from Python's `if` to
> `scf.if` is lexical per `@flyc.kernel` function, so a module-level helper gets
> a branch only by building the `scf.IfOp` itself.

That premise is **half wrong, and has been all along**. The rewrite is lexical
per *decorated* function, and `@flyc.jit` decorates too --
`jit_function.py:22` runs the same `ASTRewriter.transform` that
`kernel_function.py:461` does. A module-level helper gets the rewrite by asking
for it.

`@flyc.jit` is not "host code". `JitFunction.__call__` opens with:

```python
if ir.Context.current is not None:
    return self.func(*args, **kwargs)   # already tracing -> inline the body here
```

So the decorator means *"AST-rewrite and trace this"*, and the call site decides
the role: from Python it is a host launcher, from inside an open trace it is a
device-side inline. `@flyc.kernel` is the one with a fixed job -- it refuses to
run outside a trace at all (`kernel_function.py:161`) and emits a `gpu.func`
with a kernarg signature.

The consequence for this file: every hand-built `_scf.IfOp` and every `.select`
that exists *only because the file could not branch* is now optional.

## 1. Survey

### 1a. Hand-built control flow (7 sites)

| helper | shape | becomes |
|---|---|---|
| `cond_load` | `IfOp` + 2 `YieldOp`, i32 result | a one-line ternary |
| `_load_u64_or_zero` | `IfOp` + 2 `YieldOp`, i64 result | a one-line ternary |
| `_store_u64_if_nonnull` | `IfOp`, void | `if ptr != 0: ...` |
| `philox_report` | nested `IfOp`, void | nested `if` |
| `_over_batches` | `IfOp`, void, `const_expr` sibling | `if row < block_rows: body(...)` |
| `write_v8` | `IfOp`, void | `if aperture.cols.valid(col): write(...)` |
| `publish_transposed` | `IfOp`, void | plain `if` |

`cond_load` is the headline. Its docstring spends nine lines explaining that it
is *"a real `scf.if`, not a select, and that is the point"* -- because the
sequence-info pointers are null whenever their mode is off, and a select would
evaluate both arms and fault. **The ternary is exactly that**: the rewriter
lowers `a if c else b` to `scf_ifexp_dispatch(c, lambda: a, lambda: b)`, and the
arms are lambdas, so only the taken one runs. The entire helper collapses to:

```python
return fx.ptr_load(addr, fx.Int32) if cond else default
```

and the docstring shrinks to the one fact a reader still needs (the address is
the caller's and touches no memory).

### 1b. `.select` / `ssel` used because the file cannot branch (26 sites)

| helper | count |
|---|---|
| `decode_addressing` | 6 |
| `lse_token_pitch` | 3 |
| `resolve_window` | 3 |
| `decompose_causal_regions` | 4 |
| `make_addr_pair` | 3 |
| `lse_row_addressing` | 2 |
| `MaskedAxis` methods | 2 |
| `Aperture.read` | 1 |
| others | 2 |

These are pure-value selects. Converting them is the same readability change
already made inside the four kernels, where it was **bitwise-ISA-neutral** at
head_dim 128 and 384 in both masking modes -- the backend if-converts every one
back to a `v_cndmask`.

Worth stating plainly: **an undecorated module function must not use a
ternary.** Without the rewrite, `a if cond else b` is plain Python, so
`bool(cond)` runs on an `fx.Boolean` -- it will either raise or silently pick
one arm at trace time and bake it in. Tier 2 is therefore *gated on* Tier 0;
they cannot be done in either order.

### 1c. Logic that could move here but has not (the actual duplication)

| what | copies | notes |
|---|---|---|
| `masked_col` / `_masked_col` | **4** (fwd, dq, fuse x2) | identical left-run/right-run column map; needs a branch or a ternary |
| `wmma_acc` | 3 | trivial wrapper, never blocked -- just never moved |
| `_llvm_value` | 3 | ditto |
| `_alive` clamps (`x if alive else 0`) | 4+ | one-liners; sharing may not pay |
| LLVM passthrough / `waves_per_eu` attr block | 4 | builder-level Python, no tracing at all -- a plain function, not a jit one |

`masked_col` is the one worth doing: four copies of the same discontinuous seam
map, and the seam is precisely the thing that was got wrong during bring-up.

## 2. The boundary, measured

Spikes under `/tmp/tmp/spike/`, all run on gfx1201. A module-level `@flyc.jit`
helper called from inside a `@flyc.kernel`:

| pattern | result |
|---|---|
| closure argument called under a dynamic `if` | **works** |
| frozen-dataclass argument, method called in the *condition* | **works** |
| dataclass attribute *read* under the branch | **works** |
| dataclass **method call** under the branch | `TypeError: Cannot extract IR values from Axis(width=7)` |

The last row is the trap the file header already documents, and the header's own
workaround still applies: *call a free function so the base name is a module
rather than the object.* Every helper in 1a already obeys that -- their branch
bodies call `body(...)`, `write(...)`, `fx.ptr_load(...)`,
`_store_u64_if_nonnull(...)`. None calls a method on a parameter.

One genuinely new and favourable fact: the trap fires on names from an
*enclosing scope*, and inside one of these helpers everything arrives as a
**parameter**, which is already a local. Moving branchy code out of a kernel and
into a decorated module helper therefore makes this class of bug *less* likely,
not more.

## 3. Plan

Each step is independently committable and independently gated.

**Tier 0 -- decorate, change nothing else.** Add `@flyc.jit` to the seven
helpers in 1a and to the pure-value helpers in 1b, leaving their bodies exactly
as they are. Gate: full suite, plus bitwise ISA at hd 128/384 x both masking
modes. This proves the decorator alone is inert before any body is touched, and
it is the step that would expose a surprise in argument binding or caching.

**Tier 1 -- collapse the hand-built control flow.** Rewrite the seven bodies as
`if` / ternary. `cond_load` and `_load_u64_or_zero` become one-liners;
`philox_report` loses two nesting levels. Delete the now-stale paragraphs from
the file header and from `cond_load`'s docstring -- the "must build the IfOp
myself" rationale is gone and leaving it would actively mislead. Gate: as Tier 0.

**Tier 2 -- `ssel` / `.select` to ternaries.** The 26 sites in 1b. Mechanical,
and the same change already validated inside the kernels. Watch `MaskedAxis` and
`Aperture`: they are methods, so `self` is a parameter and a `self.foo()` under
a branch would trip the 1d boundary -- keep such calls in the condition or hoist
them. Gate: as Tier 0.

**Tier 3 -- consolidate `masked_col`.** Move the four copies to one
`fmha.masked_col(i, n_left, left_col0, right_col0, step)`. Gate: as Tier 0, and
additionally the varlen and causal suites, which are what exercise the seam.

**Tier 4 -- optional, and probably not worth it.** `wmma_acc` and `_llvm_value`
are three-line duplicates that were never blocked by anything; moving them is
tidiness, not leverage. The passthrough-attribute block is builder-level Python
and wants a plain shared function, not `@flyc.jit`. Do these only if touching
those files anyway.

## 4. Risks

- **Silent eager ternary.** The one way to get this wrong quietly: add a ternary
  to a module function and forget the decorator. Trace-time Python then picks an
  arm and bakes it in, and every test that exercises only the chosen arm passes.
  Mitigation: Tier 0 before Tier 2, and never add a ternary to this file in the
  same commit that adds the decorator.
- **Host-callable by accident.** These helpers are device-only, but a decorated
  one called with no open trace will try to *compile*, with a confusing error.
  Currently unreachable; worth one line in the file header.
- **`test_signature_parity_gfx1201.py` asserts one `@flyc.jit` per module.** It
  only scans the four kernel modules, so decorating `fmha_common` does not trip
  it -- but adding a nested jit helper *to a kernel module* would, with
  `expected one @flyc.jit, found [...]`. Two-line fix (pair the jit that calls
  the kernel) whenever that happens.
- **Perf.** Expected nil: the decorator adds one Python call and an
  `ir.Context.current` check at trace time, nothing at runtime, and the ternary
  if-converts. But it is expected, not known, until each tier's ISA gate says so.

## 5. Not doing

`stage`, `publish`, `read_batches`, `read_transposed` and the rest of the
`Aperture` surface stay free functions taking the aperture as a parameter. That
shape is what keeps object method calls out of branch bodies, which section 2
shows is still the one hard constraint. Decorating them buys nothing -- their
branching is already delegated to `_over_batches` and `write_v8`.
