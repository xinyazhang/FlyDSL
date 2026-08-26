# Plan: make the gfx950 knob classes flat, plain-data build-option classes

> **Status: option C implemented.** The plan below is kept as written, because
> the reasoning that selected C — in particular §4.2, which killed the
> "keep only the used fields" idea with a measurement — is the part worth
> re-reading if this is ever revisited. §10 records what actually shipped.

**Driver:** `flyati/sdpa-gfx950-knobs-pon-handoff.md` (AOTriton side) plus a
design correction from the owner that supersedes part of it.

**Ask, in one line:** the knob class is *the build option class* — one class,
plain data, no nested `traits` object. Where the GPU-facing code wants a traits
object, the knob class should carry the traits' fields itself, either by
inheriting the traits class or by copying its fields.

Everything measured below is against `c36f1a7e`.

## 1. The correction, and why it matters

An earlier draft of this plan proposed splitting each knob class in two — a
"request" object and a resolved `Gfx950BuildConfig`. **That is wrong and is not
what is being built.** gfx1201 is the reference, and there is exactly one class:

```python
@dataclass(frozen=True)
class FmhaKnobs:
    """How to compute it. Every field `None` means "policy decides"."""
```

`fmha_tuning_gfx1201.py:438`. Every field is `None`-defaulted, every value is a
scalar, and `resolve_knobs(meta, overrides) -> FmhaKnobs` returns *the same
class* with the `None`s filled in. The unresolved and resolved states are two
values of one type, not two types. The `None`-rather-than-a-literal-default
convention is load-bearing there and is quoted in its own docstring: it is the
only way the resolver can tell "the caller wants 1" from "the caller did not
say".

gfx950 already follows that shape. The **only** thing that breaks it is the
single `traits` field holding a nested dataclass. That is the whole defect.

## 2. Measured facts

| | forward | dK/dV | dQ |
|---|---|---|---|
| knob class | `Gfx950Knobs` | `BwdDkDvKnobs` | `BwdDqKnobs` |
| knob fields | 29 | 16 | 31 |
| traits class | `ParityDualwaveTraits` | `BwdDkDvTraits` | `BwdDqTraits` |
| traits fields | 93 | 95 | 98 |
| name collisions knob ↔ traits | **none** | none | none |
| traits values that are not PON literals | **none** | none | none |
| traits fields with **no default** | 76 | 76 | 76 |
| knob fields with no default | 0 | 0 | 0 |

All six classes are `frozen=True`. The two backward traits classes are declared
*inside* their own tuning modules (`fmha_tuning_bwd_dkdv_gfx950.py:334`,
`fmha_tuning_bwd_dq_gfx950.py:158`); only `ParityDualwaveTraits` is external
(`fmha_traits_gfx950.py:35`, a subclass of production
`dualwave.DualwaveSwpTraits`).

**Blast radius is small.** `knobs.traits` has 9 references repo-wide and only
three are real consumers — one per builder, each doing the identical
`if knobs.traits is None: raise` / `traits = knobs.traits`:

- `flash_attn_func_gfx950.py:166`
- `fmha_bwd_dkdv_gfx950.py:1493`
- `fmha_bwd_dq_gfx950.py:985`

plus two tooling probes (`tooling/probe_tr16_lanemap_gfx950.py:65`,
`tooling/probe_kv_staging.py:66`) and comments. Nothing anywhere does
`dataclasses.replace(traits, …)` or `isinstance(x, …Traits)` — checked — so
nothing depends on the traits object's *type*, only on its attributes.

## 3. Target shape

One class per kernel, unchanged in name, holding the union of its own knob
fields and the traits fields, every field `None`-defaulted, every value a
literal:

```python
@dataclass(frozen=True)
class Gfx950Knobs(FmhaKnobs):
    # ---- build options (unchanged) ----
    num_waves: int | None = None
    block_m: int | None = None
    ...
    # ---- resolved traits fields, flat ----
    BLOCK_M: int | None = None
    BLOCK_N: int | None = None
    ...
```

`resolve()` fills the traits half in its last step. The object then *is* the
traits object as far as every consumer is concerned, because consumers only
read attributes.

## 4. Copy the fields, do not inherit — and the reason is the `None` default

Both options were offered. For a **single** class they are not equivalent, and
the deciding fact is in §2: the traits classes have **76 fields with no
default**, and the knob class must be constructible unresolved.

- **Inherit** `class Gfx950Knobs(ParityDualwaveTraits)`: the 76 non-defaulted
  base fields become 76 *required* constructor arguments, so
  `fmha_knobs('gfx950', block_m=128)` stops working. The fix is to redeclare
  each of the 76 in the subclass with `= None`, which is textually the same work
  as copying **and** keeps the import. Inheritance buys nothing here.
- **Copy**: declare the traits fields locally with `None` defaults. One class,
  plain data, no import of the traits class, constructible in either state.

So between those two: **copy**. This also matches the stated motive — freedom
from importing the traits class.

One honesty note, so it is not discovered later: copying frees the *field
declarations* from `fmha_traits_gfx950`, but the tuning modules still import
`make_traits` to **derive** the values. Removing that import too means moving
the derivation itself, which is a much larger change and is **not** in this
plan. If the intent is full decoupling, say so and it becomes a separate phase.

## 4.2 "Keep only the fields that are used" — measured, and it does not work

The obvious way to avoid a 120-field class is to copy only the traits fields the
code actually reads. **Measured, that set is nearly all of them, and it is
configuration-dependent** — which makes hand-curating it the most dangerous of
the options rather than the safest.

Two measurements:

- **Statically**, over `kernels/attention/parity/*.py`, `tooling/*.py` and the
  shared `../flash_attn_utils.py`: of the 100 field names across the three
  traits classes, **94 are referenced somewhere**. Only six are not:
  `DEFAULT_STRIDE_KV_N`, `DEFAULT_STRIDE_Q_N`, `QK_SHARDS`, `SMEM_K_PAD`,
  `SMEM_LINEAR_WAVE`, `SMEM_V_PAD`.
- **Dynamically**, with a recording proxy in place of the traits object and a
  real traced call — note a *build* alone reads only ~12, because the kernel
  body is traced lazily on first call, so any measurement that stops at
  `build()` is measuring nothing: a **single** forward configuration
  (head_dim 64, dense, bf16, no features) reads **67 of 93** fields.

67 is one point of a matrix whose axes are the ladder, two MFMA families,
causal, windows, varlen, dropout, bias, GQA, paged, split-K, `d_stages` and
`vo_shards`. The union over that matrix is higher; the fields not in it are
mostly ones some *other* configuration reads.

The failure mode of getting the list wrong is the thing to weigh. A trimmed
field set does not fail at review or at build — it fails as an `AttributeError`
at trace time, in whichever configuration first reads the missing field, which
by construction is a configuration nobody built while curating the list. That
is precisely the "unforeseeable trouble" a shorter list is meant to avoid.

**So do not trim by usage.** If the class must be smaller, shrink it by not
holding traits fields at all — option C below — rather than by holding a subset.

## 4.3 The three real options

| | knob class | who builds traits | PON keys | drift risk |
|---|---|---|---|---|
| **A** flatten all traits fields | ~120 fields | `resolve()` | 121, or 28 with a `pon()` filter | needs the §4.1 test |
| **B** flatten the "used" subset | ~90 fields | `resolve()` | ~90 | **high — see §4.2** |
| **C** gfx1201's shape | **29 / 16 / 31 fields, unchanged** | the builder, from `(meta, knobs)` | 28 / 15 / 30 naturally | none |

**Recommendation: C.** It is the smallest change, it removes the nested object
without adding 90 fields anywhere, it needs no anti-drift test because nothing
is copied, and it is what the reference architecture already does —
`fmha_tuning_gfx1201.py:438` holds build options only, and the builder derives
traits from `(meta, knobs)`. "The knob class is the build option class" is
exactly the property C has and A does not.

Its cost is one line moved into each of the three builders: where they do
`traits = knobs.traits` today, they call `make_traits(meta, knobs)`. That is
five call sites total (§5.2), and the derivation itself is untouched.

**C contradicts the instruction to make the knob class carry the traits fields**,
so it is offered rather than assumed. The reason to raise it: the requirement
behind that instruction was "GPU-facing code wants a traits object", and under C
that code still gets a real traits object — from the builder, one line earlier —
so the requirement is met without the knob class growing at all.

The handoff asks us *not* to take C ("please do not solve it by having
`resolve()` stop attaching traits"), but that was stated as their preference to
avoid churn on our side, not as a constraint on the design.

### 4.1 What makes copying safe: the anti-drift test

Copying 93–98 field declarations creates a real hazard — production adds a
field to `DualwaveSwpTraits`, the copies do not follow, and the knob object
quietly stops carrying it. One test kills that class of bug, and it must be
added **in the same commit** as the copy:

```python
def test_knobs_carry_every_traits_field():
    """The knob class is the traits object; a missing field is a silent gap."""
    for KnobCls, TraitsCls in ((Gfx950Knobs, ParityDualwaveTraits), ...):
        missing = {f.name for f in fields(TraitsCls)} - {f.name for f in fields(KnobCls)}
        assert not missing, f"{KnobCls.__name__} is missing traits fields: {sorted(missing)}"
```

and its value-level partner, which is the one that proves the refactor moved
nothing:

```python
def test_resolved_knobs_match_make_traits_field_for_field():
    """For every rung x kernel, the resolved knobs equal today's make_traits."""
```

## 5. Work items

### 5.1 Flatten — REQUIRED

Per module (`fmha_tuning_gfx950.py`, `fmha_tuning_bwd_dkdv_gfx950.py`,
`fmha_tuning_bwd_dq_gfx950.py`):

1. Delete the `traits: object | None = None` field.
2. Add the traits fields as local `None`-defaulted declarations.
3. `_with_traits` (or its backward equivalent) changes from
   `replace(self, traits=traits)` to
   `replace(self, **{f.name: getattr(t, f.name) for f in fields(TraitsCls)})`,
   where `t` is still whatever `make_traits` / `make_bwd_dq_traits` returned.
   **The derivation is not touched.**
4. Keep the guard that a caller may not pass traits fields as overrides — today
   it is `if "traits" in overrides: raise`; it becomes a check against the
   traits field-name set. Without it, a caller could hand-set `BLOCK_M` and get
   a configuration the derivation never produced.

### 5.2 Migrate the five consumers — REQUIRED

The three builders drop `traits = knobs.traits` and use `knobs` directly; the
`is None` guard becomes a check on a resolved marker field (e.g.
`knobs.BLOCK_M is None` means "not resolved"). Same for the two tooling probes.

### 5.3 `GRID_AXIS_ORDER` — REQUIRED by the handoff

AOTriton computes the launch grid in C++ and does not run our `@flyc.jit`
launcher, so any grid decision the launcher makes must be visible in the knobs.
gfx950 and gfx1201 differ today:

| | grid x | grid y | grid z |
|---|---|---|---|
| gfx1201 `launch_bwd_dkdv` | `num_kv_tiles` | `num_head_k` | `nseq_idx` |
| gfx950 `launch_fmha_bwd_dkdv_gfx950` | `num_head_k` | `num_kv_blocks` | `bs_idx` |

Add a small documented int enum, set in `resolve()` for **all three** kernels
(the forward and dQ have an order too — state it rather than leave it implied).
The mapping is the interface, so it belongs in the module docstring, together
with the measured reason gfx950 is head-fastest: MI355X's eight XCDs make this
an L2-locality lever and KV-fastest measured 12–15% slower at every rung.

### 5.4 The four `None` V-layout fields — OPTIONAL

`v_half_wave`, `v_n_group`, `v_k_substep`, `v_dc_in_pair` stay `None` after
`resolve()`, meaning "use the default formula". PON renders `None` fine and the
C++ side treats it as a lookup miss, so this blocks nothing. Writing back what
the formula produced makes the record say what was *used* rather than what was
*asked*. Cheap while `resolve()` is already being edited. Owner's call.

## 6. One consequence that needs a decision

Flattening puts every traits field on the wire:

| | keys | PON chars |
|---|---|---|
| knob fields only (what the handoff asked for) | 28 | 442 |
| flattened knobs + traits | **121** | **2343** |

and 13 of those are *functional*, not perf-selection: `NUM_HEADS_Q`,
`NUM_HEADS_KV`, `HEAD_DIM`, `CAUSAL`, `DTYPE_STR`, `VARLEN`, `BIAS_TYPE`,
`ENABLE_DROPOUT`, `PAGED`, `RETURN_LSE`, `CROSS_SEQLEN`, `PAGED_BT_LDS_SIZE`,
and the LSE-layout pair. AOTriton keys those separately as the functional, so
recording them inside the perf section means one knob set renders differently
per shape.

Three ways to go:

1. **Ship all 121.** Simplest; the record becomes "everything this hsaco was
   built with".
2. **Flat object, filtered wire** — add `pon()` emitting only the knob-side
   fields (28 / 15 / 30 keys). The object satisfies the design constraint, the
   wire stays what AOTriton asked for, and the method name they requested exists
   so their call site is stable. **Recommended.**
3. Knobs plus a named subset of traits the runtime actually needs.

This is the only open question in the plan.

## 7. Verification

1. **Field-for-field equality against `make_traits`**, every ladder rung
   (32…512) × all three kernels. This is what proves the refactor moved nothing
   and is the gate to run first. Under option C this is nearly free: the traits
   the builder derives must equal what `resolve()` used to attach.
2. The anti-drift field-set test from §4.1 — **only needed under A or B.** Under
   C nothing is copied, so there is nothing to drift.
3. The handoff's §5 acceptance test: `render_pon` succeeds, round-trips, is
   space-free, and `GRID_AXIS_ORDER` plus the grid input are present and not
   `None`, for every rung × kernel.
4. `assert_matches_production` must still pass — it iterates the *parent's*
   fields, so a flattened knob class has to keep satisfying it.
5. The suites: 1038 backward, 329 forward, 350 + 3 skipped with philox. The
   forward's bitwise-vs-production test is the one that matters most, since
   every other kernel subclasses its helpers.

## 8. Risks

- **Silent drift** if production gains a traits field — mitigated by §4.1, which
  is why that test ships in the same commit rather than after.
- **A caller setting a traits field as an override** would fabricate a
  configuration the derivation never produced. Mitigated by §5.1 item 4.
- **`None` now means two things** on one object: "policy decides" for knob
  fields and "not resolved yet" for traits fields. Worth one sentence in the
  class docstring rather than leaving a reader to infer it.

## 9. Not in scope

- `fx.Tensor` → `fx.Pointer` for the kernel operands (6 forward, 9 per backward)
  — specified separately in `sdpa-flyc-gfx950-integration.md`.
- The split-K / paged `Constexpr` kernarg fold: already landed in `75553a16`,
  and AOTriton now depends on it. Nothing to do; do not regress it.
- Moving the traits *derivation* out of `fmha_traits_gfx950` (see §4).

---

# 10. Outcome — option C, as built

The `traits` field is gone from all three knob classes. Each class gained a
public `build_traits(meta)` — **the one place knobs and traits are related** —
and `resolve`'s last step became `_checked_against_traits`, which calls it,
throws the traits away and keeps only the verdict.

Discarding the object is not waste: every check inside `build_traits` names the
knob to move (the LDS cap message is the clearest — it says which of
`d_stages` / `block_n` to change, where the compiler would only say
`local memory (N) exceeds limit (163840)`), so a bad configuration still fails
at `resolve` rather than at a kernel address. The builder calls `build_traits`
again for the object itself; that is a dataclass construction, not a compile.

Measured after:

| | keys | PON chars | `GRID_AXIS_ORDER` | grid input |
|---|---|---|---|---|
| forward | 29 | 459 | 0 | `block_m` = 256 |
| dK/dV | 16 | 259 | 0 | `block_kv` = 64 |
| dQ | 31 | 486 | 0 | `block_m` = 64 |

at head_dim 128; renders, round-trips and is space-free at **every** rung of the
ladder for all three kernels, which is the handoff's §5 acceptance test.

**The gate that proves nothing moved**: the traits derived by `build_traits`
were captured across 60 configurations (10 rungs x causal/dense x 3 kernels) and
compared field-for-field against the traits `resolve()` attached at `c36f1a7e`.
**Zero fields differ.**

Two things worth knowing for the next edit:

- `BwdDkDvKnobs` does **not** inherit `Gfx950Knobs` — only `BwdDqKnobs` does. So
  it needs its own copy of `_checked_against_traits` and its own
  `GRID_AXIS_ORDER` field. Both were missed on the first pass and both failed
  loudly (`AttributeError`, then `TypeError` from `replace`), which is the good
  kind of failure.
- `GRID_AXIS_ORDER` is spelled in upper case **against** this module's
  lower-case convention, because the field name *is* the wire key: the C++ side
  reads it back as `perf().get_int("GRID_AXIS_ORDER")`. The enum is documented
  where it is defined, in `fmha_tuning_gfx950.py`, since the mapping is the
  interface.

Not done, and deliberately: §5.4's write-back of the four `None` V-layout
fields. PON renders `None` and the C++ side treats it as a lookup miss, so it
blocks nothing; it is a fidelity nicety and was left out of a change whose
value is that it moves no numbers.

One unrelated fix rode along: `tooling/probe_tr16_lanemap_gfx950.py` had a
126-character line that predates this work. It was invisible to the style gate
because the gate only checks files in the diff, and touching the file for the
`build_traits` rename brought it into scope.
