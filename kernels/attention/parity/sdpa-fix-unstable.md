# Plan: reduce the gfx1201 SDPA kernel's unstable-API surface

**Status: executed, `3f051d01..d0976d2a`, eight commits.** Every P1-P4 item
landed except the two explicitly scoped to other owners. Unstable call sites in
the two gfx1201 files went **146 -> 52**; see §9.

Derived from the `api-stability` consumer audit against `docs/api_stability.md`
at v0.3.0. **Revised after the `Aperture` refactor** (`sdpa-common-preload-
executive.md`, `a40cd565..aed002ab`), which moved most of the memory-access
layer into `fmha_common_gfx1201.py` and invalidated every line number in the
first draft. Four of the first draft's claims were also wrong; §7 lists them.

## 0. Why this is a plan and not a cleanup

An in-tree kernel is allowed to use unstable FlyDSL APIs; the audit result
`NOT STABLE-ONLY` is not by itself a defect. Two things make it worth planning:

1. **APIs we use are declared for removal in v0.4**, and we are on v0.3.0.
   That is a deadline, not a preference.
2. AOTriton is meant to dispatch this kernel's hsaco directly. Every unstable
   dependency is a thing that can break the build during an integration we do
   not control the timing of.

Everything else is triage: prefer the stable spelling where one exists at equal
cost, and *contain* what has no stable equivalent so the migration surface is
small.

## 1. How to classify (read this before adding a row)

**Do not read stability off `scripts/list_stable_apis.py` alone.** The catalog
excludes §3 deprecated APIs, so absence from it means unstable *or*
deprecated -- and those need opposite responses. Check the owning module's
`__all__`:

| in `arith.__all__`? | in §3 table? | verdict |
| --- | --- | --- |
| yes | no  | stable |
| yes | yes | deprecated, has a deadline |
| no  | --  | unstable |

Current facts for the symbols this kernel touches:

| symbol | verdict |
| --- | --- |
| `cmpi`, `FastMathFlags`, `RoundingMode`, `fastmath` | stable |
| `maxnumf`, `maximumf`, `minimumf`, `shrui`, `cmpf` | stable |
| `index_cast`, `constant_vector` | deprecated, v0.4 |
| `addi`, `andi`, `ori`, `addf`, `subf`, `mulf`, `divf` | unstable |
| `CmpIPredicate`, `MaxNumFOp` | unstable |
| `rocdl.log`, `rocdl.exp2`, `rocdl.global_load_tr_b128` | unstable |
| `flydsl.expr.typing.as_ir_value` | stable |

**Signedness rides on the type.** `_make_binop` in `expr/numeric.py` passes
each operand's class-level `signed` into the comparison, so the type picks the
opcode:

| type | width | `signed` | `<` emits |
| --- | --- | --- | --- |
| `fx.Index` | 64 | **False** | unsigned |
| `fx.Int64` | 64 | True | signed |
| `fx.Int32` | 32 | True | signed |

This is why "use the `<` operator" is not always free: on `fx.Index` it
changes the opcode. It already did once -- the O store in the `Aperture` work
turned eight `v_cmp_gt_u64` into `v_cmp_gt_i64`.

**Counting caveat.** `rocdl.WAVES_PER_EU`, `rocdl.FLAT_WORK_GROUP_SIZE` and
`gpu.func` in `flash_attn_func_gfx1201_aiw.py` are MLIR attribute-name and
op-name **string literals**, not Python attribute accesses. A regex over
`rocdl\.[a-z_]*` counts them as API uses. They are raw MLIR attribute
manipulation -- also unstable, but a different thing, and not fixable by a
spelling change.

## 2. P1 -- the v0.4 removals (hard deadline)

| deprecated API | sites | replacement | status |
| --- | --- | --- | --- |
| `fx.index_cast` | `fmha_common:124,182,183`; `aiw:808` | `fx.Int64(x)` | **done**, `aed002ab` |
| `fx.index_cast` | `kernels/common/mem_ops.py:66` | `fx.Int64(x)` | open -- shared file, §6 |
| `fx.Numeric.shrui` | `fmha_common:195` | `fx.arith.shrui(x, n)` | open, mechanical |
| `fx.Numeric.shuffle_xor` | `aiw:1297` | `fx.gpu.shuffle_xor(x, off, width)` | open, mechanical |
| `fx.Numeric.maximumf` | `pa_metadata:2157` | `fx.arith.maximumf(x, y)` | not ours, §6 |
| `fx.constant_vector` | `pa_metadata:630,761,1803` | `Numeric` / `Vector` members | not ours, §6 |

**`index_cast` is resolved, and the first draft had it wrong.** It was flagged
as the unknown-scope item because §3 names `fx.Index(x)` as the replacement,
which converts *to* index while all our uses go index -> i64. The successor for
that direction is **the target type's constructor**: `Integer.__init__` in
`expr/numeric.py` tests `isinstance(x.type, ir.IndexType)` and routes through
the same private `utils.arith.index_cast` the public entry point wraps. So the
deprecation retires the *public spelling*, not the operation, and
`fx.Int64(v)` emits exactly what `arith.index_cast(T.i64, v)` did -- verified
bitwise on eight configs.

Gate: bitwise per site. These are type plumbing and none should move an
instruction.

## 3. P2 -- stable swaps

Ordered by call sites removed per edit.

### P2.1 `FastMath` onto the stable `fastmath` context (highest value)

`FastMath` is five methods and every one of them uses an unstable builder:
`arith.addf`, `subf`, `mulf`, `divf` and `arith.MaxNumFOp`. None is in
`arith.__all__`. This is the single largest concentration of unstable calls in
a container small enough to change in one edit.

The stable form is the one §3 already names as the replacement for the
deprecated `Numeric.addf` -- *"`x + y` with a `fastmath` context"*:

```python
def add(self, a, b):
    with arith.fastmath(self.flags):
        return fx.Float32(a) + fx.Float32(b)
```

`arith.fastmath` is a stable context manager (`expr/arith.py`, in `__all__`);
`arith.maxnumf` is stable and covers `max`. Removes 5 unstable APIs and 10
`_to_raw` calls.

**Highest risk item in this plan** and it must be gated hardest: these are the
softmax inner-loop ops, the flag set is load-bearing (`ninf` once silently
deleted the KV tail mask), and the context-manager form has to prove it
attaches the *same* flags as the explicit `fastmath=` keyword. Gate at
BLOCK_DMODEL 16, 128 and 384 in both masking modes, and diff the emitted
`fastmath<...>` attributes in the MLIR dump, not just the ISA.

### P2.2 `_to_raw` / `_raw` -> `fx.as_ir_value`

54 sites (29 in `fmha_common`, 25 in `aiw`). `as_ir_value` is a superset of
`_to_raw` for everything we pass it: both return `ir.Value` unchanged and both
fall back to `ir_value()`. It additionally handles `None`, sequences and
Python literals, and consults `__extract_to_ir_values__` first -- which for a
`Numeric` returns `[self.ir_value()]`, the same answer.

Mechanical, but 54 sites, so do it in two commits (one file each) and gate each.

### P2.3 `ssel` / `smin` / `smax` in `kernels/common/utils.py`

Three lines, shared with `pa_metadata`, and it converts the most call sites per
edit outside this kernel. `<` on `fx.Int32` returns `fx.Boolean`, whose
`select` is stable, so the module becomes stable-only and loses its
`ArithValue` import.

### P2.4 `MaskedAxis.valid` -- needs a retype, not a swap

`fmha_common:336`:

```python
return arith.cmpi(arith.CmpIPredicate.slt, _to_raw(idx), _to_raw(self._bound()))
```

`arith.cmpi` is **stable**; `arith.CmpIPredicate` is **not**. So the thing to
remove is the enum, and the way to remove it is the `<` operator -- but `<` on
`fx.Index` is unsigned, which is exactly why this site spells `slt` explicitly.

The fix is therefore a retype, not a substitution: hold the axis extent and the
compared index as `fx.Int64` so `<` emits the signed compare by itself. That
collapses `cmpi` + `CmpIPredicate` + two `_to_raw` into one operator, and
`.select` on the resulting `fx.Boolean` removes the `ArithValue` wrappers in
`safe`, `gate` and `read_v8` downstream.

**Open question to settle first:** `fx.Int64(index_value)` inserts an
`index_cast`, which is a no-op on a 64-bit target but has to be *shown* to fold
rather than assumed. Run that experiment before committing to the retype; if it
does not fold, this row stays as it is and the `CmpIPredicate` dependency
stands. `_bound()` and `safe()` also feed addressing, which wants `fx.Index`,
so check whether the round trip survives.

Gate at hd100-padded and hd7-in-16, the configs where `cols.active` is true.

## 4. P3 -- the private-field write

`exe._cf` -- `aiw:157,160` and the shared `kernels/common/tensor_shim.py:26,31`.

Attaches an attribute FlyDSL does not define to a FlyDSL-owned jit object. Two
concrete hazards, not just a style point: the cache lives on a module-level
object so it persists across every configuration in the process, and it
bypasses `flyc.compile`'s own specialisation on the second and later calls.

There is no exported "compile once, then fast-dispatch" handle in
`flydsl.compiler.__all__`, which is why the workaround exists. So P3 is:

1. check whether `flyc.compile` already caches, making the wrapper redundant;
2. if it does not, hold the `CompiledFunction` in a module-level dict keyed by
   the jit object rather than writing onto it -- same effect, no FlyDSL
   internals touched;
3. fix `tensor_shim.py` too, since it is the shared copy and other kernels
   inherit the pattern.

## 5. P4 -- contain what cannot be replaced

No stable equivalent exists for these. The goal is not removal but a small,
named blast radius, so a future FlyDSL change touches one function each.

| unstable dependency | contained in | status |
| --- | --- | --- |
| `scf.IfOp` / `YieldOp` / `ir.InsertionPoint` | `_over_batches:511`, `publish_transposed:709`, `write_v8:738`, `cond_load:857` | 4 sites, all in `fmha_common` |
| `llvm.IntToPtrOp` (addrspace 3) | `lds_f32_ptr` | contained |
| `llvm.IntToPtrOp` (global) | `pointer_to_llvm_ptr` | contained |
| `fx.rocdl.global_load_tr_b128` | `global_load_tr_v8` | contained |
| `rocdl.log` / `rocdl.exp2` | `aiw:1734,1743` and the LSE epilogue | not contained -- wrap if a stable `fx.math` equivalent appears |
| `llvm.LoadOp` / `StoreOp` | `cond_load`, `lds_f32_*`, `_pointer_load/store` | **partly**: revert `seqinfo_addr` to the stable `fx.recast_iter` + `fx.ptr_load` it used before `cond_load` |
| `arith.addi` / `andi` / `ori` | scattered | shrinks after P2.1; the rest are integer address maths with no stable builder |
| `CompilationContext` | one import in `aiw` | acceptable |
| `ArithValue` | 19 sites | shrinks to near zero after P2.1, P2.3 and P2.4 |

The `scf.IfOp` count grew from 1 to 4 during the `Aperture` work, and that is
the right trade rather than a regression: those three new sites are what let
the *kernel* stop open-coding guarded and unguarded arms. All four are in one
module, which is what "contained" is supposed to mean.

`flydsl.compiler.protocol` is no longer used at all -- the carry protocol was
removed once the branching helpers moved to module scope
(`sdpa-common-preload-executive.md` §12.6).

## 6. Out of scope

- **`pa_metadata.py`.** A different kernel with its own owner and no gfx1201
  stake. Its deprecated uses are listed in P1 only because they share the v0.4
  deadline; raise them with that owner rather than churning the file.
- **`kernels/common/mem_ops.py`.** Shared with GEMM and MoE. Its one
  `index_cast` is a P1 deadline item, so it cannot simply be ignored -- but the
  edit belongs to whoever owns that file. Flag it; do not unilaterally change
  shared infrastructure.
- **Making the kernel stable-only.** Not achievable while `scf.IfOp` and the
  RDNA transpose load have no stable spelling, and not a goal in itself.

## 7. Corrections to the first draft

Recorded so the same mistakes are not re-derived:

1. **`index_cast` was called "not a drop-in" with unknown scope.** It is a
   drop-in via `fx.Int64(x)`; the §3 table simply names the other direction.
   Done in four sites for zero instruction change.
2. **"`arith.cmpi(slt, ...)` -> the `<` operator" was listed as a free swap.**
   `cmpi` is already stable; the unstable part is `CmpIPredicate`, and on
   `fx.Index` the operator changes signedness. It is a retype (P2.4), not a
   swap.
3. **`FastMath` was not in the plan at all.** It is the largest single
   concentration of unstable calls, and §3 already documents the stable form.
4. **Stability was read off the generated catalog.** That silently conflates
   unstable with deprecated; §1 gives the correct procedure.

## 8. Sequencing and gates

    P2.3 ssel -> P1.mechanical -> P2.2 as_ir_value -> P2.1 FastMath
        -> P2.4 MaskedAxis retype -> P3 -> P4.seqinfo_addr

`ssel` first: three lines, shared, most call sites converted per edit. The P1
mechanical items next because they carry the deadline and are trivial.
`as_ir_value` before `FastMath` so the latter is written in the stable dialect
once rather than twice. `P2.4` last of the P2 items because it depends on an
experiment that may say no.

Every step is bitwise-gated. Two are not expected to be free and must not be
waved through:

- **P2.1 `FastMath`** -- diff the `fastmath<...>` attributes in the MLIR dump
  as well as the ISA. Softmax inner loop.
- **P2.4 `MaskedAxis`** -- signedness change by construction. Gate at
  hd100-padded and hd7-in-16.

Gate configs, dump procedure and the three ISA-comparison rules are in
`sdpa-common-preload-executive.md` §8; do not re-derive them. Re-run the
`api-stability` skill at the end and record the delta in unstable **call
sites**, which is the number this plan is trying to move: 54 `_to_raw` + 19
`ArithValue` + 5 `FastMath` builders is the bulk of it.

---

## 9. Outcome

| unstable use                                         | before | after |
| ---------------------------------------------------- | -----: | ----: |
| `_to_raw` / `_raw`                                   |     54 |     0 |
| `ArithValue`                                         |     19 |     0 |
| `arith.CmpIPredicate`                                |      6 |     0 |
| `arith.andi` / `ori`                                 |      6 |     0 |
| `arith.addf/subf/mulf/divf`, `MaxNumFOp`             |      5 |     0 |
| deprecated (`index_cast`, `.shrui`, `.shuffle_xor`)  |      6 |     0 |
| private-field write                                  |      1 |     0 |
| `arith.addi`                                         |      3 |     3 |
| `llvm.*` ops                                         |      8 |     7 |
| `scf.*` ops                                          |      6 |     6 |
| raw `ir.*`                                           |     32 |    32 |
| `rocdl.*` ops                                        |      4 |     4 |
| **total**                                            |    146 |    52 |

Counted over `fmha_common_gfx1201.py` and `flash_attn_func_gfx1201_aiw.py`
with comments and docstrings stripped, so the prose *about* unstable APIs is
not counted as a use of one. **No v0.4 deprecation remains in either file.**

392 tests pass. Every step was bitwise-gated and only one moved an
instruction: `_dead | (...)` emits `v_or_b32 v97, v0, v97` where the
hand-built `arith.ori` emitted `v97, v97, v0`. OR is commutative and the
opcode histogram, register counts and scratch all match. The full ladder
against `a40cd565` (which also includes the whole `Aperture` refactor) has a
worst point of 0.990.

### 9.1 What is left, and why

- **`raw ir.*` (32)** is the largest remaining block and the least like the
  others. 24 of them are `ir.StringAttr` / `ir.ArrayAttr` / `ir.IntegerAttr`
  in the aiw kernel, building MLIR *attributes* by name -- `rocdl.WAVES_PER_EU`,
  `amdgpu-flat-work-group-size`. That is not an API call with a stable
  spelling to switch to; it is a dependency on MLIR attribute names, and
  removing it needs FlyDSL to expose those launch bounds as first-class knobs.
  The rest are `ir.F32Type.get()` and `ir.Type.parse`.
- **`scf.*` (6) and `llvm.*` (7)** are contained by design, in `_over_batches`,
  `publish_transposed`, `write_v8`, `cond_load`, `lds_f32_*` and
  `pointer_to_llvm_ptr`. `lds_f32_*` needs the raw load/store because it
  aliases f32 scratch over the 16-bit KV tile and there is no retyped view of
  a shared pointer.
- **`arith.addi` (3)** is pointer arithmetic inside `global_load_tr_v8` and
  `lds_f32_ptr`, both already contained.
- **`rocdl.*` (4)** are `log`, `exp2` and the RDNA transpose load. No stable
  equivalent.

### 9.2 Deliberately not done

- **`kernels/common/mem_ops.py:66`** still has a v0.4 `index_cast`. It is
  shared with GEMM and MoE; the one-line fix is `fx.Int64(x)`, verified here,
  but the edit belongs to that file's owner. **This one has a deadline** --
  raise it.
- **`pa_metadata.py`** keeps its `maximumf` and `constant_vector`, same
  deadline, same reason.
- **`kernels/common/tensor_shim.py`** keeps `exe._cf`, and this is a stronger
  "no" than the other two. There `_cf` is a de-facto public contract:
  `tests/kernels/test_rmsnorm.py` asserts on `launcher._cf` in five places and
  `kernels/norm/rmsnorm_kernel.py` documents it. Changing it needs a CDNA box
  and its owner.

### 9.3 Two findings worth keeping

**The `_cf` cache is worth 157 us, not zero.** P3 opened with "check whether
`flyc.compile` already caches, making the wrapper redundant". It does --
`JitFunction._mem_cache`, keyed on the full argument signature -- so the
tempting conclusion was to delete the wrapper. Measured against a plain
`exe(*args)`: 63 us versus 227 us at (head_dim 64, N 512), 3.5x. The cache is
about dispatch overhead, not recompilation, and only its *storage* was the
problem.

**Gate `FastMath` on the MLIR, not the ISA.** The risk there was a dropped or
widened fastmath flag, which the ISA shows only indirectly. `00_origin.mlir`
is byte-identical, with 169 float ops carrying the same flag set, and all
three `fp_mode`s were checked by hand because none has test coverage:

    noninf  fastmath<reassoc,nnan,nsz,arcp,contract,afn>   (default)
    fast    fastmath<fast>
    safe    fastmath<reassoc,nsz,arcp,contract,afn>

"fast" carries `ninf`, which once silently deleted the KV tail mask, so a
quiet widening there would be a correctness bug no test would catch.
