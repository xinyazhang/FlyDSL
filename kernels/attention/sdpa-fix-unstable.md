# Plan: reduce the gfx1201 SDPA kernel's unstable-API surface

Follows `sdpa-common-preload.md`. Derived from the `api-stability` consumer
audit of this branch against `docs/api_stability.md` at v0.3.0.

## 0. Why this is a plan and not a cleanup

An in-tree kernel is allowed to use unstable FlyDSL APIs; the audit result
`NOT STABLE-ONLY` is not by itself a defect. Two things make it worth planning:

1. **Five APIs we use are declared for removal in v0.4**, and we just moved to
   v0.3.0. That is a deadline, not a preference.
2. AOTriton is meant to dispatch this kernel's hsaco directly. Every unstable
   dependency is a thing that can break the build during an integration we do
   not control the timing of.

Everything else is triage: prefer the stable spelling where one exists at equal
cost, and *contain* what has no stable equivalent so the migration surface is
small.

## 1. P1 -- the v0.4 removals (hard deadline)

Declared in §3 of the policy with "Declared removal release: v0.4".

| deprecated API            | sites                                                    | replacement | note |
| ------------------------- | -------------------------------------------------------- | ----------- | ---- |
| `fx.arith.index_cast`     | `fmha_common:120,178,179`; `aiw:806`; `mem_ops:66`        | `fx.Index(x)` | **not a drop-in** -- see below |
| `fx.Numeric.shrui`        | `fmha_common:191`                                          | `fx.arith.shrui(x, n)` | mechanical |
| `fx.Numeric.shuffle_xor`  | `aiw:1371`; `pa_metadata:592,659,685`                      | `fx.gpu.shuffle_xor(x, off, width)` | mechanical |
| `fx.Numeric.maximumf`     | `pa_metadata:2157`                                         | `fx.arith.maximumf(x, y)` | mechanical |
| `fx.arith.constant_vector`| `pa_metadata:630,761,1803`                                 | `Numeric`/`Vector` members | needs a look |

**`index_cast` is the awkward one.** The declared replacement is `fx.Index(x)`,
which produces an *index*. Every one of our five uses goes the other way --
`index_cast(T.i64, <index>)`, index -> i64 -- because the value then feeds an
`llvm` pointer builder that wants i64. So the replacement does not fit, and
the work before v0.4 is to find the stable spelling for "index as i64", or to
establish that these sites should hold `fx.Int64` end to end instead. Do this
first; it is the only item with unknown scope.

Gate: bitwise per site. All five are type plumbing and none should move a
single instruction.

## 2. P2 -- free stable swaps

No behaviour change, no deadline, but do them before the `Tile` work so new
helpers are written in the stable dialect (`sdpa-common-preload.md` §15-16).

| from                            | to                       | sites |
| ------------------------------- | ------------------------ | ----- |
| `_to_raw` / `_raw`              | `fx.as_ir_value`         | pervasive: ~40 in aiw, ~15 in fmha_common |
| `ArithValue(pred).select(a, b)` | `(a OP b).select(a, b)`  | `common_utils.ssel` and its callers |
| `arith.cmpi(slt, ...)`          | the `<` operator          | wherever both operands are already i32 |

`ssel`/`smin`/`smax` in `kernels/common/utils.py` become stable-only outright:
`<` on `fx.Int32` returns `fx.Boolean`, a stable type whose `select` is a
stable member. That also removes the `ArithValue` import from that module.

**Order matters:** do `ssel` first. It is three lines, it is shared with
`pa_metadata`, and it converts the most call sites per edit.

## 3. P3 -- the private-field write

`exe._cf = cf` -- `aiw:160` and the shared `kernels/common/tensor_shim.py:31`.

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

## 4. P4 -- contain what cannot be replaced

No stable equivalent exists for these. The goal is not removal but a small,
named blast radius, so a future FlyDSL change touches one function each.

| unstable dependency                            | contain in            | status |
| ---------------------------------------------- | --------------------- | ------ |
| `scf.IfOp` / `YieldOp` / `ir.InsertionPoint`   | `cond_load`           | already contained |
| `llvm.IntToPtrOp` / `PtrToIntOp` (addrspace 3) | `lds_f32_ptr`         | already contained |
| `llvm.IntToPtrOp` (global)                     | `pointer_to_llvm_ptr` | already contained |
| `fx.rocdl.global_load_tr_b128`                 | `global_load_tr_v8`   | already contained |
| `llvm.LoadOp` / `StoreOp`                      | `cond_load`, `lds_f32_*`, `_pointer_load/store` | **partly**: revert `seqinfo_at` to the stable `fx.recast_iter` + `fx.ptr_load` it used before `cond_load` |
| `CompilationContext`                           | one import in aiw      | acceptable |
| `ArithValue`                                   | -- | shrinks to near zero after P2 |

The audit's own framing applies: containment is the deliverable, and the count
of unstable *call sites* matters more than the count of unstable *APIs*.

## 5. Out of scope

- **`pa_metadata.py`'s 83 upstream `arith.*` sites.** A different kernel with
  its own owner and no gfx1201 stake. Its four deprecated uses are listed in P1
  only because they share the v0.4 deadline; the rest is not ours to churn.
- **`kernels/common/mem_ops.py`'s `_fly` binding.** Shared infrastructure used
  by GEMM and MoE kernels; changing it is a tree-wide decision.
- **Making the kernel stable-only.** Not achievable while `scf.IfOp` and the
  RDNA transpose load have no stable spelling, and not a goal in itself.

## 6. Sequencing and gates

    P2.ssel -> P2.rest -> P1.mechanical -> P1.index_cast -> P3 -> P4.seqinfo_at

P2 first because it is free and shapes the `Tile` work. `P1.index_cast` last
among the P1 items because its scope is unknown until investigated.

Every step is bitwise-gated -- these are all spelling changes and none should
move an instruction. The one to watch is `P2.ssel`: it changes how the
predicate is built (`Boolean` rather than `ArithValue`), so it is the only item
that plausibly perturbs codegen. Gate it at BLOCK_DMODEL 16, 128 and 384, both
masking modes, before assuming the rest are free.

Re-run the `api-stability` skill at the end and record the delta in unstable
call sites, which is the number this plan is actually trying to move.
