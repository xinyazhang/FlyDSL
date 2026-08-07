# Executive plan: the `Aperture` refactor for gfx1201 SDPA

Self-contained. Everything needed to execute is here; nothing depends on
recalling how it was derived. Companion documents: `sdpa-common-preload.md`
(design narrative), `sdpa-fix-unstable.md` (API-stability follow-up),
`sdpa-readability-plan.md` (the parent plan; this is its Tier C/D).

Baseline: branch rebased onto `v0.3.0`, 298 tests green,
`flash_attn_func_gfx1201_aiw.py` at 2814 lines,
`fmha_common_gfx1201.py` at 875.

---

## 1. Goal

Move the memory-access layer of `flash_attn_func_gfx1201_aiw.py` into
`fmha_common_gfx1201.py` behind one abstraction, so that (a) the kernel reads
as the algorithm, (b) the backward kernels can reuse it, and (c) masking cannot
be forgotten.

**Non-goal:** making `kv_loop_body` short. Its length is mostly *interleaving*
(prefetch issue points, `sched_barrier` placement), which is the schedule and
must not move. See §9.4.

---

## 2. Vocabulary (final)

| concept | name | note |
| --- | --- | --- |
| bounded region of a tensor + its on-chip residence | `Aperture` | an opening through which a bounded part is visible |
| one bounded axis of it | `MaskedAxis` | already exists |
| values read out of an aperture | `LoadedRegion` | |
| row sub-tiles of the Q block, per wave | `ROW_SUBTILES` | today `Q_ROW_TILES` |
| column sub-tiles of the KV block | `COL_SUBTILES` | today `N_SUB_TILES` |
| columns per column sub-tile | `COLS_PER_SUBTILE` | today `K_SUB_N` |

Do **not** use `Tile`, `TileInfo`, `Fragment`, `Slab`, `Pane`, `Patch`,
`Panel`, `Operand`, `Block`, `Chunk`, `Stage`. Each was considered and
rejected: `Tile`/`Fragment` collide with CuTe/CUTLASS vocabulary that a future
cuTile port will claim; `MmaOperand` already exists in
`flydsl.expr.primitive.__all__`; the rest collide with this kernel's own words
(`BLOCK_M`, `VO_CHUNK_COLS`, prefetch stages) or connote a 1-D strip.

Class names describe residence and visibility; constant names describe
partitioning. Keeping those registers apart is what keeps this module out of
CuTe's namespace.

---

## 3. Environment facts (flydsl 0.3.0)

- `flydsl.expr.buffer_ops` **no longer exists**. Use the repo's
  `kernels/common/buffer_ops.py`, reached as
  `from gfx1201_standalone import buffer_ops`.
- `_LAZY_MODULES` is now `_BACKEND_MODULES`; `expr.rocdl` has gained
  `rdna3` / `rdna4` submodules (they contain only `s_waitcnt` helpers).
- gfx1201 now has `lib/Dialect/FlyROCDL/GFX120X/` with `MmaAtom.cpp`.
  **There is still no RDNA `CopyAtom.cpp`** (only CDNA3, CDNA4, GFX1250), so
  `fx.copy_atom_call` / `TiledCopy` cannot lower here. The layout API is not an
  option for load/store.
- `fx.make_tile` builds a CuTe **Tiler** (a partitioning of an index space).
  It is not a data container and is not what we need.

---

## 4. Hard constraints

### 4.1 Tracer: what may live in a module-level function

The AST rewrite is **lexical, per `@flyc.kernel` function**. In a module-level
helper:

- a Python `if` on a traced value **fails** with
  `cannot evaluate dynamic 'Boolean' as Python bool during tracing`;
- `range(start, stop, step, init=[...])` is **not** rewritten to `scf.for` and
  silently becomes a host-side Python loop;
- an explicit `scf.IfOp` **works**, because it needs no rewriting. This is how
  `cond_load` lives in the module.

Consequence: `_decode_side`-style branching code can move only if rewritten
onto explicit `IfOp`, and loop bodies with `range(init=)` cannot move at all.

### 4.2 Objects across a dynamic `if` — solved, use the protocol

A Python object live across a dynamic `if` fails (`state variable 'x' is …,
not an MLIR Value`; sometimes surfacing as `UnboundLocalError`). Passing it as
a *parameter* does not help.

flydsl 0.3.0 (PR #874) fixes this **and the fix is a stable API**: §2.2 of
`docs/api_stability.md` makes every name in `flydsl.compiler.protocol.__all__`
stable. Implement three methods:

```python
def __get_ir_types__(self): ...
def __extract_to_ir_values__(self): ...
@classmethod
def __construct_from_ir_values__(cls, values, exemplar=None): ...
```

`exemplar` restores non-IR fields (`active`, `elem_dtype`, LDS geometry) while
IR values thread through as `scf` state. Verified on a probe kernel.
**`Aperture` and `MaskedAxis` must implement these**; then the `qk_cols()` /
`vo_cols()` / `q_rows_axis()` factory functions are deleted.

### 4.3 Static selector vs traced predicate — do not confuse them

`KV_NEEDS_GUARD`, `V_NEEDS_GUARD`, `V_TR_NEEDS_GUARD` are **const_expr** and
harmless. They select whether a guarded form is emitted. The `scf.if` is the
*traced* predicate inside:

| static selector | traced predicate | site |
| --- | --- | --- |
| `KV_NEEDS_GUARD` | `row_valid = lds_row < BLOCK_N` | `coop_load_store_k`, `coop_store_k_lds` |
| `V_NEEDS_GUARD` | `row_valid = v_row_in_batch + off < BLOCK_N` | `coop_store_v_lds` row-major arm |
| `V_TR_NEEDS_GUARD` | `tile_ok = wave_id + l*NUM_WAVES < V_TR_TILES` | `coop_store_v_lds` transposed arm |

`_load_geom` is called separately for the QK and V/O widths, so a config can
need one guard without the other.

### 4.4 API stability

Prefer the stable spelling; new code must not add unstable surface.

| use | instead of |
| --- | --- |
| `fx.as_ir_value(x)` | `_to_raw(x)` / `_raw(x)` |
| `(a < b).select(x, y)` — `<` on `fx.Int32` returns stable `fx.Boolean` | `ArithValue(pred).select(...)` |
| the `<` operator on i32 | `arith.cmpi(slt, ...)` |
| `fx.ptr_load` / `fx.ptr_store` / `fx.recast_iter` (all stable) | `_llvm.LoadOp` / `StoreOp` |

No stable equivalent exists for `scf.IfOp`, `ir.InsertionPoint`,
`llvm.IntToPtrOp`/`PtrToIntOp`, or `fx.rocdl.global_load_tr_b128`. **Contain**
each in exactly one function. `fx.arith.index_cast` is deprecated for removal
in **v0.4**; do not add new uses.

---

## 5. The design

```python
@dataclass(frozen=True, slots=True)
class Aperture:
    rows: MaskedAxis       # extent = this tensor's seqlen; always active
    cols: MaskedAxis       # extent = this tensor's head_dim; active=PADDED_HEAD
    lds_base: int | None   # element offset; None if never staged
    lds_stride: int | None
    vec_width: int
    # + the three carry-protocol methods from §4.2
```

Four instances, one per tensor:

| aperture | rows extent | cols extent |
| --- | --- | --- |
| Q | `seqlen_q_i32` | `hdim_qk` |
| K | `seqlen_k_i32` | `hdim_qk` |
| V | `seqlen_k_i32` | `hdim_vo` |
| O | `seqlen_q_i32` | `hdim_vo` |

```python
@dataclass(frozen=True, slots=True)
class LoadedRegion:
    packs:     list[list[fx.Vector]]   # [row_subtile][k_step], masked
    rows:      list[fx.Index]          # absolute row per row-subtile
    rows_i32:  list[fx.Int32]          # same rows, signed, for the mask
    in_bounds: list[ArithValue]        # i1 per row-subtile
```

Field types were read off a traced build, not inferred. `rows` and `rows_i32`
are the same numbers at two widths and **both are required**: addressing needs
the 64-bit `fx.Index` because `toff` multiplies it by `stride_seq` (a product
that overflows 32 bits on a non-compact tensor); the causal mask needs signed
`fx.Int32` because `fx.Index` is unsigned and a negative bound compares as
enormous.

Movement functions — **no axis arguments**, the aperture knows its bounds, so
there is no unmasked spelling:

```python
load(aperture, ptr, tbase, toff, rows, rows_in_tile, col_base, ...) -> LoadedRegion
store(aperture, ptr, tbase, toff, values, rows, ...)
to_lds(aperture, lds_ptr, values, row, col, *, transposed=False)
from_lds(aperture, lds_ptr, row, col) -> list[fx.Vector]
stage(aperture, ptr, tbase, toff, lds_ptr, ...)      # vram -> lds
```

`rows` / `rows_in_tile` stay caller-computed: they encode the wave-to-row
mapping, a *schedule* decision (`ROW_SUBTILES`, `wave_q_offset`, `WMMA_M`) that
differs between fwd and bwd.

`stage` exists for portability, not deduplication: gfx1201 has no direct
VRAM→LDS instruction so it is load-then-store, while gfx950 and gfx1250 have
one. Naming it puts the seam where the ISA difference is.

---

## 6. Migration inventory

| current | calls | disposition |
| --- | --- | --- |
| `coop_load_v_global` | 6 | `load` (V) |
| `load_global_f16xN` | 5 | absorbed by `load` |
| `fmha.lds_store_vx` | 5 | absorbed by `to_lds` |
| `k_buf_base` | 4 | `Aperture.lds_base` |
| `coop_store_v_lds` | 4 | `to_lds` (V) |
| `v_buf_base` | 3 | `Aperture.lds_base` |
| `load_global_v8f16` | 3 | absorbed by `load` |
| `fmha.lds_load_v8` | 3 | `from_lds` |
| `coop_load_k_global` | 3 | `load` (K) |
| `_v_store_transposed` | 3 | `to_lds(transposed=True)` |
| `_v_store_row_major` | 3 | `to_lds` |
| `_split_ptr` | 3 | stays private, internal to `load` |
| `_load_global_half_vec` | 3 | absorbed |
| `coop_store_k_lds` | 2 | `to_lds` (K) |
| `coop_load_store_k` | 2 | `stage` |
| `_store_global_half` | 2 | `store` (O epilogue) |
| `fmha.global_load_tr_v8` | 1 | `load(transposed=True)` |

---

## 7. Steps

Each step is one commit, independently revertible, bitwise-gated.

| # | step | replaces | gate configs (§8) |
| - | --- | --- | --- |
| 1 | rename `Q_ROW_TILES`→`ROW_SUBTILES`, `N_SUB_TILES`→`COL_SUBTILES`, `K_SUB_N`→`COLS_PER_SUBTILE`; includes `FmhaKnobs.q_row_tiles` and its tuning table | — | A |
| 2 | `MaskedAxis` gains the §4.2 protocol; delete the `qk_cols()`/`vo_cols()`/`q_rows_axis()` factories, use plain variables | factories | B (16 both modes is the one that failed before) |
| 3 | `Aperture` dataclass; build the four instances; `lds_base`/`lds_stride` fields | `k_buf_base`, `v_buf_base` | A |
| 4 | `from_lds` / `to_lds` | `lds_load_v8`, `lds_store_vx`, `_v_store_row_major`, `_v_store_transposed` | C (needs V_TRANSPOSED both ways) |
| 5 | `load` for Q | the Q preload | D |
| 6 | `load` for K and V | `coop_load_k_global`, `coop_load_v_global`, `load_global_*` | C |
| 7 | `stage` | `coop_load_store_k` | B |
| 8 | `store` | `_store_global_half` | E |
| 9 | `map_subtiles` over row-subtiles for the 5 foldable `qt` loops only | — | A |

Steps 1–3 are prerequisites. 4–8 each touch one residence pair. Step 9 last.

---

## 8. Gate playbook

**Always** `export ROCM_PATH=$(rocm-sdk path --root)` first, or the JIT dies
with a bare `lld invocation failed`.

Config sets, chosen for the *arm* they select, not by habit:

| set | configs | why |
| --- | --- | --- |
| A | hd128 c0, hd128 c1 | baseline; pure-refactor steps |
| B | hd16 c0, **hd16 c1** | `KV_NEEDS_GUARD` true → the `row_valid` `scf.if` exists. hd16 **causal** is the one that caught the parameter-passing failure |
| C | hd64 c0, hd384 c0 | `V_TRANSPOSED` on vs off; hd384 also has 2 QK shards |
| D | hd128 c0/c1, hd100 padded, hd80, hd384 c1 | `rows_i32` reaching the mask; `PADDED_HEAD`; `ROW_SUBTILES==2`; the spilling build |
| E | hd100 padded | column masking on the store |

`PADDED_HEAD` configs need a padded allocation — a contiguous `(…,100)` tensor
is rejected by the pitch check. Allocate `(…, 104)[..., :100]`.

**Tests:** `cd kernels/attention && python3 -m pytest
test_flash_attn_func_gfx1201_aiw.py -q` (298 expected). Collection fails from
the repo root. Subsets: `-k varlen` (55), `-k "swa or window or gswa"` (69),
`-k causal` (116).

**ISA gate.** Dump `21_final_isa.s` by running one shape with
`FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<tmp> FLYDSL_RUNTIME_ENABLE_CACHE=0`, then
compare. Three rules learned the hard way:

1. **Re-dump the baseline from the actual parent commit.** A stale reference
   inverted a conclusion twice.
2. **Identify the KV loop by "contains `v_wmma`", never by "largest
   backward-branch region".** The size heuristic picked different loops in two
   builds and fabricated a 20-instruction regression that did not exist.
3. Compare the loop's **opcode sequence**, not just its length; register and
   branch-label numbering differ harmlessly.

**Perf.** `kernels/attention/perf_ab.py --base <rev>` is tier 2.9 and has
produced **two false regressions** (hd192 ≈0.96, hd64 ≈0.97) that the full
ladder read as neutral. Never revert on tier 2.9 alone. Use `--full`, and run
`--base X --head X` to establish the per-config self-test floor first — hd192
carries a systematic ~2.5% bias.

---

## 9. Measured facts that must not be broken

### 9.1 `rows_i32` live range
Building the i32 row copy *before* the bounds test starts its live range early
and cost **16 bytes of scratch and 6%** at BLOCK_DMODEL 384 causal. The bounds
test must use the index-typed row. If step 5 moves hd384 c1, this is why.

### 9.2 `kv_addr_hoist`
A knob, not a preference. Hoisting `row * stride_seq` out of the KV loop wins
up to 1.41× at 192 and loses to 0.59× at 384; the policy is measured per width
in `_KV_ADDR_HOIST_HEAD_DIMS`. `load`/`stage` must preserve both forms.

### 9.3 `width` is per access, not per axis
The cooperative loads are `VEC_WIDTH` wide; the Q preload is 8 wide because
`load_global_v8f16` matches the WMMA operand. Both are 8 today **from
unrelated definitions**. Do not bind `width` into `Aperture`; keep it a call
argument. (`vec_width` in `Aperture` is the LDS staging width only.)

### 9.4 The GEMM `qt` loops must not fold
In GEMM1 and GEMM2 the row-subtile loop is deliberately the *innermost* one,
inside the `ks` / V-chunk unroll, so each K operand feeds every row-subtile
while live in registers. That nesting is what `ROW_SUBTILES == 2` buys. Step 9
touches only the 5 map-shaped sites; these two keep explicit loops.

### 9.5 The out-of-range arm of `kv_off`
Redirects to `col` and not to literal `0`. The `0` arm holds one value fewer
live and still spills **more** — 272 B vs 44 B of scratch at 192, for 0.863 vs
1.172. Re-measure before changing.

---

## 10. Out of scope

- The staging/prefetch call *placement* in `kv_loop_body` — that is the
  schedule, and a full offload needs a cross-architecture framework designed
  against gfx950/gfx1201/gfx942/gfx1250 together.
- `_decode_side`-style code moving without an `IfOp` rewrite (§4.1).
- `pa_metadata.py`'s 83 upstream `arith.*` sites — another kernel's code.
- Making the module stable-only — impossible while `scf.IfOp` and the RDNA
  transpose load have no stable spelling.
- `transposed=True` on the VRAM side beyond V, until a bwd kernel needs K^T.

---

## 11. Definition of done

- The four apertures exist and every load/store in §6 goes through one.
- No `rows_axis=` / `cols_axis=` argument survives; masking is implicit.
- The three factory functions are gone.
- 298 tests pass; the full ladder is within noise of the pre-refactor revision.
- Re-run the `api-stability` skill and record the delta in unstable **call
  sites** — the number this work should move.
