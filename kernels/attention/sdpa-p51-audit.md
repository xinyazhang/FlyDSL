# P5.1 duplication audit + Phase 4 investigations

Gate for P5.2/P5.3/P5.5. Every row was checked against the tree, not against
the earlier §7.1 table, which had two wrong entries.

## 1. AIW helpers vs the shared modules

| AIW helper                        | counterpart                            | verdict |
| --------------------------------- | -------------------------------------- | ------- |
| `_pointer_load` / `_pointer_store`| `flash_attn_utils.py:523,527`          | **verbatim duplicate**; identical bodies, only the `llvm` vs `_llvm` alias differs |
| `_llvm_value`                     | `flash_attn_utils.py` `_llvm_value`    | **verbatim duplicate** |
| `pointer_to_llvm_ptr`             | `mem_ops.get_llvm_ptr`                 | **not a duplicate** -- opposite halves of a pair (see §1.1). Moved to `fmha_common.py` |
| `_ssel_i32`/`_smin`/`_smax`/`_sdiv_rd` | `common/utils`                    | **done** -- P5.4 |
| `wmma_acc`                        | `common/mma/wmma_ops.py`               | **done** -- `49764d29` |
| `_lds_load_v8` / `_lds_store_vx`  | none                                   | no LDS half exists in `mem_ops`; new code, not a move |
| `_global_load_tr_v8`              | none                                   | `global_load_tr_b128` is RDNA-only; no shared home |
| `bf16_trunc_pack_v8`              | `flash_attn_utils._bf16_trunc_pack_v8` | **unchecked** -- compare before moving either way |
| `_fadd`/`_fsub`/`_fmul`/`_fmax`   | `flash_attn_utils._fadd`...            | near-duplicate: theirs take `fm_fast` as a *parameter*, ours closes over it |
| `coop_load_*` / `coop_store_*`    | none in FlyDSL                         | AOTriton's `composed_tensors.py` is the conceptual match, not importable |
| `_col_safe`/`_col_mask`/`_apply_col_mask` | none                           | attention-specific -> `fmha_common.py` |
| `_red_addr`/`_red_store`/`_red_load` | none                                | cross-shard reduction -> `fmha_common.py` |
| `_seqinfo_at` / `_decode_side`    | none                                   | varlen ABI -> `fmha_common.py` |

### 1.1 Why `pointer_to_llvm_ptr` is not `get_llvm_ptr`

They start from different operand types and neither op accepts both:

- `get_llvm_ptr` -> `extract_aligned_pointer_as_index`, which **requires a
  memref** and rejects `!fly.ptr` (`op operand #0 must be ..., but got
  '!fly.ptr<i8, global>'`).
- `pointer_to_llvm_ptr` -> `fx.ptrtoint`, which reads `ptr.address_space` and
  so **requires the pointer**.

The house idiom for a tensor argument is the former, confirmed in
`flash_attn_utils._extract_aligned_pointer` and `_make_raw_buffer_rsrc`;
`fx.ptrtoint` appears there only on LDS pointers and iterators. Converting the
SDPA kernel to `fx.Tensor` arguments to reach it was measured and rejected --
each tensor adds a 40-byte `by_value` memref descriptor interleaved after its
pointer, 268 -> 428 kernarg bytes with every later offset shifted, and AOTriton
dispatches this hsaco directly. See `fmha_common.pointer_to_llvm_ptr`.

## 2. P4.2 -- can `fx.add_offset` replace the LDS `ptrtoint` + `addi`?

**Scope is one function, not the LDS layer.** `_lds_load_v8`, `_lds_store_vx`
and `_v_store_transposed` already do plain pointer arithmetic
(`lds_kv + fx.Int32(idx)`). The only `ptrtoint` + `addi` + `inttoptr` chain is
`_red_addr`.

`fx.add_offset(ptr, offset)` exists and shifts by **elements of the pointer's
own element type**. That is the obstacle, and it is not address space:
`_red_addr` is a *type pun*, an f32 view over the f16 `kv` array, indexed in
bytes. `add_offset` cannot express "reinterpret as f32, then index by f32
elements".

**Plausible but unproven:** both `_RED_BYTE0` and the `i*4` stride are even, so
the offset is expressible in f16 elements, and `fx.make_ptr` with an explicit
`elem_ty` is how `flash_attn_utils._make_page_view` builds a retyped pointer.
`add_offset` in f16 elements followed by `make_ptr` to f32 would avoid the
round trip. Needs a spike; the payoff is one function, so schedule accordingly.

## 3. P4.3 -- can the layout API replace `coop_load/store_*`?

**No, and it is a compiler-level answer rather than a kernel-design one.**
Atom lowering is per-subtarget in `lib/Dialect/FlyROCDL/`:

| subtarget | MmaAtom | CopyAtom |
| --------- | ------- | -------- |
| CDNA3     | yes     | yes      |
| CDNA4     | yes     | yes      |
| GFX11     | yes     | **no**   |
| GFX1250   | yes     | yes      |

There is no `CopyAtom.cpp` on the RDNA path, matching CLAUDE.md's "CopyAtom
(Buffer/LDS, CDNA3/4 only)". So `fx.copy_atom_call` has nothing to lower to on
gfx1201, and the hand-rolled cooperative load/store family is not a stylistic
holdout -- it is the only thing available on this target.

**Consequence for the plan:** P5.3 and the copy half of P5.5 lose their
"replace with the layout API" option and become plain moves, if they happen at
all. P4.3 is answered; no spike needed.
