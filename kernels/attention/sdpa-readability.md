# Overview

This readability document is specific for file
`@flash_attn_func_gfx1201_aiw.py` (referred as AIW file later).

# Function decomposition

The AIW file should only contain necessary code to make the kernel work. Any
optional or tuning knobs should be moved out of the file (to interface, for
example).

## Performance Tuning

Performance tuning relate constants and/or code should go to
`flash_attn_func_gfx1201_interface.py`. The AIW file should only care about
the correctness and functionality.

Here is an incomplete list

* `_PREFETCH_MIN_HEAD_DIM`
* `default_prefetch_dist`
* `default_block_m`
* `default_block_n`
* `resolve_shards`

# Coding Style

## Use dataclass for massive arguments on host code

Store all knobs in a dataclass and pass them to
`build_flash_attn_func_aiw_module_primary` as single object.

## Avoid using env vars since they are implicity from the API

All os.environ.get should be passed as tuning knobs

## `_FP_MODE` and `fm_fast`

Shouldn't they be computed at host and pass to the kernel instead?

## Move to a designated helper module? (Like `composed_tensors.py` in AOTriton)

An incomplete list

* `_global_load_tr_v8` (load/store)
  + Consider using Philox's approach for make it generic to `elem_dtype`?
* `_lds_load_v8` (L/S)
* `_lds_store_vx`
* `wmma_acc`
  + `if const_expr(dtype_str == "bf16")` is odd, can you make it generic by
    checking `a_v8` etc.'s properties directly?
* `_scmp_i32`, etc.
  + Common helper functions,
* `coop_load/store_*`
  + They are used to load/store tile with/without transpose. Check existing
    kernel for better practice.
* `Q preload`
  + Will be shared with backward kernel eventually.
* `gSWA: three regions over one contiguous block range`
  + Will be shared with backward kernel eventually.
* "Cross-shard S reduction"
  + Will be shared with backward kernel eventually.

In general you should consider if any component is re-usable with backward
kernel and move them to helper module(s) for sharing.

## `_decode_side`

Too long, decompose into smaller functions. Use @dataclass if flydsl supports.

## `tile_start`

Confusing. We already have `BLOCK_M/N` defining the QK tile to operate with.
Is `tile_start` operating along N axis or M axis?
Use `start_M/N` or `start_q/k` instead to name this variable, indicating their
operating direction.

# Possible bugs/Improvements

## `seqlen_q_v`

There is `seqlen_q_v = fx.Index(_seqlen_q_i32)` and then
`arith.cmpi(arith.CmpIPredicate.slt, _raw(q_start), _raw(seqlen_q_v))` etc.

We can safely assume `seqlen_q/k` always fits `i32` and we don't need to case it
to `u64`. (I didn't see seqlen_q_v being used elsewhere)

The same applies to `seqlen_k_v`. I see `tile_start` is used to compare
`seqlen_k_v` but do we really need the `tile_start` to be u64?

## `if const_expr(CAUSAL and _REVERSE_Q_TILES):`

`_REVERSE_Q_TILES` should be orthogonal to `CAUSAL`.

# Maybe outdated practice

## `_pointer_to_llvm_ptr`

Are they still required in more recent FlyDSL?

## `Vec.make_type`

Are they still required? Shouldn't they have built-in type in FlyDSL?

## LDS pointer arithmetics

I see `_lds_byte_base = _raw(fx.ptrtoint(lds_kv))` and then
`arith.addi(_lds_byte_base, _raw(off))`. We still don't have pointer
arithmetics on LDS?

## `_is_tl_l = _scmp_i32` etc.

WTF, we still can't write `x = a if flag else b` in FlyDSL?

# Questions

# `shard_qk_off = shard_id * fx.Index(QK_SLICE)`

I understand `shard_qk_off` should be a variable, but why we need
`fx.Index(QK_SLICE)` while `shard_id` should already be a variable? FlyDSL
requires explicit casting?

# Documentation

## Magic hardware values

* `K_SUB_N` = 32?
* `WMMA_LANE_K` = 8?
