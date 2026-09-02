# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Parity subclasses of the gfx950 dualwave helpers.

`kernels/attention/flash_attn_utils.py` is imported by four production kernels
(`flash_attn_generic`, `flash_attn_gfx950`, `flash_attn_fp8_gfx950`, and the
split-K combine), so it is **imported, never edited**. Everything this port
needs that differs from it lives here as a subclass.

--- What actually differs -------------------------------------------------

Less than the feature list suggests, because the production addressing is a
*special case* of the parity one rather than a different scheme. The dualwave
kernel addresses a flattened BSHD tensor:

    element(token, head, d) = token * stride_q_n + head * HEAD_DIM + d

which is the general BHSD-strided form with two slots pinned:

    element(b, h, s, d) = b * stride_0 + h * stride_1 + s * stride_2 + d
                                              ^^^^^^^^^^^^^^^^^^^^
                              stride_1 == HEAD_DIM, stride_2 == stride_q_n

So generalizing is a **change of variables, not new machinery**. Both `b` and
`h` are workgroup-uniform, so they fold into the buffer descriptor's base
address, which the production code already rebases per batch. What remains
per-access is `s * stride_2 + d`, which is the shape the existing helpers
already compute -- they just have to be handed `stride_2` where they currently
read `stride_q_n`, and a zero head offset where they currently add
`head * HEAD_DIM`.

That is why the overrides below are small and why none of them re-implements a
loop body. Three consequences worth naming, since each removes a whole class of
change:

- **`num_records` still bounds the sequence axis.** Rebasing at
  `(b, h)` and bounding at `seqlen * stride_2` elements makes an out-of-range
  row an out-of-buffer access, which returns zero in hardware rather than
  faulting. The production kernel relies on this for the ragged tail and it
  keeps working unmodified.
- **K and V get independent strides**, which the production code cannot express
  (it has one `stride_kv_n` for both). `load_k` and `load_v` are already
  separate methods that read `self.stride_kv_n_v`, so the subclass swaps the
  attribute and delegates rather than copying either body.
- **The head remap survives.** `q_head_idx = h_kv * gqa_group + group_id` is a
  permutation of head order for locality, not a correctness device. It is kept
  with runtime head counts so a parity build schedules its workgroups exactly
  as the measured baseline does.
"""

import contextlib
from dataclasses import replace

import fmha_common_gfx1201 as fmha
from fmha_common_gfx1201 import MaskedAxis
from gfx950_standalone import buffer_ops, dualwave
from philox import Philox

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

__all__ = [
    "ParityGemmHelper",
    "ParityKernelContext",
    "ParityKvGmemToLdsLoader",
    "ParityKvLdsToVgprLoader",
    "ParityQLoader",
    "ParitySoftmaxHelper",
    "ParityStoreHelper",
    "wire_ptr",
    "wire_view",
]


# --- the tensor operands are pointers on the wire --------------------------
#
# Every tensor operand of the three gfx950 parity kernels is declared
# `fx.Pointer`, not `fx.Tensor`. This is an ABI requirement, not a style
# choice: an `fx.Tensor` kernarg costs **two** slots -- the pointer, plus a
# packed shape+stride descriptor -- and AOTriton dispatches the compiled hsaco
# by filling the kernarg block itself, from a C++ struct that has a pointer and
# the strides it was already passing separately. The descriptor is 40 bytes of
# kernarg per operand that the caller has no way to fill and the kernel never
# reads.
#
# **It never reads it because every extent is already on the wire.** The
# kernels bound their buffer descriptors with products of `max_seqlen_q/k`,
# `hdim_qk/vo`, the head counts and the fifteen strides -- `seqlen_q * stride_q_seq`
# and so on -- and `batch_size` never appears as an extent at all, only as
# `batch_idx * stride_batch` with `batch_idx` coming from the grid. So the
# shapes in the descriptor duplicate scalars the ABI carries anyway, which is
# what makes dropping them safe rather than merely cheap.
#
# Measured, one build each at bf16 / h=2 / d=64 / s=256, `.kernarg_segment_size`
# from the final ISA, with VGPR counts alongside to show the wrapping is free:
#
#     forward   512 -> 296   (vgpr 164 both)
#     dQ        664 -> 352   (vgpr 190 both)
#     dK/dV     664 -> 352   (vgpr 236 both)
#
# **`Workspace` and `BlockTable` are deliberately still tensors.** Split-K and
# paged are not parity gaps -- the target is `scaled_dot_product_attention`, and
# neither is part of it -- so in every build that ships they are `Constexpr` and
# occupy no kernarg at all. There is nothing to shrink.
#
# The one thing that would have to change first if paged were ever built: the
# block table's consumer is `fx.rocdl.make_buffer_tensor(block_table)` in
# `flash_attn_utils.init_descriptors`, and that genuinely wants a tensor rather
# than an address -- unlike every operand converted here, whose only use of the
# tensor was `fx.get_iter`. A `wire_view` would not serve it, because
# `make_buffer_tensor` reads the layout it is handed. That is a real piece of
# work, not a rename.


def wire_ptr(t, elem_type=None):
    """A tensor's base address as an `fx.Pointer` argument.

    `elem_type` defaults to the tensor's **own** element type, and that default
    is load-bearing rather than cosmetic. `PointerJitArg` derives the pointer's
    assumed alignment from its element type -- `(width + 7) // 8` -- and so does
    the `fx.Tensor` path it replaces, which is what makes the two agree: a bf16
    operand keeps alignment 2 either way. Passing a byte pointer instead, as
    `fmha_abi_gfx1201.ptr_arg` does, would silently drop it to 1. Nothing fails
    when that happens; the loads just lose an alignment fact the backend was
    using, which is the kind of regression that gets attributed to the wrong
    change six weeks later.
    """
    if t is None:
        return flyc.from_c_void_p(elem_type or fx.Uint8, 0)
    if elem_type is None:
        # Keyed on the dtype's name so this module needs no torch import; it is
        # imported by the kernel builders, which run before any tensor exists.
        elem_type = _TORCH_DTYPE_TO_FX[str(t.dtype)]
    return flyc.from_c_void_p(elem_type, t.data_ptr())


_TORCH_DTYPE_TO_FX = {
    "torch.bfloat16": fx.BFloat16,
    "torch.float16": fx.Float16,
    "torch.float32": fx.Float32,
    "torch.uint8": fx.Uint8,
    "torch.int32": fx.Int32,
    "torch.int64": fx.Int64,
}


def wire_view(ptr):
    """A nominal view over `ptr`, for production code that spells `fx.get_iter`.

    `flash_attn_utils.py` is imported and never edited, and two of its methods
    reach a tensor operand through `fx.get_iter(...)`: `init_descriptors`, for
    the four dense Q/K/V/O views, and `_store_lse_row`, for LSE's base address.
    `fx.get_iter` rejects a pointer outright -- *"GetIterOp: expected
    TensorLikeType"* -- so a pointer operand cannot be handed to either
    directly. This wraps it back into something they accept.

    **The layout is a placeholder and the round trip is what makes that safe.**
    `fx.get_iter(fx.make_view(p, L))` returns `p` for any `L`, so every one of
    those call sites gets the address it wanted and none of them is affected by
    what `L` says -- all four of `init_descriptors`' views are rebased and
    re-bounded by `ParityKernelContext.init_descriptors` immediately afterwards,
    and `_store_lse_row` uses only `ptrtoint`. The value is verified to survive
    the round trip bit-exactly, and the ISA is unchanged by the wrapping.

    **What is therefore not safe is reading a size back off one of these.**
    `.shape` on a real `fx.Tensor` operand used to return the caller's true
    BHSD extents; here it returns the placeholder, which is a wrong answer
    rather than an error. No kernel does -- see the note above on why none needs
    to -- and `test_no_shape_reads_off_wire_views` is what keeps it that way.
    """
    return fx.make_view(ptr, fx.make_layout(fx.Int32(1), fx.Int32(1)))


# --- granule-general addressing --------------------------------------------
#
# Three production helpers fold constants that are only correct at granule 64.
# Each is replaced here by the same expression with the constant named, so at
# granule 64 they are the *same* arithmetic and a default build is unchanged --
# the bit-identity gate is what holds that claim.
#
# `_k_read_base` and `_ks_offset` are validated offline by
# `tooling/lds_model.py`, which reproduces family A exactly and confirms the
# granule-32 K read covers the tile once. The V pair below has no such model;
# it is measured instead.


def _anchor_v_o(traits, v_o):
    """`dualwave._anchor_v_o`, with the one-accumulator case spelled out.

    The production anchor asks for `!llvm.struct<(vector<16xf32>) x D_CHUNKS>`
    from an inline asm with `D_CHUNKS` outputs. At `D_CHUNKS == 1` LLVM rejects
    that outright -- *"inline asm with one output cannot return struct"* -- and
    the compiler aborts rather than diagnosing, so it surfaces as a crash.

    head_dim 32 is the first width to reach it: `D_CHUNKS = 32 / PV_MFMA_N = 1`.
    A single output returns the value's own type, so this is the same anchor
    with the struct wrapper dropped, not a weaker one.
    """
    if const_expr(traits.D_CHUNKS != 1):
        return dualwave._anchor_v_o(traits, v_o)
    acc = as_mlir_value(v_o[0])
    return [dualwave.llvm.inline_asm(acc.type, [acc], "", "=v,0", has_side_effects=True)]


def _slab_span_elems(rows, stride_seq, hdim=None):
    """Elements in a `(batch, head)` slab, ending on its last **real** element.

    `stride_seq` is the distance between rows and `hdim` is the width of one;
    the two coincide only when the tensor is contiguous in `(.., seqlen, hdim)`
    order. A BSHD caller hands us `(batch, seqlen, head, hdim)`, where the row
    stride is `num_heads * hdim` and the rest belongs to the *other heads* of
    that row -- so the obvious `rows * stride_seq` overshoots the slab by
    `stride_seq - hdim`.

    For an interior head that overshoot reads a neighbouring head's rows, which
    a padded head's `discard` throws away anyway. For the **last** (batch,
    head) slab it runs off the end of the tensor. AOTriton's concrete case: a
    `(3, 5, 64, 8)` Q at strides `(2560, 8, 40, 1)` puts the slab at
    `(batch 2, head 4)` at element 5152 bounded to 5152 + 64*40 = 7712, while
    the tensor ends at 7680 -- and `BLOCK_DMODEL` is 32 for `hdim_qk` 8, so the
    last row's load reaches column 31, 24 elements past the allocation, with
    the hardware range check permitting every one because they sit inside a
    `num_records` that should never have covered them. Whether that faults is
    up to the page mapping, which is why it surfaced as intermittent
    `pytest-xdist` worker deaths rather than a reproducible failure.

    **Under BHSD this changes nothing, by construction**: there
    `stride_seq == hdim`, so `(rows - 1) * stride_seq + hdim` *is*
    `rows * stride_seq` -- the same descriptor, down to the
    `oob_off == num_records` equality the D-tail store suppression relies on.
    Elsewhere the bound only shrinks, so an `oob_off` computed from the loose
    span stays at or past it and keeps being dropped.

    **`hdim` is rounded up to the load's 8-element chunk, and that is a
    departure from AOTriton's version.** Theirs ends on the last *real*
    element and warns against rounding, because for them the chunk containing
    `hdim` is off the end of the tensor for a last (batch, head) slab. This
    tree has a stricter input contract that makes the round-up not merely safe
    but required: `_check_8x_d_contract` enforces an 8-aligned D pitch
    precisely because "loads and stores are 8 columns wide, so the kernel
    touches `ceil8(hdim)` columns of every row" -- the chunk is inside the
    row's own slack by construction, and the caller was asked for it.

    Ending on the last real element instead **clips columns the kernel
    legitimately reads**: gfx950 range-checks a multi-dword buffer op per dword
    and drops any dword not wholly inside `num_records`, so at `hdim = 225`
    (bf16, two elements per dword) the dword holding columns 224 and 225 falls
    outside a bound ending at 225 and takes the *real* column 224 with it.
    Imported unrounded first, and that is exactly what happened: 36 failures in
    `test_padded_head_ignores_pad_contents`, all at `hdim % 8 == 1`.

    The round-up does not weaken the fix. On AOTriton's own example -- a
    `(3, 5, 64, 8)` Q at strides `(2560, 8, 40, 1)` -- `ceil8(8)` is 8, so the
    bound is `63*40 + 8 = 2528`, which is exactly the 2528 elements the tensor
    has left after that slab's base. The loose `64*40 = 2560` overruns it by
    32.

    The subtraction wraps at `rows == 0` -- a varlen empty sequence -- and an
    index that wraps becomes a `num_records` covering all of memory, which is a
    bigger version of this bug rather than a smaller one. The `rows > 0` select
    pins that case back to the untightened 0.

    `hdim=None` keeps the untightened bound, for callers not yet passing one.
    """
    rows_v = fx.Index(rows)
    if hdim is None:
        return rows_v * fx.Index(stride_seq)
    # `ceil8`, computed at runtime because `hdim` is a kernarg.
    width = ((fx.Index(hdim) + fx.Index(7)) // fx.Index(8)) * fx.Index(8)
    # A stride narrower than the chunk would make this negative; clamp so the
    # bound can only shrink, never grow past the loose span.
    over = fx.Index(stride_seq) - width
    trim = fx.Index((rows_v > fx.Index(0)).select((over > fx.Index(0)).select(over, fx.Index(0)), fx.Index(0)))
    return rows_v * fx.Index(stride_seq) - trim


def _bias_slab_num_records_bytes(seqlen_q, seqlen_kv, stride_b_seq_q, elem_bytes):
    """Byte bound for a `(q_row, kv_col)` bias slab, in `num_records` terms.

    `_slab_span_elems`'s problem on the bias, with the same answer.
    `stride_b_seq_q` is the distance between bias rows, not the width of one; a
    BSHD caller hands us `(batch, seqlen_q, head, seqlen_k)`, where the row
    stride is `num_heads * seqlen_k`. So `seqlen_q * stride_b_seq_q` overshoots
    by `stride_b_seq_q - seqlen_k`, and for the last (batch, head) slab it runs
    off the end -- AOTriton saw `flyc_attn_fwd` fault 128 bytes past a
    `(3, 5, 32, 16)` bias at strides `(2560, 16, 80, 1)`.

    `minui` against the untightened span pins the `seqlen_q == 0` wrap back to
    0 and is a no-op everywhere else, since `seqlen_k <= stride_b_seq_q`.

    **Do not widen this back.** Ending on the last valid element costs the
    readers something -- gfx950 range-checks a multi-dword buffer op per dword
    and drops any dword not wholly inside `num_records`, so at odd `seqlen_k` a
    wide read of the last row loses column `seqlen_k-1` to the out-of-range
    column sharing its dword. That is a load-side problem with a load-side fix;
    buying those columns back here means reading memory the caller never handed
    us.
    """
    loose = fx.Index(seqlen_q) * fx.Index(stride_b_seq_q) * fx.Index(elem_bytes)
    tight = (fx.Index(seqlen_q) * fx.Index(stride_b_seq_q) - fx.Index(stride_b_seq_q) + fx.Index(seqlen_kv)) * fx.Index(
        elem_bytes
    )
    return arith.minui(as_mlir_value(tight), as_mlir_value(loose))


def exp2_wait_state(values):
    """Hold a consumer one slot away from the `v_exp_f32` that produced its input.

    `v_exp_f32` is a quarter-rate transcendental -- it retires 16 lanes a cycle
    -- so a VALU consumer issued in the very next slot reads a partially
    written destination. CDNA requires a wait state there and
    `GCNHazardRecognizer` does not insert one for gfx950.

    **A bare `_s_nop` is not this.** `s_nop` is emitted as inline asm with no
    operands, so it orders itself against memory and nothing else; a pure VALU
    has no dependence on it and LLVM moves one across freely. AOTriton's first
    patch for this was exactly that and it fixed the one build it was written
    against by perturbing register allocation -- the `_s_nop` stayed within
    three instructions of an `exp2` in one of eight builds scanned, and
    `BLOCK_DMODEL=192, PADDED_HEAD=False` still carried two zero-gap sites with
    it in. The two `dualwave._s_nop(1)` calls already in `ParityGemmHelper.qk`
    for the head_dim 96 defect are documented as working the same accidental
    way, and this is very likely that defect seen from the other end.

    The barrier has to be one the values flow *through*. Tying every `exp2`
    result to a matching asm output (`"=v,=v,...,0,1,..."`) makes the allocator
    give each pair one register and emits one `s_nop 0`; a consumer then cannot
    precede it, because it reads the asm's result rather than the `exp2`'s.

    **`has_side_effects` must stay off.** With it on the asm becomes a
    scheduling-region boundary, and at rung 224 -- already on the 512-VGPR
    cliff -- AOTriton measured `vgpr_spill_count` going 100 -> 440 and 38-48%
    of the backward at head_dim 216. Without it the allocation is byte-for-byte
    the pre-patch one, because what holds the consumer down is the SSA def-use
    edge and not the flag.

    **Unreproduced in this tree**, and imported on the argument rather than on
    a failure: a gap scan over six dK/dV configurations (both families, both
    dtypes, with and without bias/dropout/causal), counting vector-pipe
    instructions between each `v_exp_f32` and its first consumer, found the
    minimum gap at exactly 1 everywhere and never 0 -- which is the "luck, not
    enforcement" AOTriton's own scan reports. The scan located 32-64 `exp2`
    sites per kernel, so the zero is a measurement and not a vacuous pass.

    **The forward does not get this, and that is a measurement too.** AOTriton
    reports a zero-gap site in seven of 108 bf16 forward kernels at
    `BLOCK_DMODEL` 96, 224 and 256 -- "exactly the set of rungs whose head dims
    were failing" -- which is the strongest correspondence in their write-up.
    Re-checked here at precisely those rungs *and* at the padded head dims
    inside them that their note names (72/80/88/96 and 216, plus 240/248/256),
    across dropout on and off and causal on and off: **30 configurations,
    82-128 dependent `exp2` pairs each, minimum gap exactly 1, zero sites below
    it.** So the forward sits where dK/dV sat before this barrier -- at the
    minimum, not under it -- and the copy of `_exp2_score_slice` it would take
    (its results are consumed inside, so no wrapper can interpose on a return
    value) would be a verbatim body rotting against `flash_attn_utils.py` for a
    hazard with no local evidence.

    That negative is weaker than it looks and should not be read as immunity:
    30 samples against an effect AOTriton saw in 7 of 108 leaves a real chance
    of having missed it, and exposure is per-schedule -- phase 2 measured a
    minimum gap of 8 in dQ against the exactly-1 here, from the same source.
    If a forward failure ever points this way, the mixin is the fix and the
    scan above is the thing to re-run first.

    Delete this once the hazard is modelled in `GCNHazardRecognizer` or the
    wait state is supplied where `exp2` is emitted (issue 10).
    """
    irs = [as_mlir_value(v) for v in values]
    n = len(irs)
    if const_expr(n == 1):
        # A single-output asm returns the value's own type; LLVM rejects a
        # struct return from one, which is the same edge `_anchor_v_o` hits.
        return [dualwave.llvm.inline_asm(irs[0].type, irs, "s_nop 0", "=v,0", has_side_effects=False)]
    ret_ty = dualwave.ir.Type.parse(f"!llvm.struct<({', '.join(['f32'] * n)})>")
    constraints = ",".join(["=v"] * n + [str(i) for i in range(n)])
    ret = dualwave.llvm.inline_asm(ret_ty, irs, "s_nop 0", constraints, has_side_effects=False)
    return [dualwave.llvm.extractvalue(irs[i].type, ret, [i]) for i in range(n)]


def mfma_operand_wait_state(pack):
    """Two wait states between a packed MFMA operand and the MFMA reading it.

    One step further down the chain than `exp2_wait_state`: that one keeps
    `v_exp_f32` away from the `v_cvt_pk_bf16_f32` reading it, this one keeps
    that convert away from the `v_mfma` reading *its* result as SrcA or SrcB.
    Same barrier shape, same reason a bare `_s_nop` will not do.

    **AOTriton's evidence**, at `BLOCK_DMODEL=128`, bf16, dropout, matrix bias,
    `mfma_rows=16`:

        v_cvt_pk_bf16_f32 v70, v0, v1
        s_waitcnt lgkmcnt(1)
        v_mfma_f32_16x16x32_bf16 v[54:57], v[78:81], v[70:73], v[54:57]

    `v[54:57]` came back NaN and ~1e23 against a correct 2e-3, in dK head-dim
    columns 16..31 and nowhere else, while dV out of the same kernel was always
    correct. **Non-deterministic**: two runs from one seed with bit-identical
    inputs disagreed on which rows were wrong, so it is a race and not a
    miscompiled value. Forcing all 131 `s_waitcnt` in the shipped hsaco to
    `vmcnt(0) lgkmcnt(0)` changed nothing, which rules out memory visibility --
    `s_waitcnt` retires on the scalar pipe, so an already-satisfied one between
    the two supplies no wait state at all. Swapping the convert with the
    `ds_read_b64_tr_b16` above it, a semantic no-op, makes the kernel correct.

    **Reproduced here.** The same gap scan finds `v_cvt_pk_*_f32` writes
    reaching an MFMA SrcA/SrcB with a gap of 1 in four of six configurations --
    2 sites at `d128 r16 bf16` with bias and dropout, 7 at `d192 r32`, 9 at
    `d224 r32 causal`, 1 at `d128 r16 f16` -- and the producer writes the
    *last* register of the SrcB tuple in each, which is the shape AOTriton
    names.

    **Scope is not established and the obvious generalisation is wrong.** A
    blanket "VALU write then MFMA read needs two wait states" cannot be the
    rule: ~7500 such sites across 426 mostly-passing kernels. Narrowing to this
    exact instruction pair at zero effective wait states still leaves 196.
    What separates the site that corrupts from the rest is not known.

    Delete this once the hazard is modelled or the wait state is supplied where
    `_bf16_trunc_pack_v8` builds the operand (issue 11).
    """
    ir_val = as_mlir_value(pack)
    return dualwave.llvm.inline_asm(ir_val.type, [ir_val], "s_nop 1", "=v,0", has_side_effects=False)


def _k_read_base(traits, lane_mod_32, lane_div_32):
    """`_k_lds_read_base_per_lane` with `SMEM_N_RPT` in place of a literal 8."""
    return (
        (lane_mod_32 % traits.SMEM_N_RPT) * traits.SMEM_K_LINE_STRIDE
        + (lane_mod_32 // traits.SMEM_N_RPT) * traits.D_128B_SIZE
        + lane_div_32 * traits.VEC_KV
    )


def _ks_offset(traits, ks):
    """`_swizzled_ks_offset` with `K_STEPS_PER_BAND` in place of a literal 4."""
    per_band = traits.K_STEPS_PER_BAND
    return (ks // per_band) * traits.K_LDS_TO_REG_KSTEP_OUTER_STRIDE + (
        ks % per_band
    ) * traits.K_LDS_TO_REG_KSTEP_INNER_STRIDE


def _v_dc_offset(traits, dc):
    """`_swizzled_v_dc_off` with `D_CHUNKS_PER_BAND` in place of a literal 2."""
    per_band = traits.D_CHUNKS_PER_BAND
    return (dc // per_band) * traits.V_LDS_TO_REG_DCHUNK_PAIR_STRIDE + (
        dc % per_band
    ) * traits.V_LDS_TO_REG_DCHUNK_IN_PAIR_STRIDE


def _v_imm_lo(traits, dc, k_substep):
    """`_swizzled_v_imm_lo`, in bytes, over the general dc offset."""
    return (k_substep * traits.V_LDS_TO_REG_K_SUBSTEP_STRIDE + _v_dc_offset(traits, dc)) * traits.BF16_BYTES


def _score_column_runs(kv_vectorized):
    """`[(element_index, column_offset, width)]` covering one score vector.

    The 16 f32 an MFMA lane holds are 16 *columns* of a single row -- the row
    is `lane_mod_32`, and `lane_div_32` shifts the column set -- so a bias read
    is a row-wise gather in principle and a handful of contiguous spans in
    practice. Which spans is exactly what the causal mask's threshold table
    already says, since that table maps element to column; this reads the same
    fact for a different purpose rather than restating it.

    Grouping the thresholds into consecutive runs gives four spans of four at
    the default granule and two of eight when the KV tile is vectorized. Both
    are single `buffer_load`s, which is the whole reason bias needs no LDS
    staging: the accumulator's layout hands back contiguity for free.
    """
    thresholds = []
    for thr_x, thr_y in dualwave._causal_pair_thresholds(kv_vectorized):
        thresholds.extend((thr_x, thr_y))
    runs, start = [], 0
    for i in range(1, len(thresholds) + 1):
        if i == len(thresholds) or thresholds[i] != thresholds[i - 1] + 1:
            runs.append((start, thresholds[start], i - start))
            start = i
    return runs


class ParityGemmHelper(dualwave.DualwaveGemmHelper):
    """The two GEMMs, addressed one D stage at a time.

    Under `D_STAGES > 1` a KV tile's D axis is covered in several passes, so
    neither GEMM sees all of it at once. The two are asymmetric about what that
    means, because D is a *reduction* axis for QK and an *output* axis for PV:

    - `qk_stage` accumulates into a running S that the caller carries across
      the stages, rather than seeding a fresh zero. All stages contribute to
      every element of S, and softmax cannot run until the last one has.
    - `pv_step_k` writes a disjoint slice of the O accumulator per stage, so
      the stages never meet; the stage only shifts which `v_o[dc]` is hit.

    Both take stage-relative register lists (that is what the loaders return)
    and map them to global indices here, so a stage index never has to be
    threaded into the loaders' addressing.
    """

    def qk_stage(self, v_k, q_all_scaled_bf16, acc, stage=0):
        k_lo, k_hi = v_k
        v_s_lo, v_s_hi = acc
        steps = self.traits.K_STEPS_PER_STAGE
        for ks in range_constexpr(steps):
            q_pack = dualwave._get_q_pack(self.traits, q_all_scaled_bf16, stage * steps + ks)
            v_s_lo = dualwave._mfma_acc(k_lo[ks], q_pack, v_s_lo, self.mma_atom, self.mfma_acc_vec_type)
            v_s_hi = dualwave._mfma_acc(k_hi[ks], q_pack, v_s_hi, self.mma_atom, self.mfma_acc_vec_type)
        return (v_s_lo, v_s_hi)

    def qk(self, v_k, q_all_scaled_bf16, stage=0):
        """Unstaged entry point: seed at zero and run the one stage there is."""
        if const_expr(self.traits.D_STAGES == 1):
            out = super().qk(v_k, q_all_scaled_bf16)
        else:
            out = self.qk_stage(v_k, q_all_scaled_bf16, (self.c_zero_v16f32, self.c_zero_v16f32), stage)
        # Works, and **not for the reason it looks like.** Without it head_dim
        # 96 computes a wrong answer; with it, 96 is correct across five shapes
        # in both masking modes at ~0 cost.
        #
        # It is *not* supplying a wait state. `s_nop` here is a side-effecting
        # intrinsic present during scheduling, and stripping nops from the two
        # builds gives the same 2817 instructions with **88 differing register
        # assignments** -- so this perturbs allocation, the same way anchoring
        # the K packs does, and not the same way `amdgpu-snop-padding=1` does
        # (that one is post-RA and leaves registers byte-identical).
        #
        # The underlying defect is therefore still latent. Every *documented*
        # gfx950 MFMA hazard is satisfied in the failing build: for
        # `v_mfma_f32_32x32x16_bf16`, which is XDL 8-pass on gfx950, the
        # requirements are 12 (write -> MFMA SrcA/B), 10 (-> SrcC, non-matching
        # opcode), 12 (-> VALU/DS/VMEM) and 2 (VALU -> MFMA), and a scan finds
        # zero violations. The leading hypothesis is `ds_read_b64_tr_b16`,
        # which `GCNHazardRecognizer` does not model at all.
        #
        # Located by bisection, not from the pair: a wait state here, after
        # `exp2`, or after `reduce_sum` each fix it, while after `cast_p`,
        # `lazy_rescale_o`, `load_k`, `load_v`, `reduce_max`, `sub_m` or
        # `pv_step_k` do not -- so the producer is at or before the QK MFMAs
        # and the consumer is before the P cast. Scans of every documented
        # hazard class (VALU->MFMA SrcA/B, MFMA->VALU, MFMA->MFMA by operand
        # position, VALU->DPP) found no difference between a broken and a
        # working build, so the exact instruction pair is still unidentified.
        # See `sdpa_lore_gfx950.md`.
        #
        # Costs nothing: +1.6% at head_dim 64 and +0.8% at 128, both inside
        # run-to-run noise, and it buys head_dim 96 at 932 TF against 579 for
        # the padded 128 tile.
        dualwave._s_nop(1)
        return out

    def pv_step_k(self, step, v_p, v_v, v_o, stage=0):
        if const_expr(self.traits.D_STAGES == 1):
            return super().pv_step_k(step, v_p, v_v, v_o)
        v_p_lo, v_p_hi = v_p
        v_pk = v_v[step]
        p_pk = v_p_lo[step] if const_expr(step < 2) else v_p_hi[step - 2]
        per_stage = self.traits.D_CHUNKS_PER_STAGE
        for dc in range_constexpr(per_stage):
            out = stage * per_stage + dc
            v_o[out] = dualwave._mfma_acc(v_pk[dc], p_pk, v_o[out], self.mma_atom, self.mfma_acc_vec_type)
        return v_o

    def pv(self, v_p, v_v, v_o, stage=0):
        for step in range_constexpr(4):
            v_o = self.pv_step_k(step, v_p, v_v, v_o, stage=stage)
        return v_o


class _ParityKvStaging:
    """KV DMA addressing that allows more than one issue per wave.

    A mixin because **two objects need it**: the context builds the m0 tables
    in `init_dma_m0_tables`, and the loader computes source addresses in
    `_async_load_kv_linear`. The loader subclasses the *production* context
    rather than `ParityKernelContext`, so inheritance alone would not share
    them -- and a second copy is exactly how the write and read sides of an
    LDS layout drift apart.

    **What the production formula assumes.** It places one KV tile line per
    wave per d-band, `line = wave + d * SMEM_N_RPT`, which is correct exactly
    when `SMEM_N_RPT == NUM_WAVES`. Family A satisfies that by arithmetic
    coincidence: 8 waves, and BLOCK_N 64 at 8 tokens per issue is 8 lines. A
    4-wave family covering the same BLOCK_N needs **two issues per wave**, and
    under the production formula lines 4..7 are never written at all -- the
    reads then return whatever LDS happened to hold, which is how head_dim 192
    produced non-deterministic NaN.

    **The generalisation is a change of index, not of scheme.** The flat DMA
    index is band-major, `d_flat = band * ISSUES + issue`, and

        line  = (wave + issue * NUM_WAVES) + band * SMEM_N_RPT
        token = n_in_warp * SMEM_N_RPT + (wave + issue * NUM_WAVES)

    Both collapse to the production form at `ISSUES == 1`, where `SMEM_N_RPT`
    and `NUM_WAVES` are equal. Verified against a model of the write/read
    mapping before being written: the model reproduces family A exactly, and
    is what ruled out the BLOCK_N 128 variants -- a wave's K read covers 64
    tokens, so BLOCK_N 128 would need a doubled score accumulator.
    """

    def _issue_split(self, d_flat):
        """`(band, issue)` for a flat DMA index. Band-major."""
        return d_flat // self.ISSUES_PER_WAVE, d_flat % self.ISSUES_PER_WAVE

    def _dma_line(self, d_flat):
        """The KV tile line this wave writes for `d_flat`."""
        band, issue = self._issue_split(d_flat)
        return self.wave_id_uni + issue * self.traits.NUM_WAVES + band * self.traits.SMEM_N_RPT

    def _dma_m0(self, buf_base_elems, line_stride, d_flat):
        addr = self.lds_kv_base_idx + (buf_base_elems + self._dma_line(d_flat) * line_stride) * self.traits.BF16_BYTES
        return rocdl.readfirstlane(T.i32, as_mlir_value(fx.Int32(addr)))

    def k_dma_base(self, buf_id, d):
        return self._dma_m0(dualwave._k_buf_base(self.traits, buf_id), self.traits.SMEM_K_LINE_STRIDE, d)

    def v_dma_base(self, buf_id, d):
        return self._dma_m0(dualwave._v_buf_base(self.traits, buf_id), self.traits.SMEM_V_LINE_STRIDE, d)

    # Which D stage the next DMA reads from global. Set by the loader
    # immediately before delegating, the same way `stride_kv_n_v` is, and safe
    # for the same reason: tracing is eager, so it is read while `super()`
    # runs and no branch is open across the swap.
    dma_stage = 0

    def kv_src_elem(self, src_base, d_flat):
        """Global element index for this lane's `d_flat` chunk of a KV tile.

        `band` only spans `SMEM_D_RPT` d-bands, and under `D_STAGES > 1` that
        is one *stage* of the head dim rather than all of it -- LDS holds a
        stage at a time. So the stage's base offset is the single term that
        makes staging reach global memory; everything else is unchanged, and
        at `D_STAGES == 1` the term is zero.
        """
        band, issue = self._issue_split(d_flat)
        line_n = self.wave_id + issue * self.traits.NUM_WAVES
        n_in_tile = self.n_in_warp * self.traits.SMEM_N_RPT + line_n
        global_d = self.d_bucket * self.traits.VEC_KV + band * self.traits.D_128B_SIZE
        if const_expr(self.traits.D_STAGES > 1):
            global_d = global_d + self.dma_stage * self.traits.STAGE_DIM
        return src_base + n_in_tile * self.stride_kv_n_v + global_d


class ParityKernelContext(_ParityKvStaging, dualwave.DualwaveKernelContext):
    """Dualwave context addressing arbitrary BHSD strides, with a runtime scale."""

    def __init__(
        self,
        traits,
        *,
        strides,
        sm_scale,
        num_head_q,
        num_head_k,
        hdim_qk,
        hdim_vo,
        padded_head=False,
        hdim_qk_floor=0,
        window_left=None,
        window_right=None,
        seqinfo=(None, None, None, None),
        varlen_bits=0,
        num_seqlens=0,
        Bias=None,
        bias_strides=(0, 0, 0),
        o_strides=(0, 0, 0),
        philox=(None, None, 0, None, None),
        idropout_p=0,
        dropout_scale=1.0,
        **kwargs,
    ):
        super().__init__(traits, **kwargs)
        # 9 strides in launch order: Q, K, V, each (batch, head, seq).
        # Numerically named per `sdpa-feature-gap.md`'s porting instruction --
        # the `z/h/m/k` suffixes it warns about have caused real bugs.
        #
        # **O's three are separate**, because two of the three kernels built on
        # this class have no O at all: dQ writes dQ and dK/dV writes dK and dV,
        # each under its own name with its own strides. Folding an output's
        # strides into the O slot is what made `stride_o_*` a name for
        # whichever tensor happened to be in it -- which computes the right
        # answer right up until someone reads `stride_o_seq` expecting O.
        #
        # A 12-tuple is still accepted and puts its last three in the O slot,
        # which is what the caller meant by passing them. That compatibility is
        # for the dQ kernel, which is another agent's file and still passes 12;
        # retire it once dQ moves to `o_strides`.
        (
            self.stride_q_batch,
            self.stride_q_head,
            self.stride_q_seq,
            self.stride_k_batch,
            self.stride_k_head,
            self.stride_k_seq,
            self.stride_v_batch,
            self.stride_v_head,
            self.stride_v_seq,
        ) = strides[:9]
        if len(strides) == 12:
            self.stride_o_batch, self.stride_o_head, self.stride_o_seq = strides[9:12]
        elif len(strides) == 9:
            self.stride_o_batch, self.stride_o_head, self.stride_o_seq = o_strides
        else:
            raise ValueError(f"strides must be 9 (Q, K, V) or 12 (with O); got {len(strides)}")
        self.sm_scale_arg = sm_scale
        self.num_head_q = num_head_q
        self.num_head_k = num_head_k
        # P1. `hdim_qk` is the real reduction extent, which may be narrower
        # than the compiled tile; `hdim_vo` is the real output width. They are
        # separate because the two GEMMs are not symmetric -- see
        # `ParityQLoader` and `ParityStoreHelper`.
        self.hdim_qk = hdim_qk
        self.hdim_vo = hdim_vo
        self.PADDED_HEAD = bool(padded_head)
        # Compile-time lower bound on `hdim_qk`, exclusive. The dispatcher
        # enforces it, so D columns below it need no mask.
        self.HDIM_QK_FLOOR = int(hdim_qk_floor)
        # P3. Raw window bounds, still carrying their sentinels; resolved in
        # `init_runtime_indices`, which is the first point the sequence lengths
        # this build will use are known.
        self.window_left_arg = window_left
        self.window_right_arg = window_right
        # P4. `VarlenBits` plus the four sequence-info arrays, named by role:
        # `?0` supplies lengths, `?1` supplies positions. Unread slots are
        # **null pointers**, which is safe only because the decoder branches
        # rather than selects -- see `fmha.cond_load`.
        self.seqinfo_q0, self.seqinfo_q1, self.seqinfo_k0, self.seqinfo_k1 = seqinfo
        self.varlen_bits_arg = varlen_bits
        self.num_seqlens_arg = num_seqlens
        # P5. The bias matrix and its (batch, head, seqlen_q) strides. Slot 3
        # is the KV axis and is contractually contiguous, so it is not passed.
        self.Bias = Bias
        # `_seq_q`, not `_seq`. Bias is the one rank-4 tensor here with **two**
        # sequence axes, `(batch, head, seqlen_q, seqlen_k)`, so a bare `_seq`
        # does not say which -- and the k axis is contiguous by contract and
        # therefore never passed, which makes the ambiguity easy to miss rather
        # than obviously wrong. Not `_seqq`: the doubled letter reads as a typo
        # and is skipped. Not `_seq0`: a numeric slot is exactly what the P7
        # stride rename removed, on the grounds that nothing at runtime
        # distinguishes one stride from another and spelling the axis out is
        # the only check there is.
        self.stride_b_batch, self.stride_b_head, self.stride_b_seq_q = bias_strides
        # P6. The counter is the (pointer, immediate) pair torch splits it
        # into, not one pre-summed scalar: a captured graph re-reads the
        # pointer half. The two `*_output` slots report back what was actually
        # used, so a backward pass can regenerate the identical mask.
        (
            self.philox_seed_ptr,
            self.philox_offset1,
            self.philox_offset2,
            self.philox_seed_output,
            self.philox_offset_output,
        ) = philox
        self.idropout_p = idropout_p
        self.dropout_scale_arg = dropout_scale

    # -- runtime softmax scale -------------------------------------------
    #
    # The production kernel derives the scale from head_dim
    # (`rsqrt(head_dim) * log2e`), which is only correct when the compiled
    # tile *is* the real extent. Under a padded head it is wrong, and AOTriton
    # passes `Sm_scale` regardless, so it becomes an argument.
    #
    # `* log2e` is kept folded in: every downstream exp is `exp2`, and folding
    # here means the conversion is paid once per kernel rather than per tile.
    # Pre-scaling Q by it *before* the row max -- which the production kernel
    # already does -- is the anti-FMA correction, so nothing moves.

    def init_types_and_constants(self, head_dim_runtime=None):
        super().init_types_and_constants(head_dim_runtime=head_dim_runtime)
        # DMA issues each wave makes per d-band, and the flat count that
        # follows. `NUM_DMA_*` feed both the m0 tables and the `s_waitcnt`
        # budget the pipeline is balanced against, so they have to agree.
        self.ISSUES_PER_WAVE = self.traits.SMEM_N_RPT // self.traits.NUM_WAVES
        self.NUM_DMA_K = self.traits.SMEM_D_RPT * self.ISSUES_PER_WAVE
        self.NUM_DMA_V = self.NUM_DMA_K
        self.c_sm_scale = fx.Float32(self.sm_scale_arg)
        self.c_sm_scale_log2e = fx.Float32(
            arith.mulf(
                fx.as_ir_value(fx.Float32(self.sm_scale_arg)),
                fx.as_ir_value(fx.Float32(dualwave._LOG2E)),
                fastmath=self.fm_fast,
            )
        )

    # -- runtime head counts ---------------------------------------------

    def init_thread_mapping(self):
        super().init_thread_mapping()
        # Re-derive the four head indices with runtime counts. The *mapping* is
        # the production one verbatim: `h_idx` is decomposed against the KV head
        # count and recomposed against the group size, which groups the Q heads
        # sharing a KV head. Only the operands change from constexpr to runtime,
        # so a build with matching counts schedules identically.
        num_head_k = fx.Index(self.num_head_k)
        gqa_group = fx.Index(self.num_head_q) // num_head_k
        # Published, because the dK/dV group loop's trip count is this same
        # quantity and `gqa_q_head_base` -- the head that loop walks from -- is
        # derived from it here. Recomputing it at the loop is what let a trait
        # and a kernarg drift apart in the first place (issue 9).
        self.gqa_group = gqa_group
        self.h_kv_idx = self.h_idx % num_head_k
        self.group_id = self.h_idx // num_head_k
        self.q_head_idx = self.h_kv_idx * gqa_group + self.group_id
        self.kv_head_idx = self.h_kv_idx
        if const_expr(self.traits.LPT_TILE_ORDER and self.traits.CAUSAL):
            # P7. Longest-processing-time-first: under a causal mask q-block
            # `i` walks about `i` KV tiles and `grid.y` issues in increasing
            # order, so the cheapest workgroups go first and the most expensive
            # land in the tail. Reversing the index puts them first.
            #
            # A bijection over the same index set, so the output is
            # bit-identical and only the schedule moves -- which is what makes
            # it safe as a knob. **Measured at 0.0% on gfx950** across every
            # width and shape tried, so it ships off; with 8 XCDs and this many
            # workgroups the tail imbalance it targets is already absorbed.
            # gfx1201's forward kernel measures 12-16% from the same change.
            #
            # Folded into this override rather than added as a second one: an
            # `init_thread_mapping` of its own silently replaced the head-index
            # derivation above, which only shows up once the runtime head
            # counts differ from the build's.
            self.q_block_idx = fx.Index(gpu.grid_dim.y) - fx.Index(1) - self.q_block_idx

    # -- granule-general staging ------------------------------------------

    def init_dma_thread_offsets(self):
        """Split a lane into (token, d-bucket) for the granule it stages.

        Production splits `lane // VEC_KV` by `lane % VEC_KV`, which is right
        only when a granule spans exactly `VEC_KV` lanes -- true at 64, which
        is `VEC_KV * VEC_KV`, and nowhere else. A lane always moves `VEC_KV`
        contiguous D elements, so `granule // VEC_KV` lanes cover one token's
        granule and the rest of the wave advances the token.
        """
        traits = self.traits
        self.lane_in_warp = self.tid % traits.WARP_SIZE
        self.n_in_warp = self.lane_in_warp // traits.SMEM_D_BUCKETS
        self.d_bucket = self.lane_in_warp % traits.SMEM_D_BUCKETS

    def init_lds_read_bases(self):
        super().init_lds_read_bases()
        # `_k_lds_read_base_per_lane` folds `SMEM_N_RPT` as a literal 8.
        self.k_lds_read_base_per_lane = _k_read_base(self.traits, self.lane_mod_32, self.lane_div_32)

    # -- per-tensor strides ----------------------------------------------

    def init_sequence_lengths(self, **kwargs):
        """Decode `VarlenBits` into the six scalars the rest of the kernel uses.

        The base class knows exactly one varlen shape -- cumulative
        `cu_seqlens` on both sides -- and reads it directly. `VarlenBits`
        generalizes that to three orthogonal axes per side (STACKED, LENGTH,
        POSITION), which is five useful configurations rather than one, so the
        decode replaces that branch rather than extending it.

        `fmha.decode_addressing` is gfx1201's, reused unedited: the bits mean
        the same thing on both architectures, and a second copy of a wire
        format is a second thing to keep in step.

        Three things about the shape of this:

        - **`z` is not `batch_idx`.** The workgroup's `z` selects a *sequence*;
          the decode says which *batch slice* that sequence lives in, which is
          `z` for a batched layout and 0 for a packed one. Overwriting
          `batch_idx` here is what keeps `_slab_byte_base` correct without a
          varlen branch inside it.
        - **`z` outlives that overwrite, as `seq_idx_i32`.** Anything indexed
          by *sequence* rather than by batch slice needs it, and after this
          method runs there is nowhere else to get it. `init_philox` is the
          caller that found this out the hard way: it was using `batch_idx`,
          which is 0 for every sequence of a packed layout, so all N sequences
          drew one dropout mask. Published before the branch so the dense path
          has it too -- there `z` and `batch_idx` are the same number, which is
          why the dense and padded layouts were never wrong.
        - **The reads are scalar.** `z` is workgroup-uniform, so these land in
          SGPRs and cost nothing against the VGPR budget.
        - **Row offsets stay separate from the batch index.** A packed tensor
          has `batch = 0` and a large `row_off`; a padded one has a real batch
          and `row_off = 0`. Both go through the same
          `batch * s_batch + row_off * s_seq`, which is why the descriptors
          need no varlen case at all.
        """
        traits = self.traits
        # Captured before anything can overwrite `batch_idx`, and before the
        # dense/varlen branch so both paths publish it. At entry `batch_idx` is
        # `init_block_mapping`'s `batch_q_tile_id // num_q_tiles` -- the grid's
        # sequence index, with the head and q-tile axes already divided out.
        z = fx.Int32(self.batch_idx)
        self.seq_idx_i32 = z
        if const_expr(not traits.VARLEN):
            super().init_sequence_lengths(**kwargs)
            self.lse_tokens_i32 = fx.Int32(self.seq_len_v)
            self.kv_batch_idx = self.batch_idx
            return
        q_len, q_row, q_batch = fmha.decode_addressing(
            self.varlen_bits_arg, 0, self.seq_len_v, self.seqinfo_q0, self.seqinfo_q1, z
        )
        k_len, k_row, k_batch = fmha.decode_addressing(
            self.varlen_bits_arg, 8, self.seq_len_kv_v, self.seqinfo_k0, self.seqinfo_k1, z
        )
        self.lse_tokens_i32 = fmha.lse_token_pitch(
            self.varlen_bits_arg, 0, self.seq_len_v, self.seqinfo_q0, self.seqinfo_q1, self.num_seqlens_arg
        )
        # **Each side owns its own batch index**, and this is not pedantry:
        # `0x040B` -- packed Q against `seqused_k` on a BHSD cache -- has Q
        # stacked (batch 0, large row offset) and K batched (batch z, no row
        # offset) in the *same call*. Reusing Q's index for K reads batch 0 of
        # the cache for every sequence, which is plan section 1.4 arriving as a
        # wrong answer. Measured before the split: correct for the four
        # configurations where both sides agree, wrong for the one that does
        # not.
        self.batch_idx = fx.Index(q_batch)
        self.kv_batch_idx = fx.Index(k_batch)
        self.varlen_q_row_off = fx.Index(q_row)
        self.varlen_kv_row_off = fx.Index(k_row)
        self.seqlen_q_v = fx.Index(q_len)
        self.seqlen_kv_v = fx.Index(k_len)
        self.seqlen_kv_i32 = fx.Int32(k_len)
        # `q_tok_base` / `q_tok_end` are the base class's names for the same
        # interval; the paged and split-K helpers read them.
        self.q_tok_base = fx.Index(0)
        self.q_tok_end = self.seqlen_q_v
        self.kv_tok_base = fx.Index(0)
        self.kv_tok_end = self.seqlen_kv_v

    def init_philox(self):
        """Seed, counter and this workgroup's plane origin. Prologue-only.

        **The offset scheme is `Philox.grid_plane`/`grid_offset`, and using
        them is the reproducibility contract** rather than a convenience. A
        dropout mask is generated here and *regenerated* by the backward pass
        and by the debug mask kernel; all three must agree bit for bit or the
        gradients are quietly wrong. They agree because they call the same two
        functions, not because three transcriptions of one formula happened to
        match.

        The consequence to write down, since it is invisible in any test run at
        a single tile size: **the mask is a function of element coordinates
        only.** `grid_plane` is given `max_seqlen_q`/`max_seqlen_k`, never
        `BLOCK_M`/`BLOCK_N`, so re-tuning the tile geometry cannot move a single
        random. From this phase onward that is a constraint on the tuner, not
        just a property of today's code.
        """
        if const_expr(not self.traits.ENABLE_DROPOUT):
            return
        self.philox_rng = Philox.for_arch("gfx950")
        # **`seq_idx_i32`, not `batch_idx`.** The plane is per *sequence*, and
        # the two are the same number only when the layout has a real batch
        # axis. `decode_addressing` resolves `batch_idx` to 0 for every
        # sequence of a packed layout -- correct for addressing, since a packed
        # tensor carries its origin in `row_off` instead -- so a packed call
        # drew plane `0*H + h` for all N and every sequence in the pack shared
        # one dropout mask.
        #
        # Invisible to every gate we had: `allclose` cannot see it, and the
        # bitwise varlen oracle could not either, because its reference was N
        # separate `batch_size=1` calls and those also collapse to plane 0. The
        # two sides agreed by being degenerate in the same way.
        plane = self.seq_idx_i32 * fx.Int32(self.num_head_q) + fx.Int32(self.q_head_idx)
        seed = fmha.philox_seed_value(self.philox_seed_ptr)
        offset = fmha.philox_offset_base(self.philox_offset1, self.philox_offset2)
        fmha.philox_report(self.philox_seed_output, self.philox_offset_output, seed, offset)
        self.philox_seed = seed
        self.philox_plane_base, self.philox_row_stride = self.philox_rng.grid_plane(
            offset, plane, self.seq_len_v, self.seq_len_kv_v
        )

    def init_tile_bounds(self, **kwargs):
        """Resolve the window, then let the causal bound derive from it.

        Here and not in `init_runtime_indices` for an ordering reason: a
        sentinel resolves against the sequence lengths, and those are not
        settled until `init_sequence_lengths`, which runs in between. This is
        also the first place `delta_i32` is *read* -- `causal_end_raw_i32` and
        `max_num_tiles` both come from it -- so overriding it immediately
        before `super()` is what makes the tile count follow the window's right
        bound with no further change.
        """
        if const_expr(self.traits.WINDOW):
            # **`delta_i32` *is* the right bound.** The base class sets it to
            # `seqlen_kv - seqlen_q`, which is bottom-right causal spelled as a
            # diagonal offset, and every causal site downstream -- the mask's
            # `rel`, `causal_end_raw_i32`, `max_num_tiles` -- reads it from
            # here. Re-pointing it at the resolved `window_right` therefore
            # generalizes all of them at once, and is why a window is a flag on
            # top of causal rather than a second masking mode.
            #
            # Resolution is on the device, not the host: a sentinel resolves
            # against *this sequence's* lengths, and under varlen those differ
            # per sequence. `fmha.resolve_window` is gfx1201's, reused unedited.
            left, right = fmha.resolve_window(
                self.window_left_arg,
                self.window_right_arg,
                fx.Int32(self.seqlen_q_v),
                self.seqlen_kv_i32,
            )
            self.window_left_i32 = fx.Int32(left)
            self.delta_i32 = fx.Int32(right)
        super().init_tile_bounds(**kwargs)
        if const_expr(self.traits.WINDOW and not self.traits.SPLITK):
            self._skip_dead_leading_tiles()

    def _skip_dead_leading_tiles(self):
        """Start the KV walk at the window's left edge instead of at tile 0.

        The right bound already truncates the walk, through `max_num_tiles`.
        The left bound is the mirror image and needs the *base* to move, and
        the machinery for that is already here: `[split_t0, split_t_end)` is
        the tile range the whole pipeline is written against -- the prologue
        loads `split_tile(0..2)`, the loop runs from `split_tile(3)`, and only
        split-K ever moved the base before. So this is a new *value* for an
        existing knob, not new control flow.

        A dead tile is not a wrong answer, it is a masked one: every column
        fails the left bound, `exp2` gives zero, and the tile contributes
        nothing. That is exactly why correctness cannot show this works --
        skipping is invisible to the output and only a measurement sees it.

        The lowest column any row of this Q block can reach is
        `q_start - window_left`, since `q_start` is the smallest row. Tiles
        entirely below that are dead for the whole block.
        """
        traits = self.traits
        # Clamp before dividing rather than after: `q_start - window_left` is
        # negative whenever the window reaches past the start of the sequence,
        # and `fx.Index` is unsigned, so a negative would come out enormous and
        # skip the entire range. i32 until the value is known non-negative --
        # the same rule `resolve_window` states.
        first_col_i32 = fx.Int32(self.q_start) - self.window_left_i32
        first_col_i32 = fx.Int32((first_col_i32 > fx.Int32(0)).select(first_col_i32, fx.Int32(0)))
        t0 = fx.Index(first_col_i32) // fx.Index(traits.BLOCK_N)
        # Even, because the software pipeline consumes two tiles per iteration
        # and `split_t_end` is already even; an odd base would make the segment
        # odd and leave the epilogue a tile short.
        t0 = (t0 // fx.Index(2)) * fx.Index(2)
        # The pipeline's prologue plus epilogue need four tiles to exist. The
        # base class guarantees `max_num_tiles >= 4`, so this cannot underflow.
        cap = self.split_t_end - fx.Index(4)
        self.split_t0 = fx.Index((t0 < cap).select(t0, cap))

    def init_runtime_indices(self, **kwargs):
        super().init_runtime_indices(**kwargs)
        self.stride_q_seq_v = fx.Index(self.stride_q_seq)
        self.stride_k_seq_v = fx.Index(self.stride_k_seq)
        self.stride_v_seq_v = fx.Index(self.stride_v_seq)
        self.stride_o_seq_v = fx.Index(self.stride_o_seq)
        # The production helpers read these two names. Q's seq stride is the
        # default; `ParityKvGmemToLdsLoader` swaps the KV one per tensor.
        self.stride_q_n_v = self.stride_q_seq_v
        self.stride_kv_n_v = self.stride_k_seq_v

    def _slab_byte_base(self, s0, s1, s2, row_off, head_idx, batch_idx=None):
        """Byte offset of this workgroup's (batch, head) slab.

        Both axes are workgroup-uniform, so folding them into the descriptor
        costs scalar arithmetic once instead of a per-access add.

        **`row_off` is the varlen token origin, not the batch's.** The
        production kernel's `q_tok_base` is `batch * seqlen` in dense mode,
        because there the batch axis *is* a token offset into one flat
        allocation. Here the batch has its own stride, so passing `q_tok_base`
        would count it twice. Dense passes 0; varlen will pass the cumulative
        offset with `stride_0` set to 0, which is the same decomposition
        `fmha.decode_addressing` produces on gfx1201.
        """
        if batch_idx is None:
            batch_idx = self.batch_idx
        elems = batch_idx * fx.Index(s0) + head_idx * fx.Index(s1) + row_off * fx.Index(s2)
        return elems * fx.Index(self.traits.BF16_BYTES)

    def _slab_view(self, tensor, s0, s1, s2, row_off, head_idx, rows, batch_idx=None, hdim=None):
        """A buffer view over one (batch, head) slab, bounded at its last element.

        A row past the sequence is out of the descriptor and reads as zero
        rather than faulting -- the same mechanism the production kernel uses
        for its ragged tail, restated over a stride the caller chose.

        `hdim` is the row's real extent and `s2` the distance between rows; see
        `_slab_span_elems` for why the second is not a bound for the first.
        **Optional, and defaulting to the untightened bound**, so that call
        sites in kernels this change does not cover keep their present
        behaviour rather than silently acquiring a different descriptor.
        """
        span_elems = _slab_span_elems(rows, s2, hdim)
        return dualwave._make_rebased_view(
            fx.get_iter(tensor),
            self._slab_byte_base(s0, s1, s2, row_off, head_idx, batch_idx=batch_idx),
            span_elems * fx.Index(self.traits.BF16_BYTES),
            fx.make_layout(fx.Int32(span_elems), fx.Int32(1)),
            _buf_flags_i32=self.buf_flags_i32,
            _elem_ir=self.elem_ir,
        )

    def init_descriptors(self, **kwargs):
        """Rebuild Q/K/V/O over arbitrary strides; everything else stays.

        `super()` runs first for the state that does not depend on the
        addressing scheme -- `delta_i32`, `buf_flags_i32`, `elem_ir`, the
        paged page-view constants and the debug resource -- and its four dense
        views are then replaced. The discarded ones are pure descriptor
        arithmetic with no side effects, so they fold away; the alternative is
        duplicating the half of the method that has nothing to do with strides.
        """
        traits = self.traits
        super().init_descriptors(**kwargs)

        # Varlen token origins. Dense is 0 on both sides: the batch axis has a
        # real stride here, so it must not also be spent as a token offset.
        # Under varlen these come from the decode, and the pairing is what
        # makes one expression serve every mode -- a packed tensor gets
        # `batch = 0` with a large `row_off`, a padded one a real batch with
        # `row_off = 0`.
        if const_expr(traits.VARLEN):
            self.q_row_off = self.varlen_q_row_off
            self.kv_row_off = self.varlen_kv_row_off
        else:
            self.q_row_off = fx.Index(0)
            self.kv_row_off = fx.Index(0)

        # Head folded into the base, so what remains per access is `s * stride`.
        self.q_gmem_elem_offset = self.q_start * self.stride_q_seq_v
        self.kv_gmem_elem_offset = fx.Index(0)

        # First element past O's descriptor. A store redirected here is dropped
        # by the hardware bound, which is how `ParityStoreHelper` suppresses
        # the D-tail chunks without branching.
        self.o_oob_off = self.seqlen_q_v * self.stride_o_seq_v

        if const_expr(traits.BIAS_TYPE):
            # Same slab shape as Q, which is the point: bias is indexed by
            # (batch, head, q_row, kv_col), so the varlen row origin and the
            # head fold into the base exactly as Q's do, and what remains per
            # access is `q_row * stride_b_seq_q + col`.
            #
            # A raw resource rather than a `_slab_view`: the reads here are
            # per-lane vectors of 4 or 8 elements at an address the lane
            # computes, which is `buffer_ops.buffer_load`'s shape, not the
            # copy-atom one the K/V DMA uses.
            #
            # The bound is the slab, so a row past `seqlen_q` reads zero
            # instead of faulting -- which is also the right bias for a padded
            # row -- and so does a column that runs off the last row's end.
            self.bias_rsrc = buffer_ops.create_buffer_resource(
                self.Bias,
                max_size=False,
                num_records_bytes=as_mlir_value(
                    _bias_slab_num_records_bytes(
                        self.seqlen_q_v, self.seqlen_kv_v, self.stride_b_seq_q, traits.BF16_BYTES
                    )
                ),
                base_byte_offset=as_mlir_value(
                    self._slab_byte_base(
                        self.stride_b_batch,
                        self.stride_b_head,
                        self.stride_b_seq_q,
                        self.q_row_off,
                        self.q_head_idx,
                    )
                ),
            )
        self.q_div = self._slab_view(
            self.Q,
            self.stride_q_batch,
            self.stride_q_head,
            self.stride_q_seq,
            self.q_row_off,
            self.q_head_idx,
            self.seqlen_q_v,
            hdim=self.hdim_qk,
        )
        self.o_div = self._slab_view(
            self.O,
            self.stride_o_batch,
            self.stride_o_head,
            self.stride_o_seq,
            self.q_row_off,
            self.q_head_idx,
            self.seqlen_q_v,
            # **Deliberately untightened.** This slot does not have a width the
            # base class knows: the forward puts O in it, which is `hdim_vo`
            # wide, and the dQ kernel puts dQ in it, which is Q-shaped and
            # `hdim_qk` wide -- `BwdDqStoreHelper` rebinds `self.hdim_vo` to
            # `ctx.hdim_qk` on the helper for exactly that reason. Bounding it
            # by `hdim_vo` here clipped dQ's real columns in
            # `[hdim_vo, hdim_qk)` and failed six `test_asymmetric_hdim` cases
            # the moment the tightening went in.
            #
            # This is the same slot-reuse the dK/dV stride split removes, and
            # the fix is the same: the kernel that owns the output builds its
            # own view. dK/dV now does (`dk_div` / `dv_div`, both tightened);
            # the forward and dQ still share this one, so it stays loose until
            # they do too.
        )
        if const_expr(not traits.PAGED):
            self.k_div = self._slab_view(
                self.K,
                self.stride_k_batch,
                self.stride_k_head,
                self.stride_k_seq,
                self.kv_row_off,
                self.kv_head_idx,
                self.seqlen_kv_v,
                batch_idx=self.kv_batch_idx,
                hdim=self.hdim_qk,
            )
            self.v_div = self._slab_view(
                self.V,
                self.stride_v_batch,
                self.stride_v_head,
                self.stride_v_seq,
                self.kv_row_off,
                self.kv_head_idx,
                self.seqlen_kv_v,
                batch_idx=self.kv_batch_idx,
                hdim=self.hdim_vo,
            )


class ParityQLoader(dualwave.DualwaveQLoader):
    """Q staging that zeroes the columns past `hdim_qk`.

    **Masking Q is what makes a padded head correct, and masking it is
    enough** for any finite pad -- `QK^T = sum_d Q[d] * K[d]`, so a zero in Q
    annihilates whatever K holds at the same column. K is left unmasked
    deliberately: Q is loaded once in the prologue, K once per KV tile, so this
    is the one of the two that is free. See `sdpa-close-gap-gfx950.md` for the
    residual case (a pad holding NaN or Inf, where `0 * NaN` is NaN) and the
    test that pins it.

    Whole 8-element chunks are still *loaded* past the extent, and are allowed
    to be: the D-axis pitch is contractually a multiple of 8 elements, so the
    chunk containing `hdim_qk` lands inside the allocation. What must not
    happen is those elements reaching the MFMA, which is what `discard` stops.
    """

    def load_pack(self, q_row_in_block, ks):
        pack = super().load_pack(q_row_in_block, ks)
        if const_expr(not self.PADDED_HEAD):
            return pack
        col_base = fx.Index(ks * self.traits.K_STEP_QK) + self.lane_div_32 * fx.Index(self.traits.MFMA_LANE_K)
        # Q is masked once, before the KV loop, so the mask registers die
        # immediately and the bitmask form is pure win here.
        return MaskedAxis(fx.Index(self.hdim_qk), elem_dtype=self.elem_dtype, bitmask=True).discard(
            pack, col_base, self.traits.MFMA_LANE_K
        )

    def load_all(self):
        """Q rows for this wave, for any `K_STEPS_QK`.

        The production version assembles the packs through a fixed 8 -> 16 ->
        32 tree and then takes either one 32-pack or a concatenation of two,
        which covers `K_STEPS_QK` 4 and 8 -- head_dim 64 and 128 -- and nothing
        else. head_dim 192 wants 12 packs and 256 wants 16, so the tail of that
        tree simply drops the remainder and the `Vec` constructor rejects the
        result ("shape (96,) has 96 elements, but value has type
        vector<64xbf16>"). It fails loudly, which is the good case.

        A left fold over the packs replaces the tree. `_concat_vectors` builds
        an explicit shuffle index list, so it does not need equal widths and
        the uneven steps 192 needs (32+32 -> 64, 64+32 -> 96) are fine.
        """
        traits = self.traits
        ctx = self.ctx_ref
        ctx.init_q_row()
        acc = self.load_pack(ctx.q_row_in_block, 0)
        for ks in range_constexpr(traits.K_STEPS_QK - 1):
            acc = dualwave._concat_vectors(acc, self.load_pack(ctx.q_row_in_block, ks + 1))
        return Vec(acc, (traits.K_STEPS_QK * traits.MFMA_LANE_K,), self.elem_dtype)


class ParityKvLdsToVgprLoader(dualwave.DualwaveKvLdsToVgprLoader):
    """K register reads that zero the columns past `hdim_qk`.

    Masking Q alone is enough for a *finite* pad, since `0 * x == 0`. It is not
    enough for a pad holding NaN or Inf, and a caller's D-axis padding is
    allocation slack whose contents nothing constrains -- so K is masked too.
    V needs nothing: its D columns are O's columns, and those are suppressed at
    the store instead.

    **Why the mask is the same expression as Q's.** The K tile transits LDS in
    a swizzled layout, so the column a register holds is not obviously its
    linear D index. Working the two halves against each other: the DMA writes
    LDS element `base + w*LINE + d*N_RPT*LINE + l*8 + i` holding D column
    `(l % 8) * 8 + d * 64 + i`, while `load_k` reads at
    `(lm % 8) * LINE + (lm // 8) * 64 + ld * 8 + (ks // 4) * N_RPT * LINE +
    (ks % 4) * 16`. Matching term by term gives `w = lm % 8`, `d = ks // 4` and
    `l = (lm // 8) * 8 + ld + (ks % 4) * 2`, and since `(lm // 8) * 8` vanishes
    mod 8,

        D = (ld + (ks % 4) * 2) * 8 + (ks // 4) * 64 + i
          = ks * 16 + ld * 8 + i

    which is `_q_pack_col` exactly -- the swizzle permutes *tokens* across LDS
    lines and leaves D in linear order. `k_hi` differs from `k_lo` by
    `K_LDS_TO_REG_N_STRIP_STRIDE`, an N offset, so it carries the same columns.
    """

    @contextlib.contextmanager
    def _scoped_to_stage(self):
        """Narrow `K_STEPS_QK` / `D_CHUNKS` to one stage for the duration.

        Under `D_STAGES > 1` an LDS buffer holds one stage of the head dim, so
        the inherited readers -- which loop to `K_STEPS_QK` and `D_CHUNKS` --
        would run off the end of it. The *offsets* need no adjustment:
        `_swizzled_ks_offset` and `_swizzled_v_dc_off` address d-bands within
        the buffer, and a stage-sized buffer has exactly `SMEM_D_RPT` of them.
        Only the counts are wrong.

        So this swaps the two counts rather than reimplementing the read
        loops. Copying them is the specific thing to avoid here: the write side
        (`kv_src_elem`) and the read side have to describe one LDS layout, and
        a second copy of either is how they drift apart.

        A no-op at `D_STAGES == 1`, where the two values are already equal --
        deliberately not merely equivalent, so a default build cannot differ.
        """
        if const_expr(self.traits.D_STAGES == 1):
            yield
            return
        full = self.traits
        self.traits = replace(
            full,
            K_STEPS_QK=full.K_STEPS_PER_STAGE,
            D_CHUNKS=full.D_CHUNKS_PER_STAGE,
        )
        try:
            yield
        finally:
            self.traits = full

    def _read_k_packs(self, buf_id, urk_base):
        """The inherited non-vectorized K read, with a granule-general swizzle.

        Six lines rather than a `super()` call because the production loop
        calls `_swizzled_ks_offset`, which folds `K_STEPS_PER_BAND` as a
        literal 4 -- a module function, so there is nothing to override but the
        loop that calls it. Identical arithmetic at granule 64.
        """
        traits = self.traits
        k_base = dualwave._k_buf_base(traits, buf_id)
        k_lo = [None] * traits.K_STEPS_QK
        k_hi = [None] * traits.K_STEPS_QK
        for ks in range_constexpr(traits.K_STEPS_QK):
            k_lo[ks], k_hi[ks] = self._load_k_pair(buf_id, k_base + urk_base + _ks_offset(traits, ks))
        return k_lo, k_hi

    def load_k(self, buf_id, urk_base=None, stage=0):
        with self._scoped_to_stage():
            if const_expr(self.traits.KV_VECTORIZED):
                k_lo, k_hi = super().load_k(buf_id, urk_base=urk_base)
            else:
                base = self.k_lds_read_base_per_lane if urk_base is None else urk_base
                k_lo, k_hi = self._read_k_packs(buf_id, base)
            steps = self.traits.K_STEPS_QK
        if const_expr(not self.PADDED_HEAD):
            return (k_lo, k_hi)
        # K is masked *inside* the KV loop, once per KV tile per K-step, so
        # unlike Q's prologue mask this one is on the hot path and its masks
        # stay live across everything else. Masking every step is what made a
        # padded head cost 27-54% against its own rung -- and near-independent
        # of how much pad there was, since 240-in-256 paid the same tax as
        # 129-in-192. Removing it entirely restored every padded build to its
        # rung's native rate exactly, which is what identifies this loop, and
        # nothing else about padding, as the whole cost.
        #
        # **Most of the steps cannot contain pad.** The build serves
        # `(HDIM_QK_FLOOR, BLOCK_DMODEL]` and the dispatcher enforces it, so a
        # step whose columns all lie at or below the floor is reading real
        # data. On the 32-spaced rungs consecutive rungs are 32 apart and
        # `K_STEP_QK` is 16, so exactly two steps survive at every width --
        # 2 of 16 at the 256 rung rather than 16.
        #
        # Skipping is `continue`, not a narrower loop bound: under `D_STAGES`
        # the surviving steps are the last ones globally but sit at arbitrary
        # stage-relative indices, so there is no contiguous stage-local range
        # to iterate.
        width = self.traits.MFMA_LANE_K
        masked = [ks for ks in range(steps) if ((stage * steps + ks) + 1) * self.traits.K_STEP_QK > self.HDIM_QK_FLOOR]
        if const_expr(not masked):
            return (k_lo, k_hi)
        # The bitmask form ANDs one precomputed dword instead of selecting per
        # element, but each mask is a live register. Gated on the number of
        # steps that actually survive rather than on `K_STEPS_QK`: measured
        # +21% in the 64-wide tile and -43% in the 128-wide one, where 32 extra
        # live registers turned a spill-free build into 61 spills. With two
        # steps left the wide tiles are back under that limit.
        cols = MaskedAxis(
            fx.Index(self.hdim_qk),
            elem_dtype=self.elem_dtype,
            bitmask=len(masked) * (width // 2) <= 16,
        )
        for ks in masked:
            # The mask is against the *global* D column, so the stage's base
            # has to come back in here -- `ks` is stage-relative above.
            col_base = fx.Index((stage * steps + ks) * self.traits.K_STEP_QK) + self.lane_div_32 * fx.Index(width)
            k_lo[ks] = cols.discard(k_lo[ks], col_base, width)
            k_hi[ks] = cols.discard(k_hi[ks], col_base, width)
        return (k_lo, k_hi)

    def read_v_packs(self, buf_id, urv_base):
        """The inherited non-vectorized V read, with a granule-general swizzle.

        Same reason as `_read_k_packs`: the production loop calls
        `_swizzled_v_imm_lo`, which reaches `_swizzled_v_dc_off` and its
        literal 2. Identical arithmetic at granule 64.
        """
        traits = self.traits
        lds_base = dualwave._v_buf_base(traits, buf_id) + urv_base
        v_scope = dualwave._dualwave_lds_scope("v", buf_id)
        packs = [[None] * traits.D_CHUNKS for _ in range(4)]
        for dc in range_constexpr(traits.D_CHUNKS):
            for k_substep in range_constexpr(4):
                imm_lo = _v_imm_lo(traits, dc, k_substep)
                pair = traits.V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE * traits.BF16_BYTES
                read = lambda off: dualwave._ds_read_tr_v4f16_imm(  # noqa: E731
                    lds_base,
                    off,
                    lds_kv_base_idx=self.lds_kv_base_idx,
                    v_lds_read_vec4_type=self.v_lds_read_vec4_type,
                    scope_name=v_scope,
                    scope_names=traits.LDS_SCOPE_NAMES,
                )
                a, b = read(imm_lo), read(imm_lo + pair)
                packs[k_substep][dc] = Vec(a).shuffle(Vec(b), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
        return packs

    def load_v(self, buf_id, urv_base=None, stage=0):
        with self._scoped_to_stage():
            if const_expr(self.traits.KV_VECTORIZED):
                return super().load_v(buf_id, urv_base=urv_base)
            base = self.v_lds_read_base_per_lane if urv_base is None else urv_base
            return self.read_v_packs(buf_id, base)


class ParityKvGmemToLdsLoader(_ParityKvStaging, dualwave.DualwaveKvGmemToLdsLoader):
    """K/V staging with independent per-tensor sequence strides.

    The production loader has one `stride_kv_n` for both tensors, which the
    BHSD ABI can contradict -- K and V are separate allocations and a caller
    may hand us one as BHSD and the other as BSHD. `load_k` and `load_v` are
    already separate methods that read `self.stride_kv_n_v` on the way down, so
    selecting the right stride is an attribute swap before delegating, not a
    second copy of the DMA body.

    Safe because tracing is eager: the attribute is read while `super()` runs,
    and no branch is open across the swap.
    """

    def load_k(self, tile_start, buf_id, page_id=None, stage=0):
        self.stride_kv_n_v = self.stride_k_seq_v
        self.dma_stage = stage
        return super().load_k(tile_start, buf_id, page_id=page_id)

    def load_v(self, tile_start, buf_id, page_id=None, stage=0):
        self.stride_kv_n_v = self.stride_v_seq_v
        self.dma_stage = stage
        return super().load_v(tile_start, buf_id, page_id=page_id)

    # `load_*_tile` resolve a tile index to a token offset and delegate; the
    # stage has to ride along or it is lost at that hop.
    def load_k_tile(self, tile_idx, buf_id, page_id=None, stage=0):
        self.load_k(self.tile_start(tile_idx), buf_id, page_id=page_id, stage=stage)

    def load_v_tile(self, tile_idx, buf_id, page_id=None, stage=0):
        self.load_v(self.tile_start(tile_idx), buf_id, page_id=page_id, stage=stage)

    def _async_load_kv_linear(self, dma_m0, buf_id, src_div, src_base, soffset, num_dma):
        """Issue this wave's KV DMAs, addressed through `kv_src_elem`.

        The production version calls `_linear_kv_src_elem`, which interleaves
        tokens across `NUM_WAVES` and offsets D by the flat index -- both true
        only at one issue per wave. This is the same loop with the address
        redirected, not a second copy of the DMA sequence.
        """
        for d in range_constexpr(num_dma):
            self._issue_kv_dma(src_div, dma_m0[buf_id][d], self.kv_src_elem(src_base, d), soffset)


class ParitySoftmaxHelper(dualwave.DualwaveSoftmaxHelper):
    """Softmax whose running max can never be `-inf`.

    The production `reduce_max` seeds the reduction with `-inf` and the kernel
    floors the result to `-3.0e38` only under `CAUSAL`. That is the pattern
    `sdpa-feature-gap.md` flags:

        m_i = tl.full([BLOCK_M], -3.40282e+38)   # do NOT use -inf

    A row whose scores are all `-inf` -- every key masked -- gives
    `m_i = -inf`, and then `exp2(-inf - -inf)` is `NaN` rather than 0. On
    gfx1201 this is *preventative* today and only becomes reachable with bias,
    since causal tile 0 always contains `kv = 0 <= q_row`. Here it is cheaper
    to be unconditional: seeding the reduction at the floor costs nothing (it
    replaces one constant with another) and removes the case for every masking
    mode at once, including the windows and bias still to come.

    Seeding rather than flooring afterwards is the part that matters. A floor
    applied to the *result* still lets an all-`-inf` tile reach the subtract;
    seeding means no lane ever holds `-inf` as a max in the first place.
    """

    def reduce_max(self, v_s):
        return dualwave._score_pair_max(v_s, self.c_neg_floor, self.fm_fast)

    # -- P5: bias ------------------------------------------------------------

    def _add_bias_inplace(self, v_s, tile_idx, narrow_tail=False):
        """`S += bias * log2(e)`, in place, for one KV tile.

        **After the scale and before the mask**, and both halves of that matter:

        - after, because `m_i` and the exponent live in the base-2 scaled
          domain -- Q is pre-scaled by `sm_scale * log2e` on this kernel -- so
          a bias in natural units has to cross into it. This is AOTriton's
          `qk += bias * 1.44269504089`.
        - before, because a column past `seqlen_k` must stay `-inf` rather than
          becoming `-inf + bias`. Those columns are not keys the caller hid;
          they do not exist, and neither do their bias entries.

        No runtime "does this tile need it" guard. There is nothing to skip: a
        bias build reads a bias for every live tile by definition.

        **`narrow_tail` reads the run one column at a time** instead of as one
        `dwordx2`/`dwordx4`. gfx950 range-checks a multi-dword buffer op *per
        dword* and drops any dword not wholly inside `num_records`, so a live
        column sharing a dword with an out-of-range one is dropped with it.
        `_bias_slab_num_records_bytes` ends the slab exactly on its last valid
        element, so at odd `seqlen_k` the final dword covers columns
        `(seqlen_k-1, seqlen_k)` and straddles the bound: **the last row's last
        bias entry reads back 0**, for every `(batch, head)`, with no fault and
        no NaN. Widening the bound instead would make the kernel read memory
        the caller never handed it, so the load narrows.

        Reproduced in the forward before the fix, with an all-zero bias holding
        one entry at `[.., sq-1, sk-1]` and asking whether `O`'s last row moves
        at all:

            seqlen_k    64    65    63   127   128
            last row  2e-2   0.0   0.0   0.0  1.4e-2

        -- read at even `seqlen_k`, silently dropped at odd. An aggregate error
        bound cannot see this: it is one element in `sq*sk`, which is why it
        survived both the forward's bias ladder and dQ's
        `test_db_matches_bias_gradient`.

        **Only the tiles that can contain column `seqlen_k-1` narrow.** A
        straddle needs `seqlen_k` odd, hence `seqlen_k % BLOCK_N != 0`, hence
        that tile is partial and takes the KV tail mask -- so
        `seq_pad_mask_if_needed` passes `narrow_tail` and `bias_to_lists` does
        not. An interior tile is wholly in bounds: its final dword covers
        `(c, c+1)` with `c+1 <= seqlen_k-1`, ends at or before the bound, and
        keeps the wide load.
        """
        traits = self.traits
        ctx = self.ctx_ref
        s_lo, s_hi = v_s
        lane_n_off = 8 if traits.KV_VECTORIZED else 4
        # The row is this lane's, and the descriptor already holds (batch,
        # head, row origin), so what is left is `row * pitch + column`.
        row_base = ctx.q_row * fx.Index(ctx.stride_b_seq_q)
        col_base = fx.Index(tile_idx * traits.BLOCK_N) + self.lane_div_32 * fx.Index(lane_n_off)
        log2e = fx.Float32(1.4426950408889634)
        # **Not `fm_fast` here.** `fm_fast` is MLIR's `fast`, which carries
        # `ninf` and `nnan` -- a licence to assume no infinities reach the
        # operation. A bias entry of `-inf` is how a caller spells "never
        # attend here", so bias is the first thing in this kernel that puts a
        # real infinity into *arithmetic* rather than into a select: the causal
        # mask writes `-inf` through a `cndmask` on raw bits and the KV tail
        # mask through `select`, neither of which fastmath touches.
        #
        # It happens to produce the right answer with `fast` today. That is not
        # a reason to keep it -- plan1 records `ninf` silently deleting a KV
        # tail mask on gfx1201, which is the same licence being taken up later
        # by a different pass. Two operations per element, one of them by a
        # constant, so the flag costs nothing measurable to get right.
        fm_bias = arith.FastMathFlags.contract | arith.FastMathFlags.reassoc
        for half, values in ((0, s_lo), (1, s_hi)):
            for elem0, col_off, width in _score_column_runs(traits.KV_VECTORIZED):
                run_base = row_base + col_base + fx.Index(col_off + half * 32)
                if const_expr(narrow_tail):
                    for j in range_constexpr(width):
                        one = buffer_ops.buffer_load(
                            ctx.bias_rsrc,
                            as_mlir_value(fx.Int32(run_base + fx.Index(j))),
                            vec_width=1,
                            dtype=ctx.elem_dtype,
                        )
                        # `vec_width=1` returns a **scalar** of the element
                        # type, not a one-lane vector, so it is wrapped
                        # directly rather than through `Vec(...)[0]`.
                        b = fx.Float32(ctx.elem_dtype(one).to(fx.Float32))
                        values[elem0 + j] = dualwave._fadd(
                            values[elem0 + j], dualwave._fmul(b, log2e, fm_bias), fm_bias
                        )
                    continue
                span = buffer_ops.buffer_load(
                    ctx.bias_rsrc,
                    as_mlir_value(fx.Int32(run_base)),
                    vec_width=width,
                    dtype=ctx.elem_dtype,
                )
                for j in range_constexpr(width):
                    b = fx.Float32(Vec(span, (width,), ctx.elem_dtype)[j].to(fx.Float32))
                    values[elem0 + j] = dualwave._fadd(values[elem0 + j], dualwave._fmul(b, log2e, fm_bias), fm_bias)

    def bias_to_lists(self, v_s, tile_idx):
        """`v_s_vec_to_lists`, with the bias folded in on the way through.

        The interior tiles of the dual-wave loop only unpack the scores -- they
        need no KV tail mask, being wholly in bounds -- so this is where their
        bias goes. The tiles that *do* mask get it from the
        `seq_pad_mask_if_needed` override, which keeps every tile covered
        exactly once.
        """
        lists = self.v_s_vec_to_lists(v_s)
        if const_expr(self.traits.BIAS_TYPE):
            self._add_bias_inplace(lists, tile_idx)
        return lists

    def seq_pad_mask_if_needed(self, v_s, tile_idx):
        """The KV tail mask, with the bias applied first.

        Order is the point, and it is the same one gfx1201 records: a column
        past `seqlen_k` must come out `-inf`, not `-inf + bias`. Those columns
        are not keys the caller hid -- they do not exist, and neither do their
        bias entries.

        Overriding here rather than adding a call at each site is what makes
        the *wide* body work unchanged: it masks every tile it visits, so this
        single override is its whole bias path.
        """
        if const_expr(self.traits.BIAS_TYPE):
            lists = self.v_s_vec_to_lists(v_s)
            # `narrow_tail`: these are the tiles that can hold column
            # `seqlen_k - 1`, which is the one a wide load loses at odd
            # `seqlen_k`. Interior tiles come through `bias_to_lists` and keep
            # the wide load.
            self._add_bias_inplace(lists, tile_idx, narrow_tail=True)
            v_s = dualwave._score_lists_to_vecs(lists)
        return super().seq_pad_mask_if_needed(v_s, tile_idx)

    def rescale_o(self, v_o, m_row, l_row, m_tile_max, v_p):
        """The non-lazy rescale, anchored so one accumulator does not abort LLVM.

        `_anchor_v_o` here is the parity one, which spells out the
        `D_CHUNKS == 1` case; `rescale_o` in the shared helper reaches the
        production version, which always asks an inline asm with `D_CHUNKS`
        outputs for a struct. At one output LLVM does not diagnose that -- it
        aborts with *"inline asm with one output cannot return struct"* and an
        `UNREACHABLE`, killing the process rather than raising.

        Found by the P7 sweep, which is exactly what a sweep is for: head_dim
        32 is the only width with `D_CHUNKS == 1`, `lazy_rescale=True` is the
        default and takes the other path, so this needed the one combination
        nothing had built. The body is the shared one with that single
        substitution; there is no other difference.
        """
        if const_expr(self.traits.D_CHUNKS != 1):
            return super().rescale_o(v_o, m_row, l_row, m_tile_max, v_p)
        m_new = dualwave._fmax(m_row, m_tile_max, self.fm_fast)
        corr = rocdl.exp2(T.f32, as_mlir_value(dualwave._fsub(m_row, m_new, self.fm_fast)))
        self.scale_o(v_o, corr)
        v_o = _anchor_v_o(self.traits, v_o)
        v_p = dualwave._scale_v_p(self.traits, v_p, corr, elem_dtype=self.elem_dtype, fm_fast=self.fm_fast)
        l_row = dualwave._fmul(l_row, corr, self.fm_fast)
        return v_o, m_new, l_row, v_p

    def safe_l_inv(self, l_row):
        """`1/l`, with the dropout survivor scale folded in.

        `1/(1-p)` is a per-row constant, so it belongs here and not on the
        3200-odd scores it would otherwise multiply: one extra multiply per
        output row against one per element. `dropout_args` computes the value
        host side, and both kernel bodies reach their normalisation through
        this method, so this is the whole fold.
        """
        inv = super().safe_l_inv(l_row)
        if const_expr(self.traits.ENABLE_DROPOUT):
            inv = dualwave._fmul(inv, fx.Float32(self.ctx_ref.dropout_scale_arg), self.fm_fast)
        return inv

    def cast_p(self, v_p, tile_idx=None):
        """`cast_p`, with the dropout mask applied first.

        **After `l_row`, before the O accumulation**, which is why this hangs
        off `cast_p` and not off `exp2`. The softmax denominator must be the
        *undropped* sum, or the result stops being an expectation of the
        undropped attention and the logsumexp the backward pass reads is wrong.
        Moving this one call earlier produces plausible output that is wrong by
        a per-row factor, and no shape check notices.

        The survivors are **not** scaled here. `1/(1-p)` is a per-row constant,
        so it folds into the reciprocal of `l` at the store -- one multiply per
        output row instead of one per score. `dropout_args` computes it host
        side; gfx1201 folds it the same way.

        The column runs are the bias ones. A lane's 16 scores are 16 columns of
        one row in a few contiguous spans, and the spans start at multiples of
        `randoms_per_offset`, so each is a whole number of Philox calls with no
        partial draw.
        """
        if const_expr(self.traits.ENABLE_DROPOUT):
            traits = self.traits
            ctx = self.ctx_ref
            # `v_p` is already a pair of 16-element lists here -- `exp2` hands
            # those back, where `v_s` arrives as a vector pair. No unpacking.
            lo, hi = list(v_p[0]), list(v_p[1])
            rng = ctx.philox_rng
            lane_n_off = 8 if traits.KV_VECTORIZED else 4
            col_base = fx.Int64(tile_idx * traits.BLOCK_N) + fx.Int64(self.lane_div_32 * fx.Index(lane_n_off))
            zero = fx.Float32(0.0)
            for half, values in ((0, lo), (1, hi)):
                for elem0, col_off, width in _score_column_runs(traits.KV_VECTORIZED):
                    first = rng.grid_offset(
                        ctx.philox_plane_base,
                        ctx.philox_row_stride,
                        fx.Int64(ctx.q_row),
                        col_base + fx.Int64(col_off + half * 32),
                    )
                    keep = rng.keep_span(ctx.philox_seed, first, width, ctx.idropout_p)
                    for j in range_constexpr(width):
                        values[elem0 + j] = keep[j].select(fx.Float32(values[elem0 + j]), zero)
            v_p = (lo, hi)
        # Every pack leaves through `mfma_operand_wait_state`. **Unreproduced
        # in this forward** -- a gap scan over six configurations (head dims
        # 64/96/128/224/256, both dtypes, dropout on and off, causal and not)
        # found zero exposed `v_cvt_pk_*_f32`-into-MFMA sites, having located
        # 88-344 candidate converts per kernel -- but AOTriton's scan of their
        # 216-kernel forward matrix found 44 exposed, the exposure is a
        # property of the schedule rather than of the source, and the barrier
        # measured free. One choke point covers both forward bodies, since
        # `WideSoftmaxHelper` subclasses this.
        #
        # `BwdDqSoftmaxHelper.cast_p` calls `DualwaveSoftmaxHelper.cast_p`
        # directly rather than through here, so dQ is **not** covered by this
        # and needs its own.
        p_lo_packs, p_hi_packs = super().cast_p(v_p)
        return (
            [mfma_operand_wait_state(p) for p in p_lo_packs],
            [mfma_operand_wait_state(p) for p in p_hi_packs],
        )

    # -- P3: generalized sliding window --------------------------------------
    #
    # The right bound needs no code here at all: it rides on `delta_i32`, which
    # `ParityKernelContext.init_runtime_indices` re-points at the resolved
    # `window_right`. Only the left bound is new.

    def _causal_mask_inplace(self, v_s, tile_idx, q_row_i32=None):
        """The causal right bound, then the window's left one.

        `super()` masks `col <= row + delta`. A window adds `col >= row - left`,
        and the two are independent comparisons against the same per-element
        column, so this composes rather than replacing anything.

        Plain `select`s rather than the paired inline asm the right bound uses.
        The asm packs two compares and two cndmasks per pair and exists because
        the causal mask is on the innermost path of every build; this one runs
        only in window builds, and `seq_pad_mask_inplace` next door already
        establishes the `select` form as the idiom here. Reach for the asm if a
        measurement asks for it, not before.
        """
        super()._causal_mask_inplace(v_s, tile_idx, q_row_i32=q_row_i32)
        if const_expr(not self.traits.WINDOW):
            return
        if q_row_i32 is None:
            q_row_i32 = self.ctx_ref.q_row_i32
        traits = self.traits
        s_lo, s_hi = v_s
        kv_start_i32 = fx.Int32(tile_idx * traits.BLOCK_N)
        # Same lane->column mapping the right bound uses; `s_hi` is the same
        # rows 32 columns further on, which is why its `rel` is 32 smaller.
        lane_n_off = 8 if traits.KV_VECTORIZED else 4
        lane_off_i32 = fx.Int32(self.lane_div_32) * fx.Int32(lane_n_off)
        rel_lo = fx.Int32(q_row_i32 - self.ctx_ref.window_left_i32 - kv_start_i32 - lane_off_i32)
        rel_hi = fx.Int32(rel_lo - fx.Int32(32))
        for p in range_constexpr(len(dualwave._causal_pair_thresholds(traits.KV_VECTORIZED))):
            thr_x, thr_y = dualwave._causal_pair_thresholds(traits.KV_VECTORIZED)[p]
            ix, iy = 2 * p, 2 * p + 1
            # Keep where the column has not fallen behind the left edge.
            s_lo[ix] = (rel_lo <= fx.Int32(thr_x)).select(s_lo[ix], self.c_neg_inf)
            s_lo[iy] = (rel_lo <= fx.Int32(thr_y)).select(s_lo[iy], self.c_neg_inf)
            s_hi[ix] = (rel_hi <= fx.Int32(thr_x)).select(s_hi[ix], self.c_neg_inf)
            s_hi[iy] = (rel_hi <= fx.Int32(thr_y)).select(s_hi[iy], self.c_neg_inf)

    def causal_mask_prologue_if_needed(self, v_s, tile_idx=None, kv_end_pos=None, **kwargs):
        """Two-sided version of "does this tile need masking at all?".

        The inherited test is one-sided -- `q_start + right < kv_end`, i.e. does
        the *first* row's right bound fall inside this tile. That is sufficient
        for causal, where the only partial tiles are the ones straddling the
        diagonal, and wrong for a window, whose left bound clips a second set
        of tiles somewhere else entirely.

        So the test gains a second term rather than being dropped. Masking
        unconditionally is also correct -- a mask on a fully live tile is a
        no-op -- but measured 197 us against causal's 126 at an unbounded left
        bound, because it forces the mask onto every interior tile in the walk.

        Each term is the worst case over the tile:

        - a column can overrun the right bound only if the *lowest* row's
          bound, `q_start + right`, lands inside the tile;
        - a column can fall behind the left bound only if the *highest* row's
          edge, `q_start + BLOCK_M - 1 - left`, is still above `kv_start`.
        """
        if const_expr(not self.traits.WINDOW):
            return super().causal_mask_prologue_if_needed(v_s, tile_idx=tile_idx, kv_end_pos=kv_end_pos, **kwargs)
        traits = self.traits
        if tile_idx is None:
            tile_idx = fx.Index(0)
        kv_end_tile = kwargs.get("kv_end_tile")
        if kv_end_pos is None:
            kv_end_pos = self.tile_start(tile_idx + fx.Index(1) if kv_end_tile is None else kv_end_tile)
        kv_start_pos = self.tile_start(tile_idx)
        q_start_pos_i32 = kwargs.get("q_start_pos_i32") or self.ctx_ref.q_start_pos_i32
        q_row_i32 = kwargs.get("q_row_i32") or self.ctx_ref.q_row_i32
        window_left_i32 = self.ctx_ref.window_left_i32
        delta_i32 = self.delta_i32
        mask_inplace = self._causal_mask_inplace
        to_lists = self.v_s_vec_to_lists

        @flyc.jit
        def _window_mask_if_needed(v_s, tile_idx, kv_end_pos, kv_start_pos, q_start_pos_i32, q_row_i32):
            s_lo, s_hi = v_s
            clipped_right = q_start_pos_i32 + delta_i32 < fx.Int32(kv_end_pos)
            clipped_left = fx.Int32(kv_start_pos) < q_start_pos_i32 + fx.Int32(traits.BLOCK_M - 1) - window_left_i32
            if clipped_right | clipped_left:
                lo_list, hi_list = to_lists(v_s)
                mask_inplace((lo_list, hi_list), tile_idx, q_row_i32=q_row_i32)
                s_lo, s_hi = dualwave._score_lists_to_vecs((lo_list, hi_list))
            return s_lo, s_hi

        return _window_mask_if_needed(v_s, tile_idx, kv_end_pos, kv_start_pos, q_start_pos_i32, q_row_i32)

    def rescale_o_serial(self, v_o, m_row, l_row, m_tile_max):
        """`rescale_o` without the `v_p` term, for the staged (unpipelined) loop.

        The dualwave schedule rescales O *and* the previous tile's P, because
        its softmax is split across clusters and a P from the last iteration is
        still in flight. The staged loop has no such P: it finishes each tile
        before starting the next, so at rescale time the only live state is O,
        the running max and the running sum. Passing a dummy `v_p` to
        `rescale_o` would work and would also emit a real multiply over it.

        Otherwise identical to `rescale_o`, term for term.
        """
        m_new = dualwave._fmax(m_row, m_tile_max, self.fm_fast)
        corr = rocdl.exp2(T.f32, as_mlir_value(dualwave._fsub(m_row, m_new, self.fm_fast)))
        self.scale_o(v_o, corr)
        v_o = _anchor_v_o(self.traits, v_o)
        l_row = dualwave._fmul(l_row, corr, self.fm_fast)
        return v_o, m_new, l_row


class ParityStoreHelper(dualwave.DualwaveStoreHelper):
    """O stores addressed by O's own strides.

    `_final_o_base` is the one place the production code spells O's address,
    and it spells it with *Q's* token stride plus `q_head_idx * HEAD_DIM` --
    correct only where O and Q share a layout. Overriding this single method is
    the whole change; the 128-bit store path above it is untouched.
    """

    def _final_o_base(self, q_row):
        return q_row * self.stride_o_seq_v + self.lane_div_32 * 8

    def _store_lse_row(self, m_row, l_row, q_row):
        """`_store_lse_row_unguarded`, skipped entirely when `LSE` is null.

        **LSE is an optional output, at runtime.** AOTriton's
        `attn_fwd_params::L` is documented "Can be `T2::get_null_tensor()`" and
        an inference caller that will never run a backward passes exactly that;
        their Triton kernel spells the contract as `L_not_null`. gfx1201's
        FlyDSL forward already has the equivalent (`_l_valid` in
        `flash_attn_func_gfx1201_aiw.py`); the gfx950 path simply never grew it,
        and a null `L` reached `_make_ws_rsrc` as a descriptor based at address
        0, where the store faults.

        **`RETURN_LSE` does not cover this**, and that is the point worth
        keeping. It is compile-time: `const_expr(traits.RETURN_LSE)` deletes the
        store from the binary, so `return_lse=False` is not "LSE optional", it
        is "LSE never written" -- and one binary has to serve both the caller
        that wants an LSE and the caller that passes null.

        The condition is wave-uniform, so this lowers to a scalar branch.
        Everything the store needs -- the log, the scale, the addressing --
        stays *inside* the guard, for the reason gfx1201 records at its own:
        hoisting that arithmetic out cost 8% at head_dim 256 even with the
        store still predicated, because the values then stay live across the
        epilogue for every wave, including the ones with nothing to store.

        One choke point covers both forward bodies and both layouts, since
        `WideStoreHelper` subclasses this and both LSE paths funnel here.
        """
        store = self._store_lse_row_unguarded
        lse_not_null = fx.Int64(fx.ptrtoint(fx.get_iter(self.LSE))) != fx.Int64(0)

        @flyc.jit
        def _store_lse_if_l_not_null():
            if lse_not_null:
                store(m_row, l_row, q_row)

        _store_lse_if_l_not_null()

    def _store_lse_row_unguarded(self, m_row, l_row, q_row):
        """LSE addressed through `VarlenBits`, which decides three things here.

        LSE is always **compact** -- it is the one tensor whose strides are not
        a free variable but a function of the bits, the head count and the
        token count, which is why no `lse_stride` is passed (plan section 4.2).
        Compact is not the same as fixed, though, and the production formula
        `q_head_idx * seq_len_v + q_row` pins all three of the things that
        vary:

        - **the token pitch.** `seq_len_v` is `max_seqlen_q`, which is right
          for a batched layout that pads every row-group to the longest
          sequence. A stacked Q side runs to the *batch total* instead, which
          is what `lse_token_pitch` decoded into `lse_tokens_i32`.
        - **the row origin.** A packed sequence starts at `q_row_off`, not 0.
        - **the layout.** Bits 17:16 choose `_HT` -- `(H, T)`, T contiguous,
          AOTriton's -- or `_TH`, which is Transformer Engine's. The production
          formula is `_HT` written out, so `_TH` was silently ignored.

        `fmha.lse_row_addressing` is gfx1201's and returns `(base, pitch)` so
        the per-row part stays a multiply-add. It is called with **batch 0**
        because the descriptor below already folds the batch in, and the two
        must not both count it; that works for either layout because a batch's
        rows are contiguous in both, `H * tokens` of them.

        Non-varlen builds keep the production expression exactly. It is what
        this generalizes to at `varlen_bits == 0`, `row_off == 0` and
        `tokens == seq_len_v` -- but only as a *runtime* equality, and emitting
        a select per store to rediscover a constant is not worth it on the path
        every dense build takes.
        """
        # **The runtime head count, not `traits.NUM_HEADS_Q`.** The trait is
        # compile-time, which is right only where a kernel is compiled per
        # shape. AOTriton compiles one binary for every head count and pins
        # `num_heads=1`, and this descriptor was the counterexample to that
        # pin's assumption: with the trait the per-batch slice covers `1 *
        # tokens` rows, `batch_idx` advances by one head instead of `H`, and
        # the hardware bound drops the row for every head but the first --
        # `L` comes back written for `h == 0` and untouched everywhere else,
        # with no error reported. gfx1201 already passes the runtime count to
        # `lse_row_addressing`, which is why its identical pin is sound.
        #
        # **The non-varlen shortcut is gone with it.** It delegated to
        # `super()._store_lse_row`, whose copy in `flash_attn_utils.py` bakes
        # the same trait -- so a dense build carried the same defect and there
        # was nothing left to inherit. At `varlen_bits == 0` the addressing
        # below *is* the production formula (`base = q_head_idx * tokens`,
        # `pitch = 1`), as this docstring already said; it is now reached at
        # runtime rather than as a constant, which costs two scalar ops once
        # per row in the epilogue.
        num_heads_q = fx.Index(self.num_head_q)
        tokens = fx.Index(self.lse_tokens_i32)
        per_batch = num_heads_q * tokens
        per_batch_bytes = per_batch * fx.Index(4)
        rsrc = dualwave._make_ws_rsrc(
            fx.Int64(fx.ptrtoint(fx.get_iter(self.LSE))),
            self.batch_idx * per_batch_bytes,
            per_batch_bytes,
        )
        lse_val = dualwave._fadd(
            dualwave._fmul(m_row, self.c_ln2_f, self.fm_fast),
            dualwave.fmath.log(as_mlir_value(l_row), fastmath=self.fm_fast),
            self.fm_fast,
        )
        base, pitch = fmha.lse_row_addressing(
            self.varlen_bits_arg,
            fx.Index(0),
            self.q_head_idx,
            num_heads_q,
            tokens,
            self.q_row_off,
        )
        # One writer per row: low half-wave, in-bounds row; everything else is
        # redirected to the sentinel the buffer bound drops.
        lse_local = base + q_row * pitch
        off_row = (q_row < self.seqlen_q_v).select(lse_local, per_batch)
        off = fx.Index((self.lane < fx.Index(32)).select(off_row, per_batch))
        dualwave._ws_store_f32(lse_val, off, rsrc)

    def _final_o_global(self, o_base, dc, g):
        """The store's element offset, redirected out of the buffer if past `hdim_vo`.

        A 128-bit store is all-or-nothing, so a chunk straddling `hdim_vo`
        cannot be partially written -- and does not need to be. The D pitch is
        contractually a multiple of 8 elements, so a chunk that *starts* inside
        `hdim_vo` ends inside the allocation, and writing its tail into the
        caller's own padding is exactly what that contract permits. Only
        chunks starting at or past `hdim_vo` must be suppressed.

        Suppressed by *address*, not by a branch: pushing the offset past the
        descriptor's `num_records` makes the hardware drop the store, which
        costs one select on a lane-varying value instead of an `scf.if` around
        a 128-bit store. `store_lse` uses the same device.
        """
        off = super()._final_o_global(o_base, dc, g)
        if const_expr(not self.PADDED_HEAD):
            return off
        col_base = fx.Index(dc * self.traits.D_CHUNK + 2 * g * 8) + self.lane_div_32 * fx.Index(8)
        in_range = MaskedAxis(fx.Index(self.hdim_vo)).valid(col_base)
        return fx.Index(in_range.select(fx.Index(off), self.o_oob_off))
