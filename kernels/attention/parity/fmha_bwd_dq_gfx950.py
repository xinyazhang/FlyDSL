# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Flash-attention **backward dQ (and dB)** for gfx950 -- AOTriton's `bwd_kernel_dq`.

B2 and B3 of `sdpa-bwd-plan-gfx950.md`. Dense, non-causal, bf16, the full
`LADDER` from head_dim 32 to 512, with the 8xD input contract, padded heads and
asymmetric `hdim_qk`/`hdim_vo`. The dK/dV half is B1 and lives in its own file;
there is deliberately no fused kernel (plan section 4).

    P  = exp2(qk_scale * S - lse2)        S  = Q.K^T,  qk_scale = sm_scale*log2e
    dP = dO . V^T
    dS = P * (dP - delta)                 delta = rowsum(dO*O), a host argument
    dQ = sum_j dS . K * sm_scale
    dB = dS                               (bias-gradient builds only)

--- The whole design, in one observation --------------------------------------

**Every one of the three GEMMs is a GEMM the forward already emits**, once the
right tensor is staged in the right LDS layout. Nothing below derives a new
lane map, and that is the point: the forward's `V` read path is validated end
to end, and plan section 3 says to reuse it rather than re-derive it.

| this kernel | forward equivalent | A operand (M axis) | B operand (N axis) |
|---|---|---|---|
| `S  = Q.K^T`  | `qk` | K, `load_k` | Q, `ParityQLoader` |
| `dP = dO.V^T` | `qk` | V, `load_k` | dO, same loader |
| `dQ = dS.K`   | `pv` | K^T, `load_v` | dS, packed like P |

The score-shaped accumulators are all `[kv][q]` -- M is the KV token, N is the
query row -- which is exactly the forward's `v_s`, so `seq_pad_mask_if_needed`,
`sub_m`, `exp2` and `cast_p` apply unchanged. `dS` reaches the third GEMM
through `cast_p`, so it carries the *same* K permutation `P` does, which is why
`load_v`'s transpose read lands K in the operand registers the MFMA wants with
no further shuffle.

--- Two LDS tiles, and the read that makes it two ------------------------------

GEMM1 wants K with the KV token on the MFMA's M axis and `d` contracted; GEMM3
wants K with `d` on M and the KV token contracted. Those are the forward's K
path and its V path, and the two stagings differ **only in the line padding**
(`SMEM_K_PAD` 8 elements against `SMEM_V_PAD` 32). B2 answered that by staging
K twice. B3 cannot: three slots is 199 KB at head_dim 512 against a 163840 B
cap, so the double staging is what stops the ladder rather than merely costing
DMA.

So the K tile is staged **once, in the V layout**, and read two ways:

    (V, buf 0) <- K   -> load_v(0)          GEMM3, the stock transpose read
                      -> load_k_packs()     GEMM1, the K path on V-pitch lines
    (K, buf 0) <- V   -> load_k_packs()     GEMM2, the stock K path
    buf 1                unused, and not allocated

`LDS_KV_TOTAL_SIZE` drops to one buffer -- `SMEM_K_TILE_ELEMS +
SMEM_V_TILE_ELEMS`, 133 KB at head_dim 512 -- which is a single trait field
(`BwdDqTraits`, `make_bwd_dq_traits`).

**The K tile is the one in the V region, not the other way round, and that is
deliberate.** One of the two readers has to be re-pointed at the other pitch,
and the K path is a plain `llvm.LoadOp` whose whole addressing is two
expressions; the transpose path is `ds_read_b64_tr_b16` with alias scopes, an
even-VGPR-pair constraint and two open hazards against it (`sdpa_lore_gfx950`).
Leave the fragile one stock. `BwdDqKvLdsToVgprLoader` re-points the other by
scoping two trait fields, so the *formula* is still the shared one.

The alias scopes stay truthful: the K tile is read under `lds_v0` by both of
its readers and the V tile under `lds_k0`, so no scope claims two regions are
disjoint when they are the same memory.

--- Conventions this kernel must not get wrong --------------------------------

Three, and each is a *silent* wrong answer if broken, which is why the tests
check them against our own forward rather than only against torch:

- **`qk_scale = sm_scale * log2e` is applied to the f32 scores, `sm_scale`
  alone to the dQ accumulator at the end.** AOTriton spells them
  `p = exp2(qk_scale*qk - l_i)` and `dq *= sm_scale`. The forward instead folds
  `qk_scale` into Q and rounds the product back to bf16; **this kernel
  deliberately does not**, and `BwdDqSoftmaxHelper.scale_and_sub_lse` has the
  measurement that says why.
- **Neither Q nor `dO` is pre-scaled.** `ParityQLoader.scale_all` is one
  keystroke away and multiplies every gradient by `qk_scale`; nothing checks
  shapes.
- **LSE is read in natural units and converted here.** The forward writes
  `m_row*ln2 + ln(l)`, so `lse2 = lse * log2e` (AOTriton's `l_i = ... *
  RCP_LN2`). `fmha.lse_row_addressing` owns the layout for both LSE and delta.

--- The two head extents are crossed, and that is not a detail ----------------

In the forward, the tensor read through the K register path has the *qk*
extent and the tensor written through the O store has the *vo* one. **Both are
the other way round here**, because GEMM2 reads V through the K path and the
store writes dQ, which is Q-shaped:

- two `BwdDqKvLdsToVgprLoader` instances, one per tile, differing in the LDS
  pitch *and* in which extent their padded-head mask is written against
  (plus `HDIM_VO_FLOOR`, the vo counterpart of `HDIM_QK_FLOOR`);
- `BwdDqStoreHelper`, which rebinds `hdim_vo` to the qk extent.

They coincide in every symmetric build, so only `test_asymmetric_hdim` can tell
the fix from its absence.

--- Not implemented, deliberately ---------------------------------------------

Causal, windows, varlen, dropout, bias *input*, split-K, paged, `D_STAGES`,
d-axis sharding. `BwdDqKnobs._with_traits` refuses each by name rather than
ignoring it -- every one of them would otherwise build, run and return a
correctly-shaped wrong answer, and `D_STAGES` nearly did: the forward's knob
policy turns it on above head_dim 256 and the inherited GEMM helpers then
reduce over one stage of the head dim while this loop never advances the stage.

--- Tensor argument order is the ABI ------------------------------------------

    q, k, v, b, do, dq, db, lse, delta

Four groups, and the grouping is the mnemonic: the **forward's inputs**
(`q, k, v, b`), then the **backward's input** (`do`), then this kernel's
**outputs**, then the **lower-rank** tensors (`lse`, `delta`, both rank 2).

It is also AOTriton's order -- `bwd_kernel_dq(Q, K, V, B, sm_scale, DO, DQ, DB,
L, D, ...)` and `bwd_kernel_dk_dv(Q, K, V, B, sm_scale, DO, DK, DV, L, D, ...)`
-- so a reader moving between the Triton reference and this file does not have
to re-derive the mapping, and neither does anyone eventually dispatching the
compiled hsaco directly.

`b` sits in the forward-input group even though no build here reads it yet: the
slot is held so adding bias in B6 does not move the wire format. It was
initially placed after `delta`, at the end, which is where an unused argument
naturally lands and is exactly why the grouping is written down rather than
left to accrete.
"""

from dataclasses import replace

import fmha_abi_gfx1201 as abi
import fmha_common_gfx1201 as fmha
from fmha_dualwave_gfx950 import (
    ParityGemmHelper,
    ParityKernelContext,
    ParityKvGmemToLdsLoader,
    ParityKvLdsToVgprLoader,
    ParityQLoader,
    ParitySoftmaxHelper,
    ParityStoreHelper,
)

# Private to `fmha_dualwave_gfx950`, and imported anyway: they are the
# granule-general spellings of the K read's per-lane base and k-step offset,
# and the alternative is a second copy of two formulas the read and write sides
# of an LDS layout both depend on. Reaching across within one kernel family
# beats transcribing.
from fmha_dualwave_gfx950 import _k_read_base as _parity_k_read_base
from fmha_dualwave_gfx950 import _ks_offset as _parity_ks_offset
from fmha_tuning_bwd_dq_gfx950 import BwdDqKnobs, bwd_dq_knobs
from fmha_tuning_gfx950 import FmhaInputMetadata
from gfx950_standalone import buffer_ops, dualwave

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

KERNEL_NAME = "fmha_bwd_dq_gfx950_kernel"

__all__ = [
    "KERNEL_NAME",
    "BwdDbStoreHelper",
    "BwdDqKernelContext",
    "BwdDqKvGmemToLdsLoader",
    "BwdDqKvLdsToVgprLoader",
    "BwdDqSoftmaxHelper",
    "BwdDqStoreHelper",
    "BwdRowInputLoader",
    "BwdSecondaryQLoader",
    "build_fmha_bwd_dq_gfx950_module",
    "build_fmha_bwd_dq_gfx950_module_primary",
]

_s_barrier = dualwave._s_barrier
_s_waitcnt = dualwave._s_waitcnt
_sched_barrier = dualwave._sched_barrier


def _carried(values, count):
    """An `scf.for`'s carried values as a list of `count`, however many it hands back.

    **A loop with exactly one carried value hands it back unwrapped**, because
    an `scf.for` with one result is a value rather than a tuple. This kernel
    carries `D_CHUNKS` accumulators and nothing else, so head_dim 32 -- the one
    rung where `D_CHUNKS == 1` -- is the only place that bites, and it bites
    two frames away: `loop_results[0]` on a `vector<16xf32>` returns the *first
    f32 element*, and the failure surfaces inside `_scale_o_accs` as "Cannot
    cast type to VectorType". The forward never sees this because it always
    carries `m_row` and `l_row` alongside.
    """
    if const_expr(isinstance(values, (list, tuple))):
        return list(values)
    if const_expr(count != 1):
        raise AssertionError(f"expected {count} carried values, got one unwrapped value")
    return [values]


_COMPILED = {}

# The forward's, unchanged: this kernel is the same instruction mix on the same
# schedule primitives, so a different set here would only make the two
# incomparable at the ISA level.
_COMPILE_HINTS = {
    "fast_fp_math": True,
    "unsafe_fp_math": True,
    "llvm_options": {
        "enable-post-misched": False,
        "lsr-drop-solution": True,
    },
}


class BwdDqKernelContext(ParityKernelContext):
    """The forward's parity context plus dO, dB and the two row inputs.

    Subclassed rather than ported, per the contract: the strides, the padded
    head, the varlen decode and the descriptor machinery are all inherited, and
    `O` is bound to `dQ` so `ParityStoreHelper` writes the gradient with no
    change at all. What is added is one more Q-shaped tensor (`dO`), one
    score-shaped output (`dB`), and the LSE/delta pair.
    """

    def __init__(self, traits, *, do_strides, db_strides=(0, 0, 0), DO=None, DB=None, Delta=None, **kwargs):
        super().__init__(traits, **kwargs)
        self.stride_do_batch, self.stride_do_head, self.stride_do_seq = do_strides
        # `_seq_q` for the reason `ParityKernelContext` gives for the bias
        # input's: dB is `(batch, head, seqlen_q, seqlen_k)` and a bare `_seq`
        # does not say which of the two it is.
        self.stride_db_batch, self.stride_db_head, self.stride_db_seq_q = db_strides
        self.DO = DO
        self.DB = DB
        self.Delta = Delta

    def init_runtime_indices(self, **kwargs):
        super().init_runtime_indices(**kwargs)
        self.stride_do_seq_v = fx.Index(self.stride_do_seq)

    def init_descriptors(self, **kwargs):
        """The forward's four views, plus dO's and dB's.

        `dO` gets exactly Q's treatment -- same slab shape, same row origin,
        same `seqlen_q` bound -- because it *is* Q-shaped: (batch, head, q row,
        d), with the vo head dim rather than the qk one. The bound is what
        makes a row past `seqlen_q` read zero instead of faulting, which is the
        whole reason the ragged tail needs no branch.
        """
        traits = self.traits
        super().init_descriptors(**kwargs)
        self.do_div = self._slab_view(
            self.DO,
            self.stride_do_batch,
            self.stride_do_head,
            self.stride_do_seq,
            self.q_row_off,
            self.q_head_idx,
            self.seqlen_q_v,
        )
        self.do_gmem_elem_offset = self.q_start * self.stride_do_seq_v
        if const_expr(traits.STORE_DB):
            # Same slab shape as the forward's bias descriptor -- dB is indexed
            # (batch, head, q row, kv col) and the KV axis is contractually
            # contiguous -- and a raw resource for the same reason: the stores
            # are per-lane at an address the lane computes, which is
            # `buffer_ops.buffer_store`'s shape and not the copy atom's.
            db_span = self.seqlen_q_v * fx.Index(self.stride_db_seq_q)
            # First element past the descriptor. A store redirected here is
            # dropped by the hardware bound; see `BwdDbStoreHelper`.
            self.db_oob_off = db_span
            self.db_rsrc = buffer_ops.create_buffer_resource(
                self.DB,
                max_size=False,
                num_records_bytes=as_mlir_value(db_span * fx.Index(traits.BF16_BYTES)),
                base_byte_offset=as_mlir_value(
                    self._slab_byte_base(
                        self.stride_db_batch,
                        self.stride_db_head,
                        self.stride_db_seq_q,
                        self.q_row_off,
                        self.q_head_idx,
                    )
                ),
            )

    def init_row_inputs(self):
        """Descriptors and the shared row addressing for LSE and delta.

        **One addressing for both**, which is `fmha.lse_row_addressing`'s
        stated contract: delta is produced beside LSE by the same caller and
        giving it its own decode would double the work for no expressiveness.
        Called with batch 0 because the descriptors below already fold the
        batch in, exactly as the forward's varlen LSE store does.
        """
        tokens = fx.Index(self.lse_tokens_i32)
        per_batch = fx.Index(self.num_head_q) * tokens
        per_batch_bytes = per_batch * fx.Index(4)
        self.row_base, self.row_pitch = fmha.lse_row_addressing(
            self.varlen_bits_arg,
            fx.Index(0),
            self.q_head_idx,
            self.num_head_q,
            tokens,
            self.q_row_off,
        )
        # The sentinel a row past `seqlen_q` is redirected to. Without it the
        # offset would run into the *next head's* rows, which is in bounds and
        # returns a plausible LSE for the wrong row -- finite, and wrong in a
        # way no shape check sees. The row is discarded at the store either
        # way, but a zero keeps the arithmetic in between boring.
        self.row_oob_off = per_batch
        self.lse_rsrc = dualwave._make_ws_rsrc(
            fx.Int64(fx.ptrtoint(fx.get_iter(self.LSE))), self.batch_idx * per_batch_bytes, per_batch_bytes
        )
        self.delta_rsrc = dualwave._make_ws_rsrc(
            fx.Int64(fx.ptrtoint(fx.get_iter(self.Delta))), self.batch_idx * per_batch_bytes, per_batch_bytes
        )


class BwdRowInputLoader(ParityStoreHelper):
    """`lse2` and `delta` for this lane's query row, once per kernel.

    A helper on the store base class only for the context copy it brings; it
    stores nothing. Both values are per (query row) scalars, and a lane owns
    exactly one query row (`lane_mod_32`), so these are the same shape as the
    forward's running `m_row` -- one f32 register each, broadcast over the 16
    score elements by every use.
    """

    def load(self, q_row):
        ctx = self.ctx_ref
        off = self.row_base + q_row * self.row_pitch
        off = fx.Index((q_row < self.seqlen_q_v).select(off, self.row_oob_off))
        off_i32 = as_mlir_value(fx.Int32(off))
        lse = buffer_ops.buffer_load(ctx.lse_rsrc, off_i32, vec_width=1, dtype=fx.Float32)
        delta = buffer_ops.buffer_load(ctx.delta_rsrc, off_i32, vec_width=1, dtype=fx.Float32)
        # The forward writes LSE in natural units (`m_row*ln2 + ln(l)`), and
        # every exponent downstream is `exp2`, so the conversion happens once
        # here rather than per element. AOTriton's `l_i = tl.load(...) *
        # RCP_LN2` is the same fold.
        lse2 = dualwave._fmul(fx.Float32(lse), fx.Float32(dualwave._LOG2E), self.fm_fast)
        return lse2, fx.Float32(delta)


class BwdDqSoftmaxHelper(ParitySoftmaxHelper):
    """The forward's softmax, minus the one place it trades accuracy for speed."""

    def scale_and_sub_lse(self, v_s, qk_scale, lse2):
        """`qk_scale * S - lse2`, one FMA per score element.

        **This is where the backward deliberately stops matching the forward,
        and it is worth 10x at a large `sm_scale`.** The forward folds
        `sm_scale * log2e` into Q and rounds the product back to bf16, which
        saves a multiply per score; the error that introduces is `|S| * 2^-8`
        in the *exponent*, so `P = exp2(S - lse2)` inherits it as a relative
        error. The forward tolerates that because `O` is a normalised average
        and the error largely cancels; `dS = P * (dP - delta)` does not
        normalise, and `dQ` sums it over the whole key axis.

        Measured at `B=1 H=4 S=512 d=64`, max error against an fp64 reference:

            sm_scale   Q pre-scaled (forward's fold)   scaled here
              0.05            1.8e-4                     1.6e-4
              0.25            4.3e-2                     2.0e-2
              1.00            6.9e-1                     6.8e-2

        A host model of both variants reproduces the kernel's own numbers to
        two digits, which is what identifies the Q rounding rather than
        anything else as the cause. AOTriton scales after the dot in *both*
        directions (`qk += Qk_scale * tl.dot(q, k)`), so this is its
        arithmetic, not a new choice.

        Not `dualwave._scale_sub_score_pair`: that one derives the offset from
        an *unscaled* row max (`fma(s, scale, -scale*m)`), and `lse2` is
        already in the scaled base-2 domain. Passing it there would need a
        divide by `sm_scale` to undo a multiply.

        The KV tail mask runs **after** this, not before, so no infinity ever
        reaches the FMA -- `qk_scale * -inf` is correct but puts a real
        infinity into arithmetic under `fastmath<fast>`, which plan section 5
        records as the thing that silently deleted a mask on gfx1201.
        """
        s_lo, s_hi = v_s
        scale_v = Vec.from_elements([fx.Float32(qk_scale)], fx.Float32).broadcast_to(16)
        neg_lse2 = dualwave._fsub(self.c_zero_f, lse2, self.fm_fast)
        neg_v = Vec.from_elements([fx.Float32(neg_lse2)], fx.Float32).broadcast_to(16)
        lo = fmath.fma(Vec(s_lo), scale_v, neg_v, fastmath=self.fm_fast)
        hi = fmath.fma(Vec(s_hi), scale_v, neg_v, fastmath=self.fm_fast)
        return as_mlir_value(lo), as_mlir_value(hi)


class BwdSecondaryQLoader(ParityQLoader):
    """A second Q-shaped loader, for dO.

    `ParityQLoader` reads three attributes to find its tensor -- the buffer
    view, the sequence stride and the tile's element origin -- and everything
    else about it is the lane map, which dO shares with Q exactly. So this
    rebinds those three and inherits the loop, rather than being a second copy
    of an addressing scheme.

    `hdim_qk` is rebound too: the padded-head mask inside `load_pack` is
    against the *reduction* extent for Q, and dO's is `hdim_vo` (it contracts
    with V's D axis, not K's). Inert while `PADDED_HEAD` is off, which is B2 --
    set correctly now so B3 does not have to find it.
    """

    def __init__(self, ctx, div, stride_seq_v, gmem_elem_offset, hdim):
        super().__init__(ctx)
        self.q_div = div
        self.stride_q_n_v = stride_seq_v
        self.q_gmem_elem_offset = gmem_elem_offset
        self.hdim_qk = hdim


class BwdDqKvGmemToLdsLoader(ParityKvGmemToLdsLoader):
    """One staging routine, two (tensor, region) pairs.

    The inherited `load_k` / `load_v` pair hardcodes which tensor goes to which
    region, and this kernel puts **K in the V region**. `_stage` is the
    inherited body with the tensor, the stride and the `m0` table all handed
    in, so the two calls below differ only in their arguments.

    The `m0` table *is* the pitch: `k_dma_m0` addresses lines at
    `SMEM_K_LINE_STRIDE` and `v_dma_m0` at `SMEM_V_LINE_STRIDE`, and each has to
    match the reader that later walks the region. Pairing them wrongly reads a
    tile written with a different pitch -- plausible numbers, no fault.

    Setting `stride_kv_n_v` before delegating is the inherited trick and is
    safe for the inherited reason: tracing is eager, so the attribute is read
    while the call below runs and no branch is open across the swap.
    """

    def _stage(self, tile_start, src_div, stride_seq_v, dma_m0, buf_id):
        self.stride_kv_n_v = stride_seq_v
        self.dma_stage = 0
        self._async_load_kv_linear(
            dma_m0,
            buf_id,
            src_div,
            self.kv_gmem_elem_offset,
            tile_start * stride_seq_v,
            self.NUM_DMA_K,
        )

    def stage_k(self, tile_start):
        """K into the V region: GEMM1's and GEMM3's A operand, one copy."""
        self._stage(tile_start, self.k_div, self.stride_k_seq_v, self.ctx_ref.v_dma_m0, 0)

    def stage_v(self, tile_start):
        """V into the K region: GEMM2's A operand."""
        self._stage(tile_start, self.v_div, self.stride_v_seq_v, self.ctx_ref.k_dma_m0, 0)


class BwdDqKvLdsToVgprLoader(ParityKvLdsToVgprLoader):
    """The forward's two readers, with the K path re-pointable at the V pitch.

    One instance per tile, because the *padded-head extent* differs between
    them as much as the pitch does: the K tile's D axis is `hdim_qk` and the V
    tile's is `hdim_vo`. B2 had one instance and would have masked V's columns
    against Q's extent the moment a padded head arrived -- one of the two
    crossed sites the plan's B2 outcome names.

    `v_layout` is a Python bool fixed at construction, so every `const_expr`
    below folds and neither instance emits the other's code.
    """

    def __init__(self, ctx, *, v_layout, hdim, hdim_floor):
        super().__init__(ctx)
        self.v_layout = bool(v_layout)
        # `load_k`'s mask reads these two by these names. Rebinding them is
        # what makes one class serve both extents.
        self.hdim_qk = hdim
        self.HDIM_QK_FLOOR = int(hdim_floor)

    def _lds_pack(self, elem_idx, scope_name):
        """One 8xbf16 MFMA operand pack out of LDS, under a chosen alias scope.

        `dualwave._load_k_pack_aligned` is the same four lines, except that it
        derives the scope as `lds_k{buf_id}` -- and this kernel's K tile lives
        in the `v0` region. A scope that names the wrong region is not a
        cosmetic problem: `ROCDL_LDS_Read_Tr_IntrOp` and this load are told
        they cannot alias, and here they read the same bytes.
        """
        traits = self.traits
        ptr = buffer_ops.get_element_ptr(self.lds_kv_base_ptr, byte_offset=elem_idx * traits.BF16_BYTES, elem_type=T.i8)
        return llvm.LoadOp(
            self.kv_mfma_pack_type,
            ptr,
            alignment=16,
            alias_scopes=dualwave._dualwave_lds_alias_scopes(scope_name),
            noalias_scopes=dualwave._dualwave_lds_noalias_scopes(scope_name, traits.LDS_SCOPE_NAMES),
        ).result

    def _read_k_packs(self, buf_id, urk_base):
        """The K path, over whichever region this instance owns.

        The `v_layout` arm re-points three things and no more: the line pitch,
        the k-step outer stride (which is `SMEM_N_RPT` lines) and the region
        base. Everything else -- `_parity_k_read_base`, `_parity_ks_offset`,
        the `N_STRIP` lo/hi split -- is the shared formula, evaluated against a
        `replace`d traits object rather than transcribed. The two pitches are
        the *only* difference between the layouts, which is what makes the
        substitution total.

        `urk_base` is ignored on this arm: `load_k` computes it from
        `k_lds_read_base_per_lane`, which the context derived at the K pitch.
        """
        if const_expr(not self.v_layout):
            return super()._read_k_packs(buf_id, urk_base)
        full = self.traits
        self.traits = replace(
            full,
            SMEM_K_LINE_STRIDE=full.SMEM_V_LINE_STRIDE,
            K_LDS_TO_REG_KSTEP_OUTER_STRIDE=full.SMEM_N_RPT * full.SMEM_V_LINE_STRIDE,
        )
        try:
            traits = self.traits
            scope = dualwave._dualwave_lds_scope("v", buf_id)
            base = dualwave._v_buf_base(traits, buf_id) + _parity_k_read_base(
                traits, self.lane_mod_32, self.lane_div_32
            )
            k_lo = [None] * traits.K_STEPS_QK
            k_hi = [None] * traits.K_STEPS_QK
            for ks in range_constexpr(traits.K_STEPS_QK):
                idx = base + _parity_ks_offset(traits, ks)
                k_lo[ks] = self._lds_pack(idx, scope)
                k_hi[ks] = self._lds_pack(idx + traits.K_LDS_TO_REG_N_STRIP_STRIDE, scope)
            return k_lo, k_hi
        finally:
            self.traits = full


class BwdDqStoreHelper(ParityStoreHelper):
    """The O store, told that its output is `hdim_qk` wide.

    `_final_o_global` suppresses chunks starting at or past `self.hdim_vo`,
    because in the forward the tensor it writes *is* O. Here it writes dQ,
    which is Q-shaped and `hdim_qk` wide. The two coincide in every symmetric
    build and cross the moment they do not -- the second of the two sites the
    plan's B2 outcome names.

    Rebinding the attribute rather than overriding the method: the suppression
    is one comparison inside a method that also computes the address, and a
    copy of it would be a copy of both.
    """

    def __init__(self, ctx):
        super().__init__(ctx)
        self.hdim_vo = ctx.hdim_qk


class BwdDbStoreHelper(ParityStoreHelper):
    """`dB = dS` for one KV tile, one element per store.

    **Per element, and that is a correctness decision rather than an
    oversight.** A lane's 16 scores are four runs of four contiguous KV
    columns (`_score_column_runs`), so a 4-wide vector store is available and
    is what the bias *load* uses. It is not available here: a run straddling
    `seqlen_k` would have to be partially written, and a 128-bit store is
    all-or-nothing. Suppressing the whole run instead is exact only when
    `seqlen_k` is a multiple of 4, and silently corrupting the next row of dB
    when it is not is precisely the failure mode this phase is written to
    avoid. So the tail is paid for in full, on a path that is off by default.

    Suppression is by *address*, not by a branch -- pushing the offset past the
    descriptor's `num_records` makes the hardware drop the store -- which is
    the same device `ParityStoreHelper._final_o_global` and `_store_lse_row`
    use.

    It costs 2-5.5x, measured across the ladder, and it is a *cost* rather than
    a ceiling: every rung including 512 builds and is correct with it on. The
    numbers and the register accounting are in the B3 outcome section of
    `sdpa-bwd-plan-gfx950.md`. A vectorised version needs *two* things and not
    one: a runtime second arm for the tile containing `seqlen_k`, and a
    `stride_db_seq_q` divisible by 4, since an 8-byte store off a row pitch of,
    say, 201 elements is 2-byte aligned.

    --- The source is the packed bf16, not the f32 -----------------------------

    **This shortens a live range rather than shrinking a value**, and at the
    top of the ladder that is the difference that matters. `dS` exists twice:
    as 32 f32 scores, and -- after `cast_p` -- as 4 packed `v8` bf16 vectors,
    16 VGPRs against 32. Storing from the f32 list keeps *both* alive across
    the 32-store sequence and its address arithmetic, at exactly the point
    where head_dim 384 and 512 have no registers left. Reading from the packs
    lets the f32 form die at `cast_p`.

    The values are identical: `dB` is a bf16 tensor either way, so the f32 ->
    bf16 rounding happens once regardless -- this only moves *where*. The
    element map is unchanged, because `_pack_p_v8_slices` packs list element
    `pks*8 + j` into lane slot `j` of pack `pks`, in order.
    """

    def store_tile(self, ds_packs, tile_idx, q_row):
        traits = self.traits
        ctx = self.ctx_ref
        lo_packs, hi_packs = ds_packs
        row_base = q_row * fx.Index(ctx.stride_db_seq_q)
        col_base = dualwave._seq_pad_col_base(traits, tile_idx, lane_div_32=self.lane_div_32)
        in_row = q_row < self.seqlen_q_v
        for half, packs in ((0, lo_packs), (1, hi_packs)):
            for pks in range_constexpr(len(packs)):
                vec = Vec(packs[pks], (8,), self.elem_dtype)
                for j in range_constexpr(8):
                    # The same element -> column map the KV tail mask uses.
                    # Derived once, in `flash_attn_utils`, and read here rather
                    # than transcribed: the mask and this store must agree
                    # about which column an element is, or dB is a permutation
                    # of itself.
                    r = pks * 8 + j
                    col_i32 = col_base + fx.Int32(dualwave._seq_pad_score_threshold(traits, r) + half * 32)
                    live = in_row & (col_i32 < self.seqlen_kv_i32)
                    off = fx.Index(live.select(row_base + fx.Index(col_i32), ctx.db_oob_off))
                    buffer_ops.buffer_store(as_mlir_value(vec[j]), ctx.db_rsrc, as_mlir_value(fx.Int32(off)))


def build_fmha_bwd_dq_gfx950_module_primary(meta, knobs):
    """Build the backward dQ kernel for a resolved (meta, knobs) pair.

    Same shape as the forward's builder and for the same reason: `meta` is what
    the caller asked for, `knobs` is what the tuning policy answered, and
    nothing here falls back to a policy.
    """
    if knobs.traits is None:
        raise ValueError("knobs must be resolved: call `bwd_dq_knobs(arch, ...).resolve(meta)` first")
    traits = knobs.traits

    BLOCK_DMODEL = knobs.block_dmodel
    PADDED_HEAD = knobs.padded_head
    HDIM_QK_FLOOR = knobs.hdim_qk_floor
    STORE_DB = traits.STORE_DB
    BUILD_SM_SCALE = meta.sm_scale

    # `traits.cache_tag` does not know about the tile geometry or `STORE_DB`,
    # so everything the build depends on goes in here. Two builds colliding in
    # the JIT disk cache is what a knob sweep hits first.
    _cache_tag = (
        traits.cache_tag,
        BLOCK_DMODEL,
        PADDED_HEAD,
        HDIM_QK_FLOOR,
        traits.HDIM_VO_FLOOR,
        STORE_DB,
        BUILD_SM_SCALE,
        (knobs.num_waves, knobs.block_m, knobs.block_n, knobs.head_dim_granule),
    )

    _lds_elem_dtype = dualwave.dtype_to_elem_type(traits.DTYPE_STR)

    @fx.struct
    class SharedStorage:
        kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]

    @flyc.kernel(known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def fmha_bwd_dq_gfx950_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        Bias: fx.Tensor,
        DO: fx.Tensor,
        DQ: fx.Tensor,
        DB: fx.Tensor,
        LSE: fx.Tensor,
        Delta: fx.Tensor,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        philox_seed_ptr: fx.Pointer,
        philox_offset1: fx.Pointer,
        philox_offset2: fx.Int64,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        sm_scale: fx.Float32,
        stride_q_batch: fx.Int64,
        stride_q_head: fx.Int64,
        stride_q_seq: fx.Int64,
        stride_k_batch: fx.Int64,
        stride_k_head: fx.Int64,
        stride_k_seq: fx.Int64,
        stride_v_batch: fx.Int64,
        stride_v_head: fx.Int64,
        stride_v_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stride_db_batch: fx.Int64,
        stride_db_head: fx.Int64,
        stride_db_seq_q: fx.Int64,
    ):
        ctx = BwdDqKernelContext(
            traits,
            # dQ occupies the `O` slot: it is the tensor this kernel writes
            # with `ParityStoreHelper`, and that helper reads `stride_o_*`.
            strides=(
                stride_q_batch,
                stride_q_head,
                stride_q_seq,
                stride_k_batch,
                stride_k_head,
                stride_k_seq,
                stride_v_batch,
                stride_v_head,
                stride_v_seq,
                stride_dq_batch,
                stride_dq_head,
                stride_dq_seq,
            ),
            do_strides=(stride_do_batch, stride_do_head, stride_do_seq),
            db_strides=(stride_db_batch, stride_db_head, stride_db_seq_q),
            sm_scale=sm_scale,
            num_head_q=num_head_q,
            num_head_k=num_head_k,
            hdim_qk=hdim_qk,
            hdim_vo=hdim_vo,
            padded_head=PADDED_HEAD,
            hdim_qk_floor=knobs.hdim_qk_floor,
            window_left=window_left,
            window_right=window_right,
            seqinfo=(seqinfo_q0, seqinfo_q1, seqinfo_k0, seqinfo_k1),
            varlen_bits=varlen_bits,
            num_seqlens=num_seqlens,
            Q=Q,
            K=K,
            V=V,
            O=DQ,
            DO=DO,
            DB=DB,
            Delta=Delta,
            DebugCounts=DQ,
            CuSeqQ=Q,
            CuSeqKv=Q,
            BlockTable=Q,
            Bias=Bias,
            bias_strides=(0, 0, 0),
            philox=(philox_seed_ptr, philox_offset1, philox_offset2, None, None),
            idropout_p=idropout_p,
            dropout_scale=dropout_scale,
            seq_len=max_seqlen_q,
            seq_len_kv=max_seqlen_k,
            stride_q_n=stride_q_seq,
            stride_kv_n=stride_k_seq,
            head_dim_runtime=hdim_qk,
            block_table_stride=0,
            LSE=LSE,
        )
        ctx.init_types_and_constants()
        ctx.init_runtime_indices()
        ctx.init_lds(SharedStorage)
        ctx.init_thread_mapping()
        ctx.init_sequence_lengths()
        ctx.init_descriptors()
        ctx.init_workspace()
        ctx.init_atoms_and_lds_ptrs()
        ctx.init_dma_thread_offsets()
        ctx.init_tile_bounds()
        ctx.init_active_guard()
        ctx.init_lds_read_bases()
        ctx.init_dma_m0_tables()
        ctx.init_q_row()
        ctx.init_row_inputs()

        kv_gmem_to_lds = BwdDqKvGmemToLdsLoader(ctx)
        # Two reader instances, and they differ in *both* the LDS pitch and the
        # padded-head extent. `k_reader` also serves GEMM3 through the stock
        # `load_v(0)`, since that reads the same region it does.
        k_reader = BwdDqKvLdsToVgprLoader(ctx, v_layout=True, hdim=hdim_qk, hdim_floor=HDIM_QK_FLOOR)
        v_reader = BwdDqKvLdsToVgprLoader(ctx, v_layout=False, hdim=hdim_vo, hdim_floor=traits.HDIM_VO_FLOOR)
        q_loader = ParityQLoader(ctx)
        do_loader = BwdSecondaryQLoader(ctx, ctx.do_div, ctx.stride_do_seq_v, ctx.do_gmem_elem_offset, hdim_vo)
        gemm_helper = ParityGemmHelper(ctx)
        softmax_helper = BwdDqSoftmaxHelper(ctx)
        output_store = BwdDqStoreHelper(ctx)
        row_inputs = BwdRowInputLoader(ctx)
        db_store = BwdDbStoreHelper(ctx)

        def _body():
            """One KV tile per iteration; three GEMMs and one accumulator."""
            # Neither operand is pre-scaled: `qk_scale` is applied to the f32
            # scores below, where it costs the same FMA the `lse2` subtract
            # needed anyway and does not round Q through bf16 a second time.
            # See `BwdDqSoftmaxHelper.scale_and_sub_lse`.
            q_all_bf16 = q_loader.load_all()
            do_all_bf16 = do_loader.load_all()
            lse2, delta = row_inputs.load(ctx.q_row)

            init_args = [ctx.c_zero_v16f32 for _ in range_constexpr(traits.D_CHUNKS)]
            loop_results = init_args
            for j, loop_args in range(ctx.split_tile(0), ctx.split_t_end, fx.Index(1), init=init_args):
                v_dq = _carried(loop_args, traits.D_CHUNKS)
                tile_start = ctx.tile_start(j)

                # Closes the previous iteration's readers before the DMA
                # overwrites what they were reading. One tile in flight, so
                # there is no second buffer to hide behind.
                _s_barrier()
                kv_gmem_to_lds.stage_k(tile_start)
                kv_gmem_to_lds.stage_v(tile_start)
                _s_waitcnt(0)
                _sched_barrier(0)
                _s_barrier()  # every wave's DMA has landed, not just mine

                # -- GEMM1. S, raw. The scale and the LSE subtract follow.
                v_s = gemm_helper.qk(k_reader.load_k(0), q_all_bf16)
                v_s = softmax_helper.scale_and_sub_lse(v_s, ctx.c_sm_scale_log2e, lse2)
                # Columns past `seqlen_kv` read zero from the buffer bound,
                # which is a *score of zero*, not an absent key: without the
                # mask `exp2(0 - lse2)` contributes a spurious P. After the
                # scale, so the `-inf` it writes never reaches an FMA.
                v_s = softmax_helper.seq_pad_mask_if_needed(v_s, j)

                # -- GEMM2. dP. V is read through the K path, so this is
                #    GEMM1's code with the two operands substituted -- and with
                #    the *other* reader instance, which owns the K-pitch region
                #    and masks against `hdim_vo`.
                v_dp = gemm_helper.qk(v_reader.load_k(0), do_all_bf16)

                # P = exp2(qk_scale*S - lse2). `exp2` is the forward's, split
                # into two halves there for the pipeline and simply adjacent
                # here.
                v_p = softmax_helper.exp2(v_s, 0, 16)
                v_p = softmax_helper.exp2(v_p, 16, 16)

                # dS = P * (dP - delta), elementwise over the 32 scores a lane
                # holds. `delta` is a per-row scalar and a lane owns one row.
                dp_lo, dp_hi = softmax_helper.v_s_vec_to_lists(v_dp)
                ds_lo = [None] * 16
                ds_hi = [None] * 16
                for r in range_constexpr(16):
                    ds_lo[r] = dualwave._fmul(v_p[0][r], dualwave._fsub(dp_lo[r], delta, ctx.fm_fast), ctx.fm_fast)
                    ds_hi[r] = dualwave._fmul(v_p[1][r], dualwave._fsub(dp_hi[r], delta, ctx.fm_fast), ctx.fm_fast)

                # -- GEMM3. dQ += dS . K, with K read transposed out of the V
                #    slot. `cast_p` gives dS the same K permutation P has, and
                #    the transpose read is built against that permutation, so
                #    the two line up with no further shuffle.
                v_ds = softmax_helper.cast_p((ds_lo, ds_hi))

                if const_expr(STORE_DB):
                    # **After `cast_p`, before `pv`, and reading the packs.**
                    # `dB = dS`, so nothing here is ordered by *arithmetic* --
                    # the store need only precede `sm_scale`, which is after
                    # the loop. It is ordered entirely by register pressure,
                    # and three placements were measured at `B=2 H=8 S=2048`
                    # (TFLOP/s, on the rungs that spill):
                    #
                    #     variant                        256    384    512
                    #     f32 lists, before cast_p       280    149     56
                    #     packs, after cast_p (this)     315    136     83
                    #     packs, after pv                305    146     73
                    #
                    # Reading the packs is the lever: `dS` exists as 32 f32
                    # *and*, after `cast_p`, as 16 packed bf16, and storing
                    # from the f32 form keeps both alive across the 32-store
                    # sequence. Below head_dim 256 it changes nothing -- the
                    # counts are register-identical, so the allocator was
                    # already sinking the f32 form -- and at 512 it is 1.49x.
                    #
                    # 384 prefers the f32 form by 9%, which the lore says a
                    # sweep cannot settle (it wants interleaved A/B), and it is
                    # an allocator outcome rather than a mechanism: the spill
                    # counts are not monotone with the rate in any of the three
                    # columns. This variant is kept because its one *decisive*
                    # measurement -- 1.49x at 512 -- agrees with the mechanism.
                    db_store.store_tile(v_ds, j, ctx.q_row)

                v_dq = gemm_helper.pv(v_ds, k_reader.load_v(0), v_dq)

                loop_results = yield v_dq

            v_dq = _carried(loop_results, traits.D_CHUNKS)
            # `sm_scale` once on the accumulator rather than on every dS: the
            # softmax input is `sm_scale * Q.K^T`, so the factor is linear in
            # the whole sum. AOTriton's `composed_mul_lhs(dq, sm_scale)`.
            softmax_helper.scale_o(v_dq, ctx.c_sm_scale)
            _s_barrier()
            output_store.store_final_o(v_dq, ctx.q_row)

        active = ctx.active
        if active is None:
            _body()
        else:

            @flyc.jit
            def _run_body_if_active():
                if active:
                    _body()

            _run_body_if_active()

    @flyc.jit
    def launch_fmha_bwd_dq_gfx950(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        Bias: fx.Tensor,
        DO: fx.Tensor,
        DQ: fx.Tensor,
        DB: fx.Tensor,
        LSE: fx.Tensor,
        Delta: fx.Tensor,
        batch_size: fx.Int32,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        philox_seed_ptr: fx.Pointer,
        philox_offset1: fx.Pointer,
        philox_offset2: fx.Int64,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        sm_scale: fx.Float32,
        stride_q_batch: fx.Int64,
        stride_q_head: fx.Int64,
        stride_q_seq: fx.Int64,
        stride_k_batch: fx.Int64,
        stride_k_head: fx.Int64,
        stride_k_seq: fx.Int64,
        stride_v_batch: fx.Int64,
        stride_v_head: fx.Int64,
        stride_v_seq: fx.Int64,
        stride_do_batch: fx.Int64,
        stride_do_head: fx.Int64,
        stride_do_seq: fx.Int64,
        stride_dq_batch: fx.Int64,
        stride_dq_head: fx.Int64,
        stride_dq_seq: fx.Int64,
        stride_db_batch: fx.Int64,
        stride_db_head: fx.Int64,
        stride_db_seq_q: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Make the build configuration visible to the JIT cache key.
        _ = _cache_tag
        bs_idx = fx.Index(batch_size)
        num_q_blocks = (fx.Index(max_seqlen_q) + traits.BLOCK_M - 1) // traits.BLOCK_M

        passthrough_entries = (
            [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
                ["unsafe-fp-math", "true"],
            ]
            if const_expr(traits.DAZ)
            else None
        )
        # Head-fastest, the forward's order. Not a free choice there -- it is
        # an L2-locality lever on MI355X's 8 XCDs -- and this kernel streams
        # the same K/V per (batch, head), so it inherits the reasoning.
        fmha_bwd_dq_gfx950_kernel(
            Q,
            K,
            V,
            Bias,
            DO,
            DQ,
            DB,
            LSE,
            Delta,
            seqinfo_q0,
            seqinfo_q1,
            seqinfo_k0,
            seqinfo_k1,
            varlen_bits,
            num_seqlens,
            max_seqlen_q,
            max_seqlen_k,
            window_left,
            window_right,
            philox_seed_ptr,
            philox_offset1,
            philox_offset2,
            idropout_p,
            dropout_scale,
            num_head_q,
            num_head_k,
            hdim_qk,
            hdim_vo,
            sm_scale,
            stride_q_batch,
            stride_q_head,
            stride_q_seq,
            stride_k_batch,
            stride_k_head,
            stride_k_seq,
            stride_v_batch,
            stride_v_head,
            stride_v_seq,
            stride_do_batch,
            stride_do_head,
            stride_do_seq,
            stride_dq_batch,
            stride_dq_head,
            stride_dq_seq,
            stride_db_batch,
            stride_db_head,
            stride_db_seq_q,
            value_attrs={
                "rocdl.waves_per_eu": traits.WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{traits.BLOCK_SIZE},{traits.BLOCK_SIZE}",
                "passthrough": passthrough_entries,
            },
        ).launch(
            grid=(num_head_q, num_q_blocks, bs_idx),
            block=(traits.BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    def _args(
        Q,
        K,
        V,
        DO,
        DQ,
        LSE,
        Delta,
        batch_size,
        seqlen_q,
        seqlen_k=None,
        scale=None,
        db=None,
        stream=None,
    ):
        """Every kernel argument but the stream, in launch order.

        One place that turns tensors into the wire format, so `_launch` and
        `_compile` cannot drift apart -- the same shape, and the same reason,
        as the forward's `_args`.
        """
        seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
        _ptrs, shape_meta, st = abi.prep_tensors(
            [("Q", Q), ("K", K), ("V", V), ("DO", DO), ("DQ", DQ)],
            q_heads=("DO", "DQ"),
        )
        del _ptrs  # gfx950 addresses through buffer descriptors, so it wants the tensors
        num_head_q, num_head_k, hdim_qk, hdim_vo = shape_meta

        # **A build without `padded_head` promises the tile *is* the extent**,
        # on both axes, and this is the only place that can be checked. Found
        # by a test of this file's own making: `build(head_dim=128)` handed a V
        # of width 64 resolves to `padded_head=False`, emits no mask at all,
        # and reduces `dP` over the caller's D-axis slack. The answer is finite
        # and 0.70 relative error. A real caller can make the same mistake, so
        # the guard belongs here rather than in the test that found it.
        if not PADDED_HEAD and not (hdim_qk == hdim_vo == BLOCK_DMODEL):
            raise ValueError(
                f"this build is not compiled for a padded head, so it requires hdim_qk == hdim_vo == "
                f"{BLOCK_DMODEL}; got hdim_qk {hdim_qk}, hdim_vo {hdim_vo}. Pass head_dim_v to the "
                "builder so the D-axis masks are emitted."
            )

        # **`(batch * heads, tokens)`, and the shape is shared with dK/dV.**
        # Both backward kernels take the same two row tensors and read them
        # with the same `fmha.lse_row_addressing`, so the host check is the
        # shared `abi.row_tensor_arg` rather than a second spelling of it.
        # This is the layout the forward writes LSE in -- `q_head_idx *
        # seq_len_v + q_row` inside a per-batch slab -- viewed at rank 2; a
        # caller holding `(B, H, S)` passes `lse.view(-1, S)`, which is free
        # on a contiguous tensor.
        for name, t in (("logsumexp", LSE), ("delta", Delta)):
            if t is None:
                raise ValueError(f"{name} is required: the backward reads it, it is not recomputed")
            abi.row_tensor_arg(t, name, num_head_q, seqlen_q, None)
            if t.shape[0] != int(batch_size) * num_head_q:
                raise ValueError(f"{name} must be ({int(batch_size) * num_head_q}, {seqlen_q}); got {tuple(t.shape)}")

        # A dB build must be handed a tensor and a build without dB must not
        # be: silently ignoring one returns gradients that are the right shape
        # with the bias gradient missing, and it is only ever passed by a
        # caller who believes it is being written.
        if STORE_DB and db is None:
            raise ValueError("this build has store_db=True and requires a (batch, num_heads_q, seqlen_q, seqlen_k) dB")
        if db is not None and not STORE_DB:
            raise ValueError("this build was not compiled for dB; pass store_db=True to the knobs")
        if db is not None:
            if tuple(db.shape) != (batch_size, num_head_q, seqlen_q, seqlen_k):
                raise ValueError(
                    f"dB must be (batch, num_heads_q, seqlen_q, seqlen_k) = "
                    f"{(batch_size, num_head_q, seqlen_q, seqlen_k)}, got {tuple(db.shape)}"
                )
            if db.stride(3) != 1:
                raise ValueError(f"dB needs a contiguous seqlen_k axis; strides are {tuple(db.stride())}")
            if db.dtype != Q.dtype:
                raise ValueError(f"dB must match Q's dtype ({Q.dtype}), got {db.dtype}")
        db_t = db if db is not None else DQ
        db_st = tuple(int(x) for x in db.stride()[:3]) if db is not None else (0, 0, 0)

        return (
            Q,
            K,
            V,
            DQ,  # Bias: no build here reads it; the slot is held for B6
            DO,
            DQ,
            db_t,
            LSE,
            Delta,
            int(batch_size),
            abi.NULL_PTR,
            abi.NULL_PTR,
            abi.NULL_PTR,
            abi.NULL_PTR,
            0,  # varlen_bits
            0,  # num_seqlens
            int(seqlen_q),
            int(seqlen_k),
            fmha.WINDOW_BOTRIGHT,
            fmha.WINDOW_BOTRIGHT,
            abi.NULL_PTR,
            abi.NULL_PTR,
            0,
            0,  # idropout_p
            1.0,  # dropout_scale
            num_head_q,
            num_head_k,
            hdim_qk,
            hdim_vo,
            abi.resolve_scale(
                Q, scale if scale is not None else BUILD_SM_SCALE, PADDED_HEAD, 1.0 / (BLOCK_DMODEL**0.5)
            ),
            *st,
            *db_st,
        ), stream

    def _launch(*args, **kwargs):
        packed, stream = _args(*args, **kwargs)
        with CompilationContext.compile_hints(_COMPILE_HINTS):
            return abi.run_compiled(
                _COMPILED,
                launch_fmha_bwd_dq_gfx950,
                *packed,
                stream if stream is not None else fx.Stream(None),
            )

    def _compile(*args, **kwargs):
        packed, stream = _args(*args, **kwargs)
        with CompilationContext.compile_hints(_COMPILE_HINTS):
            return flyc.compile(launch_fmha_bwd_dq_gfx950, *packed, fx.Stream(stream))

    _launch.compile = _compile
    _launch.traits = traits
    _launch.knobs = knobs
    return _launch


def build_fmha_bwd_dq_gfx950_module(arch="gfx950", **kwargs):
    """Keyword front end: name a problem, get the policy's schedule.

    `causal` defaults to **False** here where `FmhaInputMetadata` defaults it
    to True. B2 implements only the dense case and `_with_traits` refuses the
    other, so inheriting the forward's default would make every unqualified
    call raise -- and a caller who wants causal should be told it is B4, not
    told to pass an argument that is then rejected.
    """
    from dataclasses import fields as _fields

    meta_fields = {f.name for f in _fields(FmhaInputMetadata)}
    meta_kwargs = {"causal": False}
    meta_kwargs.update({k: v for k, v in kwargs.items() if k in meta_fields})
    meta = FmhaInputMetadata(**meta_kwargs)
    knob_kwargs = {k: v for k, v in kwargs.items() if k not in meta_fields}
    knobs = bwd_dq_knobs(arch, **knob_kwargs)
    if not isinstance(knobs, BwdDqKnobs):  # pragma: no cover -- defensive, the factory is typed
        raise TypeError(f"expected BwdDqKnobs, got {type(knobs).__name__}")
    return build_fmha_bwd_dq_gfx950_module_primary(meta, knobs.resolve(meta))
