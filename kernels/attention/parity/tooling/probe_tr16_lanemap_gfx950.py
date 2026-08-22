#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""B0/B3.5: what lane map does `ds_read_b64_tr_b16` actually produce?

B3.5 wants a 16-rows-per-wave MFMA family, and the question that gates the
whole layout is whether the LDS transpose read can serve a **16-row A
operand** at all. The addendum's warning is concrete: AITER's kernels that use
`v_mfma_f32_16x16x16` *exclusively* issue **zero** transpose reads, and the
ones with hundreds of them use the wide-K shapes. If the correlation is causal,
the 16-row family needs both orientations staged and the four-LDS-tile problem
comes back at exactly the widths where LDS is tightest.

This answers it by measurement rather than by reading a disassembly's
correlations. Four probes, in increasing order of what they assume:

1. **`perm`** -- the raw instruction. Every lane is given a *distinct* address
   and the LDS holds a bit pattern that names its own `(token, d)`, so the dump
   says directly which source lane and which of its four elements each output
   element came from. Assumes nothing about operands.
2. **`m32`** -- the **control**. The forward's own V address pattern and pair
   read, checked against the map the forward relies on. It must be 0 wrong; if
   it is not, the harness is broken and nothing else here means anything. (The
   lore: three wrong conclusions in this work came from a probe with no
   control.)
3. **`m16`** -- the candidate. A 16-row address pattern, one read, checked
   against the `v_mfma_f32_16x16x16` A-operand layout `m = lane % 16`,
   `k = 4 * (lane // 16) + i`.
4. **`mfma16` / `mfma16x32`** -- end to end. A real MFMA whose A operand comes
   out of the transpose read, against a host reference. This is the one that
   cannot be fooled by a lane map that is self-consistent and wrong.

    python3 probe_tr16_lanemap_gfx950.py [head_dim ...]
"""

import sys
from dataclasses import replace

import _bootstrap  # noqa: F401  (puts parity/ on sys.path)
import torch
from fmha_dualwave_gfx950 import ParityKernelContext, ParityKvGmemToLdsLoader, _v_imm_lo
from fmha_mfma16_gfx950 import a16_chunk_offset, a16_read_base, lds_elem, tok_off
from fmha_tuning_gfx950 import _GFX950_FALLBACK, FmhaInputMetadata, fmha_knobs
from gfx950_standalone import dualwave

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

# How many 16-bit words one `ds_read_b64_tr_b16` returns per lane.
TR_ELEMS = 4


def build_probe(head_dim, which, num_heads=8, geom=None):
    """A kernel that stages a V tile and dumps transpose reads of it."""
    meta = FmhaInputMetadata(num_heads=num_heads, head_dim=head_dim, causal=False, dtype_str="bf16")
    pinned = replace(fmha_knobs("gfx950"), block_dmodel=head_dim, block_dmodel_v=head_dim, padded_head=False)
    if geom is not None:
        nw, bm, bn, gran = geom
        pinned = replace(pinned, num_waves=nw, block_m=bm, block_n=bn, head_dim_granule=gran)
    knobs = _GFX950_FALLBACK.merge(pinned)._checked_modes()._with_wave_geometry()._with_traits(meta)
    traits = knobs.traits
    WHICH = which
    elem = dualwave.dtype_to_elem_type(traits.DTYPE_STR)
    scope = dualwave._dualwave_lds_scope("v", 0)

    # Reads per lane, and i32 words dumped per read. The MFMA arms dump f32
    # accumulators instead, 4 per lane.
    N_READS = {"perm": 1, "m32": 8, "m16": 4, "mfma16": 1, "mfma16x32": 1}[WHICH]
    WORDS = 4 if WHICH.startswith("mfma") else 2
    PER_LANE = N_READS * WORDS

    @fx.struct
    class SharedStorage:
        kv: fx.Array[elem, traits.LDS_KV_TOTAL_SIZE, 16]

    @flyc.kernel(known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def probe_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        Out: fx.Tensor,
        Bop: fx.Tensor,
        seq_len: fx.Int32,
        stride_s: fx.Int64,
    ):
        ctx = ParityKernelContext(
            traits,
            strides=(0, fx.Int64(head_dim), stride_s) * 4,
            sm_scale=fx.Float32(1.0),
            num_head_q=fx.Int32(num_heads),
            num_head_k=fx.Int32(num_heads),
            hdim_qk=fx.Int32(head_dim),
            hdim_vo=fx.Int32(head_dim),
            padded_head=False,
            Q=Q,
            K=K,
            V=V,
            O=O,
            DebugCounts=O,
            CuSeqQ=Q,
            CuSeqKv=Q,
            BlockTable=Q,
            seq_len=seq_len,
            seq_len_kv=seq_len,
            stride_q_n=stride_s,
            stride_kv_n=stride_s,
            head_dim_runtime=fx.Int32(head_dim),
            block_table_stride=fx.Int32(0),
            LSE=O,
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

        ParityKvGmemToLdsLoader(ctx).load_v(fx.Index(0), 0)
        dualwave._s_waitcnt(0)
        dualwave._sched_barrier(0)
        dualwave._s_barrier()

        v_base = dualwave._v_buf_base(traits, 0)
        lane = ctx.lane
        rsrc = dualwave.buffer_ops.create_buffer_resource(Out, max_size=True)
        out_base = ctx.tid * fx.Index(PER_LANE)

        def tr_read(elem_idx, imm_bytes=0):
            """One `ds_read_b64_tr_b16` at an element index of the V buffer."""
            return dualwave._ds_read_tr16_b64_imm(
                ctx.v_lds_read_vec4_type,
                fx.Int32((v_base + elem_idx) * traits.BF16_BYTES + ctx.lds_kv_base_idx),
                imm_bytes,
                scope_name=scope,
                scope_names=traits.LDS_SCOPE_NAMES,
            )

        def dump_words(slot, words):
            for w in const_expr(range(len(words))):
                dualwave.buffer_ops.buffer_store(
                    as_mlir_value(fx.Int32(words[w])),
                    rsrc,
                    as_mlir_value(fx.Int32(out_base + fx.Index(slot * WORDS + w))),
                )

        def dump_raw(slot, v4):
            pair = Vec(v4).bitcast(fx.Int32)
            dump_words(slot, [pair[0], pair[1]])

        if const_expr(WHICH == "perm"):
            # Every lane a distinct address, four contiguous elements each, so
            # the dump names its own source. Lanes 0..63 cover elements 0..255,
            # all inside line 0 of the tile and therefore all written.
            dump_raw(0, tr_read(lane * TR_ELEMS))

        elif const_expr(WHICH == "m32"):
            # The control: the forward's own V read, verbatim.
            base = ctx.v_lds_read_base_per_lane
            pair = traits.V_LDS_TO_REG_TRANSPOSE_PAIR_STRIDE * traits.BF16_BYTES
            for k_substep in const_expr(range(4)):
                imm = _v_imm_lo(traits, 0, k_substep)
                dump_raw(k_substep * 2 + 0, tr_read(base, imm))
                dump_raw(k_substep * 2 + 1, tr_read(base, imm + pair))

        elif const_expr(WHICH == "m16"):
            # The candidate. `lane // 16` moves the **token** base by 4 where
            # the forward's map moves the *d* base by 16, and there is no pair
            # read: one instruction is a whole 16x16x16 A operand.
            for c in const_expr(range(4)):
                dump_raw(c, tr_read(a16_read_base(traits, lane, 4) + a16_chunk_offset(traits, c)))

        elif const_expr(WHICH in ("mfma16", "mfma16x32")):
            # End to end: A out of the transpose read, B from global, one MFMA
            # against a host reference. A lane map can be self-consistent and
            # still be the wrong operand; this is what rules that out.
            k_ext = 16 if const_expr(WHICH == "mfma16") else 32
            quad = k_ext // 4
            mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, k_ext, ctx.elem_dtype))
            base16 = a16_read_base(traits, lane, quad)
            a = tr_read(base16)
            if const_expr(k_ext == 32):
                # Group g needs k = 8g..8g+7, so the base steps by 8 tokens and
                # the second read picks up the four after the first four.
                a2 = tr_read(base16, tok_off(traits, 4) * traits.BF16_BYTES)
                a = Vec(a).shuffle(Vec(a2), [0, 1, 2, 3, 4, 5, 6, 7]).ir_value()
            # B[k][n] with n = lane % 16 and `quad` consecutive k, which the
            # host supplies as a contiguous `Bt[n][k]`.
            b_rsrc = dualwave.buffer_ops.create_buffer_resource(Bop, max_size=True)
            b = dualwave.buffer_ops.buffer_load(
                b_rsrc,
                as_mlir_value(
                    fx.Int32((lane % fx.Index(16)) * fx.Index(k_ext) + (lane // fx.Index(16)) * fx.Index(quad))
                ),
                vec_width=quad,
                dtype=ctx.elem_dtype,
            )
            d = dualwave.fly.mma_atom_call_ssa(
                [Vec.make_type(4, fx.Float32)], mma, a, b, Vec.filled(4, 0.0, fx.Float32).ir_value()
            )
            dv = Vec(d)
            dump_words(0, [fx.Float32(dv[i]).bitcast(fx.Int32) for i in const_expr(range(4))])

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        Out: fx.Tensor,
        Bop: fx.Tensor,
        seq_len: fx.Int32,
        stride_s: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        probe_kernel(Q, K, V, O, Out, Bop, seq_len, stride_s).launch(
            grid=(1, 1, 1), block=(traits.BLOCK_SIZE, 1, 1), stream=stream
        )

    return launch, traits, PER_LANE


def _staged(traits, head_dim, num_heads, encode, dtype):
    """A BHSD-shaped, BSHD-laid-out V tensor filled by `encode(t, d)`."""
    S, H, dev = 256, num_heads, "cuda"
    T = torch.zeros(1, S, H, head_dim, device=dev, dtype=dtype).transpose(1, 2)
    tt = torch.arange(traits.BLOCK_N, device=dev)
    dd = torch.arange(head_dim, device=dev)
    T[0, 0, : traits.BLOCK_N, :] = encode(tt[:, None], dd[None, :])
    return T


def _raw_dump(head_dim, which, num_heads, encode, dtype=torch.int16, Bop=None):
    """Run one probe arm and return `(traits, got, Bop)` with `got` as int16."""
    launch, traits, per_lane = build_probe(head_dim, which, num_heads)
    src = _staged(traits, head_dim, num_heads, encode, dtype)
    Z = torch.zeros(*src.shape, device="cuda", dtype=torch.bfloat16)
    Out = torch.zeros(traits.BLOCK_SIZE * per_lane, device="cuda", dtype=torch.int32)
    if Bop is None:
        Bop = torch.zeros(1024, device="cuda", dtype=torch.bfloat16)
    v = src if dtype is torch.bfloat16 else src.view(torch.bfloat16)
    launch(Z, Z, v, Z, Out, Bop, 256, head_dim * num_heads)
    torch.cuda.synchronize()
    return traits, Out.view(traits.BLOCK_SIZE, per_lane), per_lane


def _check_td(traits, got_i16, expect, label, n_slots):
    """Compare a `(token, d)`-encoded dump against `expect(lane, slot, i)`."""
    bad = 0
    for j in range(64):
        for slot in range(n_slots):
            for i in range(TR_ELEMS):
                v = int(got_i16[j, slot * TR_ELEMS + i]) & 0xFFFF
                got = (v >> 9, v & 0x1FF)
                want = expect(j, slot, i)
                if got != want:
                    bad += 1
                    if bad <= 5:
                        print(
                            f"      lane{j:>3} slot{slot} i{i}: got (t {got[0]}, d {got[1]}), want (t {want[0]}, d {want[1]})"
                        )
    print(f"  {label:<48} {bad:>5} / {64 * n_slots * TR_ELEMS} wrong")
    return bad


def run(head_dim=64, num_heads=8):
    fails = 0
    enc = lambda t, d: t * 512 + d  # noqa: E731  (15 bits, unique over the tile)

    traits, out, _ = _raw_dump(head_dim, "perm", num_heads, enc)
    got = out.view(torch.int16).cpu()
    print(
        f"head_dim={head_dim} granule={traits.D_128B_SIZE} N_RPT={traits.SMEM_N_RPT} LINE={traits.SMEM_V_LINE_STRIDE}"
    )

    # --- 1. the raw permutation, assuming nothing about operands ---------
    bad, cross = 0, 0
    for j in range(64):
        for i in range(TR_ELEMS):
            v = int(got[j, i]) & 0xFFFF
            e = lds_elem(traits, v >> 9, v & 0x1FF)
            src = (e // TR_ELEMS, e % TR_ELEMS)
            want = (16 * (j // 16) + 4 * i + ((j % 16) // 4), j % 4)
            if src != want:
                bad += 1
                if bad <= 6:
                    print(f"      lane{j:>3} elem{i}: from (lane {src[0]}, elem {src[1]}), model says {want}")
            if src[0] // 16 != j // 16:
                cross += 1
    print(f"  {'1. raw permutation vs the model':<48} {bad:>5} / 256 wrong")
    print(f"  {'   output elements crossing a 16-lane group':<48} {cross:>5} / 256 (expect 0)")
    fails += bad + cross
    if not bad:
        print("       O[j][i] = M[16*(j//16) + 4*i + ((j%16)//4)][j%4]")
        print("       every lane's own address is honoured; groups of 16 are independent")

    # --- 2. the control: the forward's own 32-row read -------------------
    traits, out, _ = _raw_dump(head_dim, "m32", num_heads, enc)
    got = out.view(torch.int16).cpu()
    fails += _check_td(
        traits,
        got,
        lambda j, slot, i: ((slot // 2) * 16 + 4 * (j // 32) + (slot % 2) * 8 + i, j % 32),
        "2. CONTROL: 32-row map (the forward's V read)",
        8,
    )

    # --- 3. the candidate: a 16-row A operand ----------------------------
    traits, out, _ = _raw_dump(head_dim, "m16", num_heads, enc)
    got = out.view(torch.int16).cpu()
    fails += _check_td(
        traits,
        got,
        lambda j, slot, i: (4 * (j // 16) + i, slot * 16 + (j % 16)),
        "3. 16-row map: m = lane%16, k = 4*(lane//16)+i",
        4,
    )

    # --- 4/5. end to end, a real MFMA on the transposed operand ----------
    for which, k_ext in (("mfma16", 16), ("mfma16x32", 32)):
        torch.manual_seed(k_ext)
        # V[t][d] with t = k (contraction) and d = m (the A row).
        vt = torch.randn(traits.BLOCK_N, head_dim, device="cuda", dtype=torch.bfloat16)
        bt = torch.randn(16, k_ext, device="cuda", dtype=torch.bfloat16)  # Bt[n][k]
        traits, out, _ = _raw_dump(
            head_dim, which, num_heads, lambda t, d: vt[t, d], dtype=torch.bfloat16, Bop=bt.reshape(-1).contiguous()
        )
        d_got = out.view(torch.float32)[:64].cpu().float()
        # A[m][k] = V[k][m]; D[m][n] = sum_k A[m][k] * Bt[n][k]
        ref = (vt[:k_ext, :16].float().T @ bt.float().T).cpu()  # (m, n)
        bad = 0
        for j in range(64):
            for i in range(4):
                m, n = 4 * (j // 16) + i, j % 16
                g, w = d_got[j, i].item(), ref[m, n].item()
                if abs(g - w) > 2e-2 * max(1.0, abs(w)):
                    bad += 1
                    if bad <= 5:
                        print(f"      lane{j:>3} i{i} (m {m}, n {n}): got {g:.4f} want {w:.4f}")
        print(f"  {'4. v_mfma_f32_16x16x%d end to end' % k_ext:<48} {bad:>5} / 256 wrong")
        fails += bad
    return fails


if __name__ == "__main__":
    dims = [int(a) for a in sys.argv[1:]] or [64]
    bad = 0
    for hd in dims:
        bad += run(hd)
        print()
    sys.exit(1 if bad else 0)
