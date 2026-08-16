#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Truncated kernel: stage one KV tile through LDS and dump what comes back.

The attention kernel is a long pipeline, and a wrong value anywhere in it comes
out as NaN at the end. This runs **only the first two steps** -- the global->LDS
DMA and the LDS->VGPR read -- and writes the registers straight to memory, so
the KV staging can be checked on its own.

What it pins, per lane `p` and k-step `ks`, is the contract the QK GEMM relies
on:

    k_lo[ks] element i  ==  K[token = lane % 32     ][D = ks*16 + (lane//32)*8 + i]
    k_hi[ks] element i  ==  K[token = lane % 32 + 32][D = same]

Two encodings, run separately because one probe cannot distinguish them:
`K[t][d] = d` checks the D mapping, `K[t][d] = t` checks the token mapping.
bf16 represents integers exactly to 256, so both are exact for the tile sizes
here.

    python3 probe_kv_staging.py [head_dim ...]
"""

import sys
from dataclasses import replace

import _bootstrap  # noqa: F401  (puts parity/ on sys.path)
import torch
from fmha_dualwave_gfx950 import (
    ParityKernelContext,
    ParityQLoader,
    ParitySoftmaxHelper,
    ParityStoreHelper,
    ParityKvGmemToLdsLoader,
    ParityKvLdsToVgprLoader,
)
from fmha_tuning_gfx950 import _GFX950_FALLBACK, FmhaInputMetadata, fmha_knobs

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value
from gfx950_standalone import dualwave


def build_probe(head_dim, num_heads=8, which="k", buf=0):
    """A kernel that stages KV tile 0 and dumps the register packs to `Out`."""
    meta = FmhaInputMetadata(num_heads=num_heads, head_dim=head_dim, causal=False, dtype_str="bf16")
    # Deliberately bypasses `_with_widths`, so the probe can reach the planned
    # rungs (192, 256) that `LADDER` keeps unreachable while they are broken.
    # That is the whole point of a probe: it must be able to build the thing
    # under investigation.
    pinned = replace(fmha_knobs("gfx950"), block_dmodel=head_dim, block_dmodel_v=head_dim, padded_head=False)
    knobs = _GFX950_FALLBACK.merge(pinned)._checked_modes()._with_wave_geometry()._with_traits(meta)
    traits = knobs.traits
    WHICH = which
    BUF = buf
    PER_LANE = {
        "k": traits.K_STEPS_QK * 2,
        "v": 4 * traits.D_CHUNKS,
        "q": traits.K_STEPS_QK,
        "s": 4,   # two v16f32 accumulators = 32 f32, dumped as 4 slots of 8
        "pv": traits.D_CHUNKS * 2,   # D_CHUNKS v16f32 accumulators
        "ml": 1,                     # [m_row, l_row] padded to one slot
        "store": 1,                  # writes O directly; Out is unused
    }[which] * traits.MFMA_LANE_K

    elem = dualwave.dtype_to_elem_type(traits.DTYPE_STR)

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

        loader = ParityKvGmemToLdsLoader(ctx)
        reader = ParityKvLdsToVgprLoader(ctx)

        # Step 1: stage KV tile 0 into LDS buffer 0, and wait for all of it.
        if const_expr(WHICH in ("k", "s")):
            loader.load_k(fx.Index(0), BUF)
        elif const_expr(WHICH == "v"):
            loader.load_v(fx.Index(0), BUF)
        elif const_expr(WHICH in ("pv", "ml")):
            loader.load_k(fx.Index(0), 0)
            loader.load_v(fx.Index(0), 1)
        dualwave._s_waitcnt(0)
        dualwave._sched_barrier(0)
        dualwave._s_barrier()

        # Step 2: read it back into registers, and dump them.
        rsrc = dualwave.buffer_ops.create_buffer_resource(Out, max_size=True)
        base = ctx.tid * fx.Index(PER_LANE)

        def dump(slot, vec_ir):
            vec = Vec(vec_ir)
            for i in const_expr(range(traits.MFMA_LANE_K)):
                dualwave.buffer_ops.buffer_store(
                    as_mlir_value(fx.Float32(vec[i].to(fx.Float32))),
                    rsrc,
                    as_mlir_value(fx.Int32(base + fx.Index(slot * traits.MFMA_LANE_K + i))),
                )

        if const_expr(WHICH == "k"):
            k_lo, k_hi = reader.load_k(BUF)
            for ks in const_expr(range(traits.K_STEPS_QK)):
                dump(ks * 2 + 0, k_lo[ks])
                dump(ks * 2 + 1, k_hi[ks])
        elif const_expr(WHICH == "v"):
            packs = reader.load_v(BUF)
            for step in const_expr(range(4)):
                for dc in const_expr(range(traits.D_CHUNKS)):
                    dump(step * traits.D_CHUNKS + dc, packs[step][dc])
        elif const_expr(WHICH == "q"):
            # Q never transits LDS: global -> VGPR directly.
            q_all = ParityQLoader(ctx).load_all()
            for ks in const_expr(range(traits.K_STEPS_QK)):
                dump(ks, dualwave._get_q_pack(traits, q_all, ks))
        else:
            # The QK GEMM itself, on operands the probes above have verified.
            q_all = ParityQLoader(ctx).load_all()
            q_scaled = ParityQLoader(ctx).scale_all(q_all)
            k_lo, k_hi = reader.load_k(0)
            s_lo, s_hi = dualwave.DualwaveGemmHelper(ctx).qk((k_lo, k_hi), q_scaled)
            for j in const_expr(range(2)):
                dump(0 + j, Vec(s_lo).shuffle(Vec(s_lo), [j * 8 + t for t in range(8)]).ir_value())
                dump(2 + j, Vec(s_hi).shuffle(Vec(s_hi), [j * 8 + t for t in range(8)]).ir_value())

        if const_expr(WHICH == "store"):
            # The O store on its own: v_o[dc] := dc + 1, so every output column
            # must come back as (D // D_CHUNK) + 1. Tests the store's (row, D)
            # mapping without anything upstream of it.
            ctx.init_q_row()
            v_o = [
                dualwave.Vec.filled(16, float(dc + 1), fx.Float32).ir_value()
                for dc in const_expr(range(traits.D_CHUNKS))
            ]
            ParityStoreHelper(ctx).store_final_o(v_o, ctx.q_row, ctx.c_zero_f, fx.Float32(1.0))

        if const_expr(WHICH == "ml"):
            # The softmax state carried across tiles, on verified operands.
            sm = ParitySoftmaxHelper(ctx)
            gemm = dualwave.DualwaveGemmHelper(ctx)
            ql = ParityQLoader(ctx)
            q_scaled = ql.scale_all(ql.load_all())
            v_s = sm.v_s_vec_to_lists(gemm.qk(reader.load_k(0), q_scaled))
            m = sm.reduce_max(v_s)
            v_s = sm.sub_m(v_s, m)
            v_p = sm.exp2(v_s, 0, 16)
            v_p = sm.exp2(v_p, 16, 16)
            lrow = sm.reduce_sum(ctx.c_zero_f, v_p)
            for slot_i, val in const_expr(((0, "m"), (1, "l"))):
                dualwave.buffer_ops.buffer_store(
                    as_mlir_value(fx.Float32(m if val == "m" else lrow)),
                    rsrc,
                    as_mlir_value(fx.Int32(base + fx.Index(slot_i))),
                )

        if const_expr(WHICH == "pv"):
            # Prologue softmax then one full P*V, on verified-correct operands.
            sm = ParitySoftmaxHelper(ctx)
            gemm = dualwave.DualwaveGemmHelper(ctx)
            ql = ParityQLoader(ctx)
            q_scaled = ql.scale_all(ql.load_all())
            v_s = sm.v_s_vec_to_lists(gemm.qk(reader.load_k(0), q_scaled))
            m = sm.reduce_max(v_s)
            v_s = sm.sub_m(v_s, m)
            v_p = sm.exp2(v_s, 0, 16)
            v_p = sm.exp2(v_p, 16, 16)
            v_p = sm.cast_p(v_p)
            v_o = [ctx.c_zero_v16f32 for _ in const_expr(range(traits.D_CHUNKS))]
            v_o = gemm.pv(v_p, reader.load_v(1), v_o)
            for dc in const_expr(range(traits.D_CHUNKS)):
                for j in const_expr(range(2)):
                    vv = Vec(v_o[dc])
                    dump(dc * 2 + j, vv.shuffle(vv, [j * 8 + t for t in range(8)]).ir_value())

    @flyc.jit
    def launch(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor, Out: fx.Tensor,  # noqa: E741
               seq_len: fx.Int32, stride_s: fx.Int64, stream: fx.Stream = fx.Stream(None)):
        probe_kernel(Q, K, V, O, Out, seq_len, stride_s).launch(
            grid=(1, 1, 1), block=(traits.BLOCK_SIZE, 1, 1), stream=stream
        )

    return launch, traits


def run(head_dim, num_heads=8):
    S, H, dev = 256, num_heads, "cuda"

    def measure(which, encode, expect, label, src="kv", buf=0):
        launch, traits = build_probe(head_dim, num_heads, which=which, buf=buf)
        # BHSD *shape* over BSHD *memory*, so the strides the kernel is handed
        # -- head=head_dim, seq=head_dim*H -- are the tensor's real ones.
        T = torch.zeros(1, S, H, head_dim, device=dev, dtype=torch.bfloat16).transpose(1, 2)
        rows = traits.BLOCK_M if src == "q" else traits.BLOCK_N
        for t in range(rows):
            T[0, 0, t, :] = encode(t, torch.arange(head_dim, device=dev, dtype=torch.float32))
        assert T.stride(1) == head_dim and T.stride(2) == head_dim * H
        Z = torch.zeros_like(T)
        per_lane = {"k": traits.K_STEPS_QK * 2, "v": 4 * traits.D_CHUNKS,
                    "q": traits.K_STEPS_QK}[which] * traits.MFMA_LANE_K
        Out = torch.full((traits.BLOCK_SIZE * per_lane,), -1.0, device=dev, dtype=torch.float32)
        if src == "q":
            launch(T, Z, Z, Z, Out, S, head_dim * H)
        else:
            launch(Z, T, T, Z, Out, S, head_dim * H)
        torch.cuda.synchronize()
        got = Out.view(traits.BLOCK_SIZE, -1, traits.MFMA_LANE_K).float()

        bad, total = [], 0
        for lane in range(traits.WARP_SIZE):
            for slot in range(got.shape[1]):
                for i in range(traits.MFMA_LANE_K):
                    tok, d = expect(traits, lane, slot, i)
                    if d >= head_dim or tok >= (traits.BLOCK_M if src == "q" else traits.BLOCK_N):
                        continue
                    total += 1
                    want = float(encode(tok, torch.tensor(float(d))))
                    g = got[lane, slot, i].item()
                    if abs(g - want) > 1e-3:
                        bad.append((lane, slot, i, tok, d, want, g))
        print(f"  {label:<34} {len(bad):>6} / {total} wrong")
        for b in bad[:5]:
            lane, slot, i, tok, d, want, g = b
            print(f"      lane{lane:>3} slot{slot:>3} i{i}  (tok {tok:>2}, D {d:>3})  want {want:>7.1f} got {g:>10.1f}")
        return len(bad)

    def k_expect(tr, lane, slot, i):
        ks, half = slot // 2, slot % 2
        return lane % 32 + half * 32, ks * 16 + (lane // 32) * 8 + i

    def q_expect(tr, lane, slot, i):
        # wave 0 only: q_row = wave*ROWS_PER_WAVE + lane%32.
        return lane % 32, slot * 16 + (lane // 32) * 8 + i

    def v_expect(tr, lane, slot, i):
        """The V pack layout, derived from the probe rather than assumed.

        A pack is two `ds_read_b64_tr_b16` results concatenated, and the second
        is read `TRANSPOSE_PAIR_STRIDE` (one granule) away -- so a lane's eight
        elements are **two groups of four, eight tokens apart**, not eight
        consecutive tokens. The half-wave contributes 4 tokens, not 8:

            lane 0  -> tokens {0,1,2,3, 8,9,10,11}
            lane 32 -> tokens {4,5,6,7, 12,13,14,15}

        which together tile the step's 16 KV tokens exactly once.
        """
        step, dc = slot // tr.D_CHUNKS, slot % tr.D_CHUNKS
        tok = step * 16 + (lane // 32) * 4 + (i % 4) + (i // 4) * 8
        return tok, dc * 32 + (lane % 32)

    launch, traits = build_probe(head_dim, num_heads)
    print(f"head_dim={head_dim}  geom=(waves {traits.NUM_WAVES}, BM {traits.BLOCK_M}, BN {traits.BLOCK_N}, "
          f"gran {traits.D_128B_SIZE})  d_rpt={traits.SMEM_D_RPT} K_STEPS={traits.K_STEPS_QK} "
          f"D_CHUNKS={traits.D_CHUNKS}")
    n = 0
    n += measure("k", lambda t, d: d, k_expect, "K stage: K[t][d]=d  (D map)")
    n += measure("k", lambda t, d: t * torch.ones_like(d), k_expect, "K stage: K[t][d]=t  (token map)")
    n += measure("v", lambda t, d: d, v_expect, "V stage: V[t][d]=d  (D map)")
    n += measure("v", lambda t, d: t * torch.ones_like(d), v_expect, "V stage: V[t][d]=t  (token map)")
    n += measure("q", lambda t, d: d, q_expect, "Q load : Q[r][d]=d  (D map)", src="q")
    n += measure("q", lambda t, d: t * torch.ones_like(d), q_expect, "Q load : Q[r][d]=r  (row map)", src="q")
    # Buffer 1. Every probe above stages into buffer 0, but the pipeline
    # alternates -- an error in the buffer-1 base would be invisible until the
    # whole kernel runs.
    n += measure("k", lambda t, d: d, k_expect, "K stage buf1: K[t][d]=d", buf=1)
    n += measure("v", lambda t, d: d, v_expect, "V stage buf1: V[t][d]=d", buf=1)
    n += measure("v", lambda t, d: t * torch.ones_like(d), v_expect, "V stage buf1: V[t][d]=t", buf=1)
    return n


if __name__ == "__main__":
    dims = [int(a) for a in sys.argv[1:]] or [64, 128, 192]
    for hd in dims:
        run(hd)
        print()
