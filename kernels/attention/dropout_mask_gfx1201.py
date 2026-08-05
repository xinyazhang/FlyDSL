# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""The dropout mask as a tensor, mirroring AOTriton's `debug_fill_dropout_rng`.

Fills a `(B, H, Sq, Sk)` tensor with the same randoms the attention kernel
drops with. Its value is not debugging output -- it is that **the
reproducibility contract becomes testable without the attention kernel in the
way** (`sdpa-dropout-plan.md` §7).

Two things follow from that, and they are the reason this file is separate
rather than a mode of the attention kernel:

- **It shares the offset scheme rather than reimplementing it.** Both kernels
  call `Philox.grid_plane` / `grid_offset`. A second transcription of the same
  formula would make "the two agree" a statement about my typing rather than
  about the contract, and the interesting failure -- an incremental offset that
  wraps differently -- is exactly the kind that survives a careful read.
- **It has no tiling in common with the attention kernel.** `BLOCK_M` and
  `BLOCK_N` here are free parameters, so running it at several tilings and
  diffing is a direct test of §3's claim that the mask is a function of
  absolute `(batch, head, row, column)` and nothing else.

Both of AOTriton's encodings are emitted: raw `int32`, which is what the
threshold compare actually sees, and `float32` in `[0, 1)` for reading.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from philox import Philox, to_uniform_f32

THREADS = 256

_ENCODINGS = ("raw", "uniform")


def build_dropout_mask_module(
    block_m: int = 64,
    block_n: int = 32,
    philox_width: int | None = None,
    encoding: str = "raw",
):
    """A launcher filling `(B, H, Max_seqlen_q, Max_seqlen_k)` with the mask.

    `block_m` / `block_n` are the workgroup's tile and carry no meaning beyond
    scheduling -- see the module docstring. `encoding` is `"raw"` (int32, the
    value the threshold compares) or `"uniform"` (float32 in `[0, 1)`).
    """
    if encoding not in _ENCODINGS:
        raise ValueError(f"encoding must be one of {_ENCODINGS}, got {encoding!r}")
    PHILOX = Philox.for_arch() if philox_width is None else Philox(width=philox_width)
    RN = PHILOX.randoms_per_offset

    if block_n % RN:
        raise ValueError(f"block_n ({block_n}) must be a multiple of {RN}")
    COLGRPS = block_n // RN
    SLOTS = block_m * COLGRPS
    if SLOTS % THREADS:
        raise ValueError(
            f"block_m * block_n / {RN} ({SLOTS}) must be a multiple of {THREADS}"
        )
    ITERS = SLOTS // THREADS
    IS_RAW = encoding == "raw"

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def mask_kernel(
        R: fx.Pointer,
        stride_rz: fx.Int64,
        stride_rh: fx.Int64,
        stride_rm: fx.Int64,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        philox_seed: fx.Int64,
        philox_offset_base: fx.Int64,
        num_head_q: fx.Int32,
    ):
        # `ir_type` needs the MLIR context, so it is read here rather than at
        # build time.
        rptr = fx.PointerType.get(
            elem_ty=fx.Int32.ir_type if const_expr(IS_RAW) else fx.Float32.ir_type,
            address_space=fx.AddressSpace.Global,
            alignment=4,
        )
        out = fx.recast_iter(rptr, R)

        m_tile = fx.Int32(fx.Index(gpu.block_idx.x))
        n_tile = fx.Int32(fx.Index(gpu.block_idx.y))
        # A plane is one (batch, head) pair, and `blockIdx.z` *is* the plane
        # index `z * num_head_q + h` -- the same quantity the attention kernel
        # calls `off_zh`. It is decomposed only to address the output.
        plane = fx.Int32(fx.Index(gpu.block_idx.z))
        off_z = plane // num_head_q
        off_h = plane - off_z * num_head_q
        tid = fx.Int32(fx.Index(gpu.thread_idx.x))

        base, row_stride = PHILOX.grid_plane(
            philox_offset_base, plane, max_seqlen_q, max_seqlen_k
        )
        plane_addr = fx.Int64(off_z) * stride_rz + fx.Int64(off_h) * stride_rh

        for it in range_constexpr(ITERS):
            # One slot is one Philox call: RN adjacent columns of one row.
            slot = tid + fx.Int32(it * THREADS)
            row_in = slot // fx.Int32(COLGRPS)
            colgrp = slot - row_in * fx.Int32(COLGRPS)
            row = m_tile * fx.Int32(block_m) + row_in
            col = n_tile * fx.Int32(block_n) + colgrp * fx.Int32(RN)

            vals = PHILOX.u32(
                philox_seed, PHILOX.grid_offset(base, row_stride, row, col)
            )
            row_addr = plane_addr + fx.Int64(row) * stride_rm

            if row < max_seqlen_q:
                for j in range_constexpr(RN):
                    if col + fx.Int32(j) < max_seqlen_k:
                        v = vals[j] if const_expr(IS_RAW) else to_uniform_f32(vals[j])
                        fx.ptr_store(v, out + row_addr + fx.Int64(col + fx.Int32(j)))

    @flyc.jit
    def launch_dropout_mask(
        R: fx.Pointer,
        stride_rz: fx.Int64,
        stride_rh: fx.Int64,
        stride_rm: fx.Int64,
        batch_size: fx.Int32,
        num_head_q: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        philox_seed: fx.Int64,
        philox_offset_base: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        m_tiles = (fx.Index(max_seqlen_q) + (block_m - 1)) // block_m
        n_tiles = (fx.Index(max_seqlen_k) + (block_n - 1)) // block_n
        planes = fx.Index(batch_size) * fx.Index(num_head_q)
        mask_kernel(
            R, stride_rz, stride_rh, stride_rm,
            max_seqlen_q, max_seqlen_k,
            philox_seed, philox_offset_base, num_head_q,
        ).launch(
            grid=(m_tiles, n_tiles, planes),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    return launch_dropout_mask


def dropout_mask(
    batch: int,
    num_heads: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    philox_seed: int,
    philox_offset: int = 0,
    *,
    block_m: int = 64,
    block_n: int = 32,
    philox_width: int | None = None,
    encoding: str = "raw",
    device: str = "cuda",
):
    """Host wrapper: the `(B, H, Sq, Sk)` mask tensor for one attention call.

    `(max_seqlen_q, max_seqlen_k)` must be the same pair the attention kernel
    was given, since they set the offset stride and so the whole stream --
    under varlen those are the maxima, not any one sequence's lengths.
    """
    import torch

    dtype = torch.int32 if encoding == "raw" else torch.float32
    r = torch.zeros(
        batch, num_heads, max_seqlen_q, max_seqlen_k, dtype=dtype, device=device
    )
    exe = build_dropout_mask_module(
        block_m=block_m, block_n=block_n,
        philox_width=philox_width, encoding=encoding,
    )
    exe(
        flyc.from_c_void_p(fx.Uint8, r.data_ptr()),
        r.stride(0), r.stride(1), r.stride(2),
        batch, num_heads, max_seqlen_q, max_seqlen_k,
        int(philox_seed), int(philox_offset),
        fx.Stream(None),
    )
    torch.cuda.synchronize()
    return r
