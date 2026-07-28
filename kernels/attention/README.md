# `kernels/attention`

Attention kernels: dense flash-attention, paged-attention (PA) decode, MLA
decode, and the fused pre-attention ops (RoPE, QK-norm).

This file is a **router**, not a catalog. The authoritative user-facing
reference for public APIs, supported dtypes, and configuration is
[`docs/prebuilt_kernels_guide.md`](../../docs/prebuilt_kernels_guide.md); tuning
technique and vocabulary live in
[`docs/kernel_tuning_guide.md`](../../docs/kernel_tuning_guide.md). Prefer
searching the tree over trusting the summaries below.

## Dense flash-attention

Pick by architecture first — the builders are not interchangeable.

| module | scope |
|---|---|
| `flash_attn_interface.py` | High-level API for **gfx950 / gfx942**. Routes to the builders below. |
| `flash_attn_generic.py` | Generic f16/bf16 builder (gfx942-compatible, dense self/cross-attention). |
| `flash_attn_gfx950.py` | **gfx950** dual-wave software-pipelined kernel, D=64/128 bf16/f16. Requires `ds_read_tr16_b64`. |
| `flash_attn_fp8_gfx950.py` | **gfx950** DUALWAVE_SWP FP8 variant. |
| `flash_attn_utils.py` | Shared helpers for the gfx950 dual-wave family (DMA, barriers, traits). |
| `flash_attn_func_gfx1201_interface.py` | High-level API for **gfx1201 / RDNA4**. Deliberately separate from `flash_attn_interface.py`. |
| `flash_attn_func_gfx1201.py` | **gfx1201** kernel builder (WMMA, wave32). |

## Paged attention (decode)

| module | scope |
|---|---|
| `pa_decode_fp8.py` | FP8 PA decode with persistent scheduling. Start here for paged-decode changes. |
| `pa_decode_swa.py` | Sliding-window mode, imported by `pa_decode_fp8.py` (not a separate public entry point). |
| `pa_decode_tile.py` | Readable tile-programming reference for FP8 PA decode. |
| `pa_metadata.py` | Worklist scheduler (FlyDSL port of aiter's `get_pa_metadata_v1`). |
| `pa_common.py` | Shared PA helpers. |

Regression tests: [`tests/kernels/test_pa.py`](../../tests/kernels/test_pa.py);
reference semantics in `reference_masked_attention()` / `torch_mha_extend()`.

## MLA decode

| module | scope |
|---|---|
| `mla_fwd_decode.py` | MLA decode launcher. |
| `mla_fwd_decode_m16x8_fp8_fp8.py` | MLA decode kernel, nhead=128, fp8 Q / fp8 KV, bf16 out. |

## Fused pre-attention ops

| module | scope |
|---|---|
| `fused_rope_cache_kernel.py` | Fused RoPE + KV-cache write. |
| `qk_norm_rope_quant.py` | Fused per-token RMSNorm + GPT-J RoPE + optional FP8 quant. |

## gfx1201 FMHA prototype

`bench_fmha.py`, `bench_shim.py`, `flash_attn_func_gfx1201.py`, and
`flash_attn_func_gfx1201_interface.py` form a **self-contained prototype** with
conventions that differ from the rest of this directory — they import each other
by bare module name and assume the cwd is `kernels/attention`.

Read [`gfx1201_fmha.md`](gfx1201_fmha.md) before editing them. It covers how to
run the benchmark, why those four files duplicate helpers from `kernels/common`,
the gfx1201 hardware capabilities that constrain the design, and the pipelining
terminology used when discussing this work.
