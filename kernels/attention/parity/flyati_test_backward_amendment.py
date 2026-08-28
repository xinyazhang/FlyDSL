#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`test_fast`, amended to reach two bug classes it could not previously see.

**Destination: `flyati/modules/flash/tests/test_backward_amendment.py`.** It
lives here because that checkout is mounted read-only, and under the
`flyati_` prefix because `test_*.py` in this directory is collected by this
repo's own suite, where `_core_test_backward` does not exist. Rename on copy.

A `FOR_RELEASE=3` run of `test_backward.py` produced 7506 failures while
`test_fast` reported **zero**. This is `test_fast`'s grid with two axes changed
and everything else cut to the smallest set that still reproduces, so it stays
a fast test rather than becoming a second level 3.

Both gaps were read off the failure log, not guessed.

**1. Every `D_HEAD` tuple `test_fast` carries happens to pass.** It has
`(24, 152)` and `(120, 8)`; both PASSED all 768 of their runs, while
`test_hdim_qk_ne_vo` failed 6290 times. The failing pairs are the ones whose
**QK side lands exactly on a compiled tile** -- `(64, 32)`, `(128, 64)`,
`(32, 16)` -- and `test_fast`'s two do not: 24 and 120 both round *up* to a
tile, which turns padding on for an unrelated reason and masks the V/O axis by
accident. `(64, 32)` is the smallest pair with the property that matters.

**2. `test_fast` pins `bias_type = None`.** 1614 of the failures need a bias
and none was reachable.

The seqlens are *not* a gap and are left alone: `test_fast` already runs
`seqlen_q=11, seqlen_k=31` and 141 failures sit on exactly that shape. The one
addition is `seqlen_q=8`, for the family whose failures cluster at a query
extent far below `BLOCK_M` (q=8 against k=64/128/512) and which no `test_fast`
seqlen reaches.

Everything else is trimmed to two values per axis, each kept only because the
log shows failures at both. `storage_flip` is kept as an axis rather than
pinned to `test_fast`'s `True`: the `hdim_qk != hdim_vo` failures were only
ever *exercised* at `False`, that being the sole value its own test uses, so
pinning `True` would leave the direction untested.

    192 cases per backend, against `test_fast`'s 1296 and level 3's ~250k.

Run it as `test_backward.py` is run -- same `BWDOP` matrix, same `gpu_id` lease.
"""

import pytest
from _core_test_backward import (
    BWD_IMPL,
    BWDOP_ids,
    DTYPES,
    core_test_op_bwd,
    fmt_hdim,
    fmt_nheads,
)

# `test_fast` samples two; one is enough here, so that a failure is
# attributable to the parameter that changed rather than to the GQA fold.
_N_HEADS = [5]

# An int control plus the pair shape `test_fast` misses. 64 is on the compiled
# ladder, so `(64, 32)` is "QK exactly on tile, V/O narrower" -- the case where
# a single `PADDED_HEAD` flag has to be driven by the *narrower* of the two
# axes. `(32, 64)` asks the same question with the roles swapped, which is a
# different path: one masks the V/O column axis, the other the QK one.
_D_HEAD = [64, (64, 32), (32, 64)]

# 11/31 is `test_fast`'s own shape and already carries 141 known failures, so it
# is kept rather than replaced. 8 is the only addition: several families fail at
# a query extent well under BLOCK_M and nothing in `test_fast` goes below 11.
_SEQLEN_Q = [8, 11]
_SEQLEN_K = [31]

# fp32 is dropped -- not because it is uninteresting, but because every failure
# in the log is dtype0/dtype1 and this is meant to stay fast.
_DTYPES = DTYPES[:2]


@pytest.mark.parametrize('BATCH', [3])
@pytest.mark.parametrize('N_HEADS', _N_HEADS, ids=fmt_nheads)
@pytest.mark.parametrize('D_HEAD', _D_HEAD, ids=fmt_hdim)
@pytest.mark.parametrize('seqlen_q', _SEQLEN_Q)
@pytest.mark.parametrize('seqlen_k', _SEQLEN_K)
@pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5] if BWD_IMPL != 2 else [0.0])
@pytest.mark.parametrize('dtype', _DTYPES)
@pytest.mark.parametrize('sm_scale', ['l1'])
@pytest.mark.parametrize('storage_flip', [False, True])
@pytest.mark.parametrize('bias_type', [None, 'matrix'], ids=['BiasOff', 'BiasOn'])
@pytest.mark.parametrize('BWDOP', BWDOP_ids)
def test_fast(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k,
              causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
    if bias_type is not None and BWD_IMPL == 2:
        pytest.skip("Bias is not supported in AITER ASM backend")
    args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale,
            dropout_p, dtype, storage_flip, bias_type)
    core_test_op_bwd(request, args, device=gpu_id)
