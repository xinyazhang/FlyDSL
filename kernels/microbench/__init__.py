# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Standalone microbenchmarks that establish machine limits.

These measure hardware ceilings (matrix-pipeline throughput, bandwidth) rather
than any production kernel, so that kernel results can be read as a fraction of
what the part can actually do. Each module is runnable directly:

    export ROCM_PATH=$(rocm-sdk path --root)   # only for a pip-installed ROCm
    python3 kernels/microbench/wmma_peak.py
"""
