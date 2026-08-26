# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Point this suite's bare ``device="cuda"`` at the GPU its worker leased.

These tests were written single-GPU and spell the device as ``"cuda"`` with no
index, several hundred times. Under ``pytest-xdist`` that puts every worker on
GPU 0 -- four processes contending for one device while the other three idle.

`pytest-gpu-lease` (aotriton, ``python/pytest-gpu-lease``) hands each worker an
exclusive GPU through ``fcntl`` byte-range locks and exposes it as ``gpu_id``.
It deliberately does *not* set ``HIP_VISIBLE_DEVICES`` or move the current
device: it leases, and the suite decides what to do with the lease. This is
that decision, in one place, so the tests keep saying ``"cuda"``.

``torch.cuda.set_device`` rather than ``HIP_VISIBLE_DEVICES``: the env var has
to be set before the runtime initialises, which a fixture cannot promise, and
it renumbers devices so ``gpu_id`` and what the driver reports stop agreeing.
Setting the current device is what ``"cuda"`` already resolves against.

Autouse, unlike every fixture the plugin ships -- the plugin stays inert until
something asks for a lease, and asking is exactly this file's job. Without it
the plugin loads and changes nothing.

Serial runs are unaffected: with no xdist worker the plugin reports GPU 0,
which is what these tests did before.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def _use_leased_gpu(gpu_id):
    """Make the leased GPU this worker's current device, for the whole session."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    return gpu_id
