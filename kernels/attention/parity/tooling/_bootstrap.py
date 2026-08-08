# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Put `kernels/attention/parity` on `sys.path`, for scripts run from here.

The parity kernels are imported by bare module name and expect their own
directory on the path -- `gfx1201_standalone.py` explains why. A tooling script
lives one level below them, so it has to add that directory itself before any
`from flash_attn_func_gfx1201_interface import ...` can work.

Import this first, before any parity import::

    import _bootstrap  # noqa: F401
    from flash_attn_func_gfx1201_interface import flydsl_flash_attn_func_gfx1201
"""

import sys
from pathlib import Path

PARITY = Path(__file__).resolve().parents[1]

if str(PARITY) not in sys.path:
    sys.path.insert(0, str(PARITY))
