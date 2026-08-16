# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Import `kernels/*` from the working tree, from a script run in place.

The gfx950 twin of `gfx1201_standalone.py`, and it exists for the same reason:
these files are run directly out of this directory, so neither
`from ..common import ...` (no package context) nor
`from kernels.common import ...` (repository root not on `sys.path`) resolves.

It re-exports one thing the gfx1201 shim does not -- `dualwave`, which is
`kernels.attention.flash_attn_utils`. That module holds the `Dualwave*` helper
classes this kernel subclasses, and it is **imported, never edited**: it is
also imported by `flash_attn_generic.py`, `flash_attn_gfx950.py` and
`flash_attn_fp8_gfx950.py`, so a change there reaches four production kernels.
Everything gfx950-specific to this port lives in `fmha_dualwave_gfx950.py` as a
subclass.

A separate file rather than an import of the gfx1201 shim: the two differ in
what they re-export, and importing one module purely for its `sys.path`
side-effect reads as an accident waiting to be "cleaned up".
"""

import sys
from pathlib import Path

# parity/gfx950_standalone.py -> parity -> attention -> kernels -> repo
_REPO_ROOT = Path(__file__).resolve().parents[3]

if str(_REPO_ROOT) not in sys.path:
    # Front of the path: if a `kernels` package is ever installed, the working
    # tree must still win, or an edit here would silently do nothing.
    sys.path.insert(0, str(_REPO_ROOT))

from kernels.attention import flash_attn_utils as dualwave  # noqa: E402
from kernels.common import (  # noqa: E402
    buffer_ops,  # noqa: E402
    kernels_common,  # noqa: E402
    layout_utils,  # noqa: E402
    mem_ops,  # noqa: E402
    utils,  # noqa: E402
)

__all__ = [
    "buffer_ops",
    "dualwave",
    "kernels_common",
    "layout_utils",
    "mem_ops",
    "utils",
    "repo_root",
]


def repo_root() -> Path:
    """The repository root this module put on `sys.path`."""
    return _REPO_ROOT
