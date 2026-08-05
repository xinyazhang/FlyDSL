# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Import `kernels/common/*` from the working tree, from a script run in place.

The gfx1201 attention files are run directly out of this directory rather than
as part of an installed package, and that breaks the two obvious ways of
reaching shared code:

- **`from ..common import mem_ops` does not work at all.** A module run as
  `python3 foo.py` or imported by a test collected from this directory has no
  package context, so there is no `..` to be relative to.
- **`from kernels.common import mem_ops` does not work either**, because the
  repository root is not on `sys.path` -- only this directory is. `flydsl`
  resolves because it is installed in site-packages; `kernels` is not
  installed, so it simply is not found.

So this module puts the repository root on `sys.path` and re-exports the shared
modules under short names:

    from gfx1201_standalone import mem_ops, utils
    ptr = mem_ops.get_llvm_ptr(...)

**Why the path insert rather than loading the files by path.** Several of the
shared modules import each other absolutely -- `common/utils.py` does
`from kernels.common.mem_ops import global_load`, and `kernels_common.py` does
`from kernels.common.mem_ops import _create_llvm_ptr`. Loading a single file
with `importlib.util.spec_from_file_location` gives you a module whose own
imports still fail, so the package has to be importable either way. Making it
importable is the smaller change.

**What this buys, and the thing to keep true.** `kernels.common` resolves to
the *working tree*, so an edit to `kernels/common/mem_ops.py` takes effect on
the next run with no reinstall. `flydsl` still resolves to site-packages, which
is intended -- this shim is about the repository's own shared kernels, not
about the compiler.

This is scaffolding for running the prototype in place. Once these files move
under the installed package the import becomes a plain
`from kernels.common import mem_ops` and this module goes away.
"""

import sys
from pathlib import Path

# kernels/attention/gfx1201_standalone.py -> kernels/attention -> kernels -> repo
_REPO_ROOT = Path(__file__).resolve().parents[2]

if str(_REPO_ROOT) not in sys.path:
    # Front of the path: if a `kernels` package is ever installed, the working
    # tree must still win, or an edit here would silently do nothing -- which
    # is the failure this module exists to prevent.
    sys.path.insert(0, str(_REPO_ROOT))

from kernels.common import buffer_ops  # noqa: E402
from kernels.common import kernels_common  # noqa: E402
from kernels.common import layout_utils  # noqa: E402
from kernels.common import mem_ops  # noqa: E402
from kernels.common import utils  # noqa: E402
from kernels.common.mma import wmma_ops  # noqa: E402

__all__ = [
    "buffer_ops",
    "kernels_common",
    "layout_utils",
    "mem_ops",
    "utils",
    "wmma_ops",
    "repo_root",
]


def repo_root() -> Path:
    """The repository root this module put on `sys.path`.

    Exposed so a caller can resolve a sibling path without recomputing the
    `parents[2]` hop, which is wrong the moment this file moves.
    """
    return _REPO_ROOT
