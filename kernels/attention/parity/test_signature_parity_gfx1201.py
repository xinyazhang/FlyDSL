# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""The `@flyc.jit` wrapper and the `@flyc.kernel` it launches must agree.

Every launch in this directory hands the kernel its arguments **positionally**,
and the host hands the launcher *its* arguments positionally too, through
`abi.run_compiled(exe, *args)`. Three ordered lists of forty-odd untyped-at-the-
call-site scalars, and nothing checks that they line up. Two `fx.Int32` swapped
anywhere along that chain compiles, launches, and produces a plausible wrong
answer -- `num_head_k` for `hdim_vo` would just read the wrong strides.

So the two signatures are kept identical, and this is what enforces it. It is a
static test: it parses the source rather than importing it, needs no GPU, and
runs in milliseconds.

The rule is subsequence rather than equality because a launcher legitimately
takes arguments its kernel does not. `stream` is the obvious one. `batch_size` in the
mask kernel is the other kind: it sizes the grid's third axis, and the kernel
recovers its plane from `block_idx.z` instead, so passing it would be a dead
kernarg. Both are listed per module below, which is the point -- a new
divergence has to be written down here to pass, and writing it down is where
someone asks whether it should exist.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

# Arguments the launcher may carry that its kernel does not. Anything else is a
# failure, including a *reordering*, which subsequence matching also catches.
_JIT_ONLY = {
    "flash_attn_func_gfx1201_aiw.py": {"stream"},
    "fmha_bwd_dq_gfx1201_kernel.py": {"stream"},
    "fmha_bwd_dkdv_gfx1201_kernel.py": {"stream"},
    "fmha_bwd_fuse_gfx1201_kernel.py": {"stream"},
    "dropout_mask_gfx1201.py": {"stream", "batch_size"},
}

_HERE = pathlib.Path(__file__).parent


def _decorator_path(node: ast.expr) -> str:
    while isinstance(node, ast.Call):
        node = node.func
    return ast.unparse(node)


def _signatures(path: pathlib.Path) -> tuple[list[str], list[str]]:
    """`(kernel args, jit args)` for the one decorated pair in `path`."""
    kernels, jits = [], []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.FunctionDef):
            continue
        for deco in node.decorator_list:
            which = _decorator_path(deco)
            if which == "flyc.kernel":
                kernels.append(node)
            elif which == "flyc.jit":
                jits.append(node)
    assert len(kernels) == 1, f"{path.name}: expected one @flyc.kernel, found {[k.name for k in kernels]}"
    assert len(jits) == 1, f"{path.name}: expected one @flyc.jit, found {[j.name for j in jits]}"
    return ([a.arg for a in kernels[0].args.args], [a.arg for a in jits[0].args.args])


@pytest.mark.parametrize("module", sorted(_JIT_ONLY))
def test_launcher_and_kernel_agree(module):
    kernel_args, jit_args = _signatures(_HERE / module)
    allowed = _JIT_ONLY[module]

    extra = [a for a in jit_args if a not in kernel_args]
    assert set(extra) <= allowed, (
        f"{module}: the launcher takes {sorted(set(extra) - allowed)}, which its kernel does not. "
        f"Either the kernel should take it too, or add it to _JIT_ONLY with a reason."
    )
    unused = allowed - set(jit_args)
    assert not unused, f"{module}: _JIT_ONLY lists {sorted(unused)}, which the launcher no longer takes"

    # Same names in the same order, once the launcher-only ones are dropped.
    assert [a for a in jit_args if a in kernel_args] == kernel_args, (
        f"{module}: the launcher's arguments do not line up with the kernel's.\n"
        f"  kernel: {kernel_args}\n"
        f"  jit:    {[a for a in jit_args if a in kernel_args]}"
    )


def test_every_builder_module_is_covered():
    """A new kernel must be added to `_JIT_ONLY`, not silently skipped."""
    found = set()
    for path in sorted(_HERE.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        src = path.read_text()
        if "@flyc.kernel" in src and "@flyc.jit" in src:
            found.add(path.name)
    assert found == set(_JIT_ONLY), (
        f"modules with a kernel/launcher pair but no parity entry: {sorted(found - set(_JIT_ONLY))}; "
        f"entries for modules that no longer have one: {sorted(set(_JIT_ONLY) - found)}"
    )
