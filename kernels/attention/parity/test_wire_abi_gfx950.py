# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""The gfx950 kernels' tensor operands are pointers, and stay pointers.

AOTriton dispatches the compiled hsaco directly: it fills the kernarg block
from a C++ struct holding a base pointer and the strides it was already
passing. An `fx.Tensor` operand costs a second slot for a packed shape+stride
descriptor that such a caller has no way to fill and the kernel never reads, so
every tensor operand of the three gfx950 kernels is declared `fx.Pointer`.
`fmha_dualwave_gfx950.wire_ptr` and `wire_view` are the two ends of that.

Both properties this checks are the kind that break silently.

- An operand that drifts back to `fx.Tensor` still *works* from Python, because
  the launcher is called with torch tensors either way. It just moves every
  later kernarg by 40 bytes, and the caller that notices is the one reading the
  block by offset.
- A `.shape` read off a `wire_view` also still works, and returns 1 -- the
  placeholder layout -- rather than raising. Under an `fx.Tensor` operand the
  same expression returned the caller's true extent, so this is a difference
  that produces a wrong number instead of an error.

Static: parses the source, needs no GPU, runs in milliseconds. Same shape and
the same reason as `test_signature_parity_gfx1201.py`.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_HERE = pathlib.Path(__file__).parent

# `kernel name -> operands that must be `fx.Pointer`". Spelled out rather than
# inferred: the point is that adding an operand makes someone write it down.
_WIRE_OPERANDS = {
    "flash_attn_func_gfx950.py": {
        "flash_attn_func_gfx950_kernel": ["Q", "K", "V", "B", "O", "LSE"],
        "launch_flash_attn_func_gfx950": ["Q", "K", "V", "O", "LSE", "Bias"],
        # `WS` is absent on purpose: the workspace is allocated by the launcher,
        # never by an external caller, and is `Constexpr` in a non-split-K build.
        "flash_attn_splitk_combine_kernel": ["O", "LSE"],
    },
    "fmha_bwd_dq_gfx950.py": {
        "fmha_bwd_dq_gfx950_kernel": ["Q", "K", "V", "B", "DO", "DQ", "DB", "LSE", "Delta"],
        "launch_fmha_bwd_dq_gfx950": ["Q", "K", "V", "Bias", "DO", "DQ", "DB", "LSE", "Delta"],
    },
    "fmha_bwd_dkdv_gfx950.py": {
        "fmha_bwd_dkdv_gfx950_kernel": ["Q", "K", "V", "B", "DO", "DK", "DV", "LSE", "Delta"],
        "launch_fmha_bwd_dkdv_gfx950": ["Q", "K", "V", "Bias", "DO", "DK", "DV", "LSE", "Delta"],
    },
}

# Modules whose traced code may hold a `wire_view`, and so must not read a size
# back off one.
_TRACED_MODULES = [
    "flash_attn_func_gfx950.py",
    "fmha_dualwave_gfx950.py",
    "fmha_wide_gfx950.py",
    "fmha_bwd_dq_gfx950.py",
    "fmha_bwd_dq_m16_gfx950.py",
    "fmha_bwd_dkdv_gfx950.py",
    "fmha_bwd_dkdv_m16_gfx950.py",
]

# The attribute names the wire views are stored under, on the kernel contexts,
# plus the kernel parameters themselves.
_WIRE_NAMES = {"Q", "K", "V", "B", "O", "DO", "DQ", "DK", "DV", "DB", "LSE", "Delta", "Bias", "DebugCounts"}

# Reads that would have returned a real extent from an `fx.Tensor` operand.
_SIZE_READS = {"shape", "layout"}


def _functions(path: pathlib.Path) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef)}


def _decorator_path(node: ast.expr) -> str:
    while isinstance(node, ast.Call):
        node = node.func
    return ast.unparse(node)


def _traced_scopes(tree: ast.Module) -> list[ast.AST]:
    """The subtrees that become device code, and so may hold a `wire_view`.

    Two kinds, and the distinction is what makes this test usable at all: a
    function carrying `@flyc.kernel` or `@flyc.jit`, and any method of a class
    -- the kernel contexts and their helpers, which is where the views are
    stored as `self.Q` and friends.

    Everything else in these modules is host code, and host code reads shapes
    off *real* torch tensors on purpose: `_args` is where `bias.shape` is
    checked against Q's, and where the 8xD contract is enforced. Those reads are
    correct and must not be flagged.
    """
    scopes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            scopes += [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        elif isinstance(node, ast.FunctionDef):
            if any(_decorator_path(d) in ("flyc.kernel", "flyc.jit") for d in node.decorator_list):
                scopes.append(node)
    return scopes


@pytest.mark.parametrize("filename", sorted(_WIRE_OPERANDS))
def test_tensor_operands_are_pointers(filename):
    """Every listed operand is annotated `fx.Pointer`, in every listed function."""
    funcs = _functions(_HERE / filename)
    for fn_name, operands in _WIRE_OPERANDS[filename].items():
        assert fn_name in funcs, f"{filename}: no function named {fn_name}"
        annotated = {a.arg: ast.unparse(a.annotation) if a.annotation else None for a in funcs[fn_name].args.args}
        for operand in operands:
            assert operand in annotated, f"{filename}:{fn_name}: no parameter {operand!r}"
            assert annotated[operand] == "fx.Pointer", (
                f"{filename}:{fn_name}: {operand} is {annotated[operand]}, expected fx.Pointer. "
                f"A tensor operand costs a second kernarg slot for a shape+stride descriptor "
                f"the kernel never reads -- see `wire_ptr` in fmha_dualwave_gfx950.py."
            )


# `(module, kernel, launcher)` for the three pairs, and the arguments the
# launcher carries that its kernel does not. `stream` is the usual one;
# `batch_size` is the other kind -- it sizes the grid's third axis and the
# kernel recovers its plane from `block_idx.z`, so passing it would be a dead
# kernarg. Same rule and the same reasoning as `test_signature_parity_gfx1201`.
_PAIRS = [
    ("flash_attn_func_gfx950.py", "flash_attn_func_gfx950_kernel", "launch_flash_attn_func_gfx950"),
    ("fmha_bwd_dq_gfx950.py", "fmha_bwd_dq_gfx950_kernel", "launch_fmha_bwd_dq_gfx950"),
    ("fmha_bwd_dkdv_gfx950.py", "fmha_bwd_dkdv_gfx950_kernel", "launch_fmha_bwd_dkdv_gfx950"),
]
_JIT_ONLY = {"stream", "batch_size"}

# The bias operand is `B` on all three kernels and `Bias` on all three
# launchers. A pure rename -- the position is identical in every case -- so it
# is normalised rather than reported.
_ALIAS = {"B": "Bias"}

# **The forward's declarations do not line up, and this records that rather
# than fixing it.** Its kernel takes `Q K V B O LSE`; its launcher declares
# `Q K V O LSE Bias`. The call site passes them positionally in the kernel's
# order and is correct, but a reader comparing the two signatures sees Bias in
# O's slot -- exactly the trap `test_signature_parity_gfx1201` forbids outright.
#
# Not fixed here because reordering a kernel signature moves every later kernarg
# and AOTriton binds to those offsets; that is an ABI decision with a caller on
# the other end. Both backward kernels already line up exactly. Pinned to
# name-and-count so the divergence cannot grow while nobody is looking.
_ORDER_DIVERGES = {"flash_attn_func_gfx950.py"}


@pytest.mark.parametrize("filename,kernel,launcher", _PAIRS, ids=[p[0] for p in _PAIRS])
def test_launcher_and_kernel_line_up(filename, kernel, launcher):
    """The launcher hands the kernel its arguments positionally, so order is the ABI.

    `test_signature_parity_gfx1201.test_launcher_and_kernel_agree` enforces this
    for the gfx1201 kernels but cannot reach these: its `_signatures` requires
    exactly one `@flyc.kernel` and one `@flyc.jit` per module, and each of these
    modules has more -- a split-K combine kernel in the forward's case, and
    nested `@flyc.jit` helpers in the bodies. Naming the pair explicitly is what
    makes the same check apply here.

    Forty-odd untyped-at-the-call-site arguments, and two swapped anywhere along
    the chain compiles, launches, and returns a plausible wrong answer.
    """
    funcs = _functions(_HERE / filename)
    kernel_args = [_ALIAS.get(a.arg, a.arg) for a in funcs[kernel].args.args]
    jit_args = [_ALIAS.get(a.arg, a.arg) for a in funcs[launcher].args.args]

    extra = sorted(set(jit_args) - set(kernel_args))
    assert set(extra) <= _JIT_ONLY, (
        f"{filename}: the launcher takes {sorted(set(extra) - _JIT_ONLY)}, which its kernel does not. "
        f"Either the kernel should take it too, or it belongs in _JIT_ONLY with a reason."
    )
    common = [a for a in jit_args if a in kernel_args]
    if filename in _ORDER_DIVERGES:
        assert sorted(common) == sorted(kernel_args), (
            f"{filename}: the launcher and kernel no longer even carry the same arguments.\n"
            f"  kernel: {kernel_args}\n"
            f"  jit:    {common}"
        )
        return
    assert common == kernel_args, (
        f"{filename}: the launcher's arguments do not line up with the kernel's.\n"
        f"  kernel: {kernel_args}\n"
        f"  jit:    {common}"
    )


@pytest.mark.parametrize("filename", _TRACED_MODULES)
def test_no_size_read_off_a_wire_view(filename):
    """No `.shape` or `.layout` off an operand that is really a pointer.

    `wire_view` gives them a placeholder layout, so such a read returns 1
    instead of the caller's extent -- a wrong answer, not an error. Every extent
    these kernels need is on the wire already: `max_seqlen_q/k`, `hdim_qk/vo`,
    the head counts and the fifteen strides. `batch_size` is not, and does not
    need to be: it appears only as `batch_idx * stride_batch`, with `batch_idx`
    coming from the grid.
    """
    tree = ast.parse((_HERE / filename).read_text())
    bad = []
    for scope in _traced_scopes(tree):
        for node in ast.walk(scope):
            if not isinstance(node, ast.Attribute) or node.attr not in _SIZE_READS:
                continue
            base = node.value
            # `self.Q.shape`, and a bare `Q.shape` on a kernel parameter.
            name = None
            if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and base.value.id == "self":
                name = base.attr
            elif isinstance(base, ast.Name):
                name = base.id
            if name in _WIRE_NAMES:
                bad.append(f"line {node.lineno}: {ast.unparse(node)}")
    assert not bad, (
        f"{filename} reads a size off a wire view, which returns the placeholder "
        f"rather than the caller's extent:\n  " + "\n  ".join(bad)
    )
