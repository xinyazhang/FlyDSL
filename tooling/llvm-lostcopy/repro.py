import sys
from flydsl._mlir import ir
from flydsl._mlir.passmanager import PassManager

with ir.Context():
    m = ir.Module.parse(open(sys.argv[1]).read())
    PassManager.parse(
        'builtin.module(gpu-module-to-binary{format=isa opts="--amdgpu-waves-per-eu=1"})'
    ).run(m.operation)
    open(sys.argv[2], "w").write(str(m))
