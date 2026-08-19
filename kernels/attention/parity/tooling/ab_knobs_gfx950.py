"""Confirm the sweep's candidate wins by interleaved A/B on one idle GPU.

The sweep ran four GPUs at once and its margins are 2-5%, which is the range
where drift and neighbours matter. Here each pair is measured alternately in
the same process, several rounds, and the *median* of each arm is reported --
so a slow patch of wall-clock hits both arms equally.
"""

import statistics
import sys
import time

sys.path.insert(0, "/home/xinyazha/dockerhome/meff/FlyDSL/kernels/attention/parity")
import torch  # noqa: E402
from flash_attn_func_gfx950 import build_flash_attn_func_gfx950_module_primary as build_primary  # noqa: E402
from fmha_tuning_gfx950 import FmhaInputMetadata, fmha_knobs  # noqa: E402

DT = torch.bfloat16

# (head_dim, label, knob overrides). First entry of each group is the current default.
GROUPS = [
    (64, [("cur wpe2", {}), ("wpe1", dict(waves_per_eu=1)), ("wpe4", dict(waves_per_eu=4))]),
    (96, [("cur wpe2", {}), ("wpe1", dict(waves_per_eu=1)), ("wpe4", dict(waves_per_eu=4))]),
    (128, [("cur wpe2", {}), ("wpe1", dict(waves_per_eu=1))]),
    (160, [("cur wpe2", {}), ("wpe4", dict(waves_per_eu=4))]),
    (192, [("cur g64 wpe2", {}), ("g32 wpe2", dict(num_waves=4, block_m=128, block_n=64, head_dim_granule=32))]),
    (224, [("cur wpe2", {}), ("wpe4", dict(waves_per_eu=4)), ("wpe1", dict(waves_per_eu=1))]),
    (
        256,
        [
            ("cur g64 wpe2", {}),
            ("g32 wpe4", dict(num_waves=4, block_m=128, block_n=64, head_dim_granule=32, waves_per_eu=4)),
            ("g32 wpe2", dict(num_waves=4, block_m=128, block_n=64, head_dim_granule=32)),
            ("g64 wpe4", dict(waves_per_eu=4)),
        ],
    ),
    (384, [("cur g64 wpe2", {}), ("g64 wpe4", dict(waves_per_eu=4))]),
    (512, [("cur", {}), ("wpe4", dict(waves_per_eu=4))]),
]


def shape_for(hd):
    return (4, 8, 4096) if hd <= 128 else (2, 8, 4096) if hd <= 256 else (1, 8, 4096)


def timed(call, rep=30):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep


for hd, arms in GROUPS:
    B, H, S = shape_for(hd)
    q, k, v = (torch.randn(B, H, S, hd, device="cuda", dtype=DT) for _ in range(3))
    o = torch.empty(B, H, S, hd, device="cuda", dtype=DT)
    meta = FmhaInputMetadata(num_heads=8, head_dim=hd, causal=False)
    built = []
    for label, ov in arms:
        try:
            knobs = fmha_knobs("gfx950", **ov).resolve(meta)
            fn = build_primary(meta, knobs)

            def call(fn=fn, q=q, k=k, v=v, o=o):
                fn(q, k, v, o, B, S, seqlen_k=S, scale=None, lse=None)

            call()
            built.append((label, call))
        except Exception as e:  # noqa: BLE001
            print(f"hd={hd:>4} {label:>14}: BUILD {type(e).__name__}")
    torch.cuda.synchronize()
    for _ in range(10):  # shared warmup, so no arm pays the cold cost
        for _, call in built:
            call()
    samples = {label: [] for label, _ in built}
    for _ in range(7):  # interleaved rounds
        for label, call in built:
            samples[label].append(timed(call))
    flops = 2.0 * B * H * S * S * hd * 2
    base = None
    for label, _ in built:
        med = statistics.median(samples[label])
        tf = flops / med * 1e-12
        spread = (max(samples[label]) - min(samples[label])) / med * 100
        if base is None:
            base = tf
            print(f"hd={hd:>4} {label:>14}: {tf:7.1f} TF   (spread {spread:4.1f}%)   [baseline]")
        else:
            print(f"hd={hd:>4} {label:>14}: {tf:7.1f} TF   (spread {spread:4.1f}%)   {100 * (tf / base - 1):+6.1f}%")
    del q, k, v, o
    torch.cuda.empty_cache()
