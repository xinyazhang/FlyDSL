"""Knob re-sweep for the gfx950 parity kernel. One shard per GPU.

Writes one JSON record per line as each point lands, so a shard that wedges on
a pathological build loses only that point rather than the whole shard.
"""

import json
import sys
import time

sys.path.insert(0, "/home/xinyazha/dockerhome/meff/FlyDSL/kernels/attention/parity")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from flash_attn_func_gfx950 import build_flash_attn_func_gfx950_module_primary as build_primary  # noqa: E402
from fmha_tuning_gfx950 import LADDER, FmhaInputMetadata, fmha_knobs  # noqa: E402

SHARD, NSHARD, CAUSAL = int(sys.argv[1]), int(sys.argv[2]), bool(int(sys.argv[3]))
OUT = open(sys.argv[4], "w", buffering=1)

GEOMS = [(8, 256, 64, 64), (4, 128, 64, 64), (8, 128, 64, 64), (4, 64, 64, 64), (8, 64, 64, 64), (4, 128, 64, 32)]
LDS_CAP, LDS_PER = 163840, 8.3  # two KV tiles in flight, bytes per (token, D) pair


def candidates():
    for hd in LADDER:
        # The D-axis splits only buy anything where LDS or registers bind.
        stages = (1,) if hd <= 256 else (2, 4)
        shards = (1,) if hd <= 256 else (1, 2, 4)
        for nw, bm, bn, gr in GEOMS:
            if hd % gr:
                continue
            for ds in stages:
                if hd % ds or bn * (hd // ds) * LDS_PER > LDS_CAP:
                    continue  # the compile would fail, and slowly
                for vs in shards:
                    for wpe in (1, 2, 4):
                        for lz in (True, False):
                            yield dict(
                                head_dim=hd,
                                num_waves=nw,
                                block_m=bm,
                                block_n=bn,
                                head_dim_granule=gr,
                                d_stages=ds,
                                vo_shards=vs,
                                waves_per_eu=wpe,
                                lazy_rescale=lz,
                            )


def shape_for(hd):
    return (4, 8, 4096) if hd <= 128 else (2, 8, 4096) if hd <= 256 else (1, 8, 4096)


def bench(call, warmup=12, rep=30):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(rep):
            call()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / rep)
    return best


DT = torch.bfloat16
todo = [c for i, c in enumerate(candidates()) if i % NSHARD == SHARD]
print(f"shard {SHARD}: {len(todo)} candidates", file=sys.stderr, flush=True)

for n, c in enumerate(todo):
    hd = c["head_dim"]
    meta = FmhaInputMetadata(num_heads=8, head_dim=hd, causal=CAUSAL)
    try:
        knobs = fmha_knobs("gfx950", **{k: v for k, v in c.items() if k != "head_dim"}).resolve(meta)
    except Exception:
        continue  # the resolver is the legality filter
    B, H, S = shape_for(hd)
    rec = dict(c, causal=int(CAUSAL))
    try:
        q, k, v = (torch.randn(B, H, S, hd, device="cuda", dtype=DT) for _ in range(3))
        o = torch.empty(B, H, S, hd, device="cuda", dtype=DT)
        fn = build_primary(meta, knobs)

        def call(fn=fn, q=q, k=k, v=v, o=o):
            fn(q, k, v, o, B, S, seqlen_k=S, scale=None, lse=None)

        call()
        torch.cuda.synchronize()
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=CAUSAL)
        err = (o.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1.0)
        if err < 2e-2:
            s = bench(call)
            flops = 2.0 * B * H * S * S * hd * 2 * (0.5 if CAUSAL else 1.0)
            rec.update(tf=round(flops / s * 1e-12, 1), err=round(err, 5), status="ok")
        else:
            rec.update(tf=0.0, err=round(err, 5), status="WRONG")
        del q, k, v, o
    except Exception as e:  # noqa: BLE001
        rec.update(tf=0.0, err=-1.0, status=type(e).__name__)
    torch.cuda.empty_cache()
    OUT.write(json.dumps(rec) + "\n")
    if n % 10 == 0:
        print(f"shard {SHARD}: {n}/{len(todo)}", file=sys.stderr, flush=True)

print(f"shard {SHARD}: DONE", file=sys.stderr, flush=True)
