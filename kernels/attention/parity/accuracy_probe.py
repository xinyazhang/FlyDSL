"""Measure FMHA output error against an fp64 reference.

Separates *bias* (mean signed relative error, which does not cancel when
summing) from *spread* (RMS relative error). A rounding scheme that is unbiased
shows bias << spread; round-toward-zero on a strictly positive quantity like P
shows bias comparable to spread.

Run from this directory:  python3 accuracy_probe.py
"""

import torch
from flash_attn_func_gfx1201_interface import flydsl_flash_attn_func_gfx1201

BATCH, H, N, D = 1, 4, 1024, 128


def reference(q, k, v, causal):
    q64, k64, v64 = (t.to(torch.float64) for t in (q, k, v))
    s = q64 @ k64.transpose(-1, -2) / (D**0.5)
    if causal:
        n = s.shape[-1]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=s.device), 1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, -1) @ v64


def stats(out, ref):
    o, r = out.to(torch.float64), ref
    scale = r.abs().mean()
    rel = (o - r) / scale
    return rel.mean().item(), rel.pow(2).mean().sqrt().item()


def main():
    torch.manual_seed(0)
    print(f"{'dtype':>9} {'causal':>7} {'source':>10} {'bias':>12} {'rms':>12} {'bias/rms':>9}")
    for dtype in (torch.float16, torch.bfloat16):
        for causal in (False, True):
            q, k, v = (torch.randn((BATCH, N, H, D), dtype=dtype, device="cuda") for _ in range(3))
            ref = reference(q, k, v, causal)
            fly = flydsl_flash_attn_func_gfx1201(q, k, v, causal=causal)
            sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
            for label, out in (("flydsl", fly), ("torch-sdpa", sdpa)):
                b, r = stats(out, ref)
                print(
                    f"{str(dtype).split('.')[-1]:>9} {int(causal):>7} {label:>10} "
                    f"{b:>12.3e} {r:>12.3e} {abs(b) / r:>9.2f}"
                )


if __name__ == "__main__":
    main()
