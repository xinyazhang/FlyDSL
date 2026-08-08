import csv

# ---------------- Architectural constants (gfx1201) ----------------
WMMA_CYC = 16  # execution cycles per 16x16x16 WMMA (FP16->FP32)
WMMA_CYC_FP8 = 8  # FP8 variant (2x rate)
FULL_RATE = 1  # cyc / wave32 op (ADD/MUL/FMA/MAX/SUB/CVT)
TRANSC_RATE = 4  # cyc / wave32 op (EXP2, RCP, LOG2, RSQ, SQRT)
WAVE = 32
CO_EXECUTE = False  # gfx1201 WMMA cannot overlap VALU


def wmma_ops(M, N, K):
    return (M // 16) * (N // 16) * (K // 16)


def sdpa_efficiency(Br=128, Bc=128, d=128, fp8=False):
    wcyc = WMMA_CYC_FP8 if fp8 else WMMA_CYC

    # ---- WMMA cycles: QK^T (Br x Bc x d) + PV (Br x d x Bc) ----
    n_qk = wmma_ops(Br, Bc, d)
    n_pv = wmma_ops(Br, d, Bc)
    C_wmma = (n_qk + n_pv) * wcyc

    # ---- Softmax VALU cycles (score tile = Br x Bc elems) ----
    elem_waveops = (Br * Bc) // WAVE  # elementwise pass, in wave32 ops
    row_waveops = Br // WAVE  # per-row ops (rcp normalize)

    softmax = {
        "row_max (MAX)": elem_waveops * FULL_RATE,
        "x-max (SUB)": elem_waveops * FULL_RATE,
        "exp2 (EXP)": elem_waveops * TRANSC_RATE,
        "row_sum (ADD)": elem_waveops * FULL_RATE,
        "acc_rescale (FMAC)": elem_waveops * FULL_RATE,
        "normalize (RCP)": row_waveops * TRANSC_RATE,
        "cast P (CVT)": elem_waveops * FULL_RATE,
    }
    C_softmax = sum(softmax.values())

    if CO_EXECUTE:
        total = max(C_wmma, C_softmax)
    else:
        total = C_wmma + C_softmax

    eff = C_wmma / total
    return C_wmma, C_softmax, eff, softmax


# ---------------- Run a sweep ----------------
configs = [
    dict(Br=128, Bc=128, d=128, fp8=False),
    dict(Br=128, Bc=128, d=128, fp8=True),
    dict(Br=128, Bc=256, d=128, fp8=False),
    dict(Br=64, Bc=64, d=128, fp8=False),
    dict(Br=128, Bc=128, d=64, fp8=False),
]

rows = []
for c in configs:
    C_w, C_s, eff, _ = sdpa_efficiency(**c)
    rows.append({**c, "C_WMMA": C_w, "C_softmax": C_s, "efficiency_%": round(eff * 100, 1)})
    print(f"{c}  ->  WMMA={C_w:>6}  softmax={C_s:>5}  eff={eff*100:5.1f}%")

with open("sdpa_efficiency_gfx1201.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nWrote sdpa_efficiency_gfx1201.csv")
