"""Model the dualwave KV LDS write/read mapping, so a new geometry can be
checked before it is coded.

Write side (per KV tile, one d-band per DMA issue):
  wave w, issue j, band d, lane l  ->  line = (w + j*NUM_WAVES) + d*N_RPT
                                       elem = l*VEC + i
  holds global (token, D) with
      token = (l // BUCKETS) * N_RPT + line_n     [line_n = w + j*NUM_WAVES]
      D     = (l %  BUCKETS) * VEC + d*GRAN + i

Read side (K), production form generalised:
  line_n   = lane32 %  N_RPT
  in_line  = (lane32 // N_RPT) * GRAN + ld*VEC + (ks % KSPB) * KSTEP
  band     = ks // KSPB
  pack hi adds N_STRIP to in_line.
"""

VEC = 8
KSTEP = 16
LINE_ELEMS = 512  # a wave's DMA payload


def geometry(gran, block_n, num_waves, head_dim):
    n_per_wave = LINE_ELEMS // gran
    n_rpt = block_n // n_per_wave
    d_rpt = head_dim // gran
    assert block_n % n_per_wave == 0, "BLOCK_N not a multiple of tokens/issue"
    assert n_rpt % num_waves == 0, "lines do not divide across waves"
    return dict(
        gran=gran,
        block_n=block_n,
        num_waves=num_waves,
        head_dim=head_dim,
        n_per_wave=n_per_wave,
        n_rpt=n_rpt,
        d_rpt=d_rpt,
        buckets=gran // VEC,
        kspb=gran // KSTEP,
        issues=n_rpt // num_waves,
        k_steps=head_dim // KSTEP,
    )


def write_map(g):
    """{(line, elem): (token, D)} for the whole KV tile."""
    m = {}
    for w in range(g["num_waves"]):
        for j in range(g["issues"]):
            line_n = w + j * g["num_waves"]
            for d in range(g["d_rpt"]):
                line = line_n + d * g["n_rpt"]
                for lane in range(64):
                    for i in range(VEC):
                        tok = (lane // g["buckets"]) * g["n_rpt"] + line_n
                        D = (lane % g["buckets"]) * VEC + d * g["gran"] + i
                        m[(line, lane * VEC + i)] = (tok, D)
    return m


def k_read(g, lane, ks, hi, n_strip):
    """(line, [elems]) a lane reads for K at k-step `ks`."""
    lane32, ld = lane % 32, lane // 32
    line_n = lane32 % g["n_rpt"]
    band = ks // g["kspb"]
    in_line = (lane32 // g["n_rpt"]) * g["gran"] + ld * VEC + (ks % g["kspb"]) * KSTEP
    if hi:
        in_line += n_strip
    line = line_n + band * g["n_rpt"]
    return line, [in_line + i for i in range(VEC)]


def check_k(g, n_strip):
    """K reads must be in-line, cover the tile exactly once, and give
    D == ks*16 + ld*8 + i with a token that tiles [0, BLOCK_N)."""
    wm = write_map(g)
    seen, problems = {}, []
    for lane in range(64):
        ld = lane // 32
        for ks in range(g["k_steps"]):
            for hi in (0, 1):
                line, elems = k_read(g, lane, ks, hi, n_strip)
                for i, e in enumerate(elems):
                    if e >= LINE_ELEMS:
                        problems.append(f"lane{lane} ks{ks} hi{hi}: elem {e} past the written {LINE_ELEMS}")
                        continue
                    tok, D = wm[(line, e)]
                    want_D = ks * KSTEP + ld * VEC + i
                    if D != want_D:
                        problems.append(f"lane{lane} ks{ks} hi{hi} i{i}: D={D} want {want_D}")
                    seen.setdefault((ks, tok, D), []).append((lane, hi))
    # every (ks, token, D) the GEMM needs must be read exactly once
    for ks in range(g["k_steps"]):
        for tok in range(g["block_n"]):
            for sub in range(KSTEP):
                key = (ks, tok, ks * KSTEP + sub)
                if key not in seen:
                    problems.append(f"never read: ks{ks} token{tok} D{ks*KSTEP+sub}")
                elif len(seen[key]) > 1:
                    problems.append(f"read {len(seen[key])}x: ks{ks} token{tok}")
    return problems


def report(name, gran, block_n, num_waves, head_dim, n_strip=None):
    try:
        g = geometry(gran, block_n, num_waves, head_dim)
    except AssertionError as e:
        print(f"{name:<10} gran={gran} BN={block_n} waves={num_waves} hd={head_dim}: INVALID -- {e}")
        return
    ns = n_strip if n_strip is not None else 32 * VEC
    probs = check_k(g, ns)
    tag = "OK" if not probs else f"{len(probs)} problem(s)"
    print(
        f"{name:<10} gran={gran:<3} BN={block_n:<4} waves={num_waves} hd={head_dim:<4} "
        f"n_rpt={g['n_rpt']} issues={g['issues']} buckets={g['buckets']} kspb={g['kspb']} "
        f"n_strip={ns:<4} -> {tag}"
    )
    for p in probs[:3]:
        print(f"             {p}")


if __name__ == "__main__":
    print("=== family A (the known-good production geometry) ===")
    report("A", 64, 64, 8, 64)
    report("A", 64, 64, 8, 128)
    print()
    print("=== family S candidates (granule 32) ===")
    report("S/8w", 32, 128, 8, 32)
    report("S/4w", 32, 64, 4, 32)
    report("S/4w", 32, 64, 4, 96)
    print()
    print("=== family B candidates (4 waves) ===")
    report("B", 64, 128, 4, 192)
    report("B", 64, 64, 4, 192)
