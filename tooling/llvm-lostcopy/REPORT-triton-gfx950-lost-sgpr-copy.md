# SplitKit emits a live-range split whose copy-in lane mask is a strict subset of the register it then spills — reproduced from a Triton kernel, on three LLVM builds

**Target:** `amdgcn-amd-amdhsa`, `gfx950` (wave64)
**Frontend:** Triton 3.8.0 (a ~50-line kernel), reduced to a standalone `.ll`
**LLVM:** reproduces on three independent builds, including 24.0.0git @ `e2a39f504fee836e4def9581bed817ecc327b9dc`
**Reproducer:** compile-only, no GPU, ~1.3 s
**Companion:** [`REPORT-gfx950-splitkit-lost-subrange.md`](REPORT-gfx950-splitkit-lost-subrange.md) (the MLIR/FlyDSL-based report this one supports and, in two places, corrects)

---

## 0. What this document adds, in four lines

1. The defect **reproduces from a stock Triton kernel**, so no niche wheel is needed to see it. The kernel reduces to a self-contained `.ll` that `llc` can drive alone.
2. It reproduces on **three different LLVM builds**, one of which is the exact pinned revision of the companion report — so it is not an artifact of one compiler.
3. On the pinned build the offending copy is **literally tagged `lr-split`**, which is `SplitKit::buildCopy`. That is direct confirmation of the companion report's §1 mechanism, from a second frontend.
4. **The consumed form does not reproduce.** Across 10,265 Triton modules, the lost subregister is always dead or rematerialised before use. §6 gives the measurement; §7 corrects two claims in the companion report; §11 answers whether the remaining form is worth filing upstream on its own.

Separating **measured** from **inferred** throughout.

---

## 1. Environment and exact revisions

Everything below lives in one virtualenv, `~/.venvs/gfx950-10.0`, which is what makes the three-way comparison possible.

| component | version | LLVM revision | notes |
|---|---|---|---|
| Triton | `3.8.0+git4cff872c.rocm10.0.0` | **23.0.0git @ `4611156032e1ebac68b2b8f6107b1475a7c60800`** | statically linked into `triton/_C/libtriton.so` |
| ROCm SDK | 10.0.0 | **23.0.0git @ ROCm/llvm-project `8f497e0992fb7513f7f78a6f6b6f1056c375e961`** | `amdclang`, used to drive the `.ll` |
| flydsl | 0.3.2 | **24.0.0git @ `e2a39f504fee836e4def9581bed817ecc327b9dc`** | the companion report's pinned revision, inside `libFlyPythonCAPI.so.24.0git` |

**Neither 23.0.0git revision is an upstream `llvm/llvm-project` commit.** Measured: the local checkout is a full clone (595,179 commits on `upstream/main`, `is-shallow-repository` false, all release branches fetched), and `git cat-file` reports `4611156032e1…` and `8f497e0992fb…` as bad objects. Both are fork commits. **A hash-based `git merge-base --is-ancestor` test against the suspect commit is therefore unavailable**, which anyone filing an issue that names a commit should know.

### 1.1 Dating Triton's LLVM against the suspect commit `83dca924c250`

Because the hash is unavailable, Triton's LLVM was dated by **string bracketing**: `cl::opt` names and other string literals introduced upstream in a known window either are or are not present in the shipped `libtriton.so`.

| window (upstream `main`) | new string literals | present in `libtriton.so` |
|---|---|---|
| 2026-04-01 → `83dca924c250` (2026-05-13) | 54 | 37 |
| `83dca924c250` → 2026-06-20 | 54 | **16** |
| 2026-05-29 → 2026-06-25 (CodeGen/Transforms/Analysis/AMDGPU only) | 26 | **0** |
| 2026-06-20 → `e2a39f504fee` (2026-07-23) | 75 | **0** |

The 16 post-suspect hits were dated individually with `git log -S`. The **latest** is `stack-protector-guard-record`, introduced 2026-05-29 by `64bc1fae11d6`. Others include `amdgpu-spill-cfi-saved-regs` (2026-05-27, `4e39ea3d5e73`) and `ssaupdater-phi-search-limit` (2026-05-14, `536ae1dec796`).

**Inferred (high confidence):** Triton's LLVM base is upstream `main` at ≈2026-05-29, and therefore **contains `83dca924c250`** (2026-05-13). The inference assumes the fork's base is upstream `main` and that the commit was not reverted downstream; that is not directly verifiable here, because `getSubRangeForMask`/`findSubRangeForMask` are static functions with no symbols and the commit adds no string constant. The `llvm_unreachable("SubRange for this mask not found")` message is **not** a usable probe: it survives the commit at `SplitKit.cpp:402` and is present in both binaries.

So: **the suspect commit is present, and the defect reproduces.** Consistent with the companion's §8 and it adds a second revision. It does **not** prove §8 — see §8 below.

---

## 2. The reproducer

### 2.1 Kernel

`triton_kernels.py`, in this directory — a self-contained reduction of AOTriton's `bwd_preprocess`, with the `composed_tensors` decomposition removed.

```python
import triton
import triton.language as tl


@triton.jit
def bwd_preprocess_min(
    Out, DO, Delta,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_doz, stride_doh, stride_dom, stride_don,
    cu_seqlens_q, num_seqlens, max_seqlen_q, hdim_vo,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr, PADDED_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M
    offs_m = off_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    num_h = tl.num_programs(1)

    o_ptrs = (Out + off_z * stride_oz + off_h * stride_oh
              + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on)
    if num_seqlens == 0:
        seqlen_q = max_seqlen_q
    else:
        cu_start = tl.load(cu_seqlens_q + off_z)
        cu_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_end - cu_start
    do_ptrs = (DO + off_z * stride_doz + off_h * stride_doh
               + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_don)

    mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD:
        mask = mask & (offs_d[None, :] < hdim_vo)
    o = tl.load(o_ptrs, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)

    off_zh = off_z * num_h + off_h
    delta_ptrs = Delta + off_zh * max_seqlen_q + off_m + tl.arange(0, BLOCK_M)
    overflow = off_m + BLOCK_M - seqlen_q
    if overflow > 0:
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow, dtype=tl.int32)
        store_mask = boundary > tl.arange(0, BLOCK_M)
        tl.store(delta_ptrs, delta, mask=store_mask)
    else:
        tl.store(delta_ptrs, delta)
```

Three ingredients matter: two adjacent `i32` kernargs (`num_seqlens`, `max_seqlen_q`) that merge into one `S_LOAD_DWORDX2`; an `if`/`else` on the first whose two arms both need the pair afterwards; and enough register pressure (`D_HEAD=512`) to force SGPR spilling.

### 2.2 Commands

No GPU — `triton.compile` is given the target explicitly, so the HIP driver is never opened.

```sh
cd tooling/llvm-lostcopy
export LLVM_BIN=".../site-packages/_rocm_sdk_core/lib/llvm/bin"
export TRITON_DISABLE_LINE_INFO=1          # keeps the .ll free of debug metadata

python triton_repro.py triton_kernels.py bwd_preprocess_min /tmp/out \
    --sig "*fp16:16, *fp16:16, *fp32:16, u64:8, u64:8, u64:8, 1, u64:8, u64:8, u64:8, 1, *i32:16, i32, i32, i32, 128, 512, False" \
    --warps 4 --stages 1 --wpeu 2 --stem repro

python lostdef.py /tmp/out/repro.hsaco
#   repro.hsaco: 1 writelane(s) with no reaching definition
#       0x00001C74  v_writelane_b32 v255, s20, 2
```

Measured `sha256`: `repro.ll` = `12f9829ac7ad8240…`, `repro.hsaco` = `bec1414aeed34e4e…`.

### 2.3 The same thing without Triton

`repro-triton-gfx950.ll.gz` beside this file is the emitted module. It reproduces alone:

```sh
gunzip -c repro-triton-gfx950.ll.gz > /tmp/repro.ll
amdclang -x ir -c --target=amdgcn-amd-amdhsa -mcpu=gfx950 -O3 /tmp/repro.ll -o /tmp/repro.o
python lostdef.py /tmp/repro.o
#   repro.o: 1 writelane(s) with no reaching definition
#       0x00000174  v_writelane_b32 v255, s20, 2
```

`llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -O3 /tmp/repro.ll` should be equivalent; `llc` is not shipped in the ROCm tree and no LLVM build tree exists on this machine, so **`amdclang -x ir` is what was actually run** — stated plainly, because "we could not reproduce with a different driver" is exactly the ambiguity this report tries not to create for someone else.

### 2.4 On the pinned revision (24.0.0git @ `e2a39f504fee`)

No `llc` there either, but the revision *is* on this machine, static in the flydsl wheel. `mlir-translate --import-llvm` converts the `.ll` to the LLVM dialect; `wrap_gpu_module.py` wraps it in a `gpu.module` carrying `#rocdl.target<chip="gfx950">`; `drive2.py` runs `gpu-module-to-binary`, which is entirely the pinned AMDGPU backend.

```sh
mlir-translate --import-llvm /tmp/repro.ll -o /tmp/imported.mlir
python wrap_gpu_module.py /tmp/imported.mlir /tmp/repro.ll /tmp/wrapped.mlir gfx950
python drive2.py /tmp/wrapped.mlir /tmp/pinned.s
sed -i 's/amdgcn-amd-amdhsa-unknown-gfx950/amdgcn-amd-amdhsa--gfx950/' /tmp/pinned.s
amdclang -x assembler -c --target=amdgcn-amd-amdhsa -mcpu=gfx950 /tmp/pinned.s -o /tmp/pinned.o
python lostdef.py /tmp/pinned.o
#   pinned.o: 1 writelane(s) with no reaching definition
#       0x00000174  v_writelane_b32 v255, s20, 2
```

Same register, same lane, same shape as Triton's own backend produced.

*Driver sanity check:* the same `drive2.py` on the companion's MLIR module emits an `.s` with `sha256` `1452f52355effa38…` — the hash that report records.

---

## 3. MachineIR evidence — the `lr-split` marker

From `drive2.py … print-after=greedy` on the pinned build. `sreg_64` `%6690` is the merged `(num_seqlens, max_seqlen_q)` kernarg pair:

```
240B   %6690:sreg_64_xexec = S_LOAD_DWORDX2_IMM %628:sgpr_64(p4), 80, 0 ::
                             (dereferenceable invariant load (s64) from constant-pool + 80, align 16, addrspace 4)
320B   S_CMP_EQ_U32 %6690.sub0:sreg_64_xexec, 0, implicit-def $scc
...
376B   undef %6692.sub1:sreg_64_xexec = lr-split COPY %6690.sub1:sreg_64_xexec
380B   SI_SPILL_S64_SAVE %6692:sreg_64_xexec, %stack.2, implicit $exec, implicit $sgpr32 ::
                             (store (s64) into %stack.2, align 4, addrspace 5)
...
440B   undef %6691.sub1:sreg_64_xexec = lr-split COPY %6690.sub1:sreg_64_xexec
444B   SI_SPILL_S64_SAVE %6691:sreg_64_xexec, %stack.2, implicit $exec, implicit $sgpr32 ::
                             (store (s64) into %stack.2, align 4, addrspace 5)
```

`%6691` and `%6692` each occur **exactly twice** in the whole function — the `sub1`-only copy and the full-width spill. Neither has any definition of `sub0`.

The `lr-split` prefix is printed by `MachineInstr.cpp:1896` / `MIRPrinter.cpp:896` for instructions `SplitKit` creates, so this is `SplitEditor::buildCopy`, not some other partial copy. (The ROCm 23 build does not print the marker; the copies are there in the same shape, untagged.)

A second one in the same function, the other direction:

```
648B   undef %6697.sub0:sreg_64 = lr-split COPY %6695.sub0:sreg_64
652B   SI_SPILL_S64_SAVE %6697:sreg_64, %stack.3, ...
```

### 3.1 An important distinction this reproducer does *not* collapse

Two different width mismatches are in play, and conflating them would overstate what §2 shows.

| | mismatch | who disagrees | legality |
|---|---|---|---|
| **A** | copy-in mask (`sub1`) vs. the width the spiller stores (`sub0_sub1`) | `SplitKit` vs. `SIRegisterInfo`'s spill granularity | by design — `SI_SPILL_S64_SAVE` has no partial form |
| **B** | copy-in mask vs. a **later `SplitKit` copy-out** mask through the same slot | `SplitKit` vs. itself | the defect |

The companion's FlyDSL case is **B**: `%8004` is copied in with mask `{0,1,2,3,6,7}`, and a later `lr-split COPY` reads `{4,5,8..15}` out of the restored value.

**The §2 reproducer, on its own, is only A.** Its restore `%6693` reads exactly the `sub1` that was copied in. Triton *does* produce **B** — 131 times, in `bwd_kernel_fuse` (§6) — it just does not survive to the ISA there. That is the finding that carries weight, and §11 turns on it.

---

## 4. The same defect in the emitted ISA

Triton's own backend, `repro.hsaco`. The `else` arm spills the pair straight from `s[6:7]`; the `then` arm must stage it into `s[20:21]` and only one of the two `s_mov_b32` is emitted:

```asm
  1C1C: s_load_dwordx2 s[6:7], s[0:1], 0x50   ; kernarg 0x50 = (num_seqlens, max_seqlen_q)
  1C28: s_cmp_eq_u32   s6, 0
  1C2C: s_cbranch_scc1 16                     ; -> 0x1C70 when num_seqlens == 0
        ; ---- else arm: pair already in s[6:7] ----
  1C30: v_writelane_b32 v255, s6, 2
  1C40: v_writelane_b32 v255, s7, 3
  1C48: s_load_dwordx2 s[6:7], s[0:1], 0x48   ; cu_seqlens_q
  1C5C: s_load_dwordx2 s[20:21], s[6:7], 0x0
  1C68: s_sub_i32      s86, s21, s20          ; seqlen_q = end - start
  1C6C: s_branch       7                      ; -> 0x1C8C
        ; ---- then arm ----
  1C70: s_mov_b32      s21, s7                ; copy of sub1   -- present
                                              ; <<< MISSING: s_mov_b32 s20, s6  (copy of sub0)
  1C74: v_writelane_b32 v255, s20, 2          ; <<< spills s20, which has no reaching definition
  1C7C: s_mov_b32      s86, s7                ; seqlen_q = max_seqlen_q
  1C84: v_writelane_b32 v255, s21, 3
```

Hand-verified, not just detector output:

- the **only** two writes of `s20` before `0x1C74` are `s_lshl_b64 s[20:21], s[18:19], 2` at `0x1C38` and `s_load_dwordx2 s[20:21], s[6:7], 0x0` at `0x1C5C`, and **both are inside the not-taken arm** — `0x1C2C` branches over them to `0x1C70`, and `0x1C6C` jumps over `0x1C70`, so that block is reachable only from the branch that skipped both;
- branch arithmetic checks out: `0x1C2C + 4 + 16·4 = 0x1C70`, `0x1C6C + 4 + 7·4 = 0x1C8C`, matching the disassembler's `+0x170` / `+0x18c` annotations against a `0x1B00` kernel base;
- lane 2 of `v255` is written **only** at `0x1C30` and `0x1C74`, and is read back at `0x00003568`.

The identical shape, at identical addresses and registers, appears in an unrelated AOTriton production build (`bwd_preprocess-3649af6a…hsaco`, Triton 3.7.0, different venv, different LLVM build) — so this is not an artifact of the harness here.

---

## 5. What is *not* claimed: the lost lane is dead here

**Measured.** In this reproducer `sub0` is `num_seqlens`, dead after the compare. The reload at `0x3568` feeds nothing — the destination is overwritten before use — and the MachineIR agrees: the restore's only subregister use is `%6693.sub1`.

> **The mechanism, without the consequence.** `SplitKit` produced a value whose defined lanes are a strict subset of the register the spiller then stores whole, and the allocator spilled and reloaded the undefined lane. Nothing reads it.

Four predicates keep the cases apart:

| # | predicate | tool | meaning |
|---|---|---|---|
| cat1 | `v_writelane_b32` names an SGPR with no reaching definition | `lostdef.py` (pre-existing) | too weak — see §7.1 |
| cat2 | a spill slot is stored from a value with undefined lanes, no restore reads them | `mirscan.py` | mismatch **A** — the mechanism, benign |
| **cat3** | a restore reads a lane **no store into that slot ever defined on any path reaching it** | **`mircfg.py`** | mismatch **B** — the defect |
| cat3′ | the lost value is read by an instruction that is not a copy or a spill | `consumescan.py` | observable at ISA level |

`mircfg.py` is a must-analysis over the MachineIR CFG (state = lanes known-defined in the slot; merge = intersection over predecessors; a save overwrites; a restore reports `lanes-read − state`). Validated both ways:

- **fires** on the companion's failing object at the pinned revision, naming exactly the documented site:
  `%stack.3 (S512) restore %7901 in bb.37: reads [4,5,8..15], defined-on-all-paths [0,1,2,3,6,7] -> MISSING [4,5,8..15]`;
- **silent** on the known-clean sibling, on the §2 reproducer, and on every candidate subsequently hand-refuted.

`witness.py` prints a concrete block-by-block path for any hit, so "there is a path on which lane N is never written" can be checked by reading the dump rather than trusting the analysis.

---

## 6. Blast radius across Triton

Every AOTriton Triton kernel family, compiled for gfx950 at every configuration its build system generates, scanned at MachineIR level. 10,265 modules.

**Read the `attn_fwd` row with care: 843 is a sample, not the family size.** That family has 5060 configurations; the compile pool was killed by the OOM killer twice (at 48 and at 12 workers) and 843 completed. Every other row is a complete family.

| family | modules scanned | of total | cat2 (mechanism, A) | cat3 (MIR: undefined lane read, B) | cat3′ (ISA: computed with) |
|---|---|---|---|---|---|
| `bwd_kernel_dk_dv` | 2896 | 2896 (all) | 2023 | 0 | 0 |
| `bwd_kernel_dq` | 3738 | 3738 (all) | 1362 | 0 | 0 |
| `bwd_kernel_fuse` | 2644 | 2644 (all) | 2253 | **131** | 0 |
| `bwd_preprocess` | 72 | 72 (all) | 27 | 0 | 0 |
| `bwd_preprocess_varlen` | 72 | 72 (all) | 5 | 0 | 0 |
| `attn_fwd` | **843** | **of 5060 — SAMPLE, OOM-limited** | 262 | 0 | 0 |
| **total** | **10265** | — | **5932** | **131** | **0** |

**The mechanism is pervasive — 58% of Triton modules carry it.** Triton's backend does produce wide SGPR tuples (`SI_SPILL_S256` and `S128` both observed), just not the `sgpr_512` the FlyDSL kernel builds from 21 `i64` strides.

**The consumed form was not found.** All 131 `bwd_kernel_fuse` cat3 hits were followed down; none survives:

- The MIR shape is right, and it is mismatch **B**. `bwd_kernel_fuse_00454`, hand-verified: `%stack.13` holds a sign-extended index; `undef %14299.sub0 = COPY %14322.sub0` and `undef %14298.sub0 = S_MOV_B32 0` store only `sub0` (the two branches of `SplitKit::defFromParent` — a partial `buildCopy` and a `rematerializeAt`), while `%14300` and `%14317` store both lanes; five restores read `sub0` **and** `sub1`. `witness.py` gives a concrete path `bb.5 → bb.451 → bb.456 → bb.457 → bb.459 → bb.1022 → bb.461 → bb.464` reaching the reading restore while skipping both full stores (in `bb.458` and `bb.463`).
- **But it does not survive to the ISA.** In the final code every consumer regenerates the high half from the low one — `s_ashr_i32 s5, s2, 31` at `0x67A4`, `0x2238`, `0x69C0` — and the reloaded lane is only ever written back to another spill lane. `consumescan.py` reports it clean, and hand-reading all three sites agrees.
- **Not determined:** whether the intervening pass is *fixing* the disagreement deliberately or *coincidentally masking* it by rematerialising a value that happens to be cheap. That distinction matters for §11 and was not chased. It is the weakest joint in this report's argument.

The 9 of 131 objects `consumescan.py` does flag consume **exec masks** (`s_andn2_b64 s[8:9], exec`, `s_and_saveexec_b64`), not the sign-extended index the MIR finding is about. Those trace to `IMPLICIT_DEF` (258 in `bwd_kernel_fuse_02598` alone), i.e. legitimate undef. The two predicates fire on different values in the same object, so their conjunction is **not** established anywhere.

Also measured, because the companion's narrow detector reads 0 on Triton for a structural rather than a happy reason: `spillscan.py` looks for a tainted **buffer descriptor** base, and Triton's backend does not emit buffer descriptors for these accesses — it emits `global_load`/`global_store` with a 64-bit SGPR base. `globalscan.py` applies the same conjunction to the operand Triton actually uses, which has *no* bounds clamping at all. Result: **0 of 2896** on `bwd_kernel_dk_dv`.

---

## 7. Corrections to the companion report

### 7.1 `lostdef.py` does not separate the split defect from `IMPLICIT_DEF`

The companion's earlier §5 proposed the general predicate — *no `v_writelane_b32` may name an SGPR with no reaching definition* — as "the right one", and suggested it as a MachineVerifier check.

**Measured: it has a large false-positive class.** Of 2896 `bwd_kernel_dk_dv` configurations `lostdef.py` flags **1859**; at MachineIR level `mircfg.py` flags **0**. Traced by hand in `bwd_kernel_dk_dv_02894`: the two flagged writelanes spill vregs defined by `IMPLICIT_DEF` — `%10002` and `%10194`. `%10002` has both `IMPLICIT_DEF` definitions (1664B, 2384B) and real ones (`S_LOAD_DWORD_IMM`, `COPY`, `S_MOV_B32 0`), i.e. a post-SSA register genuinely undef on the paths the `IMPLICIT_DEF`s dominate. Spilling and reloading that is legal.

A verifier check phrased at writelane level would fire on every `IMPLICIT_DEF` spill in the tree. The check that discriminates is cat3: **no restore may read a spill-slot lane that no store into that slot defines on a path reaching it.** The companion's §5.1 now carries this retraction.

### 7.2 The `pop.tsv` double-count

`pop.tsv` lists **some** objects twice, once as `name-<hash>.hsaco` and once as `name.hsaco`. This is not a general property of the file — the three `flyc_*` families are hashed-only, which is why the headline numbers survive.

| family | bare rows | hashed rows | doubled? | real objects | real lostdef |
|---|---|---|---|---|---|
| `flyc_bwd_dkdv` | 0 | 216 | no | 216 | **114** |
| `flyc_bwd_dq` | 0 | 216 | no | 216 | 21 |
| `flyc_attn_fwd` | 0 | 216 | no | 216 | 19 |
| `bwd_preprocess` | 72 | 72 | **yes** | 72 | **6** |
| `bwd_preprocess_varlen` | 72 | 72 | **yes** | 72 | 0 |
| `debug_simulate_encoded_softmax` | 3 | 3 | **yes** | 3 | 0 |
| `_gemm_afp4wfp4_preshuffle_kernel` | 112 | 0 | no | 112 | 0 |

- **The headline `42` and `114` stand unchanged.**
- Corrected lostdef total: **160**, not 166.
- Corrected object total: **907** — `1054 − 72 − 72 − 3`. Correcting this took two passes; the first deduped the two obvious families and missed `debug_simulate_encoded_softmax`. Count bare and hashed rows per family rather than assuming the pattern.

All 6 real `bwd_preprocess` objects are **cat2** — `consumescan.py` reports `0 of 72 objects compute with a lost definition`. The sentence *"Twelve Triton-compiled objects carry the same defect"* was wrong twice: wrong count, and wrong claim.

### 7.3 A detector bug in the shared code

`undefscan.uses_of()` treats every operand after the first as a source. VOP3B instructions have a **second destination**: `v_mad_u64_u32 v[42:43], s[14:15], s30, v4, 0` writes `s[14:15]`. This inflates `spillscan.py` / `harmscan.py` / `consumescan.py`. Measured on the 131 cat3 candidates: 21 flagged before the fix, 9 after. `consumescan.py` carries a local `uses_of()` override for `v_mad_u64_u32`, `v_mad_i64_i32`, `v_add_co_u32`, `v_sub_co_u32`, `v_subrev_co_u32`, `v_div_scale_f32/f64`; shared `undefscan.py` was deliberately left untouched so the companion's numbers stay reproducible.

---

## 8. Relationship to the companion report and its §8 suspect

**Supported.** The companion's §1 mechanism is confirmed from a second, independent frontend, on three LLVM builds including the pinned one, with the copy explicitly tagged `lr-split`. Its §5 claim that *"this is not specific to one frontend"* is correct and now backed by a runnable reproducer. Its §11 lone-dword ISA evidence, which came from the now-weakened writelane predicate, is superseded by §3 here: a split copying `sub1` of an `sreg_64` and a spill covering `sub0_sub1`, with no `i64` anywhere in the lost value.

**Not supported by this work.** The step from mechanism to miscompile. Everything Triton produces here is cat2, or a cat3 a later pass rematerialises away. `flyc_bwd_dkdv` remains the **only** object measured to consume a lost definition — `consumescan.py` reports `1 of 2` on the two artifact objects, with the documented `s_mul_i32 s8, s10, s13` chain at `0x3F84`.

**Inferred (moderate confidence)** for why: the FlyDSL kernel's 21 `i64` strides merge into `S_LOAD_DWORDX16` / `sgpr_512` tuples, and all 21 partial-lane split bundles in that build were on `sgpr_512`. Triton's widest observed SGPR spill here is `S256`, and its 64-bit indices are cheap to rematerialise (`s_ashr_i32 x, y, 31`), so a later pass regenerates the missing half instead of consuming it. The exposure difference is tuple width and rematerialisability, not the allocator behaving differently.

**On the §8 suspect `83dca924c250`.** Triton's LLVM contains it (§1.1, inferred) and the defect reproduces there — consistent, and it adds a revision. It is *not* evidence that the commit **causes** the defect: no revision without it was tested. The discriminating experiment — build `llc` at `83dca924c250^` and at `83dca924c250`, run `repro-triton-gfx950.ll` through both, and check whether the `376B` copy acquires `sub0` — is now cheap, because the reproducer is a 170 KB `.ll` needing no Triton, MLIR, or wheel.

The companion's §8.2 methodological warning applies unchanged: check the mechanism in MachineIR, not the symptom.

---

## 9. Files in this directory

| file | what it is |
|---|---|
| `triton_kernels.py` | the reproducer kernel (§2.1) |
| `triton_repro.py` | compile-only Triton driver; explicit target, no GPU |
| `repro-triton-gfx950.ll.gz` | the emitted module — the standalone `llc` reproducer |
| `wrap_gpu_module.py` | wraps an `--import-llvm` result in a `gpu.module` + `#rocdl.target` |
| `drive2.py` | `drive.py` with a discoverable library path and `print-after` capture |
| `mirscan.py` | MachineIR: per-slot defined-vs-read lanes (union over stores; over-approximate) |
| **`mircfg.py`** | **MachineIR must-analysis over the CFG — the cat3 predicate** |
| `witness.py` | prints a concrete CFG path for a `mircfg.py` hit |
| `mirsweep.py` | runs `amdclang -print-after=greedy` + `mircfg` over a directory of `.ll` |
| `sweep.py` | compiles an AOTriton `Bare.compile` shard through `triton_repro` |
| `harmscan.py` | ISA: is the reload of a lost lane live? (superseded by `consumescan.py`) |
| `consumescan.py` | ISA: is the lost value read by something that is not a copy or a spill? |
| `globalscan.py` | `spillscan.py`'s conjunction against `global_*` saddr instead of a buffer descriptor |

Pre-existing and unmodified: `repro.py`, `drive.py`, `lostdef.py`, `spillscan.py`, `undefscan.py`, `pop.tsv`.

---

## 10. Reproduction envelope

Measured over dtype × `D_HEAD` × `PADDED_HEAD` (36 configurations, `BLOCK_M=128`):

| configuration | flagged |
|---|---|
| `D_HEAD=512`, `PADDED_HEAD=False`, fp16 / bf16 / fp32 | yes (3) |
| `D_HEAD=256`, `PADDED_HEAD=False`, fp32 | yes (1) |
| all other 32 combinations | no |

Those four are exactly the configurations AOTriton's production build flags for `bwd_preprocess`, minus the two `D_HEAD=80, PADDED_HEAD=True` cases that depend on the `composed_tensors` decomposition this reduction removes.

Attempts to reduce further **stopped reproducing**: dropping `PADDED_HEAD`/`hdim_vo` and the batch/head strides, or reducing to a single tensor, both give 0 flagged writelanes. Consistent with the companion's §4 — the defect churns under perturbation — and the reason the kernel is 50 lines rather than 15.

---

## 11. Is the mechanism-without-consequence worth reporting upstream on its own?

**File the bug, but make cat3 the claim, not cat2.** Framing cat2 as the bug would likely get the issue closed as invalid, and would take the real finding down with it.

**Measured, and decisive: cat2 occurs in 5,932 of 10,265 Triton modules — 58%.** A property holding of the majority of all compilations of a mainstream frontend is not a defect report; it is a description of how the allocator works. §3.1 says why it is benign: in cat2 the disagreement is between `SplitKit`'s lane mask and *the spiller's granularity*. `SI_SPILL_S64_SAVE` has no partial form, so storing a register whose live lanes are a subset of its width is the only thing it can do, and `SplitKit` narrowing the copy-in to the lanes live at that point is `getLiveLaneMaskAt` working correctly.

A maintainer will derive that 58% in five minutes and reasonably conclude the reporter has not distinguished a bug from normal spilling — a bad first impression that would attach to the FlyDSL evidence in the same issue.

**But the argument that this is latent wherever splitting happens is right — it just attaches to cat3.** *Nothing in `SplitEditor` prevents a copy-out asking for lanes the copy-in never provided* is a statement about mismatch **B**, and B is not hypothetical in Triton: **131 `bwd_kernel_fuse` configurations reach that MachineIR state**, hand-verified with a CFG witness path. So the intuition is supported by measurement — by a *stronger* measurement than cat2 offers. It needs the 131, not the 5,932.

**Recommended framing**, in descending order of what a maintainer can act on:

1. **The bug:** a restore reads a spill-slot lane that no store into that slot defines on a path reaching it. `SplitEditor` does not constrain a copy-out's mask to the lanes an earlier copy-in provided.
2. **Where it is observable:** `flyc_bwd_dkdv`, where the undefined pair becomes a dereferenced buffer-descriptor base. The only measured miscompile, and the headline.
3. **Evidence it is not one frontend's problem:** 131 Triton configurations reach the same MachineIR state. There the value is a sign-extended index a later pass rematerialises, so it does not reach memory — but the allocator produced the same invalid state, and whether that rematerialisation is deliberate repair or lucky coincidence was not determined (§6). That uncertainty is itself the argument: nothing in the reported code path *ensures* the repair.
4. **Suggested verifier check**, phrased at slot level (§7.1), with `mircfg.py` / `witness.py` as a reference implementation.
5. **cat2, mentioned only as context** — "the partial-copy shape is extremely common and benign; here is the number so you can rule it out as the thing being reported."

**Confidence.** High that cat2 alone should not be the claim. Moderate that the 131 cat3 cases will persuade — they do not reach memory, and a maintainer may answer that a value LLVM can rematerialise was never really lost. If that comes back, the follow-up is the §8 bisect, now cheap.

**A judgement about people rather than compilers:** if the goal is to get `83dca924c250` looked at by the person who wrote it, a short issue built on the FlyDSL disassembly plus the §3 `lr-split` MachineIR — with the Triton `.ll` attached as "a second frontend hitting the same code path" — will likely get further than any completeness argument.
