# gfx950 FMHA lore: the hazards that bite when you widen the kernel

`flash_attn_gfx950.py` and its parity port were written for head_dim 64 and
128. Every attempt to serve another width has run into the same *class* of
problem: **the compiler does not know it must insert something, and the result
is a wrong-but-finite answer.** No NaN, no fault, no out-of-range address --
just numbers that are a few percent off, or occasionally garbage.

Two instances are documented below, both found the hard way. Read the
"recognising one" section first; it is the part that generalises.

---

## Recognising one

A hazard of this class has a signature that separates it from a logic bug:

- **A semantically-null change fixes it.** Enabling a mask that masks nothing,
  copying a value through an identity, adding an `s_nop`. If the answer moves,
  the arithmetic was never the problem.
- **The instruction counts are identical.** Same MFMA count, same DS reads,
  same everything -- only the register assignment differs. Check this before
  theorising.
- **It is deterministic.** Run-to-run identical. A race would not be.
- **It appears at one width.** Register allocation changes with `K_STEPS_QK` /
  `D_CHUNKS`, so a latent hazard surfaces at whichever shape happens to place
  two registers in conflict. The other widths are lucky, not correct.

### The discriminator ladder

Apply these in order. Which one fixes it tells you the class, and each is a few
minutes' work:

| probe | what it changes | if it fixes it |
|---|---|---|
| `rocdl.sched_barrier(0)` | machine-scheduler ordering | instruction motion, late |
| empty asm, `~{memory}` clobber | IR-level ordering; no registers | instruction motion, early (LICM/sink) |
| identity asm on the value (`"=v,0"`) | forces a register copy | register allocation |
| `amdgpu-snop-padding=1` (LLVM option) | a wait state before every instruction | **hardware wait state** |

`amdgpu-snop-padding` is the sharpest instrument here and the cheapest to try.
It is far too blunt to ship -- it costs ~3x at head_dim 128 -- but a one-line
answer to "is this a hazard at all?" is worth a lot.

Note the asymmetry: an **identity asm emits no instruction**. It pins
scheduling and liveness, and it changes which registers the allocator picks,
but it supplies *no* wait state. So "the anchor fixed it" means *registers*,
not *timing*. Do not conclude you have found a wait-state bug from that alone.

---

## Hazard 1: inline asm hides memory ops from `SIInsertWaitcnts`

**Symptom.** Non-deterministic NaN at head_dim 192 and 256; head_dim 64 and 128
fine. No amount of `s_barrier` or `s_waitcnt` at the DSL level helped.

**Cause.** `_ds_read_tr16_b64_imm` emitted `ds_read_b64_tr_b16` as
`llvm.inline_asm`. `SIInsertWaitcnts` discovers outstanding LDS traffic by
scanning MIR for DS instructions; an inline asm is opaque to it, and a
`~{memory}` clobber is **not** an lgkm event. So the backend did not know a
read was in flight and inserted no `s_waitcnt lgkmcnt` before uses of the
result.

That was sound only while nothing read the destination before the kernel's own
cluster-boundary wait. Above head_dim 128 the kernel exceeds the 256
architectural-VGPR cap, the allocator starts using AGPRs, and it places
`v_accvgpr_write` copies of the destination *immediately after the read and
ahead of the wait* -- 22 unwaited uses at 192, 160 at 256, against 0 at 64 and
128.

**Rule.** *Never* emit a memory operation as inline asm. Use the ROCDL op
(`rocdl.ds_read_tr16_b64`, `rocdl.ds_read_tr8_b64`) so the backend can track
the dependency. The result type must be `vector<4xf16>` -- the intrinsic is
typed that way and `Cannot select`s on `vector<2xi32>`; bitcast afterwards.

**The sting in the tail.** Switching to the op cost 7-10%, and the cause was
*not* the waits LLVM then inserted. It was **five extra `s_waitcnt vmcnt(0)`**:
the backend treats `buffer_load ... lds` as an LDS-writing VMEM op and drains
*all* outstanding DMA before any DS read that may alias one in flight. The K
path's `ds_read_b128` provokes no such drain because it carries
`alias_scopes`/`noalias_scopes` from the `LDS_SCOPE_NAMES` scheme.
`ROCDL_LDS_Read_Tr_IntrOp` declares `requiresAliasAnalysis`, so tagging the V
reads the same way removes all five.

**Rule.** A ROCDL LDS read needs alias scopes, or you trade a correctness bug
for a 26%-of-runtime drain.

---

## Hazard 2: VALU -> MFMA wait states the compiler under-counts

**Symptom.** head_dim 96 computes a wrong answer -- ~9% of elements, scattered
in row and column, deterministic, ~2% low. Every other multiple of 32 from 32
to 256 is correct at the same granule, including 192, whose shape is exactly
twice 96's.

**What it is not**, all measured:

- not staging -- the probe reports 0/6144 wrong for K, V and Q on both buffers;
- not masking -- a *no-op* mask at an exact width is bit-identical to unpadded
  at 32/64/128/160/224, and differs only at 96, where it is correct;
- not the softmax -- `return_lse` against `torch.logsumexp` matches to 3e-4
  with zero deviating rows, so `l` and `m` are exactly right;
- not instruction motion -- neither `sched_barrier(0)` nor an IR `~{memory}`
  fence changes anything;
- not scheduling knobs -- `waves_per_eu`, `setprio`, `stagger` and
  `lazy_rescale` all leave the *identical* error.

**What it is.** `amdgpu-snop-padding=1` fixes it, which makes it a hardware
wait state. The offending pair, from the failing ISA:

```
v_perm_b32 v119, v157, v158, s14      <- VALU writes v119 (P bf16 packing)
s_nop 1                                <- LLVM inserted 2 wait states
v_mfma_f32_32x32x16_bf16 v[48:63], v[220:223], v[116:119], v[48:63]
                                          SrcB spans v119
```

CDNA requires software-inserted wait states between a VALU write and an XDL
read of that register (ISA ch. 7.6, "Dependency Resolution: Required
Independent Instructions"). LLVM recognised the hazard and emitted `s_nop 1`;
this needs more. head_dim 96 is simply the width whose allocation puts a
freshly-packed P register inside an MFMA's SrcB range.

**Status: FIXED.** `ParityGemmHelper.qk` now emits `s_nop 1` after the QK
burst. It costs +1.6% at head_dim 64 and +0.8% at 128 (both inside noise) and
buys head_dim 96 at 932 TF against 579 for the padded 128 tile.

The location came from bisection, not from identifying the pair: a wait state
after `qk`, `exp2` or `reduce_sum` each fix it; after `cast_p`,
`lazy_rescale_o`, `load_k`, `load_v`, `reduce_max`, `sub_m` or `pv_step_k` none
do. So the producer is at or before the QK MFMAs and the consumer is before the
P cast. **The exact instruction pair is still unidentified** -- scans of every
documented hazard class found no difference between a broken and a working
build -- so this is a correct fix with an incomplete explanation.

Two earlier attempts, for the record:

- Anchoring the K packs fixes it (579 -> 944 TF) but pins all `2*K_STEPS_QK`
  packs live at once -- 28 packs, 112 VGPRs at head_dim 224 -- and costs
  **-59% there**. It works by shifting registers, not by adding time.
- Putting `s_nop 0` inside `_anchor_v_p`'s asm body **does not fix it**. The
  production anchor pins the *concatenated* P vector and then shuffles the
  packs out, so the `v_perm_b32` is emitted after the anchor and the nop lands
  in the wrong place.

head_dim 96 is a normal rung again.

---

## Method notes, earned the hard way

- **Always run a known-good control.** Three wrong conclusions in this work
  came from a probe with no control: "family B is broken" (the harness was
  broken), a bad V-layout expectation, and a wrong Q buffer size. Build the new
  configuration at head_dim 64 or 128 first, where the answer is known.
- **Always test several shapes.** "The wide body is correct at 96" came from
  one small shape; at `B=4 H=8 S=4096` it is wrong. Use at least
  `(1,8,512)`, `(2,8,1024)`, `(4,8,4096)` and one ragged case.
- **Structured inputs localise far better than random ones.** Uniform Q and K
  makes S constant and isolates the GEMMs; V held constant across tokens makes
  `O[r][d] = d` for *any* weights and isolates the normaliser. Both were
  decisive here where a random-input error map only said "9% of elements".
  Beware the converse: uniform inputs cannot detect a wrong cross-lane
  reduction or a token permutation, because every value is equal.
- **`FLYDSL_RUNTIME_ENABLE_CACHE=0`, always.** `flash_attn_utils.py` and the
  parity helpers are not part of the traced closure, so the JIT cache key does
  not see edits to them. This produced 8 phantom test failures in one sitting.
- **Cap builds at ~8 minutes.** A configuration whose register allocator runs
  away is telling you it does not fit; head_dim 512 at `d_stages` 4 or 8 never
  finished. Treat a slow build as a result, not an inconvenience.
- **Check which GPU you are on, and its temperature.** A shared board at 100 C
  junction reported 218 TF where a cool one reports 1101 -- and that nearly
  produced the conclusion that a single `s_nop` cost 80%. Any perf result that
  looks structurally impossible is a machine-state result until proven
  otherwise; `rocm-smi --showtemp` first, re-measure second.
- **Check `agpr_count` alongside spills.** On this kernel the allocator either
  uses the AGPR file for the O accumulator or abandons it entirely and spills
  to scratch, and the difference is 1.4-1.6x. Four waves keeps it; eight loses
  it.
