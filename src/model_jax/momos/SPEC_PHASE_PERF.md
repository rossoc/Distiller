# Performance — making MoMos fast, not just small

**Status: exploratory. Not scheduled, not gated, nothing here blocks any other
phase.** This is a side topic captured so the reasoning is not lost. Read
`SPEC.md` §2 (the compression ledger) first; the whole point of this document is
that §2 measures the wrong half for performance purposes.

---

## 1. The problem: MoMos today is smaller but slower

`train_step.py:220` is the whole issue:

```python
gathered = state.motifs[state.mosaic]   # (M, S) — the ENTIRE dense model
```

Every weight in the model is materialised as one dense fp32 array, per step,
before the forward pass. `g_blocks` from the backward is a second one of the
same size. So peak, per step:

| | bytes/weight at peak |
|---|---|
| dense fp32 training (param + m + v + grad) | 16 |
| **MoMos today** (mosaic ~1 + `gathered` 4 + `g_blocks` 4) | **~9** |
| MoMos with a fused kernel | **~1** |

The persistent ledger reports 12–24×. The *peak* advantage is ~1.8×, because
the transient is 8 bytes/weight against a persistent 1. `SPEC.md` §2 already
concedes this — *"the transiently-gathered weights do not compress. Report
both"* — but the gate only ever prints the persistent half, so every number
reported so far is the flattering one. At N = 1.63M the transient is 6.5 MB and
invisible. At 7B parameters it is 28 GB per array, and it destroys the claim.

On speed, MoMos is currently **slower than dense**: the same matmuls, plus a
full-model gather, plus a `segment_sum` scatter over all M blocks every step.

**So fusion is not an optimisation. It is what would make the memory claim true
at scale.**

---

## 2. What fusion buys, and what it does not

Identical multiply-accumulates either way. **The win is bandwidth, not FLOPs.**
Weight traffic per weight is `dtype/S`:

| config | index traffic | vs dense fp32 |
|---|---|---|
| S=1, K=256 (uint8) | 1.0 B/wt | 4× less |
| S=2, K=4096 (uint16) | 1.0 B/wt | 4× less |
| S=4, K=4096 (uint16) | 0.5 B/wt | 8× less |
| S=4, K=256 (uint8) | 0.25 B/wt | **16× less** |

Which splits the regimes cleanly:

- **Inference decode** (memory-bound, batch 1) approaches that ratio. This is
  the prize.
- **Training and prefill** (compute-bound at large batch) gain little speed, but
  the 9 → 1 B/wt peak-memory drop is what lets a bigger model train in the same
  budget — which *is* `SPEC.md` §8's Phase E thesis.

MoMos with S>1 is exactly **vector quantization**; `S` is the VQ vector
dimension. AQLM, QuIP#, GPTVQ and Marlin-style kernels are direct prior art to
borrow from rather than reinvent.

---

## 3. What we can reuse — checked, not assumed

Verified in this environment (JAX 0.11.1):

- **Pallas is available**, including the Triton backend
  (`jax.experimental.pallas`, `jax.experimental.pallas.triton`). **It ships with
  JAX — no new dependency.** This is the "already in the library" answer.
- **No quantization library is installed** (no AQT, no qwix). AQT exists for
  JAX but does scalar int8/int4 with scale factors, *not* codebook lookup, so it
  would not help here even if added.
- **There is no off-the-shelf VQ-GEMM in JAX, Flax or optax.** This has to be
  written.

**The blocker that forces a custom kernel:** XLA will not fuse a gather into a
`dot_general` operand. There is no flag. `motifs[mosaic] @ x` always
materialises the dense weight tensor first. Pure JAX cannot express this;
Pallas (Triton on GPU, Mosaic on TPU) or an XLA custom call is required.

**Local constraint:** `jax.devices()` on this machine is CPU-only. Pallas GPU
kernels cannot be benchmarked here. Any work in §6 stages B/C needs GPU access.

---

## 4. Fixing K at 256 — and what that costs in block size

The proposal: drive training toward **K = 256** regardless of where it starts,
because 256 is the largest dictionary addressable by a `uint8` index. To recover
capacity lost by capping K, raise the block size `S`.

That trade is exact and unforgiving: with a fixed K, the information rate is

```
bits per weight = log2(K) / S = 8 / S        (K = 256)
```

Every doubling of `S` halves the bits available to describe each weight.

### 4.1 The limits, measured

At K = 256, N = 1,633,921 (the gate's model):

| S | bits/wt | B/wt | vs dense | codebook SMEM | binding? |
|---|---|---|---|---|---|
| 1 | 8.00 | 1.0019 | 12.0× | 1 KB | |
| 2 | 4.00 | 0.5038 | 23.8× | 2 KB | |
| **4** | **2.00** | **0.2575** | **46.6×** | 4 KB | **quality edge** |
| 8 | 1.00 | 0.1400 | 85.7× | 8 KB | beyond published VQ |
| 16 | 0.50 | 0.0926 | 129.6× | 16 KB | |
| 23 | 0.35 | 0.0867 | **138.4×** | 23 KB | ledger optimum |
| 32 | 0.25 | 0.0914 | 131.3× | 32 KB | |
| 64 | 0.12 | 0.1360 | 88.3× | 64 KB | SMEM edge |
| 128 | 0.06 | 0.2485 | 48.3× | 128 KB | SMEM limit |

Four separate ceilings, and only one of them actually binds:

| constraint | limit on S at K=256 | binds? |
|---|---|---|
| **Rate–distortion (quality)** | **S ≈ 4, maybe 8** | **yes — this is the answer** |
| Code validation (`state.py:94`, `S in (1,2,4)`) | S ≤ 4 | yes, today — an implementation cap, liftable |
| Codebook SMEM residency (fp32, 164 KB/SM) | S ≤ 128 | no |
| Regime cap `K ≤ M/10` (`SPEC.md` §8) | S ≤ 638 at this N | no |
| Ledger optimum (`d/dS` of `1/S + 12·K·S/N`) | S ≈ 23 here, ≈ 1510 at 7B | not a limit — an optimum quality never reaches |

**Answer: the maximum usable block size at K=256 is about 4, possibly 8.**

Note the ledger optimum is `S* = sqrt(N / (12·K))`, which grows with model size —
23 at the toy scale, ~1510 at 7B. It is irrelevant in both cases, because
quality collapses two orders of magnitude earlier. **The binding constraint is
information-theoretic, not engineering.** No kernel, no SMEM budget and no
regime rule is what stops you; 8/S bits per weight is.

For calibration against published work: S=4 gives 2 bits/weight, which is the
aggressive end of what AQLM and QuIP# achieve at usable quality. S=8 gives 1
bit/weight, beyond any published lossless result. Vector quantization does beat
scalar at equal rate, but the space-filling gain is bounded at ~0.25 bits per
dimension asymptotically — it buys a constant, not an order of magnitude.

### 4.2 The measured evidence already says S=4/K=256 is a good point

From the Phase D gate (`static` arm, 300 steps):

| cell | static loss | compression |
|---|---|---|
| S=4, K=256 | 0.00347 | 46.6× |
| S=4, K=4096 | 0.00273 | 19.3× |

**1.27× the loss for 2.4× the compression.** That is already the best
compression/quality trade in the grid, and it is reached without any of the
machinery in this document. It is the natural target configuration.

### 4.3 Caution: K=256 is where lifecycle measured *worst*

The one place the grid tested merging into a 256-motif dictionary, it hurt:

| cell | lifecycle vs single |
|---|---|
| S=1, K=256 | **2.19× worse** |
| S=4, K=256 | **2.00× worse** |
| S=2, K=256 | 0.93× |

Squeezing an already-tight dictionary removed real capacity rather than
redundancy. Any scheme that drives toward 256 live motifs is steering at the
region where the only evidence available says this goes badly. That is not a
reason to abandon it — those cells merged at fixed S, and the proposal here is
to raise S in compensation, which is untested — but it *is* a reason to measure
the S-compensated version before building anything on top of it.

---

## 5. Getting to 256: loss term, or just a target?

Two ways to drive the dictionary toward a fixed count.

### 5.1 Option A — reuse Phase D, change the floor to a target

Phase D already merges, monotonically, with tested invariants. Replacing
`min_live_frac` with `target_live: 256` — merge each pass until `live == 256`,
then stop — needs **no new loss term, no gradient interference, and no new
failure modes**. It reuses code that already has nine invariants under test.

This should be tried first. It is nearly free.

### 5.2 Option B — a loss term that pulls motifs together

The argument for Option B is real and Option A does not address it: merging only
collapses motifs that *happen* to be close. A loss term would make motifs
*become* mergeable, so that merging is near-lossless rather than destructive —
which is precisely the failure §4.3 measured.

Form: a clustering/commitment prior pulling each motif toward its nearest
neighbour, annealed, and switched off once `live == target`.

**Three risks to design against, all of which have already bitten this project:**

1. **Degenerate collapse.** A naive `sum_k min_j ||m_k − m_j||²` is minimised by
   putting every motif at one point. It needs a repulsion term, a target count,
   or a per-component floor.
2. **Fighting the hard mechanism.** A soft pull toward merging plus a hard merge
   is the same shape as the revive/merge fight that made Phase C2's lifecycle
   9.33× worse while reporting `live == K`
   (`SPEC_PHASE_D_DEFECTS.md` §2). Any such pair needs an explicit argument for
   why they compose, and a two-pass test.
3. **Optimising a proxy.** The term minimises motif proximity, not task loss at
   a fixed budget. `SPEC.md` §7's `loss_delta_around_swap` exists because the
   project already learned that quantisation error and task loss are correlated
   but not identical. The same caveat applies here, more strongly.

**Recommendation: Option A first.** Only reach for Option B if A demonstrably
fails *because* the motifs are not mergeable — and prove that is the reason
before adding a term to the loss.

---

## 6. Suggested order, if this is ever scheduled

**A. Pure JAX, no kernels.** Make the gather per-tensor and lazy so only one
tensor's dense weights are live at a time; peak transient drops from `N` to the
largest single tensor. Gather in bf16 while there. **Add a transient
bytes/weight column, and wall-clock per step, to the gate table** — without
them, every later stage is unfalsifiable, and the persistent-only ledger is now
the misleading half.

**B. Pallas forward VQ-GEMM.** Codebook in SMEM, indices streamed, no dense
materialisation. Measure decode tokens/s against dense. Known quantity — this is
what quantized-inference kernels already do. Needs GPU access (§3).

**C. Pallas backward.** Scatter-add `g_blocks` into a `K×S` SMEM accumulator,
then atomically to global, avoiding the `g_blocks` materialisation entirely.
Quantized *training* kernels are far less mature than inference ones, so this is
the research-risk step. Do it only after B proves the forward.

**D. Dictionary compaction after lifecycle.** Phase D measured S=1/K=4096 down
to 2139 live motifs, but `K` stays 4096 and the codebook stays 64 KB — only the
`active` mask changes. Renumbering live motifs to `0..L-1` and shrinking the
arrays makes lifecycle a *performance* feature: the codebook shrinks (better
SMEM fit, more room for tiles), and **if live ever drops below 256 the mosaic
dtype drops `uint16 → uint8`, halving index bandwidth outright.** That is a
direct argument for pushing `merge_quantile` harder than loss alone justifies —
and it is the mechanism that connects Phase D to this document.

---

## 7. Open question: global vs per-layer dictionary

`SPEC.md`'s opening premise is "one small, global, learned dictionary" — good
for compression and for cross-layer motif sharing. For SMEM residency, per-layer
codebooks with smaller `K_l` fit more easily and lift the `K·S` ceiling.

The index *layout* is not the issue: `tiling.ParamLayout` already carries
per-tensor offsets, so `mosaic[start:stop]` is a free view and each layer's GEMM
already reads a contiguous index block. Nothing needs restructuring there.

The real trade is dictionary scope, and it is unmeasured. At K=256 the codebook
is 1–4 KB and the question is moot; it only becomes live if K is pushed back up.
