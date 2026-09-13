# Phase C (revised) — Drift-Accumulated Reassignment

Supersedes the single-step swap criterion currently in
`train_step._propose_swaps`. Read `SPEC.md` first for the framework, the
compression ledger (§2) and the regime (§1). This document is self-contained for
the reassignment mechanism only.

**Keep unchanged:** `tiling.py`, `state.py`'s ledger, `maintenance.py`'s exact
`neighbour_graph`, and the property that a disabled reassignment config
reproduces Phase B bit-for-bit.

---

## 1. The problem, and the shape of the fix

A single step's gradient is a noisy estimate of where a block wants its weights
to go, so a swap decision made from one step is mostly noise. The fix is to
**accumulate each block's optimisation trajectory over a window** and decide
periodically, outside the jitted hot path.

That core idea is sound. Four things around it are not, and each was measured
rather than argued — do not reintroduce them:

| # | Naive approach | Measured consequence | Required instead |
|---|---|---|---|
| 1 | `accumulated_drift` at `(M, S)` fp32 | 4 bytes/weight — compression collapses from 12–24× to **2.4×** | Rotating cohort, §3 |
| 2 | High-water mark `max(mags) × 1.05` per window | Thresholds ratchet up while drift shrinks; **0 blocks eligible from window 2** | Quantile threshold, §5.4 |
| 3 | Top-K-by-magnitude for directions | Mean pairwise cosine **+0.98** — 16 slots holding one direction | Spherical k-means, §5.2 |
| 4 | Null slot as a zero row in the codebook | Similarity to zero is always 0.0; **0 of 20,000** blocks ever assigned | Explicit `max_sim < tau`, §5.3 |

A fifth, smaller: `adaptive_scale = lr / (sqrt(nu) + eps)` omits Adam's bias
correction. At `t=1`, `1 − β₂ = 0.001`, so `sqrt(nu)` is ~31× too small and the
first window's drift is ~31× inflated. Use `nu_hat`, §4.2.

---

## 2. Config

Add to `MosaicConfig`:

```python
cohort_frac:     float = 0.01    # fraction of M accumulating drift at a time
eval_window:     int   = 50      # steps per window
k_delta:         int   = 16      # codebook directions (see the S=1 clamp below)
dir_ema:         float = 0.9     # codebook direction EMA
tau_sim:         float = 0.5     # min cosine to count as "matches a direction"
jump_frac_init:  float = 0.05    # fraction of cohort eligible to jump, window 0
jump_frac_min:   float = 0.005
jump_frac_decay: float = 0.98    # per window
```

`cohort_frac = 0.0` disables reassignment entirely and must reproduce Phase B
bit-for-bit — the same escape hatch `subset_size=0` provides today.

**Validation, at construction:**
- `S == 1` ⇒ clamp `k_delta` to 2 and log it. Unit vectors in 1-D are `{+1, −1}`;
  more slots are meaningless (same degeneracy that ruled out LSH at small S).
- `k_delta >= 2`, `0 < tau_sim < 1`, `0 <= cohort_frac <= 1`.

---

## 3. The cohort — this is what keeps the compression

Only `C = max(1, int(cohort_frac * M))` blocks accumulate drift at any time.
Cohorts rotate so every block is visited every `ceil(M / C)` windows.

**Do not store a permutation.** An `(M,)` int32 index array costs `4/S`
bytes/weight — at S=2 that is 2 bytes/weight, worse than the mosaic itself.
Generate the cohort arithmetically from a multiplicative bijection:

```python
def cohort_indices(window: int, M: int, C: int, A: int, B: int) -> jnp.ndarray:
    """Layout-decorrelated block indices for this window.

    `A` is coprime to `M`, so `x -> (A*x + B) mod M` is a bijection: consecutive
    windows draw disjoint input ranges and therefore disjoint cohorts, and the
    whole of `M` is covered in ceil(M/C) windows. Computing it costs nothing and
    stores nothing, which is the point — an (M,) permutation array would cost
    more per weight than the mosaic it is meant to serve.
    """
    base = (jnp.arange(C) + window * C) % M
    return (A * base + B) % M
```

Pick `A` once at init: the smallest odd integer ≥ `0.618 * M` with
`math.gcd(A, M) == 1`. Store `A`, `B` as ints on the config (not device arrays).

**Two verified caveats — the properties are weaker than they first look:**

*Disjointness requires `C` to divide `M`.* When it does not, the final window of
a sweep wraps past `M` and reuses indices from window 0. Measured: `M=9973,
C=99` is **not** disjoint, while `M=1000/100`, `M=100000/1000` and
`M=815000/8150` are. Coverage stays 100% either way and a block accumulating
drift in two windows of one sweep is harmless, not a correctness bug — so either
round `C` down to a divisor of `M`, or keep the overlap and assert **coverage**
rather than strict disjointness in §7.2.

*Decorrelation only holds at realistic `M`.* Measured fraction of
adjacent-index pairs within one cohort: **0.00%** at M = 9,973 / 100,000 /
815,000, but **78.8%** at M = 1,000. The golden-ratio multiplier mixes poorly
when `C/M` is large and `M` small. Every M in this project's regime is ≥ 10⁵, so
this is a non-issue in practice — but do not use this scheme in a unit test at
M ≈ 10³ and conclude it is broken.

**Memory:** `drift` is `(C, S)` fp32 = `4 · cohort_frac` bytes/weight. At
`cohort_frac=0.01` that is **0.04 bytes/weight** — against a mosaic of 0.5–1.0.
Assert this in a test (§7.1): the drift buffer's shape must be `(C, S)`, never
`(M, S)`.

---

## 4. Micro-loop (inside `train_step`)

### 4.1 What is added

After step A's block gradients and before/alongside step C's motif update:

```python
cohort = cohort_indices(window, M, C, cfg.A, cfg.B)     # (C,)
scale  = adaptive_scale(opt_state, state.mosaic[cohort], cfg)   # (C, S)
drift  = drift - scale * g_blocks[cohort]                       # (C, S)
```

`state.mosaic[cohort]` is constant within a window (swaps only happen at window
end), so the motif each block is tethered to does not change mid-accumulation.

### 4.2 `adaptive_scale` — with bias correction

```python
def adaptive_scale(opt_state, motif_ids, cfg):
    """Adam's own per-motif step scale, applied to the block's gradient.

    Tethering the drift to the motif's second moment is deliberate: a block
    should not accumulate huge drift merely because its motif sits in a noisy,
    high-gradient region. `nu` is read from the optimiser rather than
    reconstructed so the scale tracks whatever the optimiser actually does.

    The bias correction is not optional. At t=1, 1 - beta2 = 0.001, so raw
    sqrt(nu) is ~31x too small and the first window's drift is ~31x inflated.
    """
    nu, count = _find_adam_nu_and_count(opt_state)        # see below
    nu_hat = nu / (1.0 - cfg.beta2 ** count.astype(jnp.float32))
    return cfg.base_lr / (jnp.sqrt(nu_hat[motif_ids]) + 1e-8)
```

`_find_adam_nu_and_count` must locate `ScaleByAdamState` by **traversing the
optimiser pytree and matching on type**, not by a hardcoded path — the state
nests differently under `optax.chain`, `MultiSteps`, etc. If no Adam state is
found, fall back to `cfg.base_lr` as a constant scale and log once. Write a test
for both the plain-`adam` and the `chain(clip_by_global_norm, adam)` layouts.

### 4.3 Window boundary

`train_step` stays jitted and shape-stable. Window-end work runs in the **macro
loop** (host-driven, outside `jax.jit`), like `maintenance.py` already does.
`train_step` returns the updated `drift`; the caller decides when to call §5.

---

## 5. Macro-loop (end of each window)

Input: `drift (C, S)`, `cohort (C,)`, `motifs (K, S)`, `mosaic`, `active`,
`graph`, `codebook_dirs (k_delta, S)`.

### 5.1 Magnitudes and normalised directions

```python
mags = jnp.linalg.norm(drift, axis=-1)                       # (C,)
nd   = drift / jnp.maximum(mags, 1e-8)[:, None]              # (C, S)
```

### 5.2 Codebook update — spherical k-means, not top-K

Persistent `codebook_dirs` are centroids on the unit sphere, refined each window
from the cohort's directions:

```python
for _ in range(cfg.kmeans_iters):          # 3-5 is plenty; it is EMA'd anyway
    assign = jnp.argmax(nd @ centroids.T, axis=-1)           # (C,)
    sums   = jax.ops.segment_sum(nd, assign, cfg.k_delta)    # (k_delta, S)
    counts = jax.ops.segment_sum(jnp.ones(C), assign, cfg.k_delta)
    new    = sums / jnp.maximum(counts, 1)[:, None]
    new    = new / (jnp.linalg.norm(new, axis=-1, keepdims=True) + 1e-8)
    centroids = jnp.where(counts[:, None] > 0, new, centroids)   # keep empty slots
```

Then EMA against the persistent codebook, **only for slots that received
assignments** — an empty slot must not be dragged toward nothing:

```python
blended = cfg.dir_ema * codebook_dirs + (1 - cfg.dir_ema) * centroids
blended = blended / (jnp.linalg.norm(blended, axis=-1, keepdims=True) + 1e-8)
codebook_dirs = jnp.where(counts[:, None] > 0, blended, codebook_dirs)
```

This fixes the correspondence problem in one move: a slot is updated from the
blocks assigned *to it*, so slot identity is stable across windows. EMA-ing slot
`d` against "whatever the d-th largest-magnitude block did this window" has no
such identity and must not be used.

Init `codebook_dirs`: k-means++ seeding from the first window's `nd`, or random
unit vectors if that is simpler — it is EMA-refined from window 1 either way.

### 5.3 Assignment, with a real null test

```python
sim      = nd @ codebook_dirs.T                    # (C, k_delta)
best_dir = jnp.argmax(sim, axis=-1)
matched  = jnp.max(sim, axis=-1) >= cfg.tau_sim    # explicit "matches something"
```

Do **not** encode "no direction" as a zero row in the codebook. Its similarity
is identically 0.0, so it wins only when every real direction scores negative —
measured at 0 of 20,000 blocks. `tau_sim` is the actual test.

### 5.4 Eligibility — quantile, not high-water mark

```python
projected = jnp.sum(drift * codebook_dirs[best_dir], axis=-1)     # (C,)
frac      = max(cfg.jump_frac_min, cfg.jump_frac_init * cfg.jump_frac_decay ** window)
thr       = jnp.quantile(projected, 1.0 - frac)
can_jump  = matched & (projected > thr)
```

A quantile self-scales as gradients shrink, so a bounded fraction of the cohort
stays eligible throughout training instead of the ratchet's hard freeze. Cooling
is then an explicit annealed `frac`, on a schedule you control, rather than an
emergent `1.05^n` that reaches "nothing can ever jump" in about two windows.

### 5.5 Target and neighbour test

```python
cur     = mosaic[cohort].astype(jnp.int32)
target  = motifs[cur] + codebook_dirs[best_dir] * projected[:, None]
nbrs    = graph[cur]                                             # (C, n_neighbors)
d_nbr   = jnp.sum((target[:, None, :] - motifs[nbrs]) ** 2, -1)
d_nbr   = jnp.where(active[nbrs], d_nbr, jnp.inf)                # never onto a dead motif
d_cur   = jnp.sum((target - motifs[cur]) ** 2, -1)               # == projected**2
best    = jnp.argmin(d_nbr, -1)
cand    = jnp.take_along_axis(nbrs, best[:, None], 1)[:, 0]
swap    = can_jump & (jnp.min(d_nbr, -1) < d_cur)
mosaic  = mosaic.at[cohort].set(jnp.where(swap, cand, cur).astype(mosaic.dtype))
```

No extra margin here: §5.4's quantile is the hysteresis.

**Memory note:** `motifs[nbrs]` is `(C, n_neighbors, S)`. At C = 1% of M that is
small, but chunk it if `C * n_neighbors * S` exceeds ~10⁸.

### 5.6 Reset

`drift = jnp.zeros((C, S))`, advance `window`, recompute `cohort` on next use.

---

## 6. Metrics to log every window

- `swap_rate` over the cohort, **and** `jump_eligible_rate` before the neighbour
  test — these separate "the threshold blocked it" from "no neighbour was better"
- `codebook_spread`: mean off-diagonal `|cos|` between `codebook_dirs`. Should
  sit well below 0.9; near 1.0 means §5.2 has collapsed to one direction
- `matched_rate`: fraction passing `tau_sim`
- `live_motifs`, `usage_entropy` (existing)
- `bytes_per_weight` including the drift buffer, so the §3 cost stays visible

---

## 7. Tests — the invariants that must hold

1. **Drift buffer shape is `(C, S)`, never `(M, S)`.** Assert directly, and
   assert measured bytes/weight ≤ mosaic + `4 * cohort_frac`.
2. **Cohorts cover all of `M`** in `ceil(M/C)` windows, and `gcd(A, M) == 1`.
   Assert strict cross-window disjointness only when `C` divides `M` — see §3's
   caveat; test at `M >= 10**4`, not at `M ~ 10**3` where the mixing degrades.
3. **Drift accumulation matches a reference**: a hand-rolled loop summing
   `-scale * g` over a window, to `atol=1e-6`.
4. **Bias correction**: at `count=1`, `adaptive_scale` matches Adam's own
   effective step to `atol=1e-6`. Without correction it is ~31× off — assert the
   corrected value, so a regression is caught.
5. **Adam state discovery** works under both `optax.adam(...)` and
   `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(...))`.
6. **Codebook spread beats top-K.** Generate drift as a dominant mode plus noise;
   assert k-means mean pairwise cosine < 0.7 while top-K-by-magnitude > 0.9.
   This is the measured property being defended.
7. **Null test fires**: blocks whose drift is near-orthogonal to every codebook
   direction must have `matched == False` and must not swap.
8. **Quantile eligibility**: exactly `round(frac * C)` blocks pass the threshold
   (before the neighbour test), for several `frac`.
9. **No swap onto an inactive motif**, under a graph where the nearest neighbour
   is dead.
10. **Mosaic dtype preserved** after reassignment.
11. **Per-block independence**: two blocks on the same motif with different drift
    must be able to reach different decisions. *This is the defect that made the
    previous gate uninformative — it must not regress.*
12. **`cohort_frac = 0.0` reproduces Phase B bit-for-bit.**
13. **S=1 clamps `k_delta` to 2.**

---

## 8. The gate

Same regime requirements as `SPEC.md` §8's Phase C blockquote — they are not
optional and the previous run was invalid for ignoring them:

- Model with **≥1M mosaicked values**: `toy_tasks.build(..., d_model=256,
  num_layers=4)` ≈ 1.64M params, ≈222 ms/step on CPU.
- **Skip any cell with `K >= M/10`** — degenerate, not a failure. Print as
  SKIPPED with the reason.
- **Print K/M** so the regime is visible.

**Make it a three-way comparison**, not two:

| arm | what it is |
|---|---|
| `static` | Phase B, no reassignment |
| `single` | current Phase C, per-step swap on `g_blocks` |
| `drift` | this document |

Two-way against `static` only tells you reassignment helps; three-way tells you
whether the drift machinery earns its complexity over the far simpler criterion
that already exists. If `drift ≈ single`, keep `single` and delete this.

**A negative result is a valid outcome.** Do not tune until it passes and then
report success. Report what you measure, with swap-rate and codebook-spread
traces over training.

---

## 9. Files

```
src/model_jax/momos/
├── drift.py        # cohort_indices, adaptive_scale, accumulate, DriftState
├── codebook.py     # spherical k-means, EMA update, assignment + null test
├── reassign.py     # §5 macro-loop: eligibility, target, neighbour test, swap
├── train_step.py   # + drift accumulation (§4); keep _propose_swaps for the
│                   #   `single` arm so the gate can compare all three
└── maintenance.py  # unchanged

tests/model_jax/momos/
├── test_drift.py
├── test_codebook.py
└── test_reassign.py

scripts/momos_phase_c.py   # extend to the three-way gate of §8
```

## 10. Verification before reporting

- `.venv/bin/python -m pytest tests/model_jax/momos/ -q` green, existing 66
  still passing
- `.venv/bin/python -m pytest tests/ -q` → exactly 2 failures, both
  `tests/test_dfm_mimir.py` (pre-existing; do not fix)
- `.venv/bin/python -m pyflakes src/model_jax/momos/*.py
  tests/model_jax/momos/*.py scripts/momos_phase_c.py` clean
- Run the three-way gate and include its real, complete output
