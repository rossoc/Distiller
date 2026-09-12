# Dynamic MoMos — implementation spec

Weight compression by dictionary learning: every weight block in the network is a
pointer into one small, global, learned dictionary. The goal is **not** to
reproduce a dense model faithfully — it is to compress hard enough that a
*larger* network fits in the same RAM and thereby outperforms the smaller dense
network that would otherwise have been trainable.

Target: `src/model_jax/momos/`. JAX 0.11.1 / Flax NNX 0.12.9 / optax 0.2.8.
Follow the conventions in `src/model_jax/` (NNX modules, `Variable.get_value()`
not `.value`, `nnx.to_flat_state`, docstrings that say *why*).

---

## 1. Regime — these constraints drive every decision

| Symbol | Meaning | Range here |
|---|---|---|
| `N` | total mosaicked weights | 10^6 – 10^9 |
| `S` | values per block | **1, 2, or 4** |
| `M` | blocks = `ceil(N/S)` | 10^6 – 10^9 |
| `K` | dictionary size | ≤ 1% of M; in practice 256 – 65536 |

**One global dictionary for the entire network.** Not per-layer, not per-role.
Forcing every tensor to draw from the same K motifs is the mechanism that induces
cross-model redundancy; it is the point, not a compromise.

---

## 2. The compression ledger — get this right or nothing else matters

Dense fp32 training costs **12 bytes/weight** (4 param + 4 Adam `m` + 4 Adam `v`).

MoMos persistent cost is dominated entirely by the **mosaic**, not the dictionary:

| S | K | mosaic dtype | bytes/weight | vs dense | dictionary (motifs+Adam) |
|---|---|---|---|---|---|
| 1 | 256 | `uint8` | 1.00 | 12× | 3 KB |
| 2 | 256 | `uint8` | 0.50 | 24× | 6 KB |
| 2 | 4096 | `uint16` | 1.00 | 12× | 96 KB |
| 4 | 4096 | `uint16` | 0.50 | 24× | 192 KB |
| 4 | 65536 | `uint16` | 0.50 | 24× | 3 MB |

**The mosaic dtype IS the compression ratio.** A `uint32` mosaic at S=1 costs
4 bytes/weight — identical to fp32, i.e. *zero* compression. Verified: JAX gathers
correctly with `uint8`/`uint16` index arrays, and `segment_sum` works after an
`.astype(int32)` on the segment ids.

### 2.1 The dictionary only amortises above a crossover size

The table above is **asymptotic**. The true cost is
`bytes/weight = dtype/S + 12·K·S/N`, so MoMos only beats dense once

    N > 12·K·S / (12 − dtype/S)

| S | K | asymptotic B/wt | crossover N |
|---|---|---|---|
| 1 | 256 | 1.00 | 279 |
| 2 | 4096 | 1.00 | 8,937 |
| 4 | 4096 | 0.50 | 17,096 |
| 4 | 65536 | 0.50 | 273,542 |

The toy SSM harness has N = 15,553 mosaicked values, which sits **below** the
S=4/K=4096 crossover — so that configuration measures *worse* than dense (0.91×)
on the toy model while being 24× better asymptotically. Report both numbers, and
assert the crossover in `test_compression.py` so the toy-scale result is never
mistaken for a failure of the method.

**Rules:**
- `mosaic` dtype is chosen from K: `uint8` if K≤256, `uint16` if K≤65536, else `uint32`.
- Assert this at construction. A silent widening to int32 destroys the entire premise.
- Do **not** bit-pack in v1. `uint8`/`uint16` already gives 12–24×; bit-packing
  adds unpack cost on every gather for at most another 2×.

Honest scope of the win: this is **persistent state** (parameters + optimiser).
Activations and the transiently-gathered weights do not compress. Report both.

---

## 3. Tiling — flat and global

All mosaicked parameters are flattened into one contiguous 1-D vector, zero-padded
to a multiple of S, and reshaped to `(M, S)`. Because S ∈ {1,2,4} divides
essentially everything, there is no shape-compatibility constraint — unlike 2-D
tiling, every parameter can participate.

```python
# tiling.py
def flatten_params(tree, include) -> tuple[jnp.ndarray, ParamLayout]
def unflatten_params(flat, layout) -> tree           # exact inverse
def to_blocks(flat, S) -> jnp.ndarray                # (M, S), zero-padded
def from_blocks(blocks, n_values) -> jnp.ndarray     # trims padding
```

`ParamLayout` records each leaf's path, shape, and offset. `unflatten_params
(flatten_params(t)) == t` must hold exactly, including dtypes.

**What to include.** Default: every parameter except 1-D norm weights
(`*/norm/weight`, `norm_f/weight`) and the per-head SSM scalars (`A_log`, `D`,
`dt_bias`). Those are a rounding error in size and are scale-sensitive. Make it a
config predicate so the ablation "mosaic literally everything" is one flag away.

### 3.1 The scale problem — the main research risk

A single dictionary must serve tensors whose natural scales differ by orders of
magnitude. Provide `scale_mode`:

- `"none"` (**default**) — pure form. Training must pull all tensors to a shared
  scale. This is the thesis under test.
- `"per_tensor"` — one learned fp32 scalar per parameter tensor (a few hundred
  floats total, i.e. free). Reconstruction is `motifs[mosaic] * scale[tensor_of_block]`.

Phase B measures both. If `"none"` diverges or plateaus badly, `"per_tensor"` is
the fallback that preserves the compression ratio essentially unchanged.

---

## 4. State

```python
@dataclass
class MosaicState:
    motifs:    jnp.ndarray   # (K, S) float32 — the only large trainable tensor
    mosaic:    jnp.ndarray   # (M,)  uint8/uint16 — the compressed model
    active:    jnp.ndarray   # (K,)  bool
    scales:    jnp.ndarray | None  # (n_tensors,) float32, or None
    opt_state: optax.OptState      # sized for (K,S) + scales, never (M,S)
    layout:    ParamLayout         # static, host-side
```

---

## 5. Micro-loop (`train_step.py`)

Order is load-bearing. Aggregate with the **pre-swap** mosaic; propose swaps from
the **optimiser's actual update**, not a raw SGD step.

```python
def reconstruct(state, cfg) -> params_tree:
    flat_blocks = state.motifs[state.mosaic]          # (M, S)
    if cfg.scale_mode == "per_tensor":
        flat_blocks = flat_blocks * state.scales[layout.block_tensor_id][:, None]
    return unflatten_params(from_blocks(flat_blocks, layout.n_values), layout)

def train_step(state, batch, rng, cfg):
    # A. block-level gradient. Differentiate w.r.t. the GATHERED blocks so the
    #    gradient is genuinely (M,S); the swap rule in D needs per-block grads.
    def loss_fn(blocks):
        return task_loss(apply_model(unflatten(blocks), batch), batch)
    loss, g_blocks = jax.value_and_grad(loss_fn)(state.motifs[state.mosaic])

    # B. aggregate EXACTLY ONCE, with the pre-swap mosaic, usage-normalised.
    seg      = state.mosaic.astype(jnp.int32)
    counts   = jax.ops.segment_sum(jnp.ones(cfg.M), seg, cfg.K)
    g_motifs = jax.ops.segment_sum(g_blocks, seg, cfg.K) / jnp.maximum(counts,1)[:,None]
    g_motifs = jnp.where(state.active[:,None], g_motifs, 0.0)

    # C. optimiser step at motif level
    updates, opt_state = optimizer.update(g_motifs, state.opt_state, state.motifs)
    motifs = optax.apply_updates(state.motifs, updates)

    # D. propose swaps for a random subset, from the REAL update
    idx    = jax.random.choice(rng, cfg.M, (cfg.subset,), replace=False)
    cur    = state.mosaic[idx].astype(jnp.int32)
    w_prop = state.motifs[cur] + updates[cur]          # where this block wants to go
    nbrs   = state.graph[cur]                           # (subset, n_neighbors)
    d_nbr  = jnp.sum((w_prop[:,None] - motifs[nbrs])**2, -1)
    d_nbr  = jnp.where(state.active[nbrs], d_nbr, jnp.inf)   # never swap onto a dead motif
    d_cur  = jnp.sum((w_prop - motifs[cur])**2, -1)
    best   = jnp.argmin(d_nbr, -1)
    cand   = jnp.take_along_axis(nbrs, best[:,None], 1)[:,0]
    swap   = jnp.min(d_nbr,-1) < d_cur - cfg.margin
    mosaic = state.mosaic.at[idx].set(
        jnp.where(swap, cand, cur).astype(state.mosaic.dtype))
    ...
```

### 5.1 Step D must use the BLOCK gradient — correction, both parts verified

The step D sketched above is wrong in two ways, each confirmed by execution.

**(a) `d_cur` is identically zero.** Comparing against the *post*-update
dictionary, `motifs[cur] == state.motifs[cur] + updates[cur] == w_prop` by
definition of `optax.apply_updates`. So `d_cur ≡ 0` and
`min(d_nbr) < d_cur - margin` can never hold for a squared distance: step D is a
silent no-op that still reports a swap rate. Compare against the **pre-update**
value.

**(b) `w_prop` is motif-constant, which defeats the purpose.**
`state.motifs[cur] + motif_updates[cur]` depends only on the *motif*, so every
block sharing a motif makes an identical decision and its own gradient never
enters. Step D then cannot do the one thing it exists for — differentiate blocks
*within* a motif. Measured consequence: swap rate decays to exactly 0.000 within
the first decile of training.

**Correct formulation.** The target for a block is where *its own* gradient pulls
its weights:

```python
eta    = ||updates[cur]|| / (||g_motifs[cur]|| + 1e-12)   # optimiser's realised
                                                          # step per unit gradient
w_prop = motifs[cur] - eta[:, None] * g_blocks[idx]       # (subset, S) — per BLOCK
d_cur  = jnp.sum((w_prop - motifs[cur]) ** 2, -1)         # varies per block
d_nbr  = jnp.sum((w_prop[:, None] - motifs[nbrs]) ** 2, -1)
```

Deriving `eta` from the optimiser rather than hardcoding `lr` preserves the
original D4 concern (a raw `-lr*grad` proposal is scale-inconsistent with Adam)
while restoring the per-block signal. `g_blocks` and `idx` must be threaded into
`_propose_swaps`; `g_blocks` is already computed in step A.

### Critical correctness note

`jax.value_and_grad(f)(motifs)` where the gather is *inside* `f` returns a
**`(K,S)`** gradient — autodiff has already aggregated, because the transpose of a
gather is a scatter-add. Differentiating w.r.t. the gathered blocks (as above)
is what yields a true `(M,S)` gradient. Applying `segment_sum` to an
already-aggregated gradient is a real bug and must not appear.

Verified identity, which becomes a test:
`grad(λm. f(m[mosaic]))(motifs) == segment_sum(grad(f)(motifs[mosaic]), mosaic, K)`

### Memory

`g_blocks` is `(M,S)` fp32 — one model's worth, transiently. Acceptable in v1
(dense training pays the same). If it dominates, the escape hatch is per-layer
processing: reconstruct, backprop and swap one parameter tensor at a time, so
only that tensor's block gradients are live. Build v1 simple; measure first.

---

## 6. Maintenance loop (`maintenance.py`) — run every `cfg.maintenance_every` steps

### 6.1 Neighbour graph: EXACT, not LSH

Sign-random-projection LSH is **useless at S ≤ 4** — measured, K=512:

| S | distinct hash codes / 512 | top-1 agreement with exact NN |
|---|---|---|
| 1 | 2 | 0.2% |
| 2 | 99 | 7.0% |
| 4 | 497 | 26.2% |

At S=1 an L2-normalised motif is literally ±1, so the hash carries one bit. Exact
all-pairs is trivially affordable here: K=4096, S=4 is 67 Mflop and a 67 MB
distance matrix.

```python
def neighbour_graph(motifs, active, n_neighbors, block=1024):
    # exact squared-L2, computed in row blocks to bound the K x K matrix
    # mask with -inf / +inf, NEVER by multiplying by a 0/1 mask:
    #   a signed score multiplied by 0 beats any negative score, so dead
    #   motifs get PROMOTED to nearest neighbour. Use jnp.where(..., inf).
```

For S=1 there is an even cheaper exact route: sort the motifs; neighbours are
adjacent in sorted order. Worth a specialisation if profiling shows the graph
costs anything.

Keep an `lsh.py` stub only if K > 65536 ever becomes real. Not needed for v1.

### 6.2 Lifecycle

Run **on the host in NumPy** — it executes once every few hundred steps, and
merging is union-find, which does not vectorise cleanly.

- **Drop**: `counts == 0` → mark inactive.
- **Merge**: if `||motif_i − motif_j|| < eps`, redirect j → i. Must be transitive:
  if j→i and i→h, every block of j ends at h. Use union-find with path compression.
- **Revive**: reinitialise a dead motif from a high-reconstruction-error block,
  **and zero its slots in `opt_state`** (a revived motif carrying a previous
  life's Adam momentum will take a wild first step).

After any lifecycle change, the mosaic must contain no index that is inactive.

---

## 7. Metrics (`metrics.py`) — log every maintenance step

- `live_motifs`, `usage_entropy` (collapse detector — the standard failure mode)
- `swap_rate` (should decay; a high plateau means `margin` is too loose)
- `reconstruction_mse` between mosaicked and a dense reference, when available
- `bytes_per_weight` actual, and the ratio vs dense
- `loss_delta_around_swap` — task loss immediately before/after a swap batch.
  The swap rule minimises *quantisation* error, not task loss; these are
  correlated but not identical. If swaps systematically raise loss, say so loudly.

---

## 8. Phases — each gate must pass before the next

**A — Tiling + state.** `tiling.py`, `state.py`.
*Gate:* `unflatten(flatten(t)) == t` exactly; mosaic dtype assertion fires for
K>256 with uint8; `bytes_per_weight` matches the §2 table.

**B — Static mosaic, motif-level Adam.** No swaps, no lifecycle. Random init
assignment (or nearest-motif from a k-means++ style seed).
*Gate:* the generic-K aggregation identity (§5) — **this, not the K=M case, is
the guard against double aggregation**; see the correction below. Plus the K=M
identity case as a weaker end-to-end smoke test, and a capacity curve: final loss
vs K ∈ {256, 1024, 4096} at S ∈ {1,2,4}, against the dense baseline, both
`scale_mode`s.

> **Correction (verified during Phase B).** An earlier revision of this spec
> called the K=M / `mosaic=arange(M)` case "the single most important test" for
> catching double aggregation. That is **wrong**. With an identity mosaic,
> `segment_sum(g, arange(K), K)` is a no-op, so a doubly-aggregating
> implementation passes the gate unchanged. Reproduced directly: correct and
> buggy agree exactly at K=M with an identity mosaic. Two things actually guard
> it — the generic-K `test_aggregation_identity`, and the fact that at realistic
> K &lt; M the blueprint's form is a hard shape error (`(K,S)` data against `(M,)`
> segment ids), so it cannot ship silently. Keep the K=M test, but for what it
> does prove: that the end-to-end step reduces to dense training.
>
> Bit-exactness at K=M holds only for a **single-tensor** model. Across a
> multi-tensor pytree it is not achievable and not a bug: plain `optax.adam`
> with no MoMos code diverges between tree-shaped and flat-concatenated layouts
> by ~0.5 float32 ULP over 20 steps, from reassociation in XLA's elementwise
> ops. Use `atol=1e-5` there.

**C — Dynamic swapping.** Micro-loop step D (see §5.1 for the corrected
formulation), exact neighbour graph recomputed each maintenance step.
*Gate:* at matched K, dynamic beats static from Phase B. If not, stop — the swap
criterion is not earning its complexity.

> **The gate must run in the target regime.** The first Phase C run was invalid
> on this point and its result should be discarded. At N = 15,553 (the small toy
> model) the K/M ratios were 1.6%–105%, never the ≤1% this method targets, and
> three of nine cells had **K ≥ M** — more motifs than blocks, so init seeds
> duplicates, most motifs go unused, and "compression" is negative. Measured
> nearest-neighbour distance in those cells was exactly 0.0.
>
> Requirements for a valid gate:
> * **~1M mosaicked values.** `toy_tasks.build(..., d_model=256, num_layers=4)`
>   gives ≈1.64M parameters (≈1.63M mosaicked), at ≈222 ms/step on CPU — so a
>   300-step cell is ≈1 minute and a full sweep is minutes, not hours.
> * **Skip any cell with K ≥ M/10.** Degenerate or out of regime. Omit them and
>   say why; do not report them as failures.
> * **Print K/M as a column** so the regime is visible in the output rather than
>   inferred.

**D — Lifecycle.** Drop / merge / revive on host.
*Gate:* matches Phase C loss with measurably fewer live motifs; usage entropy
stays healthy; chain-merge test (j→i→h) leaves no block pointing at a dead motif.

**E — Mamba2 backbone.** Behind a config flag; the dense path stays default.
*Gate:* loss curve tracks the dense JAX path at matched hyperparameters, with
measured `bytes_per_weight`. Then the actual thesis test — a MoMos model with
~10× the parameters, trained in the same memory budget, beating the dense model
that fits without it.

---

## 9. Validation harness — use what already exists

`src/model_jax/toy_tasks.py` provides a 56k-parameter SSM regressor and two
synthetic tasks (`cumsum`, `adding`) that train in seconds on CPU with
established dense baselines (cumsum 0.0031 vs 0.156 baseline; adding 0.00009).
`scripts/ssm_adding_problem.py` runs them. Every phase gate should run there
first — it makes each phase falsifiable in under a minute on a laptop.

Add `scripts/momos_capacity.py` that sweeps (S, K, scale_mode) on a toy task and
prints the capacity curve plus bytes/weight.

---

## 10. Files

```
src/model_jax/momos/
├── __init__.py
├── tiling.py        # flatten/unflatten/to_blocks/from_blocks, ParamLayout
├── state.py         # MosaicState, dtype selection, init, bytes_per_weight
├── train_step.py    # micro-loop
├── maintenance.py   # exact neighbour graph + host lifecycle
└── metrics.py

tests/model_jax/momos/
├── test_tiling.py        # round-trip, padding, dtype selection
├── test_gradients.py     # the aggregation identity; K=M bit-exactness
├── test_train_step.py    # swap uses optimiser update; no dead-motif swaps
├── test_maintenance.py   # -inf masking; chain merge; revive resets Adam
└── test_compression.py   # bytes/weight matches the §2 table
```

Run `pytest tests/model_jax/momos/ -q` plus the full suite; the pre-existing
2 failures in `tests/test_dfm_mimir.py` are unrelated and expected.
