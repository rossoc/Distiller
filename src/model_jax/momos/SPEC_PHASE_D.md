# Phase D — Lifecycle: merge / drop (monotonic simplification)

Read `SPEC.md` first for the framework, the compression ledger (§2) and the
regime (§1). `SPEC.md` §6.2 is the four-line sketch this document expands;
where the two disagree, this one is the corrected version and says so.

**Keep unchanged:** `tiling.py`, `state.py`'s ledger, `maintenance.py`'s exact
`neighbour_graph`, `train_step.py`'s micro-loop, and the property that a
disabled lifecycle config reproduces Phase C bit-for-bit.

**Do not touch `drift.py` / `codebook.py` / `reassign.py`.** Phase C2 is parked,
not adopted (`src/config/momos/drift.yaml` records the measurements). The
adopted reassignment method is `single` — `train_step._propose_swaps`.

---

## 1. What this phase is for

Phase B/C fix `K` motifs at init and never change how many there are, so the
dictionary can only ever get more redundant: motifs drift close enough together
to be interchangeable, or end up with no blocks pointing at them at all. Either
way the model keeps paying `K·S·12` bytes for capacity it does not use.

Phase D makes the live dictionary **simplify monotonically as training
proceeds**: merge near-duplicates, drop the unused, and never add anything
back. `live_motifs` is non-increasing by construction, which is exactly what
`SPEC.md` §8's Phase D gate asks to see — *"matches Phase C loss with measurably
fewer live motifs"*.

The compression ledger does not move. `K` is fixed — lifecycle changes which of
the `K` slots are live, never how many exist, so `bytes_per_weight` is identical
before and after. The win this phase is chasing is **equal loss at fewer live
motifs**, which is what licenses lowering `K` in a later run.

---

## 2. Revive is deliberately not part of this phase

An earlier revision specified a third operation — revive, reinitialising freed
slots by splitting oversubscribed motifs. **It was implemented, measured, and
removed.** Do not reintroduce it.

Two independent reasons, both on the record:

**It is not well defined as `SPEC.md` §6.2 states it.** That section says to
revive "from a high-reconstruction-error block". There is no such quantity here:
a block's weights *are* `motifs[mosaic[m]]`, so reconstruction is exact by
construction and every block sharing a motif is byte-identical. There is no
residual to rank blocks by and no stored per-block target. The fallback —
revive-as-split, the k-means dead-centroid rule — is well defined but is what
the measurement below rejected.

**It measured catastrophically, and in a way that masked its own damage.**
Running `[static, single, lifecycle]` with split-revive enabled and no cap:

```
 S     K      static   live    H       single   live    H    lifecycle   live    H    lc/static
 1  1024     0.00282   1024  6.93     0.00306   1024  6.93     0.01663   1024  6.92      5.91x
 1  4096     0.00224   4096  8.32     0.00176   4096  8.32     0.02088   4096  8.30      9.33x
```

`live == K` in both cells, so the arm looks inert on the metric the gate reads —
while the loss is 6–9× worse than doing nothing. Merge and drop fired hard
(measured: 732/1024 and 3862/4096 motifs sit within `merge_eps=1e-3` of a
neighbour at init), then revive refilled every freed slot by splitting donors,
every pass. The dictionary was violently restructured three times and `live`
reported no change at all. The damage scales with merge volume — 71% of motifs
eligible gives 5.91×, 94% gives 9.33× — which is the signature of churn, not of
a tuning miss.

Revive also generated most of this phase's accidental complexity: an opt_state
pytree walk, a `split_jitter > merge_eps` construction assertion, a donor-count
bookkeeping loop, and a two-pass oscillation test. All of that goes away with it.

---

## 3. The one correction to SPEC.md §6.2 that remains

### Usage counts need no hot-path change

An earlier read of this phase claimed lifecycle needs per-motif usage counters
maintained in the training loop. **That is wrong and would be a costly
mistake** — it would add `(K,)` of mutable state to the jitted step for a
quantity that is already derivable.

Counts are a pure function of the mosaic, and `metrics.usage_entropy`
(`metrics.py:50`) already computes them exactly this way:

```python
seg = state.mosaic.astype(jnp.int32)
counts = jax.ops.segment_sum(jnp.ones_like(seg, dtype=jnp.float32), seg, K)
```

Lifecycle runs on the host every few hundred steps. Recompute counts there from
`state.mosaic`. Do not add a counter to `train_step`. Extract the counts
expression into a shared `metrics.usage_counts(state) -> (K,) int` and have
`usage_entropy` call it, rather than writing the same `segment_sum` twice.

---

## 4. Config

Add to `MosaicConfig`:

```python
lifecycle_every: int   = 0      # maintenance passes between lifecycle runs; 0 disables
merge_quantile:  float = 0.02   # fraction of live motifs merged per pass
merge_eps:       float = 1e-4   # RELATIVE floor: always merge closer than this
min_live_frac:   float = 0.25   # never merge below this fraction of K
```

`lifecycle_every = 0` disables the phase entirely and **must** reproduce Phase C
bit-for-bit — the same escape hatch `subset_size=0` and `cohort_frac=0.0`
already provide.

**Validation, at construction:**
- `0 <= merge_quantile < 1`
- `merge_eps >= 0`
- `0 < min_live_frac <= 1`
- `lifecycle_every >= 0`

### 4.1 Merge must be quantile-driven, not threshold-driven

This is the single most important design constraint in this document, and a
fixed `merge_eps` **cannot** satisfy it. Measured nearest-neighbour distance per
motif, relative to the RMS motif norm, on the real initialised dictionary:

```
 S     K    RMSnorm  min d/scale      p50   <1e-3   <1e-2   <1e-1
 1   256    0.08533     8.88e-06  1.01e-03     128     245     255
 1  1024    0.04948     3.39e-07  4.95e-04     732    1000    1017
 1  4096    0.04365     0.00e+00  1.37e-04    3862    4058    4076
 2   256    0.03281     1.04e-03  6.44e-02       0       6     192
 2  1024    0.03508     9.73e-04  2.96e-02       2      70     955
 2  4096    0.03752     0.00e+00  1.30e-02      25    1412    4009
 4   256    0.03440     0.00e+00  2.98e-01       3       3      10
 4  1024    0.07783     0.00e+00  9.01e-02       4       4     612
 4  4096    0.06485     0.00e+00  7.56e-02       8      10    2835
```

At `1e-3`, S=1/K=4096 merges **94% of the dictionary on the first pass** while
S=4/K=256 merges **3 motifs**. That is a ~500× swing in effect from one
constant, and it is structural rather than a tuning problem: nearest-neighbour
spacing in `S` dimensions scales like `K^(-1/S)`, so it is `K⁻¹` at S=1 and
`K^(-1/4)` at S=4. No constant threshold spans that.

**So merge takes the bottom `merge_quantile` of the live nearest-neighbour
distance distribution each pass**, which self-scales across every S and K. This
is the same lesson `SPEC_PHASE_C2.md` §5.4 already learned when it replaced a
high-water-mark threshold with a quantile; do not relearn it here.

`merge_eps` survives only as a **floor**: a pair closer than `merge_eps * scale`
merges regardless of quantile, so exact duplicates (note the `0.00e+00` entries
above — init does produce them) are always collapsed.

### 4.2 Both epsilons are RELATIVE, not absolute

`scale_mode="none"` means motifs sit at whatever scale the donor weights happen
to have, and that scale differs per model and drifts during training. Define the
scale once per pass, from the live motifs only:

```python
scale = sqrt(mean(sum(motifs[active] ** 2, axis=-1)))   # RMS motif norm
```

If `scale` is 0 or non-finite (a degenerate dictionary), skip the pass and
record that in the metrics rather than dividing by it.

---

## 5. The pass

Run **on the host in NumPy**, every `lifecycle_every` maintenance steps (so
every `lifecycle_every * maintenance_every` training steps). Union-find does not
vectorise cleanly and this executes rarely; do not try to jit it.

Order is **merge → drop**, and it is load-bearing:

1. **Merge** collapses near-duplicates and rewrites the mosaic. This is what
   *creates* newly-unused motifs — a motif all of whose blocks were just
   redirected to a root now has count 0.
2. **Drop** marks count-0 motifs inactive.

Recompute counts from the mosaic **after** merge, not before. A pass that
reuses pre-merge counts for the drop step will drop the wrong motifs.

### 5.1 Merge

Candidate pairs are near-duplicates among **active** motifs. Do not materialise
the full `K×K` pair matrix — at `K=65536` that is 17 GB. Reuse
`maintenance.neighbour_graph`, which already computes exact nearest neighbours
in row blocks with dead motifs masked by `+inf` (never by a 0/1 multiplicative
mask — see that module's docstring for why).

Per pass:

- `d[k]` = distance from motif `k` to its nearest live neighbour, for live `k`.
- `thr = max(quantile(d, merge_quantile), merge_eps * scale)` — the quantile
  sets the dose, the floor guarantees duplicates always go.
- Candidate pairs are `(k, nn[k])` with `d[k] <= thr`.
- **Union-find with path compression**, root = **lowest index** in the
  component. Transitivity is the whole point: given `j→i` and `i→h`, every block
  of `j` must end at `h`, not at the dead intermediate `i`. Resolve with `find()`
  on every mosaic entry, never with a single-level lookup table.
- `mosaic = root_of[mosaic]` — vectorised gather, and it **must preserve the
  mosaic's dtype** (`uint8`/`uint16`). A silent widening to int64 through NumPy
  fancy indexing destroys the compression premise; assert the dtype after.
- Every non-root member: `active[member] = False`.
- The root's value is left as-is. Members are within `thr` of it by
  construction, so averaging buys nothing measurable.

Note on cliques: a clique of size <= n_neighbors + 1 is guaranteed to collapse
fully in one pass because all pairwise edges are captured in the candidate set.
A clique larger than n_neighbors + 1 collapses fully in one pass if its
n_neighbors-NN subgraph over the clique is connected; if clustered into dense
sub-groups whose intra-cluster distances are smaller than inter-cluster distances,
each sub-cluster collapses in the first pass, and the remaining representatives
merge across subsequent passes.

### 5.2 Drop

`counts == 0` and `active` → `active = False`, with counts recomputed from the
post-merge mosaic (§3, §5). No mosaic rewrite is needed: a motif with zero
blocks has nothing pointing at it, which is what makes dropping it safe.

### 5.3 The floor

Merging is now **irreversible** — nothing replenishes the dictionary. Enforce
`min_live_frac`:

- Before merging, compute how many components the candidate set would collapse.
  If the result would put `live` below `ceil(min_live_frac * K)`, **raise `thr`
  until it does not** (equivalently: take only the closest pairs that fit the
  budget). Do not simply skip the pass — that makes the floor a cliff.
- Never let `active.sum()` reach 0.

---

## 6. The trap: irreversible collapse

With revive gone, §6's old oscillation is impossible — but the opposite failure
becomes possible and must be tested for. Every pass only ever removes motifs, so
a `merge_quantile` that is slightly too high compounds: 2% per pass over 30
passes is `0.98^30 ≈ 55%` of the dictionary gone, and there is no mechanism that
notices or recovers.

Two guards, and the tests in §7 must exercise both:

- `min_live_frac` is a hard floor (§5.3).
- The gate reports `live` per arm per cell (§9), so collapse is visible in the
  table rather than inferred from a loss regression.

The healthy signature is `live` falling quickly at first — init duplicates and
genuinely redundant motifs — then flattening well above the floor. `live`
pinned *at* the floor means `merge_quantile` is too high, not that the floor is
doing its job.

---

## 7. Invariants — the tests that must hold

`tests/model_jax/momos/test_lifecycle.py`:

1. **No block points at a dead motif.** After any pass, `active[mosaic].all()`.
   This is `SPEC.md` §6.2's closing line and the single most important property.
2. **Chain merge.** Construct `j→i→h` explicitly (three motifs within `thr` in a
   chain) and assert every block of `j` *and* of `i` ends at `h`, with `i` and
   `j` both inactive. A single-level remap passes a `j→i` test and fails this
   one; that is the point.
3. **Mosaic dtype survives.** `mosaic.dtype` unchanged by a pass. Assert for
   `uint8` (K≤256) and `uint16` (e.g. K=300) separately — NumPy fancy indexing
   is where this silently widens.
4. **Monotonicity.** `live_after <= live_before`, for every pass, always. This
   is the property that makes this phase "simplification"; nothing may add a
   motif back.
5. **The floor holds.** With `merge_quantile` set absurdly high (e.g. 0.9) over
   several passes, `live` never falls below `ceil(min_live_frac * K)` and never
   reaches 0.
6. **`lifecycle_every = 0` is bit-for-bit Phase C.** Same seed, same data, same
   final `motifs`/`mosaic` as a run with the lifecycle module not invoked at
   all. Use `atol=0` — this is an identity claim, not an approximation.
7. **Drop never drops a used motif.**
8. **Counts agree with the mosaic.** `usage_counts(state).sum() == M`, and every
   index in `mosaic` is `< K`.
9. **Quantile self-scales.** At fixed `merge_quantile`, the fraction of motifs
   merged in one pass is within a stated tolerance across `S ∈ {1,2,4}` at
   matched K. This is the §4.1 defect's regression test — a fixed-threshold
   implementation fails it by orders of magnitude.

---

## 8. Metrics to log every pass

Add to `metrics.py`, alongside the existing `live_motifs` / `usage_entropy`:

- `usage_counts(state) -> (K,) int` — the shared helper §3 asks for;
  `usage_entropy` must be refactored to call it.
- `n_merged`, `n_dropped` — per pass.
- `live_motifs` and `usage_entropy` before and after the pass, so the effect is
  attributable to the pass rather than to the training in between.
- `merge_scale` and the realised `merge_thr`, so a degenerate or drifting
  dictionary is visible rather than inferred.
- `floor_clamped: bool` — whether §5.3 had to raise `thr` this pass. A run where
  this is true every pass is over-merging.

---

## 9. The gate

`SPEC.md` §8 Phase D: *"matches Phase C loss with measurably fewer live motifs;
usage entropy stays healthy."*

The comparison harness already exists and is Hydra-driven. Add
`src/config/momos/lifecycle.yaml` as a new arm (`method: lifecycle`, inheriting
`single`'s `subset_size=64`/`margin=1e-6` plus the §4 knobs) and run:

```bash
python scripts/momos_phase_c.py 'gate.arms=[static,single,lifecycle]'
```

Regime requirements are unchanged and not optional — `~1.63M` mosaicked values
(`d_model=256, num_layers=4`), skip `K >= M/10`, skip any cell whose compressed
footprint reaches dense fp32, print `K/M` and the ledger. The gate script
already enforces all of these.

`run_gate` reports `live` and `entropy` columns per arm. Both halves of the
claim must be read together:

| lifecycle result | verdict |
|---|---|
| loss ≈ `single`, `live` < K | **pass** — the dictionary simplified for free |
| loss ≈ `single`, `live` == K | inert; merge never fired |
| loss worse, `live` < K | over-merging; lower `merge_quantile` |
| loss worse, `live` == K | broken; this is what revive produced (§2) |

### 9.1 Cadence

`lifecycle_every: 1` against the default `maintenance_every: 50` gives 6 passes
in the 300-step gate. **Do not start at 2.** Three passes was measured to be too
few to separate signal from noise in this harness, and the same dose trap
already cost two full grids on Phase C2 (`drift.yaml` records it). If 6 passes
shows nothing, raise the step count rather than the quantile.

**A negative result is a valid outcome.** Report what is measured. Do not tune
until it passes and then report success.

---

## 10. Files

```
src/model_jax/momos/
├── lifecycle.py    # NEW — union-find merge, drop, the floor; host NumPy
├── state.py        # + §4 config knobs and their validation
├── metrics.py      # + usage_counts, lifecycle ledger; refactor usage_entropy
├── maintenance.py  # unchanged — reuse neighbour_graph for merge candidates
└── train_step.py   # unchanged — no hot-path counter (§3)

src/config/momos/
└── lifecycle.yaml  # NEW — the gate arm

scripts/momos_phase_c.py   # live/entropy columns per arm (§9)

tests/model_jax/momos/
└── test_lifecycle.py      # NEW — §7's invariants
```

## 11. Verification before reporting

- `.venv/bin/python -m pytest tests/model_jax/momos/ -q` green; the existing 98
  still passing (89 momos + 9 config)
- `.venv/bin/python -m pytest tests/ -q` → exactly 2 failures, both
  `tests/test_dfm_mimir.py` (pre-existing; do not fix)
- `.venv/bin/python -m pyflakes src/model_jax/momos/*.py
  tests/model_jax/momos/*.py scripts/momos_phase_c.py` clean
- Run the gate of §9 and include its real, complete output
