# Phase D — Lifecycle: drop / merge / revive

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

Phase B/C fix `K` motifs at init and never change how many there are. Two
failure modes follow, and `metrics.usage_entropy` exists specifically to detect
the first:

- **Collapse.** Blocks pile onto a handful of motifs. The effective dictionary
  is far smaller than `K`, so the model is paying `K·S·12` bytes for capacity
  it is not using.
- **Waste.** Motifs drift close enough together to be interchangeable, or end
  up with no blocks at all. Same cost, same non-use.

Lifecycle reclaims those slots: **merge** near-duplicates, **drop** the unused,
**revive** the reclaimed slots against the motifs that are carrying too much.

The compression ledger does not move. `K` is fixed — lifecycle changes which of
the `K` slots are live, never how many exist, so `bytes_per_weight` is identical
before and after. What should improve is loss at matched `K`, or equal loss at
smaller `K`.

---

## 2. Two corrections to SPEC.md §6.2

### 2.1 Usage counts need no hot-path change

An earlier read of this phase (mine, in conversation) claimed lifecycle needs
per-motif usage counters maintained in the training loop. **That is wrong and
would be a costly mistake** — it would add `(K,)` of mutable state to the jitted
step for a quantity that is already derivable.

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

### 2.2 "High-reconstruction-error block" is not a well-defined quantity

`SPEC.md` §6.2 says to revive a dead motif "from a high-reconstruction-error
block". **There is no such thing in this scheme.** A block's weights *are*
`motifs[mosaic[m]]` — reconstruction is exact by construction, and every block
sharing a motif has byte-identical values. There is no residual to rank blocks
by, and no stored per-block target.

The nearest well-defined analogue is step D's own misfit,
`d_cur = ||w_prop − motifs[cur]||²` where
`w_prop = motifs[cur] − eta · g_blocks[idx]` (`train_step.py:175-180`) — how far
a block wants to be from where its motif actually sits. But `g_blocks` is the
`(M, S)` transient that only exists inside `train_step`, and surfacing it to the
host every maintenance step is exactly the memory cost §2 of `SPEC.md` forbids.

**So revive is specified as a split instead** (§5). This is the standard
dead-centroid rule from k-means, it needs only `mosaic` + `motifs`, and it is
the operation that actually moves `usage_entropy`.

---

## 3. Config

Add to `MosaicConfig`:

```python
lifecycle_every:  int   = 0      # maintenance passes between lifecycle runs; 0 disables
merge_eps:        float = 1e-3   # RELATIVE distance below which two motifs merge
split_jitter:     float = 1e-2   # RELATIVE perturbation applied on split
revive_frac:      float = 0.5    # fraction of a donor's blocks handed to the revived slot
max_revives:      int   = 0      # cap per pass; 0 = no cap
```

`lifecycle_every = 0` disables the phase entirely and **must** reproduce Phase C
bit-for-bit — the same escape hatch `subset_size=0` and `cohort_frac=0.0`
already provide.

**Validation, at construction:**
- `split_jitter > merge_eps`, strictly. See §6 — this is not a style preference,
  it is what stops revive and merge from undoing each other every pass.
- `0 <= revive_frac < 1`. At `1.0` the donor keeps nothing and the split is a
  rename, not a split.
- `merge_eps >= 0`, `lifecycle_every >= 0`, `max_revives >= 0`.

### 3.1 Both epsilons are RELATIVE, not absolute

`scale_mode="none"` means motifs sit at whatever scale the donor weights
happen to have, and that scale differs per model and drifts during training. An
absolute `merge_eps` would merge everything on one model and nothing on the
next.

Define the scale once per pass, from the live motifs only:

```python
scale = sqrt(mean(sum(motifs[active] ** 2, axis=-1)))   # RMS motif norm
eps   = merge_eps * scale
```

and use `eps` for the merge test. Apply `split_jitter * scale` the same way.
If `scale` is 0 or non-finite (a degenerate dictionary), skip the pass and say
so in the metrics rather than dividing by it.

---

## 4. The pass — order is load-bearing

Run **on the host in NumPy**, every `lifecycle_every` maintenance steps (so
every `lifecycle_every * maintenance_every` training steps). Union-find does not
vectorise cleanly and this executes rarely; do not try to jit it.

The three operations must run **merge → drop → revive**, in that order:

1. **Merge** collapses near-duplicates and rewrites the mosaic. This is what
   *creates* newly-unused motifs — a motif all of whose blocks were just
   redirected to a root now has count 0.
2. **Drop** marks count-0 motifs inactive. Running it before merge would miss
   everything merge is about to free, wasting a whole pass.
3. **Revive** fills the slots the first two freed. Running it before drop would
   find nothing to fill.

Recompute counts from the mosaic *after* merge, not before. A pass that reuses
pre-merge counts for the drop step will drop the wrong motifs.

---

## 5. The three operations

### 5.1 Merge

Candidate pairs are near-duplicates among **active** motifs:
`||motifs[i] − motifs[j]|| < eps`, `i < j`.

Do not materialise the full `K×K` pair matrix — at `K=65536` that is 17 GB.
Reuse `maintenance.neighbour_graph`, which already computes exact nearest
neighbours in row blocks with dead motifs masked by `+inf` (never by a 0/1
multiplicative mask — see that module's docstring for why). Consider only each
motif's `n_neighbors` candidates; anything closer than `eps` is necessarily
among them.

**Union-find with path compression**, root = **lowest index** in the component.
Transitivity is the whole point: given `j→i` and `i→h`, every block of `j` must
end at `h`, not at the dead intermediate `i`. Resolve with `find()` on every
mosaic entry, never with a single-level lookup table.

Then, in one pass:
- `mosaic = root_of[mosaic]` — vectorised gather, and it **must preserve the
  mosaic's dtype** (`uint8`/`uint16`). A silent widening to int64 through NumPy
  fancy indexing destroys the compression premise; assert the dtype after.
- Every non-root member of a component: `active[member] = False`.
- The root's value is left as-is. The members are within `eps` of it by
  construction, so averaging them buys nothing measurable and costs a
  usage-weighted mean.

### 5.2 Drop

`counts == 0` and `active` → `active = False`.

Recompute `counts` from the post-merge mosaic (§2.1, §4). No mosaic rewrite is
needed: a motif with zero blocks has nothing pointing at it, which is what makes
dropping it safe.

Never drop the last live motif. If `active.sum()` would reach 0, skip the drop
and record it — a dictionary with no live motifs cannot be swapped onto and the
run is dead from that point.

### 5.3 Revive — as a split

For each dead slot, up to `max_revives`:

- **Donor** = the active motif with the highest count. Skip the whole revive if
  the best donor has fewer than 2 blocks (nothing to split).
- **Value**: `motifs[dead] = motifs[donor] * (1 + split_jitter * scale * u)`
  where `u` is a deterministic unit-norm perturbation from the pass's seeded
  RNG. Seed it from the step number so a re-run reproduces.
- **Blocks**: hand a deterministic `revive_frac` of the donor's blocks to the
  revived index. Deterministic means *by position* — e.g. every block whose
  index within the donor's block list falls in the first `revive_frac` — not a
  fresh random draw per pass.
- **`active[dead] = True`**.
- **Zero the revived row in `opt_state`.** `SPEC.md` §6.2 is emphatic and
  correct: a revived motif carrying its previous life's Adam momentum takes a
  wild first step. `opt_state` is `optimizer.init({"motifs": ..., "scales":
  ...})` (`state.py:228`), so for `optax.adam` this means row `dead` of both
  `mu["motifs"]` and `nu["motifs"]`. Walk the optimiser pytree and zero that row
  in **every** leaf shaped `(K, S)` rather than hardcoding `mu`/`nu` by name —
  it must not silently no-op if the optimiser is changed to one with more
  moments.

**Why a split works even though the two motifs start nearly equal.** All blocks
of a motif have identical values, so there is no intra-cluster spread to split
along. The two slots diverge anyway: each receives the *aggregated gradient of
its own block subset* (`SPEC.md` §5 step B), and those subsets differ, so their
updates differ from the first step onward. The jitter only breaks the initial
tie; the gradient does the actual separating.

---

## 6. The trap: revive and merge must not fight

A revived motif starts at `motifs[donor] * (1 + split_jitter * scale * u)`,
i.e. at distance `≈ split_jitter * scale` from its donor. Merge collapses any
pair closer than `merge_eps * scale`.

If `split_jitter <= merge_eps`, **the very next pass merges every motif this
pass revived, straight back into its donor** — and the blocks that were split
off are handed back too. The net effect is an expensive no-op that looks like it
is working: `live_motifs` oscillates, `usage_entropy` returns to where it
started, and the loss never moves.

This is why `split_jitter > merge_eps` is a construction-time assertion (§3) and
why `test_lifecycle.py` must contain a **two-pass** test, not just a one-pass
one — a single pass cannot observe this failure at all.

---

## 7. Invariants — the tests that must hold

`tests/model_jax/momos/test_lifecycle.py`:

1. **No block points at a dead motif.** After any pass,
   `active[mosaic].all()`. This is `SPEC.md` §6.2's closing line and the single
   most important property here.
2. **Chain merge.** Construct `j→i→h` explicitly (three motifs within `eps` in a
   chain) and assert every block of `j` *and* of `i` ends at `h`, with `i` and
   `j` both inactive. A single-level remap passes a `j→i` test and fails this
   one; that is the point.
3. **Mosaic dtype survives.** `mosaic.dtype` is unchanged by a pass. Assert for
   `uint8` (K≤256) and `uint16` separately — NumPy fancy indexing is where this
   silently widens.
4. **Revive zeroes the optimiser row.** Every `(K, S)`-shaped leaf of
   `opt_state` is exactly 0 at the revived index, and **unchanged at every other
   index** — the second half is what catches a fix that zeroes too much.
5. **Two-pass stability** (§6). With `split_jitter > merge_eps`, run two passes
   and assert the motifs revived in pass 1 are still alive and still distinct
   from their donors after pass 2. Then assert the construction check rejects
   `split_jitter <= merge_eps`.
6. **`lifecycle_every = 0` is bit-for-bit Phase C.** Same seed, same data, same
   final `motifs`/`mosaic` as a run with the lifecycle module not invoked at
   all. Use `atol=0` — this is an identity claim, not an approximation.
7. **Drop never drops a used motif**, and never empties the dictionary
   (`active.sum() >= 1` always).
8. **Counts agree with the mosaic.** `usage_counts(state).sum() == M`, and every
   index in `mosaic` is `< K`.
9. **Entropy does not collapse.** Over a short training run with lifecycle on,
   `usage_entropy` at the end is not lower than at the start by more than a
   stated tolerance. Lifecycle exists to *raise* effective dictionary use; a
   pass that reliably lowers entropy is a bug, not a tuning issue.

---

## 8. Metrics to log every pass

Add to `metrics.py`, alongside the existing `live_motifs` / `usage_entropy`:

- `usage_counts(state) -> (K,) int` — the shared helper §2.1 asks for;
  `usage_entropy` must be refactored to call it.
- `n_merged`, `n_dropped`, `n_revived` — per pass.
- `live_motifs` and `usage_entropy` before and after the pass, so the effect is
  attributable to the pass rather than to the training in between.
- `merge_scale` — the `scale` of §3.1, so a degenerate or drifting dictionary is
  visible rather than inferred.

---

## 9. The gate

`SPEC.md` §8 Phase D: *"matches Phase C loss with measurably fewer live motifs;
usage entropy stays healthy; chain-merge test (j→i→h) leaves no block pointing
at a dead motif."*

The comparison harness already exists and is Hydra-driven. Add
`src/config/momos/lifecycle.yaml` as a new arm (`method: lifecycle`, inheriting
`single`'s `subset_size=64`/`margin=1e-6` plus the §3 knobs) and run:

```bash
python scripts/momos_phase_c.py 'gate.arms=[static,single,lifecycle]'
```

Regime requirements are unchanged and not optional — `~1.63M` mosaicked values
(`d_model=256, num_layers=4`), skip `K >= M/10`, skip any cell whose compressed
footprint reaches dense fp32, print `K/M` and the ledger. The gate script
already enforces all of these.

**`run_gate` must gain `live` and `entropy` columns per arm**, because "matches
Phase C loss with measurably fewer live motifs" is a two-number claim and the
table currently prints only loss. A lifecycle arm that matches `single` on loss
while reporting `live == K` has not passed — it has just not done anything.

**A negative result is a valid outcome.** Report what is measured. Do not tune
until it passes and then report success. Specifically: if lifecycle matches
`single` on loss at equal `live_motifs`, it is inert and should be reported as
inert — that is exactly the failure Phase C2 shipped with at its defaults
(`drift.yaml` records it), and it was invisible until the losses were compared
to five decimals.

---

## 10. Files

```
src/model_jax/momos/
├── lifecycle.py    # NEW — union-find merge, drop, split-revive; host NumPy
├── state.py        # + §3 config knobs and their validation
├── metrics.py      # + usage_counts, lifecycle ledger; refactor usage_entropy
├── maintenance.py  # unchanged — reuse neighbour_graph for merge candidates
└── train_step.py   # unchanged — no hot-path counter (§2.1)

src/config/momos/
└── lifecycle.yaml  # NEW — the gate arm

scripts/momos_phase_c.py   # + live/entropy columns per arm (§9)

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
