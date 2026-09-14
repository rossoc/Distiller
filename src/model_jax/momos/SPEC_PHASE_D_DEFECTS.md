# Phase D — known defects, deferred

Phase D (`lifecycle.py`, merge + drop) is **implemented and its tests pass**
(110 in `tests/model_jax/momos/`, 10 of them in `test_lifecycle.py`). The
defects below were found by review *after* that, are not fixed, and are recorded
here rather than patched immediately.

Read `SPEC_PHASE_D.md` first — it is the design this file records deviations
from. Where the two disagree about what the code *does*, this file is right;
where they disagree about what the code *should* do, `SPEC_PHASE_D.md` is.

**Nothing here blocks the phase from running.** The gate executes and produces
real numbers. What D1/D2 mean is that the merge *dose* currently measured is not
the dose the config asks for, so any tuning done before they are fixed will have
to be redone.

Status key: **[verified]** = reproduced directly against this working tree;
**[reported]** = from review, plausible from the code but not independently
reproduced here.

---

## Priority

| # | Defect | Severity | Fix together? |
|---|---|---|---|
| D1 | Merge only uses the nearest-neighbour column | medium | **yes, with D2** |
| D2 | `merge_quantile` delivers ~half its documented dose | medium | **yes, with D1** |
| D3 | Invariant 6's test cannot fail | medium | standalone |
| D4 | Hardcoded arm names in the gate harness | low | standalone |
| D5 | Header/data column widths disagree | low | standalone |
| D6 | `merge_thr` reports 0.0 when the floor saturates | low | standalone |
| D7 | Degenerate `scale` skips `drop` as well as merge | low | standalone |
| D8 | Dead code and a stale docstring | trivial | standalone |

---

# Execution plan — P1 → D1 → D2 → F1

Four stages, in this order. Each stage states what changes, what to measure, and
the rule that decides whether it worked. **Do not carry a measurement across a
stage boundary** — every stage changes something that moves the numbers.

## Why this order

- **P1 first, and it is measurable in isolation.** P1 touches `state.init` and
  `train_step`, which *every* arm shares — `static` and `single` included. So it
  can be measured with lifecycle switched off entirely, which means the merge
  dose defects (D1/D2) cannot contaminate the result. This is the only stage
  here that is cleanly separable, so it goes first.
- **D1 then D2, but only one re-measurement between them.** They are a single
  calibration change split across two edits (see "D1 + D2 interact" below).
  Measuring after D1 alone would produce a number that D2 immediately
  invalidates.
- **F1 last.** Seed replication is only worth its ~80 minutes once the mechanism
  it measures is calibrated. Running it now would replicate an artefact.

## Baseline

The 9-cell, 3-arm grid at `4744.9s` (in the conversation record, and summarised
under F1) is the pre-change baseline. Keep it for comparison; do not re-run it.

---

## Stage 1 — P1: reserve a permanent zero motif

**Implement** per §P1 below: `motifs[0] = 0` at init, gradient masked and the row
re-zeroed after the optimiser step, Adam slots zeroed, `drop_step` exempting
index 0, `usage_counts[0]` exposed as a sparsity metric, all behind
`reserve_zero_motif: bool = False`.

Merge needs **no** special case: the union-find root rule is already "lowest
index", so any component containing index 0 roots there and near-zero motifs
merge *into* zero.

**Measure with lifecycle OFF** — `'gate.arms=[static,single]'` — comparing
`reserve_zero_motif` false vs true. This isolates P1 completely from D1/D2.
Cells: S=1/K=1024, S=2/K=1024, S=4/K=256, plus **S=1/K=4096** (the cell with the
most near-zero weights, so the most likely place for a zero-collapse).

**Decision rule — read `usage_counts[0]` and loss together**, never separately:

| `usage_counts[0]` | loss vs `reserve_zero_motif: false` | reading |
|---|---|---|
| ~0 | unchanged | inert — the zero motif attracts nothing |
| moderate | unchanged or better | **the intended outcome**: free structured sparsity |
| large | much worse | zero-collapse; the escape hatch stays off by default |

**Done when:** the bit-for-bit invariants still hold with
`reserve_zero_motif: false` (this is why it is opt-in), and the table above has
a measured answer.

---

## Stage 2 — D1: draw merge candidates from all neighbour columns

**Implement** the D1 fix: candidate edges from all `n_neighbors` columns with
`d <= thr`, keeping column 0 for the quantile.

**Do not run the gate.** D2 lands next and changes the dose again.

**Do run** the unit suite, and add the clique test D1's repro describes — motifs
at `0, 0.001, 0.0025, 0.0035` with `thr = 0.004` must collapse to **one**
component, not two.

**Then resolve the §5.1 question**: either re-verify `SPEC_PHASE_D.md` §5.1's
clique claim now holds, or correct §5.1 to describe the real guarantee.

---

## Stage 3 — D2: make `merge_quantile` mean what it says

**Implement** the D2 fix — canonicalise candidate pairs to `(min, max)` and size
the eligible set so realised deactivations ≈ `merge_quantile * live`.

**Also settle `merge_eps`.** The baseline grid's headline wins came from the
floor, not the quantile: seven of nine cells merged 5.9–7.4% (the quantile), while
S=1/K=1024 merged 18.7% and S=1/K=4096 merged 47.8% (the floor). `merge_eps`
should mean "numerically indistinguishable", not "close" — drop it roughly two
orders, to `1e-6` or below, so the quantile is the dose knob and the floor only
catches the exact duplicates init genuinely produces.

**Tighten the test:** `test_quantile_self_scales` currently accepts `[0.02, 0.08]`,
wide enough to hide the factor of two it was meant to catch. Assert realised
deactivation fraction against `merge_quantile` within a tight tolerance, at
matched K across `S ∈ {1,2,4}`.

**Now re-measure**, on the same four cells as Stage 1 plus S=1/K=4096:

- `thr / scale` must **not** equal `merge_eps` — if it does, the floor is still
  binding and the quantile is still inert.
- merged fraction should be ≈ `merge_quantile` per pass at **every** cell,
  regardless of S and K. That uniformity is the property §4.1 exists to deliver
  and the thing this stage proves.

**Expect the S=1 wins to shrink or move.** They were produced by the
un-calibrated floor. A smaller, uniform, *explicable* effect is the better
outcome here than a large unexplained one.

**Before leaving this stage, fix D5** (header/data column widths). Stage 4's
output is what gets read; a table whose headers drift three characters per arm
is how a verdict gets misread.

---

## Stage 4 — F1: seed replication

Run §F1 exactly as specified: three seeds × three cells
(S=1/K=1024, S=2/K=1024, S=4/K=256), `'gate.arms=[static,single,lifecycle]'`.

Report **mean and spread** per arm per cell. Apply F1's decision rule without
softening it: *a cell whose inter-seed spread exceeds its lifecycle-vs-single
gap has not demonstrated anything*, and is reported as such — not as a pass and
not as a fail.

Consider adding `gate.seeds: [0,1,2]` first (F1's harness-gap note): it removes
nine manual invocations and amortises the dense baseline, which is currently
recomputed per run.

**This is the stage that decides whether Phase D passed.** Everything before it
is calibration.

---

## Not in the sequence

`D3`, `D4`, `D6`, `D7`, `D8` do not gate the measurements above and can be done
in any order alongside them. `D3` is the one worth not deferring indefinitely —
invariant 6's test currently cannot fail, so the suite is quietly weaker than
its green count suggests.



## D1 — Merge only ever considers the nearest neighbour **[verified]**

**Location:** `src/model_jax/momos/lifecycle.py:99`

```python
nn_indices = graph_np[live_indices, 0]
```

`maintenance.neighbour_graph` is called with `n_neighbors` and returns a
`(K, n_neighbors)` array, then every column but the first is discarded. The
candidate edge set is therefore only `k → argmin`, and **`cfg.n_neighbors` is a
no-op for merge** beyond forcing it to be ≥ 1.

**This falsifies a claim in `SPEC_PHASE_D.md` §5.1**, which asserts:

> Note that a clique larger than `n_neighbors` still collapses fully in one
> pass: each member's nearest neighbours are all within `thr`, and union-find
> closes the component transitively across rows.

That reasoning is only valid if all `n_neighbors` columns feed the candidate
set. With column 0 alone, mutually-close groups fragment into disjoint
mutual-nearest-neighbour pairs instead of collapsing.

**Reported repro:** motifs at `0, 0.001, 0.0025, 0.0035` (all pairwise distances
≤ 0.0035) with `thr = 0.004` gives `n_merged = 2`, not 3 — `nn[0]=1, nn[1]=0,
nn[2]=3, nn[3]=2` forms two disjoint components, so motifs 0 and 2 both stay
live despite being 0.0025 apart, inside `thr`.

**Fix:** draw candidate edges from all `n_neighbors` columns, keeping only pairs
with `d <= thr`. Keep column 0 for the quantile computation (see D2).

**Then either** re-verify §5.1's clique claim holds, **or** correct §5.1 to
describe what the implementation actually guarantees.

---

## D2 — `merge_quantile` delivers ~half its documented dose **[verified]**

**Location:** `src/model_jax/momos/lifecycle.py:112`

```python
eligible_mask = d_array <= thr
```

This selects the bottom-`q` fraction of *motifs*, which is correct. But close
motifs are almost always **mutual** nearest neighbours, so both `(k, nn[k])` and
`(nn[k], k)` enter the candidate list and collapse into a single union. Two
eligible motifs therefore produce one deactivation.

`state.py` documents the knob as "fraction of live motifs merged per pass";
the realised figure is about half that.

**Measured** (K=512, random motifs, exact union-find over the same candidate
rule):

```
 S     K      q  eligible%  deactivated%  ratio
 1   512   0.02     2.34%        1.17%   0.59
 1   512   0.05     5.08%        2.54%   0.51
 1   512   0.10    10.35%        5.27%   0.53
 1   512   0.20    20.12%       10.94%   0.55
 2   512   0.02     2.34%        1.17%   0.59
 2   512   0.20    20.12%       10.94%   0.55
 4   512   0.02     2.34%        1.17%   0.59
 4   512   0.20    20.31%       11.91%   0.60
```

Confirmed in a real gate run: `merge_quantile=0.02`, live=1024 → `merged=11`
(1.07%, half of 2%).

**Note what is NOT broken here.** The eligible fraction is exactly `q` at S=1, 2
and 4 alike, so `SPEC_PHASE_D.md` §4.1's self-scaling across `S` — the property
that section exists to guarantee — **does work**. This is a calibration and
naming gap, not the structural defect §4.1 was written to prevent.

**Fix, one of:**
- de-duplicate candidate pairs (canonicalise each to `(min, max)`) and size the
  eligible set so that realised deactivations ≈ `q * live`; or
- redefine the knob in `state.py` and `SPEC_PHASE_D.md` §4 as "fraction of live
  motifs *examined*", and accept ≈ `q/2` merged.

The first keeps the knob meaning what its name says; prefer it.

---

### D1 + D2 interact — do not fix them independently

Fixing D1 widens the candidate edge set (more merging). Fixing D2 roughly
doubles the realised dose. Applied naively together they could **≈4× the merge
rate**.

There is also a consistency problem once D1 lands: the quantile is computed on
column-0 distances while candidates would come from all columns, so threshold
and candidate set would no longer measure the same quantity. Decide explicitly
whether `thr` stays a column-0 quantile (a dose proxy) or becomes a quantile
over the full candidate-edge distance distribution.

**Re-measure after the combined fix**, using the diagnostic that caught the
original calibration problem: if `thr / scale` equals `merge_eps` exactly, the
floor is binding and the quantile is inert.

**Add the regression test D2 lacks:** `test_quantile_self_scales` currently
accepts a `[0.02, 0.08]` band, which is wide enough to hide a factor of two.
Assert the realised deactivation fraction against `merge_quantile` within a tight
tolerance, at matched K across `S ∈ {1,2,4}`.

---

## D3 — Invariant 6's test cannot fail **[verified]**

**Location:** `tests/model_jax/momos/test_lifecycle.py:331`

`SPEC_PHASE_D.md` §7.6 requires the same final `motifs`/`mosaic` at `atol=0`.
The test instead:

- compares only `loss`, `live` and `entropy` — it captures
  `loss_s, _, _, live_s, ent_s, _` and never looks at the arrays; and
- compares a `single` config against a `lifecycle` config **both with
  `lifecycle_every: 0`**, so both take the identical branch in `train_momos`
  and no lifecycle code runs in either.

It therefore cannot catch a regression where `lifecycle_every=0` still perturbs
state — which is exactly the invariant it is named for.

**Fix:** compare the final `mosaic` and `motifs` arrays with `atol=0`. This needs
`train_momos` to expose them (it currently returns
`loss, swaps, spreads, live, entropy, lifecycle_history`), or a lower-level
comparison that bypasses the gate harness entirely. The second is preferable —
it keeps the invariant independent of the gate script's return signature.

---

## D4 — Hardcoded arm names in the gate harness **[verified]**

**Location:** `scripts/momos_phase_c.py:253`

```python
if arm in ("single", "lifecycle"):
```

This gates both the swap history and the entire lifecycle block on literal arm
names, directly contradicting `train_momos`' own docstring:

> The branch below is on that name rather than on a hardcoded arm list, so a new
> method is a new yaml file, not an edit here.

A new `src/config/momos/foo.yaml` with `lifecycle_every: 4` would run the swap
loop, never call `lifecycle_pass`, record no swap history, and warn about
none of it.

**Fix:** gate on behaviour — `cfg.subset_size > 0` for swap history,
`cfg.lifecycle_every > 0` for the lifecycle block.

---

## D5 — Header and data column widths disagree **[reported]**

**Location:** `scripts/momos_phase_c.py:394`

Per-arm header cells are wider than the data cells beneath them: `a[:5]+'_lv'`
and `a[:5]+'_h'` are 8 and 7 characters in `:>6` fields, while the rows print
`{lives[a]:>6d}` and `{entropies[a]:>6.2f}`. Each arm contributes 26 header
characters against 23 data characters, so the header drifts ~3 characters right
per arm:

```
... B/wt_inf   static stati_lv stati_h   single singl_lv singl_h lifecycl lifec_lv lifec_h ...
...    1.000  0.01596   1024   6.93  0.01547   1024   6.93  0.01525   1002   6.90 ...
```

**Fix:** shorten to `a[:3]+'_lv'` / `a[:4]+'_h'`, or widen the data fields to
match. Misaligned columns in the gate table are how a misread verdict starts.

---

## D6 — `merge_thr` reports 0.0 when the floor saturates **[verified]**

**Location:** `src/model_jax/momos/lifecycle.py:120`

```python
realised_thr = 0.0 if clamped else thr
```

When `budget == 0` (live already at `min_live_frac * K`) the early return
reports `thr=0.0000e+00 clamped=True`. `SPEC_PHASE_D.md` §8 asks for the
realised threshold so "a degenerate or drifting dictionary is visible rather
than inferred" — and `thr=0.0` reads as a degenerate dictionary rather than a
saturated floor. The two states need different responses.

**Fix:** report the computed `thr` alongside `clamped=True`.

---

## D7 — Degenerate `scale` skips `drop` as well as merge **[reported]**

**Location:** `src/model_jax/momos/lifecycle.py:262`

`if scale <= 0.0` returns before `drop_step`. But drop depends on nothing except
the mosaic, and when `merge_quantile > 0` the threshold does not depend on
`scale` either. A degenerate dictionary therefore never reclaims unused slots and
reports `n_dropped=0` indefinitely.

`SPEC_PHASE_D.md` §4.2 says to skip the *scale-derived* part of the pass;
skipping drop is a stricter reading than the spec requires.

**Fix:** on degenerate `scale`, skip merge only and still run drop.

---

## D8 — Dead code and a stale docstring **[reported]**

**Location:** `src/model_jax/momos/lifecycle.py:250, :271, :284, :76`

- `lifecycle_pass`'s `step` parameter is never used. It was needed by revive's
  seeded RNG; revive was removed (`SPEC_PHASE_D.md` §2) and the parameter
  outlived it.
- `merge_scale=scale if np.isfinite(scale) else 0.0` is unreachable —
  `merge_scale()` already returns `0.0` for non-finite input.
- `counts` returned from `drop_step` is bound and discarded.
- `merge_step`'s docstring says it "Rewrites `mosaic` in place" when it returns
  a new array. The early-return path additionally aliases the caller's array
  rather than copying it, unlike `active` — inconsistent even if currently
  harmless.

---

---

# Follow-up experiments — before Phase D is recorded as passed

The gate has run and produced real numbers (`4744.9s`, 9 cells, 3 arms). It is
**not yet a verdict**, for the two reasons below. Both are cheap relative to
having to redo the tuning later.

## F1 — Seed replication: the current grid cannot resolve the effect it reports

**One seed was run (`gate.seed: 0`).** That is not enough to separate a 15%
effect from run-to-run variation, and the grid contains direct evidence of how
large that variation is. `single` vs `static` — a mechanism that should be
consistently mildly helpful — came out:

```
single/static:  0.88  1.09  0.79  0.34  0.86  0.36  0.99  1.69  0.75
```

A 0.34x–1.69x spread across cells. Any lifecycle ratio inside roughly 0.8–1.4
is therefore indistinguishable from noise, which covers four of the nine cells:
S=2/K=256 (0.93x), S=2/K=1024 (0.83x), S=2/K=4096 (1.13x), S=4/K=4096 (1.40x).

Note also that S=4/K=1024's headline "0.36x vs single" is mostly `single` being
anomalously bad in that cell (1.69x worse than `static`); against `static` it is
0.60x. Ratios against `single` are unstable wherever `single` itself is.

**Only S=1/K=1024 (0.39x) and S=1/K=4096 (0.35x) clear that noise floor.**

**Run three seeds on three cells**, chosen to span the observed range rather
than to re-measure the whole grid:

| S | K | why this cell |
|---|---|---|
| 1 | 1024 | the strongest robust win (0.39x vs single, 18.7% merged) |
| 2 | 1024 | mid-range, quantile-bound (0.83x, 6.2% merged) |
| 4 | 256 | one of the ~2x regressions (2.00x worse, 7.0% merged) |

```bash
for s in 0 1 2; do
  .venv/bin/python scripts/momos_phase_c.py \
    'gate.arms=[static,single,lifecycle]' \
    'gate.s=[1]' 'gate.k=[1024]' gate.seed=$s
done   # repeat for (S=2,K=1024) and (S=4,K=256)
```

Report mean and spread per arm per cell, not single numbers. **A cell whose
inter-seed spread exceeds its lifecycle-vs-single gap has not demonstrated
anything**, and should be reported that way rather than as a pass or a fail.

**Budget:** ~527s per cell for 3 arms (derived from the 4744.9s full grid), so
9 cell-runs is ~80 min — about the same as the full grid, because the dense
baseline is recomputed per invocation.

**Harness gap:** `gate.seed` is a scalar, so this needs three separate
invocations per cell. Consider adding `gate.seeds: [0, 1, 2]` and aggregating
inside `run_gate`, which would also amortise the dense baseline across seeds.

## F2 — Init-dedup control: is Phase D repairing a bad initialisation?

The two cells that pass decisively are S=1/K=1024 and S=1/K=4096 — **exactly
where initialisation produces the most near-duplicate motifs**. Measured at init
(the §4.1 table in `SPEC_PHASE_D.md`), the fraction of motifs whose nearest
neighbour is within `1e-3` relative:

```
S=1:  K=256 -> 50%    K=1024 -> 71%    K=4096 -> 94%
S=2:  K=256 ->  0%    K=1024 ->  0.2%  K=4096 ->  0.6%
S=4:  K=256 ->  1%    K=1024 ->  0.4%  K=4096 ->  0.2%
```

This is the `K^(-1/S)` scaling: at S=1 the motifs are scalars, so K of them
packed into a 1-D range are inevitably close, and it gets worse as K grows.

**So there are two competing explanations for the S=1 win**, and the grid cannot
distinguish them:

1. Lifecycle provides ongoing value by continuously reclaiming redundancy.
2. `state.init` produces duplicate motifs at S=1, and lifecycle is a one-time
   repair for that — value that a better init would capture for free, without a
   periodic host pass.

**The control:** dedup at init — k-means++ seeding with a minimum-separation
constraint, or simply re-drawing duplicate motifs — then re-run `static`,
`single` and `lifecycle` on S=1/K=1024 and S=1/K=4096.

- If `static`/`single` close most of the gap, **the fix belongs in `state.init`**
  and Phase D's remaining value is much smaller than the grid suggests.
- If the gap persists, the redundancy is regenerated during training and
  lifecycle is doing structural work.

Either answer is worth knowing before tuning `merge_quantile`, and this is the
cheaper experiment of the two.

---

# Open design proposals

## P1 — Reserve a permanent zero motif

**Requested, not yet specified or implemented.** The goal: guarantee that a
block can always be assigned weights of exactly zero, and that motifs which
drift close to zero collapse onto it rather than lingering as small non-zero
values. This turns "go to zero" into a reachable state of the assignment rather
than something gradient descent has to approximate, and gives the dictionary a
learned structured-sparsity channel.

**Put the zero motif at index 0.** The existing union-find rule (root = lowest
index, `lifecycle.py`) then makes it a fixed point of merge for free: any
component containing index 0 roots at index 0, so every near-zero motif merges
*into* zero and never the other way round. That is exactly the requested
"motifs close to 0 go to zero" semantics, with no special case in the merge
path.

**What must change:**

- **`state.init`** — set `motifs[0] = 0` after seeding, and assign blocks with
  the zero motif present in the candidate set.
- **`train_step`** — the zero motif must not be trained off zero. Mask its
  gradient in step B and re-zero `motifs[0]` after step C. This is a hot-path
  change, but a masked write on one row, not a new `(K,)` of state. Zero its
  Adam slots too, or they accumulate against a row that is forced back to zero
  every step.
- **`lifecycle.drop_step`** — exempt index 0 unconditionally. It will frequently
  have `count == 0` early in training and must survive that.
- **`metrics`** — `usage_counts[0]` becomes a sparsity metric (fraction of
  blocks pinned to exactly zero). Worth its own column in the gate table; it is
  the number that says whether this feature is doing anything.

**Must be opt-in:** `reserve_zero_motif: bool = False`. Turning it on changes
Phase B and Phase C behaviour, not just Phase D, so it would otherwise break
the bit-for-bit invariants (`SPEC_PHASE_D.md` §7.6 and the Phase B/C
equivalents) that the escape hatches exist to protect.

**The risk to measure, not assume:** at S=1 a great many weights sit near zero,
so a large fraction of blocks could collapse onto motif 0. That is either
excellent (free structured sparsity at no ledger cost) or model-destroying, and
which one it is cannot be argued from first principles. Gate it on
`usage_counts[0]` and loss together, the same way §9 requires `live` and loss to
be read as a pair — a run where most blocks pin to zero and the loss collapses
is the failure mode to watch for.

**Ledger note:** `K` is unchanged, so `bytes_per_weight` does not move. One of
the K slots is reserved rather than added.


---

# Phase D verdict (post D3–D8, post F1/F2)

D3–D8 are fixed (all standalone, no dose recalibration needed — D1/D2 were
already fixed earlier in this working tree). Full suite: 120/122 passing; the
2 failures (`test_lifecycle_every_zero_is_bit_for_bit_phase_c`,
`test_cohort_frac_zero_reproduces_phase_b_bit_for_bit`) are pre-existing
GPU/XLA float32 reassociation flakes, reproduced identically on a clean `main`
via `git stash` before any of this work — not a regression.

**F2** (dedup-at-init control, S=1 only): dedup dramatically improved
`static`/`single` (up to 5.6x at K=1024) but improved `lifecycle` far less,
flipping the S=1/K=1024 ranking from lifecycle-wins to single-wins.

**F1 re-run with `dedup_init=true`** (all three original F1 cells, 3 seeds each):

| Cell | static | single | lifecycle | best | verdict |
|---|---|---|---|---|---|
| S=1/K=1024 | 0.00430 | 0.00525 ± 0.00185 | 0.00532 ± 0.00127 | static | INCONCLUSIVE (gap 0.00007) |
| S=2/K=1024 | 0.00500 | 0.00418 ± 0.00032 | 0.00481 ± 0.00460 | single | INCONCLUSIVE (gap 0.00063) |
| S=4/K=256  | 0.00692 | 0.00457 ± 0.00156 | 0.01085 ± 0.00772 | single | INCONCLUSIVE (gap 0.00629) |

Every cell is still formally INCONCLUSIVE by F1's spread-vs-gap rule. But the
means are decisive in direction: `single` beats `lifecycle` in **all three**
cells post-dedup (it did not pre-dedup — lifecycle had the edge at S=1/K=1024
and S=2/K=1024 in the original 3-seed run), and at S=4/K=256 `lifecycle` is now
the worst arm by a wide margin (1.57x worse than static, vs single's 0.66x).
`lifecycle` beats `static` in only 1/3 cells; `single` beats `static` in 2/3.

**Verdict: Phase D is recorded as NOT PASSED.** Once the init-duplication
confound (F2's hypothesis #2) is controlled for, the lifecycle merge/drop
machinery shows no measured edge over the much simpler per-step `single` swap
rule, and actively underperforms it at S=4/K=256. The mechanism is implemented
correctly (D1–D8 fixed, unit-tested, invariants hold) but has not demonstrated
value in this regime. Further `merge_quantile`/`merge_eps` tuning is not
recommended until a cell shows a gap that survives more seeds — S=4/K=256 is
the closest (spread 0.00772 vs gap 0.00629) and the cheapest place to add
seeds if this is revisited.

**Next SPEC phase:** Phase E (Mamba2 backbone integration) is unblocked
either way — lifecycle is opt-in (`lifecycle_every: 0` disables it, reducing
to `single`), so Phase E does not depend on this verdict. Work continuing
after this point ([[momos-memory-efficiency]] below) targets inference/training
memory rather than Phase E or further lifecycle tuning, per explicit user
redirection.

---

## Verified clean — do not re-investigate

The review checked and confirmed these hold; they are recorded so the next pass
does not spend time on them:

- The dtype-preservation assert holds for `uint8`, `uint16` and `uint32`.
- `np.bincount` accepts all three mosaic dtypes.
- The `min_live_frac` floor and the monotonicity invariant hold across both
  merge and drop.
- Union-find root selection is genuinely min-index.
- `_propose_swaps`' `state.active[nbrs]` mask means a motif deactivated by
  lifecycle can never be swapped back onto.
- Lifecycle-induced `active`/`mosaic` changes do not retrigger a jit recompile.
