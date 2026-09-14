# Phase E — MoMos on the real Mamba2 backbone

**Status: landed end-to-end, behind a config flag; dense stays default.**
`src/train_jax.py`'s `run_fold` now takes `momos_backbone.enabled=true` as a
real Hydra override (`src/config/momos_backbone/default.yaml`) and routes to
`_run_momos_fold`, a full training-loop implementation parallel to the dense
path (own jitted step/eval, own checkpointing, same `DistillerDataModule`/
`DataLoader` plumbing). Composing the root config with **no** overrides
leaves `momos_backbone.enabled: false`, so every existing run is unaffected
— checked directly by `test_momos_backbone_is_off_by_default`. This records
what `model_jax.momos.integration` implements, what it was tested against,
and what SPEC.md §8's Phase E gate still needs before this is a pass, not
just a working prototype.

## What's implemented

`model_jax.momos.integration` (`integration.py`) generalises `train_step.py`
from `model_jax.toy_tasks`' flat SSM regressor to any real `nnx.Module`:

- **`ModelBundle`** splits a model via `nnx.split(model, nnx.Param, ...)` into
  `rest` (non-`nnx.Param` state — e.g. `mimir_mamba2`'s frozen `Donor`
  tables) and the dictionary-managed params, and carries a *third* bucket,
  `excluded` — the `nnx.Param` leaves `tiling.default_include` rejects
  (`A_log`, `D`, `dt_bias`, norm weights). Phases A-D left those frozen at
  init (documented, deliberate, and fine for a toy task's norm weight); on a
  real SSM, `A_log` sets the recurrence's decay rate and has to train.
- **`step`** trains the dictionary and the excluded leaves in the *same*
  backward pass — `jax.value_and_grad` over three argnums (gathered blocks,
  scales, excluded leaves) — so enabling MoMos does not double a step's
  backward cost the way training them via two separate `value_and_grad`
  calls would.
- `tiling.unflatten_params` / `train_step.reconstruct` gained an
  `excluded_values` override (default `None`, bit-for-bit unchanged) so
  reconstruction can restore the excluded leaves' *trained* values rather
  than their frozen init.
- **Found and fixed while wiring this**: `tiling.default_include`'s
  `norm_f/weight` exclusion only matched the exact top-level path
  `("norm_f", "weight")`; the real model nests it under `("backbone",
  "norm_f", "weight")`, so that scale-sensitive tensor was silently being
  mosaicked. Fixed to match by suffix, like the `norm/weight` rule already
  does (`tiling.py`, `test_tiling.py::test_default_include_excludes_norm_f_weight_when_nested`).

## The Hydra wiring

- `src/config/momos_backbone/default.yaml` — a new config group,
  `enabled: false` by default, carrying every `MosaicConfig` field
  (`S`, `K`, `scale_mode`, `subset_size`, `margin`, `n_neighbors`,
  `reserve_zero_motif`, `dedup_init`) plus one integration-only knob,
  `dict_lr_mult` (the dictionary's own learning rate is
  `training.learning_rate * dict_lr_mult` — see "Measured result" below for
  why this needs its own value, not the dense arm's).
- `train_jax.run_fold` reads `cfg.momos_backbone.enabled` right after
  computing `total_steps` and, if true, returns early into
  `_run_momos_fold(...)` — a separate function, not an `if`/`else` woven
  through the rest of `run_fold`, so the dense path's existing code is
  unmoved and untouched by this change; enabling MoMos can only be reached
  by explicitly setting the flag.
- `_run_momos_fold` builds `MosaicConfig` via `_momos_config_from_cfg`
  (filters `momos_backbone`'s yaml down to real `MosaicConfig` fields, the
  same pattern `scripts/momos_phase_c.py::_mosaic_config` already uses),
  then `integration.init_bundle` + `integration.make_jit_step`/
  `make_jit_eval` for a jit-compiled loop, and reuses the existing
  `DistillerDataModule`/checkpoint-manager/Optuna-pruning code paths
  unchanged.
- **Why a separate jit-wrapping layer** (`integration.make_jit_step`,
  `make_jit_eval`, `bundle_arrays`, `bundle_from_arrays`, all in
  `integration.py`): `MosaicState`/`ModelBundle` are plain dataclasses, not
  registered JAX pytrees — `ParamLayout` holds a `numpy` array, which is
  unhashable, and a pytree's static/aux data must be hashable, so
  registration is not just undone, it is not available. `scripts/
  momos_phase_c.py`'s `core_drift`/`core_single` already solved this by
  closing `layout`/`cfg` over from the enclosing scope and jitting a
  function whose real arguments are only array leaves; `make_jit_step`/
  `make_jit_eval` package that same idiom once, tested directly against
  eager `integration.step` for numerical equivalence
  (`test_make_jit_step_matches_eager_step`, `test_make_jit_eval_matches_
  merged_model_loss` — both to `atol=1e-5`, SPEC.md §8's own multi-tensor
  eager-vs-jit-reassociation caveat, not a new tolerance invented here).
- **Checkpointing** saves the *reconstructed dense* params
  (`integration.merged_model`), not the compact dictionary, so
  `predict.py`/`model_jax/export.py` keep working unchanged on a MoMos
  checkpoint. Saving the compact form instead — the actual memory win —
  is real future work tied to `SPEC_PHASE_PERF.md`, not part of this
  landing.
- Lifecycle (merge/drop) is **not** reachable from `momos_backbone` yet —
  only `subset_size`/`margin` (static swap) knobs are wired. Per this
  project's own Phase D verdict (`SPEC_PHASE_D_DEFECTS.md`), lifecycle has
  not demonstrated value over plain swapping, so wiring it into a real
  training loop before that changes would be adding untested complexity to
  chase a mechanism that has not earned it yet.

## What it was tested against

`tests/model_jax/momos/test_integration.py` (12 tests) — the real
`model_jax.mimir_mamba2.MimirMamba2Model` architecture (Mamba2 backbone +
donor projections), with toy donor tables (same pattern as
`tests/model_jax/test_mimir_mamba2_jax.py` — no network call). Covers: donor
tables never move, excluded leaves do move, reconstruction matches an
independent hand-rebuild, loss decreases over training, and the jit-wrapped
core (`make_jit_step`/`make_jit_eval`) matches eager `step`/`merged_model`.

`tests/model_jax/momos/test_train_jax_integration.py` (5 tests) — the actual
Hydra wiring: config composes with `momos_backbone.enabled: false` by
default (and stays that way with no override), composes correctly with
overrides, `_momos_config_from_cfg` builds a valid `MosaicConfig` from the
composed config, and `train_jax._run_momos_fold` itself — the function
`run_fold` calls — runs a real short training loop (synthetic data, real
Hydra-composed `cfg`, real checkpoint write to a temp dir) and drives the
loss down over more epochs than fewer.

`scripts/momos_phase_e.py` — the same real architecture, trained on the
**real** district-heating ground-truth data (`data/data_district_heating.xlsx`,
the same file `src/train.py`'s PyTorch path uses), tokenized with a
deterministic fake tokenizer (no real donor tokenizer is available in this
environment — `outputs/donor_jax` does not exist here, and building it
needs a network download). What's real: the text, the architecture, the
loss, the training loop. What's a stand-in: the tokenizer and the donor
tables' *values* (not their role).

## Measured result — not yet a gate pass

One run, 124,962 trainable params, S=1/K=4096 (2.4 B/wt, 5.0x dense, K/M ≈
3.3% — in target regime), 150 steps, `lr=3e-3` for both arms:

```
step   0:  dense=7.63  momos=7.26   (momos starts lower — motif seeding
                                      from real blocks helps here)
step  74:  dense=1.16  momos=1.42   ratio 1.23x
step 149:  dense=1.15  momos=3.15   ratio 2.73x
```

MoMos learns (loss drops from init) but does **not** track the dense curve —
it degrades again after step ~75 rather than continuing to converge.
Tripling the dictionary optimiser's learning rate relative to the excluded
leaves' moved the mid-run ratio closer (1.83x at step 111) but reproduced the
same late-run divergence (2.73x at step 149) — so this looks like an
optimisation/stability issue (dictionary LR, schedule, or Adam moment
resets — none tried yet), not evidence the mechanism cannot work here.

**This is not SPEC.md §8's Phase E gate pass.** The gate wants the loss curve
to *track* dense at matched hyperparameters; a 2-3x gap that grows over
training does not qualify. Recorded honestly rather than as a pass, per this
project's own convention (`SPEC_PHASE_D_DEFECTS.md`'s F1/F2 verdicts).

## Before calling this gate passed

1. A learning-rate sweep for the dictionary optimiser specifically (not
   reused from the dense arm's `lr` — the two are updating structurally
   different things and there is no reason to expect the same value is
   right for both).
2. A cosine/warmup schedule, matching `train_jax.py`'s dense path, rather
   than the flat LR both arms use here — Phase C/D's toy-task gates never
   needed one at 300 steps; this run's late-training divergence may be
   exactly what a decay schedule exists to prevent.
3. More than one seed — `SPEC_PHASE_D_DEFECTS.md` §F1 already established
   that this project's own single-seed grids are not enough to separate a
   real effect from noise; the same caution applies here.
4. Only once (1)-(3) are done: SPEC.md §8's actual thesis test — a MoMos
   model with ~10x the parameters, trained in the same memory budget,
   beating the dense model that fits without it. Nothing here attempted
   that; this is a mechanism smoke test, not the thesis test.
