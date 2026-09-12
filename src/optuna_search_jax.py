# -*- coding: utf-8 -*-
"""Optuna hyperparameter search for the JAX path.

Mirrors ``src/optuna_search.py``: each trial suggests hyperparameters, runs a
full K-fold CV via ``cv_jax.run_kfold_cv`` (with per-fold pruning), and reports
the mean validation loss.

    python src/optuna_search_jax.py optuna.n_trials=50

The search-space dispatch, ``MedianPruner`` and SQLite storage are reused
verbatim. **One thing is not reusable**, and it is the kind of difference that
only shows up on a bad night: ``optuna_search.py`` passes
``catch=(torch.OutOfMemoryError,)`` so a trial that pushes ``batch_size`` or
``d_model`` too far is recorded as failed instead of killing the study. JAX
raises its own exhaustion error instead, so the equivalent type is resolved
below — without it, the first OOM trial would take the whole search down.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple, Type

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf

from cv_jax import run_kfold_cv
from utils import seed_all

log = logging.getLogger(__name__)


def _oom_exceptions() -> Tuple[Type[BaseException], ...]:
    """Exception types that mean "this trial did not fit", not "the run is broken".

    JAX surfaces device-memory exhaustion as ``XlaRuntimeError`` (a
    RESOURCE_EXHAUSTED status) from jaxlib. The import path has moved between
    jaxlib versions, so it is resolved defensively — a search that loses its
    OOM handling to an ImportError would be worse than one that catches
    slightly too much.
    """
    caught: list = [MemoryError]
    try:
        from jaxlib.xla_extension import XlaRuntimeError  # type: ignore

        caught.append(XlaRuntimeError)
    except ImportError:  # pragma: no cover - depends on jaxlib layout
        try:
            from jax.errors import JaxRuntimeError  # type: ignore

            caught.append(JaxRuntimeError)
        except ImportError:
            log.warning(
                "Could not resolve JAX's runtime-error type; an out-of-memory "
                "trial will fail the whole study instead of being recorded as "
                "a failed trial."
            )
    return tuple(caught)


def objective(trial: optuna.Trial, cfg: DictConfig) -> float:
    """Optuna objective: K-fold CV mean validation loss."""
    search = cfg.optuna.search_space
    for param_name, spec in search.items():
        if spec._target_.endswith("suggest_float"):
            log_flag = spec.get("log", False)
            trial.suggest_float(param_name, spec.low, spec.high, log=log_flag)
        elif spec._target_.endswith("suggest_int"):
            trial.suggest_int(param_name, spec.low, spec.high)
        elif spec._target_.endswith("suggest_categorical"):
            trial.suggest_categorical(param_name, spec.choices)

    summary = run_kfold_cv(cfg, trial)
    return summary["mean_val_loss"]


# ---------------------------------------------------------------------------
# Hydra entrypoint — Optuna search
# ---------------------------------------------------------------------------


@hydra.main(
    config_path=str(Path(__file__).parent / "config"),
    config_name="config_jax",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    seed_all(cfg.training.seed)

    out_dir = Path(cfg.training.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Starting Optuna search: %d trials × %d-fold CV",
        cfg.optuna.n_trials,
        cfg.optuna.n_folds,
    )

    storage_url = cfg.optuna.storage or None

    pruner_cls_name = cfg.optuna.pruner._target_.split(".")[-1]
    pruner_cls = getattr(optuna.pruners, pruner_cls_name)
    pruner_kwargs = {k: v for k, v in cfg.optuna.pruner.items() if k != "_target_"}
    pruner = pruner_cls(**pruner_kwargs)

    study = optuna.create_study(
        study_name=cfg.optuna.study_name,
        storage=storage_url,
        direction=cfg.optuna.mode,
        pruner=pruner,
        load_if_exists=True,
    )

    def optuna_objective(trial: optuna.Trial) -> float:
        return objective(trial, cfg)

    study.optimize(
        optuna_objective,
        n_trials=cfg.optuna.n_trials,
        show_progress_bar=True,
        catch=_oom_exceptions(),
    )

    log.info("Optuna study complete.")
    log.info("Best value: %.6f", study.best_value)
    log.info("Best params: %s", study.best_params)

    results = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "direction": cfg.optuna.mode,
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name,
            }
            for t in study.trials
        ],
    }
    results_path = out_dir / "optuna_study_results_jax.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved Optuna results to %s", results_path)


if __name__ == "__main__":
    main()
