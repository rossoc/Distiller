# -*- coding: utf-8 -*-
"""Optuna hyperparameter search orchestration for Distiller.

Each trial suggests hyperparameters, then runs a full K-fold CV (via
``cv.run_kfold_cv``, with per-fold pruning) and reports the mean validation
loss across folds as the trial's objective value.

    python src/optuna_search.py optuna.n_trials=50
"""

import json
import logging
from pathlib import Path

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf

from cv import run_kfold_cv
from utils import configure_cuda_fast_path, resolve_device, seed_all

log = logging.getLogger(__name__)


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
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    resolve_device(cfg.trainer.accelerator)
    configure_cuda_fast_path(enable_benchmark=bool(cfg.get("compile", False)))
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
        optuna_objective, n_trials=cfg.optuna.n_trials, show_progress_bar=True
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
    results_path = out_dir / "optuna_study_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved Optuna results to %s", results_path)


if __name__ == "__main__":
    main()
