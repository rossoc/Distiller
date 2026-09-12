# -*- coding: utf-8 -*-
"""K-fold cross-validation orchestration for the JAX path.

Mirrors ``src/cv.py`` one-for-one: loops ``train_jax.run_fold`` over every fold
and aggregates validation loss. Used directly for a plain CV run (this file's
``python src/cv_jax.py`` CLI), and imported by ``optuna_search_jax.py`` so each
trial reuses the same fold loop with per-fold pruning.

    python src/cv_jax.py
    python src/cv_jax.py data=rnn training.learning_rate=5.0e-5
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import optuna
from omegaconf import DictConfig, OmegaConf

from data.loader import count_by_target
from lit_datamodule import DistillerDataModule
from train_jax import run_fold
from utils import seed_all

log = logging.getLogger(__name__)


def run_kfold_cv(
    cfg: DictConfig, trial: Optional[optuna.Trial] = None
) -> Dict[str, Any]:
    """Run K-fold CV once (optionally under an Optuna trial) and summarize it."""
    # Sample-only datamodule (no set_module() call, so no tokenizer is loaded)
    # purely for the dataset-size/target-count summary stats below.
    stats_dm = DistillerDataModule(cfg, fold=-1)
    stats_dm.setup()

    fold_losses: List[float] = []
    fold_results: List[Dict[str, Any]] = []

    for fold_idx in range(cfg.optuna.n_folds):
        if trial is not None and trial.should_prune():
            raise optuna.TrialPruned()

        if trial is None:
            log.info("=== Fold %d/%d ===", fold_idx + 1, cfg.optuna.n_folds)

        loss = run_fold(cfg, fold_idx, trial)
        fold_losses.append(loss)
        fold_results.append({"fold": fold_idx + 1, "val_loss": loss})

        if trial is not None:
            trial.report(float(np.mean(fold_losses)), fold_idx + 1)

    train_samples = stats_dm.get_train_samples()
    test_samples = stats_dm.get_test_samples()

    return {
        "n_folds": cfg.optuna.n_folds,
        "fold_results": fold_results,
        "fold_val_losses": fold_losses,
        "mean_val_loss": float(np.mean(fold_losses)),
        "std_val_loss": float(np.std(fold_losses)),
        "min_val_loss": float(np.min(fold_losses)),
        "max_val_loss": float(np.max(fold_losses)),
        "n_train_samples": len(train_samples),
        "n_test_samples": len(test_samples),
        "target_counts_train": count_by_target(train_samples),
        "target_counts_test": count_by_target(test_samples),
    }


# ---------------------------------------------------------------------------
# Hydra entrypoint — plain K-fold CV run (no Optuna)
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

    # No resolve_device()/configure_cuda_fast_path() here: those are torch
    # knobs. JAX picks its backend from the installed jaxlib, and the mesh
    # (cfg.mesh_shape) is what decides device use — see model_jax/sharding.py.
    seed_all(cfg.training.seed)

    out_dir = Path(cfg.training.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Running K-fold CV (%d folds, no Optuna)...", cfg.optuna.n_folds)
    summary = run_kfold_cv(cfg, trial=None)

    summary_path = out_dir / "cv_results_jax.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Saved CV results to %s", summary_path)
    log.info(
        "Validation loss: %.6f ± %.6f  (min=%.6f max=%.6f, n=%d folds)",
        summary["mean_val_loss"],
        summary["std_val_loss"],
        summary["min_val_loss"],
        summary["max_val_loss"],
        summary["n_folds"],
    )


if __name__ == "__main__":
    main()
