# -*- coding: utf-8 -*-
"""K-fold cross-validation orchestration for Distiller.

Loops ``train.run_fold`` over every fold and aggregates validation loss.
Used directly for a plain CV run (this file's ``python src/cv.py`` CLI), and
imported by ``optuna_search.py`` so each Optuna trial reuses the exact same
fold loop (with per-fold pruning) instead of a second copy of it.
"""

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
from train import run_fold
from utils import configure_cuda_fast_path, resolve_device, seed_all

log = logging.getLogger(__name__)


def run_kfold_cv(cfg: DictConfig, trial: Optional[optuna.Trial] = None) -> Dict[str, Any]:
    """Run K-fold CV once (optionally under an Optuna trial) and summarize it.

    If *trial* is given, checks ``trial.should_prune()`` before each fold and
    reports the running mean validation loss after each fold — the same
    per-fold pruning ``objective()`` used to do itself before this loop moved
    here so both the plain CV path and the Optuna path share one
    implementation.
    """
    # Sample-only datamodule (no set_module() call, so no model/tokenizer is
    # loaded) purely for the dataset-size/target-count summary stats below.
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

    log.info("Running K-fold CV (%d folds, no Optuna)...", cfg.optuna.n_folds)
    summary = run_kfold_cv(cfg, trial=None)

    summary_path = out_dir / "cv_results.json"
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
