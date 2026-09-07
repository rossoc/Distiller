# -*- coding: utf-8 -*-
"""Training core for Distiller — autoregressive DFM-Mimir fine-tuning.

This module exposes exactly one reusable entry point: ``run_fold``. It knows
how to train (and validate) DFM-Mimir on a single K-fold CV fold, optionally
under an Optuna trial. It does NOT know about K-fold orchestration or
hyperparameter search — those live in ``cv.py`` and ``optuna_search.py``
respectively, both of which import ``run_fold`` from here.

    # K-fold CV (no Optuna)
    python src/cv.py

    # Optuna search (K-fold CV per trial)
    python src/optuna_search.py optuna.n_trials=50

    # Override anything via Hydra
    python src/cv.py data.test_frac=0.2 training.learning_rate=5.0e-5
"""

import inspect
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Silence noisy third-party warnings / tips that only obscure real errors.
# ---------------------------------------------------------------------------
# Lightning 2.6 prints "💡 Tip: ... litlogger/litmodels" INFO messages via its
# rank_zero logger; bump that logger to WARNING so they don't clutter output.
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.fabric.utilities.rank_zero").setLevel(logging.WARNING)

# Deprecation warnings inside Lightning internals (harmless for our usage).
warnings.filterwarnings(
    "ignore",
    message=".*isinstance\\(treespec, LeafSpec\\).*",
)
warnings.filterwarnings(
    "ignore",
    message=".*does not have many workers which may be a bottleneck.*",
)
# bf16-mixed precision just makes the model-summary size estimate inexact —
# harmless, not worth a warning every fold.
warnings.filterwarnings(
    "ignore",
    message=".*Precision bf16-mixed is not supported by the model summary.*",
)

import hydra
import optuna
import torch
import torch.multiprocessing
from omegaconf import DictConfig, OmegaConf, open_dict

# Allow TF32 matmuls/cudnn on Ampere+ (Blackwell) for a free speed boost.
# Relevant to any process that imports this module directly (not only the
# cv.py/optuna_search.py CLIs, which additionally call
# utils.configure_cuda_fast_path() for the benchmark-mode knob).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from callback import OptunaPruningCallback, EpochProgressCallback, _best_checkpoint
from lit_datamodule import DistillerDataModule
from model.dfm_mimir import DFMMimirModule
from utils import dataloader_runtime, format_seconds, seed_all

log = logging.getLogger(__name__)

torch.multiprocessing.set_sharing_strategy("file_system")


# ---------------------------------------------------------------------------
# Trial hyperparameter merging
# ---------------------------------------------------------------------------


def _apply_trial_overrides(
    model_kwargs: Dict[str, Any],
    cfg: DictConfig,
    trial_params: Dict[str, Any],
) -> None:
    """Merge Optuna's suggested hyperparameters into *model_kwargs* / *cfg.training*.

    A suggested param overrides the corresponding ``model_kwargs`` entry when
    one exists by that name, and/or the ``cfg.training`` entry when one exists
    by that name — a param can hit either, both, or neither (search-space
    names don't have to match either one).
    """
    for key, value in trial_params.items():
        if key in model_kwargs:
            model_kwargs[key] = value
        if key in cfg.training:
            with open_dict(cfg.training):
                cfg.training[key] = value


# ---------------------------------------------------------------------------
# Single CV fold — the one function this module exposes
# ---------------------------------------------------------------------------


def run_fold(
    cfg: DictConfig,
    fold_idx: int,
    trial: Optional[optuna.Trial] = None,
) -> float:
    """Train one CV fold and return the best validation loss.

    Builds the model, a ``DistillerDataModule`` scoped to ``fold_idx``, and a
    Lightning ``Trainer``, then fits and returns ``eval_loss``.

    If *trial* is given, uses Optuna pruning callbacks and no checkpoint is
    written (avoids dozens of trial x fold checkpoint files during search).
    Otherwise this fold's single best-on-validation-loss checkpoint is saved
    to ``{cfg.training.output_dir}/fold_{fold_idx}``.

    Each fold/trial is re-seeded independently (offset from the base seed by
    fold and trial identity) so it is reproducible in isolation, rather than
    depending on how much randomness every prior fold/trial consumed from the
    shared global RNG stream.
    """
    offset = fold_idx + (trial.number * 1000 if trial is not None else 0)
    seed_all(cfg.training.seed + offset)

    model_kwargs: Dict[str, Any] = {
        "model_id": cfg.model.model_id,
        "trust_remote_code": cfg.model.trust_remote_code,
        "dtype": cfg.model.dtype,
        "learning_rate": cfg.training.learning_rate,
        "weight_decay": cfg.training.weight_decay,
        "warmup_ratio": cfg.training.warmup_ratio,
        "lr_scheduler": cfg.training.lr_scheduler,
        "hrm_cycles": (
            dict(cfg.model.hrm_cycles)
            if getattr(cfg.model, "hrm_cycles", None) is not None
            else None
        ),
    }

    if trial is not None:
        _apply_trial_overrides(model_kwargs, cfg, trial.params)

    module = DFMMimirModule(**{
        k: v
        for k, v in model_kwargs.items()
        if k in inspect.signature(DFMMimirModule.__init__).parameters
    })

    # Build the fold's dataloaders via the shared datamodule — the single
    # source of truth for row-splitting/K-fold-indexing/tokenization/
    # DataLoader construction (worker persistence, pinning, prefetching).
    datamodule = DistillerDataModule(
        cfg, fold=fold_idx, runtime=dataloader_runtime(cfg.runtime)
    )
    datamodule.set_module(module)
    datamodule.setup()
    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()

    callbacks: List[Any] = [EpochProgressCallback(log_every_n_epochs=1)]
    if trial is not None:
        callbacks.append(OptunaPruningCallback(trial, monitor="eval_loss"))
        # Suppress progress bar + model summary during search
        trainer_cfg = OmegaConf.create(
            OmegaConf.to_container(cfg.trainer, resolve=True)
        )
        trainer_cfg.enable_progress_bar = False
        trainer_cfg.enable_model_summary = False
    else:
        # Not an Optuna trial: keep exactly one checkpoint for this fold —
        # its own best-on-validation-loss epoch.
        callbacks.append(
            _best_checkpoint(str(Path(cfg.training.output_dir) / f"fold_{fold_idx}"))
        )
        trainer_cfg = cfg.trainer

    # Optional W&B logger. NOTE: must be False, not None, to actually disable
    # logging — Lightning's Trainer.__init__ treats logger=None as "use the
    # default logger" (attaches a CSVLogger), only logger=False means "no
    # logger" (which is what _log()'s "stay silent when disabled" logic
    # assumes).
    logger: Any = False
    if getattr(cfg, "wandb", None) and cfg.wandb.enabled:
        from lightning.pytorch.loggers import WandbLogger

        # One run per fold (grouped so folds can be selected together and
        # compared/aggregated in the W&B UI), rather than every fold's
        # metrics landing on a single run. `reinit=True` forces a fresh
        # wandb run here even though the previous fold's run object is still
        # alive in this process (we finish() it below, but that belt-and-
        # braces this against a differently-ordered/early-returning caller).
        run_name = cfg.wandb.name or cfg.wandb.group or "cv"
        run_name = f"{run_name}_fold{fold_idx}"
        if trial is not None:
            run_name = f"{run_name}_trial{trial.number}"

        logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity or None,
            name=run_name,
            group=cfg.wandb.group or None,
            job_type="trial" if trial is not None else "cv",
            offline=cfg.wandb.offline,
            tags=list(cfg.wandb.tags or []),
            reinit=True,
        )

    trainer = hydra.utils.instantiate(
        trainer_cfg,
        callbacks=callbacks,
        logger=logger,
        # The Trainer target is fixed (config.yaml always points it at
        # lightning.pytorch.Trainer) — whitelist exactly that target rather
        # than the blanket (and, previously, ineffective) all-targets bypass.
        _target_whitelist_="lightning.pytorch.Trainer",
    )

    start = time.time()
    trainer.fit(module, train_dl, val_dl)
    elapsed = format_seconds(time.time() - start)

    if trial is not None:
        log.info(
            "  fold done in %s — val_loss=%.4f",
            elapsed,
            float(trainer.callback_metrics.get("eval_loss", float("inf"))),
        )

    val_loss = trainer.callback_metrics.get("eval_loss")
    result: float
    if val_loss is not None:
        result = float(val_loss.cpu().item())
    else:
        # Fallback from logger connector
        logs = trainer.logger_connector._metrics if trainer.logger_connector else {}
        if "eval_loss" in logs:
            values = logs["eval_loss"]
            result = float(values[-1]) if values else float("inf")
        else:
            result = float("inf")

    # Close out this fold's W&B run so the next fold's WandbLogger starts a
    # fresh run instead of reusing this (still-active-in-process) one.
    if logger:
        logger.experiment.finish()

    # Release the model + trainer from GPU before the next trial/fold loads,
    # so VRAM doesn't accumulate across repeated loads in one process.
    del module, trainer, train_dl, val_dl, datamodule
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result
