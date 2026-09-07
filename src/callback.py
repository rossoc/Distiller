# -*- coding: utf-8 -*-
"""Factory helpers for Lightning callbacks, including Optuna pruning."""

from __future__ import annotations

from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

from optuna import TrialPruned
from optuna.trial import Trial


class EpochProgressCallback(Callback):
    """Lightweight per-epoch heartbeat so training is visibly progressing even
    when the progress bar is disabled (e.g. during an Optuna search).

    Logs one INFO line at the end of each training epoch with the latest
    train/eval losses pulled from ``trainer.callback_metrics``.
    """

    def __init__(self, log_every_n_epochs: int = 1) -> None:
        self.log_every_n_epochs = log_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        import logging

        _log = logging.getLogger("distiller.epoch")
        train_loss = trainer.callback_metrics.get("train_loss")
        eval_loss = trainer.callback_metrics.get("eval_loss")
        _log.info(
            "epoch %d/%d — train_loss=%.4f eval_loss=%s",
            trainer.current_epoch + 1,
            trainer.max_epochs,
            float(train_loss) if train_loss is not None else float("nan"),
            f"{float(eval_loss):.4f}" if eval_loss is not None else "n/a",
        )


class OptunaPruningCallback(Callback):
    """Report ``monitor`` to an Optuna trial after every validation epoch and
    raise ``TrialPruned`` when the trial should be pruned.

    A local reimplementation, not ``optuna_integration``'s
    ``PyTorchLightningPruningCallback`` — avoids a second extra dependency
    and its history of breaking across Lightning versions.

    Inherits from ``lightning.pytorch.callbacks.Callback`` so Lightning 2.6+
    finds the standard hook stubs (``setup``, ``on_exception``, ...) it calls
    on every callback; without this base class Lightning raises
    ``AttributeError: object has no attribute 'setup'`` / ``'on_exception'``.
    """

    def __init__(self, trial: Trial, monitor: str = "eval_loss") -> None:
        self.trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        epoch = trainer.current_epoch
        self.trial.report(float(value), step=epoch)
        if self.trial.should_prune():
            raise TrialPruned(
                f"trial {self.trial.number} pruned at epoch {epoch} "
                f"({self.monitor}={float(value):.4f})"
            )


def _best_checkpoint(checkpoint_dir: str) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="eval_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )


def _early_stopping(patience: int | None) -> EarlyStopping | None:
    if patience is None or patience <= 0:
        return None
    return EarlyStopping(
        monitor="eval_loss", mode="min", patience=patience, verbose=False
    )


def build_callbacks(checkpoint_dir: str, patience: int | None = None) -> list:
    """Standard callback set: best-checkpoint saving + optional early stopping."""
    callbacks = [_best_checkpoint(checkpoint_dir)]
    early_stopping = _early_stopping(patience)
    if early_stopping is not None:
        callbacks.append(early_stopping)
    return callbacks
