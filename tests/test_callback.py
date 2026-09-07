# -*- coding: utf-8 -*-
"""Tests for callback.py — using lightweight stub trainer/pl_module objects,
no real Lightning Trainer needed."""

from __future__ import annotations

import logging

import pytest
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from optuna import TrialPruned

from callback import (
    EpochProgressCallback,
    OptunaPruningCallback,
    _best_checkpoint,
    _early_stopping,
    build_callbacks,
)


class _StubTrainer:
    def __init__(self, current_epoch=0, max_epochs=10, metrics=None, sanity_checking=False):
        self.current_epoch = current_epoch
        self.max_epochs = max_epochs
        self.callback_metrics = metrics or {}
        self.sanity_checking = sanity_checking


class _StubTrial:
    def __init__(self, should_prune: bool, number: int = 0):
        self._should_prune = should_prune
        self.number = number
        self.reports = []

    def report(self, value, step):
        self.reports.append((value, step))

    def should_prune(self):
        return self._should_prune


# ---------------------------------------------------------------------------
# EpochProgressCallback
# ---------------------------------------------------------------------------


def test_epoch_progress_callback_logs_every_n_epochs(caplog):
    cb = EpochProgressCallback(log_every_n_epochs=2)
    trainer = _StubTrainer(current_epoch=0, metrics={"train_loss": 1.0})
    with caplog.at_level(logging.INFO, logger="distiller.epoch"):
        cb.on_train_epoch_end(trainer, None)
    # epoch 1 (0-indexed 0) is not a multiple of 2 -> no log
    assert len(caplog.records) == 0


def test_epoch_progress_callback_logs_on_matching_epoch(caplog):
    cb = EpochProgressCallback(log_every_n_epochs=2)
    trainer = _StubTrainer(current_epoch=1, metrics={"train_loss": 1.0, "eval_loss": 2.0})
    with caplog.at_level(logging.INFO, logger="distiller.epoch"):
        cb.on_train_epoch_end(trainer, None)
    assert len(caplog.records) == 1
    assert "eval_loss=2.0000" in caplog.records[0].message


def test_epoch_progress_callback_handles_missing_eval_loss(caplog):
    cb = EpochProgressCallback(log_every_n_epochs=1)
    trainer = _StubTrainer(current_epoch=0, metrics={"train_loss": 1.0})
    with caplog.at_level(logging.INFO, logger="distiller.epoch"):
        cb.on_train_epoch_end(trainer, None)
    assert "n/a" in caplog.records[0].message


# ---------------------------------------------------------------------------
# OptunaPruningCallback
# ---------------------------------------------------------------------------


def test_optuna_pruning_callback_skips_sanity_checking():
    trial = _StubTrial(should_prune=True)
    cb = OptunaPruningCallback(trial, monitor="eval_loss")
    trainer = _StubTrainer(metrics={"eval_loss": 5.0}, sanity_checking=True)
    cb.on_validation_end(trainer, None)  # must not raise
    assert trial.reports == []


def test_optuna_pruning_callback_no_metric_yet_is_noop():
    trial = _StubTrial(should_prune=True)
    cb = OptunaPruningCallback(trial, monitor="eval_loss")
    trainer = _StubTrainer(metrics={})
    cb.on_validation_end(trainer, None)  # must not raise
    assert trial.reports == []


def test_optuna_pruning_callback_reports_and_prunes():
    trial = _StubTrial(should_prune=True)
    cb = OptunaPruningCallback(trial, monitor="eval_loss")
    trainer = _StubTrainer(current_epoch=3, metrics={"eval_loss": 0.5})
    with pytest.raises(TrialPruned):
        cb.on_validation_end(trainer, None)
    assert trial.reports == [(0.5, 3)]


def test_optuna_pruning_callback_reports_without_pruning():
    trial = _StubTrial(should_prune=False)
    cb = OptunaPruningCallback(trial, monitor="eval_loss")
    trainer = _StubTrainer(current_epoch=1, metrics={"eval_loss": 0.9})
    cb.on_validation_end(trainer, None)  # must not raise
    assert trial.reports == [(0.9, 1)]


# ---------------------------------------------------------------------------
# _best_checkpoint / _early_stopping / build_callbacks
# ---------------------------------------------------------------------------


def test_best_checkpoint_config():
    cb = _best_checkpoint("some/dir")
    assert isinstance(cb, ModelCheckpoint)
    assert cb.dirpath.endswith("some/dir")
    assert cb.monitor == "eval_loss"
    assert cb.mode == "min"
    assert cb.save_top_k == 1
    assert cb.save_last is False


def test_early_stopping_none_when_no_patience():
    assert _early_stopping(None) is None
    assert _early_stopping(0) is None
    assert _early_stopping(-1) is None


def test_early_stopping_config_when_patience_given():
    es = _early_stopping(5)
    assert isinstance(es, EarlyStopping)
    assert es.monitor == "eval_loss"
    assert es.mode == "min"
    assert es.patience == 5


def test_build_callbacks_without_early_stopping():
    callbacks = build_callbacks("dir", patience=None)
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], ModelCheckpoint)


def test_build_callbacks_with_early_stopping():
    callbacks = build_callbacks("dir", patience=3)
    assert len(callbacks) == 2
    assert isinstance(callbacks[0], ModelCheckpoint)
    assert isinstance(callbacks[1], EarlyStopping)
