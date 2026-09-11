# -*- coding: utf-8 -*-
"""Tests for train.py::_apply_trial_overrides and _resolve_trainer_precision
— plain functions of a dict/string + config, no real Optuna Trial/model/
Hydra entrypoint needed."""

from __future__ import annotations

from omegaconf import OmegaConf

from train import _apply_trial_overrides, _resolve_trainer_precision


def _cfg():
    return OmegaConf.create(
        {"training": {"learning_rate": 2e-5, "weight_decay": 0.01, "seed": 42}}
    )


def test_overrides_model_kwargs_only():
    model_kwargs = {"learning_rate": 2e-5, "dtype": "bf16"}
    cfg = OmegaConf.create({"training": {}})
    _apply_trial_overrides(model_kwargs, cfg, {"dtype": "fp32"})
    assert model_kwargs["dtype"] == "fp32"


def test_overrides_cfg_training_only():
    model_kwargs = {"learning_rate": 2e-5}
    cfg = _cfg()
    _apply_trial_overrides(model_kwargs, cfg, {"seed": 7})
    assert cfg.training.seed == 7
    # model_kwargs untouched — "seed" isn't a key in it.
    assert "seed" not in model_kwargs


def test_overrides_both_when_key_present_in_each():
    model_kwargs = {"learning_rate": 2e-5}
    cfg = _cfg()
    _apply_trial_overrides(model_kwargs, cfg, {"learning_rate": 5e-5})
    assert model_kwargs["learning_rate"] == 5e-5
    assert cfg.training.learning_rate == 5e-5


def test_ignores_unrelated_trial_params():
    model_kwargs = {"learning_rate": 2e-5}
    cfg = _cfg()
    _apply_trial_overrides(model_kwargs, cfg, {"some_unrelated_param": 123})
    assert model_kwargs == {"learning_rate": 2e-5}
    assert "some_unrelated_param" not in cfg.training


def test_empty_trial_params_is_noop():
    model_kwargs = {"learning_rate": 2e-5}
    cfg = _cfg()
    before = dict(cfg.training)
    _apply_trial_overrides(model_kwargs, cfg, {})
    assert model_kwargs == {"learning_rate": 2e-5}
    assert dict(cfg.training) == before


# ---------------------------------------------------------------------------
# _resolve_trainer_precision
# ---------------------------------------------------------------------------


def test_fp32_dtype_forces_32_true_precision():
    assert _resolve_trainer_precision("fp32", "bf16-true") == "32-true"


def test_fp32_dtype_leaves_matching_precision_alone():
    # Already "32-true" -- no log spam, same value returned.
    assert _resolve_trainer_precision("fp32", "32-true") == "32-true"


def test_bf16_dtype_leaves_precision_untouched():
    assert _resolve_trainer_precision("bf16", "bf16-true") == "bf16-true"
    # Even an unusual choice (e.g. bf16-mixed) is the caller's call to make
    # for the bf16 case -- only fp32 is forced.
    assert _resolve_trainer_precision("bf16", "bf16-mixed") == "bf16-mixed"


def test_missing_dtype_leaves_precision_untouched():
    # model.dtype not set at all (falls back to MimirMamba2Module's own
    # "bf16" default) must not be mistaken for the fp32 case.
    assert _resolve_trainer_precision(None, "bf16-true") == "bf16-true"
