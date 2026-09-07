# -*- coding: utf-8 -*-
"""Tests for model/dfm_mimir.py — configure_optimizers' warmup/scheduler
dispatch, the HRM-cycles config patch in __init__, and the _log helper.

None of this downloads the real HF model: configure_optimizers/_log are
exercised on a bare object built via __new__ (a tiny real nn.Linear-based
model for AdamW construction, no HF download); the HRM-cycles test
monkeypatches AutoTokenizer/AutoModelForCausalLM so the real __init__ runs
end-to-end against fakes.
"""

from __future__ import annotations

from types import SimpleNamespace

import lightning as L
import pytest
import torch
import torch.nn as nn

import model.dfm_mimir as dfm_mimir_mod
from model.dfm_mimir import DFMMimirModule


def _bare() -> DFMMimirModule:
    """A DFMMimirModule instance with LightningModule/nn.Module machinery
    initialized but none of the real __init__ body (no tokenizer/model
    download) — enough to exercise methods that only touch a few attributes.
    """
    bare = DFMMimirModule.__new__(DFMMimirModule)
    L.LightningModule.__init__(bare)
    return bare


# ---------------------------------------------------------------------------
# configure_optimizers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.layer_norm = nn.LayerNorm(4)


def _bare_module(lr_scheduler="cosine", warmup_ratio=0.1, total_steps=100):
    bare = _bare()
    bare.model = _TinyModel()
    # `hparams` has no setter (it's a Lightning property backed by
    # `_hparams`, populated normally by save_hyperparameters()) — write the
    # backing attribute directly.
    bare._hparams = SimpleNamespace(
        weight_decay=0.01,
        learning_rate=1e-4,
        warmup_ratio=warmup_ratio,
        lr_scheduler=lr_scheduler,
    )
    bare.trainer = SimpleNamespace(estimated_stepping_batches=total_steps)
    return bare


def test_configure_optimizers_groups_params_by_no_decay():
    bare = _bare_module()
    result = DFMMimirModule.configure_optimizers(bare)
    optimizer = result["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)
    decay_group, no_decay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == 0.01
    assert no_decay_group["weight_decay"] == 0.0
    # linear.weight -> decay; linear.bias + both layer_norm params -> no-decay.
    assert len(decay_group["params"]) == 1
    assert len(no_decay_group["params"]) == 3


def test_configure_optimizers_warmup_steps_and_cosine_dispatch(monkeypatch):
    calls = {}

    def fake_cosine(optimizer, num_warmup_steps, num_training_steps):
        calls.update(warmup=num_warmup_steps, total=num_training_steps)
        return "COSINE"

    monkeypatch.setattr(dfm_mimir_mod, "get_cosine_schedule_with_warmup", fake_cosine)
    bare = _bare_module(lr_scheduler="cosine", warmup_ratio=0.2, total_steps=50)
    result = DFMMimirModule.configure_optimizers(bare)

    assert calls == {"warmup": 10, "total": 50}  # int(50 * 0.2)
    assert result["lr_scheduler"]["scheduler"] == "COSINE"
    assert result["lr_scheduler"]["interval"] == "step"
    assert result["lr_scheduler"]["frequency"] == 1


def test_configure_optimizers_linear_dispatch(monkeypatch):
    calls = {}

    def fake_linear(optimizer, num_warmup_steps, num_training_steps):
        calls.update(warmup=num_warmup_steps, total=num_training_steps)
        return "LINEAR"

    monkeypatch.setattr(dfm_mimir_mod, "get_linear_schedule_with_warmup", fake_linear)
    bare = _bare_module(lr_scheduler="linear", warmup_ratio=0.1, total_steps=200)
    result = DFMMimirModule.configure_optimizers(bare)

    assert calls == {"warmup": 20, "total": 200}
    assert result["lr_scheduler"]["scheduler"] == "LINEAR"


def test_configure_optimizers_unknown_scheduler_yields_no_scheduler():
    bare = _bare_module(lr_scheduler="none")
    result = DFMMimirModule.configure_optimizers(bare)
    assert "lr_scheduler" not in result
    assert "optimizer" in result


# ---------------------------------------------------------------------------
# _log
# ---------------------------------------------------------------------------


def _bare_with_log():
    bare = _bare()
    logged = {}
    bare.log = lambda name, value, **kwargs: logged.update(
        {"name": name, "value": value, **kwargs}
    )
    return bare, logged


def test_log_uses_logger_true_when_trainer_logger_attached():
    bare, logged = _bare_with_log()
    bare.trainer = SimpleNamespace(logger=object())
    DFMMimirModule._log(bare, "train_loss", 1.23, on_step=True)
    assert logged == {"name": "train_loss", "value": 1.23, "on_step": True, "logger": True}


def test_log_uses_logger_false_when_trainer_logger_is_none():
    bare, logged = _bare_with_log()
    bare.trainer = SimpleNamespace(logger=None)
    DFMMimirModule._log(bare, "eval_loss", 4.56)
    assert logged["logger"] is False


# ---------------------------------------------------------------------------
# HRM-cycles config patch (real __init__, fake HF backends)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.eos_token_id = 1


class _FakeConfig:
    def __init__(self, H_cycles=2, L_cycles=3, L_bp_cycles=None) -> None:
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.L_bp_cycles = L_bp_cycles
        self.use_cache = True


class _FakeCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeConfig()
        self.linear = nn.Linear(2, 2)


@pytest.fixture
def fake_hf_backend(monkeypatch):
    monkeypatch.setattr(
        dfm_mimir_mod.AutoTokenizer,
        "from_pretrained",
        lambda model_id, trust_remote_code=True: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        dfm_mimir_mod.AutoModelForCausalLM,
        "from_pretrained",
        lambda model_id, trust_remote_code=True, dtype=None: _FakeCausalLM(),
    )


def test_hrm_cycles_patches_model_config(fake_hf_backend):
    module = DFMMimirModule(
        model_id="fake-model", hrm_cycles={"H_cycles": 1, "L_cycles": 1}
    )
    assert module.model.config.H_cycles == 1
    assert module.model.config.L_cycles == 1
    assert module.model.L_bp_cycles_padded == [1]


def test_hrm_cycles_none_leaves_model_config_untouched(fake_hf_backend):
    module = DFMMimirModule(model_id="fake-model", hrm_cycles=None)
    assert module.model.config.H_cycles == 2  # default, untouched
    assert module.model.config.L_cycles == 3
    assert not hasattr(module.model, "L_bp_cycles_padded")


# ---------------------------------------------------------------------------
# forward — non-finite loss guard
#
# A batch whose labels are entirely -100 (e.g. the prompt alone already used
# up max_length, leaving no output tokens to predict) makes HF's
# CrossEntropyLoss(reduction="mean") divide by zero valid tokens -> nan,
# without raising. forward() must not let that nan through: it would
# poison the whole epoch's averaged train/eval loss (mean of anything
# containing nan is nan).
# ---------------------------------------------------------------------------


def _bare_forward_module(loss_value):
    bare = _bare()
    fake_loss = torch.tensor(loss_value, requires_grad=True)
    fake_output = SimpleNamespace(loss=fake_loss, logits=torch.zeros(1, 1, 4))
    bare.model = lambda **kwargs: fake_output
    return bare


def test_forward_replaces_nan_loss_with_zero():
    bare = _bare_forward_module(float("nan"))
    loss, _ = DFMMimirModule.forward(
        bare, input_ids=torch.zeros(1, 1, dtype=torch.long), labels=torch.full((1, 1), -100)
    )
    assert loss.item() == 0.0
    assert torch.isfinite(loss)


def test_forward_replaces_inf_loss_with_zero():
    bare = _bare_forward_module(float("inf"))
    loss, _ = DFMMimirModule.forward(
        bare, input_ids=torch.zeros(1, 1, dtype=torch.long), labels=torch.full((1, 1), -100)
    )
    assert loss.item() == 0.0


def test_forward_leaves_finite_loss_untouched():
    bare = _bare_forward_module(2.5)
    loss, _ = DFMMimirModule.forward(
        bare, input_ids=torch.zeros(1, 1, dtype=torch.long), labels=torch.zeros(1, 1, dtype=torch.long)
    )
    assert loss.item() == 2.5
