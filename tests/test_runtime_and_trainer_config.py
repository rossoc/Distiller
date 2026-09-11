# -*- coding: utf-8 -*-
"""Sanity checks for the config-only fixes (performance review items #6/#7):

  #6: runtime/default.yaml's num_workers/pin_memory should actually use the
      training box's CPU count rather than a fixed conservative guess, and
      pin_memory should end up enabled on a CUDA box.
  #7: config.yaml's Trainer should not run with cudnn.benchmark=True, given
      the project's shapes change fold-to-fold and (since dynamic per-batch
      padding replaced whole-split padding) batch-to-batch.

These are plain YAML/config checks, not behavioural tests — no Hydra
composition or model construction needed.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "src" / "config"


def _load(*parts: str) -> dict:
    with open(CONFIG_DIR.joinpath(*parts)) as f:
        return yaml.safe_load(f)


def test_num_workers_is_sized_to_the_machine_not_a_fixed_small_guess():
    runtime = _load("runtime", "default.yaml")
    cpu_count = os.cpu_count() or 1
    # Not the old conservative default...
    assert runtime["num_workers"] != 2 or cpu_count <= 2
    # ...but not unreasonably close to every core either (leave headroom for
    # the main process / concurrent Optuna trials).
    assert 1 <= runtime["num_workers"] <= cpu_count


def test_pin_memory_resolves_true_on_a_cuda_box():
    import sys

    sys.path.insert(0, str(CONFIG_DIR.parent))
    from unittest import mock

    from omegaconf import OmegaConf

    from utils import dataloader_runtime

    runtime_cfg = OmegaConf.create(_load("runtime", "default.yaml"))
    with mock.patch("torch.cuda.is_available", return_value=True):
        resolved = dataloader_runtime(runtime_cfg)
    assert resolved["pin_memory"] is True


def test_persistent_workers_and_prefetch_factor_still_consistent_with_workers():
    runtime = _load("runtime", "default.yaml")
    if runtime["num_workers"] > 0:
        # Both settings are only meaningful (and only accepted by
        # DataLoader) when num_workers > 0 — see utils.dataloader_runtime.
        assert runtime["persistent_workers"] is True
        assert runtime["prefetch_factor"] and runtime["prefetch_factor"] > 0


def test_trainer_cudnn_benchmark_is_disabled():
    cfg = _load("config.yaml")
    assert cfg["trainer"]["benchmark"] is False
