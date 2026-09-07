# -*- coding: utf-8 -*-
"""Tests for utils.py — reproducibility, device resolution, formatting."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from utils import format_seconds, resolve_device, seed_all


# ---------------------------------------------------------------------------
# seed_all
# ---------------------------------------------------------------------------


def test_seed_all_reproducible_across_calls():
    seed_all(123)
    a_random = random.random()
    a_np = np.random.rand()
    a_torch = torch.rand(3)

    seed_all(123)
    b_random = random.random()
    b_np = np.random.rand()
    b_torch = torch.rand(3)

    assert a_random == b_random
    assert a_np == b_np
    assert torch.equal(a_torch, b_torch)


def test_seed_all_different_seeds_diverge():
    seed_all(1)
    a = torch.rand(3)
    seed_all(2)
    b = torch.rand(3)
    assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


def test_resolve_device_cpu_always_available():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", None, raising=False)
    assert resolve_device("auto") == "cpu"


def test_resolve_device_auto_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_resolve_device_cuda_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError):
        resolve_device("cuda")


def test_resolve_device_mps_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends, "mps", None, raising=False)
    with pytest.raises(RuntimeError):
        resolve_device("mps")


def test_resolve_device_unknown_mode_raises():
    with pytest.raises(ValueError):
        resolve_device("tpu")


# ---------------------------------------------------------------------------
# format_seconds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "00:00"),
        (59, "00:59"),
        (60, "01:00"),
        (3600, "01:00:00"),
        (3661, "01:01:01"),
        (-5, "00:00"),
    ],
)
def test_format_seconds(seconds, expected):
    assert format_seconds(seconds) == expected
