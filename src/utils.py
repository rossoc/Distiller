# -*- coding: utf-8 -*-
"""Small standalone helpers for reproducibility, device selection, and CUDA tuning."""

from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(mode: str) -> str:
    """Resolve a device mode string to a concrete runtime device.

    Args:
        mode: One of ``auto``, ``cuda``, ``mps``, or ``cpu``.

    Returns:
        Resolved device string.
    """
    mode = str(mode).lower()
    has_mps = (
        bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    )

    if mode == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if has_mps else "cpu"
    elif mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but it is not available.")
        return "cuda"
    elif mode == "mps":
        if not has_mps:
            raise RuntimeError("Requested MPS but it is not available.")
        return "mps"
    elif mode == "cpu":
        return "cpu"
    raise ValueError(f"Unknown device mode: {mode}")


def configure_cuda_fast_path(enable_benchmark: bool = False) -> None:
    """Enable common CUDA fast-path settings when CUDA is available."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if enable_benchmark:
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def dataloader_runtime(runtime_cfg: DictConfig) -> Dict[str, Any]:
    """Resolve ``cfg.runtime`` into concrete DataLoader kwargs.

    Single source of truth for worker count / pinning / persistence /
    prefetching, shared by ``train.py``, ``cv.py``, and ``predict.py`` so
    they don't each hand-roll (and drift on) their own DataLoader setup.
    ``persistent_workers``/``prefetch_factor`` are only meaningful (and only
    accepted by ``torch.utils.data.DataLoader``) when ``num_workers > 0``.
    """
    num_workers = int(runtime_cfg.num_workers)
    pin_memory = runtime_cfg.pin_memory
    pin_memory = torch.cuda.is_available() if pin_memory == "auto" else bool(pin_memory)

    if num_workers > 0:
        prefetch_factor = runtime_cfg.get("prefetch_factor", None)
        return {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": bool(runtime_cfg.persistent_workers),
            "prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else None,
        }
    return {
        "num_workers": 0,
        "pin_memory": pin_memory,
        "persistent_workers": False,
        "prefetch_factor": None,
    }


def format_seconds(seconds: float) -> str:
    """Format seconds as ``MM:SS`` or ``HH:MM:SS``."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
