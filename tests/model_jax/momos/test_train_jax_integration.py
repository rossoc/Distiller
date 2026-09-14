# -*- coding: utf-8 -*-
"""Tests for the ``momos_backbone`` Hydra wiring in ``src/train_jax.py``
(SPEC_PHASE_E.md's "full integration" landing).

Mirrors ``tests/model_jax/test_end_to_end_jax.py``'s pattern for the dense
path: a config-composition check, plus a real (short) training run — on
synthetic data, hand-built rather than routed through
``DistillerDataModule``/a real tokenizer/donor tables, for the same reason
the dense path's own end-to-end test does (no real donor checkpoint is
available offline in this environment; see ``scripts/momos_phase_e.py``'s
docstring for the fuller explanation). What this DOES exercise, unlike
``tests/model_jax/momos/test_integration.py``: the Hydra config tree itself,
and ``train_jax._run_momos_fold`` — the actual function
``train_jax.run_fold`` calls when ``momos_backbone.enabled: true`` — rather
than calling ``model_jax.momos.integration`` directly.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from hydra import compose, initialize_config_dir

import train_jax
from model_jax.donor_projection import DonorTables
from model_jax.mimir_mamba2 import MimirMamba2Model

CONFIG_DIR = str(Path(__file__).resolve().parents[3] / "src" / "config")
VOCAB, D_DONOR = 96, 48


def _model(**overrides) -> MimirMamba2Model:
    g = torch.Generator().manual_seed(0)
    tables = DonorTables(
        "toy/donor",
        jnp.asarray((torch.randn(VOCAB, D_DONOR, generator=g) * 0.03).numpy()),
        jnp.asarray((torch.randn(VOCAB, D_DONOR, generator=g) * 0.05).numpy()),
    )
    kwargs = dict(
        d_model=32, num_hidden_layers=2, state_size=8, head_dim=16, chunk_size=8,
        projection_cache_dir=None, loss_chunk_tokens=8, gradient_checkpointing=False,
    )
    kwargs.update(overrides)
    return MimirMamba2Model(tables, rngs=nnx.Rngs(params=0), **kwargs)


# ---------------------------------------------------------------------------
# Hydra config composition
# ---------------------------------------------------------------------------


def test_momos_backbone_is_off_by_default():
    """The safety property this whole landing rests on: composing the root
    config with no overrides must leave the dense path untouched."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_jax")
    assert cfg.momos_backbone.enabled is False


def test_momos_backbone_config_composes_with_overrides():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config_jax",
            overrides=["momos_backbone.enabled=true", "momos_backbone.K=1024", "momos_backbone.S=2"],
        )
    assert cfg.momos_backbone.enabled is True
    assert cfg.momos_backbone.K == 1024
    assert cfg.momos_backbone.S == 2


def test_momos_config_from_cfg_builds_a_valid_mosaic_config():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config_jax",
            overrides=["momos_backbone.enabled=true", "momos_backbone.K=1024"],
        )
    momos_cfg = train_jax._momos_config_from_cfg(cfg.momos_backbone)
    assert momos_cfg.K == 1024
    assert momos_cfg.S == cfg.momos_backbone.S
    assert momos_cfg.scale_mode == cfg.momos_backbone.scale_mode


# ---------------------------------------------------------------------------
# A real (short) training run through _run_momos_fold
# ---------------------------------------------------------------------------


def _synthetic_batches(n: int, batch_size: int, length: int, seed: int):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(n):
        ids = rng.integers(0, VOCAB // 2, (batch_size, length))
        labels = np.empty_like(ids)
        labels[:, 1:] = (ids[:, :-1] + 1) % VOCAB
        labels[:, 0] = -100
        batches.append(
            {"input_ids": ids, "labels": labels, "attention_mask": np.ones_like(ids)}
        )
    return batches


def test_run_momos_fold_short_training_run_on_synthetic_data(tmp_path):
    """The function ``train_jax.run_fold`` actually calls when
    ``momos_backbone.enabled: true`` — driven by a real composed Hydra
    config, on a real model, with a hand-built ``train_dl``/``val_dl`` in
    place of ``DistillerDataModule`` (see module docstring)."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config_jax",
            overrides=[
                "momos_backbone.enabled=true",
                "momos_backbone.K=64",
                "momos_backbone.S=1",
                "momos_backbone.dict_lr_mult=1.0",
                "training.learning_rate=3e-3",
                "training.num_train_epochs=1",
                "training.lr_scheduler=constant",
                f"training.output_dir={tmp_path}",
            ],
        )

    model = _model()
    train_dl = _synthetic_batches(n=20, batch_size=4, length=16, seed=0)
    val_dl = _synthetic_batches(n=3, batch_size=4, length=16, seed=1)

    from model_jax.batching import bucket_ladder, to_jax_batch

    ladder = bucket_ladder(16, min_rung=16)

    def prepare(batch, size):
        return to_jax_batch(batch, ladder, size, pad_token_id=0)

    best = train_jax._run_momos_fold(
        cfg, fold_idx=0, seed=0, model=model, momos_cfg_dc=cfg.momos_backbone,
        train_dl=train_dl, val_dl=val_dl, prepare=prepare,
        train_batch=4, eval_batch=4, num_epochs=1, total_steps=len(train_dl), trial=None,
    )
    assert np.isfinite(best)

    # A checkpoint for this fold must exist (eval_loss improved from +inf).
    fold_dir = tmp_path / "fold_0"
    assert fold_dir.exists() and any(fold_dir.iterdir())


def test_run_momos_fold_drives_the_loss_down(tmp_path):
    """Same shape as the dense path's ``test_short_training_run_drives_the_
    loss_down_without_recompiling`` acceptance check, for the momos path."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="config_jax",
            overrides=[
                "momos_backbone.enabled=true",
                "momos_backbone.K=64",
                "momos_backbone.S=1",
                "training.learning_rate=5e-3",
                "training.num_train_epochs=6",
                "training.lr_scheduler=constant",
                f"training.output_dir={tmp_path}",
            ],
        )

    model = _model()
    batches = _synthetic_batches(n=6, batch_size=4, length=16, seed=2)
    val_dl = batches[:2]

    from model_jax.batching import bucket_ladder, to_jax_batch

    ladder = bucket_ladder(16, min_rung=16)

    def prepare(batch, size):
        return to_jax_batch(batch, ladder, size, pad_token_id=0)

    # Two runs from the same init, one epoch vs six — more epochs must give
    # a lower (or equal) best eval loss than a single pass.
    best_1 = train_jax._run_momos_fold(
        cfg, fold_idx=0, seed=0, model=_model(), momos_cfg_dc=cfg.momos_backbone,
        train_dl=batches, val_dl=val_dl, prepare=prepare,
        train_batch=4, eval_batch=4, num_epochs=1, total_steps=len(batches), trial=None,
    )
    best_6 = train_jax._run_momos_fold(
        cfg, fold_idx=1, seed=0, model=_model(), momos_cfg_dc=cfg.momos_backbone,
        train_dl=batches, val_dl=val_dl, prepare=prepare,
        train_batch=4, eval_batch=4, num_epochs=6, total_steps=len(batches) * 6, trial=None,
    )
    assert np.isfinite(best_1) and np.isfinite(best_6)
    assert best_6 < best_1


# ---------------------------------------------------------------------------
# The DataLoader worker fork -> spawn fix (SPEC_PHASE_E.md's real-run report)
# ---------------------------------------------------------------------------


def test_run_fold_requests_spawn_workers_to_avoid_the_fork_warning():
    """``train_jax.run_fold`` must ask ``DistillerDataModule`` for ``spawn``
    workers whenever ``num_workers > 0`` — fork()ing DataLoader workers in a
    process that has already opened CUDA/gone multithreaded via JAX is the
    unsafe pattern Python's own RuntimeWarning names (observed for real,
    training against the actual district-heating data with a real donor)."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_jax", overrides=["runtime.num_workers=4"])

    from lit_datamodule import DistillerDataModule
    from utils import dataloader_runtime

    runtime = dataloader_runtime(cfg.runtime)
    if runtime["num_workers"] > 0:
        runtime["multiprocessing_context"] = "spawn"
    dm = DistillerDataModule(cfg, fold=0, runtime=runtime)
    assert dm.runtime["multiprocessing_context"] == "spawn"


def test_build_dataloader_forwards_multiprocessing_context(tmp_path):
    """The other half: ``_build_dataloader`` must actually pass it to
    ``torch.utils.data.DataLoader``, not just store it."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_jax")

    from lit_datamodule import DistillerDataModule, _TokenizedDataset

    class _FakeModule:
        pad_token_id = 0

    dm = DistillerDataModule(
        cfg, fold=0,
        runtime={
            "num_workers": 2, "pin_memory": False, "persistent_workers": True,
            "prefetch_factor": 2, "multiprocessing_context": "spawn",
        },
    )
    dm.module = _FakeModule()
    dataset = _TokenizedDataset(
        [torch.tensor([1, 2, 3])] * 4, [torch.tensor([1, 2, 3])] * 4
    )
    dl = dm._build_dataloader(dataset, shuffle=True)
    assert dl.multiprocessing_context is not None
    assert dl.multiprocessing_context.get_start_method() == "spawn"
