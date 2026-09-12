# -*- coding: utf-8 -*-
"""End-to-end checks for the JAX path: configs, export, and a real training run.

Three things that only fail when the pieces are put together:

1. The Hydra config tree actually composes, and the keys ``train_jax`` reads
   exist where it reads them (in particular: epochs/batch/accumulation under
   ``training``, not under a trainer group — which is what keeps Optuna's
   overrides effective).
2. The exported PyTorch model produces the same logits as the JAX model it came
   from — the algebraic-identity check that makes the checkpoint-format break
   survivable.
3. A short training run on synthetic data drives the loss down, and does so
   without recompiling on every batch.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from hydra import compose, initialize_config_dir

from model_jax.batching import bucket_ladder, to_jax_batch
from model_jax.donor_projection import DonorTables
from model_jax.export import export_to_torch
from model_jax.mimir_mamba2 import MimirMamba2Model
from train_jax import build_optimizer

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "src" / "config")
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


def test_jax_config_tree_composes():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_jax")

    assert cfg.model.kind == "mamba2_jax"
    assert cfg.model.donor_dir
    # No Lightning Trainer block on this path — that is the whole reason for a
    # second root config.
    assert "trainer" not in cfg


def test_loop_knobs_live_under_training_so_optuna_overrides_reach_them():
    """If any of these moved to the root config, the search space would go dead.

    ``_apply_trial_overrides`` only writes into ``model_kwargs`` and
    ``cfg.training``; a knob read from anywhere else is invisible to Optuna.
    """
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_jax")

    for key in (
        "num_train_epochs", "batch_size", "gradient_accumulation_steps",
        "max_grad_norm", "learning_rate", "weight_decay", "warmup_ratio",
        "lr_scheduler", "eval_batch_multiplier", "seed", "output_dir",
    ):
        assert key in cfg.training, key


def test_every_searched_parameter_reaches_the_model_or_the_training_config():
    """A search-space name matching neither target would be silently ignored."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_jax", overrides=["optuna=mamba2_jax"])

    # apply_trial_overrides writes a suggestion into model_kwargs (built from
    # everything under cfg.model) and/or cfg.training. A name in neither is
    # suggested, recorded in the trial, and then silently discarded.
    for name in cfg.optuna.search_space:
        assert name in cfg.model or name in cfg.training, (
            f"searched parameter {name!r} reaches neither cfg.model nor "
            f"cfg.training — it would be suggested and then discarded"
        )

    # Of the ones that land in cfg.model, each is either a model-constructor
    # argument or a knob train_jax reads off model_kwargs itself.
    import inspect

    accepted = set(inspect.signature(MimirMamba2Model.__init__).parameters)
    loop_read = {"projection_lr_mult", "freeze_backbone"}
    for name in cfg.optuna.search_space:
        if name in cfg.model:
            assert name in accepted or name in loop_read, name


def test_jax_optuna_study_is_separate_from_the_pytorch_one():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        jax_cfg = compose(config_name="config_jax", overrides=["optuna=mamba2_jax"])
        torch_cfg = compose(config_name="config", overrides=["optuna=mamba2"])

    assert jax_cfg.optuna.study_name != torch_cfg.optuna.study_name
    assert jax_cfg.optuna.storage != torch_cfg.optuna.storage


def test_pytorch_config_tree_still_composes_unchanged():
    """The port must not have destabilised the working path."""
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=["model=mamba2"])
    assert cfg.model.kind == "mamba2"
    assert cfg.trainer._target_ == "lightning.pytorch.Trainer"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_exported_torch_model_matches_the_jax_model_logits():
    """The algebraic identity the whole export rests on."""
    model = _model()
    exported = export_to_torch(model)

    ids_np = np.random.default_rng(0).integers(0, VOCAB, (2, 11))
    jax_logits = np.asarray(model(jnp.asarray(ids_np)))
    with torch.no_grad():
        torch_logits = exported(
            input_ids=torch.as_tensor(ids_np), use_cache=False
        ).logits.numpy()

    assert jax_logits.shape == torch_logits.shape == (2, 11, VOCAB)
    rel = np.abs(jax_logits - torch_logits).max() / max(np.abs(torch_logits).max(), 1e-6)
    print(f"\nexport parity: max abs {np.abs(jax_logits - torch_logits).max():.3e}, "
          f"max rel {rel:.3e}")
    assert rel < 1e-4


def test_exported_model_carries_no_donor_and_is_smaller_than_the_donor():
    model = _model()
    exported = export_to_torch(model)

    names = dict(exported.named_parameters())
    assert not any("donor" in n for n in names)
    assert names["backbone.embeddings.weight"].shape == (VOCAB, 32)
    assert names["lm_head.weight"].shape == (VOCAB, 32)
    # Exported tables are d_model-wide, not d_donor-wide — that is the point.
    assert 32 < D_DONOR


# ---------------------------------------------------------------------------
# A real (short) training run
# ---------------------------------------------------------------------------


def test_short_training_run_drives_the_loss_down_without_recompiling():
    """Phase D's acceptance check, at toy scale and on synthetic data.

    Also pins the reason ``batching.py`` exists: with the bucket ladder the
    jitted step compiles once per (batch, rung) shape, not once per batch.
    """
    model = _model(gradient_checkpointing=True)
    tx, _ = build_optimizer(
        nnx.state(model, nnx.Param),
        learning_rate=3e-3, weight_decay=0.01, projection_lr_mult=1.0,
        warmup_ratio=0.1, lr_scheduler="cosine", max_grad_norm=1.0,
        grad_accum_steps=1, total_steps=30, freeze_backbone=False,
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    # Python bodies run only when a function is traced, so a plain counter is
    # an exact trace count — nnx.jit does not expose jax.jit's _cache_size.
    traces = []

    @nnx.jit
    def train_step(model, optimizer, batch):
        traces.append(batch["input_ids"].shape)

        def loss_fn(m):
            return m.loss(batch["input_ids"], batch["labels"], batch["attention_mask"])

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    # A genuinely learnable signal, which takes some care to construct: the
    # loss shifts by one (hidden[t] predicts labels[t+1]), so labels must be a
    # function of ids up to and including t. `labels[t+1] = ids[t] + 1` is —
    # `labels[t] = ids[t] + 1` would not be, since ids[t+1] is random noise the
    # model cannot see, and the loss would sit flat at chance no matter how
    # well the loop works.
    rng = np.random.default_rng(0)
    ladder = bucket_ladder(32, min_rung=16)

    def make_batch(length: int):
        ids = rng.integers(0, VOCAB // 2, (4, length))
        labels = np.empty_like(ids)
        labels[:, 1:] = (ids[:, :-1] + 1) % VOCAB
        labels[:, 0] = -100  # nothing precedes the first position
        return to_jax_batch(
            {"input_ids": ids, "labels": labels,
             "attention_mask": np.ones_like(ids)},
            ladder, batch_size=4, pad_token_id=0,
        )

    # Vary the raw length on every step, exactly like the real dataloader does.
    lengths = [12, 17, 9, 30, 21, 14]
    losses = []
    for step in range(30):
        batch = make_batch(lengths[step % len(lengths)])
        losses.append(float(train_step(model, optimizer, batch)))

    assert all(np.isfinite(losses)), losses
    first, last = np.mean(losses[:5]), np.mean(losses[-5:])
    print(f"\ntraining run: loss {first:.4f} -> {last:.4f} over 30 steps")
    assert last < first * 0.9, f"loss did not fall: {first:.4f} -> {last:.4f}"

    # Six distinct raw lengths over 30 steps, but only the ladder's rungs as
    # distinct shapes — so the step traced a handful of times, not thirty.
    shapes = set(traces)
    print(f"traces: {len(traces)} for shapes {sorted(shapes)} over 30 steps")
    assert len(traces) <= len(ladder), (
        f"{len(traces)} traces for {len(ladder)} rungs — the bucket ladder is "
        f"not containing recompilation"
    )
    assert {s[1] for s in shapes} <= set(ladder)
