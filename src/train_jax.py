# -*- coding: utf-8 -*-
"""Training core for the JAX/Flax path — the counterpart of ``src/train.py``.

Exposes one reusable entry point, :func:`run_fold`, with the same signature and
contract as the PyTorch one: train a single K-fold CV fold, optionally under an
Optuna trial, and return the best validation loss. ``cv_jax.py`` and
``optuna_search_jax.py`` import it, exactly as their PyTorch siblings import
``train.run_fold``.

    python src/cv_jax.py model=mamba2_jax data=rnn
    python src/optuna_search_jax.py model=mamba2_jax data=rnn optuna=mamba2_jax

What replaces Lightning, and where the equivalences are load-bearing:

* ``trainer.fit`` -> the epoch/batch loop at the bottom of this file.
* ``configure_optimizers``' four parameter groups -> ``optax.multi_transform``
  keyed by parameter path (see ``model_jax.mimir_mamba2.param_group``).
* ``trainer.gradient_clip_val`` -> ``optax.clip_by_global_norm``. Easy to lose
  in a hand-rolled loop, and losing it silently changes training: Lightning
  clips to ``training.max_grad_norm`` (1.0) today, *after* accumulation, so the
  chain here is ``MultiSteps(chain(clip, groups))`` and not the other way round.
* ``trainer.accumulate_grad_batches`` -> ``optax.MultiSteps``.
* ``trainer.estimated_stepping_batches`` -> computed explicitly, because there
  is nothing to ask.
* ``ModelCheckpoint(save_top_k=1, monitor="eval_loss")`` -> orbax's
  ``CheckpointManager`` with ``best_fn``/``max_to_keep=1``.
* ``OptunaPruningCallback`` -> a direct ``trial.report``/``should_prune`` call
  in the validation branch of the loop.

Every epoch/batch/accumulation knob is read from ``cfg.training``, never from a
trainer group. That is not cosmetic: ``_apply_trial_overrides`` writes Optuna's
suggestions into ``model_kwargs`` and ``cfg.training`` only, so a knob read from
anywhere else would silently ignore the search space (today the Lightning
config reaches them through ``${training.*}`` interpolation).
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import optuna
import orbax.checkpoint as ocp
from flax import nnx
from omegaconf import DictConfig, OmegaConf, open_dict

from lit_datamodule import DistillerDataModule
from model_jax import sharding as sharding_lib
from model_jax.batching import bucket_ladder, describe_ladder, to_jax_batch
from model_jax.donor_projection import load_donor_tables
from model_jax.factory import DEFAULT_KIND, build_model
from model_jax.mimir_mamba2 import param_group
from utils import dataloader_runtime, format_seconds

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trial hyperparameter merging
# ---------------------------------------------------------------------------


class _TokenizerHolder:
    """What ``DistillerDataModule.set_module`` actually needs.

    The datamodule takes a "module ref, for tokenizer access" and reads exactly
    two attributes off it (``.tokenizer``, ``.pad_token_id``) — see its own
    comment saying so. The JAX model has neither and should not: tokenization
    is a host-side concern that has nothing to do with the computation graph.
    This adapter supplies them from the converted donor directory, so the
    datamodule is reused as-is with no changes and no ``trust_remote_code``.
    """

    def __init__(self, donor_dir: str) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(donor_dir))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id


def apply_trial_overrides(
    model_kwargs: Dict[str, Any],
    cfg: DictConfig,
    trial_params: Dict[str, Any],
) -> None:
    """Merge Optuna's suggested hyperparameters into *model_kwargs* / *cfg.training*.

    A duplicate of ``train._apply_trial_overrides`` — the logic is
    framework-agnostic (plain dicts and DictConfig), but importing it from
    ``train.py`` would drag that module's import-time torch/Lightning setup
    (multiprocessing strategy, TF32 flags, warning filters) onto the JAX path.
    Fifteen lines is cheaper than that coupling.
    """
    for key, value in trial_params.items():
        if key in model_kwargs:
            model_kwargs[key] = value
        if key in cfg.training:
            with open_dict(cfg.training):
                cfg.training[key] = value


# ---------------------------------------------------------------------------
# Optimiser
# ---------------------------------------------------------------------------


def build_schedule(
    name: str, peak_lr: float, total_steps: int, warmup_ratio: float
) -> optax.Schedule:
    """Warmup + cosine/linear decay, matching the PyTorch path's HF schedules.

    ``get_cosine_schedule_with_warmup`` ramps 0 -> peak over ``warmup_steps``
    then decays over the remaining ``total_steps - warmup_steps``; optax's
    ``warmup_cosine_decay_schedule`` splits its ``decay_steps`` the same way, so
    passing the same two numbers gives the same curve.
    """
    warmup_steps = int(total_steps * warmup_ratio)
    if name == "cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=peak_lr,
            warmup_steps=warmup_steps,
            decay_steps=max(total_steps, warmup_steps + 1),
            end_value=0.0,
        )
    if name == "linear":
        return optax.join_schedules(
            [
                optax.linear_schedule(0.0, peak_lr, max(warmup_steps, 1)),
                optax.linear_schedule(
                    peak_lr, 0.0, max(total_steps - warmup_steps, 1)
                ),
            ],
            boundaries=[max(warmup_steps, 1)],
        )
    return optax.constant_schedule(peak_lr)


def build_optimizer(
    params,
    *,
    learning_rate: float,
    weight_decay: float,
    projection_lr_mult: float,
    warmup_ratio: float,
    lr_scheduler: str,
    max_grad_norm: float,
    grad_accum_steps: int,
    total_steps: int,
    freeze_backbone: bool,
) -> Tuple[optax.GradientTransformation, Any]:
    """Four parameter groups: ``{projection, body} x {decay, no-decay}``.

    The frozen donor tables cannot reach the optimiser by construction — they
    are ``Donor`` variables, not ``nnx.Param``, so they are not in *params* at
    all. Same guarantee as the PyTorch path's "buffers, not parameters", by a
    different mechanism.

    Composition order matters and mirrors Lightning's: accumulate first, then
    clip the accumulated gradient, then apply the per-group transform. Wrapping
    it the other way round would clip each micro-batch separately, which is a
    different (and weaker) constraint.
    """
    base_schedule = build_schedule(lr_scheduler, learning_rate, total_steps, warmup_ratio)
    proj_schedule = build_schedule(
        lr_scheduler, learning_rate * projection_lr_mult, total_steps, warmup_ratio
    )

    def adamw(schedule, decay: float) -> optax.GradientTransformation:
        return optax.adamw(learning_rate=schedule, weight_decay=decay)

    body = optax.set_to_zero() if freeze_backbone else None
    transforms = {
        "proj_decay": adamw(proj_schedule, weight_decay),
        "proj_no_decay": adamw(proj_schedule, 0.0),
        "body_decay": body or adamw(base_schedule, weight_decay),
        "body_no_decay": body or adamw(base_schedule, 0.0),
    }

    labels = _label_tree(params)
    grouped = optax.multi_transform(transforms, labels)
    chained = optax.chain(optax.clip_by_global_norm(max_grad_norm), grouped)
    tx = optax.MultiSteps(chained, every_k_schedule=max(grad_accum_steps, 1))
    return tx, labels


def _label_tree(params):
    """Map every parameter path to its optimiser-group name."""
    flat = nnx.to_flat_state(params)
    labelled = [(path, param_group(path)) for path, _ in flat]
    counts: Dict[str, int] = {}
    for _, group in labelled:
        counts[group] = counts.get(group, 0) + 1
    log.info("Optimiser groups: %s", counts)
    return nnx.from_flat_state(labelled)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def make_checkpoint_manager(directory: str, enabled: bool) -> Optional[Any]:
    """Best-on-eval-loss checkpointing, mirroring ``_best_checkpoint``.

    Saves ``nnx.state(model, nnx.Param)`` only — the donor tables are ``Donor``
    variables and so are excluded by construction, the same 805 MiB-per-table
    saving the PyTorch path gets from ``persistent=False``.

    NOTE: orbax checkpoints are **not** interchangeable with PyTorch
    ``state_dict``s. This is a hard format break; ``model_jax/export.py`` is the
    bridge for anything that has to cross it.
    """
    if not enabled:
        return None
    path = Path(directory).absolute()
    path.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=1,
        best_fn=lambda metrics: float(metrics["eval_loss"]),
        best_mode="min",
        create=True,
    )
    return ocp.CheckpointManager(path, options=options)


# ---------------------------------------------------------------------------
# Single CV fold
# ---------------------------------------------------------------------------


def run_fold(
    cfg: DictConfig,
    fold_idx: int,
    trial: Optional[optuna.Trial] = None,
) -> float:
    """Train one CV fold and return the best validation loss.

    Same contract as ``train.run_fold``: builds the model and a
    ``DistillerDataModule`` scoped to *fold_idx*, trains, and returns
    ``eval_loss``. Under a trial, reports to Optuna after every validation pass
    and honours pruning; otherwise writes this fold's single best checkpoint to
    ``{cfg.training.output_dir}/fold_{fold_idx}``.

    Seeding: the PyTorch path re-seeds global RNG per fold/trial. JAX has no
    global RNG, so the same offset feeds an explicit ``nnx.Rngs`` — reproducible
    in isolation, and not dependent on how much randomness earlier folds drew.
    """
    training = cfg.training
    offset = fold_idx + (trial.number * 1000 if trial is not None else 0)
    seed = int(training.seed) + offset
    rngs = nnx.Rngs(params=seed)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    kind = model_cfg.pop("kind", DEFAULT_KIND)
    donor_dir = model_cfg.pop("donor_dir", None) or model_cfg.get("donor_model_id")

    model_kwargs: Dict[str, Any] = dict(model_cfg)
    if trial is not None:
        apply_trial_overrides(model_kwargs, cfg, trial.params)

    tables = load_donor_tables(donor_dir, dtype=jnp.bfloat16)
    model = build_model(kind, model_kwargs, tables=tables, rngs=rngs)

    # ------------------------------------------------------------------
    # Data — the PyTorch DataLoaders, unchanged; only the boundary differs.
    # ------------------------------------------------------------------
    tokenizer_holder = _TokenizerHolder(donor_dir)
    datamodule = DistillerDataModule(
        cfg, fold=fold_idx, runtime=dataloader_runtime(cfg.runtime)
    )
    datamodule.set_module(tokenizer_holder)
    datamodule.setup()
    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()

    pad_token_id = int(tokenizer_holder.pad_token_id or 0)
    ladder = bucket_ladder(int(cfg.data.max_length))
    train_batch = int(training.batch_size)
    eval_batch = max(1, int(train_batch * int(training.eval_batch_multiplier)))
    log.info("Batching: %s", describe_ladder(ladder, train_batch))

    mesh = sharding_lib.make_mesh(cfg.get("mesh_shape"))

    # ------------------------------------------------------------------
    # Optimiser — every knob from cfg.training, so Optuna overrides land.
    # ------------------------------------------------------------------
    grad_accum = max(int(training.gradient_accumulation_steps), 1)
    num_epochs = int(training.num_train_epochs)
    steps_per_epoch = max(len(train_dl) // grad_accum, 1)
    total_steps = max(steps_per_epoch * num_epochs, 1)

    params = nnx.state(model, nnx.Param)
    tx, _labels = build_optimizer(
        params,
        learning_rate=float(training.learning_rate),
        weight_decay=float(training.weight_decay),
        projection_lr_mult=float(model_kwargs.get("projection_lr_mult", 1.0)),
        warmup_ratio=float(training.warmup_ratio),
        lr_scheduler=str(training.lr_scheduler),
        max_grad_norm=float(training.max_grad_norm),
        grad_accum_steps=grad_accum,
        total_steps=total_steps,
        freeze_backbone=bool(model_kwargs.get("freeze_backbone", False)),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    log.info(
        "Fold %d: %d epochs x %d optimiser steps (grad_accum=%d) = %d total",
        fold_idx,
        num_epochs,
        steps_per_epoch,
        grad_accum,
        total_steps,
    )

    # ------------------------------------------------------------------
    # Jitted steps
    # ------------------------------------------------------------------
    @nnx.jit
    def train_step(model, optimizer, batch):
        def loss_fn(m):
            return m.loss(batch["input_ids"], batch["labels"], batch["attention_mask"])

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    @nnx.jit
    def eval_step(model, batch):
        return model.loss(batch["input_ids"], batch["labels"], batch["attention_mask"])

    def prepare(batch, size):
        return sharding_lib.shard_batch(
            mesh, to_jax_batch(batch, ladder, size, pad_token_id)
        )

    def evaluate() -> float:
        total, count = 0.0, 0
        for batch in val_dl:
            loss = float(eval_step(model, prepare(batch, eval_batch)))
            if math.isfinite(loss):
                total += loss
                count += 1
        return total / count if count else float("inf")

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    manager = make_checkpoint_manager(
        str(Path(training.output_dir) / f"fold_{fold_idx}"), enabled=trial is None
    )
    best = float("inf")
    start = time.time()
    global_step = 0

    try:
        for epoch in range(num_epochs):
            running, seen = 0.0, 0
            for batch in train_dl:
                loss = float(train_step(model, optimizer, prepare(batch, train_batch)))
                running += loss
                seen += 1
                global_step += 1

            eval_loss = evaluate()
            train_loss = running / seen if seen else float("nan")
            log.info(
                "epoch %d/%d — train_loss=%.4f eval_loss=%.4f",
                epoch + 1,
                num_epochs,
                train_loss,
                eval_loss,
            )

            if eval_loss < best:
                best = eval_loss
                if manager is not None:
                    manager.save(
                        global_step,
                        args=ocp.args.StandardSave(
                            jax.tree.map(np.asarray, nnx.state(model, nnx.Param))
                        ),
                        metrics={"eval_loss": eval_loss},
                    )

            if trial is not None:
                trial.report(eval_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
    finally:
        if manager is not None:
            manager.wait_until_finished()
            manager.close()

    elapsed = format_seconds(time.time() - start)
    log.info("  fold done in %s — val_loss=%.4f", elapsed, best)
    return best


def restore_params(directory: str, model) -> None:
    """Load a saved best checkpoint's params back into *model*, in place."""
    manager = ocp.CheckpointManager(Path(directory).absolute())
    step = manager.best_step() or manager.latest_step()
    if step is None:
        raise FileNotFoundError(f"no checkpoint in {directory}")
    target = jax.tree.map(np.asarray, nnx.state(model, nnx.Param))
    restored = manager.restore(step, args=ocp.args.StandardRestore(target))
    nnx.update(model, restored)
    manager.close()


# ---------------------------------------------------------------------------
# Hydra entrypoint — a single fold, for smoke-testing the loop
# ---------------------------------------------------------------------------


@hydra.main(
    config_path=str(Path(__file__).parent / "config"),
    config_name="config_jax",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    loss = run_fold(cfg, fold_idx=0)
    log.info("fold 0 best eval_loss = %.6f", loss)


if __name__ == "__main__":
    main()
