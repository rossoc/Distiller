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

import os

# Must be set before `import jax` — JAX reads these at backend init, not at
# call time. Default (unset) is `XLA_PYTHON_CLIENT_PREALLOCATE=true` at
# `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`: JAX grabs 75% of the GPU's *free*
# memory as one arena on the very first op, regardless of what the model
# actually needs — confirmed directly: a script doing nothing but
# `jnp.zeros((100,))` on this project's hardware still requested 11.6 GiB.
# That is why `nvidia-smi` showed ~12-13GB "used" for a 21M-parameter model
# whose real measured peak (`jax.devices()[0].memory_stats()
# ['peak_bytes_in_use']`, with preallocation off) is ~4.5-4.9GB — see
# SPEC_PHASE_PERF.md for the fuller writeup. Capping the fraction, rather
# than disabling preallocation outright, keeps its performance benefit
# (fewer allocator round-trips) while leaving headroom for XLA's one-time
# autotuning compile spike — the actual cause of the OOM this was chasing,
# not steady-state peak, which barely changes with `loss_chunk_tokens`.
# `setdefault` means an explicit env var (or a smaller card) still wins.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

# jaxlib's native runtime (PJRT/TSL) logs things like the `cuda_timer.cc
# "Delay kernel timed out"` autotuning-timer notice straight to stderr via
# C++ absl logging — it never goes through Python's `logging` module, so no
# logger-level change above can touch it. TSL inherits TensorFlow's
# verbosity env var; 2 = drop INFO and WARNING, keep ERROR/FATAL, which is
# exactly this notice's level. Harmless to suppress: it's about autotuning
# *measurement* accuracy, not training correctness.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import dataclasses
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import optuna
import orbax.checkpoint as ocp
import torch
from flax import nnx
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from lit_datamodule import DistillerDataModule
from model_jax import sharding as sharding_lib
from model_jax.batching import bucket_ladder, describe_ladder, to_jax_batch
from model_jax.donor_projection import load_donor_tables
from model_jax.factory import DEFAULT_KIND, build_model
from model_jax.mimir_mamba2 import IGNORE_INDEX, param_group
from model_jax.momos import integration as momos_integration
from model_jax.momos.state import MosaicConfig
from utils import dataloader_runtime, format_seconds

# Persist XLA's compiled executables — and, per `jax_persistent_cache_enable_xla_caches`
# (default: `xla_gpu_per_fusion_autotune_cache_dir`), its autotuning results — to disk
# *across process runs*, not just within one. Without this, every
# `python src/train_jax.py` invocation recompiles and re-autotunes all 5+5 bucket
# shapes from scratch — the ~2.5-3min AOT warmup in `run_fold`, plus the
# `cuda_timer.cc "Delay kernel timed out"` warnings XLA's autotuner logs while
# benchmarking candidate kernels — even when nothing about the model or shapes
# changed since the last run. On a cache hit, `train_step.lower(...).compile()`
# loads the executable straight from disk instead of re-tracing/re-autotuning, so
# the AOT warmup collapses to whatever it costs to read cache files. First run
# after a code/shape/model-arch change still pays the full cost and repopulates
# the cache. `jax_persistent_cache_min_compile_time_secs` defaults to 1s, well
# under our ~30s/shape compiles, so nothing else needs configuring.
_JAX_CACHE_DIR = Path(__file__).resolve().parent.parent / ".jax_cache"
_JAX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_JAX_CACHE_DIR))

# A level strictly between INFO (20) and WARNING (30), for the handful of
# lines an interactive run actually wants to see live: per-epoch train/eval
# loss and the final per-fold summary. Everything else in this file (and in
# every other module's logger, and third-party ones — absl, jax, orbax,
# hydra — none of which set an explicit level of their own, so they inherit
# whatever the root logger's level is) stays at plain INFO/DEBUG and is
# filtered out once `main()` raises the root logger to SUMMARY, without
# having to touch every `log.info(...)` call site across the codebase. Set
# the env var `DISTILLER_LOG_LEVEL=INFO` (or DEBUG) before running to opt
# back into the full, verbose log for debugging.
SUMMARY = 25
logging.addLevelName(SUMMARY, "SUMMARY")

log = logging.getLogger(__name__)


def configure_logging() -> None:
    """Raise the root logger to ``SUMMARY`` — shared by this file's and
    ``predict_jax.py``'s ``main()``.

    ``logging.basicConfig(...)`` would be a no-op in either entrypoint —
    Hydra has already attached its own handler to the root logger by the
    time ``main()`` runs — so the level is set directly on the root logger
    instead, which works regardless of who owns the handler. Every other
    logger in the process (this project's own modules, and third-party
    ones: absl, jax, orbax, hydra) has no explicit level of its own, so it
    inherits this. Override with ``DISTILLER_LOG_LEVEL=INFO`` (or ``DEBUG``)
    to get the full, verbose log back for debugging.
    """
    level_name = os.environ.get("DISTILLER_LOG_LEVEL", "SUMMARY").upper()
    level = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "SUMMARY": SUMMARY,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }.get(level_name, SUMMARY)
    logging.getLogger().setLevel(level)


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
                optax.linear_schedule(peak_lr, 0.0, max(total_steps - warmup_steps, 1)),
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
    base_schedule = build_schedule(
        lr_scheduler, learning_rate, total_steps, warmup_ratio
    )
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


_MOSAIC_CONFIG_FIELDS = {f.name for f in dataclasses.fields(MosaicConfig)}


def _momos_config_from_cfg(momos_cfg: DictConfig) -> MosaicConfig:
    """Build a ``MosaicConfig`` from ``cfg.momos_backbone``.

    Forwards only real ``MosaicConfig`` fields, so ``momos_backbone``'s yaml
    can carry knobs that are not dataclass fields (``dict_lr_mult``,
    ``enabled``) without a filter list here going stale every time SPEC adds
    a field — the same pattern ``scripts/momos_phase_c.py``'s
    ``_mosaic_config`` already uses for the gate script's own config groups.
    """
    raw = OmegaConf.to_container(momos_cfg, resolve=True)
    kwargs = {k: v for k, v in raw.items() if k in _MOSAIC_CONFIG_FIELDS}
    return MosaicConfig(**kwargs)


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
    runtime = dataloader_runtime(cfg.runtime)
    # JAX has already opened CUDA and gone multithreaded in this process by
    # the time DataLoader workers start — fork()ing them afterward (torch's
    # Linux default) is the unsafe pattern Python's own RuntimeWarning
    # names. "spawn" avoids it; see lit_datamodule._build_dataloader.
    if runtime["num_workers"] > 0:
        runtime["multiprocessing_context"] = "spawn"
        # Pairs with "spawn": torch's default CPU-tensor sharing strategy
        # ("file_descriptor") sends one open fd per tensor across the
        # worker boundary. In a process that already has many fds open
        # (JAX/XLA, orbax's async checkpoint writer), that exhausts the
        # descriptor limit — "unable to open shared memory object ...
        # Too many open files". "file_system" shares via named temp files
        # instead of fds, avoiding the limit entirely. Process-global, but
        # only worker IPC uses it, so it's harmless for anything else.
        torch.multiprocessing.set_sharing_strategy("file_system")
    datamodule = DistillerDataModule(cfg, fold=fold_idx, runtime=runtime)
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

    momos_cfg_dc = cfg.get("momos_backbone")
    momos_enabled = bool(momos_cfg_dc.enabled) if momos_cfg_dc is not None else False
    if momos_enabled:
        return _run_momos_fold(
            cfg,
            fold_idx,
            seed,
            model,
            momos_cfg_dc,
            train_dl,
            val_dl,
            prepare=lambda batch, size: sharding_lib.shard_batch(
                mesh, to_jax_batch(batch, ladder, size, pad_token_id)
            ),
            train_batch=train_batch,
            eval_batch=eval_batch,
            num_epochs=num_epochs,
            total_steps=total_steps,
            trial=trial,
        )

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

    def _dummy_batch(rung: int, size: int):
        shaped = {
            "input_ids": np.zeros((size, rung), dtype=np.int64),
            "labels": np.full((size, rung), IGNORE_INDEX, dtype=np.int64),
            "attention_mask": np.zeros((size, rung), dtype=np.int64),
        }
        return sharding_lib.shard_batch(
            mesh, to_jax_batch(shaped, ladder, size, pad_token_id)
        )

    def _compile_train(rung: int) -> None:
        train_step.lower(model, optimizer, _dummy_batch(rung, train_batch)).compile()

    def _compile_eval(rung: int) -> None:
        eval_step.lower(model, _dummy_batch(rung, eval_batch)).compile()

    def _warmup_bucket_shapes():
        """AOT-compile ``train_step`` for every ladder rung, then kick off
        ``eval_step`` compiles in the background.

        Without this, the first batch of each new bucket shape triggers a
        real (blocking) XLA compile mid-epoch — that is the 0%-then-100% SM
        sawtooth seen in the first ~15 steps of a fold (5 rungs, each
        stalling on its own compile). ``jit(...).lower(...).compile()``
        populates the same shape-keyed cache a normal call would, but never
        *executes* the function, so it cannot touch optimiser/schedule state
        (grad-accum counters, LR step) the way calling ``train_step`` with a
        dummy batch would.

        Two overlapping tricks, both safe because compilation is CPU-side
        (XLA needs the device only to *run* an executable, not to build one)
        and never executes the traced function:

        * The 5 train shapes are compiled from a thread pool. The XLA/PJRT
          compile call releases the GIL, so the compiles do genuinely run
          concurrently (confirmed via aggregate CPU% during the pool) —
          but this is not a clean 5x: XLA's own backend codegen already
          uses multiple threads per compile, so 5-way pooling oversubscribes
          the machine rather than dividing the work evenly. Measured here:
          204s sequential vs 157s pooled (~1.3x) for 5 shapes on 16 cores.
          Still a real, free win — just don't expect it to divide by
          ``len(ladder)``.
        * The 5 eval shapes are handed to a *second*, non-blocking pool
          right after, so they compile in the background while the epoch
          loop is already dispatching real (GPU-bound) train steps — their
          cost is hidden behind training instead of adding to startup. The
          first call to ``evaluate()`` joins that pool once; if the epoch
          took longer than the eval compiles (virtually always true), the
          join is immediate.
        """
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=len(ladder)) as pool:
            list(pool.map(_compile_train, ladder))
        log.info(
            "Precompiled %d train shapes in %s (parallel)",
            len(ladder),
            format_seconds(time.time() - t0),
        )
        eval_pool = ThreadPoolExecutor(max_workers=len(ladder))
        eval_futures = [eval_pool.submit(_compile_eval, rung) for rung in ladder]
        return eval_pool, eval_futures

    warmup_start = time.time()
    eval_pool, eval_futures = _warmup_bucket_shapes()
    eval_compiled = False

    def evaluate() -> float:
        nonlocal eval_compiled
        if not eval_compiled:
            t0 = time.time()
            for f in eval_futures:
                f.result()
            eval_pool.shutdown(wait=False)
            eval_compiled = True
            log.info(
                "Eval shapes ready (joined background compile in %s)",
                format_seconds(time.time() - t0),
            )
        total, count = 0.0, 0
        val_progress = tqdm(val_dl, desc="validation", unit="step", leave=False)
        for batch in val_progress:
            loss = float(eval_step(model, prepare(batch, eval_batch)))
            if math.isfinite(loss):
                total += loss
                count += 1
                val_progress.set_postfix(loss=f"{total / count:.4f}")
        val_progress.close()
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
            progress = tqdm(
                train_dl,
                desc=f"epoch {epoch + 1}/{num_epochs}",
                unit="step",
                leave=False,
            )
            for batch in progress:
                loss = float(train_step(model, optimizer, prepare(batch, train_batch)))
                running += loss
                seen += 1
                global_step += 1
                progress.set_postfix(loss=f"{loss:.4f}", avg=f"{running / seen:.4f}")
            progress.close()

            eval_loss = evaluate()
            train_loss = running / seen if seen else float("nan")
            log.log(
                SUMMARY,
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
    log.log(SUMMARY, "  fold done in %s — val_loss=%.4f", elapsed, best)
    return best


def _run_momos_fold(
    cfg: DictConfig,
    fold_idx: int,
    seed: int,
    model: nnx.Module,
    momos_cfg_dc: DictConfig,
    train_dl,
    val_dl,
    *,
    prepare,
    train_batch: int,
    eval_batch: int,
    num_epochs: int,
    total_steps: int,
    trial: Optional[optuna.Trial],
) -> float:
    """The MoMos-enabled counterpart of ``run_fold``'s optimiser/loop section
    (SPEC_PHASE_E.md). Kept as a separate function rather than branching
    inline throughout ``run_fold`` so the dense path above is untouched code,
    not code now living inside an ``if``/``else`` — enabling MoMos can only
    ever be reached by explicitly setting ``momos_backbone.enabled: true``.

    Checkpoints the *reconstructed dense* params (``integration.merged_model``),
    not the compact dictionary — so anything downstream that loads a
    checkpoint (``predict.py``, ``model_jax/export.py``) keeps working
    unchanged. Saving the compact representation instead is real future
    work (it is the actual memory win the dictionary exists for) but is not
    part of this landing; see ``SPEC_PHASE_PERF.md``.
    """
    training = cfg.training
    momos_cfg = _momos_config_from_cfg(momos_cfg_dc)
    log.info(
        "Fold %d: MoMos backbone enabled — S=%d K=%d scale_mode=%s subset_size=%d "
        "(SPEC_PHASE_E.md)",
        fold_idx,
        momos_cfg.S,
        momos_cfg.K,
        momos_cfg.scale_mode,
        momos_cfg.subset_size,
    )

    base_schedule = build_schedule(
        str(training.lr_scheduler),
        float(training.learning_rate),
        total_steps,
        float(training.warmup_ratio),
    )
    dict_schedule = build_schedule(
        str(training.lr_scheduler),
        float(training.learning_rate) * float(momos_cfg_dc.get("dict_lr_mult", 1.0)),
        total_steps,
        float(training.warmup_ratio),
    )
    dict_optimizer = optax.adam(dict_schedule)
    # Every leaf tiling.default_include excludes on this architecture
    # (A_log, D, dt_bias, norm weights) is the "body_no_decay" group under
    # mimir_mamba2.param_group — see SPEC_PHASE_E.md — so one optimiser, not
    # train_jax's usual four-group multi_transform, is correct here. That
    # grouping does not generalise as-is to a model whose excluded leaves
    # aren't all no-decay/body; this integration has only been built and
    # tested against MimirMamba2Model.
    excluded_optimizer = optax.adamw(base_schedule, weight_decay=0.0)

    def momos_loss_fn(m, batch):
        return m.loss(batch["input_ids"], batch["labels"], batch["attention_mask"])

    bundle = momos_integration.init_bundle(
        model, momos_cfg, dict_optimizer, excluded_optimizer, jax.random.PRNGKey(seed)
    )
    n_mosaicked = bundle.mosaic.layout.n_values
    log.info(
        "MoMos: %d values mosaicked into K=%d motifs (S=%d)",
        n_mosaicked,
        momos_cfg.K,
        momos_cfg.S,
    )

    core_step = momos_integration.make_jit_step(
        bundle.mosaic.layout,
        bundle.mosaic.cfg,
        bundle.graphdef,
        momos_loss_fn,
        dict_optimizer,
        excluded_optimizer,
    )
    core_eval = momos_integration.make_jit_eval(
        bundle.mosaic.layout, bundle.mosaic.cfg, bundle.graphdef, momos_loss_fn
    )
    arrays = list(momos_integration.bundle_arrays(bundle))

    def evaluate(current_arrays) -> float:
        (
            motifs,
            mosaic_idx,
            active,
            scales,
            _opt_state,
            excluded,
            _excl_opt_state,
            rest,
        ) = current_arrays
        total, count = 0.0, 0
        val_progress = tqdm(val_dl, desc="validation", unit="step", leave=False)
        for batch in val_progress:
            loss = float(
                core_eval(
                    motifs,
                    mosaic_idx,
                    active,
                    scales,
                    excluded,
                    rest,
                    prepare(batch, eval_batch),
                )
            )
            if math.isfinite(loss):
                total += loss
                count += 1
                val_progress.set_postfix(loss=f"{total / count:.4f}")
        val_progress.close()
        return total / count if count else float("inf")

    manager = make_checkpoint_manager(
        str(Path(training.output_dir) / f"fold_{fold_idx}"), enabled=trial is None
    )
    best = float("inf")
    start = time.time()
    global_step = 0
    rng = jax.random.PRNGKey(seed + 1)

    try:
        for epoch in range(num_epochs):
            running, seen = 0.0, 0
            progress = tqdm(
                train_dl,
                desc=f"epoch {epoch + 1}/{num_epochs} (momos)",
                unit="step",
                leave=False,
            )
            for batch in progress:
                rng, sub = jax.random.split(rng)
                *arrays, loss, _swap = core_step(
                    *arrays, sub, prepare(batch, train_batch)
                )
                loss = float(loss)
                running += loss
                seen += 1
                global_step += 1
                progress.set_postfix(loss=f"{loss:.4f}", avg=f"{running / seen:.4f}")
            progress.close()

            eval_loss = evaluate(arrays)
            train_loss = running / seen if seen else float("nan")
            log.log(
                SUMMARY,
                "epoch %d/%d — train_loss=%.4f eval_loss=%.4f (momos)",
                epoch + 1,
                num_epochs,
                train_loss,
                eval_loss,
            )

            if eval_loss < best:
                best = eval_loss
                if manager is not None:
                    final_bundle = momos_integration.bundle_from_arrays(
                        tuple(arrays),
                        layout=bundle.mosaic.layout,
                        cfg=bundle.mosaic.cfg,
                        graphdef=bundle.graphdef,
                    )
                    merged = momos_integration.merged_model(final_bundle)
                    manager.save(
                        global_step,
                        args=ocp.args.StandardSave(
                            jax.tree.map(np.asarray, nnx.state(merged, nnx.Param))
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
    log.log(SUMMARY, "  fold done in %s — val_loss=%.4f (momos)", elapsed, best)
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
    configure_logging()
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    loss = run_fold(cfg, fold_idx=0)
    log.log(SUMMARY, "fold 0 best eval_loss = %.6f", loss)


if __name__ == "__main__":
    main()
