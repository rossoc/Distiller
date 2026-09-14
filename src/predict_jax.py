# -*- coding: utf-8 -*-
"""Inference / evaluation for the JAX/Flax path — counterpart of predict.py.

Scores a train_jax.py checkpoint on the **validation split** of a given
fold (not the held-out test set). Mirrors predict.py's per-sample exact-
match, per-field, and whole-line accuracy.

Usage:
    python src/predict_jax.py predict.checkpoint_dir=outputs/.../fold_0 predict.fold=0
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax import nnx
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data.loader import extract_field_name
from lit_datamodule import DistillerDataModule
from model_jax import sharding as sharding_lib
from model_jax.batching import bucket_ladder, to_jax_batch
from model_jax.donor_projection import load_donor_tables
from model_jax.factory import DEFAULT_KIND, build_model
from model_jax.mimir_mamba2 import IGNORE_INDEX
from train_jax import SUMMARY, configure_logging, restore_params
from utils import dataloader_runtime

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer adapter (same role as train_jax._TokenizerHolder)
# ---------------------------------------------------------------------------


class _TokenizerHolder:
    def __init__(self, donor_dir: str) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(donor_dir))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id


# ---------------------------------------------------------------------------
# Model rebuild
# ---------------------------------------------------------------------------


def _build_model(cfg: DictConfig):
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    kind = model_cfg.pop("kind", DEFAULT_KIND)
    donor_dir = model_cfg.pop("donor_dir", None) or model_cfg.get("donor_model_id")
    tables = load_donor_tables(donor_dir, dtype=jnp.bfloat16)
    model = build_model(kind, model_cfg, tables=tables, rngs=nnx.Rngs(0))
    return model, donor_dir


def _is_momos(cfg: DictConfig) -> bool:
    momos_cfg = cfg.get("momos_backbone")
    return bool(momos_cfg.enabled) if momos_cfg is not None else False


def _load_momos(cfg: DictConfig, checkpoint_dir: str) -> nnx.Module:
    """Load a MoMos checkpoint into the reconstructed dense model."""
    import dataclasses as _dc
    import optax

    from model_jax.mimos.integration import init_bundle, merged_model
    from model_jax.momos.state import MosaicConfig

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    kind = model_cfg.pop("kind", DEFAULT_KIND)
    donor_dir = model_cfg.pop("donor_dir", None) or model_cfg.get("donor_model_id")
    tables = load_donor_tables(donor_dir, dtype=jnp.bfloat16)
    base = build_model(kind, model_cfg, tables=tables, rngs=nnx.Rngs(0))

    raw = OmegaConf.to_container(cfg.momos_backbone, resolve=True)
    fields = {f.name for f in _dc.fields(MosaicConfig)}
    momos_cfg = MosaicConfig(**{k: v for k, v in raw.items() if k in fields})

    dummy_opt = optax.adam(1.0)
    bundle = init_bundle(base, momos_cfg, dummy_opt, dummy_opt, jax.random.PRNGKey(0))
    merged = merged_model(bundle)
    restore_params(checkpoint_dir, merged)
    return merged


# ---------------------------------------------------------------------------
# Metrics (mirrors predict.py.evaluate_on_test)
# ---------------------------------------------------------------------------


def _top_errors_per_field(
    all_predictions: List[Dict[str, Any]],
    target_cols: List[str],
    top_k: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Most common (ground_truth -> prediction) mismatches, per target field.

    Restricted to `target_cols` — same reason `per_column_accuracy` is: a row
    whose prompt got truncated past its `<field>?` marker (see
    `extract_field_name`'s fallback) reports a bogus field name, not a real
    target column, and its "errors" would be meaningless here too.
    """
    by_field: Dict[str, Counter] = defaultdict(Counter)
    for p in all_predictions:
        if p["field"] not in target_cols:
            continue
        if p["ground_truth"] != p["prediction"]:
            by_field[p["field"]][(p["ground_truth"], p["prediction"])] += 1

    return {
        field: [
            {"ground_truth": gt, "prediction": pred, "count": count}
            for (gt, pred), count in by_field[field].most_common(top_k)
        ]
        for field in target_cols
    }


def _compute_metrics(
    all_predictions: List[Dict[str, Any]],
    target_cols: List[str],
    checkpoint: str,
    output_dir: Path,
) -> Dict[str, Any]:
    exact_matches = sum(
        1 for p in all_predictions if p["ground_truth"] == p["prediction"]
    )
    exact_match_rate = exact_matches / len(all_predictions) if all_predictions else 0.0

    field_correct: Dict[str, int] = defaultdict(int)
    field_total: Dict[str, int] = defaultdict(int)
    for p in all_predictions:
        f = p["field"]
        field_total[f] += 1
        if p["ground_truth"] == p["prediction"]:
            field_correct[f] += 1

    per_field_accuracy = {
        f: {
            "correct": field_correct[f],
            "total": field_total[f],
            "accuracy": field_correct[f] / field_total[f] if field_total[f] else 0.0,
        }
        for f in sorted(field_total)
    }

    _empty = {"correct": 0, "total": 0, "accuracy": 0.0}
    per_column_accuracy = {
        col: per_field_accuracy.get(col, _empty) for col in target_cols
    }

    per_sample_accuracy = [
        1.0 if p["ground_truth"] == p["prediction"] else 0.0 for p in all_predictions
    ]

    n_fields = len(target_cols)
    n_lines = len(per_sample_accuracy) // n_fields
    whole_line_correct = sum(
        1
        for i in range(n_lines)
        if all(per_sample_accuracy[i * n_fields + j] == 1.0 for j in range(n_fields))
    )
    whole_line_accuracy = whole_line_correct / n_lines if n_lines else 0.0

    avg_accuracy_per_field = (
        sum(m["accuracy"] for m in per_column_accuracy.values()) / len(target_cols)
        if target_cols
        else 0.0
    )
    top_errors_per_field = _top_errors_per_field(all_predictions, target_cols)

    metrics = {
        "checkpoint": checkpoint,
        "n_val_samples": len(all_predictions),
        "n_val_rows": n_lines,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_match_rate,
        "avg_accuracy_per_sample": exact_match_rate,
        "avg_accuracy_per_field": avg_accuracy_per_field,
        "per_sample_accuracy": per_sample_accuracy,
        "per_field_accuracy": per_field_accuracy,
        "per_column_accuracy": per_column_accuracy,
        "top_errors_per_field": top_errors_per_field,
        "whole_line_accuracy": whole_line_accuracy,
        "predictions": all_predictions,
    }

    out_path = output_dir / "val_predictions.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.log(SUMMARY, "Saved validation predictions to %s", out_path)
    return metrics


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------


def evaluate(cfg: DictConfig, fold_idx: int, checkpoint_dir: str) -> Dict[str, Any]:
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _is_momos(cfg):
        log.info("MoMos checkpoint detected — loading dense merged model")
        model = _load_momos(cfg, checkpoint_dir)
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        donor_dir = model_cfg.pop("donor_dir", None) or model_cfg.get("donor_model_id")
    else:
        model, donor_dir = _build_model(cfg)
        restore_params(checkpoint_dir, model)
    model.eval()

    tokenizer_holder = _TokenizerHolder(donor_dir)
    pad_token_id = int(tokenizer_holder.pad_token_id or 0)

    runtime = dataloader_runtime(cfg.runtime)
    if runtime["num_workers"] > 0:
        runtime["multiprocessing_context"] = "spawn"
        torch.multiprocessing.set_sharing_strategy("file_system")

    datamodule = DistillerDataModule(cfg, fold=fold_idx, runtime=runtime)
    datamodule.set_module(tokenizer_holder)
    datamodule.setup()
    val_dl = datamodule.val_dataloader()

    ladder = bucket_ladder(int(cfg.data.max_length))
    eval_batch = max(
        1, int(int(cfg.training.batch_size) * int(cfg.training.eval_batch_multiplier))
    )

    mesh = sharding_lib.make_mesh(cfg.get("mesh_shape"))

    def prepare(batch, size: int):
        return sharding_lib.shard_batch(
            mesh, to_jax_batch(batch, ladder, size, pad_token_id)
        )

    @nnx.jit
    def eval_step(model, batch):
        return model.argmax_token_ids(batch["input_ids"], batch["attention_mask"])

    target_cols = cfg.data.target_columns
    all_predictions: List[Dict[str, Any]] = []

    for batch in tqdm(val_dl, desc="validation", unit="step", leave=False):
        rows = batch["input_ids"].shape[0]
        jax_batch = prepare(batch, rows)
        pred_ids: np.ndarray = np.asarray(eval_step(model, jax_batch))  # [rows, seq]
        labels_np = np.asarray(jax_batch["labels"])
        input_ids_np = np.asarray(jax_batch["input_ids"])

        # Shift right by one so predicted[t] aligns with labels[t]
        # (logit at t predicts token t+1), mirroring predict.py.
        predicted = np.concatenate(
            [np.full((rows, 1), -1, dtype=pred_ids.dtype), pred_ids[:, :-1]], axis=1
        )

        for i in range(rows):
            true_mask = labels_np[i] != -100
            pred_text = (
                tokenizer_holder.tokenizer.decode(
                    predicted[i][true_mask], skip_special_tokens=True
                )
                if true_mask.any()
                else ""
            )
            true_text = (
                tokenizer_holder.tokenizer.decode(
                    labels_np[i][true_mask], skip_special_tokens=True
                )
                if true_mask.any()
                else ""
            )
            prompt_mask = labels_np[i] == -100
            inp_text = (
                tokenizer_holder.tokenizer.decode(
                    input_ids_np[i][prompt_mask], skip_special_tokens=True
                )
                if prompt_mask.any()
                else ""
            )
            all_predictions.append(
                {
                    "input": inp_text,
                    "field": extract_field_name(inp_text),
                    "ground_truth": true_text.strip(),
                    "prediction": pred_text.strip(),
                }
            )

    metrics = _compute_metrics(all_predictions, target_cols, checkpoint_dir, output_dir)
    return metrics


# ---------------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------------


@hydra.main(
    config_path=str(Path(__file__).parent / "config"),
    config_name="config_jax",
)
def main(cfg: DictConfig) -> None:
    configure_logging()
    checkpoint_dir = cfg.predict.get("checkpoint_dir")
    fold_idx = int(cfg.predict.get("fold", 0))
    if not checkpoint_dir:
        log.error(
            "predict.checkpoint_dir is required — point it at the orbax "
            "checkpoint directory, e.g. outputs/.../fold_0"
        )
        sys.exit(1)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    metrics = evaluate(cfg, fold_idx, str(checkpoint_dir))
    log.log(
        SUMMARY,
        "Exact match rate (per sample): %.4f (%d/%d)",
        metrics["exact_match_rate"],
        metrics["exact_matches"],
        metrics["n_val_samples"],
    )
    log.log(
        SUMMARY,
        "Whole-line accuracy (%d rows): %.4f (%d/%d lines fully correct)",
        metrics["n_val_rows"],
        metrics["whole_line_accuracy"],
        round(metrics["whole_line_accuracy"] * metrics["n_val_rows"]),
        metrics["n_val_rows"],
    )
    log.log(
        SUMMARY,
        "Average accuracy per field: %.4f",
        metrics["avg_accuracy_per_field"],
    )
    log.log(SUMMARY, "Per-field accuracy:")
    # `per_column_accuracy` — not the raw `per_field_accuracy` — restricted to
    # cfg.data.target_columns. A row whose prompt got truncated past its
    # `<field>?` marker (see data.loader.extract_field_name's fallback) is
    # otherwise reported under a garbage "field" name (a stray word from the
    # truncated source text), which `per_field_accuracy` still carries for
    # debugging but has no business showing up here.
    for f, m in metrics.get("per_column_accuracy", {}).items():
        log.log(
            SUMMARY, "  %s: %d/%d = %.4f", f, m["correct"], m["total"], m["accuracy"]
        )

    log.log(SUMMARY, "Top errors per field:")
    for f, errors in metrics.get("top_errors_per_field", {}).items():
        log.log(SUMMARY, "  %s:", f)
        if not errors:
            log.log(SUMMARY, "    (no errors)")
        for e in errors:
            log.log(
                SUMMARY,
                "    %r -> %r  (x%d)",
                e["ground_truth"],
                e["prediction"],
                e["count"],
            )


if __name__ == "__main__":
    main()

