# -*- coding: utf-8 -*-
"""Inference / evaluation entrypoint for Distiller.

IMPORTANT: this script NEVER trains. It only loads an already-trained
checkpoint (provided via ``predict.checkpoint_dir``) and either:

  * scores it on the held-out test set (mode="evaluate"), or
  * generates completions for user-supplied prompts (mode="generate").

If no ``predict.checkpoint_dir`` is supplied the script exits with an error —
there is no base-model fallback and no training path.

Usage (this is a plain Hydra entrypoint, not argparse — every option,
including ``mode``, is a ``key=value`` override):
    # Score a trained checkpoint on the held-out test set
    python src/predict.py predict.mode=evaluate \
        predict.checkpoint_dir=outputs/final_model

    # Score a Lightning .ckpt instead
    python src/predict.py predict.mode=evaluate \
        predict.checkpoint_dir=outputs/best.ckpt

    # Generate completions for custom prompts
    python src/predict.py predict.mode=generate \
        predict.checkpoint_dir=outputs/final_model \
        predict.prompts='["Some input text fuel_type"]'
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from data.loader import _FIELD_MARKER_RE
from lit_datamodule import DistillerDataModule
from model.factory import DEFAULT_KIND, filter_kwargs, module_class
from utils import dataloader_runtime

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load a trained checkpoint (HF model dir OR Lightning .ckpt)
# ---------------------------------------------------------------------------

def _load_module_from_checkpoint(
    cfg: DictConfig,
    checkpoint_dir: str,
) -> L.LightningModule:
    """Load a trained module from a checkpoint path.

    Which class to load is decided by ``cfg.model.kind`` — the same selector
    training used — and the rest of ``cfg.model`` is forwarded to whichever of
    ``load_from_checkpoint``/``from_pretrained`` runs, keeping only the keys
    that class actually names (the two kinds disagree on, for instance,
    ``model_id`` vs ``donor_model_id``).

    Supports both a HuggingFace model directory (saved by
    ``module.save_pretrained``) and a Lightning ``.ckpt`` file. The two are
    distinguished by the filesystem: a ``.ckpt`` file is loaded via
    ``load_from_checkpoint``; anything else is treated as a model directory.
    """
    path = Path(checkpoint_dir)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    cls = module_class(model_cfg.pop("kind", DEFAULT_KIND))

    if path.is_file() and path.suffix == ".ckpt":
        log.info("Loading Lightning checkpoint from %s", path)
        module = cls.load_from_checkpoint(
            str(path),
            map_location="cpu",
            **filter_kwargs(cls.__init__, model_cfg),
        )
        module.eval()
        if torch.cuda.is_available():
            module.to("cuda")
        return module

    if not path.exists():
        log.error("Checkpoint path does not exist: %s", path)
        sys.exit(1)

    log.info("Loading trained model directory from %s", path)
    # No .to() here: each class's from_pretrained owns its own device
    # placement, because they load differently — DFMMimirModule hands the job
    # to accelerate via device_map="auto" (and moving such a model afterwards
    # raises), while MimirMamba2Module places itself.
    return cls.from_pretrained(
        **filter_kwargs(cls.from_pretrained, {**model_cfg, "save_path": str(path)})
    )


# ---------------------------------------------------------------------------
# Evaluate a checkpoint on the held-out test set (no training)
# ---------------------------------------------------------------------------

def evaluate_on_test(
    cfg: DictConfig,
    module: L.LightningModule,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Score an already-trained checkpoint on the held-out test set.

    Builds the same test split the training pipeline held out, via the same
    ``DistillerDataModule`` used for training (fold=-1 selects the held-out
    test set, never touched during CV), runs greedy decoding over it, and
    reports an exact-match rate. Does NOT train and does NOT save a model.
    """
    output_dir = output_dir or Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_cols = cfg.data.target_columns

    datamodule = DistillerDataModule(
        cfg, fold=-1, runtime=dataloader_runtime(cfg.runtime)
    )
    datamodule.set_module(module)
    datamodule.setup()

    test_samples = datamodule.get_test_samples()
    log.info("Test samples: %d", len(test_samples))

    # Extract the target column name for each sample. build_samples formats
    # the field marker as "<{col}>?", placed either at the tail of the input
    # (default) or the head (data.prompt_first=true, for Mamba2 — see
    # data.loader.build_samples_by_row), so locate it by regex rather than
    # assuming a fixed end. This makes per-field/per-column accuracy correct
    # regardless of ordering or tokenizer merging/decoding artifacts.
    sample_fields = []
    for s in test_samples:
        m = _FIELD_MARKER_RE.search(s["input"])
        col_name = m.group(1) if m else s["input"].rsplit(" ", 1)[-1].strip("<>?")
        sample_fields.append(col_name)

    test_dl = datamodule.test_dataloader()

    module.eval()
    all_predictions: List[Dict[str, Any]] = []

    # global_idx tracks the position within the original (unshuffled) sample
    # list so that sample_fields (built from the source DataFrame, not the
    # tokenized/decoed text) is indexed correctly across batches.
    global_idx = 0

    with torch.no_grad():
        for batch in test_dl:
            input_ids = batch["input_ids"].to(module.device)
            attention_mask = batch["attention_mask"].to(module.device)
            labels = batch["labels"]

            # Causal-LM convention: logits[:, t, :] predicts the token at
            # position t+1, not t (this is also how the model's own internal
            # loss aligns logits/labels). Comparing argmax(logits[:, t, :])
            # directly against labels[:, t] reads every prediction one token
            # too early, which silently produces a systematically shifted
            # decoded answer (e.g. ground truth "1.0" decoded as ".01",
            # "Ajva" as "vaAj" — every character present, just rotated by
            # however many characters that value's first token spans).
            # Shift right by one so predicted[:, t] lines up with labels[:, t].
            #
            # argmax_token_ids, rather than argmax over module.model(...).logits:
            # both models implement it, and the Mamba2 one reduces the
            # 262144-wide logits chunk by chunk instead of holding a
            # [batch, seq, 262144] tensor (6.4 GiB at batch 16) all at once.
            pred_ids = module.argmax_token_ids(input_ids, attention_mask)
            predicted = torch.nn.functional.pad(pred_ids[:, :-1], (1, 0), value=-1)
            for i in range(len(input_ids)):
                # Decode ONLY the target region (labels != -100). The model is
                # teacher-forced on the full sequence, so argmax over the input
                # region would just reconstruct the prompt; comparing the
                # full-sequence decode against the target-only ground truth
                # would make exact match impossible. Restricting both sides to
                # the target span yields a meaningful per-sample accuracy.
                true_mask = labels[i] != -100
                pred_text = module.tokenizer.decode(
                    predicted[i][true_mask], skip_special_tokens=True
                ) if true_mask.any() else ""
                true_text = module.tokenizer.decode(
                    labels[i][true_mask], skip_special_tokens=True
                ) if true_mask.any() else ""
                # Decode only the prompt portion (tokens before the target
                # span) so the "input" field in predictions doesn't include
                # the ground-truth output text that was concatenated during
                # tokenization.
                prompt_mask = labels[i] == -100
                inp_text = module.tokenizer.decode(
                    input_ids[i][prompt_mask], skip_special_tokens=True
                ) if prompt_mask.any() else ""
                all_predictions.append({
                    "input": inp_text,
                    "field": sample_fields[global_idx + i],
                    "ground_truth": true_text.strip(),
                    "prediction": pred_text.strip(),
                })
            global_idx += len(input_ids)

    # Per-sample exact-match accuracy = average accuracy per sample.
    exact_matches = sum(
        1 for p in all_predictions if p["ground_truth"] == p["prediction"]
    )
    exact_match_rate = exact_matches / len(all_predictions) if all_predictions else 0.0

    # Per-field accuracy: group samples by the target column name.
    # We use p["field"] (set from sample_fields above, derived from the original
    # pre-tokenization sample input) rather than re-extracting from the decoded
    # input text, which can lose or merge tokens and produce wrong field names.
    from collections import defaultdict as _dd

    field_correct: Dict[str, int] = _dd(int)
    field_total: Dict[str, int] = _dd(int)
    for p in all_predictions:
        field = p["field"]
        field_total[field] += 1
        if p["ground_truth"] == p["prediction"]:
            field_correct[field] += 1

    per_field_accuracy = {
        f: {
            "correct": field_correct[f],
            "total": field_total[f],
            "accuracy": field_correct[f] / field_total[f] if field_total[f] else 0.0,
        }
        for f in sorted(field_total)
    }

    # Per target-column accuracy: group by the config column name. field
    # names are exactly the target_cols entries (build_samples appends the
    # raw column name as the last token), so this is just per_field_accuracy
    # filtered/defaulted to target_cols — no separate accumulation needed.
    _empty_field = {"correct": 0, "total": 0, "accuracy": 0.0}
    per_column_accuracy = {
        col: per_field_accuracy.get(col, _empty_field) for col in target_cols
    }

    # Per-sample accuracy: each sample contributes 1.0 (match) or 0.0 (no match).
    # The average accuracy per sample is the exact_match_rate above, but we also
    # emit a per-sample vector so callers can compute the mean over any subset.
    per_sample_accuracy = [
        1.0 if p["ground_truth"] == p["prediction"] else 0.0
        for p in all_predictions
    ]

    # Whole-line accuracy: groups samples by source row (every n_fields samples
    # come from the same DataFrame row, since build_samples iterates rows →
    # target_columns). A line is "fully correct" only when ALL fields for that
    # row match exactly.
    n_fields = len(target_cols)
    n_lines = len(per_sample_accuracy) // n_fields
    whole_line_correct = sum(
        1
        for i in range(n_lines)
        if all(
            per_sample_accuracy[i * n_fields + j] == 1.0
            for j in range(n_fields)
        )
    )
    whole_line_accuracy = (
        whole_line_correct / n_lines if n_lines else 0.0
    )

    # DFM-Mimir-only: the Mamba2 module has no recurrent-cycle partition, so
    # the key stays absent there rather than reporting a pair of Nones.
    h_cycles = getattr(module.model.config, "H_cycles", None)
    l_cycles = getattr(module.model.config, "L_cycles", None)
    hrm_cycles = (
        {"hrm_cycles": {"H_cycles": h_cycles, "L_cycles": l_cycles}}
        if h_cycles is not None
        else {}
    )

    metrics = {
        "checkpoint": str(cfg.predict.checkpoint_dir),
        "n_test_samples": len(all_predictions),
        "n_test_rows": n_lines,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_match_rate,
        "avg_accuracy_per_sample": exact_match_rate,
        "per_sample_accuracy": per_sample_accuracy,
        "per_field_accuracy": per_field_accuracy,
        "per_column_accuracy": per_column_accuracy,
        "whole_line_accuracy": whole_line_accuracy,
        **hrm_cycles,
        "predictions": all_predictions,
    }

    preds_path = output_dir / "test_predictions.json"
    with open(preds_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info("Saved test predictions to %s", preds_path)

    return metrics


# ---------------------------------------------------------------------------
# Generate predictions for custom prompts
# ---------------------------------------------------------------------------

def generate_for_prompts(
    cfg: DictConfig,
    module: L.LightningModule,
    prompts: List[str],
    output_path: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """Generate predictions for a list of prompts using a trained checkpoint."""
    output_path = output_path or Path(cfg.training.output_dir) / "custom_predictions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for prompt in prompts:
        pred = module.generate(
            prompt=prompt,
            max_new_tokens=cfg.predict.get("max_new_tokens", 128),
            temperature=cfg.predict.get("temperature", 0.7),
            do_sample=cfg.predict.get("do_sample", True),
        )
        results.append({"prompt": prompt, "prediction": pred})
        log.info("Prompt: %s", prompt[:80])
        log.info("Prediction: %s", pred)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved predictions to %s", output_path)

    return results


# ---------------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------------

@hydra.main(
    config_path=str(Path(__file__).parent / "config"),
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    checkpoint_dir = cfg.predict.get("checkpoint_dir", None)
    if not checkpoint_dir:
        log.error(
            "No checkpoint provided. Set predict.checkpoint_dir to a trained "
            "model directory (final_model/) or a Lightning .ckpt file. "
            "predict.py does not train and will not fall back to a base model."
        )
        sys.exit(1)

    mode = cfg.predict.mode
    module = _load_module_from_checkpoint(cfg, checkpoint_dir)

    if mode == "evaluate":
        metrics = evaluate_on_test(cfg, module)
        log.info(
            "Average accuracy per sample: %.4f (%d/%d exact matches)",
            metrics["avg_accuracy_per_sample"],
            metrics["exact_matches"],
            metrics["n_test_samples"],
        )
        log.info(
            "Whole-line accuracy (%d rows): %.4f (%d/%d lines fully correct)",
            metrics["n_test_rows"],
            metrics["whole_line_accuracy"],
            round(metrics["whole_line_accuracy"] * metrics["n_test_rows"]),
            metrics["n_test_rows"],
        )
        if "hrm_cycles" in metrics:
            log.info(
                "HRM cycles: H=%s L=%s",
                metrics["hrm_cycles"]["H_cycles"],
                metrics["hrm_cycles"]["L_cycles"],
            )
        log.info("Per-field accuracy:")
        for f, m in metrics.get("per_field_accuracy", {}).items():
            log.info("  %s: %d/%d = %.4f", f, m["correct"], m["total"], m["accuracy"])
        log.info("Per-column accuracy:")
        for col, m in metrics.get("per_column_accuracy", {}).items():
            log.info(
                "  %s: %d/%d = %.4f", col, m["correct"], m["total"], m["accuracy"]
            )

    elif mode == "generate":
        prompts = cfg.predict.prompts
        if not prompts:
            log.error(
                "No prompts provided. Set predict.prompts in config or override "
                "on the command line, e.g. "
                'predict.prompts=\'["your prompt here"]\''
            )
            sys.exit(1)
        generate_for_prompts(cfg, module, prompts)

    else:
        log.error(
            "Unknown mode: %s. Use 'evaluate' or 'generate'. "
            "(interactive mode was removed — predict.py only scores/evaluates "
            "a supplied checkpoint.)", mode
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
