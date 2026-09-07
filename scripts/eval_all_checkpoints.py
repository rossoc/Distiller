#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate every per-fold best checkpoint under cfg.training.output_dir
and report average accuracy per sample for each.

Reuses predict.py's evaluate_on_test() with the HRM config from
src/config/model/default.yaml (H_cycles=1, L_cycles=1 = no same-level recursion).

Usage:
    cd /home/local/4BC/Distiller
    .venv/bin/python scripts/eval_all_checkpoints.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Make src/ importable so we can import from predict, model, data, etc.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from predict import evaluate_on_test, _load_module_from_checkpoint  # noqa: E402

log = logging.getLogger(__name__)


@hydra.main(
    config_path=str(SRC_DIR / "config"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Each fold's single best-on-validation-loss checkpoint lives at
    # {output_dir}/fold_{fold_idx}/best.ckpt (see train.py::run_fold).
    checkpoint_dir = Path(cfg.training.output_dir)
    ckpt_files = sorted(checkpoint_dir.glob("fold_*/best.ckpt"))
    if not ckpt_files:
        log.error("No fold checkpoints found under %s", checkpoint_dir)
        sys.exit(1)

    log.info("Found %d checkpoints under %s", len(ckpt_files), checkpoint_dir)

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ckpt in ckpt_files:
        # Every fold's checkpoint is named "best.ckpt" (see _best_checkpoint in
        # callback.py) — qualify it by its parent fold_N/ directory so results
        # and output filenames don't collide across folds.
        name = f"{ckpt.parent.name}/{ckpt.name}"
        log.info("=== Evaluating %s ===", name)

        cfg.predict.checkpoint_dir = str(ckpt)
        cfg.predict.mode = "evaluate"

        module = _load_module_from_checkpoint(cfg, str(ckpt))
        metrics = evaluate_on_test(cfg, module)

        results.append({
            "checkpoint": name,
            "avg_accuracy_per_sample": metrics["avg_accuracy_per_sample"],
            "whole_line_accuracy": metrics["whole_line_accuracy"],
            "exact_matches": metrics["exact_matches"],
            "n_test_samples": metrics["n_test_samples"],
            "n_test_rows": metrics["n_test_rows"],
            "hrm_cycles": metrics["hrm_cycles"],
            "per_column_accuracy": {
                col: m["accuracy"]
                for col, m in metrics["per_column_accuracy"].items()
            },
        })

        # Save per-checkpoint full predictions
        safe_name = name.replace("/", "_").replace(".ckpt", "")
        out_path = output_dir / f"test_predictions_{safe_name}.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        log.info("  Saved per-checkpoint results to %s", out_path)

    # Save summary
    summary_path = output_dir / "all_checkpoint_accuracies.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved summary to %s", summary_path)

    # Print table
    print("\n" + "=" * 95)
    print(f"{'Checkpoint':<42} {'Avg Acc/Sample':>16} {'Whole-Line':>12} {'Matches':>10} {'HRM':>10}")
    print("=" * 95)
    for r in results:
        hrm = f"H={r['hrm_cycles']['H_cycles']},L={r['hrm_cycles']['L_cycles']}"
        print(f"{r['checkpoint']:<42} {r['avg_accuracy_per_sample']:>16.4f} "
              f"{r['whole_line_accuracy']:>12.4f} {r['exact_matches']:>10} {hrm:>10}")
    print("=" * 95)
    overall = sum(r['avg_accuracy_per_sample'] for r in results) / len(results)
    overall_wl = sum(r['whole_line_accuracy'] for r in results) / len(results)
    print(f"{'OVERALL AVERAGE':<42} {overall:>16.4f} {overall_wl:>12.4f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
