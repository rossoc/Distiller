import json
from collections import defaultdict


def parse_label_fields(label_text: str) -> dict:
    """
    Parse a label string into a dictionary of field-value pairs.

    Args:
        label_text: String in format "Field1: value1\nField2: value2\n..."

    Returns:
        Dictionary mapping field names to values
    """
    fields = {}
    for line in label_text.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    return fields


def parse_inference(json_path: str) -> list[tuple[dict, dict]]:
    """
    Extract ground truth and predictions.

    Args:
        json_path: Path to the inference results JSON file

    Returns:
        List of Dictionaries for target and prediction
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])

    res = []
    for result in results:
        ground_truth = result.get("ground_truth", "")
        predictions = result.get("predictions", [])

        gt_fields = parse_label_fields(ground_truth)

        # Check top-1 prediction
        if predictions:
            top1_pred = predictions[0].get("text", "")
            pred_fields = parse_label_fields(top1_pred)

        res.append((gt_fields, pred_fields))

    return res


def simple_count(result):
    tot = {key: 0 for key in result[0][0].keys()}
    correct = {key: 0 for key in result[0][0].keys()}
    for sample in result:
        for key, value in sample[0].items():
            tot[key] += 1

            if value == sample[1][key]:
                correct[key] += 1
    return tot, correct


def compute_match_statistics(json_path: str, top_k: int = 1) -> dict:
    """
    Compute match statistics between ground truth and predictions.

    Args:
        json_path: Path to the inference results JSON file
        top_k: Number of top predictions to consider (default: 1)

    Returns:
        Dictionary with match statistics
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    n_samples = len(results)

    # Field-level match counters
    field_matches = defaultdict(int)
    field_totals = defaultdict(int)

    # Exact match counters
    exact_matches_top1 = 0
    exact_matches_topk = 0

    # Score statistics
    scores = []

    for result in results:
        ground_truth = result.get("ground_truth", "")
        predictions = result.get("predictions", [])

        gt_fields = parse_label_fields(ground_truth)

        # Check top-1 prediction
        if predictions:
            top1_pred = predictions[0].get("text", "")
            top1_score = predictions[0].get("score")

            if top1_score is not None:
                scores.append(top1_score)

            pred_fields = parse_label_fields(top1_pred)

            # Check exact match for top-1
            if pred_fields == gt_fields:
                exact_matches_top1 += 1

            # Check field-level matches for top-1
            for field, gt_value in gt_fields.items():
                field_totals[field] += 1
                if pred_fields.get(field) == gt_value:
                    field_matches[field] += 1

            # Check top-k exact match
            for pred in predictions[:top_k]:
                pred_fields_k = parse_label_fields(pred.get("text", ""))
                if pred_fields_k == gt_fields:
                    exact_matches_topk += 1
                    break

    # Compute statistics
    stats = {
        "n_samples": n_samples,
        "top_k": top_k,
        "exact_match_top1": exact_matches_top1,
        "exact_match_topk": exact_matches_topk,
        "exact_match_rate_top1": exact_matches_top1 / n_samples if n_samples > 0 else 0,
        "exact_match_rate_topk": exact_matches_topk / n_samples if n_samples > 0 else 0,
        "field_accuracy": {},
        "avg_score": sum(scores) / len(scores) if scores else 0,
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
    }

    # Compute field-level accuracy
    for field in field_totals:
        stats["field_accuracy"][field] = field_matches[field] / field_totals[field]

    return stats


def print_match_statistics(stats: dict):
    """
    Print match statistics in a readable format.

    Args:
        stats: Dictionary returned by compute_match_statistics
    """
    print("=" * 60)
    print("Match Statistics")
    print("=" * 60)
    print(f"Total samples: {stats['n_samples']}")
    print(f"Top-k considered: {stats['top_k']}")
    print()
    print("Exact Match Rates:")
    print(
        f"  Top-1: {stats['exact_match_rate_top1']:.2%} ({stats['exact_match_top1']}/{stats['n_samples']})"
    )
    print(
        f"  Top-{stats['top_k']}: {stats['exact_match_rate_topk']:.2%} ({stats['exact_match_topk']}/{stats['n_samples']})"
    )
    print()
    print("Score Statistics:")
    print(f"  Average: {stats['avg_score']:.4f}")
    print(f"  Min: {stats['min_score']:.4f}")
    print(f"  Max: {stats['max_score']:.4f}")
    print()
    print("Field-Level Accuracy:")
    for field, acc in sorted(stats["field_accuracy"].items()):
        print(f"  {field}: {acc:.2%}")
    print("=" * 60)


def compute_per_label_metrics(result: list[tuple[dict, dict]]) -> dict:
    """
    Compute comprehensive per-label metrics for diffusion model evaluation.

    For each label field, computes:
    - Accuracy: exact match rate
    - Confusion matrix (for categorical fields)
    - Error analysis: most common mispredictions
    - Value distribution comparison (GT vs Pred)

    Args:
        result: List of (ground_truth_dict, prediction_dict) tuples

    Returns:
        Dictionary with per-label metrics
    """
    # Get all field names
    all_fields = set()
    for gt, pred in result:
        all_fields.update(gt.keys())

    metrics = {}

    for field in sorted(all_fields):
        field_metrics = {
            "field_name": field,
            "n_samples": 0,
            "n_correct": 0,
            "accuracy": 0.0,
            "confusion": {},
            "gt_value_counts": {},
            "pred_value_counts": {},
            "top_errors": [],
        }

        gt_counts = defaultdict(int)
        pred_counts = defaultdict(int)
        confusion = defaultdict(lambda: defaultdict(int))
        errors = defaultdict(int)

        for gt, pred in result:
            if field not in gt:
                continue

            gt_val = gt.get(field, "")
            pred_val = pred.get(field, "")

            field_metrics["n_samples"] += 1
            gt_counts[gt_val] += 1
            pred_counts[pred_val] += 1
            confusion[gt_val][pred_val] += 1

            if gt_val == pred_val:
                field_metrics["n_correct"] += 1
            else:
                errors[(gt_val, pred_val)] += 1

        # Compute accuracy
        if field_metrics["n_samples"] > 0:
            field_metrics["accuracy"] = (
                field_metrics["n_correct"] / field_metrics["n_samples"]
            )

        # Convert confusion matrix to regular dict
        field_metrics["confusion"] = {
            gt: dict(preds) for gt, preds in confusion.items()
        }

        # Store value distributions
        field_metrics["gt_value_counts"] = dict(gt_counts)
        field_metrics["pred_value_counts"] = dict(pred_counts)

        # Get top 5 most common errors
        sorted_errors = sorted(errors.items(), key=lambda x: x[1], reverse=True)
        field_metrics["top_errors"] = [
            {"gt": gt, "pred": pred, "count": count}
            for (gt, pred), count in sorted_errors[:5]
        ]

        metrics[field] = field_metrics

    return metrics


def print_per_label_metrics(metrics: dict):
    """
    Print per-label metrics in a readable format.

    Args:
        metrics: Dictionary returned by compute_per_label_metrics
    """
    print("\n" + "=" * 80)
    print("PER-LABEL METRICS")
    print("=" * 80)

    for field, field_metrics in metrics.items():
        print(f"\n{'─' * 80}")
        print(f"Field: {field}")
        print(f"{'─' * 80}")
        print(
            f"Accuracy: {field_metrics['accuracy']:.2%} "
            f"({field_metrics['n_correct']}/{field_metrics['n_samples']})"
        )

        # Print value distributions
        print("\nGround Truth Value Distribution:")
        sorted_gt = sorted(
            field_metrics["gt_value_counts"].items(), key=lambda x: x[1], reverse=True
        )[:10]
        for val, count in sorted_gt:
            pct = count / field_metrics["n_samples"] * 100
            print(f"  {val!r:40s} : {count:5d} ({pct:5.1f}%)")

        print("\nPrediction Value Distribution:")
        sorted_pred = sorted(
            field_metrics["pred_value_counts"].items(), key=lambda x: x[1], reverse=True
        )[:10]
        for val, count in sorted_pred:
            pct = count / max(field_metrics["n_samples"], 1) * 100
            print(f"  {val!r:40s} : {count:5d} ({pct:5.1f}%)")

        # Print top errors
        if field_metrics["top_errors"]:
            print("\nTop 5 Most Common Errors:")
            for i, error in enumerate(field_metrics["top_errors"], 1):
                print(
                    f"  {i}. GT: {error['gt']!r:35s} → Pred: {error['pred']!r:35s} "
                    f"(count: {error['count']})"
                )

        # Print confusion matrix summary (only for fields with limited unique values)
        unique_gt = len(field_metrics["gt_value_counts"])
        unique_pred = len(field_metrics["pred_value_counts"])
        if unique_gt <= 10 and unique_pred <= 10 and unique_gt > 1:
            print("\nConfusion Matrix (rows=GT, cols=Pred):")
            # Get all unique values
            all_vals = sorted(
                set(field_metrics["gt_value_counts"].keys())
                | set(field_metrics["pred_value_counts"].keys())
            )
            # Header
            print("     ", end="")
            for val in all_vals[:8]:
                short_val = val[:10] if val else "<empty>"
                print(f"{short_val:>12}", end="")
            print()
            # Rows
            for gt_val in all_vals[:8]:
                print(f"{gt_val[:10] if gt_val else '<empty>':>5}", end="")
                for pred_val in all_vals[:8]:
                    count = field_metrics["confusion"].get(gt_val, {}).get(pred_val, 0)
                    if count > 0:
                        print(f"{count:>12}", end="")
                    else:
                        print(f"{'·':>12}", end="")
                print()

    print("\n" + "=" * 80)
