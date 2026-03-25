from typing import Optional
from datetime import datetime
from pathlib import Path
import argparse
import json
import yaml


def create_output_folder(output_dir: str, run_name: Optional[str] = None) -> Path:
    """
    Create a timestamped output folder for this training run.

    Args:
        output_dir: Base output directory
        run_name: Optional custom name for the run

    Returns:
        Path to the created folder
    """
    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        # Generate timestamp-based name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"decoder_{timestamp}"

    run_path = base_path / run_name

    # Handle name collisions
    if run_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_name = f"decoder_{timestamp}"
        run_path = base_path / run_name

    run_path.mkdir(parents=True, exist_ok=True)

    return run_path


def log_training_config(run_path: Path, args: argparse.Namespace, data_info: dict):
    """
    Save the training configuration to a JSON file.

    Args:
        run_path: Path to the output folder
        args: Parsed command line arguments
        data_info: Dataset information dictionary
    """
    config = {
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "model": {
            "output_dim": 768,
            "emb_dim": args.emb_dim,
            "num_layers": args.num_layers,
            "fwd_dim": args.fwd_dim,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "precision": args.precision,
            "min_epochs": args.min_epochs,
            "min_steps": args.min_steps,
        },
        "data": data_info,
        "early_stopping": {
            "enabled": args.early_stopping,
            "patience": args.early_stopping_patience,
        },
    }

    config_path = run_path / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Training config saved to: {config_path}")


def log_test_results(results, run_path):
    test_results_path = run_path / "test_results.json"
    with open(test_results_path, "w") as f:
        json.dump(results, f, indent=2)


def write_data_config(key, value, config):
    with open(config, "r") as file:
        settings = yaml.safe_load(file)

    settings[key] = value

    with open(config, "w") as file:
        yaml.safe_dump(settings, file)


def read_data_config(config):
    with open(config, "r") as file:
        settings = yaml.safe_load(file)

    return settings
