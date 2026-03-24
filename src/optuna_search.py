#!/usr/bin/env python3
"""
Hyperparameter optimization script for the Embedding Decoder using Optuna.

This script uses Optuna to search for optimal hyperparameters:
- num_heads: Number of attention heads
- num_layers: Number of transformer decoder layers
- fwd_dim: Feed-forward dimension
- learning_rate: Learning rate for optimizer

Multiple trials can be run in parallel to utilize available GPU/CPU resources.

Usage:
    python src/optuna_search.py --n_trials 20 --n_parallel 3 --epochs 30

Output:
    outputs/optuna/study_{timestamp}/
    ├── best_trial.json          # Best trial configuration
    ├── study_results.json       # All trial results
    ├── optimization_history.png # Optimization history plot
    ├── parameter_importances.png # Parameter importance plot
    └── trial_{N}/               # Individual trial outputs
        ├── best_model.pt
        ├── training_config.json
        └── lightning_logs/
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add src directory to path for imports when running as script
if __name__ == "__main__":
    script_dir = Path(__file__).parent.absolute()
    sys.path.insert(0, str(script_dir))

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar

from model.diffusion import EmbeddingDecoderLightning
from model.lightning_interfaces import BestModelSaveCallback
from data.datamodule import EmbeddingDecoderDataModule
from util.logger import save_training_config
from util.randomness import setup_run


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization for Embedding Decoder using Optuna"
    )

    # Optuna arguments
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of trials to run (default: 20)",
    )
    parser.add_argument(
        "--n_parallel",
        type=int,
        default=1,
        help="Number of parallel trials to run (default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout in hours for the entire study (default: None)",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Name for this study (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Database URL for distributed optimization (default: in-memory)",
    )

    # Search space arguments
    parser.add_argument(
        "--num_heads_choices",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help="Choices for num_heads (default: 4 8 12 16)",
    )
    parser.add_argument(
        "--num_layers_choices",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="Choices for num_layers (default: 2 4 6 8)",
    )
    parser.add_argument(
        "--fwd_dim_choices",
        type=int,
        nargs="+",
        default=[1024, 516, 2048, 4096],
        help="Choices for fwd_dim (default: 1024 2048 3072 4096)",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=1e-5,
        help="Minimum learning rate for log-uniform search (default: 1e-5)",
    )
    parser.add_argument(
        "--lr_max",
        type=float,
        default=1e-3,
        help="Maximum learning rate for log-uniform search (default: 1e-3)",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/optuna",
        help="Base directory for outputs (default: outputs/optuna)",
    )

    # Data arguments
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length for embeddings",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.5, help="Ratio of data for training"
    )
    parser.add_argument(
        "--eval_ratio", type=float, default=0.1, help="Ratio of data for evaluation"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.4, help="Ratio of data for testing"
    )

    # Model arguments (fixed, not optimized)
    parser.add_argument(
        "--emb_dim", type=int, default=768, help="Model embedding dimension"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--loss_alpha",
        type=float,
        default=0.5,
        help="Weight for MSE loss in combined loss",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Device arguments
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda", "mps"],
        help="Accelerator to use",
    )
    parser.add_argument(
        "--devices", type=int, default=1, help="Number of devices to use per trial"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        choices=[
            "64",
            "32-true",
            "32",
            "16-true",
            "16-mixed",
            "bf16-true",
            "bf16-mixed",
        ],
        help="Precision for training",
    )

    # Callback arguments
    parser.add_argument(
        "--early_stopping", action="store_true", help="Enable early stopping"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--min_epochs", type=int, default=1, help="Minimum number of epochs to train"
    )
    parser.add_argument(
        "--min_steps", type=int, default=None, help="Minimum number of steps to train"
    )
    parser.add_argument(
        "--pruning_patience",
        type=int,
        default=5,
        help="Patience for Optuna pruning (epochs without improvement)",
    )
    parser.add_argument(
        "--progress_bar_refresh_rate",
        type=int,
        default=100,
        help="Update progress bar every N batches (default: 100)",
    )
    parser.add_argument(
        "--disable_progress_bar",
        action="store_true",
        default=True,
        help="Disable progress bar entirely (default: True for Optuna)",
    )

    # Logging arguments
    parser.add_argument(
        "--log_every_n_steps", type=int, default=50, help="Log every N steps"
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=1.0,
        help="Validation check interval (1.0 = every epoch)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable WandB logging",
    )

    return parser.parse_args()


def suggest_hyperparameters(trial: optuna.Trial, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Suggest hyperparameters for a trial.

    Args:
        trial: Optuna trial object
        args: Command-line arguments

    Returns:
        Dictionary of hyperparameters
    """
    return {
        "num_heads": trial.suggest_categorical("num_heads", args.num_heads_choices),
        "num_layers": trial.suggest_categorical("num_layers", args.num_layers_choices),
        "fwd_dim": trial.suggest_categorical("fwd_dim", args.fwd_dim_choices),
        "learning_rate": trial.suggest_float(
            "learning_rate", args.lr_min, args.lr_max, log=True
        ),
    }


def create_trial_directory(output_dir: Path, trial_number: int) -> Path:
    """Create a directory for a specific trial."""
    trial_dir = output_dir / f"trial_{trial_number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    return trial_dir


def run_trial(
    trial: optuna.Trial,
    args: argparse.Namespace,
    study_output_dir: Path,
) -> float:
    """
    Run a single Optuna trial.

    Args:
        trial: Optuna trial object
        args: Command-line arguments
        study_output_dir: Output directory for the study

    Returns:
        Validation loss to optimize
    """
    # Suggest hyperparameters
    hparams = suggest_hyperparameters(trial, args)

    # Create trial directory
    trial_dir = create_trial_directory(study_output_dir, trial.number)

    # Set up random seed for this trial
    trial_seed = args.seed + trial.number
    args.seed = trial_seed

    # Setup the run path
    run_path = trial_dir
    args.run_name = f"trial_{trial.number:03d}"
    args.output_dir = str(run_path.parent)

    # Create data module
    datamodule = EmbeddingDecoderDataModule(
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=trial_seed,
    )

    # Create Lightning module with suggested hyperparameters
    model = EmbeddingDecoderLightning(
        output_dim=768,
        emb_dim=args.emb_dim,
        num_layers=hparams["num_layers"],
        fwd_dim=hparams["fwd_dim"],
        num_heads=hparams["num_heads"],
        dropout=args.dropout,
        learning_rate=hparams["learning_rate"],
        weight_decay=args.weight_decay,
        loss_alpha=args.loss_alpha,
    )

    # Set hyperparameters on args for save_training_config
    args.num_layers = hparams["num_layers"]
    args.fwd_dim = hparams["fwd_dim"]
    args.num_heads = hparams["num_heads"]
    args.learning_rate = hparams["learning_rate"]

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_path / "checkpoints",
        filename="checkpoint-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )

    best_model_callback = BestModelSaveCallback(run_path / "best_model.pt")

    # Pruning callback
    pruning_callback = PyTorchLightningPruningCallback(
        trial=trial, monitor="val_loss"
    )

    callbacks = [checkpoint_callback, best_model_callback, pruning_callback]

    if args.early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.early_stopping_patience,
            verbose=True,
        )
        callbacks.append(early_stopping_callback)

    # Add progress bar callback with configurable refresh rate
    if not args.disable_progress_bar:
        progress_bar_callback = RichProgressBar(
            refresh_rate=args.progress_bar_refresh_rate
        )
        callbacks.append(progress_bar_callback)

    # Setup logger
    if not args.disable_wandb:
        import wandb

        wandb_run = wandb.init(
            project="embedding-decoder-optuna",
            name=f"trial_{trial.number:03d}",
            dir=str(run_path),
            reinit=True,
        )
        wandb_run.config.update(hparams)

        logger = L.pytorch.loggers.WandbLogger(experiment=wandb_run)
    else:
        logger = False

    # Create trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        min_epochs=args.min_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=not args.disable_progress_bar,
        enable_model_summary=not args.disable_progress_bar,
    )

    # Setup data and save config
    datamodule.setup()
    data_info = datamodule.get_dataset_info()
    save_training_config(run_path, args, data_info)

    # Save trial-specific hyperparameters
    trial_config = {
        "trial_number": trial.number,
        "hyperparameters": hparams,
        "fixed_params": {
            "emb_dim": args.emb_dim,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "loss_alpha": args.loss_alpha,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        },
        "data_info": data_info,
    }

    with open(run_path / "trial_config.json", "w") as f:
        json.dump(trial_config, f, indent=2)

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Get the best validation loss
    best_val_loss = trainer.callback_metrics.get("val_loss", float("inf")).item()

    # Report back to Optuna
    trial.report(best_val_loss, trainer.current_epoch)

    # Handle pruning based on the intermediate value
    if trial.should_prune():
        raise optuna.TrialPruned()

    return best_val_loss


def save_study_results(study: optuna.Study, output_dir: Path):
    """Save study results and create visualizations."""
    # Save best trial
    best_trial = study.best_trial
    best_trial_data = {
        "number": best_trial.number,
        "value": best_trial.value,
        "params": best_trial.params,
    }

    with open(output_dir / "best_trial.json", "w") as f:
        json.dump(best_trial_data, f, indent=2)

    # Save all trial results
    all_trials = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            all_trials.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "datetime": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            })

    with open(output_dir / "study_results.json", "w") as f:
        json.dump({
            "study_name": study.study_name,
            "n_trials": len(study.trials),
            "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "n_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
            "best_trial": best_trial_data,
            "all_trials": all_trials,
        }, f, indent=2)

    # Create visualizations
    try:
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_image(output_dir / "optimization_history.png")

        # Parameter importances
        fig = plot_param_importances(study)
        fig.write_image(output_dir / "parameter_importances.png")

        # Parallel coordinate plot
        fig = plot_parallel_coordinate(study)
        fig.write_image(output_dir / "parallel_coordinate.png")

        print(f"\nVisualizations saved to {output_dir}")
    except Exception as e:
        print(f"\nWarning: Could not create visualizations: {e}")
        print("Install plotly and kaleido for visualization support:")
        print("  pip install plotly kaleido")


def run_parallel_trials(
    study: optuna.Study,
    args: argparse.Namespace,
    study_output_dir: Path,
):
    """
    Run trials in parallel using Optuna's built-in parallelization.

    Args:
        study: Optuna study object
        args: Command-line arguments
        study_output_dir: Output directory for the study
    """
    # Use Optuna's built-in parallelization via study.optimize with n_jobs
    study.optimize(
        lambda trial: run_trial(trial, args, study_output_dir),
        n_trials=args.n_trials,
        timeout=args.timeout * 3600 if args.timeout else None,
        n_jobs=args.n_parallel,  # Use Optuna's built-in parallelization
        show_progress_bar=True,
    )


def main():
    args = parse_args()

    # Create study directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = args.study_name or f"study_{timestamp}"
    study_output_dir = Path(args.output_dir) / study_name
    study_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Optuna Hyperparameter Optimization")
    print(f"{'=' * 60}")
    print(f"Study name: {study_name}")
    print(f"Output directory: {study_output_dir.absolute()}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Parallel trials: {args.n_parallel}")
    print(f"{'=' * 60}\n")

    # Create sampler and pruner
    sampler = TPESampler(seed=args.seed, multivariate=True)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=args.pruning_patience)

    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # Set study direction
    print("Optimizing for minimum validation loss")

    # Run optimization (handles both sequential and parallel via n_jobs)
    run_parallel_trials(study, args, study_output_dir)

    # Print results
    print(f"\n{'=' * 60}")
    print("Optimization Complete!")
    print(f"{'=' * 60}")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

    if study.trials and any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best validation loss: {study.best_value:.4f}")
        print("\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

    print(f"\nOutput directory: {study_output_dir.absolute()}")
    print(f"Best trial config: {study_output_dir / 'best_trial.json'}")
    print(f"All results: {study_output_dir / 'study_results.json'}")
    print(f"{'=' * 60}\n")

    # Save study results and create visualizations
    save_study_results(study, study_output_dir)


if __name__ == "__main__":
    main()
