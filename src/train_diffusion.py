#!/usr/bin/env python3
"""
Training script for the Embedding Decoder using PyTorch Lightning.

This script trains the decoder to predict target embeddings from input embeddings.
The best model (lowest validation loss) and training configuration are saved to
a timestamped folder in outputs/.

Usage:
    python src/train_lightning.py --epochs 50 --batch_size 32

Output:
    outputs/decoder_YYYYMMDD_HHMMSS/
    ├── best_model.pt       # Best model weights
    ├── training_config.json # Training configuration including seed
    └── lightning_logs/     # TensorBoard logs
"""

import argparse

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
from lightning.pytorch.loggers import WandbLogger

from model.diffusion_trainer import DiffusionTrainer
from model.callback import BestModelSaveCallback
from data.datamodule import EmbeddingDecoderDataModule
from util.logger import log_training_config, log_test_results
from util.randomness import setup_run
from util.verbose_output import (
    print_verbose_setup_diffusion,
    print_verbose_training_complete,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Embedding Decoder with Lightning"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Base directory for outputs (default: outputs)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this run (default: auto-generated timestamp)",
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

    # Model arguments
    parser.add_argument(
        "--emb_dim", type=int, default=768, help="Model embedding dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of transformer decoder layers"
    )
    parser.add_argument(
        "--fwd_dim", type=int, default=2048, help="Feed-forward dimension"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--loss_alpha",
        type=float,
        default=0.5,
        help="Weight for MSE loss in combined loss",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for regularization",
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="Gradient clipping value",
    )
    parser.add_argument(
        "--use_warmup",
        action="store_true",
        default=True,
        help="Use learning rate warmup",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, help="Number of warmup epochs"
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
        "--devices", type=int, default=1, help="Number of devices to use"
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
        "--progress_bar_refresh_rate",
        type=int,
        default=100,
        help="Update progress bar every N batches (default: 100)",
    )
    parser.add_argument(
        "--disable_progress_bar",
        action="store_true",
        help="Disable progress bar entirely",
    )
    parser.add_argument(
        "--save_last_checkpoint",
        action="store_true",
        help="Save last.ckpt every epoch (default: False for performance)",
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

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup the run
    run_path = setup_run(args)

    datamodule = EmbeddingDecoderDataModule(
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = DiffusionTrainer(
        output_dim=768,
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        fwd_dim=args.fwd_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_alpha=args.loss_alpha,
        gradient_clip_val=args.gradient_clip_val,
        use_warmup=args.use_warmup,
        warmup_epochs=args.warmup_epochs,
    )

    callbacks = [BestModelSaveCallback(run_path)]

    if args.early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.early_stopping_patience,
            verbose=False,
        )
        callbacks += [early_stopping_callback]

    # Add progress bar callback with configurable refresh rate
    if not args.disable_progress_bar:
        progress_bar_callback = RichProgressBar(
            refresh_rate=args.progress_bar_refresh_rate
        )
        callbacks += [progress_bar_callback]

    # Setup logger
    logger = WandbLogger(
        save_dir=str(run_path),
        name=args.run_name or "decoder",
        project="simple-diffusion",
    )

    # Create trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        min_epochs=args.min_epochs,
        min_steps=args.min_steps,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=not args.disable_progress_bar,
        enable_model_summary=True,
    )

    # Save training config (after data module setup to get dataset info)
    print("\nSetting up data...")
    datamodule.setup()
    data_info = datamodule.get_dataset_info()
    log_training_config(run_path, args, data_info)

    # Print training summary
    print_verbose_setup_diffusion(args)

    trainer.fit(model, datamodule=datamodule)

    test_results = trainer.test(model, datamodule=datamodule)

    log_test_results(test_results, run_path)

    print_verbose_training_complete(run_path)


if __name__ == "__main__":
    main()
