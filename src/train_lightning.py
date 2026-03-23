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
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from lightning_module import EmbeddingDecoderLightning
from lightning_datamodule import EmbeddingDecoderDataModule, Datasets_Variations


def parse_args():
    parser = argparse.ArgumentParser(description="Train Embedding Decoder with Lightning")
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Base directory for outputs (default: outputs)"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for this run (default: auto-generated timestamp)"
    )
    
    # Data arguments
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for embeddings"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.5,
        help="Ratio of data for training"
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.1,
        help="Ratio of data for evaluation"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.4,
        help="Ratio of data for testing"
    )
    
    # Model arguments
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=768,
        help="Model embedding dimension"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of transformer decoder layers"
    )
    parser.add_argument(
        "--fwd_dim",
        type=int,
        default=2048,
        help="Feed-forward dimension"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    parser.add_argument(
        "--loss_alpha",
        type=float,
        default=0.5,
        help="Weight for MSE loss in combined loss"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Device arguments
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "cuda", "mps"],
        help="Accelerator to use"
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        choices=["64", "32-true", "32", "16-true", "16-mixed", "bf16-true", "bf16-mixed"],
        help="Precision for training"
    )
    
    # Callback arguments
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Patience for early stopping"
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=1,
        help="Minimum number of epochs to train"
    )
    parser.add_argument(
        "--min_steps",
        type=int,
        default=None,
        help="Minimum number of steps to train"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="Log every N steps"
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=1.0,
        help="Validation check interval (1.0 = every epoch)"
    )
    
    return parser.parse_args()


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


def save_training_config(run_path: Path, args: argparse.Namespace, data_info: dict):
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
            "loss_alpha": args.loss_alpha,
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


class BestModelSaveCallback(L.Callback):
    """
    Custom callback to save the best model weights at the end of training.
    """
    
    def __init__(self, save_path: Path):
        """
        Initialize the callback.
        
        Args:
            save_path: Path to save the best model
        """
        super().__init__()
        self.save_path = save_path
        self.best_val_loss = float('inf')
        
    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Called at the end of each validation epoch."""
        val_loss = trainer.callback_metrics.get("val_loss")
        
        if val_loss is not None and val_loss.item() < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            
            # Save the best model
            self._save_model(pl_module)
            print(f"New best model saved (val_loss={val_loss.item():.4f})")
    
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Called at the end of training to ensure we have a saved model."""
        if not self.save_path.exists():
            print("Saving final model (no validation improvements during training)...")
            self._save_model(pl_module)
    
    def _save_model(self, pl_module: L.LightningModule):
        """Save the model weights and config."""
        checkpoint = {
            "model_state_dict": pl_module.state_dict(),
            "model_config": pl_module.get_model_config(),
            "hyperparameters": dict(pl_module.hparams),
            "best_val_loss": self.best_val_loss,
        }
        
        torch.save(checkpoint, self.save_path)


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    L.seed_everything(args.seed)
    
    # Create output folder
    run_path = create_output_folder(args.output_dir, args.run_name)
    print(f"\n{'='*60}")
    print(f"Training run: {run_path.name}")
    print(f"Output folder: {run_path.absolute()}")
    print(f"{'='*60}\n")
    
    # Create data module
    print("Initializing data module...")
    datamodule = EmbeddingDecoderDataModule(
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=0,  # Set to >0 for faster data loading on Linux
        seed=args.seed,
    )
    
    # Create Lightning module
    print("Creating model...")
    model = EmbeddingDecoderLightning(
        output_dim=768,
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        fwd_dim=args.fwd_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_alpha=args.loss_alpha,
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_path / "checkpoints",
        filename="checkpoint-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    
    best_model_callback = BestModelSaveCallback(run_path / "best_model.pt")
    
    callbacks = [checkpoint_callback, best_model_callback]
    
    if args.early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.early_stopping_patience,
            verbose=True,
        )
        callbacks.append(early_stopping_callback)
    
    # Setup logger (use CSVLogger as default, TensorBoard if available)
    try:
        logger = TensorBoardLogger(
            save_dir=str(run_path),
            name="lightning_logs",
        )
        print("Using TensorBoard logger")
    except ModuleNotFoundError:
        logger = CSVLogger(
            save_dir=str(run_path),
            name="csv_logs",
        )
        print("TensorBoard not available, using CSV logger")
    
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
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Save training config (after data module setup to get dataset info)
    print("\nSetting up data...")
    datamodule.setup()
    data_info = datamodule.get_dataset_info()
    save_training_config(run_path, args, data_info)
    
    # Print training summary
    print(f"\n{'='*60}")
    print("Training Configuration:")
    print(f"{'='*60}")
    print(f"  Model: {args.emb_dim}d emb, {args.num_layers} layers, {args.num_heads} heads")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Seed: {args.seed}")
    print(f"  Train samples: {data_info['train_samples']}")
    print(f"  Eval samples: {data_info['eval_samples']}")
    print(f"  Test samples: {data_info['test_samples']}")
    print(f"{'='*60}\n")
    
    # Train
    print("Starting training...")
    trainer.fit(model, datamodule=datamodule)
    
    # Test on test set
    print("\nRunning final evaluation on test set...")
    test_results = trainer.test(model, datamodule=datamodule)
    
    # Save test results
    test_results_path = run_path / "test_results.json"
    with open(test_results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Output folder: {run_path.absolute()}")
    print(f"Best model: {run_path / 'best_model.pt'}")
    print(f"Config: {run_path / 'training_config.json'}")
    print(f"Logs: {run_path / 'lightning_logs'}")
    print(f"Checkpoints: {run_path / 'checkpoints'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
