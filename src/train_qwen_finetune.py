#!/usr/bin/env python3
"""
Training script for fine-tuning LLMs (Qwen, Llama, etc.) using LoRA/QLoRA.

This script trains language models on custom downstream tasks using parameter-efficient
fine-tuning methods. The best model and training configuration are saved to a
timestamped folder in outputs/.

Usage:
    # Basic QLoRA fine-tuning (recommended for most cases)
    python src/train_qwen_finetune.py --data_path data/instructions.json

    # Full fine-tuning (more resources required)
    python src/train_qwen_finetune.py --use_lora False --use_quantization False

    # Custom LoRA configuration
    python src/train_qwen_finetune.py --lora_rank 16 --lora_alpha 32 --learning_rate 1e-4

Output:
    outputs/finetuned_YYYYMMDD_HHMMSS/
    ├── adapter_model/      # LoRA adapter weights (if using LoRA)
    ├── tokenizer/          # Tokenizer files
    ├── finetune_config.json # Training configuration
    └── lightning_logs/     # Training logs
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from model.qwen_finetuner import (
    QwenFineTuner,
    FineTuningConfig,
    LoRAConfig,
    QuantizationConfig,
    create_finetuning_config,
)
from data.finetune_datamodule import FinetuneDataModule
from util.randomness import setup_run


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune LLMs (Qwen, Llama) with LoRA/QLoRA"
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
        "--data_path",
        type=str,
        required=True,
        help="Path to training data (JSON, JSONL, CSV, or TXT)",
    )
    parser.add_argument(
        "--format_type",
        type=str,
        default="instruction",
        choices=["instruction", "chat", "text"],
        help="Format of the training data",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Qwen3.5-0.8B-Q8_0",
        help="HuggingFace model name to fine-tune",
    )
    parser.add_argument(
        "--use_lora",
        type=bool,
        default=True,
        help="Use LoRA adapters for parameter-efficient fine-tuning",
    )
    parser.add_argument(
        "--use_quantization",
        type=bool,
        default=True,
        help="Use 4-bit quantization (QLoRA) for memory efficiency",
    )

    # LoRA arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank (higher = more parameters, better quality)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha scaling factor",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout rate",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=None,
        help="Target modules for LoRA (auto-detected if not specified)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for learning rate scheduler",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "no"],
        help="Mixed precision mode",
    )

    # Logging arguments
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Evaluate every N steps",
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
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use",
    )

    # WandB arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Use Weights & Biases for logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="distiller-finetuning",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB entity (username/team)",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Data loading
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for data loading",
    )

    return parser.parse_args()


def create_run_name(args) -> str:
    """Create a run name based on configuration."""
    if args.run_name:
        return args.run_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model_name.split("/")[-1]
    lora_tag = f"lora_r{args.lora_rank}" if args.use_lora else "full"
    quant_tag = "qlora" if args.use_quantization and args.use_lora else "fp16"

    return f"finetuned_{model_short}_{lora_tag}_{quant_tag}_{timestamp}"


def main():
    args = parse_args()

    # Setup reproducibility
    setup_run(seed=args.seed)

    # Create output directory
    run_name = create_run_name(args)
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")

    # Create fine-tuning configuration
    config = create_finetuning_config(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        use_quantization=args.use_quantization,
        # Override with any custom args
        use_lora=args.use_lora,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_checkpointing=args.gradient_checkpointing,
        mixed_precision=args.mixed_precision,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        output_dir=str(output_dir),
        run_name=run_name,
    )

    # Print configuration
    print("=" * 60)
    print("Fine-tuning Configuration")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"LoRA: {config.use_lora} (r={config.lora.r}, alpha={config.lora.lora_alpha})")
    print(f"Quantization: {config.use_quantization}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Max length: {config.max_length}")
    print(f"Mixed precision: {config.mixed_precision}")
    print(f"Output dir: {output_dir}")
    print("=" * 60 + "\n")

    # Initialize data module
    datamodule = FinetuneDataModule(
        data_path=args.data_path,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        format_type=args.format_type,
        validation_split=args.validation_split,
    )

    # Initialize model
    model = QwenFineTuner(config=config)
    model.setup_model()

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="checkpoint-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=2,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    progress_bar = RichProgressBar()

    # Setup logger
    loggers = []
    if args.use_wandb:
        wandb_logger = WandbLogger(
            name=run_name,
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "model_name": config.model_name,
                "lora_rank": config.lora.r,
                "lora_alpha": config.lora.lora_alpha,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "epochs": config.num_epochs,
                "use_quantization": config.use_quantization,
            },
        )
        loggers.append(wandb_logger)

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=config.num_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.mixed_precision if args.mixed_precision != "no" else "32-true",
        gradient_clip_val=1.0,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        callbacks=[checkpoint_callback, lr_monitor, progress_bar],
        loggers=loggers,
        default_root_dir=output_dir,
        val_check_interval=args.eval_steps if args.eval_steps > 0 else 1.0,
    )

    # Train
    print("\nStarting training...\n")
    trainer.fit(model, datamodule=datamodule)

    # Save final model
    print("\nSaving final model...")
    model.save_model(output_dir / "adapter_model")

    # Save training configuration
    training_config = {
        "model_name": config.model_name,
        "use_lora": config.use_lora,
        "use_quantization": config.use_quantization,
        "lora_config": {
            "r": config.lora.r,
            "lora_alpha": config.lora.lora_alpha,
            "lora_dropout": config.lora.lora_dropout,
            "target_modules": config.lora.target_modules,
        },
        "training_config": {
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_length": config.max_length,
            "warmup_ratio": config.warmup_ratio,
            "weight_decay": config.weight_decay,
            "mixed_precision": config.mixed_precision,
        },
        "data_config": {
            "data_path": args.data_path,
            "format_type": args.format_type,
            "validation_split": args.validation_split,
        },
        "run_name": run_name,
        "seed": args.seed,
    }

    with open(output_dir / "finetune_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Model saved to: {output_dir}")
    print(f"Configuration saved to: {output_dir / 'finetune_config.json'}")
    print(f"{'='*60}\n")

    # Print final metrics
    if trainer.callback_metrics:
        print("Final metrics:")
        for key, value in trainer.callback_metrics.items():
            print(f"  {key}: {value.item() if hasattr(value, 'item') else value}")


if __name__ == "__main__":
    main()
