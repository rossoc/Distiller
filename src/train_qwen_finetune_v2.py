#!/usr/bin/env python3
"""
Training script for fine-tuning Qwen with Unsloth and SFTTrainer (v2).
"""
import argparse
import os
import torch
from src.data.qwen_unsloth_dataset_v2 import load_qwen_dataset_v2
from src.model.qwen_finetuner_v2 import QwenFinetunerV2

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with Unsloth v2")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3.5-0.8B", help="Model name")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--max_steps", type=int, default=100, help="Max training steps")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--wandb_project", type=str, default="qwen3.5-finetune", help="W&B project name")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()

    os.environ["WANDB_PROJECT"] = args.wandb_project

    # Load dataset
    dataset_dict = load_qwen_dataset_v2(args.data_path, validation_split=0)
    train_dataset = dataset_dict["train"]
    
    training_args = {
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": 5,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": args.seed,
        "output_dir": args.output_dir,
        "report_to": "wandb",
    }

    # Initialize and train model
    finetuner = QwenFinetunerV2(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        **training_args
    )
    
    trainer_stats = finetuner.train(train_dataset)
    
    print("Training finished.")
    print(trainer_stats)

if __name__ == "__main__":
    main()
