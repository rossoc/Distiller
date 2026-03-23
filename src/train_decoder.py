#!/usr/bin/env python3
"""
Training script for the Embedding Decoder.

This script trains the decoder to predict target embeddings from input embeddings.
The training uses pre-computed embeddings to avoid redundant computation.

Flow:
1. Load dataset (X = input texts, y = target texts)
2. Pre-compute embeddings for all X and y using gemma_encoder
3. Train decoder to predict y_embeddings from x_embeddings
4. Save the trained decoder model

At training time, FAISS is not needed - we directly compare predicted embeddings
to target embeddings using MSE or Cosine Similarity loss.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data.dataset import create_datasets, embedding_collate_fn
from model.decoder import Decoder


class EmbeddingLoss(nn.Module):
    """
    Loss function for embedding prediction.

    Combines MSE loss with cosine similarity loss for better embedding alignment.
    """

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Weight for MSE loss (1-alpha for cosine loss)
        """
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction='none')

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # MSE loss (per-element)
        mse_loss = self.mse(predicted, target)
        
        # Apply padding mask if provided
        if padding_mask is not None:
            # Mask shape: (batch, seq_len), expand to match loss shape
            mask = padding_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            mse_loss = mse_loss * (~mask)
        
        mse_loss = mse_loss.mean()

        # Cosine similarity loss
        pred_norm = torch.norm(predicted, p=2, dim=-1, keepdim=True)
        tgt_norm = torch.norm(target, p=2, dim=-1, keepdim=True)
        pred_normalized = predicted / (pred_norm + 1e-8)
        target_normalized = target / (tgt_norm + 1e-8)

        cosine_sim = torch.sum(pred_normalized * target_normalized, dim=-1)
        
        # Apply padding mask to cosine loss
        if padding_mask is not None:
            cosine_sim = cosine_sim * (~padding_mask)
            cosine_loss = 1 - cosine_sim.sum() / (~padding_mask).sum()
        else:
            cosine_loss = 1 - cosine_sim.mean()

        # Combined loss
        return self.alpha * mse_loss + (1 - self.alpha) * cosine_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Embedding Decoder")
    
    # Data arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/decoder",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for embeddings"
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
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log every N batches"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs"
    )
    
    return parser.parse_args()


def train_epoch(
    model: Decoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    criterion: nn.Module,
    epoch: int,
    device: str,
    log_interval: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_embeddings = batch["input_embeddings"].to(device)
        target_embeddings = batch["target_embeddings"].to(device)
        memory_padding_mask = batch["input_attention_mask"].to(device) == 0
        tgt_padding_mask = batch["target_attention_mask"].to(device) == 0

        # Forward pass
        optimizer.zero_grad()
        predicted_embeddings = model(
            memory=input_embeddings,
            tgt=target_embeddings,  # Teacher forcing
            memory_padding_mask=memory_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )

        # Compute loss (only on non-padded positions)
        loss = criterion(predicted_embeddings, target_embeddings, tgt_padding_mask)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Logging
        total_loss += loss.item()
        num_batches += 1

        if batch_idx % log_interval == 0:
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return total_loss / num_batches


def evaluate(
    model: Decoder,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_embeddings = batch["input_embeddings"].to(device)
            target_embeddings = batch["target_embeddings"].to(device)
            memory_padding_mask = batch["input_attention_mask"].to(device) == 0
            tgt_padding_mask = batch["target_attention_mask"].to(device) == 0

            # Forward pass
            predicted_embeddings = model(
                memory=input_embeddings,
                tgt=target_embeddings,
                memory_padding_mask=memory_padding_mask,
                tgt_padding_mask=tgt_padding_mask,
            )

            # Compute loss
            loss = criterion(predicted_embeddings, target_embeddings, tgt_padding_mask)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save args to config file
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Training configuration:")
    print(f"  Device: {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Model: emb_dim={args.emb_dim}, layers={args.num_layers}, heads={args.num_heads}")
    
    # Create datasets
    print("\n" + "="*50)
    print("Loading and preprocessing datasets...")
    print("="*50)
    
    train_dataset, eval_dataset, test_dataset = create_datasets(
        train_ratio=0.5,
        eval_ratio=0.1,
        test_ratio=0.4,
        max_length=args.max_length,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=embedding_collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=embedding_collate_fn,
    )
    
    # Create model
    print("\nCreating decoder model...")
    model = Decoder(
        output_dim=768,
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        fwd_dim=args.fwd_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model = model.to(args.device)
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.learning_rate * 0.1,
    )
    
    # Create loss function
    criterion = EmbeddingLoss(alpha=args.loss_alpha)
    
    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    best_eval_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            epoch=epoch,
            device=args.device,
            log_interval=args.log_interval,
        )
        
        print(f"\nTraining loss: {train_loss:.4f}")
        
        # Evaluate
        eval_loss = evaluate(
            model=model,
            dataloader=eval_loader,
            criterion=criterion,
            device=args.device,
        )
        print(f"Evaluation loss: {eval_loss:.4f}")
        
        # Save best model
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            checkpoint_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "eval_loss": eval_loss,
                "config": {
                    "emb_dim": args.emb_dim,
                    "num_layers": args.num_layers,
                    "fwd_dim": args.fwd_dim,
                    "num_heads": args.num_heads,
                    "dropout": args.dropout,
                    "output_dim": 768,
                }
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "eval_loss": eval_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best evaluation loss: {best_eval_loss:.4f}")
    print(f"Model saved to {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
