"""
Lightning module for the Decoder2 (Vocabulary-based Decoder).

Wraps the Decoder2 model with Lightning's training interface.
"""

import torch
import torch.nn as nn
import lightning as L
from torch.optim.lr_scheduler import OneCycleLR
from dataclasses import dataclass

from typing import Optional, Dict, Any
from model.decoder_2 import Decoder2


@dataclass
class Decoder2TrainerHparams:
    vocab_size: int = 256000
    emb_dim: int = 768
    num_layers: int = 6
    fwd_dim: int = 2048
    num_heads: int = 8
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    use_warmup: bool = True
    warmup_epochs: int = 5


class Decoder2Trainer(L.LightningModule):
    """
    Lightning module for training the vocabulary-based decoder (Decoder2).
    """

    hparams: Decoder2TrainerHparams

    def __init__(
        self,
        vocab_size: int = 256000,
        emb_dim: int = 768,
        num_layers: int = 6,
        fwd_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        gradient_clip_val: float = 1.0,
        use_warmup: bool = True,
        warmup_epochs: int = 5,
    ):
        """
        Initialize the Lightning module.

        Args:
            vocab_size: Vocabulary size (256000 for Gemma tokenizer)
            emb_dim: Model hidden dimension
            num_layers: Number of transformer decoder layers
            fwd_dim: Feed-forward dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            gradient_clip_val: Gradient clipping value (disabled if <= 0)
            use_warmup: Whether to use learning rate warmup
            warmup_epochs: Number of warmup epochs
        """
        super().__init__()
        self.save_hyperparameters()

        # Create the decoder model
        self.model = Decoder2(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            num_layers=num_layers,
            fwd_dim=fwd_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # For tracking best validation loss
        self.best_val_loss = float("inf")

    def forward(
        self, memory, tgt=None, memory_padding_mask=None, tgt_padding_mask=None
    ):
        """Forward pass through the decoder."""
        return self.model(
            memory=memory,
            tgt=tgt,
            memory_mask=memory_padding_mask,
            tgt_mask=tgt_padding_mask,
        )

    def training_step(self, batch, batch_idx):
        """Training step."""
        input_embeddings = batch["input_embeddings"]
        target_tokens = batch["target_tokens"]
        memory_mask = batch["input_attention_mask"] == 0
        tgt_mask = batch["target_attention_mask"] == 0

        # Forward pass
        # output shape: (batch, seq_len, vocab_size)
        logits = self.model(
            memory=input_embeddings,
            tgt=target_tokens,
            memory_mask=memory_mask,
            tgt_mask=tgt_mask,
        )

        # Reshape for cross-entropy: (batch * seq_len, vocab_size)
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        target_flat = target_tokens.view(-1)

        # Compute loss
        loss = self.criterion(logits_flat, target_flat)

        # Compute accuracy for monitoring
        with torch.no_grad():
            predictions = logits_flat.argmax(dim=-1)
            # Ignore padding tokens in accuracy calculation
            non_padding_mask = target_flat != -100
            correct = (predictions == target_flat) & non_padding_mask
            accuracy = correct.sum().float() / non_padding_mask.sum().clamp(min=1)

        # Log metrics
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("train_accuracy", accuracy, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_embeddings = batch["input_embeddings"]
        target_tokens = batch["target_tokens"]
        memory_padding_mask = batch["input_attention_mask"] == 0
        tgt_padding_mask = batch["target_attention_mask"] == 0

        # Forward pass
        logits = self(
            memory=input_embeddings,
            tgt=target_tokens,
            memory_padding_mask=memory_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )

        # Reshape for cross-entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        target_flat = target_tokens.view(-1)

        # Compute loss
        val_loss = self.criterion(logits_flat, target_flat)

        # Compute accuracy
        with torch.no_grad():
            predictions = logits_flat.argmax(dim=-1)
            non_padding_mask = target_flat != -100
            correct = (predictions == target_flat) & non_padding_mask
            accuracy = correct.sum().float() / non_padding_mask.sum().clamp(min=1)

        # Log metrics
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("val_accuracy", accuracy, on_epoch=True, logger=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        input_embeddings = batch["input_embeddings"]
        target_tokens = batch["target_tokens"]
        memory_padding_mask = batch["input_attention_mask"] == 0
        tgt_padding_mask = batch["target_attention_mask"] == 0

        # Forward pass
        logits = self(
            memory=input_embeddings,
            tgt=target_tokens,
            memory_padding_mask=memory_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )

        # Reshape for cross-entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        target_flat = target_tokens.view(-1)

        # Compute loss
        test_loss = self.criterion(logits_flat, target_flat)

        # Compute accuracy
        with torch.no_grad():
            predictions = logits_flat.argmax(dim=-1)
            non_padding_mask = target_flat != -100
            correct = (predictions == target_flat) & non_padding_mask
            accuracy = correct.sum().float() / non_padding_mask.sum().clamp(min=1)

        # Log metrics
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, logger=True)
        self.log(
            "test_accuracy", accuracy, on_step=False, on_epoch=True, logger=True
        )

        return test_loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler with warmup."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # Cosine annealing scheduler with warmup
        if self.hparams.use_warmup:
            total_steps = self.trainer.estimated_stepping_batches
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=int(total_steps),
                pct_start=self.hparams.warmup_epochs
                / max(self.trainer.max_epochs or 0, 1),
                anneal_strategy="cos",
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs or 0,
                eta_min=self.hparams.learning_rate * 0.1,
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }

        # Configure optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def get_model_config(self):
        """Get the model configuration as a dictionary."""
        return {
            "vocab_size": self.hparams.vocab_size,
            "emb_dim": self.hparams.emb_dim,
            "num_layers": self.hparams.num_layers,
            "fwd_dim": self.hparams.fwd_dim,
            "num_heads": self.hparams.num_heads,
            "dropout": self.hparams.dropout,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
        strict: bool = True,
    ) -> "Decoder2Trainer":
        """
        Load a Decoder2Trainer model from a checkpoint file.

        Handles multiple checkpoint formats:
        - Custom format (model_state_dict, model_config, hyperparameters)
        - Standard Lightning format (state_dict, hyper_parameters)

        Args:
            checkpoint_path: Path to the .ckpt or .pt model file
            device: Device to load the model on (default: "cpu")
            strict: Whether to strictly enforce that the keys match

        Returns:
            Loaded Decoder2Trainer model in eval mode

        Example:
            >>> model = Decoder2Trainer.from_checkpoint("outputs/best_model.pt", device="cuda")
        """
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # Extract hyperparameters - try multiple formats
        if "hyperparameters" in checkpoint:
            hparams = checkpoint["hyperparameters"]
            print("Loaded hyperparameters (custom format)")
        elif "hyper_parameters" in checkpoint:
            hparams = checkpoint["hyper_parameters"]
            print("Loaded hyperparameters (Lightning format)")
        elif "model_config" in checkpoint:
            hparams = checkpoint.get("model_config", {})
            print("Loaded model_config (fallback format)")
        else:
            # Fallback to default config
            hparams = {
                "vocab_size": 256000,
                "emb_dim": 768,
                "num_layers": 6,
                "fwd_dim": 2048,
                "num_heads": 8,
                "dropout": 0.1,
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "gradient_clip_val": 1.0,
                "use_warmup": True,
                "warmup_epochs": 5,
            }
            print("Using default hyperparameters (not found in checkpoint)")

        # Create model from hyperparameters
        model = cls(
            vocab_size=hparams.get("vocab_size", 256000),
            emb_dim=hparams.get("emb_dim", 768),
            num_layers=hparams.get("num_layers", 6),
            fwd_dim=hparams.get("fwd_dim", 2048),
            num_heads=hparams.get("num_heads", 8),
            dropout=hparams.get("dropout", 0.1),
            learning_rate=hparams.get("learning_rate", 1e-4),
            weight_decay=hparams.get("weight_decay", 0.01),
            gradient_clip_val=hparams.get("gradient_clip_val", 1.0),
            use_warmup=hparams.get("use_warmup", True),
            warmup_epochs=hparams.get("warmup_epochs", 5),
        )

        # Load state dict - handle multiple formats
        if "model_state_dict" in checkpoint:
            # Custom format: model_state_dict contains Decoder2 weights with "model." prefix
            state_dict = checkpoint["model_state_dict"]
            # Strip "model." prefix to get plain Decoder2 weights
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            print("Loaded state dict (custom format, stripped 'model.' prefix)")
        elif "state_dict" in checkpoint:
            # Standard Lightning format
            state_dict = checkpoint["state_dict"]
            # Strip "model." prefix if present
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            print("Loaded state dict (Lightning format)")
        else:
            # Fallback: assume checkpoint is the state dict itself
            state_dict = checkpoint
            print("Loaded state dict (raw format)")

        model.model.load_state_dict(state_dict, strict=strict)
        model.to(device)
        model.eval()

        print("Model loaded successfully")
        print(f"  - Layers: {hparams.get('num_layers', 6)}")
        print(f"  - Embedding dim: {hparams.get('emb_dim', 768)}")
        print(f"  - Feed-forward dim: {hparams.get('fwd_dim', 2048)}")
        print(f"  - Heads: {hparams.get('num_heads', 8)}")
        print(f"  - Vocab size: {hparams.get('vocab_size', 256000)}")

        return model
