"""
Lightning module for the Embedding Decoder.

Wraps the Decoder model with Lightning's training interface.
"""

import torch
import torch.nn as nn
import lightning as L
from torch.optim.lr_scheduler import OneCycleLR
from dataclasses import dataclass

from typing import Optional, Dict, Any
from model.decoder import Decoder


@dataclass
class TrainerHparams:
    output_dim: int = 768
    emb_dim: int = 768
    num_layers: int = 6
    fwd_dim: int = 2048
    num_heads: int = 8
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    loss_alpha: float = 0.5
    gradient_clip_val: float = 1.0
    use_warmup: bool = True
    warmup_epochs: int = 5


class DiffusionTrainer(L.LightningModule):
    """
    Lightning module for training the embedding decoder.
    """

    hparams: TrainerHparams

    def __init__(
        self,
        output_dim: int = 768,
        emb_dim: int = 768,
        num_layers: int = 6,
        fwd_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        loss_alpha: float = 0.5,
        gradient_clip_val: float = 1.0,
        use_warmup: bool = True,
        warmup_epochs: int = 5,
    ):
        """
        Initialize the Lightning module.

        Args:
            output_dim: Output embedding dimension (768 for EmbeddingGemma)
            emb_dim: Model hidden dimension
            num_layers: Number of transformer decoder layers
            fwd_dim: Feed-forward dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            loss_alpha: Weight for MSE loss in combined loss (1-alpha for cosine)
            label_smoothing: Label smoothing factor for regularization
            gradient_clip_val: Gradient clipping value (disabled if <= 0)
            use_warmup: Whether to use learning rate warmup
            warmup_epochs: Number of warmup epochs
        """
        super().__init__()
        self.save_hyperparameters()

        # Create the decoder model
        self.model = Decoder(
            output_dim=output_dim,
            emb_dim=emb_dim,
            num_layers=num_layers,
            fwd_dim=fwd_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Loss function
        self.criterion = DiffusionLoss(alpha=loss_alpha)

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
        target_embeddings = batch["target_embeddings"]
        memory_mask = batch["input_attention_mask"] == 0
        tgt_mask = batch["target_attention_mask"] == 0

        max_tgt_len = target_embeddings.shape[1]
        init_tgt = input_embeddings[:, :max_tgt_len, :].clone()

        # Forward pass
        predicted_embeddings = self.model(
            memory=input_embeddings,
            tgt=init_tgt,
            memory_mask=memory_mask,
            tgt_mask=tgt_mask,
        )

        # Compute loss
        loss = self.criterion(predicted_embeddings, target_embeddings, tgt_mask)

        # Compute additional metrics for monitoring
        with torch.no_grad():
            # Cosine similarity metric
            avg_cosine_sim = cosine_similarity(
                predicted_embeddings, target_embeddings, tgt_mask
            )

            # MSE metric
            mse = ((predicted_embeddings - target_embeddings) ** 2).mean()

        # Log metrics
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log("train_cosine_sim", avg_cosine_sim, on_epoch=True, logger=True)
        self.log("train_mse", mse, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_embeddings = batch["input_embeddings"]
        target_embeddings = batch["target_embeddings"]
        memory_padding_mask = batch["input_attention_mask"] == 0
        tgt_padding_mask = batch["target_attention_mask"] == 0

        # Forward pass
        predicted_embeddings = self(
            memory=input_embeddings,
            tgt=target_embeddings,
            memory_padding_mask=memory_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )

        # Compute loss
        val_loss = self.criterion(
            predicted_embeddings, target_embeddings, tgt_padding_mask
        )

        # Compute additional metrics
        with torch.no_grad():
            # Cosine similarity metric
            avg_cosine_sim = cosine_similarity(
                predicted_embeddings, target_embeddings, tgt_padding_mask
            )

        # Log metrics
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("val_cosine_sim", avg_cosine_sim, on_epoch=True, logger=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        input_embeddings = batch["input_embeddings"]
        target_embeddings = batch["target_embeddings"]
        memory_padding_mask = batch["input_attention_mask"] == 0
        tgt_padding_mask = batch["target_attention_mask"] == 0

        # Forward pass
        predicted_embeddings = self(
            memory=input_embeddings,
            tgt=target_embeddings,
            memory_padding_mask=memory_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )

        # Compute loss
        test_loss = self.criterion(
            predicted_embeddings, target_embeddings, tgt_padding_mask
        )

        # Compute cosine similarity metric
        with torch.no_grad():
            avg_cosine_sim = cosine_similarity(
                predicted_embeddings, target_embeddings, tgt_padding_mask
            )

        # Log metrics
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, logger=True)
        self.log(
            "test_cosine_sim", avg_cosine_sim, on_step=False, on_epoch=True, logger=True
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
        # Note: gradient clipping is handled by Trainer via trainer.clip_grad_val
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def get_model_config(self):
        """Get the model configuration as a dictionary."""
        return {
            "output_dim": self.hparams.output_dim,
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
    ) -> "DiffusionTrainer":
        """
        Load a DiffusionTrainer model from a checkpoint file.

        Handles multiple checkpoint formats:
        - Custom format (model_state_dict, model_config, hyperparameters)
        - Standard Lightning format (state_dict, hyper_parameters)

        Args:
            checkpoint_path: Path to the .ckpt or .pt model file
            device: Device to load the model on (default: "cpu")
            strict: Whether to strictly enforce that the keys match

        Returns:
            Loaded DiffusionTrainer model in eval mode

        Example:
            >>> model = DiffusionTrainer.from_checkpoint("outputs/best_model.pt", device="cuda")
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
                "output_dim": 768,
                "emb_dim": 768,
                "num_layers": 6,
                "fwd_dim": 2048,
                "num_heads": 8,
                "dropout": 0.1,
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "loss_alpha": 0.5,
                "gradient_clip_val": 1.0,
                "use_warmup": True,
                "warmup_epochs": 5,
            }
            print("Using default hyperparameters (not found in checkpoint)")

        # Create model from hyperparameters
        model = cls(
            output_dim=hparams.get("output_dim", 768),
            emb_dim=hparams.get("emb_dim", 768),
            num_layers=hparams.get("num_layers", 6),
            fwd_dim=hparams.get("fwd_dim", 2048),
            num_heads=hparams.get("num_heads", 8),
            dropout=hparams.get("dropout", 0.1),
            learning_rate=hparams.get("learning_rate", 1e-4),
            weight_decay=hparams.get("weight_decay", 0.01),
            loss_alpha=hparams.get("loss_alpha", 0.5),
            gradient_clip_val=hparams.get("gradient_clip_val", 1.0),
            use_warmup=hparams.get("use_warmup", True),
            warmup_epochs=hparams.get("warmup_epochs", 5),
        )

        # Load state dict - handle multiple formats
        if "model_state_dict" in checkpoint:
            # Custom format: model_state_dict contains Decoder weights with "model." prefix
            state_dict = checkpoint["model_state_dict"]
            # Strip "model." prefix to get plain Decoder weights
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

        return model


def cosine_similarity(
    pred: torch.Tensor, target: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute cosine similarity between predictions and targets.

    Args:
        pred: Predicted embeddings of shape (batch, seq_len, dim)
        target: Target embeddings of shape (batch, seq_len, dim)
        tgt_mask: Boolean padding mask of shape (batch, seq_len), where True = padding

    Returns:
        Average cosine similarity (scalar tensor), ignoring padding tokens if mask provided
    """
    pred_norm = torch.norm(pred, p=2, dim=-1)
    tgt_norm = torch.norm(target, p=2, dim=-1)
    cos_sim = (pred * target).sum(dim=-1) / (pred_norm * tgt_norm + 1e-8)

    if tgt_mask is not None:
        cos_sim = cos_sim * (~tgt_mask)
        avg_cos_sim = cos_sim.sum() / (~tgt_mask).sum()
    else:
        avg_cos_sim = cos_sim.mean()

    return avg_cos_sim


class DiffusionLoss(nn.Module):
    """
    Loss function for embedding prediction.

    Combines MSE loss with cosine similarity loss for better embedding alignment.
    Includes optional label smoothing for regularization.
    """

    def __init__(self, alpha: float = 0.5, label_smoothing: float = 0.0):
        """
        Args:
            alpha: Weight for MSE loss (1-alpha for cosine loss)
            label_smoothing: Label smoothing factor for MSE loss (0 = no smoothing)
        """
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mse_loss = self.mse(predicted, target)

        # Apply padding mask if provided
        if mask is not None:
            mse_loss = mse_loss * (~mask.unsqueeze(-1))
            num_valid_tokens = (~mask).sum() * predicted.shape[-1]
            mse_loss = mse_loss.sum() / num_valid_tokens.clamp(min=1)
        else:
            mse_loss = mse_loss.mean()

        cosine_loss = 1 - cosine_similarity(predicted, target, mask)

        return self.alpha * mse_loss + (1 - self.alpha) * cosine_loss
