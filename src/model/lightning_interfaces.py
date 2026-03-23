import lightning as L
from pathlib import Path
import torch


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
        self.best_val_loss = float("inf")

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
