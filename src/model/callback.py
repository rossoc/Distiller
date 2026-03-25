import lightning as L
from pathlib import Path
import torch
import gc


class BestModelSaveCallback(L.Callback):
    """
    Custom callback to save the best model weights at the end of training.

    Saves synchronously to avoid memory accumulation from pending tasks.
    """

    def __init__(self, save_path: Path):
        """
        Initialize the callback.

        Args:
            save_path: Path to save the best model
        """
        super().__init__()
        self.save_path = save_path / "best_model.pt"
        self.best_val_loss = float("inf")

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Called at the end of each validation epoch."""
        val_loss = trainer.callback_metrics.get("val_loss")

        if val_loss is not None and val_loss.item() < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            # Save the best model synchronously
            self._save_model(pl_module)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Called at the end of training to ensure we have a saved model."""
        if not self.save_path.exists():
            print("Saving final model (no validation improvements during training)...")
            self._save_model(pl_module)

    def _save_model(self, pl_module: L.LightningModule):
        """
        Save the model weights and config synchronously.

        Args:
            pl_module: The Lightning module to save
        """
        # Ensure parent directory exists
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # Move state dict to CPU to avoid GPU memory accumulation
        state_dict = {k: v.cpu() for k, v in pl_module.state_dict().items()}

        checkpoint = {
            "model_state_dict": state_dict,
            "model_config": pl_module.get_model_config(),  # type: ignore
            "hyperparameters": dict(pl_module.hparams),
            "best_val_loss": self.best_val_loss,
        }

        # Write to disk synchronously
        torch.save(checkpoint, self.save_path)

        # Clean up CPU memory
        del state_dict, checkpoint
        gc.collect()
