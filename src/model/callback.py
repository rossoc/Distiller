import lightning as L
from pathlib import Path
import torch
from concurrent.futures import ThreadPoolExecutor


class BestModelSaveCallback(L.Callback):
    """
    Custom callback to save the best model weights at the end of training.

    Uses asynchronous disk writes to avoid blocking training.
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
        self._pending_save = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="model_save"
        )

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Called at the end of each validation epoch."""
        val_loss = trainer.callback_metrics.get("val_loss")

        if val_loss is not None and val_loss.item() < self.best_val_loss:
            self.best_val_loss = val_loss.item()

            # Save the best model (async disk write)
            self._save_model(pl_module)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Called at the end of training to ensure we have a saved model."""
        # Wait for any pending save to complete
        self._executor.shutdown(wait=True)

        if not self.save_path.exists():
            print("Saving final model (no validation improvements during training)...")
            self._save_model(pl_module, sync=True)

    def _save_model(self, pl_module: L.LightningModule, sync: bool = False):
        """
        Save the model weights and config.

        Args:
            pl_module: The Lightning module to save
            sync: If True, wait for disk write to complete
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

        # Write to disk asynchronously (non-blocking)
        def _write_checkpoint():
            torch.save(checkpoint, self.save_path)

        if sync:
            _write_checkpoint()
        else:
            self._executor.submit(_write_checkpoint)
