"""
Lightning data module for the Embedding Decoder dataset.
"""

from typing import Optional
from torch.utils.data import DataLoader
import lightning as L

from .util import (
    load_dataset,
)

from model.encoder import gemma_encoder
from .dataset import (
    EmbeddingDecoderDataset,
    embedding_collate_fn,
)

from util.randomness import set_seed


class EmbeddingDecoderDataModule(L.LightningDataModule):
    """
    Lightning data module for the embedding decoder dataset.

    Handles dataset creation, pre-computation of embeddings, and data loading.
    """

    def __init__(
        self,
        train_ratio: float = 0.5,
        eval_ratio: float = 0.1,
        test_ratio: float = 0.4,
        schema: str = "simple_diffusion",
        max_length: int = 512,
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42,
    ):
        """
        Initialize the data module.

        Args:
            train_ratio: Ratio of data for training
            eval_ratio: Ratio of data for evaluation
            test_ratio: Ratio of data for testing
            dataset_variation: Which dataset variation to use
            max_length: Maximum sequence length for embeddings
            batch_size: Batch size for data loading
            num_workers: Number of workers for data loading
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.train_ratio = train_ratio
        self.eval_ratio = eval_ratio
        self.test_ratio = test_ratio
        self.data_schema = schema
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # Datasets will be created in setup()
        self.encoder = gemma_encoder()
        self.train_data = None
        self.eval_data = None
        self.test_data = None
        self.train_dataset: EmbeddingDecoderDataset = None
        self.eval_dataset: EmbeddingDecoderDataset = None
        self.test_dataset: EmbeddingDecoderDataset = None

        self._dataset_created = False

    def setup(self, stage: Optional[str] = None):
        """
        Create datasets.

        This is called on each process for each GPU.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        set_seed(self.seed)

        self.train_data, self.eval_data, self.test_data = load_dataset(
            (self.train_ratio, self.eval_ratio, self.test_ratio), "simple_diffusion"
        )

    def _build_dataloader(self, X, y, shuffle=True) -> DataLoader:
        dataset = EmbeddingDecoderDataset(X, y, self.encoder, self.max_length)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=embedding_collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        X_train, y_train = self.train_data  # type: ignore
        return self._build_dataloader(X_train, y_train)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        X_eval, y_eval = self.eval_data  # type: ignore
        return self._build_dataloader(X_eval, y_eval, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        X_test, y_test = self.test_data  # type: ignore
        return self._build_dataloader(X_test, y_test, shuffle=False)

    def get_dataset_info(self) -> dict:
        """Get information about the datasets."""
        return {
            "train_ratio": self.train_ratio,
            "eval_ratio": self.eval_ratio,
            "test_ratio": self.test_ratio,
            "data_schema": self.data_schema,
            "max_length": self.max_length,
            "train_samples": len(self.train_data[0]) if self.train_data else 0,
            "eval_samples": len(self.eval_data[0]) if self.eval_data else 0,
            "test_samples": len(self.test_data[0]) if self.test_data else 0,
        }
