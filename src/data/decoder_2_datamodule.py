"""
Lightning data module for the Decoder2 (Vocabulary-based Decoder) dataset.
"""

from typing import Optional
from torch.utils.data import DataLoader
import lightning as L

from model.encoder import gemma_encoder
from model.gemma_tokenizer import get_gemma_tokenizer
from data.decoder_2_dataset import (
    Decoder2Dataset,
    decoder2_collate_fn,
)

from util.randomness import set_seed


class Decoder2DataModule(L.LightningDataModule):
    """
    Lightning data module for the Decoder2 dataset.

    Handles dataset creation, pre-computation of embeddings and tokens, and data loading.
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
        gguf_model_path: str = "models/embeddinggemma-300M-Q8.gguf",
    ):
        """
        Initialize the data module.

        Args:
            train_ratio: Ratio of data for training
            eval_ratio: Ratio of data for evaluation
            test_ratio: Ratio of data for testing
            schema: Which dataset schema to use
            max_length: Maximum sequence length for embeddings/tokens
            batch_size: Batch size for data loading
            num_workers: Number of workers for data loading
            seed: Random seed for reproducibility
            gguf_model_path: Path to the GGUF model for tokenizer
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
        self.gguf_model_path = gguf_model_path

        # Datasets will be created in setup()
        self.encoder = None
        self.tokenizer = None
        self.train_data = None
        self.eval_data = None
        self.test_data = None
        self.train_dataset: Decoder2Dataset = None
        self.eval_dataset: Decoder2Dataset = None
        self.test_dataset: Decoder2Dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        Create datasets.

        This is called on each process for each GPU.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        set_seed(self.seed)

        # Load encoder and tokenizer
        self.encoder = gemma_encoder()
        self.tokenizer = get_gemma_tokenizer(self.gguf_model_path)

        # Load dataset
        from data.util import load_dataset
        
        self.train_data, self.eval_data, self.test_data = load_dataset(
            (self.train_ratio, self.eval_ratio, self.test_ratio), 
            "simple_diffusion"
        )

    def _build_dataloader(self, X, y, shuffle=True) -> DataLoader:
        dataset = Decoder2Dataset(
            X_texts=X,
            y_texts=y,
            encoder=self.encoder,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=decoder2_collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        X_train, y_train = self.train_data
        return self._build_dataloader(X_train, y_train)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        X_eval, y_eval = self.eval_data
        return self._build_dataloader(X_eval, y_eval, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        X_test, y_test = self.test_data
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
            "vocab_size": self.tokenizer.vocab_size if self.tokenizer else 256000,
        }
