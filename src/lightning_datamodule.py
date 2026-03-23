"""
Lightning data module for the Embedding Decoder dataset.
"""

import torch
from torch.utils.data import DataLoader
import lightning as L

from data.dataset import (
    create_datasets,
    EmbeddingDecoderDataset,
    embedding_collate_fn,
)
from data.util import Datasets_Variations, set_seed


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
        dataset_variation: Datasets_Variations = Datasets_Variations.SIMPLE_DIFFUSION,
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
        self.dataset_variation = dataset_variation
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        # Datasets will be created in setup()
        self.train_dataset: EmbeddingDecoderDataset = None
        self.eval_dataset: EmbeddingDecoderDataset = None
        self.test_dataset: EmbeddingDecoderDataset = None
        
    def setup(self, stage: str = None):
        """
        Create datasets.
        
        This is called on each process for each GPU.
        
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        # Set seed for reproducibility
        set_seed(self.seed)
        
        # Create datasets
        self.train_dataset, self.eval_dataset, self.test_dataset = create_datasets(
            train_ratio=self.train_ratio,
            eval_ratio=self.eval_ratio,
            test_ratio=self.test_ratio,
            dataset_variation=self.dataset_variation,
            max_length=self.max_length,
        )
        
        print(f"DataModule setup complete:")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Eval samples: {len(self.eval_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")
        
    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=embedding_collate_fn,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=embedding_collate_fn,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=embedding_collate_fn,
            pin_memory=True,
        )
    
    def get_dataset_info(self) -> dict:
        """Get information about the datasets."""
        return {
            "train_ratio": self.train_ratio,
            "eval_ratio": self.eval_ratio,
            "test_ratio": self.test_ratio,
            "dataset_variation": self.dataset_variation.name,
            "max_length": self.max_length,
            "train_samples": len(self.train_dataset) if self.train_dataset else 0,
            "eval_samples": len(self.eval_dataset) if self.eval_dataset else 0,
            "test_samples": len(self.test_dataset) if self.test_dataset else 0,
        }
