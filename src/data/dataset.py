"""
Dataset module for the Embedding Decoder training.

Handles pre-computing embeddings for training data and creating
PyTorch datasets for efficient training.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from data.util import load_dataset, Datasets_Variations
from model.encoder import gemma_encoder


class EmbeddingDecoderDataset(Dataset):
    """
    Dataset for training the embedding decoder.
    
    Each sample contains:
    - input_embeddings: Embeddings of the input text (memory) - shape (seq_len, 768)
    - target_embeddings: Embeddings of the target text - shape (seq_len, 768)
    - target_text: The target text string (for FAISS index building)
    """
    
    def __init__(
        self,
        X_texts: List[str],
        y_texts: List[str],
        encoder,
        max_length: int = 512,
    ):
        """
        Initialize the dataset.
        
        Args:
            X_texts: List of input text strings
            y_texts: List of target text strings
            encoder: Encoder model to convert text to embeddings
            max_length: Maximum sequence length for encoding
        """
        self.X_texts = X_texts
        self.y_texts = y_texts
        self.encoder = encoder
        self.max_length = max_length
        
        # Pre-compute all embeddings
        print("Pre-computing input embeddings...")
        self.input_embeddings = self._encode_texts(X_texts)
        
        print("Pre-computing target embeddings...")
        self.target_embeddings = self._encode_texts(y_texts)
        
        print(f"Dataset ready with {len(self)} samples")
        
    def _encode_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Encode a list of texts to sequences of embeddings."""
        embeddings = []
        # Use small batch size to avoid LlamaBatch overflow
        batch_size = 4
        
        for i, text in enumerate(texts):
            try:
                # Get embeddings for single text - returns list of token embeddings
                emb_list = self.encoder.embed([text])
                if emb_list and len(emb_list) > 0:
                    # emb_list[0] is the list of token embeddings
                    # Each token embedding is 768-dim
                    token_embs = np.array(emb_list[0], dtype=np.float32)
                    
                    # Handle shape: could be (num_tokens, 768) or (768,) for single token
                    if token_embs.ndim == 1:
                        token_embs = token_embs.reshape(1, -1)
                    
                    # Truncate to max_length if needed
                    if len(token_embs) > self.max_length:
                        token_embs = token_embs[:self.max_length]
                    
                    embeddings.append(token_embs)
                else:
                    # Fallback to zeros
                    embeddings.append(np.zeros((1, 768), dtype=np.float32))
            except Exception as e:
                # Fallback to zeros if encoding fails
                embeddings.append(np.zeros((1, 768), dtype=np.float32))
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Encoded {i + 1}/{len(texts)} texts...")
        
        return embeddings
    
    def __len__(self):
        return len(self.X_texts)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        return {
            "input_embeddings": torch.tensor(self.input_embeddings[idx], dtype=torch.float32),
            "target_embeddings": torch.tensor(self.target_embeddings[idx], dtype=torch.float32),
            "target_text": self.y_texts[idx],
            "input_text": self.X_texts[idx],
        }


def embedding_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching variable-length embedding sequences.
    
    Pads sequences to the maximum length in the batch.
    """
    batch_size = len(batch)
    
    # Get max sequence lengths
    max_input_len = max(sample["input_embeddings"].shape[0] for sample in batch)
    max_target_len = max(sample["target_embeddings"].shape[0] for sample in batch)
    
    # Get embedding dimension (assume 768)
    emb_dim = 768
    
    # Create padded tensors
    input_embeddings = torch.zeros(batch_size, max_input_len, emb_dim, dtype=torch.float32)
    target_embeddings = torch.zeros(batch_size, max_target_len, emb_dim, dtype=torch.float32)
    input_attention_mask = torch.zeros(batch_size, max_input_len, dtype=torch.long)
    target_attention_mask = torch.zeros(batch_size, max_target_len, dtype=torch.long)
    
    # Fill in the data
    for i, sample in enumerate(batch):
        inp_len = sample["input_embeddings"].shape[0]
        tgt_len = sample["target_embeddings"].shape[0]
        
        input_embeddings[i, :inp_len, :] = sample["input_embeddings"]
        input_attention_mask[i, :inp_len] = 1
        
        target_embeddings[i, :tgt_len, :] = sample["target_embeddings"]
        target_attention_mask[i, :tgt_len] = 1
    
    return {
        "input_embeddings": input_embeddings,
        "target_embeddings": target_embeddings,
        "input_attention_mask": input_attention_mask,
        "target_attention_mask": target_attention_mask,
        "target_text": [sample["target_text"] for sample in batch],
        "input_text": [sample["input_text"] for sample in batch],
    }


def create_datasets(
    train_ratio: float = 0.5,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.4,
    dataset_variation: Datasets_Variations = Datasets_Variations.SIMPLE_DIFFUSION,
    max_length: int = 512,
) -> Tuple[EmbeddingDecoderDataset, EmbeddingDecoderDataset, EmbeddingDecoderDataset]:
    """
    Create train, eval, and test datasets.
    
    Args:
        train_ratio: Ratio of data for training
        eval_ratio: Ratio of data for evaluation
        test_ratio: Ratio of data for testing
        dataset_variation: Which dataset variation to use
        max_length: Maximum sequence length
        
    Returns:
        train_dataset, eval_dataset, test_dataset
    """
    from src.data.util import set_seed
    set_seed(42)  # For reproducibility
    
    # Load and split data
    (X_train, y_train), (X_eval, y_eval), (X_test, y_test) = load_dataset(
        dataset_variation=dataset_variation,
        split_ratio=(train_ratio, eval_ratio, test_ratio)
    )
    
    # Convert to lists
    X_train_list = X_train.tolist()
    y_train_list = y_train.tolist()
    X_eval_list = X_eval.tolist()
    y_eval_list = y_eval.tolist()
    X_test_list = X_test.tolist()
    y_test_list = y_test.tolist()
    
    print(f"Loaded {len(X_train)} training samples")
    print(f"Loaded {len(X_eval)} evaluation samples")
    print(f"Loaded {len(X_test)} test samples")
    
    # Create encoder
    encoder = gemma_encoder()
    
    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = EmbeddingDecoderDataset(
        X_train_list, y_train_list, encoder, max_length
    )
    
    print("\nCreating evaluation dataset...")
    eval_dataset = EmbeddingDecoderDataset(
        X_eval_list, y_eval_list, encoder, max_length
    )
    
    print("\nCreating test dataset...")
    test_dataset = EmbeddingDecoderDataset(
        X_test_list, y_test_list, encoder, max_length
    )
    
    return train_dataset, eval_dataset, test_dataset


def build_faiss_index_from_dataset(
    dataset: EmbeddingDecoderDataset,
    save_path: str,
) -> 'FAISSRetriever':
    """
    Build a FAISS index from a dataset's target texts and embeddings.
    
    This is used at inference time to convert predicted embeddings back to text.
    
    Args:
        dataset: Dataset containing target texts and embeddings
        save_path: Path to save the FAISS index
        
    Returns:
        retriever: FAISSRetriever with built index
    """
    from model.faiss_retriever import build_retriever_from_texts
    
    # Get all target texts and embeddings
    texts = dataset.y_texts
    embeddings = np.array([d["target_embeddings"].numpy() for d in dataset])
    
    # Build and save retriever
    retriever = build_retriever_from_texts(
        texts=texts,
        embeddings=embeddings,
        save_path=save_path
    )
    
    return retriever
