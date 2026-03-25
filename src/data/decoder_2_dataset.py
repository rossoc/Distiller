"""
Dataset module for the Decoder2 (Vocabulary-based Decoder) training.

Handles pre-computing embeddings for input texts and tokenizing target texts
for training the vocabulary-based decoder.
"""

from tqdm.auto import trange
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
import numpy as np


class Decoder2Dataset(Dataset):
    """
    Dataset for training the vocabulary-based decoder (Decoder2).

    Each sample contains:
    - input_embeddings: Embeddings of the input text (memory) - shape (seq_len, 768)
    - target_tokens: Token IDs of the target text - shape (seq_len,)
    - target_text: The target text string (for reference)
    - input_text: The input text string (for reference)
    """

    def __init__(
        self,
        X_texts: List[str],
        y_texts: List[str],
        encoder,
        tokenizer,
        max_length: int = 2048,
        batch_size: int = 64,
    ):
        """
        Initialize the dataset.

        Args:
            X_texts: List of input text strings
            y_texts: List of target text strings
            encoder: Encoder model to convert input text to embeddings
            tokenizer: Tokenizer to convert target text to token IDs
            max_length: Maximum sequence length for encoding/tokenization
        """
        self.X_texts = X_texts
        self.y_texts = y_texts
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size

        # Pre-compute input embeddings
        self.input_embeddings = self._encode_texts(X_texts, "Input")
        
        # Pre-compute target tokens
        self.target_tokens = self._tokenize_texts(y_texts, "Target")
        
        print(f"Dataset ready with {len(self)} samples")

    def _encode_texts(self, texts: List[str], is_input: str) -> list[np.ndarray]:
        """Encode texts in batches for better performance."""
        all_embeddings = []
        print(f"Processing {is_input} Embeddings")

        # Process in chunks of batch_size
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            try:
                # Batch Encode: Pass the whole list to the encoder
                batch_embs = self.encoder.embed(batch)

                for emb_list in batch_embs:
                    if emb_list and len(emb_list) > 0:
                        token_embs = np.array(emb_list, dtype=np.float32)

                        # Ensure 2D (sequence_length, 768)
                        if token_embs.ndim == 1:
                            token_embs = token_embs.reshape(1, -1)

                        # Truncate
                        if len(token_embs) > self.max_length:
                            token_embs = token_embs[: self.max_length]

                        all_embeddings.append(token_embs)
                    else:
                        all_embeddings.append(np.zeros((1, 768), dtype=np.float32))

            except Exception as e:
                # If a whole batch fails, append zeros for each item
                for _ in range(len(batch)):
                    all_embeddings.append(np.zeros((1, 768), dtype=np.float32))

        return all_embeddings

    def _tokenize_texts(self, texts: List[str], desc: str) -> List[np.ndarray]:
        """Tokenize texts in batches for better performance."""
        all_tokens = []
        print(f"Processing {desc} Tokens")

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            try:
                # Tokenize batch
                for text in batch:
                    # Tokenize text
                    tokens = self.tokenizer.encode(text)
                    
                    # Convert to numpy array
                    token_array = np.array(tokens, dtype=np.int64)
                    
                    # Truncate if needed
                    if len(token_array) > self.max_length:
                        token_array = token_array[: self.max_length]
                    
                    all_tokens.append(token_array)
                    
            except Exception as e:
                # If tokenization fails, append empty array
                for _ in range(len(batch)):
                    all_tokens.append(np.array([], dtype=np.int64))

        return all_tokens

    def __len__(self):
        return len(self.X_texts)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {
            "input_embeddings": torch.tensor(
                self.input_embeddings[idx], dtype=torch.float32
            ),
            "target_tokens": torch.tensor(
                self.target_tokens[idx], dtype=torch.long
            ),
            "target_text": self.y_texts[idx],
            "input_text": self.X_texts[idx],
        }


def decoder2_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching variable-length sequences.

    Pads input embeddings and target tokens to the maximum length in the batch.
    """
    batch_size = len(batch)

    # Get max sequence lengths
    max_input_len = max(sample["input_embeddings"].shape[0] for sample in batch)
    max_target_len = max(sample["target_tokens"].shape[0] for sample in batch)

    # Get embedding dimension (assume 768)
    emb_dim = 768

    # Create padded tensors
    input_embeddings = torch.zeros(
        batch_size, max_input_len, emb_dim, dtype=torch.float32
    )
    # Use -100 for padding (ignored by CrossEntropyLoss)
    target_tokens = torch.full(
        (batch_size, max_target_len), -100, dtype=torch.long
    )
    input_attention_mask = torch.zeros(batch_size, max_input_len, dtype=torch.long)
    target_attention_mask = torch.zeros(batch_size, max_target_len, dtype=torch.long)

    # Fill in the data
    for i, sample in enumerate(batch):
        inp_len = sample["input_embeddings"].shape[0]
        tgt_len = sample["target_tokens"].shape[0]

        input_embeddings[i, :inp_len, :] = sample["input_embeddings"]
        input_attention_mask[i, :inp_len] = 1

        target_tokens[i, :tgt_len] = sample["target_tokens"]
        target_attention_mask[i, :tgt_len] = 1

    return {
        "input_embeddings": input_embeddings,
        "target_tokens": target_tokens,
        "input_attention_mask": input_attention_mask,
        "target_attention_mask": target_attention_mask,
        "target_text": [sample["target_text"] for sample in batch],
        "input_text": [sample["input_text"] for sample in batch],
    }
