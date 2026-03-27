"""
Dataset module for the Embedding Decoder training.

Handles pre-computing embeddings for training data and creating
PyTorch datasets for efficient training.
"""

from tqdm.auto import trange
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
import numpy as np
from model.faiss_retriever import FAISSRetriever


class EmbeddingDecoderDataset(Dataset):
    """
    Dataset for training the embedding decoder.

    Each sample contains:
    - input_embeddings: Embeddings of the input text (memory) - shape (seq_len, emb_dim)
    - target_embeddings: Embeddings of the target text - shape (seq_len, emb_dim)
    - target_text: The target text string (for FAISS index building)
    """

    def __init__(
        self,
        X_texts: List[str],
        y_texts: List[str],
        encoder,
        max_length: int = 2048,
        batch_size: int = 64,
        emb_dim: int = 768,
    ):
        """
        Initialize the dataset.

        Args:
            X_texts: List of input text strings
            y_texts: List of target text strings
            encoder: Encoder model to convert text to embeddings
            max_length: Maximum sequence length for encoding
            batch_size: Batch size for encoding
            emb_dim: Embedding dimension (768 for gemma, 1024 for qwen)
        """
        self.X_texts = X_texts
        self.y_texts = y_texts
        self.encoder = encoder
        self.max_length = max_length
        self.batch_size = batch_size
        self.emb_dim = emb_dim

        # Pre-compute all embeddings
        self.input_embeddings = self._encode_texts(X_texts, "Input")
        self.target_embeddings = self._encode_texts(y_texts, "Target")
        print(f"Dataset ready with {len(self)} samples")

    def _encode_texts(self, texts: List[str], is_input: str) -> list[np.ndarray]:
        """Encode texts in batches for better performance."""
        all_embeddings = []
        print(f"Processing {is_input} Embeddings")

        # Use smaller batch size to avoid KV cache exhaustion
        # Especially important for larger models like Qwen
        encode_batch_size = min(self.batch_size, 16)

        # Process in chunks of batch_size
        for i in range(0, len(texts), encode_batch_size):
            batch = texts[i : i + encode_batch_size]

            try:
                # 1. Batch Encode: Pass the whole list to the encoder
                # This returns a list of lists (one list of token embs per text)
                batch_embs = self.encoder.embed(batch)

                for emb_list in batch_embs:
                    if emb_list and len(emb_list) > 0:
                        token_embs = np.array(emb_list, dtype=np.float32)

                        # Ensure 2D (sequence_length, emb_dim)
                        if token_embs.ndim == 1:
                            token_embs = token_embs.reshape(1, -1)

                        # Truncate
                        if len(token_embs) > self.max_length:
                            token_embs = token_embs[: self.max_length]

                        all_embeddings.append(token_embs)
                    else:
                        all_embeddings.append(np.zeros((1, self.emb_dim), dtype=np.float32))

            except Exception as e:
                # If batch encoding fails, fall back to per-sample encoding
                print(f"  Batch encoding failed, falling back to per-sample: {e}")
                for text in batch:
                    try:
                        emb_list = self.encoder.embed([text])[0]
                        if emb_list and len(emb_list) > 0:
                            token_embs = np.array(emb_list, dtype=np.float32)
                            if token_embs.ndim == 1:
                                token_embs = token_embs.reshape(1, -1)
                            if len(token_embs) > self.max_length:
                                token_embs = token_embs[: self.max_length]
                            all_embeddings.append(token_embs)
                        else:
                            all_embeddings.append(np.zeros((1, self.emb_dim), dtype=np.float32))
                    except Exception as inner_e:
                        all_embeddings.append(np.zeros((1, self.emb_dim), dtype=np.float32))

        return all_embeddings

    def __len__(self):
        return len(self.X_texts)

    def __getitem__(self, idx) -> Dict[str, Any]:
        return {
            "input_embeddings": torch.tensor(
                self.input_embeddings[idx], dtype=torch.float32
            ),
            "target_embeddings": torch.tensor(
                self.target_embeddings[idx], dtype=torch.float32
            ),
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

    # Infer embedding dimension from first sample
    emb_dim = batch[0]["input_embeddings"].shape[1]

    # Create padded tensors
    input_embeddings = torch.zeros(
        batch_size, max_input_len, emb_dim, dtype=torch.float32
    )
    target_embeddings = torch.zeros(
        batch_size, max_target_len, emb_dim, dtype=torch.float32
    )
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


def build_faiss_index_from_dataset(
    dataset: EmbeddingDecoderDataset,
    save_path: str,
) -> FAISSRetriever:
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
        texts=texts, embeddings=embeddings, save_path=save_path
    )

    return retriever
