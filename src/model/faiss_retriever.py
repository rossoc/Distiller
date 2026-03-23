"""
FAISS Retriever Module

This module handles building and searching the FAISS index for converting
predicted embeddings back to text during inference.

The FAISS index stores all possible output sentences as vectors, allowing
nearest neighbor search to find the closest matching text for a predicted embedding.
"""

import os
import pickle
from typing import List, Tuple
import numpy as np
import faiss


class FAISSRetriever:
    """
    FAISS-based retriever for converting embeddings to text.
    
    At inference time:
    1. The decoder predicts a vector (embedding)
    2. The vector is normalized to unit length
    3. FAISS searches for the nearest neighbor in the index
    4. Returns the corresponding text for the closest match
    """
    
    def __init__(self, embedding_dim: int = 768):
        """
        Initialize the FAISS retriever.
        
        Args:
            embedding_dim: Dimension of the embeddings (768 for EmbeddingGemma)
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.texts = []
        self.embeddings = None
        
    def build_index(self, texts: List[str], embeddings: np.ndarray):
        """
        Build the FAISS index from texts and their embeddings.
        
        Args:
            texts: List of text strings to index
            embeddings: Numpy array of shape (num_texts, embedding_dim)
                       containing pre-computed embeddings
        """
        if len(texts) != len(embeddings):
            raise ValueError(
                f"Number of texts ({len(texts)}) must match number of embeddings "
                f"({len(embeddings)})"
            )
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )
        
        self.texts = texts
        self.embeddings = embeddings.copy()
        
        # Normalize embeddings to unit length for cosine similarity
        # FAISS uses inner product, which equals cosine similarity for normalized vectors
        faiss.normalize_L2(self.embeddings)
        
        # Create FAISS index (Inner Product = Cosine Similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.embeddings)
        
        print(f"FAISS index built with {len(self.texts)} entries")
        
    def search(self, query_embeddings: np.ndarray, k: int = 1) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Search for the nearest neighbors of query embeddings.
        
        Args:
            query_embeddings: Numpy array of shape (batch_size, embedding_dim)
                             containing predicted embeddings to search
            k: Number of nearest neighbors to return
            
        Returns:
            results: List of lists of texts (batch_size, k)
            scores: List of lists of similarity scores (batch_size, k)
        """
        if self.index is None:
            raise RuntimeError("FAISS index not built. Call build_index() first.")
        
        # Normalize query embeddings
        query_embeddings = query_embeddings.copy()
        faiss.normalize_L2(query_embeddings)
        
        # Search
        scores, indices = self.index.search(query_embeddings, k)
        
        # Convert indices to texts
        results = []
        score_list = []
        for i in range(len(query_embeddings)):
            result_texts = []
            result_scores = []
            for j in range(k):
                if indices[i, j] < len(self.texts):
                    result_texts.append(self.texts[indices[i, j]])
                    result_scores.append(float(scores[i, j]))
            results.append(result_texts)
            score_list.append(result_scores)
        
        return results, score_list
    
    def search_single(self, query_embedding: np.ndarray, k: int = 1) -> Tuple[List[str], List[float]]:
        """
        Search for a single query embedding.
        
        Args:
            query_embedding: Numpy array of shape (embedding_dim,)
            k: Number of nearest neighbors to return
            
        Returns:
            texts: List of k nearest neighbor texts
            scores: List of k similarity scores
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        results, scores = self.search(query_embedding, k)
        return results[0], scores[0]
    
    def save(self, filepath: str):
        """
        Save the FAISS index and texts to disk.
        
        Args:
            filepath: Path to save the index (will save .index and .texts files)
        """
        if self.index is None:
            raise RuntimeError("No index to save. Call build_index() first.")
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save texts and raw embeddings
        with open(f"{filepath}.data", "wb") as f:
            pickle.dump({
                "texts": self.texts,
                "embeddings": self.embeddings,
                "embedding_dim": self.embedding_dim
            }, f)
        
        print(f"FAISS index saved to {filepath}.index and {filepath}.data")
        
    def load(self, filepath: str):
        """
        Load the FAISS index and texts from disk.
        
        Args:
            filepath: Path to the saved index (without extension)
        """
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.index")
        
        # Load texts and embeddings
        with open(f"{filepath}.data", "rb") as f:
            data = pickle.load(f)
            self.texts = data["texts"]
            self.embeddings = data["embeddings"]
            self.embedding_dim = data["embedding_dim"]
        
        print(f"FAISS index loaded with {len(self.texts)} entries")


def build_retriever_from_texts(texts: List[str], embeddings: np.ndarray, 
                                save_path: str = None) -> FAISSRetriever:
    """
    Convenience function to build a FAISS retriever from texts and embeddings.
    
    Args:
        texts: List of text strings to index
        embeddings: Numpy array of pre-computed embeddings
        save_path: Optional path to save the index
        
    Returns:
        retriever: FAISSRetriever instance with built index
    """
    retriever = FAISSRetriever(embedding_dim=embeddings.shape[1])
    retriever.build_index(texts, embeddings)
    
    if save_path:
        retriever.save(save_path)
    
    return retriever
