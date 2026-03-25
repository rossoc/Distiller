"""
Token Extractor Module

This module provides functionality to convert predicted embeddings back to tokens
using cosine similarity with the model's embedding weight matrix.

The approach follows the method described in note/extract_token.md:
1. Load the embedding weight matrix from the GGUF model
2. For each predicted embedding, compute cosine similarity with all token vectors
3. Select the token with highest similarity (argmax)
4. Decode token IDs to text using the tokenizer
"""

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TokenExtractionResult:
    """Result of token extraction for a single sequence."""
    token_ids: List[int]
    tokens: List[str]
    text: str
    confidence_scores: List[float]
    sequence_length: int


class TokenExtractor:
    """
    Extracts tokens from predicted embeddings using cosine similarity.
    
    This class loads the embedding weight matrix and vocabulary from the GGUF model,
    then provides methods to convert predicted embeddings to token sequences.
    
    Example:
        >>> extractor = TokenExtractor.from_gguf_model("models/embeddinggemma-300M-Q8.gguf")
        >>> # For a predicted embedding of shape (768,)
        >>> token_id, confidence = extractor.extract_token(predicted_embedding)
        >>> # For a sequence of embeddings (seq_len, 768)
        >>> result = extractor.extract_sequence(predicted_embeddings)
        >>> print(result.text)
    """
    
    def __init__(
        self,
        embedding_weights: np.ndarray,
        vocab_data: Dict[str, Any],
    ):
        """
        Initialize the token extractor.
        
        Args:
            embedding_weights: Weight matrix of shape (vocab_size, embedding_dim)
            vocab_data: Dictionary containing vocabulary information with keys:
                - tokens: List of token strings
                - bos_token_id: Beginning of sequence token ID
                - eos_token_id: End of sequence token ID
        """
        self.embedding_weights = embedding_weights  # (vocab_size, embedding_dim)
        self.vocab_size = embedding_weights.shape[0]
        self.embedding_dim = embedding_weights.shape[1]
        
        self.tokens = vocab_data['tokens']
        self.bos_token_id = vocab_data['bos_token_id']
        self.eos_token_id = vocab_data['eos_token_id']
        
        # Validate dimensions
        assert len(self.tokens) == self.vocab_size, \
            f"Vocab size mismatch: {len(self.tokens)} tokens vs {self.vocab_size} weights"
        
        # Pre-compute normalized weights for cosine similarity
        # This is done once during initialization for efficiency
        self._normalized_weights = None
    
    @property
    def normalized_weights(self) -> torch.Tensor:
        """Lazily compute and cache normalized embedding weights."""
        if self._normalized_weights is None:
            weights_tensor = torch.from_numpy(self.embedding_weights)
            self._normalized_weights = F.normalize(weights_tensor, p=2, dim=-1)
        return self._normalized_weights
    
    @classmethod
    def from_gguf_model(
        cls,
        model_path: str,
        cache_dir: Optional[str] = None,
    ) -> "TokenExtractor":
        """
        Load token extractor from a GGUF model file.
        
        This method extracts the embedding weights and vocabulary from the GGUF file
        and caches them for faster subsequent loads.
        
        Args:
            model_path: Path to the GGUF model file
            cache_dir: Directory to cache extracted weights (default: same directory as model)
        
        Returns:
            TokenExtractor instance with loaded weights and vocabulary
        
        Example:
            >>> extractor = TokenExtractor.from_gguf_model("models/embeddinggemma-300M-Q8.gguf")
        """
        model_path = Path(model_path)
        if cache_dir is None:
            cache_dir = model_path.parent
        
        cache_dir = Path(cache_dir)
        weights_cache_path = cache_dir / "token_embedding_weights.npy"
        vocab_cache_path = cache_dir / "tokenizer_vocab.json"
        
        # Try to load from cache first
        if weights_cache_path.exists() and vocab_cache_path.exists():
            print(f"Loading cached weights from {weights_cache_path}")
            embedding_weights = np.load(weights_cache_path)
            
            with open(vocab_cache_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            print(f"Loaded cached weights: {embedding_weights.shape}")
            return cls(embedding_weights, vocab_data)
        
        # Extract from GGUF file
        print(f"Extracting weights from GGUF model: {model_path}")
        embedding_weights, vocab_data = cls._extract_from_gguf(model_path)
        
        # Cache the extracted data
        print(f"Caching extracted weights to {weights_cache_path}")
        np.save(weights_cache_path, embedding_weights)
        
        with open(vocab_cache_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        return cls(embedding_weights, vocab_data)
    
    @staticmethod
    def _extract_from_gguf(
        model_path: Path
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract embedding weights and vocabulary from a GGUF file.
        
        Args:
            model_path: Path to the GGUF model file
        
        Returns:
            Tuple of (embedding_weights, vocab_data)
        """
        import gguf
        
        reader = gguf.GGUFReader(model_path, 'r')
        
        # Find and extract token embedding weights
        token_embd = None
        for tensor in reader.tensors:
            if tensor.name == 'token_embd.weight':
                token_embd = tensor
                break
        
        if token_embd is None:
            raise ValueError("Token embedding tensor not found in GGUF file")

        # Dequantize weights
        from gguf import dequantize
        embedding_weights = dequantize(token_embd.data, token_embd.tensor_type)

        # Ensure correct shape (vocab_size, embedding_dim)
        if embedding_weights.shape == (768, 262144):
            embedding_weights = embedding_weights.T

        # Extract vocabulary using .contents() method
        tokens_field = reader.fields['tokenizer.ggml.tokens']
        tokens = tokens_field.contents()  # Returns list of strings

        # Get special token IDs
        bos_field = reader.fields['tokenizer.ggml.bos_token_id']
        eos_field = reader.fields['tokenizer.ggml.eos_token_id']
        bos_id = int(bos_field.parts[0][0])
        eos_id = int(eos_field.parts[0][0])

        vocab_data = {
            'tokens': tokens,
            'bos_token_id': bos_id,
            'eos_token_id': eos_id,
            'vocab_size': len(tokens)
        }

        return embedding_weights, vocab_data
    
    def extract_token(
        self,
        predicted_embedding: np.ndarray,
        return_confidence: bool = True,
    ) -> Tuple[int, Optional[float]]:
        """
        Extract a single token from a predicted embedding.
        
        Args:
            predicted_embedding: Predicted embedding vector of shape (embedding_dim,)
            return_confidence: Whether to return confidence score
        
        Returns:
            Tuple of (token_id, confidence_score) or (token_id, None)
        
        Example:
            >>> embedding = np.random.randn(768)
            >>> token_id, confidence = extractor.extract_token(embedding)
        """
        # Convert to tensor and normalize
        pred_tensor = torch.from_numpy(predicted_embedding).float()
        pred_normalized = F.normalize(pred_tensor, p=2, dim=-1)
        
        # Compute cosine similarity with all token embeddings
        # Shape: (vocab_size,)
        similarities = torch.matmul(pred_normalized, self.normalized_weights.T)
        
        # Get token with highest similarity
        token_id = torch.argmax(similarities).item()
        
        if return_confidence:
            confidence = similarities[token_id].item()
            return token_id, confidence
        
        return token_id, None
    
    def extract_sequence(
        self,
        predicted_embeddings: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        stop_at_eos: bool = True,
        return_confidence: bool = True,
    ) -> TokenExtractionResult:
        """
        Extract a sequence of tokens from predicted embeddings.
        
        Args:
            predicted_embeddings: Array of shape (seq_len, embedding_dim)
            attention_mask: Boolean mask of shape (seq_len,), True = valid token
            stop_at_eos: Whether to stop decoding at EOS token
            return_confidence: Whether to return confidence scores
        
        Returns:
            TokenExtractionResult containing token IDs, tokens, text, and confidence scores
        
        Example:
            >>> embeddings = np.random.randn(50, 768)
            >>> result = extractor.extract_sequence(embeddings)
            >>> print(f"Decoded text: {result.text}")
            >>> print(f"Average confidence: {np.mean(result.confidence_scores):.3f}")
        """
        seq_len = predicted_embeddings.shape[0]
        
        # Create attention mask if not provided (all valid)
        if attention_mask is None:
            attention_mask = np.ones(seq_len, dtype=bool)
        
        # Process all positions at once for efficiency
        # Shape: (seq_len, embedding_dim)
        pred_tensor = torch.from_numpy(predicted_embeddings).float()
        pred_normalized = F.normalize(pred_tensor, p=2, dim=-1)
        
        # Compute cosine similarities: (seq_len, vocab_size)
        similarities = torch.matmul(pred_normalized, self.normalized_weights.T)
        
        # Get token IDs: (seq_len,)
        token_ids = torch.argmax(similarities, dim=-1).tolist()
        
        # Get confidence scores
        if return_confidence:
            confidence_scores = []
            for i, tok_id in enumerate(token_ids):
                if attention_mask[i]:
                    conf = similarities[i, tok_id].item()
                else:
                    conf = 0.0
                confidence_scores.append(conf)
        else:
            confidence_scores = [None] * seq_len
        
        # Convert token IDs to tokens
        tokens = [self.tokens[tid] for tid in token_ids]
        
        # Handle EOS stopping
        if stop_at_eos:
            eos_positions = [i for i, tid in enumerate(token_ids) if tid == self.eos_token_id]
            if eos_positions:
                first_eos = eos_positions[0]
                token_ids = token_ids[:first_eos + 1]
                tokens = tokens[:first_eos + 1]
                if return_confidence:
                    confidence_scores = confidence_scores[:first_eos + 1]
        
        # Filter by attention mask
        filtered_token_ids = []
        filtered_tokens = []
        filtered_confidences = []
        
        for i, (tid, tok, conf) in enumerate(zip(token_ids, tokens, confidence_scores)):
            if i < len(attention_mask) and not attention_mask[i]:
                continue
            filtered_token_ids.append(tid)
            filtered_tokens.append(tok)
            if return_confidence:
                filtered_confidences.append(conf)
        
        # Decode tokens to text
        text = self.detokenize(filtered_tokens)
        
        return TokenExtractionResult(
            token_ids=filtered_token_ids,
            tokens=filtered_tokens,
            text=text,
            confidence_scores=filtered_confidences if return_confidence else [],
            sequence_length=len(filtered_token_ids)
        )
    
    def extract_batch(
        self,
        predicted_embeddings: np.ndarray,
        attention_masks: Optional[np.ndarray] = None,
        stop_at_eos: bool = True,
        return_confidence: bool = True,
    ) -> List[TokenExtractionResult]:
        """
        Extract token sequences for a batch of predicted embeddings.
        
        Args:
            predicted_embeddings: Array of shape (batch_size, seq_len, embedding_dim)
            attention_masks: Boolean mask of shape (batch_size, seq_len)
            stop_at_eos: Whether to stop decoding at EOS token
            return_confidence: Whether to return confidence scores
        
        Returns:
            List of TokenExtractionResult for each sample in the batch
        
        Example:
            >>> embeddings = np.random.randn(32, 50, 768)  # batch of 32
            >>> results = extractor.extract_batch(embeddings)
            >>> for i, result in enumerate(results):
            ...     print(f"Sample {i}: {result.text}")
        """
        batch_size = predicted_embeddings.shape[0]
        results = []
        
        for i in range(batch_size):
            emb = predicted_embeddings[i]
            mask = attention_masks[i] if attention_masks is not None else None
            
            result = self.extract_sequence(
                emb,
                attention_mask=mask,
                stop_at_eos=stop_at_eos,
                return_confidence=return_confidence
            )
            results.append(result)
        
        return results
    
    def detokenize(self, tokens: List[str]) -> str:
        """
        Convert a list of tokens to text.
        
        This uses a simple concatenation approach. For more sophisticated
        detokenization (handling spaces, special tokens, etc.), you may
        want to use a dedicated tokenizer.
        
        Args:
            tokens: List of token strings
        
        Returns:
            Detokenized text string
        """
        # Simple detokenization - join tokens
        # Note: This is a basic approach. Gemma uses SentencePiece which
        # encodes spaces as '▁' (U+2581) prefix.
        text = ''.join(tokens)
        
        # Handle SentencePiece-style space encoding
        text = text.replace('▁', ' ')
        
        return text
    
    def get_token_text(self, token_id: int) -> str:
        """Get the text representation of a token ID."""
        return self.tokens[token_id]
    
    def get_confidence_stats(
        self,
        results: List[TokenExtractionResult]
    ) -> Dict[str, float]:
        """
        Compute confidence statistics for a list of extraction results.
        
        Args:
            results: List of TokenExtractionResult
        
        Returns:
            Dictionary with confidence statistics (mean, std, min, max)
        """
        all_confidences = []
        for result in results:
            all_confidences.extend(result.confidence_scores)
        
        if not all_confidences:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        conf_array = np.array(all_confidences)
        return {
            'mean': float(np.mean(conf_array)),
            'std': float(np.std(conf_array)),
            'min': float(np.min(conf_array)),
            'max': float(np.max(conf_array)),
        }
