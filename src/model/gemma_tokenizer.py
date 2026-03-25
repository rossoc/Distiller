"""
Tokenizer wrapper for Gemma vocabulary.

Provides a simple interface for tokenization and detokenization
using the Gemma vocabulary from the GGUF model.
"""

import torch
from typing import List, Optional
from model.token_extractor import TokenExtractor


class GemmaTokenizer:
    """
    Simple tokenizer wrapper for Gemma vocabulary.
    
    Uses the vocabulary from TokenExtractor to encode/decode text.
    """
    
    def __init__(self, token_extractor: TokenExtractor):
        """
        Initialize the tokenizer.
        
        Args:
            token_extractor: TokenExtractor instance with loaded vocabulary
        """
        self.token_extractor = token_extractor
        self.tokens = token_extractor.tokens
        self.vocab_size = token_extractor.vocab_size
        self.bos_token_id = token_extractor.bos_token_id
        self.eos_token_id = token_extractor.eos_token_id
        
        # Build token-to-id mapping for fast lookup
        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}
    
    @classmethod
    def from_gguf_model(cls, model_path: str) -> "GemmaTokenizer":
        """
        Load tokenizer from GGUF model.
        
        Args:
            model_path: Path to the GGUF model file
            
        Returns:
            GemmaTokenizer instance
        """
        extractor = TokenExtractor.from_gguf_model(model_path)
        return cls(extractor)
    
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Encode text to token IDs.
        
        Uses a simple approach: finds the best matching token for each
        character/word segment. For production use, consider using
        SentencePiece tokenizer directly.
        
        Args:
            text: Input text to tokenize
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
            max_length: Maximum sequence length (truncates if longer)
            
        Returns:
            List of token IDs
        """
        # Simple tokenization approach using the vocabulary
        # This is a basic implementation - for better tokenization,
        # use SentencePiece directly
        
        tokens = []
        
        # Add BOS token if requested
        if add_bos:
            tokens.append(self.bos_token_id)
        
        # Simple character-level tokenization as fallback
        # In practice, you'd want to use SentencePiece for proper subword tokenization
        text = text.replace(' ', '▁')  # SentencePiece space encoding
        
        # Try to match longest substrings from vocabulary
        i = 0
        while i < len(text):
            matched = False
            # Try longest matches first
            for length in range(min(16, len(text) - i), 0, -1):
                substring = text[i:i + length]
                if substring in self.token_to_id:
                    tokens.append(self.token_to_id[substring])
                    i += length
                    matched = True
                    break
            
            if not matched:
                # Skip unmatched character
                i += 1
        
        # Add EOS token if requested
        if add_eos:
            tokens.append(self.eos_token_id)
        
        # Truncate if needed
        if max_length is not None and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        return tokens
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip BOS/EOS tokens
            
        Returns:
            Decoded text string
        """
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in [self.bos_token_id, self.eos_token_id]:
                continue
            tokens.append(self.tokens[token_id])
        
        # Join tokens and handle SentencePiece space encoding
        text = ''.join(tokens)
        text = text.replace('▁', ' ')
        
        return text
    
    def batch_encode(
        self,
        texts: List[str],
        add_bos: bool = True,
        add_eos: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
        pad_token_id: int = -100,
    ) -> List[List[int]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of texts to encode
            add_bos: Whether to add BOS tokens
            add_eos: Whether to add EOS tokens
            max_length: Maximum sequence length
            padding: Whether to pad sequences to max_length
            pad_token_id: Token ID to use for padding
            
        Returns:
            List of token ID sequences
        """
        encoded = [
            self.encode(text, add_bos=add_bos, add_eos=add_eos, max_length=max_length)
            for text in texts
        ]
        
        if padding and max_length is not None:
            max_len = max(len(seq) for seq in encoded)
            for i, seq in enumerate(encoded):
                if len(seq) < max_len:
                    encoded[i] = seq + [pad_token_id] * (max_len - len(seq))
        
        return encoded


def get_gemma_tokenizer(model_path: str = "models/embeddinggemma-300M-Q8.gguf") -> GemmaTokenizer:
    """
    Get a Gemma tokenizer instance.
    
    Args:
        model_path: Path to the GGUF model file
        
    Returns:
        GemmaTokenizer instance
    """
    return GemmaTokenizer.from_gguf_model(model_path)
