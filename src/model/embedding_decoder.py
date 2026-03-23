"""
Embedding Decoder Module

This module provides a decoder-only model that can generate text from embeddings
produced by EmbeddingGemma or similar encoder models.

Architecture:
    Text -> EmbeddingGemma (Encoder) -> Embedding (768-dim) 
    -> Projection Layer -> Decoder Input Embeddings -> Gemma Decoder -> Text
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass


@dataclass
class EmbeddingDecoderConfig:
    """Configuration for the EmbeddingDecoder model."""
    
    encoder_model_name: str = "google/embeddinggemma-300m"
    decoder_model_name: str = "google/gemma-2-2b"
    embedding_dim: int = 768  # EmbeddingGemma output dimension
    decoder_hidden_size: int = 2304  # Gemma-2-2B hidden size
    projection_hidden_dim: int = 2048  # Hidden dimension in projection MLP
    dropout: float = 0.1
    freeze_encoder: bool = True
    use_mlp_projection: bool = True  # Use MLP instead of linear projection


class EmbeddingDecoderModel(nn.Module):
    """
    A model that decodes text from embeddings.
    
    Takes embeddings from an encoder model (e.g., EmbeddingGemma) and generates
    text using a decoder-only language model.
    
    The key innovation is a projection layer that maps encoder embeddings to
    the decoder's input embedding space, enabling text generation from
    fixed-dimensional vectors.
    """
    
    def __init__(self, config: EmbeddingDecoderConfig):
        super().__init__()
        self.config = config
        
        # Load the decoder model (we'll use its embeddings and transformer)
        self.decoder = AutoModelForCausalLM.from_pretrained(
            config.decoder_model_name,
            torch_dtype=torch.float32,
            attn_implementation="eager",  # Use eager for compatibility
        )
        
        # Get decoder's embedding dimension
        decoder_embed_dim = self.decoder.config.hidden_size
        
        # Projection layer: maps encoder embeddings to decoder input space
        if config.use_mlp_projection:
            self.projection = nn.Sequential(
                nn.Linear(config.embedding_dim, config.projection_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.projection_hidden_dim, decoder_embed_dim),
            )
        else:
            self.projection = nn.Linear(config.embedding_dim, decoder_embed_dim)
        
        # Initialize projection layer weights
        self._init_projection_weights()
        
    def _init_projection_weights(self):
        """Initialize projection layer with Xavier initialization."""
        for module in self.projection.modules() if hasattr(self.projection, 'modules') else [self.projection]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training.
        
        Args:
            input_embeddings: Tensor of shape (batch_size, seq_len, embedding_dim)
                - Pre-computed embeddings from the encoder
            attention_mask: Tensor of shape (batch_size, seq_len)
                - Mask to avoid padding
            labels: Tensor of shape (batch_size, seq_len), optional
                - Labels for computing language modeling loss
            
        Returns:
            loss: Optional tensor if labels are provided
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Project embeddings to decoder input space
        projected_embeddings = self.projection(input_embeddings)
        
        # Get decoder's embedding layer to add positional encodings implicitly
        # We'll use the decoder model directly with inputs_embeds
        outputs = self.decoder(
            inputs_embeds=projected_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=False,
            **kwargs
        )
        
        loss = outputs[0] if labels is not None else None
        logits = outputs[1] if len(outputs) > 1 else outputs[0]
        
        return loss, logits
    
    @torch.no_grad()
    def generate(
        self,
        input_embeddings: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text from embeddings using autoregressive decoding.
        
        Args:
            input_embeddings: Tensor of shape (batch_size, embedding_dim)
                - Single embedding vector per sample (e.g., from mean pooling)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for token repetition
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
        
        Returns:
            generated_tokens: Tensor of shape (batch_size, generated_seq_len)
        """
        self.eval()
        
        batch_size = input_embeddings.shape[0]
        device = input_embeddings.device
        
        # Project embedding to decoder input space
        # Shape: (batch_size, 1, decoder_hidden_size)
        projected = self.projection(input_embeddings.unsqueeze(1))
        
        generated_ids_list = []
        
        for i in range(batch_size):
            # Start with the projected embedding as initial input
            current_embeds = projected[i:i+1]  # (1, 1, decoder_hidden_size)
            generated_ids = []
            
            for _ in range(max_new_tokens):
                # Get decoder output for current position
                outputs = self.decoder(inputs_embeds=current_embeds)
                next_token_logits = outputs.logits[:, -1, :]  # (1, vocab_size)
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-p sampling
                if top_p < 1.0:
                    next_token_logits = self._apply_top_p_filtering(
                        next_token_logits, top_p=top_p
                    )
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for EOS
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
                
                generated_ids.append(next_token.item())
                
                # Get embedding for next token
                next_token_embeds = self.decoder.get_input_embeddings()(next_token)
                
                # Concatenate with previous embeddings
                current_embeds = torch.cat([current_embeds, next_token_embeds.unsqueeze(1)], dim=1)
            
            generated_ids_list.append(torch.tensor(generated_ids, device=device))
        
        # Pad sequences to same length
        max_len = max(len(ids) for ids in generated_ids_list)
        padded_ids = torch.full(
            (batch_size, max_len), 
            pad_token_id or 0, 
            dtype=torch.long, 
            device=device
        )
        for i, ids in enumerate(generated_ids_list):
            if len(ids) > 0:
                padded_ids[i, :len(ids)] = ids
        
        return padded_ids
    
    def _apply_top_p_filtering(
        self, 
        logits: torch.Tensor, 
        top_p: float,
        min_tokens_to_keep: int = 1
    ) -> torch.Tensor:
        """Apply top-p (nucleus) sampling filtering."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least min_tokens_to_keep tokens
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        
        return logits


class EmbeddingEncoderWrapper:
    """
    Wrapper for loading and using EmbeddingGemma as an encoder.
    
    This wrapper handles loading the encoder model and producing embeddings
    that can be fed to the EmbeddingDecoderModel.
    """
    
    def __init__(
        self, 
        model_name: str = "google/embeddinggemma-300m",
        device: str = "cpu"
    ):
        self.device = device
        self.model_name = model_name
        
        # Load the embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
            self.using_sentence_transformers = True
        except ImportError:
            # Fallback to transformers
            from transformers import AutoModel, AutoTokenizer
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.using_sentence_transformers = False
    
    @torch.no_grad()
    def encode(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = False
    ) -> torch.Tensor:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
        
        Returns:
            embeddings: Tensor of shape (num_texts, embedding_dim)
        """
        if self.using_sentence_transformers:
            embeddings = self.model.encode(
                texts, 
                batch_size=batch_size, 
                show_progress_bar=show_progress,
                convert_to_tensor=True
            )
        else:
            # Manual encoding with transformers
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                # Mean pooling
                embeddings = self._mean_pooling(
                    outputs.last_hidden_state, 
                    inputs['attention_mask']
                )
                all_embeddings.append(embeddings)
            
            embeddings = torch.cat(all_embeddings, dim=0)
        
        return embeddings
    
    def _mean_pooling(
        self, 
        model_output: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


def load_embedding_decoder(
    decoder_path: str,
    encoder_model_name: str = "google/embeddinggemma-300m",
    decoder_model_name: str = "google/gemma-2-2b",
    device: str = "cpu"
) -> Tuple[EmbeddingDecoderModel, EmbeddingEncoderWrapper]:
    """
    Load a trained embedding decoder model and its encoder.
    
    Args:
        decoder_path: Path to saved decoder checkpoint
        encoder_model_name: Name of the encoder model
        decoder_model_name: Name of the base decoder model
        device: Device to load models on
    
    Returns:
        decoder_model: Loaded EmbeddingDecoderModel
        encoder: EmbeddingEncoderWrapper
    """
    # Create config
    config = EmbeddingDecoderConfig(
        encoder_model_name=encoder_model_name,
        decoder_model_name=decoder_model_name,
    )
    
    # Initialize model
    decoder_model = EmbeddingDecoderModel(config)
    
    # Load trained weights
    checkpoint = torch.load(decoder_path, map_location=device, weights_only=True)
    decoder_model.load_state_dict(checkpoint['model_state_dict'])
    decoder_model.to(device)
    decoder_model.eval()
    
    # Load encoder
    encoder = EmbeddingEncoderWrapper(encoder_model_name, device=device)
    
    return decoder_model, encoder
