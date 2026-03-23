import torch
import torch.nn as nn
from typing import Optional, Tuple


class Decoder(nn.Module):
    """
    Transformer-based decoder that predicts embeddings.
    
    Input: embeddings of the input text (the "memory")
    Output: list of vectors of dimension 768 (EmbeddingGemma embedding size)
    
    The decoder learns to predict the target embeddings given the input embeddings.
    """
    
    def __init__(
        self,
        output_dim: int = 768,
        emb_dim: int = 768,
        num_layers: int = 6,
        fwd_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize the decoder.
        
        Args:
            output_dim: Output embedding dimension (768 for EmbeddingGemma)
            emb_dim: Model hidden dimension
            num_layers: Number of transformer decoder layers
            fwd_dim: Feed-forward dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.emb_dim = emb_dim
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=fwd_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output = nn.Linear(emb_dim, output_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        memory: torch.Tensor,
        tgt: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            memory: Input embeddings (batch, seq_len, emb_dim)
            tgt: Optional target embeddings for teacher forcing (batch, tgt_seq_len, emb_dim)
            memory_padding_mask: Mask for memory padding (batch, seq_len)
            tgt_padding_mask: Mask for target padding (batch, tgt_seq_len)
            
        Returns:
            predicted_embeddings: (batch, seq_len, output_dim)
        """
        batch_size = memory.shape[0]
        device = memory.device
        
        # Convert padding masks to key_padding_mask format for TransformerDecoder
        # TransformerDecoder expects key_padding_mask as (batch, seq_len) boolean tensor
        # where True means padding
        memory_key_padding_mask = memory_padding_mask if memory_padding_mask is not None else None
        tgt_key_padding_mask = tgt_padding_mask if tgt_padding_mask is not None else None
        
        # If no target provided, use memory as target (for inference)
        if tgt is None:
            x = memory
            x = self.decoder(
                x,
                memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            return self.output(x)
        
        # Teacher forcing mode (training)
        x = self.decoder(
            tgt,
            memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        
        return self.output(x)
    
    @torch.no_grad()
    def generate(
        self,
        memory: torch.Tensor,
        max_len: int = 10,
        start_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate embeddings autoregressively.
        
        Args:
            memory: Input embeddings (batch, seq_len, emb_dim)
            max_len: Maximum number of embeddings to generate
            start_token: Optional start token embedding
            
        Returns:
            generated: (batch, max_len, output_dim)
        """
        batch_size = memory.shape[0]
        device = memory.device
        
        # Use a learned start token or zeros
        if start_token is None:
            start_token = torch.zeros(batch_size, 1, self.emb_dim, device=device)
        elif start_token.dim() == 2:
            start_token = start_token.unsqueeze(1)
        
        generated = start_token
        
        for _ in range(max_len):
            # Decode one step
            x = self.decoder(generated, memory)
            next_emb = self.output(x[:, -1:, :])
            
            # Append to generated
            generated = torch.cat([generated, next_emb], dim=1)
        
        # Remove the start token
        return generated[:, 1:, :]


class EmbeddingPredictionHead(nn.Module):
    """
    Simple MLP head for predicting embeddings from embeddings.
    
    This is an alternative to the Transformer decoder for simpler use cases.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 2048,
        output_dim: int = 768,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
