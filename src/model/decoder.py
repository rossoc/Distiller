import torch
import torch.nn as nn
from typing import Optional


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
        self.num_layers = num_layers

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
        memory_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.decoder(
            tgt,
            memory,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=memory_mask,
        )

        return self.output(x)
