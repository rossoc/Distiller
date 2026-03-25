import torch
import torch.nn as nn
from typing import Optional


class Decoder2(nn.Module):
    """
    Transformer-based decoder that predicts vocabulary tokens.

    Input: embeddings of the input text (the "memory")
    Output: logits over the vocabulary (vocab_size = 256000 for Gemma)

    The decoder learns to predict token IDs from the vocabulary given the input embeddings.
    """

    def __init__(
        self,
        vocab_size: int = 256000,
        emb_dim: int = 768,
        num_layers: int = 6,
        fwd_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize the decoder.

        Args:
            vocab_size: Vocabulary size (256000 for Gemma tokenizer)
            emb_dim: Model hidden dimension
            num_layers: Number of transformer decoder layers
            fwd_dim: Feed-forward dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

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

        self.output = nn.Linear(emb_dim, vocab_size)

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
        # Ensure masks are boolean type
        if memory_mask is not None:
            memory_mask = memory_mask.bool()
        if tgt_mask is not None:
            tgt_mask = tgt_mask.bool()
        
        x = self.decoder(
            tgt,
            memory,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=memory_mask,
        )

        return self.output(x)
