"""Memory embedding model: same transformer base as code model + temporal projection.

Architecture:
- Same 6-layer transformer encoder as CodeEncoder
- Additional temporal projection head for query-time re-ranking
- Structured input: [TYPE] {type} [DESC] {description} [BODY] {body}

The temporal component is NOT baked into the embedding. Instead, the model
produces a content embedding, and temporal features are applied at query time
as a re-ranking score.
"""

from __future__ import annotations

import torch
import torch.nn as nn

MEMORY_TYPE_TOKENS = ["[USER]", "[FEEDBACK]", "[PROJECT]", "[REFERENCE]"]
STRUCTURE_TOKENS = ["[TYPE]", "[DESC]", "[BODY]"]


class MemoryEncoder(nn.Module):
    """Memory embedding model with temporal awareness."""

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode memory content to embedding.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) 1 for real tokens, 0 for padding

        Returns:
            (batch, hidden_dim) normalized content embeddings
        """
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.layer_norm(x)
        x = self.dropout(x)

        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Mean pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)

        return nn.functional.normalize(pooled, p=2, dim=-1)


def preprocess_memory(
    memory_type: str,
    description: str,
    body: str,
) -> str:
    """Format a memory file into structured input for the encoder."""
    type_token = {
        "user": "[USER]",
        "feedback": "[FEEDBACK]",
        "project": "[PROJECT]",
        "reference": "[REFERENCE]",
    }.get(memory_type, "[USER]")
    return f"[TYPE] {type_token} [DESC] {description} [BODY] {body}"
