"""Code embedding model: 6-layer transformer encoder with 384-dim output.

Architecture:
- Tokenizer: WordPiece with language-specific special tokens ([RUST], [PYTHON], etc.)
- Encoder: 6 transformer layers, 384 hidden dim, 6 attention heads
- Pooling: Mean pooling over non-padding tokens
- Output: 384-dimensional embedding (L2-normalized)

~22M parameters. Designed for fast local inference via ONNX Runtime.
"""

from __future__ import annotations

import torch
import torch.nn as nn

LANGUAGE_TOKENS = ["[RUST]", "[PYTHON]", "[JAVASCRIPT]", "[TYPESCRIPT]", "[GO]", "[OTHER]"]


class CodeEncoder(nn.Module):
    """Code embedding model for source code retrieval."""

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        max_seq_len: int = 512,
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
        """Encode input tokens to embeddings.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) 1 for real tokens, 0 for padding

        Returns:
            (batch, hidden_dim) normalized embeddings
        """
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.layer_norm(x)
        x = self.dropout(x)

        # Create causal-free attention mask for transformer
        if attention_mask is not None:
            # Convert 0/1 mask to additive mask (0 → -inf for masked positions)
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Mean pooling over non-padding tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)

        # L2 normalize
        return nn.functional.normalize(pooled, p=2, dim=-1)


def preprocess_code(code: str, language: str | None = None) -> str:
    """Add language token prefix to code for embedding."""
    lang_token = {
        "rust": "[RUST]",
        "python": "[PYTHON]",
        "javascript": "[JAVASCRIPT]",
        "typescript": "[TYPESCRIPT]",
        "go": "[GO]",
    }.get(language or "", "[OTHER]")
    return f"{lang_token} {code}"
