"""Token pooling over transformer sequence outputs."""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int, att_dropout: float = 0.1) -> None:
        super().__init__()
        self.attention_vector = nn.Parameter(torch.randn(hidden_size))
        self.dropout = nn.Dropout(att_dropout)

    def forward(
        self, token_embeddings: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        scores = torch.matmul(token_embeddings, self.attention_vector)
        if mask is not None:
            scores = scores.float().masked_fill(mask == 0, -1e9).to(scores.dtype)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(token_embeddings * weights.unsqueeze(-1), dim=1)
        return self.dropout(pooled)


class MultiHeadAttentionPooling(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int = 4, att_dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=att_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(att_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self, token_embeddings: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        key_padding_mask = (mask == 0) if mask is not None else None
        attn_output, _ = self.mha(
            token_embeddings,
            token_embeddings,
            token_embeddings,
            key_padding_mask=key_padding_mask,
        )
        pooled = attn_output.mean(dim=1)
        pooled = self.layer_norm(pooled)
        return self.dropout(pooled)
