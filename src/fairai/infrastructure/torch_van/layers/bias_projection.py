"""Bias projection layer — subtract estimated bias component from pooled emb."""

from __future__ import annotations

import torch
import torch.nn as nn


class BiasProjectionLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        lambda_factor: float = 1.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        curr_dim = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, in_dim))
        self.projection = nn.Sequential(*layers)
        self.lambda_factor = float(lambda_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias_component = self.projection(x)
        return x - self.lambda_factor * bias_component
