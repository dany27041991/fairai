"""Residual MLP block used in VAN classifier head."""

from __future__ import annotations

import torch
import torch.nn as nn

from fairai.infrastructure.torch_van.layers.dynamic_dropout import DynamicDropout


class ResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.LayerNorm(out_features),
            DynamicDropout(dropout),
        )
        self.shortcut: nn.Module
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.shortcut(x)
