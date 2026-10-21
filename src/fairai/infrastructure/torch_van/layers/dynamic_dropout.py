"""Stochastic dropout with inverted scaling (AS-IS DynamicDropout)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DynamicDropout(nn.Module):
    def __init__(self, dropout_prob: float = 0.5) -> None:
        super().__init__()
        self.dropout_prob = float(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.dropout_prob == 0.0:
            return x
        keep = 1.0 - self.dropout_prob
        mask = (torch.rand_like(x) > self.dropout_prob).to(x.dtype) / keep
        return x * mask
