"""Thesis PPO action selection (temperature / ε-greedy)."""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def select_action_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Thesis ``select_action_temperature`` — sample from softmax(logits / T)."""
    adjusted = logits / max(float(temperature), 1e-8)
    probs = F.softmax(adjusted, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action, probs


def select_action_epsilon(
    logits: torch.Tensor,
    temperature: float = 1.0,
    epsilon: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Thesis ``select_action_epsilon`` — ε-greedy over temperature sampling."""
    if random.random() < float(epsilon):
        n = int(logits.size(-1))
        action = torch.tensor(random.randint(0, n - 1), device=logits.device)
        probs = F.softmax(logits, dim=-1)
        return action, probs
    return select_action_temperature(logits, temperature=temperature)
