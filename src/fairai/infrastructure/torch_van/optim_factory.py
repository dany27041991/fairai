"""Optimizer / scheduler / grad-clip helpers for VAN training."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


def adaptive_clip_grad(
    parameters: Iterable[torch.nn.Parameter],
    clip_factor: float,
    *,
    eps: float = 1e-3,
    norm_type: float = 2.0,
) -> None:
    """AS-IS adaptive per-parameter clip (Thesis)."""
    for param in parameters:
        if param.grad is None:
            continue
        param_norm = param.data.norm(norm_type)
        grad_norm = param.grad.data.norm(norm_type)
        if param_norm == 0:
            continue
        clip_coef = clip_factor * (param_norm + eps) / (grad_norm + eps)
        if clip_coef < 1:
            param.grad.data.mul_(clip_coef)


def build_optimizer(model: torch.nn.Module, optim_cfg: dict[str, Any]) -> AdamW:
    lr = float(optim_cfg.get("learning_rate", 1e-5))
    weight_decay = float(optim_cfg.get("weight_decay", 0.01))
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(
    optimizer: AdamW,
    *,
    optim_cfg: dict[str, Any],
    num_training_steps: int,
):
    warmup_pct = float(optim_cfg.get("warmup_pct", 0.1))
    warmup_steps = max(1, int(warmup_pct * num_training_steps))
    kind = str(optim_cfg.get("scheduler", "cosine")).lower()
    if kind == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )


def clip_gradients(
    model: torch.nn.Module,
    *,
    clip_factor: float,
    max_norm: float = 1.0,
) -> None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    if clip_factor > 0:
        adaptive_clip_grad(model.parameters(), clip_factor=clip_factor)
