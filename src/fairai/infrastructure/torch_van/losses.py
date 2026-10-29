"""Focal + BiasAware losses (AS-IS Thesis semantics)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """CE + focal modulation with optional label smoothing and per-class alpha."""

    def __init__(
        self,
        *,
        alpha: float | list[float] = 1.0,
        gamma: float = 1.0,
        num_labels: int = 3,
        label_smoothing: float = 0.0,
        reduction: str = "none",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        if isinstance(alpha, (float, int)):
            alpha_t = torch.full((num_labels,), float(alpha))
        else:
            alpha_t = torch.tensor(list(alpha), dtype=torch.float32)
            if alpha_t.numel() != num_labels:
                raise ValueError("len(alpha) must equal num_labels")
        self.register_buffer("alpha", alpha_t)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        alpha_t = self.alpha.to(device=inputs.device, dtype=ce.dtype)[targets]
        loss = alpha_t * (1.0 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BiasAwareLoss(nn.Module):
    """Wraps a per-example base loss with ethical penalties and optional class weights."""

    def __init__(
        self,
        base_loss_fn: nn.Module,
        *,
        penalty_stereo: float = 1.0,
        penalty_anti: float = 1.0,
        penalty_neutral: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.penalty_stereo = float(penalty_stereo)
        self.penalty_anti = float(penalty_anti)
        self.penalty_neutral = float(penalty_neutral)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.detach().float())
        else:
            self.class_weights = None  # type: ignore[assignment]

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.base_loss_fn(inputs, targets)
        if self.class_weights is not None:
            factor = self.class_weights.to(inputs.device)[targets]
        else:
            factor = torch.ones_like(loss)
        factor = factor.clone()
        factor[targets == 0] *= self.penalty_stereo
        factor[targets == 1] *= self.penalty_anti
        factor[targets == 2] *= self.penalty_neutral
        return (loss * factor).mean()


def compute_class_weights(
    labels: list[int] | torch.Tensor,
    *,
    num_classes: int = 3,
) -> torch.Tensor:
    """Thesis ``compute_class_weights``: ``total / (C · count_c)`` per class."""
    if isinstance(labels, torch.Tensor):
        label_list = [int(x) for x in labels.tolist()]
    else:
        label_list = [int(x) for x in labels]
    counts = [0] * num_classes
    for lab in label_list:
        if 0 <= lab < num_classes:
            counts[lab] += 1
    total = sum(counts)
    weights = [
        (total / (num_classes * c) if c > 0 else 0.0) for c in counts
    ]
    return torch.tensor(weights, dtype=torch.float32)


def build_bias_aware_loss(
    loss_cfg: dict,
    *,
    num_labels: int = 3,
    class_weights: torch.Tensor | None = None,
) -> BiasAwareLoss:
    focal = FocalLoss(
        alpha=float(loss_cfg.get("alpha", 1.0)),
        gamma=float(loss_cfg.get("gamma", 1.0)),
        num_labels=num_labels,
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
        reduction="none",
    )
    return BiasAwareLoss(
        focal,
        penalty_stereo=float(loss_cfg.get("penalty_stereo", 1.0)),
        penalty_anti=float(loss_cfg.get("penalty_anti", 1.0)),
        penalty_neutral=float(loss_cfg.get("penalty_neutral", 1.0)),
        class_weights=class_weights,
    )


def build_train_criterion(
    loss_cfg: dict,
    *,
    num_labels: int = 3,
    labels_for_weights: list[int] | None = None,
) -> nn.Module:
    """Build train loss from YAML.

    ``loss.kind``: ``bias_aware`` (default Thesis) | ``ce`` (plain CrossEntropy).
    ``loss.use_class_weights``: when True (default), Thesis inverse-frequency weights.
    """
    kind = str(loss_cfg.get("kind", "bias_aware")).lower()
    use_cw = bool(loss_cfg.get("use_class_weights", True))
    class_weights: torch.Tensor | None = None
    if use_cw and labels_for_weights is not None:
        class_weights = compute_class_weights(labels_for_weights, num_classes=num_labels)

    if kind in {"ce", "cross_entropy", "crossentropy"}:
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights)
        return nn.CrossEntropyLoss()

    return build_bias_aware_loss(
        loss_cfg, num_labels=num_labels, class_weights=class_weights
    )
