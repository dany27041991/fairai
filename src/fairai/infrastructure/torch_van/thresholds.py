"""Post-hoc decision rules using temperature + conf/margin/var thresholds.

Thesis cells 5/7: Monte Carlo dropout + dynamic threshold
``dynamic_threshold = conf - 0.5 * max_var``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fairai.domain.labels import Label


def predict_from_logits(
    logits: torch.Tensor,
    *,
    temp: float = 1.0,
    conf: float = 0.0,
    margin: float = 0.0,
    var: float = 0.0,
    apply_gates: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pred_labels, probs).

    When ``apply_gates`` is True, low-confidence / low-margin / high-variance
    predictions are forced to NEUTRAL (label 2). Single-pass gate (batch eval).
    For Thesis MC dropout + dynamic threshold, use ``predict_mc_dropout``.
    """
    calibrated = logits / max(float(temp), 1e-8)
    probs = F.softmax(calibrated, dim=-1)
    top2 = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1)
    pred = top2.indices[:, 0]
    if not apply_gates:
        return pred, probs

    conf_score = top2.values[:, 0]
    margin_score = top2.values[:, 0] - top2.values[:, 1]
    var_score = probs.var(dim=-1)
    neutral = torch.full_like(pred, int(Label.NEUTRAL))
    keep = (conf_score >= conf) & (margin_score >= margin) & (var_score <= var)
    if conf <= 0 and margin <= 0 and var >= 1.0:
        return pred, probs
    pred = torch.where(keep, pred, neutral)
    return pred, probs


def predict_mc_dropout(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    temp: float = 1.0,
    conf: float = 0.4,
    margin: float = 0.3,
    var: float = 0.01,
    num_samples: int = 10,
    abstain_as_neutral: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Thesis ``classify_text`` — MC dropout mean/var + dynamic conf gate.

    Returns ``(pred_labels, mean_probs, var_probs)``. Abstentions become NEUTRAL
    when ``abstain_as_neutral`` (CLI-friendly); set False to get -1 like Thesis None.
    """
    was_training = bool(model.training)
    model.train()  # enable dropout
    probs_stack: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(max(int(num_samples), 1)):
            logits, _ = model(input_ids, attention_mask)
            calibrated = logits / max(float(temp), 1e-8)
            probs_stack.append(F.softmax(calibrated, dim=-1))
    if was_training:
        model.train()
    else:
        model.eval()

    stacked = torch.stack(probs_stack, dim=0)  # (S, B, C)
    mean_probs = stacked.mean(dim=0)
    var_probs = stacked.var(dim=0, unbiased=False)
    top2 = torch.topk(mean_probs, k=min(2, mean_probs.size(-1)), dim=-1)
    pred = top2.indices[:, 0].clone()
    max_prob = top2.values[:, 0]
    margin_score = top2.values[:, 0] - top2.values[:, 1]
    # Thesis classify_text: variance of the *winning* class only
    max_var = var_probs.gather(1, pred.unsqueeze(1)).squeeze(1)
    dynamic_threshold = float(conf) - 0.5 * max_var
    reject = (max_prob < dynamic_threshold) | (margin_score < margin) | (max_var > var)
    if abstain_as_neutral:
        pred = torch.where(reject, torch.full_like(pred, int(Label.NEUTRAL)), pred)
    else:
        pred = torch.where(reject, torch.full_like(pred, -1), pred)
    return pred, mean_probs, var_probs
