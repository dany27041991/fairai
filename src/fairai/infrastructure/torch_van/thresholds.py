"""Post-hoc decision rules using temperature + conf/margin/var thresholds."""

from __future__ import annotations

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
    predictions are forced to NEUTRAL (label 2), matching the Thesis calibration
    spirit. Exact Thesis composite may be refined in M3; this gate is conservative.
    """
    calibrated = logits / max(float(temp), 1e-8)
    probs = F.softmax(calibrated, dim=-1)
    top2 = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1)
    pred = top2.indices[:, 0]
    if not apply_gates:
        return pred, probs

    conf_score = top2.values[:, 0]
    margin_score = top2.values[:, 0] - top2.values[:, 1]
    # variance across class probabilities
    var_score = probs.var(dim=-1)
    neutral = torch.full_like(pred, int(Label.NEUTRAL))
    keep = (conf_score >= conf) & (margin_score >= margin) & (var_score <= var)
    # If thresholds are near-zero / unused, keep argmax
    if conf <= 0 and margin <= 0 and var >= 1.0:
        return pred, probs
    pred = torch.where(keep, pred, neutral)
    return pred, probs
