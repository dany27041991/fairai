import torch
import torch.nn as nn

from fairai.domain.labels import Label
from fairai.infrastructure.torch_van.thresholds import predict_from_logits, predict_mc_dropout


def test_argmax_without_gates() -> None:
    logits = torch.tensor([[0.1, 5.0, 0.2], [9.0, 0.0, 0.0]])
    pred, probs = predict_from_logits(logits, apply_gates=False)
    assert pred.tolist() == [1, 0]
    assert probs.shape == (2, 3)


def test_gates_force_neutral_on_flat_distribution() -> None:
    logits = torch.zeros(1, 3)
    pred, _ = predict_from_logits(
        logits, temp=1.0, conf=0.9, margin=0.5, var=0.01, apply_gates=True
    )
    assert int(pred.item()) == int(Label.NEUTRAL)


class _StubVan(nn.Module):
    """Minimal stand-in: forward(input_ids, attention_mask) -> (logits, None)."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self._logits = logits
        self.drop = nn.Dropout(0.0)

    def forward(self, input_ids, attention_mask):  # noqa: ANN001
        _ = self.drop(input_ids.float())
        return self._logits, None


def test_mc_dropout_dynamic_gate_abstain() -> None:
    # Flat logits → reject under high conf → -1 when abstain_as_neutral=False
    model = _StubVan(torch.zeros(1, 3))
    ids = torch.zeros(1, 4, dtype=torch.long)
    mask = torch.ones(1, 4, dtype=torch.long)
    pred, mean_p, var_p = predict_mc_dropout(
        model,
        ids,
        mask,
        temp=1.0,
        conf=0.9,
        margin=0.5,
        var=0.01,
        num_samples=5,
        abstain_as_neutral=False,
    )
    assert pred.shape == (1,)
    assert mean_p.shape == (1, 3)
    assert var_p.shape == (1, 3)
    assert int(pred.item()) == -1


def test_composite_metric_imported() -> None:
    from fairai.domain.analysis_stats import composite_metric

    # accuracy 1.0 on 2 kept × coverage 2/3
    assert abs(composite_metric([0, 1, 2], [0, 1, -1], null_value=-1) - (2 / 3)) < 1e-9
