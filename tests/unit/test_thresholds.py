import torch

from fairai.domain.labels import Label
from fairai.infrastructure.torch_van.thresholds import predict_from_logits


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
