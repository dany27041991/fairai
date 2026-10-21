import torch

from fairai.infrastructure.torch_van.losses import BiasAwareLoss, FocalLoss, build_bias_aware_loss


def test_focal_loss_shapes() -> None:
    crit = FocalLoss(alpha=1.5, gamma=2.0, num_labels=3, label_smoothing=0.05)
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 0])
    out = crit(logits, targets)
    assert out.shape == (4,)


def test_bias_aware_mean_scalar() -> None:
    loss = build_bias_aware_loss(
        {
            "alpha": 1.0,
            "gamma": 1.0,
            "label_smoothing": 0.0,
            "penalty_stereo": 2.0,
            "penalty_anti": 1.5,
            "penalty_neutral": 1.0,
        }
    )
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    value = loss(logits, targets)
    assert value.ndim == 0
    assert torch.isfinite(value)


def test_focal_as_bias_base() -> None:
    base = FocalLoss(reduction="none")
    wrap = BiasAwareLoss(base, penalty_stereo=3.0)
    logits = torch.zeros(2, 3)
    targets = torch.tensor([0, 2])
    assert torch.isfinite(wrap(logits, targets))
