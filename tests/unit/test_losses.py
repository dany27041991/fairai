import torch

from fairai.infrastructure.torch_van.losses import (
    BiasAwareLoss,
    FocalLoss,
    build_bias_aware_loss,
    build_train_criterion,
    compute_class_weights,
)


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


def test_compute_class_weights_and_wiring() -> None:
    # Balanced 2/2/2 → each weight = 1.0
    w = compute_class_weights([0, 0, 1, 1, 2, 2], num_classes=3)
    assert torch.allclose(w, torch.ones(3))
    # Thesis imbalance: rare class gets higher weight
    w2 = compute_class_weights([0, 0, 0, 0, 1, 2], num_classes=3)
    assert float(w2[1]) > float(w2[0])
    loss = build_bias_aware_loss({"alpha": 1.0, "gamma": 1.0}, class_weights=w2)
    assert loss.class_weights is not None
    value = loss(torch.randn(4, 3), torch.tensor([0, 1, 2, 0]))
    assert torch.isfinite(value)


def test_build_train_criterion_disables_class_weights() -> None:
    crit = build_train_criterion(
        {"kind": "bias_aware", "use_class_weights": False, "alpha": 1.0, "gamma": 1.0},
        num_labels=3,
        labels_for_weights=[0, 1, 2],
    )
    assert isinstance(crit, BiasAwareLoss)
    assert crit.class_weights is None
