"""Baselines / ablation config + architecture flags."""

from pathlib import Path

import torch

from fairai.application.config_merge import load_experiment_config
from fairai.infrastructure.torch_van.checkpoint import arch_config_from_yaml
from fairai.infrastructure.torch_van.losses import build_train_criterion
from fairai.infrastructure.torch_van.van_classifier import VanArchConfig, VanClassifier

ROOT = Path(__file__).resolve().parents[2]


def test_baseline_ce_config_flags() -> None:
    cfg = load_experiment_config(ROOT / "configs/experiments/baseline_ce_smoke.yaml")
    assert cfg["loss"]["kind"] == "ce"
    assert cfg["loss"]["use_class_weights"] is False
    assert cfg["architecture"]["use_bias_projection"] is False
    assert cfg["train"]["weighted_sampler"] is False


def test_ablation_configs_load() -> None:
    nobpl = load_experiment_config(
        ROOT / "configs/experiments/ablation_no_bias_proj_smoke.yaml"
    )
    assert nobpl["architecture"]["use_bias_projection"] is False
    assert "gamma" in nobpl["loss"]
    assert nobpl["loss"].get("kind", "bias_aware") == "bias_aware"

    nocw = load_experiment_config(
        ROOT / "configs/experiments/ablation_no_class_weights_smoke.yaml"
    )
    assert nocw["loss"]["use_class_weights"] is False
    assert nocw["loss"]["penalty_stereo"] == 1.0
    assert nocw["train"]["weighted_sampler"] is False


def test_frozen_lr_config() -> None:
    cfg = load_experiment_config(
        ROOT / "configs/experiments/baseline_frozen_lr_smoke.yaml"
    )
    assert cfg["experiment_kind"] == "frozen_lr"
    assert cfg["logistic"]["class_weight"] == "balanced"


def test_build_train_criterion_ce_and_bias_aware() -> None:
    labels = [0, 0, 1, 1, 2, 2]
    ce = build_train_criterion(
        {"kind": "ce", "use_class_weights": False},
        num_labels=3,
        labels_for_weights=labels,
    )
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 0])
    assert torch.isfinite(ce(logits, targets))

    ba = build_train_criterion(
        {
            "kind": "bias_aware",
            "use_class_weights": True,
            "alpha": 1.0,
            "gamma": 1.0,
            "penalty_stereo": 2.0,
            "penalty_anti": 1.0,
            "penalty_neutral": 1.0,
        },
        num_labels=3,
        labels_for_weights=labels,
    )
    assert torch.isfinite(ba(logits, targets))


def test_van_without_bias_projection_forward() -> None:
    arch = VanArchConfig(use_bias_projection=False, multihead_num_heads=1, num_layers=1)
    model = VanClassifier(arch, init_backbone_weights=False)
    assert not any("bias_layer.projection" in k for k in model.state_dict())
    ids = torch.randint(0, 100, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)
    logits, pooled = model(ids, mask)
    assert logits.shape == (2, 3)
    assert pooled.shape[0] == 2


def test_arch_config_reads_use_bias_projection() -> None:
    cfg = load_experiment_config(ROOT / "configs/experiments/baseline_ce_smoke.yaml")
    arch = arch_config_from_yaml(cfg)
    assert arch.use_bias_projection is False
