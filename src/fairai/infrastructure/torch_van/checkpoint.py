"""Load VAN checkpoints and YAML-aligned architecture configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from fairai.infrastructure.torch_van.van_classifier import VanArchConfig, VanClassifier


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def arch_config_from_yaml(data: dict[str, Any]) -> VanArchConfig:
    arch = data.get("architecture", {})
    return VanArchConfig(
        backbone=str(data.get("backbone", VanArchConfig.backbone)),
        num_labels=int(data.get("num_labels", 3)),
        dropout=float(arch.get("gen_dropout", 0.3)),
        dropout_reg=float(arch.get("gen_dropout_reg", 0.5)),
        num_layers=int(arch.get("gen_num_layers", 2)),
        hidden_size=int(arch.get("gen_hidden_size", 512)),
        att_dropout=float(arch.get("att_dropout", 0.1)),
        multihead_num_heads=int(arch.get("multihead_num_heads", 1)),
        bias_dim=int(arch.get("bias_dim", 256)),
        bias_num_layers=int(arch.get("bias_num_layers", 1)),
        bias_dropout=float(arch.get("bias_dropout", 0.0)),
        bias_lambda=float(arch.get("bias_lambda", 1.0)),
    )


def thresholds_from_yaml(data: dict[str, Any]) -> dict[str, float]:
    thr = data.get("thresholds", {})
    return {
        "conf": float(thr.get("conf", 0.0)),
        "margin": float(thr.get("margin", 0.0)),
        "var": float(thr.get("var", 0.0)),
        "temp": float(thr.get("temp", 1.0)),
    }


def load_van_classifier(
    checkpoint_path: Path,
    config_path: Path,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[VanClassifier, dict[str, float], dict[str, Any]]:
    raw = load_yaml(config_path)
    arch = arch_config_from_yaml(raw)
    model = VanClassifier(arch, init_backbone_weights=False)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(state)}")
    # AS-IS saves a raw state_dict OrderedDict
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"load_state_dict issues missing={missing} unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model, thresholds_from_yaml(raw), raw
