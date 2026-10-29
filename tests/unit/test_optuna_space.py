"""Unit tests for Optuna search-space helpers (no GPU)."""

from __future__ import annotations

from fairai.domain.optuna_space import apply_params_to_config, suggest_hyperparams


class _FakeTrial:
    def __init__(self) -> None:
        self._i = 0

    def suggest_float(self, name: str, low: float, high: float, **kwargs):  # noqa: ANN003
        self._i += 1
        return float(low + (high - low) * 0.25)

    def suggest_int(self, name: str, low: int, high: int, **kwargs):  # noqa: ANN003
        return int(low)

    def suggest_categorical(self, name: str, choices):  # noqa: ANN001
        return choices[0]


def test_suggest_van_includes_bpl() -> None:
    p = suggest_hyperparams(_FakeTrial(), "van")
    assert p["use_bias_projection"] is True
    assert "bias_dim" in p
    assert p["loss_kind"] == "bias_aware"


def test_suggest_ce_disables_weights() -> None:
    p = suggest_hyperparams(_FakeTrial(), "ce")
    assert p["use_bias_projection"] is False
    assert p["loss_kind"] == "ce"
    assert p["use_class_weights"] is False
    assert p["weighted_sampler"] is False


def test_apply_params_sets_optim() -> None:
    base = {
        "architecture": {},
        "loss": {},
        "optim": {},
        "train": {},
    }
    p = suggest_hyperparams(_FakeTrial(), "no_bpl")
    cfg = apply_params_to_config(base, p)
    assert cfg["architecture"]["use_bias_projection"] is False
    assert "learning_rate" in cfg["optim"]
    assert cfg["train"]["weighted_sampler"] is True
