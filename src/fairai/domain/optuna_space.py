"""Optuna search spaces for fair matched HPO (VAN / CE / −BPL)."""

from __future__ import annotations

from typing import Any, Literal

Family = Literal["van", "ce", "no_bpl"]

FAMILIES: tuple[Family, ...] = ("van", "ce", "no_bpl")


def suggest_hyperparams(trial: Any, family: Family) -> dict[str, Any]:
    """Sample hyperparameters aligned with thesis VAN Optuna keys."""
    params: dict[str, Any] = {
        "gen_dropout": trial.suggest_float("gen_dropout", 0.0, 0.5),
        "gen_dropout_reg": trial.suggest_float("gen_dropout_reg", 0.0, 0.8),
        "att_dropout": trial.suggest_float("att_dropout", 0.0, 0.5),
        "gen_num_layers": trial.suggest_int("gen_num_layers", 2, 8),
        "gen_hidden_size": trial.suggest_categorical(
            "gen_hidden_size", [512, 768, 1024, 1536, 2048]
        ),
        "multihead_num_heads": trial.suggest_categorical(
            "multihead_num_heads", [4, 8]
        ),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 3e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "warmup_pct": trial.suggest_float("warmup_pct", 0.0, 0.4),
        "clip_grad": trial.suggest_float("clip_grad", 0.1, 2.0),
        "scheduler": trial.suggest_categorical("scheduler", ["cosine", "linear"]),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15),
        "alpha": trial.suggest_float("alpha", 0.5, 2.5),
        "gamma": trial.suggest_float("gamma", 0.5, 3.0),
    }

    if family == "van":
        params.update(
            {
                "use_bias_projection": True,
                "bias_dim": trial.suggest_categorical(
                    "bias_dim", [256, 512, 768, 896, 1024]
                ),
                "bias_num_layers": trial.suggest_int("bias_num_layers", 1, 2),
                "bias_dropout": trial.suggest_float("bias_dropout", 0.0, 0.5),
                "bias_lambda": trial.suggest_float("bias_lambda", 0.5, 3.0),
                "penalty_stereo": trial.suggest_float("penalty_stereo", 0.5, 5.0),
                "penalty_anti": trial.suggest_float("penalty_anti", 0.5, 5.0),
                "penalty_neutral": trial.suggest_float("penalty_neutral", 0.5, 5.0),
                "loss_kind": "bias_aware",
                "use_class_weights": True,
                "weighted_sampler": True,
            }
        )
    elif family == "no_bpl":
        params.update(
            {
                "use_bias_projection": False,
                "penalty_stereo": trial.suggest_float("penalty_stereo", 0.5, 5.0),
                "penalty_anti": trial.suggest_float("penalty_anti", 0.5, 5.0),
                "penalty_neutral": trial.suggest_float("penalty_neutral", 0.5, 5.0),
                "loss_kind": "bias_aware",
                "use_class_weights": True,
                "weighted_sampler": True,
            }
        )
    else:  # ce
        params.update(
            {
                "use_bias_projection": False,
                "penalty_stereo": 1.0,
                "penalty_anti": 1.0,
                "penalty_neutral": 1.0,
                "loss_kind": "ce",
                "use_class_weights": False,
                "weighted_sampler": False,
                # CE still uses focal-ish alpha/gamma only if loss supports; keep for parity
                # of optim/arch search; loss kind ce ignores penalties.
            }
        )
    return params


def apply_params_to_config(base: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy-ish overlay of sampled params onto a loaded experiment config."""
    cfg = {
        **base,
        "architecture": dict(base.get("architecture", {})),
        "loss": dict(base.get("loss", {})),
        "optim": dict(base.get("optim", {})),
        "train": dict(base.get("train", {})),
    }
    arch = cfg["architecture"]
    loss = cfg["loss"]
    optim = cfg["optim"]
    train = cfg["train"]

    arch["gen_dropout"] = float(params["gen_dropout"])
    arch["gen_dropout_reg"] = float(params["gen_dropout_reg"])
    arch["att_dropout"] = float(params["att_dropout"])
    arch["gen_num_layers"] = int(params["gen_num_layers"])
    arch["gen_hidden_size"] = int(params["gen_hidden_size"])
    arch["multihead_num_heads"] = int(params["multihead_num_heads"])
    arch["use_bias_projection"] = bool(params["use_bias_projection"])
    if params.get("use_bias_projection"):
        arch["bias_dim"] = int(params["bias_dim"])
        arch["bias_num_layers"] = int(params["bias_num_layers"])
        arch["bias_dropout"] = float(params["bias_dropout"])
        arch["bias_lambda"] = float(params["bias_lambda"])

    loss["kind"] = str(params["loss_kind"])
    loss["alpha"] = float(params["alpha"])
    loss["gamma"] = float(params["gamma"])
    loss["label_smoothing"] = float(params["label_smoothing"])
    loss["penalty_stereo"] = float(params["penalty_stereo"])
    loss["penalty_anti"] = float(params["penalty_anti"])
    loss["penalty_neutral"] = float(params["penalty_neutral"])
    loss["use_class_weights"] = bool(params["use_class_weights"])

    optim["learning_rate"] = float(params["learning_rate"])
    optim["weight_decay"] = float(params["weight_decay"])
    optim["warmup_pct"] = float(params["warmup_pct"])
    optim["clip_grad"] = float(params["clip_grad"])
    optim["scheduler"] = str(params["scheduler"])

    train["weighted_sampler"] = bool(params["weighted_sampler"])
    return cfg
