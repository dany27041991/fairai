"""Optuna HPO runner: equal-budget TPE search with WHM(train-iCAT) objective."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from fairai.application.config_merge import load_experiment_config
from fairai.domain.optuna_space import (
    FAMILIES,
    Family,
    apply_params_to_config,
    suggest_hyperparams,
)
from fairai.infrastructure.artifact_paths import ArtifactPaths
from fairai.infrastructure.torch_van.checkpointing import write_json


def _study_dir(paths: ArtifactPaths, study_name: str) -> Path:
    d = paths.root / "optuna" / study_name
    d.mkdir(parents=True, exist_ok=True)
    (d / "trials").mkdir(exist_ok=True)
    return d


def _write_trial_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Drop inherits so the trial file is self-contained.
    payload = {k: v for k, v in cfg.items() if k != "inherits"}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run_optuna_hpo(
    *,
    family: Family,
    base_config: Path,
    n_trials: int = 100,
    study_name: str | None = None,
    artifacts: ArtifactPaths | None = None,
    device: str = "cuda",
    max_samples: int | None = None,
    trial_epochs: int | None = 20,
    trial_patience: int | None = 3,
    seed: int = 42,
    storage: str | None = None,
) -> dict[str, Any]:
    """Run TPE+MedianPruner HPO; persist study under artifacts/optuna/<name>/."""
    try:
        import optuna
        from optuna.exceptions import TrialPruned
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "optuna is required for HPO: pip install 'optuna>=3.6'"
        ) from exc

    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")

    paths = artifacts or ArtifactPaths.from_env()
    name = study_name or f"optuna_fair_{family}"
    out_dir = _study_dir(paths, name)
    base = load_experiment_config(base_config)

    # Ensure matched WHM protocol (anti-peek).
    train = dict(base.get("train", {}))
    train["selection_metric"] = "combined"
    train["f1_weight"] = float(train.get("f1_weight", 1.88))
    train["icat_subset"] = "train"
    train["icat_every"] = int(train.get("icat_every", 2))
    train.setdefault("stereoset_splits", "splits/stereoset")
    train.setdefault("dataset_train", "processed/van_split_seed42_train.jsonl")
    train.setdefault("dataset_val", "processed/van_split_seed42_val.jsonl")
    base["train"] = train

    db_path = out_dir / "study.db"
    storage_url = storage or f"sqlite:///{db_path.resolve()}"
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(
        study_name=name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial: optuna.Trial) -> float:
        from fairai.application.train_van import train_van

        params = suggest_hyperparams(trial, family)
        cfg = apply_params_to_config(base, params)
        cfg["name"] = f"{name}_trial{trial.number}"
        cfg["device"] = device
        tcfg = cfg["train"]
        if trial_epochs is not None:
            tcfg["epochs"] = int(trial_epochs)
        if trial_patience is not None:
            tcfg["patience"] = int(trial_patience)
        tcfg["seed"] = seed
        tcfg["amp"] = bool(tcfg.get("amp", True)) and str(device).startswith("cuda")

        trial_yaml = out_dir / "trials" / f"trial_{trial.number:04d}.yaml"
        _write_trial_yaml(trial_yaml, cfg)
        write_json(out_dir / "trials" / f"trial_{trial.number:04d}_params.json", params)

        def on_epoch(row: dict[str, float]) -> None:
            score = float(row.get("combined", row.get("val_f1_macro", 0.0)))
            trial.report(score, step=int(row["epoch"]))
            if trial.should_prune():
                raise TrialPruned()

        try:
            result = train_van(
                config_path=trial_yaml,
                artifacts=paths,
                device=device,
                max_samples=max_samples,
                run_id=f"{name}-t{trial.number}",
                seed=seed,
                on_epoch=on_epoch,
            )
        except TrialPruned:
            raise

        value = float(
            result.best_combined
            if result.best_combined is not None
            else result.best_val_f1
        )
        trial.set_user_attr("best_val_f1", result.best_val_f1)
        trial.set_user_attr("best_avg_icat", result.best_avg_icat)
        trial.set_user_attr("run_dir", result.run_dir)
        trial.set_user_attr("epochs_ran", result.epochs_ran)
        return value

    study.optimize(objective, n_trials=int(n_trials), catch=(Exception,))

    complete = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
    if not complete:
        raise RuntimeError(
            f"Optuna study {name!r}: no COMPLETE trials "
            f"(n={len(study.trials)}). Check CUDA/torch and trial logs."
        )
    best = study.best_trial
    params_path = out_dir / "trials" / f"trial_{best.number:04d}_params.json"
    if not params_path.is_file():
        raise FileNotFoundError(f"best trial params missing: {params_path}")
    full_params = json.loads(params_path.read_text(encoding="utf-8"))

    best_cfg = apply_params_to_config(base, full_params)
    best_cfg["name"] = f"optuna_fair_{family}_best"
    # Final train uses full epochs/patience from rigorous base / van_v2
    best_cfg["train"]["epochs"] = int(base.get("train", {}).get("epochs", 100))
    best_cfg["train"]["patience"] = int(base.get("train", {}).get("patience", 5))
    best_yaml = out_dir / "best_config.yaml"
    _write_trial_yaml(best_yaml, best_cfg)
    write_json(out_dir / "best_trial_params.json", full_params)

    summary = {
        "study_name": name,
        "family": family,
        "n_trials": len(study.trials),
        "best_trial": best.number,
        "best_value": float(best.value) if best.value is not None else None,
        "best_val_f1": best.user_attrs.get("best_val_f1"),
        "best_avg_icat": best.user_attrs.get("best_avg_icat"),
        "best_run_dir": best.user_attrs.get("run_dir"),
        "best_config": str(best_yaml),
        "best_params_path": str(out_dir / "best_trial_params.json"),
        "storage": storage_url,
        "protocol": {
            "selection_metric": "combined",
            "icat_subset": "train",
            "f1_weight": 1.88,
            "trial_epochs": trial_epochs,
            "trial_patience": trial_patience,
            "max_samples": max_samples,
            "report": "val F1 + StereoSet test iCAT after final seeds",
            "note": "HPO may use max_samples subsample; final suite trains on full split",
        },
    }
    write_json(out_dir / "study_summary.json", summary)
    return summary


def write_finalize_suite(
    *,
    artifacts: ArtifactPaths | None = None,
    families: tuple[Family, ...] = ("van", "ce", "no_bpl"),
    seeds: tuple[int, ...] = (42, 123, 7),
    out_suite: Path | None = None,
    study_prefix: str = "optuna_fair",
    config_prefix: str = "optuna_fair",
    run_id_prefix: str = "opta",
    eval_dataset: str = "processed/stereoset/f1_test_leakfree.jsonl",
    eval_out_dir: str = "results/metrics/eval_reports_opta",
    out_metrics: str = "results/metrics/optuna_fair_compare.json",
    suite_name: str | None = None,
) -> Path:
    """Copy each family's best_config.yaml and write a finalize suite YAML."""
    paths = artifacts or ArtifactPaths.from_env()
    experiments: list[dict[str, Any]] = []
    for fam in families:
        best = paths.root / "optuna" / f"{study_prefix}_{fam}" / "best_config.yaml"
        if not best.is_file():
            raise FileNotFoundError(f"missing best config for {fam}: {best}")
        repo_cfg = Path.cwd() / "configs" / "experiments" / f"{config_prefix}_{fam}_best.yaml"
        repo_cfg.parent.mkdir(parents=True, exist_ok=True)
        repo_cfg.write_text(best.read_text(encoding="utf-8"), encoding="utf-8")
        for seed in seeds:
            experiments.append(
                {
                    "config": f"experiments/{config_prefix}_{fam}_best.yaml",
                    "run_id": f"{run_id_prefix}-{fam}-s{seed}",
                    "seed": seed,
                }
            )

    name = suite_name or f"suite_{study_prefix}"
    suite = {
        "name": name,
        "device": "cuda",
        "aggregate": True,
        "experiments": experiments,
        "eval": {
            "dataset": eval_dataset,
            "batch_size": 64,
            "subset": "test",
            "stereoset_splits": "splits/stereoset",
            "eval_out_dir": eval_out_dir,
        },
        "out": out_metrics,
    }
    target = out_suite or (Path.cwd() / "configs/experiments" / f"{name}.yaml")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")
    return target
