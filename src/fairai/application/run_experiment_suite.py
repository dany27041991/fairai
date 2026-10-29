"""Run a YAML suite of baseline/ablation experiments + compare metrics."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fairai.application.config_merge import load_experiment_config
from fairai.application.evaluate_experiment import compare_experiment_runs
from fairai.infrastructure.artifact_paths import ArtifactPaths
from fairai.infrastructure.torch_van.checkpoint import load_yaml
from fairai.infrastructure.torch_van.checkpointing import write_json


def _resolve_under(root: Path, rel: str | None) -> Path | None:
    if rel is None or rel == "" or str(rel).lower() in {"null", "none", "~"}:
        return None
    path = Path(str(rel))
    if path.is_file() or path.is_dir():
        return path
    return root / path


def run_experiment_suite(
    *,
    suite_path: Path,
    artifacts: ArtifactPaths | None = None,
    device: str | None = None,
    skip_train: bool = False,
) -> dict[str, Any]:
    suite = load_yaml(suite_path)
    paths = artifacts or ArtifactPaths.from_env()
    configs_root = suite_path.parent.parent  # configs/experiments/x → configs/
    device = device or str(suite.get("device", "cpu"))
    eval_cfg = dict(suite.get("eval", {}))
    max_samples = eval_cfg.get("max_samples")
    max_samples_i = int(max_samples) if max_samples is not None else None

    eval_dataset = _resolve_under(
        paths.data_dir, eval_cfg.get("dataset")
    ) or _resolve_under(Path.cwd(), eval_cfg.get("dataset"))
    splits_dir = _resolve_under(
        paths.data_dir, eval_cfg.get("stereoset_splits")
    ) or _resolve_under(Path.cwd(), eval_cfg.get("stereoset_splits"))

    run_dirs: list[Path] = []
    train_results: list[dict[str, Any]] = []

    for entry in suite.get("experiments", []):
        if isinstance(entry, str):
            entry = {"config": entry}
        cfg_rel = str(entry["config"])
        cfg_path = Path(cfg_rel)
        if not cfg_path.is_file():
            cfg_path = (configs_root / cfg_rel).resolve()
            if not cfg_path.is_file():
                # suite under configs/experiments → sibling file
                cfg_path = (suite_path.parent / Path(cfg_rel).name).resolve()

        run_id = entry.get("run_id") or f"{Path(cfg_rel).stem}-{time.strftime('%Y%m%d-%H%M%S')}"
        existing = entry.get("run_dir")
        if skip_train or existing:
            run_dir = Path(existing) if existing else paths.model_dir / "runs" / run_id
            if not run_dir.is_dir():
                raise FileNotFoundError(f"run_dir missing for skip-train: {run_dir}")
            run_dirs.append(run_dir)
            continue

        cfg = load_experiment_config(cfg_path, configs_root=configs_root)
        kind = str(cfg.get("experiment_kind", cfg.get("kind", "van"))).lower()
        train_max = entry.get("max_samples", suite.get("train_max_samples"))
        train_max_i = int(train_max) if train_max is not None else None

        entry_seed = entry.get("seed")
        seed_i = int(entry_seed) if entry_seed is not None else None

        if kind in {"frozen_lr", "frozen_sbert_lr", "logistic"}:
            from fairai.application.train_frozen_lr import train_frozen_lr

            result = train_frozen_lr(
                config_path=cfg_path,
                artifacts=paths,
                device=device,
                max_samples=train_max_i,
                run_id=run_id,
                seed=seed_i,
            )
            train_results.append(result.to_dict())
            run_dirs.append(Path(result.run_dir))
        else:
            from fairai.application.train_van import train_van

            result = train_van(
                config_path=cfg_path,
                artifacts=paths,
                device=device,
                max_samples=train_max_i,
                run_id=run_id,
                seed=seed_i,
            )
            train_results.append(result.to_dict())
            run_dirs.append(Path(result.run_dir))

    out_rel = str(suite.get("out", "results/metrics/baselines_compare.json"))
    out_path = Path(out_rel) if Path(out_rel).is_absolute() else paths.root / out_rel

    eval_out = _resolve_under(paths.root, eval_cfg.get("eval_out_dir"))
    if eval_out is None:
        eval_out = paths.root / "results" / "metrics" / "eval_reports"
    table = compare_experiment_runs(
        run_dirs=run_dirs,
        eval_dataset=eval_dataset if eval_dataset and eval_dataset.exists() else None,
        splits_dir=splits_dir if splits_dir and Path(splits_dir).exists() else None,
        device=device,
        batch_size=int(eval_cfg.get("batch_size", 32)),
        max_samples=max_samples_i,
        subset=str(eval_cfg.get("subset", "all")),
        out_path=out_path,
        eval_out_dir=eval_out,
    )
    payload: dict[str, Any] = {
        "suite": str(suite_path),
        "device": device,
        "run_dirs": [str(p) for p in run_dirs],
        "train_results": train_results,
        "compare": table,
        "compare_path": str(out_path),
    }
    if bool(suite.get("aggregate", False)):
        from fairai.application.aggregate_rigorous import aggregate_compare_table

        payload["aggregate"] = aggregate_compare_table(out_path)
    write_json(out_path.parent / "suite_last.json", payload)
    return payload
