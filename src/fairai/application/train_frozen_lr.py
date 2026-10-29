"""Baseline: frozen MiniLM embeddings + LogisticRegression."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from fairai.application.config_merge import load_experiment_config
from fairai.infrastructure.artifact_paths import ArtifactPaths
from fairai.infrastructure.baselines.frozen_encoder import encode_texts
from fairai.infrastructure.data.jsonl_repo import load_examples
from fairai.infrastructure.torch_van.checkpointing import write_json


@dataclass(slots=True)
class FrozenLrResult:
    run_dir: str
    best_val_f1: float
    metrics_path: str
    model_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_frozen_lr(
    *,
    config_path: Path,
    artifacts: ArtifactPaths | None = None,
    device: str | None = None,
    max_samples: int | None = None,
    run_id: str | None = None,
    val_fraction: float = 0.1,
    seed: int | None = None,
) -> FrozenLrResult:
    cfg = load_experiment_config(config_path)
    paths = artifacts or ArtifactPaths.from_env()
    train_cfg = dict(cfg.get("train", {}))
    lr_cfg = dict(cfg.get("logistic", {}))

    device = device or str(cfg.get("device", "cpu"))
    seed_i = int(seed if seed is not None else train_cfg.get("seed", 42))
    backbone = str(cfg.get("backbone", "sentence-transformers/all-MiniLM-L6-v2"))
    max_len = int(cfg.get("max_len", 512))
    batch_size = int(train_cfg.get("batch_size", 64))

    def _resolve(rel: str) -> Path:
        p = paths.data_dir / rel
        if p.is_file():
            return p
        alt = Path(rel)
        return alt if alt.is_file() else paths.processed(Path(rel).name)

    limit = max_samples if max_samples is not None else train_cfg.get("max_samples")
    limit_i = int(limit) if limit is not None else None
    train_rel = train_cfg.get("dataset_train")
    val_rel = train_cfg.get("dataset_val")
    if train_rel and val_rel:
        dataset_path = _resolve(str(train_rel))
        train_ex = load_examples(dataset_path, max_samples=limit_i)
        val_ex = load_examples(_resolve(str(val_rel)), max_samples=None)
    else:
        dataset_rel = str(train_cfg.get("dataset", "processed/van_train_leakage_safe.jsonl"))
        dataset_path = _resolve(dataset_rel)
        examples = load_examples(dataset_path, max_samples=limit_i)
        if len(examples) < 4:
            raise ValueError(f"Need ≥4 examples, got {len(examples)}")
        labels = [int(ex.label) for ex in examples]
        train_ex, val_ex = train_test_split(
            examples,
            test_size=val_fraction,
            random_state=seed_i,
            stratify=labels if len(set(labels)) > 1 else None,
        )
    if len(train_ex) < 2 or len(val_ex) < 1:
        raise ValueError(f"Split too small: train={len(train_ex)} val={len(val_ex)}")

    x_train = encode_texts(
        [ex.sentence for ex in train_ex],
        backbone=backbone,
        device=device,
        max_len=max_len,
        batch_size=batch_size,
    )
    y_train = np.asarray([int(ex.label) for ex in train_ex], dtype=np.int64)
    x_val = encode_texts(
        [ex.sentence for ex in val_ex],
        backbone=backbone,
        device=device,
        max_len=max_len,
        batch_size=batch_size,
    )
    y_val = np.asarray([int(ex.label) for ex in val_ex], dtype=np.int64)

    clf = LogisticRegression(
        max_iter=int(lr_cfg.get("max_iter", 2000)),
        C=float(lr_cfg.get("C", 1.0)),
        class_weight=lr_cfg.get("class_weight", "balanced"),
        solver=str(lr_cfg.get("solver", "lbfgs")),
        random_state=seed_i,
        n_jobs=int(lr_cfg.get("n_jobs", -1)),
    )
    clf.fit(x_train, y_train)
    val_pred = clf.predict(x_val)
    val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))

    rid = run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = paths.model_dir / "runs" / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "logistic.joblib"
    joblib.dump(
        {
            "classifier": clf,
            "backbone": backbone,
            "max_len": max_len,
            "normalize": True,
            "num_labels": int(cfg.get("num_labels", 3)),
        },
        model_path,
    )

    metrics = {
        "run_id": rid,
        "config": str(config_path),
        "experiment": str(cfg.get("name", "frozen_lr")),
        "kind": "frozen_sbert_lr",
        "dataset": str(dataset_path),
        "n_train": len(train_ex),
        "n_val": len(val_ex),
        "seed": seed_i,
        "best_val_f1_macro": val_f1,
        "device": device,
        "ablation": {
            "use_bias_projection": False,
            "loss_kind": "logistic",
            "use_class_weights": lr_cfg.get("class_weight") is not None,
            "weighted_sampler": False,
            "freeze_backbone": True,
        },
    }
    metrics_path = run_dir / "metrics.json"
    write_json(metrics_path, metrics)
    write_json(run_dir / "config.snapshot.json", cfg)

    return FrozenLrResult(
        run_dir=str(run_dir),
        best_val_f1=val_f1,
        metrics_path=str(metrics_path),
        model_path=str(model_path),
    )
