"""Use-case: train VAN from YAML (smoke subset or full run)."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from fairai.application.config_merge import load_experiment_config
from fairai.domain.example import Example
from fairai.domain.icat import compute_combined_score
from fairai.infrastructure.artifact_paths import ArtifactPaths
from fairai.infrastructure.data.jsonl_repo import load_examples
from fairai.infrastructure.torch_van.checkpoint import (
    arch_config_from_yaml,
    thresholds_from_yaml,
)
from fairai.infrastructure.torch_van.checkpointing import save_state_dict, write_json
from fairai.infrastructure.torch_van.encoded_dataset import EncodedDataset
from fairai.infrastructure.torch_van.icat_probe import probe_avg_icat
from fairai.infrastructure.torch_van.losses import build_train_criterion
from fairai.infrastructure.torch_van.optim_factory import build_optimizer, build_scheduler
from fairai.infrastructure.torch_van.train_loop import evaluate_f1, train_one_epoch
from fairai.infrastructure.torch_van.van_classifier import VanClassifier

EpochCallback = Callable[[dict[str, float]], None]


@dataclass(slots=True)
class TrainResult:
    run_dir: str
    best_val_f1: float
    epochs_ran: int
    metrics_path: str
    best_checkpoint: str
    best_combined: float | None = None
    best_avg_icat: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_data(paths: ArtifactPaths, rel: str) -> Path:
    dataset_path = paths.data_dir / rel
    if dataset_path.is_file():
        return dataset_path
    alt = Path(rel)
    if alt.is_file():
        return alt
    return paths.processed(Path(rel).name)


def _load_split(
    *,
    paths: ArtifactPaths,
    train_cfg: dict[str, Any],
    dataset_rel: str,
    max_samples: int | None,
    val_fraction: float,
    seed: int,
) -> tuple[list[Example], list[Example], Path]:
    train_rel = train_cfg.get("dataset_train")
    val_rel = train_cfg.get("dataset_val")
    if train_rel and val_rel:
        train_path = _resolve_data(paths, str(train_rel))
        val_path = _resolve_data(paths, str(val_rel))
        train_ex = load_examples(train_path, max_samples=max_samples)
        val_ex = load_examples(val_path, max_samples=None)
        if len(train_ex) < 2 or len(val_ex) < 1:
            raise ValueError(f"Fixed split too small: train={len(train_ex)} val={len(val_ex)}")
        return train_ex, val_ex, train_path

    dataset_path = _resolve_data(paths, dataset_rel)
    examples = load_examples(dataset_path, max_samples=max_samples)
    if len(examples) < 4:
        raise ValueError(f"Need ≥4 examples to train/val-split, got {len(examples)}")
    labels = [int(ex.label) for ex in examples]
    train_ex, val_ex = train_test_split(
        examples,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels if len(set(labels)) > 1 else None,
    )
    return train_ex, val_ex, dataset_path


def train_van(
    *,
    config_path: Path,
    artifacts: ArtifactPaths | None = None,
    device: str | None = None,
    max_samples: int | None = None,
    run_id: str | None = None,
    val_fraction: float = 0.1,
    seed: int | None = None,
    on_epoch: EpochCallback | None = None,
) -> TrainResult:
    cfg = load_experiment_config(config_path)
    paths = artifacts or ArtifactPaths.from_env()
    train_cfg = dict(cfg.get("train", {}))
    optim_cfg = dict(cfg.get("optim", {}))
    loss_cfg = dict(cfg.get("loss", {}))

    device = device or str(cfg.get("device", "cpu"))
    seed_i = int(seed if seed is not None else train_cfg.get("seed", 42))
    _set_seed(seed_i)
    train_cfg["seed"] = seed_i

    dataset_rel = str(train_cfg.get("dataset", "processed/van_train_leakage_safe.jsonl"))
    limit = max_samples if max_samples is not None else train_cfg.get("max_samples")
    limit_i = int(limit) if limit is not None else None
    train_ex, val_ex, dataset_path = _load_split(
        paths=paths,
        train_cfg=train_cfg,
        dataset_rel=dataset_rel,
        max_samples=limit_i,
        val_fraction=val_fraction,
        seed=seed_i,
    )
    backbone = str(cfg.get("backbone", "sentence-transformers/all-MiniLM-L6-v2"))
    max_len = int(cfg.get("max_len", 512))
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    batch_size = int(train_cfg.get("batch_size", 8))
    num_labels = int(cfg.get("num_labels", 3))
    train_labels = [int(ex.label) for ex in train_ex]
    use_weighted_sampler = bool(train_cfg.get("weighted_sampler", True))
    train_ds = EncodedDataset(train_ex, tokenizer, max_len)
    if use_weighted_sampler and train_labels:
        label_t = torch.tensor(train_labels, dtype=torch.long)
        class_counts = torch.bincount(label_t, minlength=num_labels).float().clamp_min(1.0)
        sample_w = 1.0 / class_counts[label_t]
        sampler = WeightedRandomSampler(
            weights=sample_w,
            num_samples=len(sample_w),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        EncodedDataset(val_ex, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=False,
    )

    arch = arch_config_from_yaml(cfg)
    model = VanClassifier(arch, init_backbone_weights=True).to(device)
    if bool(train_cfg.get("freeze_backbone", False)):
        for param in model.base_model.parameters():
            param.requires_grad = False

    criterion = build_train_criterion(
        loss_cfg, num_labels=num_labels, labels_for_weights=train_labels
    )
    if device != "cpu":
        criterion = criterion.to(device)

    epochs = int(train_cfg.get("epochs", 1))
    grad_accum = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    num_steps = max(1, (len(train_loader) * epochs) // grad_accum)
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, optim_cfg=optim_cfg, num_training_steps=num_steps)

    use_amp = bool(train_cfg.get("amp", False)) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    clip_factor = float(optim_cfg.get("clip_factor", 1.0))
    patience = int(train_cfg.get("patience", 5))

    selection = str(train_cfg.get("selection_metric", "f1")).lower()
    use_combined = selection in {"combined", "whm", "f1_icat"}
    f1_weight = float(train_cfg.get("f1_weight", 1.88))
    icat_every = max(1, int(train_cfg.get("icat_every", 1)))
    icat_subset = str(train_cfg.get("icat_subset", "test"))
    splits_rel = str(train_cfg.get("stereoset_splits", "splits/stereoset"))
    splits_dir = _resolve_data(paths, splits_rel) if use_combined else None
    # splits_dir is a directory — resolve under data_dir
    if use_combined:
        cand = paths.data_dir / splits_rel
        splits_dir = cand if cand.is_dir() else Path(splits_rel)

    rid = run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = paths.model_dir / "runs" / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_score = -1.0
    best_icat: float | None = None
    best_path = run_dir / "best_model.pt"
    no_improve = 0
    history: list[dict[str, float]] = []
    last_icat = 0.0

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(
            model,
            train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_accum_steps=grad_accum,
            clip_factor=clip_factor,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_f1 = evaluate_f1(model, val_loader, device=device)
        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": avg_loss,
            "val_f1_macro": val_f1,
        }
        if use_combined and splits_dir is not None and (
            epoch == 1 or epoch % icat_every == 0 or epoch == epochs
        ):
            last_icat = probe_avg_icat(
                model,
                tokenizer,
                splits_dir=splits_dir,
                device=device,
                max_len=max_len,
                batch_size=min(64, batch_size * 2),
                subset=icat_subset,
            )
            row["avg_icat"] = last_icat
            row["combined"] = compute_combined_score(
                val_f1, last_icat, f1_weight=f1_weight
            )
        history.append(row)
        if on_epoch is not None:
            on_epoch(row)

        if use_combined:
            score = float(row.get("combined", compute_combined_score(val_f1, last_icat, f1_weight=f1_weight)))
            improved = score > best_score
            if improved:
                best_score = score
                best_f1 = val_f1
                best_icat = last_icat
        else:
            improved = val_f1 > best_f1
            if improved:
                best_f1 = val_f1

        if improved:
            no_improve = 0
            save_state_dict(model, best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    final_path = run_dir / "final_model.pt"
    save_state_dict(model, final_path)
    if not best_path.is_file():
        save_state_dict(model, best_path)

    metrics = {
        "run_id": rid,
        "config": str(config_path),
        "experiment": str(cfg.get("name", rid)),
        "dataset": str(dataset_path),
        "n_train": len(train_ex),
        "n_val": len(val_ex),
        "seed": seed_i,
        "selection_metric": selection,
        "best_val_f1_macro": best_f1,
        "best_avg_icat": best_icat,
        "best_combined": best_score if use_combined else None,
        "epochs_ran": len(history),
        "history": history,
        "device": device,
        "thresholds": thresholds_from_yaml(cfg),
        "ablation": {
            "use_bias_projection": bool(arch.use_bias_projection),
            "loss_kind": str(loss_cfg.get("kind", "bias_aware")),
            "use_class_weights": bool(loss_cfg.get("use_class_weights", True)),
            "weighted_sampler": use_weighted_sampler,
            "freeze_backbone": bool(train_cfg.get("freeze_backbone", False)),
        },
    }
    metrics_path = run_dir / "metrics.json"
    write_json(metrics_path, metrics)
    write_json(run_dir / "config.snapshot.json", {**cfg, "train": train_cfg})

    return TrainResult(
        run_dir=str(run_dir),
        best_val_f1=best_f1,
        epochs_ran=len(history),
        metrics_path=str(metrics_path),
        best_checkpoint=str(best_path),
        best_combined=best_score if use_combined else None,
        best_avg_icat=best_icat,
    )
