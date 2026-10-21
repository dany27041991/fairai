"""Use-case: train VAN from YAML (smoke subset or full run)."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from fairai.application.config_merge import load_experiment_config
from fairai.infrastructure.artifact_paths import ArtifactPaths
from fairai.infrastructure.data.jsonl_repo import load_examples
from fairai.infrastructure.torch_van.checkpoint import (
    arch_config_from_yaml,
    thresholds_from_yaml,
)
from fairai.infrastructure.torch_van.checkpointing import save_state_dict, write_json
from fairai.infrastructure.torch_van.encoded_dataset import EncodedDataset
from fairai.infrastructure.torch_van.losses import build_bias_aware_loss
from fairai.infrastructure.torch_van.optim_factory import build_optimizer, build_scheduler
from fairai.infrastructure.torch_van.train_loop import evaluate_f1, train_one_epoch
from fairai.infrastructure.torch_van.van_classifier import VanClassifier


@dataclass(slots=True)
class TrainResult:
    run_dir: str
    best_val_f1: float
    epochs_ran: int
    metrics_path: str
    best_checkpoint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_van(
    *,
    config_path: Path,
    artifacts: ArtifactPaths | None = None,
    device: str | None = None,
    max_samples: int | None = None,
    run_id: str | None = None,
    val_fraction: float = 0.1,
) -> TrainResult:
    cfg = load_experiment_config(config_path)
    paths = artifacts or ArtifactPaths.from_env()
    train_cfg = dict(cfg.get("train", {}))
    optim_cfg = dict(cfg.get("optim", {}))
    loss_cfg = dict(cfg.get("loss", {}))

    device = device or str(cfg.get("device", "cpu"))
    seed = int(train_cfg.get("seed", 42))
    _set_seed(seed)

    dataset_rel = str(train_cfg.get("dataset", "processed/van_train_leakage_safe.jsonl"))
    dataset_path = paths.data_dir / dataset_rel
    if not dataset_path.is_file():
        # allow absolute / cwd-relative
        alt = Path(dataset_rel)
        dataset_path = alt if alt.is_file() else paths.processed(Path(dataset_rel).name)

    limit = max_samples if max_samples is not None else train_cfg.get("max_samples")
    limit_i = int(limit) if limit is not None else None
    examples = load_examples(dataset_path, max_samples=limit_i)
    if len(examples) < 4:
        raise ValueError(f"Need ≥4 examples to train/val-split, got {len(examples)}")

    labels = [int(ex.label) for ex in examples]
    train_ex, val_ex = train_test_split(
        examples,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    backbone = str(cfg.get("backbone", "sentence-transformers/all-MiniLM-L6-v2"))
    max_len = int(cfg.get("max_len", 512))
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    batch_size = int(train_cfg.get("batch_size", 8))
    train_loader = DataLoader(
        EncodedDataset(train_ex, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        EncodedDataset(val_ex, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=False,
    )

    arch = arch_config_from_yaml(cfg)
    model = VanClassifier(arch, init_backbone_weights=True).to(device)
    criterion = build_bias_aware_loss(loss_cfg, num_labels=int(cfg.get("num_labels", 3)))

    epochs = int(train_cfg.get("epochs", 1))
    grad_accum = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    num_steps = max(1, (len(train_loader) * epochs) // grad_accum)
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer, optim_cfg=optim_cfg, num_training_steps=num_steps)

    use_amp = bool(train_cfg.get("amp", False)) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    clip_factor = float(optim_cfg.get("clip_factor", 1.0))
    patience = int(train_cfg.get("patience", 5))

    rid = run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = paths.model_dir / "runs" / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_path = run_dir / "best_model.pt"
    no_improve = 0
    history: list[dict[str, float]] = []

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
        history.append({"epoch": float(epoch), "train_loss": avg_loss, "val_f1_macro": val_f1})
        if val_f1 > best_f1:
            best_f1 = val_f1
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
        "dataset": str(dataset_path),
        "n_train": len(train_ex),
        "n_val": len(val_ex),
        "best_val_f1_macro": best_f1,
        "epochs_ran": len(history),
        "history": history,
        "device": device,
        "thresholds": thresholds_from_yaml(cfg),
    }
    metrics_path = run_dir / "metrics.json"
    write_json(metrics_path, metrics)
    write_json(run_dir / "config.snapshot.json", cfg)

    return TrainResult(
        run_dir=str(run_dir),
        best_val_f1=best_f1,
        epochs_ran=len(history),
        metrics_path=str(metrics_path),
        best_checkpoint=str(best_path),
    )
