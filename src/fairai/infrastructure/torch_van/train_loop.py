"""Single-epoch train / val helpers."""

from __future__ import annotations

from typing import Any

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from fairai.infrastructure.torch_van.optim_factory import clip_gradients


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: str,
    grad_accum_steps: int,
    clip_factor: float,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
) -> float:
    model.train()
    total = 0.0
    optimizer.zero_grad(set_to_none=True)
    amp_device = "cuda" if str(device).startswith("cuda") else "cpu"
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast(amp_device, enabled=use_amp):
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels) / grad_accum_steps

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % grad_accum_steps == 0 or step == len(loader):
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                clip_gradients(model, clip_factor=clip_factor)
                scaler.step(optimizer)
                scaler.update()
            else:
                clip_gradients(model, clip_factor=clip_factor)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total += float(loss.item()) * grad_accum_steps

    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate_f1(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: str,
) -> float:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]
        logits, _ = model(input_ids, attention_mask)
        pred = logits.argmax(dim=-1).cpu()
        y_true.extend(labels.tolist())
        y_pred.extend(pred.tolist())
    if not y_true:
        return 0.0
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
