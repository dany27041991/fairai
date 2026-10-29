"""Use-case: classification metrics on a labeled JSONL with a VAN checkpoint."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from fairai.domain.labels import Label
from fairai.infrastructure.data.jsonl_repo import load_examples
from fairai.infrastructure.torch_van.checkpoint import load_van_and_tokenizer
from fairai.infrastructure.torch_van.encoded_dataset import EncodedDataset
from fairai.infrastructure.torch_van.thresholds import predict_from_logits


@dataclass(slots=True)
class ClassificationReport:
    accuracy: float
    f1_macro: float
    f1_weighted: float
    n_samples: int
    per_class: dict
    label_distribution_pred: dict[str, int]
    label_distribution_true: dict[str, int]
    confusion_matrix: list[list[int]]
    y_true: list[int]
    y_pred: list[int]
    by_bias_type: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        # Omit raw vectors from default JSON (large); keep matrix.
        data = asdict(self)
        data.pop("y_true", None)
        data.pop("y_pred", None)
        return data


def evaluate_classification(
    *,
    checkpoint_path: Path,
    config_path: Path,
    dataset_path: Path,
    device: str = "cpu",
    batch_size: int = 32,
    max_samples: int | None = None,
    max_len: int = 512,
    apply_threshold_gates: bool = False,
) -> ClassificationReport:
    bundle = load_van_and_tokenizer(
        checkpoint_path, config_path, device=device, default_max_len=max_len
    )
    examples = load_examples(dataset_path, max_samples=max_samples)
    if not examples:
        raise ValueError(f"No examples in {dataset_path}")

    loader = DataLoader(
        EncodedDataset(examples, bundle.tokenizer, bundle.max_len),
        batch_size=batch_size,
        shuffle=False,
    )
    thresholds = bundle.thresholds

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits, _ = bundle.model(input_ids, attention_mask)
            pred, _ = predict_from_logits(
                logits,
                temp=thresholds["temp"] if apply_threshold_gates else 1.0,
                conf=thresholds["conf"] if apply_threshold_gates else 0.0,
                margin=thresholds["margin"] if apply_threshold_gates else 0.0,
                var=thresholds["var"] if apply_threshold_gates else 1.0,
                apply_gates=apply_threshold_gates,
            )
            y_true.extend(batch["label"].tolist())
            y_pred.extend(pred.cpu().tolist())

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=[Label(i).display_name for i in range(3)],
        output_dict=True,
        zero_division=0,
    )
    dist_true = {Label(i).name: y_true.count(i) for i in range(3)}
    dist_pred = {Label(i).name: y_pred.count(i) for i in range(3)}
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    by_bias: dict[str, dict] = {}
    groups: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        groups[ex.bias_type or "null"].append(i)
    for bias, idxs in sorted(groups.items()):
        yt = [y_true[i] for i in idxs]
        yp = [y_pred[i] for i in idxs]
        by_bias[bias] = {
            "n": len(idxs),
            "f1_macro": float(
                f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0)
            ),
            "accuracy": float(accuracy_score(yt, yp)),
        }
    return ClassificationReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        n_samples=len(y_true),
        per_class=report,
        label_distribution_pred=dist_pred,
        label_distribution_true=dist_true,
        confusion_matrix=cm.tolist(),
        y_true=list(y_true),
        y_pred=list(y_pred),
        by_bias_type=by_bias,
    )
