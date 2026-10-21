"""Use-case: classification metrics on a labeled JSONL with a VAN checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from fairai.domain.example import Example
from fairai.domain.labels import Label
from fairai.infrastructure.data.jsonl_repo import load_examples
from fairai.infrastructure.torch_van.checkpoint import load_van_classifier
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

    def to_dict(self) -> dict:
        return asdict(self)


class _EncodedDataset(Dataset):
    def __init__(
        self,
        examples: list[Example],
        tokenizer: AutoTokenizer,
        max_len: int,
    ) -> None:
        sentences = [ex.sentence for ex in examples]
        enc = tokenizer(
            sentences,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor([int(ex.label) for ex in examples], dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.size(0))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }


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
    model, thresholds, raw = load_van_classifier(
        checkpoint_path, config_path, device=device
    )
    backbone = str(raw.get("backbone", "sentence-transformers/all-MiniLM-L6-v2"))
    max_len = int(raw.get("max_len", max_len))
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    examples = load_examples(dataset_path, max_samples=max_samples)
    if not examples:
        raise ValueError(f"No examples in {dataset_path}")

    loader = DataLoader(
        _EncodedDataset(examples, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=False,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits, _ = model(input_ids, attention_mask)
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
    return ClassificationReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        n_samples=len(y_true),
        per_class=report,
        label_distribution_pred=dist_pred,
        label_distribution_true=dist_true,
    )
