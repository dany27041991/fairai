"""Use-case: run VAN inference on free-form texts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer

from fairai.domain.labels import Label
from fairai.infrastructure.torch_van.checkpoint import load_van_classifier
from fairai.infrastructure.torch_van.thresholds import predict_from_logits


@dataclass(frozen=True, slots=True)
class InferenceItem:
    text: str
    label: int
    label_name: str
    probs: dict[str, float]


def infer_texts(
    texts: list[str],
    *,
    checkpoint_path: Path,
    config_path: Path,
    device: str = "cpu",
    apply_threshold_gates: bool = False,
) -> list[InferenceItem]:
    if not texts:
        return []
    model, thresholds, raw = load_van_classifier(
        checkpoint_path, config_path, device=device
    )
    backbone = str(raw.get("backbone", "sentence-transformers/all-MiniLM-L6-v2"))
    max_len = int(raw.get("max_len", 512))
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    enc = tokenizer(
        texts,
        max_length=max_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits, _ = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        )
        pred, probs = predict_from_logits(
            logits,
            temp=thresholds["temp"],
            conf=thresholds["conf"],
            margin=thresholds["margin"],
            var=thresholds["var"],
            apply_gates=apply_threshold_gates,
        )
    results: list[InferenceItem] = []
    for i, text in enumerate(texts):
        label_i = int(pred[i].item())
        p = probs[i].cpu().tolist()
        results.append(
            InferenceItem(
                text=text,
                label=label_i,
                label_name=Label(label_i).display_name,
                probs={
                    Label.STEREOTYPE.display_name: float(p[0]),
                    Label.ANTI_STEREOTYPE.display_name: float(p[1]),
                    Label.NEUTRAL.display_name: float(p[2]),
                },
            )
        )
    return results
