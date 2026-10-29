"""Classification metrics for Thesis few-shot Baseline vs Few-Shot cell."""

from __future__ import annotations

from typing import Any, Sequence

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from fairai.infrastructure.llm.labels import get_numeric_id
from fairai.infrastructure.llm.prompts import gold_label


def score_int_labels(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> tuple[float, float]:
    """Raw 0/1/2 accuracy + macro-F1 (VAN few-shot smoke; no LABELS_INFO remap)."""
    if not y_true:
        return 0.0, 0.0
    return (
        float(accuracy_score(list(y_true), list(y_pred))),
        float(f1_score(list(y_true), list(y_pred), average="macro", zero_division=0)),
    )


def compute_classification_metrics(
    examples: Sequence[dict[str, Any]],
    predictions: Sequence[str | int | None],
) -> dict[str, float]:
    """Thesis ``compute_classification_metrics`` — drops ``None`` preds."""
    filtered_ex: list[dict[str, Any]] = []
    filtered_pred: list[str] = []
    for example, pred in zip(examples, predictions, strict=True):
        key = "None" if pred is None or pred == "" else str(pred)
        if key == "None":
            continue
        filtered_ex.append(example)
        filtered_pred.append(key)
    if not filtered_ex:
        return {
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "balanced_acc": 0.0,
        }
    y_true = [get_numeric_id(gold_label(ex)) for ex in filtered_ex]
    y_pred = [get_numeric_id(p) for p in filtered_pred]
    return {
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def accuracy_excluding_none(
    examples: Sequence[dict[str, Any]],
    predictions: Sequence[str | int | None],
) -> float:
    """Thesis ``evaluate_against_true_labels`` accuracy (exclude None)."""
    pairs: list[tuple[int, int]] = []
    for example, pred in zip(examples, predictions, strict=True):
        key = "None" if pred is None or pred == "" else str(pred)
        if key == "None":
            continue
        gold = gold_label(example)
        if gold is None:
            continue
        pairs.append((get_numeric_id(gold), get_numeric_id(key)))
    if not pairs:
        return 0.0
    y_true, y_pred = zip(*pairs, strict=True)
    return float(accuracy_score(list(y_true), list(y_pred)))
