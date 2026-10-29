"""Thesis few-shot ``LABELS_INFO`` + scoring helpers."""

from __future__ import annotations

from typing import Any

# From FAIRAI-Thesis few-shot eval cell (Baseline vs Few-Shot).
LABELS_INFO: dict[str, dict[str, Any]] = {
    "2": {"numeric": 3, "display": "Neutral", "order": 0},
    "None": {"numeric": 0, "display": "Not assignable", "order": 1},
    "1": {"numeric": -1, "display": "Antistereotyped", "order": 2},
    "0": {"numeric": -2, "display": "Stereotypical", "order": 3},
}


def get_numeric_id(label: str | int | None) -> int:
    key = "None" if label is None or label == "" else str(label)
    return int(LABELS_INFO[key]["numeric"]) if key in LABELS_INFO else 0


def get_display_name(label: str | int | None) -> str:
    key = "None" if label is None or label == "" else str(label)
    return str(LABELS_INFO[key]["display"]) if key in LABELS_INFO else "Unknown"


def get_confusion_order(*, exclude_none: bool = True) -> tuple[list[int], list[str]]:
    sorted_labels = sorted(LABELS_INFO.keys(), key=lambda k: int(LABELS_INFO[k]["order"]))
    if exclude_none:
        sorted_labels = [k for k in sorted_labels if k != "None"]
    numeric = [int(LABELS_INFO[k]["numeric"]) for k in sorted_labels]
    display = [str(LABELS_INFO[k]["display"]) for k in sorted_labels]
    return numeric, display


def improvement_percentage(old_value: float, new_value: float) -> float:
    """Thesis ``improvement_percentage`` — % change from old → new."""
    if abs(old_value) < 1e-8:
        if abs(new_value) < 1e-8:
            return 0.0
        return 100.0
    return ((new_value - old_value) / abs(old_value)) * 100
