"""Thesis ANALISYS cell 47 — MC predictions accuracy / CM / error by bias."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.metrics import classification_report, confusion_matrix

from fairai.domain.analysis_stats import accuracy, accuracy_by_group, error_rate_by_group
from fairai.infrastructure.analysis.mc_options import load_mc_option_rows
from fairai.infrastructure.torch_van.checkpointing import write_json


@dataclass(slots=True)
class McPredictionReport:
    n_total: int
    n_predicted: int
    n_missing_pred: int
    accuracy_global: float
    accuracy_by_bias: dict[str, float]
    error_rate_by_bias: dict[str, float]
    classification_report: dict[str, Any]
    confusion_matrix: list[list[int]]
    labels: list[Any]
    plot_paths: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_predicted": self.n_predicted,
            "n_missing_pred": self.n_missing_pred,
            "accuracy_global": self.accuracy_global,
            "accuracy_by_bias": self.accuracy_by_bias,
            "error_rate_by_bias": self.error_rate_by_bias,
            "classification_report": self.classification_report,
            "confusion_matrix": self.confusion_matrix,
            "labels": [str(x) for x in self.labels],
            "plot_paths": self.plot_paths,
        }


def analyze_mc_predictions(
    dataset_path: Path,
    *,
    out_dir: Path | None = None,
    write_plots: bool = True,
) -> McPredictionReport:
    rows = load_mc_option_rows(dataset_path)
    pred_rows = [r for r in rows if r.label_predicted is not None]
    y_true = [r.label for r in pred_rows]
    y_pred = [r.label_predicted for r in pred_rows]
    groups = [r.item_bias for r in pred_rows]
    labels = sorted(set(y_true) | set(y_pred), key=lambda x: str(x))
    cm = confusion_matrix(y_true, y_pred, labels=labels) if y_true else []
    report = (
        classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        if y_true
        else {}
    )
    acc_by = accuracy_by_group(y_true, y_pred, groups) if y_true else {}
    err_by = error_rate_by_group(y_true, y_pred, groups) if y_true else {}

    plot_paths: list[str] = []
    if write_plots and out_dir is not None and y_true:
        from fairai.infrastructure.viz.bars import plot_horizontal_counts
        from fairai.infrastructure.viz.confusion import plot_confusion_matrix

        out_dir.mkdir(parents=True, exist_ok=True)
        cm_path = plot_confusion_matrix(
            cm,
            class_names=[str(x) for x in labels],
            title="Confusion Matrix",
            out_path=out_dir / "mc_confusion_matrix.png",
        )
        plot_paths.append(str(cm_path))
        err_labels = list(err_by.keys())
        err_vals = [err_by[k] for k in err_labels]
        err_path = plot_horizontal_counts(
            err_labels,
            err_vals,
            xlabel="Error Rate",
            ylabel="item_bias",
            title="Error Rate by Type of Bias (item_bias)",
            out_path=out_dir / "mc_error_rate_by_bias.png",
        )
        plot_paths.append(str(err_path))

    result = McPredictionReport(
        n_total=len(rows),
        n_predicted=len(pred_rows),
        n_missing_pred=len(rows) - len(pred_rows),
        accuracy_global=accuracy(y_true, y_pred) if y_true else 0.0,
        accuracy_by_bias=acc_by,
        error_rate_by_bias=err_by,
        classification_report=report,
        confusion_matrix=cm.tolist() if hasattr(cm, "tolist") else list(cm),
        labels=labels,
        plot_paths=plot_paths,
    )
    if out_dir is not None:
        write_json(out_dir / "mc_predictions_report.json", result.to_dict())
    return result
