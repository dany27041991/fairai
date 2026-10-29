"""Thesis RL eval plots (cell 15) — summarize + bar charts from row metrics."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from fairai.domain.analysis_stats import compute_bleu, distinct_n, summarize_metric_groups
from fairai.infrastructure.torch_van.checkpointing import write_json


@dataclass(slots=True)
class GenMetricsReport:
    summary: list[dict[str, Any]]
    plot_paths: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary, "plot_paths": self.plot_paths}


def load_metric_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "rows" in data:
            return list(data["rows"])
        if isinstance(data, list):
            return list(data)
        raise ValueError("JSON must be a list of rows or {rows: [...]}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def analyze_gen_metrics(
    metrics_path: Path,
    *,
    out_dir: Path | None = None,
    write_plots: bool = True,
    metrics: Sequence[str] = ("perplexity", "distinct_1", "distinct_2", "bleu"),
) -> GenMetricsReport:
    rows = load_metric_rows(metrics_path)
    # Coerce numeric fields
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for m in metrics:
            if m in item and item[m] not in (None, ""):
                item[m] = float(item[m])
        cleaned.append(item)
    summary = summarize_metric_groups(cleaned, metrics=metrics)
    plot_paths: list[str] = []
    if write_plots and out_dir is not None and summary:
        from fairai.infrastructure.viz.bars import plot_mean_std_bars

        out_dir.mkdir(parents=True, exist_ok=True)
        for m in metrics:
            labels = [str(s.get("model", s.get("group", ""))) for s in summary]
            means = [float(s.get(f"{m}_mean", 0.0)) for s in summary]
            stds = [float(s.get(f"{m}_std", 0.0)) for s in summary]
            if not any(means):
                continue
            path = plot_mean_std_bars(
                labels,
                means,
                stds,
                ylabel=f"{m} (mean ± std)",
                title=f"{m} Comparison: Baseline vs RL",
                out_path=out_dir / f"gen_{m}.png",
                colors=["skyblue", "salmon"][: len(labels)],
            )
            plot_paths.append(str(path))

    report = GenMetricsReport(summary=summary, plot_paths=plot_paths)
    if out_dir is not None:
        write_json(out_dir / "gen_metrics_report.json", report.to_dict())
    return report


def analyze_metric_bars(
    payload: dict[str, Any] | Path,
    *,
    out_dir: Path,
    filename: str = "metric_bars.png",
) -> Path:
    """Thesis few-shot style grouped bars: {categories, series: {name: values}}."""
    if isinstance(payload, Path):
        data = json.loads(payload.read_text(encoding="utf-8"))
    else:
        data = payload
    from fairai.infrastructure.viz.bars import plot_grouped_bars

    return plot_grouped_bars(
        list(data["categories"]),
        {k: list(v) for k, v in dict(data["series"]).items()},
        ylabel=str(data.get("ylabel", "Score")),
        title=str(data.get("title", "Metric comparison")),
        out_path=out_dir / filename,
    )


def compute_distinct_for_texts(texts: Sequence[str], n: int = 1) -> list[float]:
    return [distinct_n(t, n) for t in texts]


def score_generated_pair(reference: str, hypothesis: str) -> dict[str, float]:
    """Thesis per-sample metrics without loading CausalLM (post-hoc scoring)."""
    return {
        "distinct_1": distinct_n(hypothesis, 1),
        "distinct_2": distinct_n(hypothesis, 2),
        "bleu": compute_bleu(reference, hypothesis),
    }
