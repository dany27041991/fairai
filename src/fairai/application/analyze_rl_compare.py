"""Thesis ANALISYS cell 46 — paired Baseline vs RL metric comparison + bars."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fairai.domain.analysis_stats import PairedCompareResult, compare_paired
from fairai.infrastructure.torch_van.checkpointing import write_json


@dataclass(slots=True)
class RlCompareReport:
    results: list[PairedCompareResult]
    plot_paths: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "plot_paths": self.plot_paths,
        }


def _pairs_from_payload(payload: Mapping[str, Any]) -> list[tuple[str, list[float], list[float]]]:
    """Accept either {metric: {baseline:[], treatment:[]}} or list of metric blocks."""
    out: list[tuple[str, list[float], list[float]]] = []
    if "metrics" in payload and isinstance(payload["metrics"], list):
        for block in payload["metrics"]:
            out.append(
                (
                    str(block["name"]),
                    [float(x) for x in block["baseline"]],
                    [float(x) for x in block.get("rl", block.get("treatment", []))],
                )
            )
        return out
    for name, block in payload.items():
        if not isinstance(block, Mapping):
            continue
        if "baseline" not in block:
            continue
        treatment = block.get("rl", block.get("treatment"))
        if treatment is None:
            continue
        out.append(
            (
                str(name),
                [float(x) for x in block["baseline"]],
                [float(x) for x in treatment],
            )
        )
    return out


def analyze_rl_compare(
    metrics: Mapping[str, Any] | Sequence[Mapping[str, Any]] | Path,
    *,
    out_dir: Path | None = None,
    alpha: float = 0.05,
    write_plots: bool = True,
) -> RlCompareReport:
    if isinstance(metrics, Path):
        payload = json.loads(metrics.read_text(encoding="utf-8"))
    elif isinstance(metrics, Sequence) and not isinstance(metrics, Mapping):
        payload = {"metrics": list(metrics)}
    else:
        payload = dict(metrics)

    results: list[PairedCompareResult] = []
    plot_paths: list[str] = []
    for name, baseline, treatment in _pairs_from_payload(payload):
        result = compare_paired(baseline, treatment, metric=name, alpha=alpha)
        results.append(result)
        if write_plots and out_dir is not None:
            from fairai.infrastructure.viz.bars import plot_mean_std_bars

            path = plot_mean_std_bars(
                ["Baseline", "RL"],
                [result.baseline.mean, result.treatment.mean],
                [result.baseline.std, result.treatment.std],
                ylabel=name,
                title=f"Comparison of {name}: Baseline vs RL",
                out_path=out_dir / f"rl_compare_{_slug(name)}.png",
                colors=["skyblue", "salmon"],
            )
            plot_paths.append(str(path))

    report = RlCompareReport(results=results, plot_paths=plot_paths)
    if out_dir is not None:
        write_json(out_dir / "rl_compare_report.json", report.to_dict())
    return report


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
