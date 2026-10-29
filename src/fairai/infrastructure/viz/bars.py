"""Bar charts used across Thesis ANALISYS / RL / few-shot cells."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from fairai.infrastructure.viz.savefig import require_matplotlib, save_figure


def plot_mean_std_bars(
    labels: Sequence[str],
    means: Sequence[float],
    stds: Sequence[float],
    *,
    ylabel: str,
    title: str,
    out_path: Path,
    colors: Sequence[str] | None = None,
) -> Path:
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 5))
    x = list(range(len(labels)))
    bars = ax.bar(
        x,
        list(means),
        yerr=list(stds),
        capsize=8,
        color=list(colors) if colors is not None else None,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(list(labels))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, mean in zip(bars, means, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    return save_figure(fig, out_path)


def plot_grouped_bars(
    categories: Sequence[str],
    series: dict[str, Sequence[float]],
    *,
    ylabel: str,
    title: str,
    out_path: Path,
) -> Path:
    """Side-by-side bars (Thesis few-shot Baseline vs Few-Shot)."""
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    n_series = len(series)
    width = 0.8 / max(n_series, 1)
    x = list(range(len(categories)))
    for i, (name, values) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * width
        ax.bar([xi + offset for xi in x], list(values), width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(list(categories), rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return save_figure(fig, out_path)


def plot_horizontal_counts(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> Path:
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(list(labels), list(values))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return save_figure(fig, out_path)
