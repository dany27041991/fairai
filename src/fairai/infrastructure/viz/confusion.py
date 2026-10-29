"""Confusion-matrix heatmaps (Thesis VAN eval + few-shot / MC analysis)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from fairai.infrastructure.viz.savefig import require_matplotlib, save_figure


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]] | object,
    *,
    class_names: Sequence[str],
    title: str,
    out_path: Path,
    cmap: str = "Blues",
) -> Path:
    plt = require_matplotlib()
    try:
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "seaborn is required for confusion heatmaps. Install with: pip install 'fairai[viz]'"
        ) from exc

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    return save_figure(fig, out_path)
