"""Matplotlib/seaborn helpers (optional `fairai[viz]`)."""

from fairai.infrastructure.viz.bars import plot_grouped_bars, plot_mean_std_bars
from fairai.infrastructure.viz.confusion import plot_confusion_matrix
from fairai.infrastructure.viz.savefig import require_matplotlib, save_figure

__all__ = [
    "plot_confusion_matrix",
    "plot_grouped_bars",
    "plot_mean_std_bars",
    "require_matplotlib",
    "save_figure",
]
