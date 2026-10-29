"""Figure I/O with non-interactive Agg backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def require_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plots. Install with: pip install 'fairai[viz]'"
        ) from exc
    return plt


def save_figure(fig: Any, path: Path, *, dpi: int = 120) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt = require_matplotlib()
    plt.close(fig)
    return path
