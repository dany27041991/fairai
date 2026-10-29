"""Save / load training run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_state_dict(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    ensure_ascii: bool = True,
) -> None:
    """Write pretty JSON (analyze reports, metrics, manifests)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=ensure_ascii) + "\n",
        encoding="utf-8",
    )


# Alias used by analyze_* report writers
write_report_json = write_json
