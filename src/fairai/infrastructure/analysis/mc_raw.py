"""Shared StereoSet MC JSON root parsing (prompts list or train/val/test)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_mc_raw(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_mc_prompts(raw: Any) -> list[dict[str, Any]]:
    """Thesis ``prompts`` list, or train/validation/test splits, or bare list."""
    if isinstance(raw, list):
        return [p for p in raw if isinstance(p, dict)]
    if not isinstance(raw, dict):
        raise ValueError(f"Unrecognized MC JSON root type: {type(raw)}")
    if "prompts" in raw:
        return [p for p in (raw.get("prompts") or []) if isinstance(p, dict)]
    if any(k in raw for k in ("train", "validation", "test")):
        prompts: list[dict[str, Any]] = []
        for split in ("train", "validation", "test"):
            prompts.extend(p for p in (raw.get(split) or []) if isinstance(p, dict))
        return prompts
    raise ValueError("MC JSON must have 'prompts' or train/validation/test splits")


def split_mc_prompts(raw: Any) -> dict[str, list[dict[str, Any]]] | None:
    """Return native train/validation/test dict if present; else None."""
    if not isinstance(raw, dict):
        return None
    if "train" not in raw and "validation" not in raw and "test" not in raw:
        return None
    return {
        split: [p for p in (raw.get(split) or []) if isinstance(p, dict)]
        for split in ("train", "validation", "test")
        if split in raw
    }
