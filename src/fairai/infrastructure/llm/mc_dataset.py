"""Load StereoSet multiple-choice JSON (select_options*)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fairai.infrastructure.llm.prompts import is_valid_example


def load_mc_file(path: Path) -> dict[str, list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "train" in raw:
        return {
            split: [ex for ex in (raw.get(split) or []) if is_valid_example(ex)]
            for split in ("train", "validation", "test")
            if split in raw
        }
    if isinstance(raw, dict) and "prompts" in raw:
        prompts = [ex for ex in raw["prompts"] if is_valid_example(ex)]
        # synthetic split for smoke when only prompts list exists
        n = len(prompts)
        return {
            "train": prompts[: max(1, int(0.7 * n))],
            "validation": prompts[int(0.7 * n) : int(0.85 * n)],
            "test": prompts[int(0.85 * n) :],
        }
    raise ValueError(f"Unrecognized MC dataset schema in {path}")
