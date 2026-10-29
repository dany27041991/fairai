"""Load StereoSet multiple-choice JSON (select_options*)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fairai.infrastructure.analysis.mc_raw import (
    iter_mc_prompts,
    load_mc_raw,
    split_mc_prompts,
)
from fairai.infrastructure.llm.prompts import is_valid_example


def load_mc_file(path: Path) -> dict[str, list[dict[str, Any]]]:
    raw = load_mc_raw(path)
    native = split_mc_prompts(raw)
    if native is not None:
        return {
            split: [ex for ex in examples if is_valid_example(ex)]
            for split, examples in native.items()
        }
    prompts = [ex for ex in iter_mc_prompts(raw) if is_valid_example(ex)]
    # synthetic split for smoke when only prompts list exists
    n = len(prompts)
    return {
        "train": prompts[: max(1, int(0.7 * n))],
        "validation": prompts[int(0.7 * n) : int(0.85 * n)],
        "test": prompts[int(0.85 * n) :],
    }
