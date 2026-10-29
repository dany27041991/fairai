"""Thesis cell 10 — merge train/validation/test MC preds into flat ``prompts``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fairai.infrastructure.torch_van.checkpointing import write_json


@dataclass(slots=True)
class MergeMcReport:
    n_prompts: int
    splits_used: list[str]
    out_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_prompts": self.n_prompts,
            "splits_used": self.splits_used,
            "out_path": self.out_path,
        }


def merge_stereoset_select_options(
    input_path: Path,
    output_path: Path,
    *,
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> MergeMcReport:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    prompts: list[Any] = []
    used: list[str] = []
    for split in splits:
        split_items = data.get(split, [])
        if not isinstance(split_items, list):
            raise ValueError(f"The key '{split}' does not contain a list.")
        if split_items:
            used.append(split)
        prompts.extend(split_items)
    write_json(output_path, {"prompts": prompts}, ensure_ascii=False)
    return MergeMcReport(n_prompts=len(prompts), splits_used=used, out_path=str(output_path))
