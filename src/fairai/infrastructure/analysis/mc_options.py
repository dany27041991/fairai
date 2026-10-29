"""Load StereoSet MC JSON used by Thesis cell 47 / few-shot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fairai.infrastructure.analysis.mc_raw import iter_mc_prompts, load_mc_raw


@dataclass(slots=True)
class McOptionRow:
    sentence: str
    item_bias: str
    label: Any
    label_predicted: Any | None
    correct_original: Any | None


def load_mc_option_rows(path: Path) -> list[McOptionRow]:
    rows: list[McOptionRow] = []
    for prompt in iter_mc_prompts(load_mc_raw(path)):
        item_bias = str(prompt.get("item_bias", "unknown"))
        for opt in prompt.get("options") or []:
            rows.append(
                McOptionRow(
                    sentence=str(opt.get("sentence", "")),
                    item_bias=item_bias,
                    label=opt.get("label"),
                    label_predicted=opt.get("label_predicted"),
                    correct_original=opt.get("correct"),
                )
            )
    return rows
