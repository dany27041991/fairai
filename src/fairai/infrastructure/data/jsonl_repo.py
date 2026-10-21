"""JSONL dataset repository for FAIRAI examples."""

from __future__ import annotations

import json
from pathlib import Path

from fairai.domain.example import Example


def load_examples(path: Path, *, max_samples: int | None = None) -> list[Example]:
    examples: list[Example] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(Example.from_record(json.loads(line)))
            if max_samples is not None and len(examples) >= max_samples:
                break
    return examples
