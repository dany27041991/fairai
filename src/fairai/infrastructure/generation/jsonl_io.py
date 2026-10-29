"""Read/write Example JSONL streams."""

from __future__ import annotations

import json
from pathlib import Path

from fairai.domain.example import Example


def write_examples(path: Path, examples: list[Example]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            handle.write(json.dumps(ex.to_record(), ensure_ascii=False) + "\n")
    return len(examples)


def load_examples_jsonl(path: Path, *, max_samples: int | None = None) -> list[Example]:
    """Deprecated alias — prefer ``jsonl_repo.load_examples``."""
    from fairai.infrastructure.data.jsonl_repo import load_examples

    return load_examples(path, max_samples=max_samples)
