"""Remove train candidates that overlap StereoSet evaluation text."""

from __future__ import annotations

from pathlib import Path

from fairai.domain.example import Example
from fairai.infrastructure.data.stereoset_flatten import flatten_split_file
from fairai.infrastructure.generation.normalize import normalize_sentence


def stereoset_forbidden_sentences(splits_dir: Path) -> set[str]:
    """Normalized sentence strings from StereoSet test (inter + intra)."""
    forbidden: set[str] = set()
    for name in ("intersentence_test.json", "intrasentence_test.json"):
        path = splits_dir / name
        if not path.is_file():
            continue
        for concat in (True, False):
            for ex in flatten_split_file(path, concatenate_context=concat):
                forbidden.add(normalize_sentence(ex.sentence).lower())
                # also raw sentence without SEP baggage
                forbidden.add(normalize_sentence(ex.sentence.split("[SEP]")[-1]).lower())
    return forbidden


def filter_leakage(
    examples: list[Example],
    forbidden: set[str],
) -> tuple[list[Example], int]:
    kept: list[Example] = []
    removed = 0
    for ex in examples:
        key = normalize_sentence(ex.sentence).lower()
        if key in forbidden:
            removed += 1
            continue
        kept.append(ex)
    return kept, removed
