"""Merge / dedupe / balanced sample for generation pipeline."""

from __future__ import annotations

import random
from collections import defaultdict

from fairai.domain.example import Example
from fairai.infrastructure.generation.normalize import normalize_sentence


def dedupe_examples(examples: list[Example]) -> list[Example]:
    seen: set[str] = set()
    out: list[Example] = []
    for ex in examples:
        key = normalize_sentence(ex.sentence).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


def merge_examples(*groups: list[Example]) -> list[Example]:
    merged: list[Example] = []
    for group in groups:
        merged.extend(group)
    return dedupe_examples(merged)


def sample_balanced(
    examples: list[Example],
    *,
    max_samples: int,
    seed: int = 42,
) -> list[Example]:
    """Balanced sample across available labels (as equal as possible)."""
    rng = random.Random(seed)
    by_label: dict[int, list[Example]] = defaultdict(list)
    for ex in examples:
        by_label[int(ex.label)].append(ex)
    labels = sorted(by_label)
    if not labels:
        return []
    for bucket in by_label.values():
        rng.shuffle(bucket)

    n_labs = len(labels)
    base = max_samples // n_labs
    rem = max_samples - base * n_labs
    quotas = {lab: base + (1 if i < rem else 0) for i, lab in enumerate(labels)}

    picked: list[Example] = []
    leftovers: list[Example] = []
    for lab in labels:
        bucket = by_label[lab]
        q = quotas[lab]
        picked.extend(bucket[:q])
        leftovers.extend(bucket[q:])
    if len(picked) < max_samples and leftovers:
        need = max_samples - len(picked)
        picked.extend(leftovers[:need])
    rng.shuffle(picked)
    return picked[:max_samples]
