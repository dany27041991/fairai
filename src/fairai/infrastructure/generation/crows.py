"""CrowS-Pairs CSV → labeled Example records."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from fairai.domain.example import Example
from fairai.domain.labels import Label
from fairai.infrastructure.generation.normalize import (
    capitalize_sentence,
    ensure_period,
    normalize_sentence,
)
from fairai.infrastructure.generation.synonyms import generate_synonym_sentences


def iter_crows_examples(
    csv_path: Path,
    *,
    expand_synonyms: bool = False,
    max_synonym_samples: int = 3,
    max_rows: int | None = None,
) -> Iterator[Example]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            sent_more = capitalize_sentence(ensure_period(normalize_sentence(row["sent_more"])))
            sent_less = capitalize_sentence(ensure_period(normalize_sentence(row["sent_less"])))
            bias_type = str(row.get("bias_type") or "unknown")
            kind = str(row.get("stereo_antistereo") or "").strip().lower()
            if kind == "stereo":
                label_more, label_less = Label.STEREOTYPE, Label.ANTI_STEREOTYPE
            elif kind == "antistereo":
                label_more, label_less = Label.ANTI_STEREOTYPE, Label.STEREOTYPE
            else:
                continue

            pairs = [(sent_more, label_more), (sent_less, label_less)]
            for sentence, label in pairs:
                yield Example(sentence=sentence, label=label, bias_type=bias_type)
                if expand_synonyms:
                    for syn in generate_synonym_sentences(
                        sentence, max_samples=max_synonym_samples
                    )[1:]:
                        yield Example(
                            sentence=capitalize_sentence(ensure_period(syn)),
                            label=label,
                            bias_type=bias_type,
                        )
