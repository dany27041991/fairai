"""Load pre-processed Thesis GenDataset JSONL pools (WinoBias / ToxiGen / …)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from fairai.domain.example import Example
from fairai.domain.labels import Label
from fairai.infrastructure.data.jsonl_repo import load_examples
from fairai.infrastructure.generation.normalize import (
    capitalize_sentence,
    ensure_period,
    normalize_sentence,
)


# Relative to a Thesis ``DEVELOP/datasets`` (or fairai ``artifacts/datasets/raw``) root.
DEFAULT_GLOBS: dict[str, tuple[str, ...]] = {
    "winobias": ("wino_bias/*.jsonl", "winobias/*.jsonl"),
    "toxigen": ("toxigen/*.jsonl",),
    "squad": ("squad/*.jsonl",),
    "tatoeba": ("tatoeba/*.jsonl",),
    "stereoset": (
        "stereoset/stereoset_dataset.jsonl",
        "stereoset/stereoset_intrasentence_dataset.jsonl",
        "stereoset/combined_dataset.jsonl",
    ),
}


def resolve_source_files(
    root: Path,
    source: str,
    globs: tuple[str, ...] | None = None,
) -> list[Path]:
    patterns = globs or DEFAULT_GLOBS.get(source, ())
    found: list[Path] = []
    for pattern in patterns:
        found.extend(sorted(root.glob(pattern)))
    # de-dupe preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for path in found:
        if path.is_file() and path not in seen:
            seen.add(path)
            out.append(path)
    return out


def iter_van_format_jsonl(path: Path, *, max_samples: int | None = None) -> Iterator[Example]:
    """Records with sentence/label[/bias_type] (WinoBias, ToxiGen, SQuAD, Tatoeba)."""
    yield from load_examples(path, max_samples=max_samples)


def iter_stereoset_pool_jsonl(path: Path, *, max_samples: int | None = None) -> Iterator[Example]:
    """StereoSet pool JSONL may nest ``synonym_sentences`` (Thesis GenDataset)."""
    n = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if max_samples is not None and n >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            label_raw = rec.get("label")
            if label_raw is None:
                continue
            try:
                label = Label.from_int(int(label_raw))
            except ValueError:
                continue
            bias = rec.get("bias_type")
            if bias in ("", "none", None):
                bias = None
            sentences = [str(rec.get("sentence", "")).strip()]
            for syn in rec.get("synonym_sentences") or []:
                s = str(syn).strip()
                if s:
                    sentences.append(s)
            for sentence in sentences:
                if not sentence:
                    continue
                text = capitalize_sentence(ensure_period(normalize_sentence(sentence)))
                yield Example(sentence=text, label=label, bias_type=str(bias) if bias else None)
                n += 1
                if max_samples is not None and n >= max_samples:
                    return


def load_named_source(
    root: Path,
    source: str,
    *,
    max_samples: int | None = None,
    globs: tuple[str, ...] | None = None,
) -> list[Example]:
    files = resolve_source_files(root, source, globs=globs)
    if not files:
        return []
    out: list[Example] = []
    remaining = max_samples
    for path in files:
        if remaining is not None and remaining <= 0:
            break
        if source == "stereoset":
            chunk = list(iter_stereoset_pool_jsonl(path, max_samples=remaining))
        else:
            chunk = list(iter_van_format_jsonl(path, max_samples=remaining))
        out.extend(chunk)
        if remaining is not None:
            remaining -= len(chunk)
    return out
