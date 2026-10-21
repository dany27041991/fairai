"""Flatten StereoSet context records into labeled sentence examples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FlatStereoExample:
    sentence: str
    label: int
    example_id: str
    bias_type: str
    context_id: str


def load_stereoset_split(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def flatten_record(
    record: dict,
    *,
    concatenate_context: bool = True,
    sep_token: str = "[SEP]",
    valid_labels: frozenset[int] = frozenset({0, 1, 2}),
) -> list[FlatStereoExample]:
    sentences_block = record.get("sentences") or {}
    sentences = sentences_block.get("sentence") or []
    gold_labels = sentences_block.get("gold_label") or []
    sentence_ids = sentences_block.get("id") or []
    bias_type = str(record.get("bias_type") or "unknown")
    context_id = str(record.get("id") or "")
    context = str(record.get("context") or "") if concatenate_context else ""

    if not (
        isinstance(sentences, list)
        and isinstance(gold_labels, list)
        and isinstance(sentence_ids, list)
    ):
        return []
    if not (len(sentences) == len(gold_labels) == len(sentence_ids)):
        return []

    out: list[FlatStereoExample] = []
    for sentence, label, sid in zip(sentences, gold_labels, sentence_ids, strict=True):
        label_i = int(label)
        if label_i not in valid_labels:
            continue
        if concatenate_context and context:
            text = f"{context} {sep_token} {sentence}"
        else:
            text = str(sentence)
        out.append(
            FlatStereoExample(
                sentence=text,
                label=label_i,
                example_id=str(sid),
                bias_type=bias_type,
                context_id=context_id,
            )
        )
    return out


def flatten_split_file(
    path: Path,
    *,
    concatenate_context: bool = True,
    sep_token: str = "[SEP]",
) -> list[FlatStereoExample]:
    flat: list[FlatStereoExample] = []
    for record in load_stereoset_split(path):
        flat.extend(
            flatten_record(
                record,
                concatenate_context=concatenate_context,
                sep_token=sep_token,
            )
        )
    return flat


def resolve_split_paths(
    splits_dir: Path,
    *,
    kind: str,
    subset: str,
) -> list[Path]:
    """kind: intersentence|intrasentence; subset: train|test|all."""
    kinds = [kind] if kind != "both" else ["intersentence", "intrasentence"]
    paths: list[Path] = []
    for k in kinds:
        if subset in ("train", "all"):
            paths.append(splits_dir / f"{k}_train.json")
        if subset in ("test", "all"):
            paths.append(splits_dir / f"{k}_test.json")
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing StereoSet splits: " + ", ".join(map(str, missing)))
    return paths
