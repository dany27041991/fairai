"""Audit / optionally strip residual StereoSet-test leakage from a train JSONL (GAP-D4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fairai.infrastructure.data.jsonl_repo import load_examples
from fairai.infrastructure.generation.jsonl_io import write_examples
from fairai.infrastructure.generation.leakage import (
    filter_leakage,
    stereoset_forbidden_sentences,
)
from fairai.infrastructure.generation.normalize import normalize_sentence
from fairai.infrastructure.torch_van.checkpointing import write_json


@dataclass(slots=True)
class LeakageAuditReport:
    dataset: str
    splits_dir: str
    n_total: int
    n_forbidden_keys: int
    n_hits: int
    hit_rate: float
    sample_hits: list[str]
    out_strict: str | None
    n_written: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_train_leakage(
    dataset_path: Path,
    *,
    splits_dir: Path,
    out_strict: Path | None = None,
    out_report: Path | None = None,
    max_sample_hits: int = 20,
    max_samples: int | None = None,
) -> LeakageAuditReport:
    """Report residual overlaps; optionally write a stricter JSONL (never overwrites input)."""
    examples = load_examples(dataset_path, max_samples=max_samples)
    forbidden = stereoset_forbidden_sentences(splits_dir)
    hits = [
        ex
        for ex in examples
        if normalize_sentence(ex.sentence).lower() in forbidden
    ]
    n_hits = len(hits)
    written: int | None = None
    out_path: str | None = None
    if out_strict is not None:
        if out_strict.resolve() == dataset_path.resolve():
            raise ValueError(
                "Refuse to overwrite canonical dataset; choose a different --out-strict"
            )
        kept, removed = filter_leakage(examples, forbidden)
        n_hits = removed
        written = write_examples(out_strict, kept)
        out_path = str(out_strict)

    report = LeakageAuditReport(
        dataset=str(dataset_path),
        splits_dir=str(splits_dir),
        n_total=len(examples),
        n_forbidden_keys=len(forbidden),
        n_hits=n_hits,
        hit_rate=(n_hits / len(examples)) if examples else 0.0,
        sample_hits=[ex.sentence[:200] for ex in hits[:max_sample_hits]],
        out_strict=out_path,
        n_written=written,
    )
    if out_report is not None:
        write_json(out_report, report.to_dict())
    return report
