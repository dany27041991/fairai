"""Validate VAN training JSONL schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fairai.domain.example import Example
from fairai.domain.labels import Label


@dataclass(frozen=True, slots=True)
class SchemaIssue:
    line: int
    message: str


def validate_record(record: dict[str, Any], *, line: int = 0) -> list[SchemaIssue]:
    issues: list[SchemaIssue] = []
    if "sentence" not in record:
        issues.append(SchemaIssue(line, "missing sentence"))
    elif not str(record["sentence"]).strip():
        issues.append(SchemaIssue(line, "empty sentence"))
    if "label" not in record:
        issues.append(SchemaIssue(line, "missing label"))
    else:
        try:
            Label.from_int(int(record["label"]))
        except Exception as exc:  # noqa: BLE001
            issues.append(SchemaIssue(line, f"bad label: {exc}"))
    return issues


def validate_examples(examples: list[Example]) -> list[SchemaIssue]:
    issues: list[SchemaIssue] = []
    for i, ex in enumerate(examples, start=1):
        issues.extend(validate_record(ex.to_record(), line=i))
    return issues
