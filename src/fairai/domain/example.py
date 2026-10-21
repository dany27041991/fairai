"""Domain example entity (pure Python, no torch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fairai.domain.labels import Label


@dataclass(frozen=True, slots=True)
class Example:
    """One labeled sentence (optionally with bias type metadata)."""

    sentence: str
    label: Label
    bias_type: str | None = None
    example_id: str | None = None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "sentence": self.sentence,
            "label": int(self.label),
        }
        if self.bias_type is not None:
            record["bias_type"] = self.bias_type
        if self.example_id is not None:
            record["id"] = self.example_id
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Example:
        if "sentence" not in record or "label" not in record:
            raise KeyError("record must contain 'sentence' and 'label'")
        bias = record.get("bias_type")
        if bias is None or bias == "none" or bias == "":
            bias = None
        return cls(
            sentence=str(record["sentence"]),
            label=Label.from_int(int(record["label"])),
            bias_type=str(bias) if bias is not None else None,
            example_id=str(record["id"]) if "id" in record else None,
        )
