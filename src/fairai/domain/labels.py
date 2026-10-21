"""Domain labels for stereotype classification (pure Python)."""

from __future__ import annotations

from enum import IntEnum


class Label(IntEnum):
    """Three-way stereotype label used across FAIRAI datasets."""

    STEREOTYPE = 0
    ANTI_STEREOTYPE = 1
    NEUTRAL = 2

    @classmethod
    def from_int(cls, value: int) -> Label:
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"Invalid label {value!r}; expected 0, 1, or 2") from exc

    @property
    def display_name(self) -> str:
        return {
            Label.STEREOTYPE: "Stereotypical",
            Label.ANTI_STEREOTYPE: "Anti-stereotypical",
            Label.NEUTRAL: "Neutral",
        }[self]
