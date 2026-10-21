"""Enforce per-file line budget (coding standard: max 500 LOC)."""

from __future__ import annotations

from pathlib import Path


def count_lines(path: Path) -> int:
    return sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))


def find_oversized_python_files(
    root: Path,
    *,
    limit: int = 500,
) -> list[tuple[Path, int]]:
    root = root.resolve()
    offenders: list[tuple[Path, int]] = []
    for path in sorted(root.rglob("*.py")):
        n = count_lines(path)
        if n > limit:
            offenders.append((path, n))
    return offenders
