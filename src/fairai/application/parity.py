"""Compare metric dicts against a baseline with absolute tolerances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParityCheck:
    key: str
    baseline: float
    actual: float
    tol: float
    ok: bool


def check_parity(
    actual: dict[str, Any],
    baseline: dict[str, Any],
    tolerances: dict[str, float],
) -> list[ParityCheck]:
    """``tolerances`` maps dotted keys (e.g. ``icat.avg_icat``) to abs tol."""

    def dig(data: dict[str, Any], dotted: str) -> float:
        cur: Any = data
        for part in dotted.split("."):
            cur = cur[part]
        return float(cur)

    results: list[ParityCheck] = []
    for key, tol in tolerances.items():
        base = dig(baseline, key)
        act = dig(actual, key)
        results.append(
            ParityCheck(
                key=key,
                baseline=base,
                actual=act,
                tol=float(tol),
                ok=abs(act - base) <= float(tol),
            )
        )
    return results


def parity_ok(checks: list[ParityCheck]) -> bool:
    return all(c.ok for c in checks)
