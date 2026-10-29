#!/usr/bin/env python3
"""Compare processed jsonl line counts to AS-IS expected table."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "van_train_leakage_safe.jsonl": 50409,
    "van_final.jsonl": 55817,
    "van_minimal_24k.jsonl": 24000,
}


def main() -> int:
    processed = ROOT / "artifacts/datasets/processed"
    report: dict[str, dict] = {}
    ok = True
    for name, expected in EXPECTED.items():
        path = processed / name
        n = sum(1 for _ in path.open(encoding="utf-8")) if path.is_file() else -1
        match = n == expected
        ok = ok and match
        report[name] = {"expected": expected, "actual": n, "pass": match}
    out = ROOT / "docs/verification/EVIDENCE/wave_a_counts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
