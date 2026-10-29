#!/usr/bin/env python3
"""Print Table B LaTeX rows + claim summary from optuna_fair_compare_agg.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/results/metrics/optuna_fair_compare_agg.json")
    if not path.is_file():
        print(f"missing {path}", file=sys.stderr)
        return 1
    rows = {r["family"]: r for r in json.loads(path.read_text())["rows"]}

    def fmt(fam: str, label: str) -> str:
        r = rows[fam]
        return (
            f"{label} & "
            f"${r['eval_f1_macro_mean']:.3f}\\pm{r['eval_f1_macro_std']:.3f}$ & "
            f"${r['avg_icat_mean']:.1f}\\pm{r['avg_icat_std']:.1f}$\\\\"
        )

    order = [
        ("van", "VAN full (3 seeds)"),
        ("no_bpl", "VAN w/o BPL"),
        ("ce", "Fine-tuned CE (no BPL)"),
    ]
    for key, label in order:
        # family names may be optuna_fair_X_best or stripped
        match = None
        for fam, r in rows.items():
            if key in fam.replace("-", "_"):
                match = fam
                break
        if match is None:
            print(f"% missing family matching {key}", file=sys.stderr)
            continue
        r = rows[match]
        print(
            f"{label} & "
            f"${r['eval_f1_macro_mean']:.3f}\\pm{r['eval_f1_macro_std']:.3f}$ & "
            f"${r['avg_icat_mean']:.1f}\\pm{r['avg_icat_std']:.1f}$\\\\"
        )

    # ranking helpers
    def score(r: dict) -> tuple[float, float]:
        return float(r["avg_icat_mean"]), float(r["eval_f1_macro_mean"])

    ranked_icat = sorted(rows.values(), key=lambda r: r["avg_icat_mean"], reverse=True)
    ranked_f1 = sorted(rows.values(), key=lambda r: r["eval_f1_macro_mean"], reverse=True)
    print("% best iCAT:", ranked_icat[0]["family"], ranked_icat[0]["avg_icat_mean"])
    print("% best F1:", ranked_f1[0]["family"], ranked_f1[0]["eval_f1_macro_mean"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
