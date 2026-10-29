#!/usr/bin/env python3
"""Print Table B LaTeX rows from optuna_fair_compare_agg.json (and optional WHM)."""

from __future__ import annotations

import json
import math
from pathlib import Path

LABELS = {
    "van": "VAN full (3 seeds)",
    "no_bpl": r"VAN w/o BPL",
    "ce": "Fine-tuned CE (no BPL)",
}


def _fmt(mean: float, std: float, digits: int = 3) -> str:
    if math.isnan(mean):
        return "---"
    if digits == 3 and mean < 1:
        return f"${mean:.{digits}f}\\pm{std:.{digits}f}$"
    return f"${mean:.{digits}f}\\pm{std:.{digits}f}$"


def main() -> None:
    path = Path("artifacts/results/metrics/optuna_fair_compare_agg.json")
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    by = {r["family"]: r for r in rows}
    order = ["van", "no_bpl", "ce"]
    print("% auto from", path)
    for key in order:
        # family names after aggregate strip: van / no_bpl / ce
        fam = None
        for k, r in by.items():
            if key in k.replace("optuna_fair_", "").replace("opta-", ""):
                fam = r
                break
        if fam is None:
            # try exact
            fam = by.get(key) or by.get(f"optuna_fair_{key}_best")
        if fam is None:
            print(f"% missing family {key}")
            continue
        f1 = _fmt(fam["eval_f1_macro_mean"], fam["eval_f1_macro_std"], 3)
        # iCAT often ~80-90
        ic = _fmt(fam["avg_icat_mean"], fam["avg_icat_std"], 1)
        label = LABELS.get(key, key)
        print(f"{label} & {f1} & {ic}\\\\")


if __name__ == "__main__":
    main()
