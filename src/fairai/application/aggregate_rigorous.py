"""Aggregate multi-seed rigorous suite rows → mean±std table."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from fairai.infrastructure.torch_van.checkpointing import write_json


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(var)


def _family(experiment: str | None, run_dir: str) -> str:
    name = (experiment or Path(run_dir).name).lower()
    name = re.sub(r"[-_]s\d+$", "", name)
    name = re.sub(r"^(rig-|rigc-|opta-|ovp-|released-)", "", name)
    return name


def aggregate_compare_table(
    compare_path: Path,
    *,
    out_path: Path | None = None,
) -> dict[str, Any]:
    raw = json.loads(Path(compare_path).read_text(encoding="utf-8"))
    rows = list(raw.get("rows", []))
    by_fam: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        fam = _family(row.get("experiment"), str(row.get("run_dir", "")))
        by_fam[fam].append(row)

    summary_rows: list[dict[str, Any]] = []
    for fam, items in sorted(by_fam.items()):
        f1s = [float(x["eval_f1_macro"]) for x in items if x.get("eval_f1_macro") is not None]
        icats = [float(x["avg_icat"]) for x in items if x.get("avg_icat") is not None]
        val_f1s = [
            float(x["best_val_f1_macro"])
            for x in items
            if x.get("best_val_f1_macro") is not None
        ]
        f1_m, f1_s = _mean_std(f1s)
        ic_m, ic_s = _mean_std(icats)
        vf_m, vf_s = _mean_std(val_f1s)
        summary_rows.append(
            {
                "family": fam,
                "n_seeds": len(items),
                "eval_f1_macro_mean": f1_m,
                "eval_f1_macro_std": f1_s,
                "avg_icat_mean": ic_m,
                "avg_icat_std": ic_s,
                "best_val_f1_macro_mean": vf_m,
                "best_val_f1_macro_std": vf_s,
                "run_dirs": [x.get("run_dir") for x in items],
            }
        )

    payload = {
        "source": str(compare_path),
        "n_families": len(summary_rows),
        "rows": summary_rows,
    }
    target = out_path or Path(compare_path).with_name(
        Path(compare_path).stem + "_agg.json"
    )
    write_json(target, payload)
    payload["out_path"] = str(target)
    return payload
