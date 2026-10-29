"""Sample + heuristic audit of neutral (label=2) select-options / F1-test rows."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fairai.infrastructure.data.jsonl_repo import load_examples
from fairai.infrastructure.torch_van.checkpointing import write_json

AUDIT_LABELS = ("neutral", "subtle_stereo", "unrelated_junk", "unclear")


def _heuristic_flag(sentence: str) -> str:
    """Cheap automatic flag (not a substitute for human audit)."""
    text = sentence.strip()
    # Prefer option sentence after [SEP] if present
    if "[SEP]" in text:
        text = text.split("[SEP]", 1)[-1].strip()
    words = text.split()
    if len(words) <= 2:
        return "unrelated_junk"
    if len(words) <= 4 and text.lower() in {
        "and it was!",
        "and it was.",
        "i see.",
        "ok.",
        "yes.",
        "no.",
    }:
        return "unrelated_junk"
    return "neutral"


@dataclass(slots=True)
class NeutralAuditReport:
    n_neutrals_pool: int
    n_sample: int
    n_controls: int
    heuristic_flags: dict[str, int]
    heuristic_junk_rate: float
    sample_path: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_neutrals(
    *,
    f1_jsonl: Path,
    select_options_path: Path | None,
    out_sample: Path,
    out_report: Path,
    n_neutral: int = 180,
    n_control: int = 50,
    seed: int = 42,
) -> NeutralAuditReport:
    examples = load_examples(f1_jsonl)
    neutrals = [ex for ex in examples if int(ex.label) == 2]
    controls = [ex for ex in examples if int(ex.label) in (0, 1)]
    rng = random.Random(seed)
    neu_sample = neutrals if len(neutrals) <= n_neutral else rng.sample(neutrals, n_neutral)
    ctrl_sample = (
        controls if len(controls) <= n_control else rng.sample(controls, n_control)
    )

    flags = Counter(_heuristic_flag(ex.sentence) for ex in neu_sample)
    junk_rate = flags.get("unrelated_junk", 0) / max(len(neu_sample), 1)

    rows: list[dict[str, Any]] = []
    for ex in neu_sample:
        rows.append(
            {
                "sentence": ex.sentence,
                "gold_label": 2,
                "bias_type": ex.bias_type,
                "heuristic": _heuristic_flag(ex.sentence),
                "human_label": None,
                "allowed": list(AUDIT_LABELS),
            }
        )
    for ex in ctrl_sample:
        rows.append(
            {
                "sentence": ex.sentence,
                "gold_label": int(ex.label),
                "bias_type": ex.bias_type,
                "heuristic": None,
                "human_label": None,
                "allowed": list(AUDIT_LABELS),
                "control": True,
            }
        )
    rng.shuffle(rows)
    out_sample.parent.mkdir(parents=True, exist_ok=True)
    out_sample.write_text(
        json.dumps({"schema": AUDIT_LABELS, "items": rows}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    extra = {}
    if select_options_path and select_options_path.is_file():
        extra["select_options"] = str(select_options_path)

    report = NeutralAuditReport(
        n_neutrals_pool=len(neutrals),
        n_sample=len(neu_sample),
        n_controls=len(ctrl_sample),
        heuristic_flags=dict(sorted(flags.items())),
        heuristic_junk_rate=float(junk_rate),
        sample_path=str(out_sample),
        note=(
            "Fill human_label in the sample JSON; heuristic_junk_rate is automatic only. "
            "If human junk rate > 0.10, filter those items from F1-test and re-eval."
        ),
    )
    payload = report.to_dict()
    payload.update(extra)
    write_json(out_report, payload)
    return report


def filter_f1_jsonl_by_heuristic(
    *,
    in_jsonl: Path,
    out_jsonl: Path,
    drop_flags: frozenset[str] = frozenset({"unrelated_junk"}),
) -> dict[str, int]:
    """Drop label-2 rows whose heuristic flag is in drop_flags; keep stereo/anti."""
    kept = []
    dropped = 0
    for ex in load_examples(in_jsonl):
        if int(ex.label) == 2 and _heuristic_flag(ex.sentence) in drop_flags:
            dropped += 1
            continue
        kept.append(ex)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for ex in kept:
            fh.write(json.dumps(ex.to_record(), ensure_ascii=False) + "\n")
    return {"kept": len(kept), "dropped_neutral_junk": dropped}
