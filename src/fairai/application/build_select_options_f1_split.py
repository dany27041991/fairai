"""Build a leakage-free F1 test JSONL from StereoSet select-options."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fairai.domain.example import Example
from fairai.domain.labels import Label
from fairai.infrastructure.data.jsonl_repo import load_examples
from fairai.infrastructure.generation.normalize import normalize_sentence
from fairai.infrastructure.torch_van.checkpointing import write_json

SEP = "[SEP]"
MIN_PROMPTS_TARGET = 150


def leakage_key(text: str) -> str:
    return normalize_sentence(text).lower()


def concat_text(context: str, sentence: str, *, sep: str = SEP) -> str:
    ctx = (context or "").strip()
    sent = (sentence or "").strip()
    if ctx:
        return f"{ctx} {sep} {sent}"
    return sent


def load_forbidden_from_jsonl(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        for ex in load_examples(path):
            keys.add(leakage_key(ex.sentence))
    return keys


def load_forbidden_from_select_split(
    prompts: list[dict[str, Any]], *, sep: str = SEP
) -> set[str]:
    keys: set[str] = set()
    for pr in prompts:
        ctx = str(pr.get("context") or "")
        for opt in pr.get("options") or []:
            sent = str(opt.get("sentence") or "")
            keys.add(leakage_key(sent))
            keys.add(leakage_key(concat_text(ctx, sent, sep=sep)))
    return keys


def prompt_has_leakage(
    prompt: dict[str, Any], forbidden: set[str], *, sep: str = SEP
) -> bool:
    ctx = str(prompt.get("context") or "")
    for opt in prompt.get("options") or []:
        sent = str(opt.get("sentence") or "")
        if leakage_key(sent) in forbidden:
            return True
        if leakage_key(concat_text(ctx, sent, sep=sep)) in forbidden:
            return True
    return False


def flatten_prompt_to_examples(
    prompt: dict[str, Any], *, sep: str = SEP, source_split: str
) -> list[Example]:
    ctx = str(prompt.get("context") or "")
    item_bias = prompt.get("item_bias")
    out: list[Example] = []
    for i, opt in enumerate(prompt.get("options") or []):
        label = Label.from_int(int(opt["label"]))
        sent = str(opt.get("sentence") or "")
        bias = opt.get("bias_type") or item_bias
        out.append(
            Example(
                sentence=concat_text(ctx, sent, sep=sep),
                label=label,
                bias_type=str(bias) if bias else None,
                example_id=f"{source_split}:{i}:{leakage_key(sent)[:24]}",
            )
        )
    return out


def write_examples_jsonl(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_record(), ensure_ascii=False) + "\n")


@dataclass(slots=True)
class F1SplitReport:
    n_forbidden_keys: int
    n_prompts_test_in: int
    n_prompts_test_kept: int
    n_prompts_val_promoted: int
    n_prompts_total: int
    n_options: int
    overlap_exact_vs_forbidden: int
    by_item_bias: dict[str, int]
    by_label: dict[str, int]
    min_prompts_target: int
    below_target: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_select_options_f1_split(
    *,
    select_options_path: Path,
    corpus_jsonl_paths: list[Path],
    out_jsonl: Path,
    out_report: Path,
    sep: str = SEP,
    min_prompts: int = MIN_PROMPTS_TARGET,
) -> F1SplitReport:
    raw = json.loads(select_options_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict with train/validation/test in {select_options_path}")

    train_prompts = list(raw.get("train") or [])
    val_prompts = list(raw.get("validation") or [])
    test_prompts = list(raw.get("test") or [])

    forbidden = load_forbidden_from_jsonl(corpus_jsonl_paths)
    forbidden |= load_forbidden_from_select_split(train_prompts, sep=sep)

    kept_test = [p for p in test_prompts if not prompt_has_leakage(p, forbidden, sep=sep)]
    promoted: list[dict[str, Any]] = []
    if len(kept_test) < min_prompts:
        for p in val_prompts:
            if prompt_has_leakage(p, forbidden, sep=sep):
                continue
            promoted.append(p)
            if len(kept_test) + len(promoted) >= min_prompts:
                break

    selected = kept_test + promoted
    examples: list[Example] = []
    for p in kept_test:
        examples.extend(flatten_prompt_to_examples(p, sep=sep, source_split="test"))
    for p in promoted:
        examples.extend(flatten_prompt_to_examples(p, sep=sep, source_split="validation"))

    # Acceptance: zero exact match vs forbidden
    overlap = sum(1 for ex in examples if leakage_key(ex.sentence) in forbidden)
    if overlap:
        # Drop residual (should be rare if prompt-level filter worked)
        examples = [ex for ex in examples if leakage_key(ex.sentence) not in forbidden]
        overlap = sum(1 for ex in examples if leakage_key(ex.sentence) in forbidden)

    write_examples_jsonl(out_jsonl, examples)

    bias_counts = Counter((ex.bias_type or "null") for ex in examples)
    label_counts = Counter(str(int(ex.label)) for ex in examples)
    report = F1SplitReport(
        n_forbidden_keys=len(forbidden),
        n_prompts_test_in=len(test_prompts),
        n_prompts_test_kept=len(kept_test),
        n_prompts_val_promoted=len(promoted),
        n_prompts_total=len(selected),
        n_options=len(examples),
        overlap_exact_vs_forbidden=overlap,
        by_item_bias=dict(sorted(bias_counts.items())),
        by_label=dict(sorted(label_counts.items())),
        min_prompts_target=min_prompts,
        below_target=len(selected) < min_prompts,
    )
    payload = report.to_dict()
    payload["paths"] = {
        "select_options": str(select_options_path),
        "out_jsonl": str(out_jsonl),
        "corpus": [str(p) for p in corpus_jsonl_paths],
    }
    write_json(out_report, payload)
    return report
