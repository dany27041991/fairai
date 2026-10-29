"""Use-case: LM / SS / ICAT on flattened StereoSet splits with VAN."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from fairai.domain.icat import IcatBundle, bundle_from_probs
from fairai.infrastructure.data.stereoset_flatten import (
    FlatStereoExample,
    flatten_split_file,
    resolve_split_paths,
)
from fairai.infrastructure.torch_van.checkpoint import load_van_and_tokenizer


@dataclass(slots=True)
class IcatReport:
    intersentence: IcatBundle | None
    intrasentence: IcatBundle | None
    avg_icat: float | None
    by_bias_type: dict[str, dict]
    by_bias_type_intersentence: dict[str, dict]
    by_bias_type_intrasentence: dict[str, dict]
    n_total: int
    subset: str
    concatenate_context: bool

    def to_dict(self) -> dict:
        return {
            "intersentence": self.intersentence.to_dict() if self.intersentence else None,
            "intrasentence": self.intrasentence.to_dict() if self.intrasentence else None,
            "avg_icat": self.avg_icat,
            # Pooled across splits (baseline / docs); Thesis also prints per-split below.
            "by_bias_type": self.by_bias_type,
            "by_bias_type_intersentence": self.by_bias_type_intersentence,
            "by_bias_type_intrasentence": self.by_bias_type_intrasentence,
            "n_total": self.n_total,
            "subset": self.subset,
            "concatenate_context": self.concatenate_context,
        }


class _FlatDataset(Dataset):
    def __init__(
        self, examples: list[FlatStereoExample], tokenizer: Any, max_len: int
    ) -> None:
        texts = [ex.sentence for ex in examples]
        enc = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "index": idx,
        }


def _collect_probs(
    model: torch.nn.Module,
    examples: list[FlatStereoExample],
    tokenizer: Any,
    *,
    device: str,
    max_len: int,
    batch_size: int,
) -> list[tuple[float, float, float]]:
    loader = DataLoader(
        _FlatDataset(examples, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=False,
    )
    rows: list[tuple[float, float, float] | None] = [None] * len(examples)
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            probs = torch.softmax(logits, dim=-1).cpu()
            for i, idx in enumerate(batch["index"].tolist()):
                p = probs[i].tolist()
                rows[idx] = (float(p[0]), float(p[1]), float(p[2]))
    assert all(r is not None for r in rows)
    return rows  # type: ignore[return-value]


def evaluate_icat(
    *,
    checkpoint_path: Path,
    config_path: Path,
    splits_dir: Path,
    subset: str = "all",
    device: str = "cpu",
    batch_size: int = 32,
    concatenate_context: bool | None = None,
    report_bias_types: bool = True,
) -> IcatReport:
    # concatenate_context=None → AS-IS defaults (inter True / intra False)
    # False → force no concat for both; True → force concat for both
    bundle = load_van_and_tokenizer(checkpoint_path, config_path, device=device)
    model, tokenizer, max_len = bundle.model, bundle.tokenizer, bundle.max_len
    sep = tokenizer.sep_token or "[SEP]"

    bundles: dict[str, IcatBundle] = {}
    bias_rows: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    bias_rows_by_kind: dict[str, dict[str, list[tuple[float, float, float]]]] = {
        "intersentence": defaultdict(list),
        "intrasentence": defaultdict(list),
    }
    n_total = 0

    for kind in ("intersentence", "intrasentence"):
        # AS-IS Thesis: inter=True (context [SEP] sentence), intra=False (filled sentence only)
        concat = True if kind == "intersentence" else False
        if concatenate_context is False:
            concat = False
        paths = resolve_split_paths(splits_dir, kind=kind, subset=subset)
        examples: list[FlatStereoExample] = []
        for path in paths:
            examples.extend(
                flatten_split_file(path, concatenate_context=concat, sep_token=sep)
            )
        if not examples:
            continue
        probs = _collect_probs(
            model,
            examples,
            tokenizer,
            device=device,
            max_len=max_len,
            batch_size=batch_size,
        )
        bundles[kind] = bundle_from_probs(probs)
        n_total += len(examples)
        if report_bias_types:
            for ex, row in zip(examples, probs, strict=True):
                bias_rows[ex.bias_type].append(row)
                bias_rows_by_kind[kind][ex.bias_type].append(row)

    inter = bundles.get("intersentence")
    intra = bundles.get("intrasentence")
    avg = None
    if inter is not None and intra is not None:
        avg = (inter.icat + intra.icat) / 2.0
    elif inter is not None:
        avg = inter.icat
    elif intra is not None:
        avg = intra.icat

    by_bias = {
        bias: bundle_from_probs(rows).to_dict()
        for bias, rows in sorted(bias_rows.items())
    }
    by_bias_inter = {
        bias: bundle_from_probs(rows).to_dict()
        for bias, rows in sorted(bias_rows_by_kind["intersentence"].items())
    }
    by_bias_intra = {
        bias: bundle_from_probs(rows).to_dict()
        for bias, rows in sorted(bias_rows_by_kind["intrasentence"].items())
    }
    if not report_bias_types:
        by_bias, by_bias_inter, by_bias_intra = {}, {}, {}

    return IcatReport(
        intersentence=inter,
        intrasentence=intra,
        avg_icat=avg,
        by_bias_type=by_bias,
        by_bias_type_intersentence=by_bias_inter,
        by_bias_type_intrasentence=by_bias_intra,
        n_total=n_total,
        subset=subset,
        concatenate_context=(
            False if concatenate_context is False else True  # True means AS-IS defaults used
        ),
    )


def write_gold_scores_json(
    *,
    checkpoint_path: Path,
    config_path: Path,
    splits_dir: Path,
    out_path: Path,
    subset: str = "all",
    device: str = "cpu",
    batch_size: int = 32,
) -> Path:
    """Thesis cell-3 main: write ``gold_scores.json`` with inter + intra lists."""
    inter = generate_gold_score_entries(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        splits_dir=splits_dir,
        kind="intersentence",
        subset=subset,
        device=device,
        batch_size=batch_size,
        concatenate_context=True,
    )
    intra = generate_gold_score_entries(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        splits_dir=splits_dir,
        kind="intrasentence",
        subset=subset,
        device=device,
        batch_size=batch_size,
        concatenate_context=False,
    )
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"intersentence": inter, "intrasentence": intra}, indent=4) + "\n",
        encoding="utf-8",
    )
    return out_path


def generate_gold_score_entries(
    *,
    checkpoint_path: Path,
    config_path: Path,
    splits_dir: Path,
    kind: str,
    subset: str = "all",
    device: str = "cpu",
    batch_size: int = 32,
    concatenate_context: bool = True,
) -> list[dict]:
    """Reproduce AS-IS gold_scores entries: P(model)[gold_label] per sentence id."""
    bundle = load_van_and_tokenizer(checkpoint_path, config_path, device=device)
    model, tokenizer, max_len = bundle.model, bundle.tokenizer, bundle.max_len
    sep = tokenizer.sep_token or "[SEP]"

    paths = resolve_split_paths(splits_dir, kind=kind, subset=subset)
    examples: list[FlatStereoExample] = []
    for path in paths:
        examples.extend(
            flatten_split_file(path, concatenate_context=concatenate_context, sep_token=sep)
        )
    loader = DataLoader(
        _FlatDataset(examples, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=False,
    )
    results: list[dict | None] = [None] * len(examples)
    with torch.no_grad():
        for batch in loader:
            logits, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            probs = torch.softmax(logits, dim=-1).cpu()
            for i, idx in enumerate(batch["index"].tolist()):
                gold = examples[idx].label
                results[idx] = {
                    "id": examples[idx].example_id,
                    "score": float(probs[i][gold].item()),
                }
    assert all(r is not None for r in results)
    return results  # type: ignore[return-value]
