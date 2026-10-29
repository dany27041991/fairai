"""In-training StereoSet iCAT probe (no checkpoint reload)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from fairai.domain.icat import IcatBundle, bundle_from_probs
from fairai.infrastructure.data.stereoset_flatten import (
    FlatStereoExample,
    flatten_split_file,
    resolve_split_paths,
)


class _FlatDataset(torch.utils.data.Dataset):
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


@torch.no_grad()
def probe_avg_icat(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    splits_dir: Path,
    device: str,
    max_len: int = 512,
    batch_size: int = 64,
    subset: str = "test",
) -> float:
    """Average inter/intra iCAT for an in-memory classifier (AS-IS concat rules)."""
    model.eval()
    sep = tokenizer.sep_token or "[SEP]"
    bundles: dict[str, IcatBundle] = {}
    for kind in ("intersentence", "intrasentence"):
        concat = kind == "intersentence"
        examples: list[FlatStereoExample] = []
        for path in resolve_split_paths(splits_dir, kind=kind, subset=subset):
            examples.extend(
                flatten_split_file(path, concatenate_context=concat, sep_token=sep)
            )
        if not examples:
            continue
        loader = DataLoader(
            _FlatDataset(examples, tokenizer, max_len),
            batch_size=batch_size,
            shuffle=False,
        )
        rows: list[tuple[float, float, float] | None] = [None] * len(examples)
        for batch in loader:
            logits, _ = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            probs = torch.softmax(logits, dim=-1).cpu()
            for i, idx in enumerate(batch["index"].tolist()):
                p = probs[i].tolist()
                rows[idx] = (float(p[0]), float(p[1]), float(p[2]))
        bundles[kind] = bundle_from_probs(rows)  # type: ignore[arg-type]
    inter = bundles.get("intersentence")
    intra = bundles.get("intrasentence")
    if inter is not None and intra is not None:
        return (inter.icat + intra.icat) / 2.0
    if inter is not None:
        return float(inter.icat)
    if intra is not None:
        return float(intra.icat)
    return 0.0
