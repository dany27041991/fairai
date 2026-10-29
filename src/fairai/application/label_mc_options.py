"""Thesis cell 5 — ``update_dataset_with_predictions`` on StereoSet MC JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fairai.infrastructure.torch_van.checkpoint import load_van_and_tokenizer
from fairai.infrastructure.torch_van.checkpointing import write_json
from fairai.infrastructure.torch_van.thresholds import predict_mc_dropout


@dataclass(slots=True)
class LabelMcReport:
    input_path: str
    output_path: str
    n_options: int
    n_predicted: int
    n_abstained: int
    splits: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "n_options": self.n_options,
            "n_predicted": self.n_predicted,
            "n_abstained": self.n_abstained,
            "splits": self.splits,
        }


def update_dataset_with_predictions(
    input_path: Path,
    output_path: Path,
    *,
    checkpoint_path: Path,
    config_path: Path,
    device: str = "cpu",
    mc_samples: int = 10,
    max_len: int | None = None,
    splits: tuple[str, ...] | None = None,
) -> LabelMcReport:
    """Label each MC option with Thesis MC-dropout gates.

    Text = ``context [SEP] option_sentence`` (Thesis cell 5).
    Abstentions → ``label_predicted: null``.
    """
    bundle = load_van_and_tokenizer(
        checkpoint_path, config_path, device=device, default_max_len=max_len or 512
    )
    model, thresholds, tokenizer = bundle.model, bundle.thresholds, bundle.tokenizer
    max_len = int(max_len or bundle.max_len)
    sep = tokenizer.sep_token or "[SEP]"

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object with split keys (train/validation/test)")

    split_keys = list(splits) if splits else [k for k, v in data.items() if isinstance(v, list)]
    n_options = 0
    n_pred = 0
    n_abs = 0

    for split in split_keys:
        items = data.get(split, [])
        if not isinstance(items, list):
            continue
        for item in items:
            context = str(item.get("context", ""))
            for option in item.get("options", []):
                n_options += 1
                sentence = str(option.get("sentence", ""))
                text = f"{context} {sep} {sentence}".strip() if context else sentence
                enc = tokenizer(
                    text,
                    max_length=max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                pred, _, _ = predict_mc_dropout(
                    model,
                    enc["input_ids"].to(device),
                    enc["attention_mask"].to(device),
                    temp=thresholds["temp"],
                    conf=thresholds["conf"],
                    margin=thresholds["margin"],
                    var=thresholds["var"],
                    num_samples=mc_samples,
                    abstain_as_neutral=False,
                )
                label_i = int(pred[0].item())
                if label_i < 0:
                    option["label_predicted"] = None
                    n_abs += 1
                else:
                    option["label_predicted"] = label_i
                    n_pred += 1

    write_json(output_path, data, ensure_ascii=False)
    return LabelMcReport(
        input_path=str(input_path),
        output_path=str(output_path),
        n_options=n_options,
        n_predicted=n_pred,
        n_abstained=n_abs,
        splits=split_keys,
    )
