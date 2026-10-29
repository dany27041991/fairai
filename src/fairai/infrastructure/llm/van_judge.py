"""Score MC options with a frozen VAN classifier (no generative LLM required)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fairai.infrastructure.torch_van.checkpoint import load_van_and_tokenizer


class VanOptionJudge:
    """Pick the option with highest P(neutral) — aligns with AS-IS few-shot teaching."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        config: Path,
        device: str = "cpu",
        prefer_label: int = 2,
    ) -> None:
        bundle = load_van_and_tokenizer(checkpoint, config, device=device)
        self.model = bundle.model
        self.device = device
        self.prefer_label = prefer_label
        self.max_len = bundle.max_len
        self.tokenizer = bundle.tokenizer

    @torch.no_grad()
    def score_sentence(self, sentence: str) -> torch.Tensor:
        enc = self.tokenizer(
            sentence,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        logits, _ = self.model(
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device),
        )
        return torch.softmax(logits[0].float().cpu(), dim=-1)

    def predict_label(self, example: dict[str, Any]) -> int | None:
        options = example.get("options") or []
        if not options:
            return None
        best_i = 0
        best_score = -1.0
        for i, opt in enumerate(options):
            probs = self.score_sentence(str(opt.get("sentence", "")))
            score = float(probs[self.prefer_label])
            if score > best_score:
                best_score = score
                best_i = i
        return int(options[best_i]["label"])
