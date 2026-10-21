"""Tokenized Dataset for labeled JSONL examples."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from fairai.domain.example import Example


class EncodedDataset(Dataset):
    def __init__(
        self,
        examples: list[Example],
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
    ) -> None:
        sentences = [ex.sentence for ex in examples]
        enc = tokenizer(
            sentences,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor([int(ex.label) for ex in examples], dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.size(0))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }
