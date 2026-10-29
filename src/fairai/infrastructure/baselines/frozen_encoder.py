"""Frozen MiniLM mean-pool embeddings for linear baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class _TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: Any, max_len: int) -> None:
        enc = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self) -> int:
        return int(self.input_ids.size(0))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


@torch.no_grad()
def encode_texts(
    texts: list[str],
    *,
    backbone: str,
    device: str = "cpu",
    max_len: int = 512,
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """Return ``[N, H]`` float32 embeddings (optionally L2-normalized)."""
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModel.from_pretrained(backbone).to(device)
    model.eval()
    loader = DataLoader(
        _TextDataset(texts, tokenizer, max_len),
        batch_size=batch_size,
        shuffle=False,
    )
    chunks: list[np.ndarray] = []
    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        pooled = mean_pool(out.last_hidden_state, batch["attention_mask"].to(device))
        if normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        chunks.append(pooled.cpu().numpy().astype(np.float32))
    del model
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


@dataclass(slots=True)
class FrozenEncoder:
    """Reusable encoder handle for train + iCAT (loads backbone once)."""

    backbone: str
    device: str = "cpu"
    max_len: int = 512
    normalize: bool = True
    tokenizer: Any = field(init=False, repr=False)
    model: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone)
        self.model = AutoModel.from_pretrained(self.backbone).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, texts: list[str], *, batch_size: int = 64) -> np.ndarray:
        if not texts:
            hidden = int(self.model.config.hidden_size)
            return np.zeros((0, hidden), dtype=np.float32)
        loader = DataLoader(
            _TextDataset(texts, self.tokenizer, self.max_len),
            batch_size=batch_size,
            shuffle=False,
        )
        chunks: list[np.ndarray] = []
        for batch in loader:
            out = self.model(
                input_ids=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
            )
            pooled = mean_pool(out.last_hidden_state, batch["attention_mask"].to(self.device))
            if self.normalize:
                pooled = F.normalize(pooled, p=2, dim=-1)
            chunks.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)
