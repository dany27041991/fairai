"""Text normalization helpers for dataset generation."""

from __future__ import annotations

import re


def normalize_sentence(sentence: str) -> str:
    text = re.sub(r"\s+", " ", sentence)
    text = re.sub(r"\s([?.!,\"](?:\s|$))", r"\1", text)
    return text.strip()


def normalize_punctuation(text: str) -> str:
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ensure_period(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def capitalize_sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]
