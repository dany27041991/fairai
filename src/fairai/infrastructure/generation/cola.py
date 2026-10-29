"""Optional CoLA grammar filter (Thesis GenDataset ``verify_grammar``)."""

from __future__ import annotations

import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "textattack/bert-base-uncased-CoLA"


def filter_cola_sentences(
    sentences: Sequence[str],
    *,
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = 4,
    device: int | str = -1,
) -> list[str]:
    """Keep sentences predicted acceptable by CoLA (LABEL_1).

    No-op (returns input) if ``transformers`` / model unavailable.
    """
    if not sentences:
        return []
    try:
        from transformers import pipeline
    except ImportError:
        logger.info("transformers missing — CoLA filter skipped")
        return list(sentences)

    try:
        clf = pipeline("text-classification", model=model_name, device=device)
    except Exception as exc:  # noqa: BLE001
        logger.info("CoLA model load failed (%s) — filter skipped", exc)
        return list(sentences)

    kept: list[str] = []
    for i in range(0, len(sentences), batch_size):
        batch = list(sentences[i : i + batch_size])
        results = clf(batch)
        for sentence, result in zip(batch, results, strict=True):
            label = result.get("label") if isinstance(result, dict) else None
            if label in ("LABEL_1", "1", "acceptable"):
                kept.append(sentence)
    # preserve order, unique
    return list(dict.fromkeys(kept))
