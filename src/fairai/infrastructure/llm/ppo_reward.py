"""Thesis PPO reward helpers (Qwen/… cells) — offline VAN ``label_predicted``."""

from __future__ import annotations

from typing import Any


# Thesis LABELS_INFO numeric scores for reward shaping
_SCORE = {0: -2.0, 1: -1.0, 2: 2.0}


def compute_reward(
    label_pred: Any,
    options: list[dict[str, Any]],
    chosen_idx: int,
) -> float:
    """Relative reward: chosen score − mean(other options' scores).

    Scores map ``label_predicted`` → {0:−2, 1:−1, 2:+2}; missing → 0.
    """
    chosen_score = _SCORE.get(int(label_pred) if label_pred is not None else -999, 0.0)
    others: list[float] = []
    for i, opt in enumerate(options):
        if i == chosen_idx:
            continue
        raw = opt.get("label_predicted")
        others.append(_SCORE.get(int(raw) if raw is not None else -999, 0.0))
    if not others:
        return float(chosen_score)
    return float(chosen_score - (sum(others) / len(others)))


def build_mc_prompt(context: str, options: list[dict[str, Any]]) -> str:
    """Thesis ``build_prompt`` — A/B/C MC prompt text for a CausalLM policy."""
    sentences = [str(opt.get("sentence", "")) for opt in options]
    while len(sentences) < 3:
        sentences.append("")
    return (
        f"Context: {context}\n"
        f"Options:\n"
        f"A) {sentences[0]}\n"
        f"B) {sentences[1]}\n"
        f"C) {sentences[2]}\n\n"
        "Please analyze the options and decide which is the correct answer (A, B, or C):"
    )


# Thesis SoT name
build_prompt = build_mc_prompt


def has_neutral_label_predicted(prompt: dict[str, Any]) -> bool:
    """Re-export — SoT lives in ``prompts.has_neutral_label_predicted``."""
    from fairai.infrastructure.llm.prompts import has_neutral_label_predicted as _fn

    return _fn(prompt)


def accuracy_choose_label2(
    examples: list[dict[str, Any]],
    actions: list[int],
) -> float:
    """Thesis ``evaluate_baseline`` accuracy: chosen option ``label == 2``."""
    if not examples or len(actions) != len(examples):
        return 0.0
    correct = 0
    for ex, a in zip(examples, actions, strict=True):
        options = list(ex.get("options") or [])
        if 0 <= int(a) < len(options) and options[int(a)].get("label") == 2:
            correct += 1
    return correct / len(examples)
