"""Use-case: run VAN inference on free-form texts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from fairai.domain.labels import Label
from fairai.infrastructure.torch_van.checkpoint import load_van_and_tokenizer
from fairai.infrastructure.torch_van.thresholds import predict_from_logits


@dataclass(frozen=True, slots=True)
class InferenceItem:
    text: str
    label: int
    label_name: str
    probs: dict[str, float]
    abstained: bool = False
    prob_variance: dict[str, float] | None = None
    bias_warning: bool = False  # Thesis PROMPT TEST: warn if label ∈ {0, 1}


def infer_texts(
    texts: list[str],
    *,
    checkpoint_path: Path,
    config_path: Path,
    device: str = "cpu",
    apply_threshold_gates: bool = False,
    mc_samples: int = 0,
) -> list[InferenceItem]:
    """Infer labels. ``mc_samples>0`` enables Thesis MC dropout + dynamic gate."""
    if not texts:
        return []
    bundle = load_van_and_tokenizer(checkpoint_path, config_path, device=device)
    model, thresholds, tokenizer = bundle.model, bundle.thresholds, bundle.tokenizer
    enc = tokenizer(
        texts,
        max_length=bundle.max_len,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    var_probs = None
    if mc_samples > 0:
        from fairai.infrastructure.torch_van.thresholds import predict_mc_dropout

        pred_raw, probs, var_probs = predict_mc_dropout(
            model,
            input_ids,
            attention_mask,
            temp=thresholds["temp"],
            conf=thresholds["conf"],
            margin=thresholds["margin"],
            var=thresholds["var"],
            num_samples=mc_samples,
            abstain_as_neutral=False,
        )
        abstained_flags = (pred_raw < 0).tolist()
        pred = pred_raw.clone()
        pred[pred < 0] = int(Label.NEUTRAL)
    else:
        with torch.no_grad():
            logits, _ = model(input_ids, attention_mask)
            pred, probs = predict_from_logits(
                logits,
                temp=thresholds["temp"],
                conf=thresholds["conf"],
                margin=thresholds["margin"],
                var=thresholds["var"],
                apply_gates=apply_threshold_gates,
            )
        abstained_flags = [False] * len(texts)

    results: list[InferenceItem] = []
    for i, text in enumerate(texts):
        label_i = int(pred[i].item())
        if label_i < 0:
            label_i = int(Label.NEUTRAL)
        p = probs[i].cpu().tolist()
        vmap = None
        if var_probs is not None:
            vv = var_probs[i].cpu().tolist()
            vmap = {
                Label.STEREOTYPE.display_name: float(vv[0]),
                Label.ANTI_STEREOTYPE.display_name: float(vv[1]),
                Label.NEUTRAL.display_name: float(vv[2]),
            }
        results.append(
            InferenceItem(
                text=text,
                label=label_i,
                label_name=Label(label_i).display_name,
                probs={
                    Label.STEREOTYPE.display_name: float(p[0]),
                    Label.ANTI_STEREOTYPE.display_name: float(p[1]),
                    Label.NEUTRAL.display_name: float(p[2]),
                },
                abstained=bool(abstained_flags[i]),
                prob_variance=vmap,
                bias_warning=(
                    not bool(abstained_flags[i])
                    and label_i in (int(Label.STEREOTYPE), int(Label.ANTI_STEREOTYPE))
                ),
            )
        )
    return results
