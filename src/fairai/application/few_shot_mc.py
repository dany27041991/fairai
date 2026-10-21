"""Few-shot StereoSet MC evaluation (VAN judge smoke / documented run)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score

from fairai.infrastructure.artifact_paths import ArtifactPaths
from fairai.infrastructure.llm.mc_dataset import load_mc_file
from fairai.infrastructure.llm.prompts import (
    build_few_shot_prompt,
    build_test_prompt,
    gold_label,
)
from fairai.infrastructure.llm.van_judge import VanOptionJudge
from fairai.infrastructure.torch_van.checkpointing import write_json


@dataclass(slots=True)
class FewShotResult:
    n: int
    accuracy: float
    f1_macro: float
    backend: str
    metrics_path: str
    sample_prompt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_few_shot_mc(
    *,
    dataset_path: Path,
    checkpoint: Path,
    config: Path,
    device: str = "cpu",
    split: str = "test",
    max_examples: int = 64,
    n_shots: int = 3,
    artifacts: ArtifactPaths | None = None,
    out_name: str = "llm_fewshot_van.json",
) -> FewShotResult:
    paths = artifacts or ArtifactPaths.from_env()
    splits = load_mc_file(dataset_path)
    train = splits.get("train") or []
    test = splits.get(split) or splits.get("test") or []
    if not test:
        raise ValueError(f"No examples in split={split}")

    shots = train[:n_shots]
    prefix = build_few_shot_prompt(shots) if shots else ""
    sample_prompt = build_test_prompt(test[0], few_shot_prefix=prefix)

    judge = VanOptionJudge(checkpoint=checkpoint, config=config, device=device)
    y_true: list[int] = []
    y_pred: list[int] = []
    for ex in test[:max_examples]:
        gold = gold_label(ex)
        pred = judge.predict_label(ex)
        if gold is None or pred is None:
            continue
        y_true.append(gold)
        y_pred.append(pred)

    if not y_true:
        raise RuntimeError("No scorable MC examples")

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out = paths.metrics_dir() / out_name
    payload = {
        "backend": "van_neutral_judge",
        "dataset": str(dataset_path),
        "split": split,
        "n": len(y_true),
        "n_shots": n_shots,
        "accuracy": acc,
        "f1_macro": f1,
        "note": (
            "Generative HF LLM optional; this smoke uses frozen VAN v2 to pick the "
            "option with highest P(neutral), matching AS-IS few-shot teaching bias."
        ),
    }
    write_json(out, payload)
    return FewShotResult(
        n=len(y_true),
        accuracy=acc,
        f1_macro=f1,
        backend="van_neutral_judge",
        metrics_path=str(out),
        sample_prompt=sample_prompt[:2000],
    )
