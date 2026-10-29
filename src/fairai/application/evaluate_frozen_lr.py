"""F1 / accuracy eval for frozen MiniLM + LogisticRegression runs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from fairai.application.evaluate_classification import ClassificationReport
from fairai.domain.labels import Label
from fairai.infrastructure.baselines.frozen_encoder import encode_texts
from fairai.infrastructure.data.jsonl_repo import load_examples


def evaluate_frozen_lr(
    *,
    model_path: Path,
    dataset_path: Path,
    device: str = "cpu",
    batch_size: int = 64,
    max_samples: int | None = None,
) -> ClassificationReport:
    bundle = joblib.load(model_path)
    clf = bundle["classifier"]
    backbone = str(bundle["backbone"])
    max_len = int(bundle.get("max_len", 512))

    examples = load_examples(dataset_path, max_samples=max_samples)
    if not examples:
        raise ValueError(f"No examples in {dataset_path}")

    x = encode_texts(
        [ex.sentence for ex in examples],
        backbone=backbone,
        device=device,
        max_len=max_len,
        batch_size=batch_size,
        normalize=bool(bundle.get("normalize", True)),
    )
    y_true = [int(ex.label) for ex in examples]
    y_pred = [int(p) for p in clf.predict(x).tolist()]

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=[Label(i).display_name for i in range(3)],
        output_dict=True,
        zero_division=0,
    )
    dist_true = {Label(i).name: y_true.count(i) for i in range(3)}
    dist_pred = {Label(i).name: y_pred.count(i) for i in range(3)}
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    by_bias: dict[str, dict] = {}
    groups: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        groups[ex.bias_type or "null"].append(i)
    for bias, idxs in sorted(groups.items()):
        yt = [y_true[i] for i in idxs]
        yp = [y_pred[i] for i in idxs]
        by_bias[bias] = {
            "n": len(idxs),
            "f1_macro": float(
                f1_score(yt, yp, average="macro", labels=[0, 1, 2], zero_division=0)
            ),
            "accuracy": float(accuracy_score(yt, yp)),
        }
    return ClassificationReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        n_samples=len(y_true),
        per_class=report,
        label_distribution_pred=dist_pred,
        label_distribution_true=dist_true,
        confusion_matrix=cm.tolist(),
        y_true=list(y_true),
        y_pred=list(y_pred),
        by_bias_type=by_bias,
    )
