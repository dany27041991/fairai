"""StereoSet ICAT for frozen MiniLM + LogisticRegression."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

from fairai.application.evaluate_icat import IcatReport
from fairai.domain.icat import IcatBundle, bundle_from_probs
from fairai.infrastructure.baselines.frozen_encoder import FrozenEncoder
from fairai.infrastructure.data.stereoset_flatten import (
    FlatStereoExample,
    flatten_split_file,
    resolve_split_paths,
)


def _probs_for_examples(
    examples: list[FlatStereoExample],
    *,
    encoder: FrozenEncoder,
    clf: object,
    batch_size: int,
) -> list[tuple[float, float, float]]:
    texts = [ex.sentence for ex in examples]
    x = encoder.encode(texts, batch_size=batch_size)
    proba = np.asarray(clf.predict_proba(x), dtype=np.float64)
    # Ensure columns align to classes 0,1,2
    classes = list(getattr(clf, "classes_", [0, 1, 2]))
    out: list[tuple[float, float, float]] = []
    for row in proba:
        mapped = {int(c): float(p) for c, p in zip(classes, row, strict=True)}
        out.append((mapped.get(0, 0.0), mapped.get(1, 0.0), mapped.get(2, 0.0)))
    return out


def evaluate_icat_frozen_lr(
    *,
    model_path: Path,
    splits_dir: Path,
    subset: str = "all",
    device: str = "cpu",
    batch_size: int = 64,
    concatenate_context: bool | None = None,
    report_bias_types: bool = True,
) -> IcatReport:
    bundle = joblib.load(model_path)
    clf = bundle["classifier"]
    encoder = FrozenEncoder(
        backbone=str(bundle["backbone"]),
        device=device,
        max_len=int(bundle.get("max_len", 512)),
        normalize=bool(bundle.get("normalize", True)),
    )
    sep = encoder.tokenizer.sep_token or "[SEP]"

    bundles: dict[str, IcatBundle] = {}
    bias_rows: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    bias_rows_by_kind: dict[str, dict[str, list[tuple[float, float, float]]]] = {
        "intersentence": defaultdict(list),
        "intrasentence": defaultdict(list),
    }
    n_total = 0

    for kind in ("intersentence", "intrasentence"):
        concat = True if kind == "intersentence" else False
        if concatenate_context is False:
            concat = False
        paths = resolve_split_paths(splits_dir, kind=kind, subset=subset)
        examples: list[FlatStereoExample] = []
        for path in paths:
            examples.extend(
                flatten_split_file(path, concatenate_context=concat, sep_token=sep)
            )
        if not examples:
            continue
        probs = _probs_for_examples(
            examples, encoder=encoder, clf=clf, batch_size=batch_size
        )
        bundles[kind] = bundle_from_probs(probs)
        n_total += len(examples)
        if report_bias_types:
            for ex, row in zip(examples, probs, strict=True):
                bias_rows[ex.bias_type].append(row)
                bias_rows_by_kind[kind][ex.bias_type].append(row)

    inter = bundles.get("intersentence")
    intra = bundles.get("intrasentence")
    avg = None
    if inter is not None and intra is not None:
        avg = (inter.icat + intra.icat) / 2.0
    elif inter is not None:
        avg = inter.icat
    elif intra is not None:
        avg = intra.icat

    by_bias = {
        bias: bundle_from_probs(rows).to_dict()
        for bias, rows in sorted(bias_rows.items())
    }
    by_bias_inter = {
        bias: bundle_from_probs(rows).to_dict()
        for bias, rows in sorted(bias_rows_by_kind["intersentence"].items())
    }
    by_bias_intra = {
        bias: bundle_from_probs(rows).to_dict()
        for bias, rows in sorted(bias_rows_by_kind["intrasentence"].items())
    }
    if not report_bias_types:
        by_bias, by_bias_inter, by_bias_intra = {}, {}, {}

    return IcatReport(
        intersentence=inter,
        intrasentence=intra,
        avg_icat=avg,
        by_bias_type=by_bias,
        by_bias_type_intersentence=by_bias_inter,
        by_bias_type_intrasentence=by_bias_intra,
        n_total=n_total,
        subset=subset,
        concatenate_context=False if concatenate_context is False else True,
    )
