"""Unit tests for Thesis analysis stats (no matplotlib required)."""

from __future__ import annotations

import json
from pathlib import Path

from fairai.application.analyze_gen_metrics import analyze_gen_metrics, load_metric_rows
from fairai.application.analyze_mc_predictions import analyze_mc_predictions
from fairai.application.analyze_rl_compare import analyze_rl_compare
from fairai.application.merge_mc_options import merge_stereoset_select_options
from fairai.domain.analysis_stats import (
    accuracy_by_group,
    compare_paired,
    composite_metric,
    compute_bleu,
    distinct_n,
    lexical_diversity,
    mean_std,
    summarize_metric_groups,
)


def test_mean_std_and_distinct() -> None:
    stats = mean_std([1.0, 2.0, 3.0])
    assert stats.n == 3
    assert abs(stats.mean - 2.0) < 1e-9
    assert distinct_n("a b a b c", 1) > 0
    assert lexical_diversity(["hello world", "hello there"]) > 0
    assert 0.0 <= compute_bleu("the cat sat", "the cat sat") <= 1.0


def test_compare_paired_without_scipy() -> None:
    result = compare_paired([0.20, 0.22, 0.21], [0.28, 0.30, 0.29], metric="F1")
    assert result.treatment.mean > result.baseline.mean
    assert result.metric == "F1"


def test_accuracy_by_group() -> None:
    acc = accuracy_by_group([0, 0, 1, 1], [0, 1, 1, 1], ["race", "race", "gender", "gender"])
    assert acc["race"] == 0.5
    assert acc["gender"] == 1.0


def test_summarize_gen_metrics(tmp_path: Path) -> None:
    rows = [
        {
            "model": "Baseline",
            "perplexity": 10.0,
            "distinct_1": 0.4,
            "distinct_2": 0.7,
            "bleu": 0.1,
        },
        {
            "model": "Baseline",
            "perplexity": 12.0,
            "distinct_1": 0.5,
            "distinct_2": 0.8,
            "bleu": 0.2,
        },
        {"model": "RL", "perplexity": 8.0, "distinct_1": 0.6, "distinct_2": 0.9, "bleu": 0.3},
    ]
    summary = summarize_metric_groups(rows)
    assert len(summary) == 2
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    assert len(load_metric_rows(path)) == 3
    report = analyze_gen_metrics(path, out_dir=tmp_path / "out", write_plots=False)
    assert report.summary
    assert (tmp_path / "out" / "gen_metrics_report.json").is_file()


def test_composite_and_merge(tmp_path: Path) -> None:
    assert abs(composite_metric([0, 1, 2], [0, -1, 2], null_value=-1) - (1.0 * (2 / 3))) < 1e-9
    src = tmp_path / "pred.json"
    src.write_text(
        json.dumps(
            {
                "train": [{"id": 1}],
                "validation": [{"id": 2}],
                "test": [{"id": 3}],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "merged.json"
    report = merge_stereoset_select_options(src, out)
    assert report.n_prompts == 3
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["prompts"]) == 3


def test_dataset_dist(tmp_path: Path) -> None:
    from fairai.application.analyze_dataset_dist import analyze_dataset_dist

    path = tmp_path / "data.jsonl"
    rows = [
        {"sentence": "short one", "label": 0, "bias_type": "race"},
        {"sentence": "another longer sentence here", "label": 1, "bias_type": "race"},
        {"sentence": "neutral text", "label": 2, "bias_type": "gender"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    report = analyze_dataset_dist(path, out_dir=tmp_path / "dist", write_plots=False)
    assert report.n_samples == 3
    assert "stereotype" in report.label_counts


def test_rl_compare_and_mc(tmp_path: Path) -> None:
    metrics = {
        "metrics": [
            {"name": "Precision", "baseline": [0.2, 0.3], "rl": [0.4, 0.5]},
        ]
    }
    report = analyze_rl_compare(metrics, out_dir=tmp_path / "rl", write_plots=False)
    assert report.results[0].treatment.mean == 0.45

    mc = {
        "prompts": [
            {
                "item_bias": "race",
                "options": [
                    {"sentence": "a", "label": 0, "label_predicted": 0},
                    {"sentence": "b", "label": 1, "label_predicted": 0},
                    {"sentence": "c", "label": 2, "label_predicted": None},
                ],
            }
        ]
    }
    mc_path = tmp_path / "mc.json"
    mc_path.write_text(json.dumps(mc), encoding="utf-8")
    mc_report = analyze_mc_predictions(mc_path, out_dir=tmp_path / "mc", write_plots=False)
    assert mc_report.n_total == 3
    assert mc_report.n_predicted == 2
    assert mc_report.n_missing_pred == 1
    assert abs(mc_report.accuracy_global - 0.5) < 1e-9
