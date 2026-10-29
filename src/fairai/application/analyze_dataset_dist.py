"""Thesis ANALISYS cell 48 — label/bias distributions and sentence lengths."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fairai.domain.analysis_stats import LABEL_NAMES, avg_length_by, count_by, label_counts_per_bias
from fairai.domain.example import Example
from fairai.infrastructure.data.jsonl_repo import load_examples
from fairai.infrastructure.torch_van.checkpointing import write_json


@dataclass(slots=True)
class DatasetDistReport:
    n_samples: int
    bias_counts: dict[str, int]
    label_counts: dict[str, int]
    label_per_bias: dict[str, dict[str, int]]
    avg_len_by_bias: dict[str, float]
    avg_len_by_label: dict[str, float]
    avg_len_by_bias_label: dict[str, dict[str, float]]
    plot_paths: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "bias_counts": self.bias_counts,
            "label_counts": self.label_counts,
            "label_per_bias": self.label_per_bias,
            "avg_len_by_bias": self.avg_len_by_bias,
            "avg_len_by_label": self.avg_len_by_label,
            "avg_len_by_bias_label": self.avg_len_by_bias_label,
            "plot_paths": self.plot_paths,
        }


def _avg_len_bias_label(examples: list[Example]) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for ex in examples:
        bias = ex.bias_type if ex.label != 2 else None  # Thesis: neutral → no bias
        key = str(bias if bias is not None else "No bias")
        buckets[key][int(ex.label)].append(len(ex.sentence.split()))
    return {
        bias: {
            LABEL_NAMES.get(lab, str(lab)): (sum(vals) / len(vals) if vals else 0.0)
            for lab, vals in labs.items()
        }
        for bias, labs in buckets.items()
    }


def analyze_dataset_dist(
    dataset_path: Path,
    *,
    out_dir: Path | None = None,
    max_samples: int | None = None,
    write_plots: bool = True,
) -> DatasetDistReport:
    examples = load_examples(dataset_path, max_samples=max_samples)
    labels = [int(ex.label) for ex in examples]
    # Match Thesis: neutral rows contribute bias=None
    biases = [ex.bias_type if int(ex.label) != 2 else None for ex in examples]
    sentences = [ex.sentence for ex in examples]

    bias_counts = count_by(biases)
    label_counts = {LABEL_NAMES.get(k, str(k)): v for k, v in Counter(labels).items()}
    label_per = label_counts_per_bias(labels, biases)
    avg_bias = avg_length_by(sentences, biases)
    avg_label = {
        LABEL_NAMES.get(int(k), str(k)): v
        for k, v in avg_length_by(sentences, labels).items()
    }
    avg_bias_label = _avg_len_bias_label(examples)

    plot_paths: list[str] = []
    if write_plots and out_dir is not None and examples:
        plot_paths.extend(
            _write_dist_plots(
                out_dir,
                bias_counts,
                label_counts,
                label_per,
                avg_bias,
                avg_label,
                avg_bias_label,
            )
        )

    report = DatasetDistReport(
        n_samples=len(examples),
        bias_counts=bias_counts,
        label_counts=label_counts,
        label_per_bias=label_per,
        avg_len_by_bias=avg_bias,
        avg_len_by_label=avg_label,
        avg_len_by_bias_label=avg_bias_label,
        plot_paths=plot_paths,
    )
    if out_dir is not None:
        write_json(out_dir / "dataset_dist_report.json", report.to_dict())
    return report


def _write_dist_plots(
    out_dir: Path,
    bias_counts: dict[str, int],
    label_counts: dict[str, int],
    label_per: dict[str, dict[str, int]],
    avg_bias: dict[str, float],
    avg_label: dict[str, float],
    avg_bias_label: dict[str, dict[str, float]],
) -> list[str]:
    from fairai.infrastructure.viz.bars import plot_horizontal_counts
    from fairai.infrastructure.viz.savefig import require_matplotlib, save_figure

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    paths.append(
        str(
            plot_horizontal_counts(
                list(bias_counts.keys()),
                list(bias_counts.values()),
                xlabel="Count",
                ylabel="Bias Type",
                title="Distribution of Bias Types",
                out_path=out_dir / "dist_bias_types.png",
            )
        )
    )
    paths.append(
        str(
            plot_horizontal_counts(
                list(label_counts.keys()),
                list(label_counts.values()),
                xlabel="Count",
                ylabel="Label",
                title="Overall Label Distribution",
                out_path=out_dir / "dist_labels.png",
            )
        )
    )
    paths.append(
        str(
            plot_horizontal_counts(
                list(avg_bias.keys()),
                list(avg_bias.values()),
                xlabel="Average Length (words)",
                ylabel="Bias Type",
                title="Average Sentence Length per Bias Type",
                out_path=out_dir / "dist_avg_len_bias.png",
            )
        )
    )
    paths.append(
        str(
            plot_horizontal_counts(
                list(avg_label.keys()),
                list(avg_label.values()),
                xlabel="Average Length (words)",
                ylabel="Label",
                title="Average Sentence Length per Label",
                out_path=out_dir / "dist_avg_len_label.png",
            )
        )
    )

    # Stacked label-per-bias (Thesis cell 48)
    plt = require_matplotlib()
    bias_keys = list(label_per.keys())
    label_order = ["stereotype", "anti-stereotype", "neutral"]
    fig, ax = plt.subplots(figsize=(12, 7))
    left = [0.0] * len(bias_keys)
    for lab in label_order:
        widths = [float(label_per[b].get(lab, 0)) for b in bias_keys]
        ax.barh(bias_keys, widths, left=left, label=lab)
        left = [left[i] + widths[i] for i in range(len(bias_keys))]
    ax.set_title("Label Distribution per Bias Type")
    ax.set_xlabel("Count")
    ax.set_ylabel("Bias Type")
    ax.legend(title="Labels")
    paths.append(str(save_figure(fig, out_dir / "dist_label_per_bias.png")))

    # Grouped avg length by bias × label (Thesis cell 48 plot #6)
    if avg_bias_label:
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        bias_list = list(avg_bias_label.keys())
        width = 0.25
        y = list(range(len(bias_list)))
        for i, lab in enumerate(label_order):
            vals = [float(avg_bias_label[b].get(lab, 0.0)) for b in bias_list]
            ax2.barh([yi + i * width for yi in y], vals, height=width, label=lab)
        ax2.set_yticks([yi + width for yi in y])
        ax2.set_yticklabels(bias_list)
        ax2.set_xlabel("Average Length (words)")
        ax2.set_ylabel("Bias Type")
        ax2.set_title("Average Sentence Length per Bias Type and Label")
        ax2.legend(title="Labels")
        paths.append(str(save_figure(fig2, out_dir / "dist_avg_len_bias_label.png")))
    return paths
