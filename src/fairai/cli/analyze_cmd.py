"""CLI: fairai analyze — Thesis ANALISYS / plots (cells 46–49 + RL/few-shot charts)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from fairai.cli.json_io import echo_json

analyze_app = typer.Typer(help="Thesis analysis & plots (distributions, CM, text, RL).")


@analyze_app.command("neutral-audit")
def neutral_audit_cmd(
    f1_jsonl: Path = typer.Option(
        Path("artifacts/datasets/processed/stereoset/f1_test_leakfree.jsonl"),
        "--f1-jsonl",
        exists=True,
        dir_okay=False,
    ),
    out_sample: Path = typer.Option(
        Path("artifacts/analysis/neutral_audit_sample.json"),
        "--out-sample",
    ),
    out_report: Path = typer.Option(
        Path("artifacts/analysis/neutral_audit.json"),
        "--out-report",
        "-o",
    ),
    apply_heuristic_filter: bool = typer.Option(
        False,
        "--apply-heuristic-filter",
        help="Write filtered JSONL dropping heuristic unrelated_junk neutrals.",
    ),
    out_filtered: Path = typer.Option(
        Path("artifacts/datasets/processed/stereoset/f1_test_leakfree_filtered.jsonl"),
        "--out-filtered",
    ),
) -> None:
    """Sample neutrals for human audit + heuristic junk flags."""
    from fairai.application.audit_neutrals import (
        audit_neutrals,
        filter_f1_jsonl_by_heuristic,
    )

    report = audit_neutrals(
        f1_jsonl=f1_jsonl,
        select_options_path=Path(
            "artifacts/datasets/processed/stereoset/stereoset_select_options_pred.json"
        ),
        out_sample=out_sample,
        out_report=out_report,
    )
    echo_json(report.to_dict())
    if apply_heuristic_filter:
        stats = filter_f1_jsonl_by_heuristic(in_jsonl=f1_jsonl, out_jsonl=out_filtered)
        echo_json(stats)
        typer.echo(f"wrote {out_filtered}")
    typer.echo(f"wrote {out_sample} and {out_report}")


@analyze_app.command("f1-split")
def f1_split_cmd(
    select_options: Path = typer.Option(
        Path("artifacts/datasets/processed/stereoset/stereoset_select_options_pred.json"),
        "--select-options",
        exists=True,
        dir_okay=False,
        help="StereoSet select-options JSON with train/validation/test.",
    ),
    corpus: list[Path] = typer.Option(
        [
            Path("artifacts/datasets/processed/van_split_seed42_train.jsonl"),
            Path("artifacts/datasets/processed/van_split_seed42_val.jsonl"),
        ],
        "--corpus",
        help="Corpus JSONL files whose sentences are forbidden (repeatable).",
    ),
    out_jsonl: Path = typer.Option(
        Path("artifacts/datasets/processed/stereoset/f1_test_leakfree.jsonl"),
        "--out-jsonl",
    ),
    out_report: Path = typer.Option(
        Path("artifacts/analysis/f1_test_leakfree_report.json"),
        "--out-report",
        "-o",
    ),
    min_prompts: int = typer.Option(150, "--min-prompts"),
) -> None:
    """Build leakage-free F1 test JSONL (unrelated→neutral select-options)."""
    from fairai.application.build_select_options_f1_split import (
        build_select_options_f1_split,
    )

    report = build_select_options_f1_split(
        select_options_path=select_options,
        corpus_jsonl_paths=list(corpus),
        out_jsonl=out_jsonl,
        out_report=out_report,
        min_prompts=min_prompts,
    )
    echo_json(report.to_dict())
    typer.echo(f"wrote {out_jsonl} and {out_report}")


@analyze_app.command("leakage")
def leakage_cmd(
    dataset: Path = typer.Option(
        Path("artifacts/datasets/processed/van_train_leakage_safe.jsonl"),
        "--dataset",
        "-d",
        exists=True,
        dir_okay=False,
        help="Train JSONL to audit (GAP-D4).",
    ),
    splits_dir: Path = typer.Option(
        Path("artifacts/datasets/splits/stereoset"),
        "--splits-dir",
        exists=True,
        file_okay=False,
    ),
    out_report: Path = typer.Option(
        Path("artifacts/analysis/leakage_audit.json"),
        "--out-report",
        "-o",
    ),
    out_strict: Path | None = typer.Option(
        None,
        "--out-strict",
        help="Optional stricter JSONL (must differ from --dataset).",
    ),
    max_samples: int | None = typer.Option(None, "--max-samples"),
) -> None:
    """Audit residual StereoSet-test overlaps in a train JSONL (GAP-D4)."""
    from fairai.application.audit_leakage import audit_train_leakage

    report = audit_train_leakage(
        dataset,
        splits_dir=splits_dir,
        out_strict=out_strict,
        out_report=out_report,
        max_samples=max_samples,
    )
    echo_json(report.to_dict())


@analyze_app.command("label-mc")
def label_mc_cmd(
    input_path: Path = typer.Option(
        Path("artifacts/datasets/processed/stereoset/stereoset_select_options.json"),
        "--input",
        "-i",
        exists=True,
        dir_okay=False,
        help="MC JSON with train/validation/test (Thesis cell 5 input).",
    ),
    output_path: Path = typer.Option(
        Path("artifacts/datasets/processed/stereoset/stereoset_select_options_pred.json"),
        "--output",
        "-o",
        help="Write label_predicted per option.",
    ),
    checkpoint: Path = typer.Option(
        Path("artifacts/models/van/v2/final_model.pt"), "--checkpoint", "-k"
    ),
    config: Path = typer.Option(Path("configs/models/van_v2.yaml"), "--config", "-c"),
    device: str = typer.Option("cpu", "--device"),
    mc_samples: int = typer.Option(10, "--mc-samples", help="Thesis MC dropout samples."),
) -> None:
    """Label MC options with VAN + MC gates (Thesis update_dataset_with_predictions)."""
    from fairai.application.label_mc_options import update_dataset_with_predictions

    report = update_dataset_with_predictions(
        input_path,
        output_path,
        checkpoint_path=checkpoint,
        config_path=config,
        device=device,
        mc_samples=mc_samples,
    )
    echo_json(report.to_dict())


@analyze_app.command("merge-mc")
def merge_mc_cmd(
    input_path: Path = typer.Option(
        Path("artifacts/datasets/processed/stereoset/stereoset_select_options_pred.json"),
        "--input",
        "-i",
        exists=True,
        dir_okay=False,
    ),
    output_path: Path = typer.Option(
        Path("artifacts/datasets/processed/stereoset/stereoset_select_options.json"),
        "--output",
        "-o",
    ),
) -> None:
    """Merge train/validation/test MC preds → flat prompts (Thesis cell 10)."""
    from fairai.application.merge_mc_options import merge_stereoset_select_options

    report = merge_stereoset_select_options(input_path, output_path)
    echo_json(report.to_dict())


@analyze_app.command("rl-compare")
def rl_compare_cmd(
    metrics: Path = typer.Option(
        ...,
        "--metrics",
        "-m",
        help="JSON with paired Baseline/RL arrays per metric.",
        exists=True,
        dir_okay=False,
    ),
    out_dir: Path = typer.Option(Path("artifacts/analysis/rl_compare"), "--out-dir", "-o"),
    alpha: float = typer.Option(0.05, "--alpha"),
    no_plots: bool = typer.Option(False, "--no-plots"),
) -> None:
    """Paired t-test + bars (Thesis ANALISYS cell 46)."""
    from fairai.application.analyze_rl_compare import analyze_rl_compare

    report = analyze_rl_compare(
        metrics, out_dir=out_dir, alpha=alpha, write_plots=not no_plots
    )
    echo_json(report.to_dict())


@analyze_app.command("mc")
def mc_cmd(
    dataset: Path = typer.Option(
        Path("artifacts/datasets/processed/stereoset/stereoset_select_options.json"),
        "--dataset",
        "-d",
        exists=True,
        dir_okay=False,
    ),
    out_dir: Path = typer.Option(Path("artifacts/analysis/mc"), "--out-dir", "-o"),
    no_plots: bool = typer.Option(False, "--no-plots"),
) -> None:
    """MC prediction accuracy / CM / error by bias (Thesis cell 47)."""
    from fairai.application.analyze_mc_predictions import analyze_mc_predictions

    report = analyze_mc_predictions(dataset, out_dir=out_dir, write_plots=not no_plots)
    echo_json(report.to_dict())


@analyze_app.command("dataset")
def dataset_cmd(
    dataset: Path = typer.Option(
        Path("artifacts/datasets/processed/van_minimal_24k.jsonl"),
        "--dataset",
        "-d",
        exists=True,
        dir_okay=False,
    ),
    out_dir: Path = typer.Option(Path("artifacts/analysis/dataset"), "--out-dir", "-o"),
    max_samples: int | None = typer.Option(None, "--max-samples"),
    no_plots: bool = typer.Option(False, "--no-plots"),
) -> None:
    """Bias/label distributions + length plots (Thesis cell 48)."""
    from fairai.application.analyze_dataset_dist import analyze_dataset_dist

    report = analyze_dataset_dist(
        dataset,
        out_dir=out_dir,
        max_samples=max_samples,
        write_plots=not no_plots,
    )
    echo_json(report.to_dict())


@analyze_app.command("text")
def text_cmd(
    dataset: Path = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="JSONL with sentence/label/bias_type (Thesis cell 49).",
        exists=True,
        dir_okay=False,
    ),
    out_dir: Path = typer.Option(Path("artifacts/analysis/text"), "--out-dir", "-o"),
    max_samples: int | None = typer.Option(None, "--max-samples"),
    device: str = typer.Option("cpu", "--device"),
    skip_embeddings: bool = typer.Option(False, "--skip-embeddings"),
    skip_tsne: bool = typer.Option(False, "--skip-tsne"),
    no_plots: bool = typer.Option(False, "--no-plots"),
) -> None:
    """Lexical / sentiment / TF-IDF / optional t-SNE (Thesis cell 49)."""
    from fairai.application.analyze_text_semantics import analyze_text_semantics

    report = analyze_text_semantics(
        dataset,
        out_dir=out_dir,
        max_samples=max_samples,
        write_plots=not no_plots,
        skip_embeddings=skip_embeddings,
        skip_tsne=skip_tsne,
        device=device,
    )
    echo_json(report.to_dict())


@analyze_app.command("gen-metrics")
def gen_metrics_cmd(
    metrics: Path = typer.Option(
        ...,
        "--metrics",
        "-m",
        help="CSV/JSON rows with model,perplexity,distinct_*,bleu.",
        exists=True,
        dir_okay=False,
    ),
    out_dir: Path = typer.Option(Path("artifacts/analysis/gen_metrics"), "--out-dir", "-o"),
    no_plots: bool = typer.Option(False, "--no-plots"),
) -> None:
    """RL generation metric bars (Thesis EVALUATION RL MODEL)."""
    from fairai.application.analyze_gen_metrics import analyze_gen_metrics

    report = analyze_gen_metrics(metrics, out_dir=out_dir, write_plots=not no_plots)
    echo_json(report.to_dict())


@analyze_app.command("metric-bars")
def metric_bars_cmd(
    metrics: Path = typer.Option(
        ...,
        "--metrics",
        "-m",
        help='JSON: {categories, series, title?, ylabel?}',
        exists=True,
        dir_okay=False,
    ),
    out_dir: Path = typer.Option(Path("artifacts/analysis/metric_bars"), "--out-dir", "-o"),
    filename: str = typer.Option("metric_bars.png", "--filename"),
) -> None:
    """Grouped bars (Thesis few-shot Baseline vs Few-Shot charts)."""
    from fairai.application.analyze_gen_metrics import analyze_metric_bars

    path = analyze_metric_bars(metrics, out_dir=out_dir, filename=filename)
    echo_json({"plot_path": str(path)})


@analyze_app.command("confusion")
def confusion_cmd(
    matrix_json: Path = typer.Option(
        ...,
        "--matrix",
        "-m",
        help="JSON list-of-lists CM, or eval report with confusion_matrix key.",
        exists=True,
        dir_okay=False,
    ),
    out: Path = typer.Option(Path("artifacts/analysis/confusion_matrix.png"), "--out", "-o"),
    title: str = typer.Option("Confusion Matrix", "--title"),
    labels: str = typer.Option(
        "Stereotypical,Anti-stereotypical,Neutral",
        "--labels",
        help="Comma-separated class names.",
    ),
) -> None:
    """Plot a confusion matrix (Thesis VAN evaluate_model / few-shot)."""
    from fairai.infrastructure.viz.confusion import plot_confusion_matrix

    raw = json.loads(matrix_json.read_text(encoding="utf-8"))
    matrix = raw["confusion_matrix"] if isinstance(raw, dict) else raw
    names = [s.strip() for s in labels.split(",") if s.strip()]
    path = plot_confusion_matrix(matrix, class_names=names, title=title, out_path=out)
    echo_json({"plot_path": str(path)})
