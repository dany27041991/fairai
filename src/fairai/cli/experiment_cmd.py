"""CLI: baselines, ablations, unified F1+iCAT reports."""

from __future__ import annotations

from pathlib import Path

import typer

from fairai.cli.json_io import echo_json

experiment_app = typer.Typer(help="Baselines / ablations + paper metrics (F1, iCAT).")


@experiment_app.command("train-lr")
def train_lr_cmd(
    config: Path = typer.Option(
        Path("configs/experiments/baseline_frozen_lr.yaml"),
        "--config",
        "-c",
    ),
    device: str | None = typer.Option(None, "--device"),
    max_samples: int | None = typer.Option(None, "--max-samples"),
    run_id: str | None = typer.Option(None, "--run-id"),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts", "-a"),
) -> None:
    """Train frozen MiniLM + LogisticRegression baseline."""
    from fairai.application.train_frozen_lr import train_frozen_lr
    from fairai.infrastructure.artifact_paths import ArtifactPaths

    result = train_frozen_lr(
        config_path=config,
        artifacts=ArtifactPaths.from_env(root=artifacts),
        device=device,
        max_samples=max_samples,
        run_id=run_id,
    )
    echo_json(result.to_dict())


@experiment_app.command("eval-run")
def eval_run_cmd(
    run_dir: Path = typer.Option(..., "--run-dir", "-r", help="artifacts/models/runs/<id>"),
    eval_dataset: Path | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Labeled JSONL for F1/accuracy (optional).",
    ),
    splits_dir: Path | None = typer.Option(
        None,
        "--splits-dir",
        help="StereoSet splits dir for iCAT (optional).",
    ),
    device: str = typer.Option("cpu", "--device"),
    batch_size: int = typer.Option(32, "--batch-size"),
    max_samples: int | None = typer.Option(None, "--max-samples"),
    subset: str = typer.Option("all", "--subset"),
) -> None:
    """Write F1 and/or iCAT into ``<run_dir>/eval_report.json``."""
    from fairai.application.evaluate_experiment import evaluate_experiment_run

    if eval_dataset is None and splits_dir is None:
        typer.secho("Provide --dataset and/or --splits-dir", fg=typer.colors.RED)
        raise typer.Exit(1)
    result = evaluate_experiment_run(
        run_dir=run_dir,
        eval_dataset=eval_dataset,
        splits_dir=splits_dir,
        device=device,
        batch_size=batch_size,
        max_samples=max_samples,
        subset=subset,
    )
    echo_json(result.to_dict())
    typer.echo(f"wrote {result.out_path}")


@experiment_app.command("compare")
def compare_cmd(
    run_dir: list[Path] = typer.Option(
        ...,
        "--run-dir",
        "-r",
        help="Repeat for each run to include in the table.",
    ),
    out: Path = typer.Option(
        Path("artifacts/results/metrics/baselines_compare.json"),
        "--out",
        "-o",
    ),
    eval_dataset: Path | None = typer.Option(None, "--dataset", "-d"),
    splits_dir: Path | None = typer.Option(None, "--splits-dir"),
    device: str = typer.Option("cpu", "--device"),
    batch_size: int = typer.Option(32, "--batch-size"),
    max_samples: int | None = typer.Option(None, "--max-samples"),
    subset: str = typer.Option("all", "--subset"),
) -> None:
    """Evaluate many runs → one comparison JSON (paper table)."""
    from fairai.application.evaluate_experiment import compare_experiment_runs

    table = compare_experiment_runs(
        run_dirs=list(run_dir),
        eval_dataset=eval_dataset,
        splits_dir=splits_dir,
        device=device,
        batch_size=batch_size,
        max_samples=max_samples,
        subset=subset,
        out_path=out,
    )
    echo_json(table)
    typer.echo(f"wrote {out}")


@experiment_app.command("suite")
def suite_cmd(
    config: Path = typer.Option(
        Path("configs/experiments/suite_smoke.yaml"),
        "--config",
        "-c",
        help="Suite YAML listing experiment configs + eval paths.",
    ),
    device: str | None = typer.Option(None, "--device"),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts", "-a"),
    skip_train: bool = typer.Option(
        False, "--skip-train", help="Only eval existing run_dirs from suite YAML."
    ),
) -> None:
    """Train listed experiments (or skip) then compare F1/iCAT."""
    from fairai.application.run_experiment_suite import run_experiment_suite
    from fairai.infrastructure.artifact_paths import ArtifactPaths

    result = run_experiment_suite(
        suite_path=config,
        artifacts=ArtifactPaths.from_env(root=artifacts),
        device=device,
        skip_train=skip_train,
    )
    echo_json(result)


@experiment_app.command("optuna")
def optuna_cmd(
    family: str = typer.Option(
        ...,
        "--family",
        "-f",
        help="Search family: van | ce | no_bpl",
    ),
    config: Path = typer.Option(
        Path("configs/experiments/optuna_fair_base.yaml"),
        "--config",
        "-c",
        help="Base YAML (WHM train-iCAT matched split).",
    ),
    n_trials: int = typer.Option(100, "--n-trials", "-n"),
    study_name: str | None = typer.Option(None, "--study-name"),
    device: str = typer.Option("cuda", "--device"),
    max_samples: int | None = typer.Option(
        None, "--max-samples", help="Cap train size (smoke)."
    ),
    trial_epochs: int | None = typer.Option(
        20, "--trial-epochs", help="Max epochs per HPO trial (None=YAML)."
    ),
    trial_patience: int | None = typer.Option(3, "--trial-patience"),
    seed: int = typer.Option(42, "--seed"),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts", "-a"),
) -> None:
    """Equal-budget Optuna TPE search (WHM on StereoSet-train iCAT)."""
    from fairai.application.run_optuna_hpo import run_optuna_hpo
    from fairai.domain.optuna_space import FAMILIES
    from fairai.infrastructure.artifact_paths import ArtifactPaths

    fam = family.strip().lower()
    if fam not in FAMILIES:
        typer.secho(f"family must be one of {FAMILIES}", fg=typer.colors.RED)
        raise typer.Exit(1)
    summary = run_optuna_hpo(
        family=fam,  # type: ignore[arg-type]
        base_config=config,
        n_trials=n_trials,
        study_name=study_name,
        artifacts=ArtifactPaths.from_env(root=artifacts),
        device=device,
        max_samples=max_samples,
        trial_epochs=trial_epochs,
        trial_patience=trial_patience,
        seed=seed,
    )
    echo_json(summary)
    typer.echo(f"best_config: {summary.get('best_config')}")


@experiment_app.command("optuna-finalize")
def optuna_finalize_cmd(
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts", "-a"),
    out: Path = typer.Option(
        Path("configs/experiments/suite_optuna_fair.yaml"),
        "--out",
        "-o",
    ),
) -> None:
    """Write suite_optuna_fair.yaml from each family's best_config.yaml."""
    from fairai.application.run_optuna_hpo import write_finalize_suite
    from fairai.infrastructure.artifact_paths import ArtifactPaths

    path = write_finalize_suite(
        artifacts=ArtifactPaths.from_env(root=artifacts),
        out_suite=out,
    )
    typer.echo(f"wrote {path}")
