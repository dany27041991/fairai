"""CLI commands for ICAT evaluation and parity."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from fairai.cli.json_io import echo_json, write_and_echo
from fairai.infrastructure.torch_van.checkpointing import write_json

icat_app = typer.Typer(help="StereoSet LM/SS/ICAT evaluation.")


@icat_app.callback(invoke_without_command=True)
def icat_default(
    ctx: typer.Context,
    checkpoint: Path = typer.Option(
        Path("artifacts/models/van/v2/final_model.pt"), "--checkpoint", "-k"
    ),
    config: Path = typer.Option(Path("configs/models/van_v2.yaml"), "--config", "-c"),
    splits_dir: Path = typer.Option(
        Path("artifacts/datasets/splits/stereoset"), "--splits-dir"
    ),
    subset: str = typer.Option("all", "--subset", help="train|test|all"),
    device: str = typer.Option("cpu", "--device"),
    batch_size: int = typer.Option(32, "--batch-size"),
    force_no_context: bool = typer.Option(
        False,
        "--force-no-context",
        help="Force no context concat for both splits (default: AS-IS inter/intra rules).",
    ),
    out: Path | None = typer.Option(None, "--out"),
    baseline: Path | None = typer.Option(
        None, "--baseline", help="JSON baseline for parity check"
    ),
    write_baseline: bool = typer.Option(
        False, "--write-baseline", help="Write this run as baseline JSON"
    ),
    write_gold: Path | None = typer.Option(
        None,
        "--write-gold",
        help="Thesis gold_scores.json path (P(gold_label) per sentence id).",
    ),
) -> None:
    """Run ICAT on StereoSet splits (default command)."""
    if ctx.invoked_subcommand is not None:
        return
    from fairai.application.evaluate_icat import evaluate_icat, write_gold_scores_json
    from fairai.application.parity import check_parity, parity_ok

    if subset not in {"train", "test", "all"}:
        typer.secho("subset must be train|test|all", fg=typer.colors.RED)
        raise typer.Exit(1)

    if write_gold is not None:
        path = write_gold_scores_json(
            checkpoint_path=checkpoint,
            config_path=config,
            splits_dir=splits_dir,
            out_path=write_gold,
            subset=subset,
            device=device,
            batch_size=batch_size,
        )
        typer.echo(f"wrote gold scores {path}")

    report = evaluate_icat(
        checkpoint_path=checkpoint,
        config_path=config,
        splits_dir=splits_dir,
        subset=subset,
        device=device,
        batch_size=batch_size,
        concatenate_context=False if force_no_context else None,
    )
    payload = report.to_dict()
    if out is not None:
        write_and_echo(payload, out=out)
        typer.echo(f"wrote {out}")
    else:
        echo_json(payload)

    baseline_path = baseline
    if write_baseline:
        target = baseline_path or Path("artifacts/results/metrics/baseline_van_v2_icat.json")
        write_json(target, payload)
        typer.echo(f"baseline written {target}")
        baseline_path = None

    if baseline_path is not None:
        base = json.loads(baseline_path.read_text(encoding="utf-8"))
        tols = {
            "avg_icat": 1.0,
            "intersentence.icat": 1.5,
            "intrasentence.icat": 1.5,
            "intersentence.lm": 2.0,
            "intrasentence.lm": 2.0,
        }
        checks = check_parity(payload, base, tols)
        for c in checks:
            flag = "OK" if c.ok else "FAIL"
            typer.echo(
                f"[{flag}] {c.key}: actual={c.actual:.4f} baseline={c.baseline:.4f} tol={c.tol}"
            )
        if not parity_ok(checks):
            raise typer.Exit(2)
