"""CLI commands for ICAT evaluation and parity."""

from __future__ import annotations

import json
from pathlib import Path

import typer

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
) -> None:
    """Run ICAT on StereoSet splits (default command)."""
    if ctx.invoked_subcommand is not None:
        return
    from fairai.application.evaluate_icat import evaluate_icat
    from fairai.application.parity import check_parity, parity_ok

    if subset not in {"train", "test", "all"}:
        typer.secho("subset must be train|test|all", fg=typer.colors.RED)
        raise typer.Exit(1)

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
    typer.echo(json.dumps(payload, indent=2))

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}")

    baseline_path = baseline
    if write_baseline:
        target = baseline_path or Path("artifacts/results/metrics/baseline_van_v2_icat.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
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
