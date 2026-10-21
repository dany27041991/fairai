"""FAIRAI CLI entrypoint (Typer). Keep thin — logic lives in application/."""

from __future__ import annotations

from pathlib import Path

import typer

from fairai import __version__
from fairai.cli.icat_cmd import icat_app

app = typer.Typer(
    name="fairai",
    help="FAIRAI / VAN research CLI — reproducible bias classification.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(icat_app, name="icat")


def _print_version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _main(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_print_version,
        is_eager=True,
    ),
) -> None:
    """FAIRAI CLI."""


@app.command("import-asis")
def import_asis(
    source: Path = typer.Option(
        ...,
        "--source",
        "-s",
        help="Path to TESI MAGISTRALE (AS-IS root).",
        exists=True,
        file_okay=False,
    ),
    dest: Path = typer.Option(
        Path("artifacts"),
        "--dest",
        "-d",
        help="Destination artifacts root.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan only, no copy."),
) -> None:
    """Copy canonical AS-IS datasets/checkpoints into artifacts/."""
    from fairai.application.import_artifacts import import_asis_artifacts

    result = import_asis_artifacts(source, dest, dry_run=dry_run)
    typer.echo(f"copied:  {len(result.copied)}")
    typer.echo(f"skipped: {len(result.skipped)}")
    if result.missing_optional:
        typer.echo("missing optional:")
        for item in result.missing_optional:
            typer.echo(f"  - {item}")
    typer.echo(f"manifest: {result.manifest_path}")


@app.command("manifest")
def manifest_cmd(
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts", "-a"),
    verify: bool = typer.Option(False, "--verify", help="Verify hashes on disk."),
) -> None:
    """Show or verify artifacts/manifest.sha256.json."""
    import json

    from fairai.infrastructure.hashing import sha256_file

    path = artifacts / "manifest.sha256.json"
    if not path.is_file():
        typer.secho(f"Manifest not found: {path}", fg=typer.colors.RED)
        raise typer.Exit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not verify:
        typer.echo(json.dumps(data, indent=2))
        return
    ok = True
    for entry in data.get("files", []):
        target = artifacts / entry["relpath"]
        if not target.is_file():
            typer.secho(f"MISSING {entry['relpath']}", fg=typer.colors.RED)
            ok = False
            continue
        actual = sha256_file(target)
        if actual != entry["sha256"]:
            typer.secho(f"HASH MISMATCH {entry['relpath']}", fg=typer.colors.RED)
            ok = False
        else:
            typer.echo(f"ok {entry['relpath']}")
    raise typer.Exit(0 if ok else 1)


@app.command("check-file-length")
def check_file_length(
    root: Path = typer.Option(Path("src"), "--root"),
    limit: int = typer.Option(500, "--limit", help="Max lines per .py file."),
) -> None:
    """Fail if any source file exceeds the line budget (SOLID / maintainability)."""
    from fairai.infrastructure.file_budget import find_oversized_python_files

    offenders = find_oversized_python_files(root, limit=limit)
    if not offenders:
        typer.echo(f"OK: no files over {limit} lines under {root}")
        return
    for path, n in offenders:
        typer.secho(f"{n:4d}  {path}", fg=typer.colors.RED)
    raise typer.Exit(1)


@app.command("eval")
def eval_cmd(
    checkpoint: Path = typer.Option(
        Path("artifacts/models/van/v2/final_model.pt"),
        "--checkpoint",
        "-k",
    ),
    config: Path = typer.Option(Path("configs/models/van_v2.yaml"), "--config", "-c"),
    dataset: Path = typer.Option(
        Path("artifacts/datasets/processed/van_minimal_24k.jsonl"),
        "--dataset",
        "-d",
    ),
    max_samples: int | None = typer.Option(512, "--max-samples"),
    batch_size: int = typer.Option(32, "--batch-size"),
    device: str = typer.Option("cpu", "--device"),
    gates: bool = typer.Option(False, "--gates", help="Apply threshold gates."),
    out: Path | None = typer.Option(None, "--out", help="Write JSON report."),
) -> None:
    """Classification eval on a labeled JSONL with VAN checkpoint."""
    import json

    from fairai.application.evaluate_classification import evaluate_classification

    report = evaluate_classification(
        checkpoint_path=checkpoint,
        config_path=config,
        dataset_path=dataset,
        device=device,
        batch_size=batch_size,
        max_samples=max_samples,
        apply_threshold_gates=gates,
    )
    payload = report.to_dict()
    typer.echo(json.dumps(payload, indent=2))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"wrote {out}")


@app.command("generate")
def generate_cmd(
    config: Path = typer.Option(Path("configs/data/generate.yaml"), "--config", "-c"),
    sources: str | None = typer.Option(
        None, "--sources", help="Comma list: crows,jsonl (default from YAML)."
    ),
    max_samples: int | None = typer.Option(None, "--max-samples"),
    synonyms: bool = typer.Option(False, "--synonyms", help="Enable WordNet expansion."),
    out: str | None = typer.Option(None, "--out", help="Relative path under datasets/."),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts", "-a"),
) -> None:
    """Regenerate a schema-valid JSONL subset (does not overwrite canonical processed/)."""
    import json

    from fairai.application.generate_dataset import generate_dataset
    from fairai.infrastructure.artifact_paths import ArtifactPaths

    source_list = [s.strip() for s in sources.split(",") if s.strip()] if sources else None
    result = generate_dataset(
        config_path=config,
        artifacts=ArtifactPaths.from_env(root=artifacts),
        sources=source_list,
        max_samples=max_samples,
        expand_synonyms=synonyms or None,
        out_name=out,
    )
    typer.echo(json.dumps(result.to_dict(), indent=2))
    if result.n_schema_issues:
        typer.secho(f"schema issues dropped: {result.n_schema_issues}", fg=typer.colors.YELLOW)


@app.command("train")
def train_cmd(
    config: Path = typer.Option(
        Path("configs/experiments/smoke_cpu.yaml"),
        "--config",
        "-c",
        help="Experiment YAML (may inherit models/van_v2.yaml).",
    ),
    device: str | None = typer.Option(None, "--device", help="cpu|cuda (overrides YAML)."),
    max_samples: int | None = typer.Option(
        None, "--max-samples", help="Cap dataset size (smoke)."
    ),
    run_id: str | None = typer.Option(None, "--run-id"),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts", "-a"),
) -> None:
    """Train VAN; writes checkpoint + metrics under artifacts/models/runs/."""
    import json

    from fairai.application.train_van import train_van
    from fairai.infrastructure.artifact_paths import ArtifactPaths

    result = train_van(
        config_path=config,
        artifacts=ArtifactPaths.from_env(root=artifacts),
        device=device,
        max_samples=max_samples,
        run_id=run_id,
    )
    typer.echo(json.dumps(result.to_dict(), indent=2))


@app.command("smoke")
def smoke_cmd(
    device: str = typer.Option("cpu", "--device"),
    checkpoint: Path = typer.Option(
        Path("artifacts/models/van/v2/final_model.pt"),
        "--checkpoint",
        "-k",
    ),
    config: Path = typer.Option(Path("configs/models/van_v2.yaml"), "--config", "-c"),
    skip_forward: bool = typer.Option(
        False, "--skip-forward", help="Skip VAN load/forward (CI without artifacts)."
    ),
) -> None:
    """CI smoke: pytest + file budget + optional VAN forward."""
    from fairai.application.smoke import run_smoke, smoke_passed

    repo_root = Path(__file__).resolve().parents[3]
    # Installed editable: src/fairai/cli -> parents[3] = repo root when running from src layout
    if not (repo_root / "pyproject.toml").is_file():
        repo_root = Path.cwd()

    result = run_smoke(
        repo_root=repo_root,
        checkpoint=checkpoint,
        config=config,
        device=device,
        skip_forward=skip_forward,
    )
    typer.echo(result.detail)
    if smoke_passed(result):
        typer.secho("smoke OK", fg=typer.colors.GREEN)
        raise typer.Exit(0)
    typer.secho("smoke FAILED", fg=typer.colors.RED)
    raise typer.Exit(1)


@app.command("infer")
def infer_cmd(
    text: list[str] | None = typer.Option(
        None, "--text", "-t", help="Text to classify (repeatable)."
    ),
    file: Path | None = typer.Option(None, "--file", "-f", help="One text per line."),
    checkpoint: Path = typer.Option(
        Path("artifacts/models/van/v2/final_model.pt"),
        "--checkpoint",
        "-k",
    ),
    config: Path = typer.Option(Path("configs/models/van_v2.yaml"), "--config", "-c"),
    device: str = typer.Option("cpu", "--device"),
    gates: bool = typer.Option(False, "--gates"),
) -> None:
    """Infer stereotype labels for texts."""
    import json

    from fairai.application.infer_texts import infer_texts

    texts: list[str] = list(text or [])
    if file is not None:
        texts.extend(
            line.strip()
            for line in file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not texts:
        typer.secho("Provide --text and/or --file", fg=typer.colors.RED)
        raise typer.Exit(1)
    from dataclasses import asdict

    items = infer_texts(
        texts,
        checkpoint_path=checkpoint,
        config_path=config,
        device=device,
        apply_threshold_gates=gates,
    )
    typer.echo(json.dumps([asdict(item) for item in items], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
