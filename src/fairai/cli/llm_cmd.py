"""CLI: fairai llm few-shot | ppo-smoke."""

from __future__ import annotations

from pathlib import Path

import typer

llm_app = typer.Typer(help="LLM alignment helpers (few-shot MC, PPO smoke).")


@llm_app.command("few-shot")
def few_shot_cmd(
    dataset: Path = typer.Option(
        Path("artifacts/datasets/processed/stereoset/stereoset_select_options_pred.json"),
        "--dataset",
        "-d",
    ),
    checkpoint: Path = typer.Option(
        Path("artifacts/models/van/v2/final_model.pt"), "--checkpoint", "-k"
    ),
    config: Path = typer.Option(Path("configs/models/van_v2.yaml"), "--config", "-c"),
    device: str = typer.Option("cpu", "--device"),
    split: str = typer.Option("test", "--split"),
    max_examples: int = typer.Option(64, "--max-examples"),
    n_shots: int = typer.Option(3, "--n-shots"),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts", "-a"),
) -> None:
    """StereoSet MC few-shot smoke using frozen VAN as option judge."""
    from fairai.application.few_shot_mc import run_few_shot_mc
    from fairai.cli.json_io import echo_json
    from fairai.infrastructure.artifact_paths import ArtifactPaths

    result = run_few_shot_mc(
        dataset_path=dataset,
        checkpoint=checkpoint,
        config=config,
        device=device,
        split=split,
        max_examples=max_examples,
        n_shots=n_shots,
        artifacts=ArtifactPaths.from_env(root=artifacts),
    )
    payload = result.to_dict()
    # keep chat readable — truncate prompt in stdout
    prompt = payload.pop("sample_prompt", "")
    echo_json(payload)
    typer.echo("--- sample prompt (trunc) ---")
    typer.echo(prompt[:800])


@llm_app.command("ppo-smoke")
def ppo_smoke_cmd(
    dataset: Path = typer.Option(
        Path("artifacts/datasets/processed/stereoset/stereoset_select_options_pred.json"),
        "--dataset",
        "-d",
    ),
    steps: int = typer.Option(64, "--steps"),
    seed: int = typer.Option(42, "--seed"),
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts", "-a"),
) -> None:
    """REINFORCE smoke on MC env (not full Qwen PPO)."""
    from fairai.application.ppo_smoke import run_ppo_smoke
    from fairai.cli.json_io import echo_json
    from fairai.infrastructure.artifact_paths import ArtifactPaths

    result = run_ppo_smoke(
        dataset_path=dataset,
        steps=steps,
        seed=seed,
        artifacts=ArtifactPaths.from_env(root=artifacts),
    )
    echo_json(result.to_dict())
