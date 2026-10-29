#!/usr/bin/env python3
"""Entrypoint: Optuna fair HPO on experiment-gpu (CUDA) image."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path("/app")
    os.chdir(root)
    (root / "artifacts" / "logs").mkdir(parents=True, exist_ok=True)

    print("=== install fairai (editable) + optuna ===", flush=True)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-e",
            ".",
            "typer>=0.12",
            "pyyaml>=6.0",
            "scikit-learn>=1.4",
            "transformers>=4.40",
            "tqdm>=4.66",
            "joblib>=1.3",
            "optuna>=3.6",
        ],
    )

    import torch

    print(
        f"=== torch {torch.__version__} cuda={torch.cuda.is_available()} "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} ===",
        flush=True,
    )

    family = os.environ.get("FAIRAI_OPTUNA_FAMILY", "ce")
    n_trials = os.environ.get("FAIRAI_OPTUNA_N_TRIALS", "100")
    trial_epochs = os.environ.get("FAIRAI_OPTUNA_TRIAL_EPOCHS", "20")
    trial_patience = os.environ.get("FAIRAI_OPTUNA_TRIAL_PATIENCE", "3")
    max_samples = os.environ.get("FAIRAI_OPTUNA_MAX_SAMPLES", "")
    study_name = os.environ.get("FAIRAI_OPTUNA_STUDY", f"optuna_fair_{family}")
    device = os.environ.get("FAIRAI_DEVICE", "cuda")
    config = os.environ.get(
        "FAIRAI_OPTUNA_CONFIG", "configs/experiments/optuna_fair_base.yaml"
    )

    cmd = [
        "fairai",
        "experiment",
        "optuna",
        "-f",
        family,
        "-c",
        config,
        "--n-trials",
        n_trials,
        "--trial-epochs",
        trial_epochs,
        "--trial-patience",
        trial_patience,
        "--device",
        device,
        "--study-name",
        study_name,
        "--artifacts",
        "/app/artifacts",
    ]
    if max_samples.strip():
        cmd.extend(["--max-samples", max_samples.strip()])
    print("=== running:", " ".join(cmd), "===", flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
