#!/usr/bin/env python3
"""Entrypoint for docker compose experiment-gpu (vLLM CUDA base image)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path("/app")
    os.chdir(root)
    log_dir = root / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=== install fairai (editable) ===", flush=True)
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
        ],
    )

    import torch

    print(
        f"=== torch {torch.__version__} cuda={torch.cuda.is_available()} "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} ===",
        flush=True,
    )

    suite = os.environ.get(
        "FAIRAI_SUITE", "configs/experiments/suite_full.yaml"
    )
    device = os.environ.get("FAIRAI_DEVICE", "cuda")
    cmd = [
        "fairai",
        "experiment",
        "suite",
        "-c",
        suite,
        "--device",
        device,
        "--artifacts",
        "/app/artifacts",
    ]
    print("=== running:", " ".join(cmd), "===", flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
