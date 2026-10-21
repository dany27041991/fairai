"""Resolve artifact directories from environment (infrastructure adapter)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default.resolve()


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """Central path resolver — no hardcoded machine paths in business logic."""

    root: Path
    data_dir: Path
    model_dir: Path
    results_dir: Path
    log_dir: Path
    hf_cache: Path

    @classmethod
    def from_env(cls, root: Path | None = None) -> ArtifactPaths:
        base = (root or Path("artifacts")).resolve()
        return cls(
            root=base,
            data_dir=_env_path("FAIRAI_DATA_DIR", base / "datasets"),
            model_dir=_env_path("FAIRAI_MODEL_DIR", base / "models"),
            results_dir=_env_path("FAIRAI_RESULTS_DIR", base / "results"),
            log_dir=_env_path("FAIRAI_LOG_DIR", base / "logs"),
            hf_cache=_env_path("HF_HOME", base / "hf_cache"),
        )

    def processed(self, name: str) -> Path:
        return self.data_dir / "processed" / name

    def splits(self, *parts: str) -> Path:
        return self.data_dir.joinpath("splits", *parts)

    def raw(self, *parts: str) -> Path:
        return self.data_dir.joinpath("raw", *parts)

    def van_checkpoint(self, version: str, filename: str = "final_model.pt") -> Path:
        return self.model_dir / "van" / version / filename

    def metrics_dir(self) -> Path:
        return self.results_dir / "metrics"

    def gold_scores(self) -> Path:
        return self.results_dir / "gold_scores.json"

    def manifest(self) -> Path:
        return self.root / "manifest.sha256.json"

    def ensure_layout(self) -> None:
        for path in (
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "splits",
            self.model_dir / "van",
            self.metrics_dir(),
            self.log_dir,
            self.hf_cache,
        ):
            path.mkdir(parents=True, exist_ok=True)
