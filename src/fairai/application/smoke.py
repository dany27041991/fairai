"""CI / Compose smoke: unit tests + optional VAN forward pass."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SmokeResult:
    pytest_ok: bool
    file_budget_ok: bool
    forward_ok: bool | None
    detail: str


def run_smoke(
    *,
    repo_root: Path,
    checkpoint: Path,
    config: Path,
    device: str = "cpu",
    skip_forward: bool = False,
) -> SmokeResult:
    """Run pytest + file-budget check; optionally load VAN and forward one batch."""
    details: list[str] = []

    pytest_rc = subprocess.call(
        [sys.executable, "-m", "pytest", "-q", "tests/unit"],
        cwd=repo_root,
    )
    pytest_ok = pytest_rc == 0
    details.append(f"pytest={'ok' if pytest_ok else 'fail'}")

    from fairai.infrastructure.file_budget import find_oversized_python_files

    offenders = find_oversized_python_files(repo_root / "src", limit=500)
    file_budget_ok = not offenders
    if offenders:
        details.append(f"oversized={[f'{p}:{n}' for p, n in offenders]}")
    else:
        details.append("file_budget=ok")

    forward_ok: bool | None = None
    if skip_forward:
        details.append("forward=skipped")
    elif not checkpoint.is_file():
        forward_ok = None
        details.append(f"forward=skip (missing {checkpoint})")
    else:
        try:
            import torch

            from fairai.infrastructure.torch_van.checkpoint import load_van_classifier

            model, _thr, _raw = load_van_classifier(
                checkpoint, config, device=device, strict=True
            )
            model.eval()
            ids = torch.randint(0, 1000, (2, 16), device=device)
            mask = torch.ones(2, 16, dtype=torch.long, device=device)
            with torch.no_grad():
                logits, pooled = model(ids, mask)
            assert logits.shape == (2, 3)
            assert pooled.shape[0] == 2
            forward_ok = True
            details.append("forward=ok")
        except Exception as exc:  # noqa: BLE001 — surface in smoke report
            forward_ok = False
            details.append(f"forward=fail ({exc})")

    return SmokeResult(
        pytest_ok=pytest_ok,
        file_budget_ok=file_budget_ok,
        forward_ok=forward_ok,
        detail="; ".join(details),
    )


def smoke_passed(result: SmokeResult) -> bool:
    if not (result.pytest_ok and result.file_budget_ok):
        return False
    if result.forward_ok is False:
        return False
    return True
