"""Use-case: import AS-IS thesis artifacts into fairai/artifacts."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fairai.application.asis_mapping import CANONICAL_SPECS, CopySpec
from fairai.infrastructure.hashing import sha256_file
from fairai.infrastructure.torch_van.checkpointing import write_json


@dataclass(slots=True)
class ImportResult:
    copied: list[str]
    skipped: list[str]
    missing_required: list[str]
    missing_optional: list[str]
    manifest_path: Path


def import_asis_artifacts(
    source_root: Path,
    artifacts_root: Path,
    *,
    specs: tuple[CopySpec, ...] = CANONICAL_SPECS,
    dry_run: bool = False,
) -> ImportResult:
    source_root = source_root.resolve()
    artifacts_root = artifacts_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"AS-IS root not found: {source_root}")

    copied: list[str] = []
    skipped: list[str] = []
    missing_required: list[str] = []
    missing_optional: list[str] = []
    manifest_files: list[dict] = []

    for spec in specs:
        src = source_root / spec.relative_source
        dest = artifacts_root / spec.relative_dest
        if not src.is_file():
            (missing_required if spec.required else missing_optional).append(
                spec.relative_source
            )
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        src_hash = sha256_file(src)
        if dest.is_file() and sha256_file(dest) == src_hash:
            skipped.append(spec.relative_dest)
        elif dry_run:
            copied.append(f"(dry-run) {spec.relative_dest}")
        else:
            shutil.copy2(src, dest)
            copied.append(spec.relative_dest)

        target = dest if dest.is_file() else src
        manifest_files.append(
            {
                "relpath": spec.relative_dest,
                "sha256": src_hash if not dest.is_file() else sha256_file(target),
                "bytes": target.stat().st_size if target.is_file() else src.stat().st_size,
                "role": spec.role,
                "required": spec.required,
            }
        )

    manifest_path = artifacts_root / "manifest.sha256.json"
    if not dry_run:
        payload = {
            "created_at": datetime.now(UTC).isoformat(),
            "source_root": str(source_root),
            "files": manifest_files,
        }
        write_json(manifest_path, payload)

    if missing_required:
        raise FileNotFoundError(
            "Missing required AS-IS files:\n  - " + "\n  - ".join(missing_required)
        )

    return ImportResult(
        copied=copied,
        skipped=skipped,
        missing_required=missing_required,
        missing_optional=missing_optional,
        manifest_path=manifest_path,
    )
