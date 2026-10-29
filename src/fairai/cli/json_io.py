"""Shared CLI JSON echo / write helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import typer

from fairai.infrastructure.torch_van.checkpointing import write_json


def echo_json(payload: Any, *, ensure_ascii: bool = True) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=ensure_ascii))


def write_and_echo(
    payload: Mapping[str, Any],
    *,
    out: Path | None = None,
    ensure_ascii: bool = True,
) -> None:
    echo_json(payload, ensure_ascii=ensure_ascii)
    if out is not None:
        write_json(out, dict(payload), ensure_ascii=ensure_ascii)
