"""Load YAML experiment configs with optional `inherits:` deep-merge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fairai.infrastructure.torch_van.checkpoint import load_yaml


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if key == "inherits":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_experiment_config(path: Path, *, configs_root: Path | None = None) -> dict[str, Any]:
    """Load YAML; resolve `inherits:` against configs_root."""
    data = load_yaml(path)
    inherits = data.get("inherits")
    if not inherits:
        return data
    root = configs_root
    if root is None:
        # configs/experiments[/rigorous]/x.yaml → configs/
        root = path.parent
        for parent in [path.parent, *path.parents]:
            if (parent / "models").is_dir() and (parent / "experiments").is_dir():
                root = parent
                break
    parent_path = (root / str(inherits)).resolve()
    parent = load_yaml(parent_path)
    # parent may itself inherit
    if parent.get("inherits"):
        parent = load_experiment_config(parent_path, configs_root=root)
    return _deep_merge(parent, data)
