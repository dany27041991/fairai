"""Use-case: generate / filter / sample training JSONL (M5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from fairai.application.config_merge import load_experiment_config
from fairai.domain.example import Example
from fairai.infrastructure.artifact_paths import ArtifactPaths
from fairai.infrastructure.generation.crows import iter_crows_examples
from fairai.infrastructure.generation.jsonl_io import load_examples_jsonl, write_examples
from fairai.infrastructure.generation.leakage import filter_leakage, stereoset_forbidden_sentences
from fairai.infrastructure.generation.merge import dedupe_examples, merge_examples, sample_balanced
from fairai.infrastructure.generation.schema import validate_examples
from fairai.infrastructure.torch_van.checkpointing import write_json


@dataclass(slots=True)
class GenerateResult:
    out_path: str
    n_written: int
    n_removed_leakage: int
    n_schema_issues: int
    sources_used: list[str]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_extra_jsonl(paths: list[Path], *, max_per_file: int | None) -> list[Example]:
    out: list[Example] = []
    for path in paths:
        if path.is_file():
            out.extend(load_examples_jsonl(path, max_samples=max_per_file))
    return out


def generate_dataset(
    *,
    config_path: Path,
    artifacts: ArtifactPaths | None = None,
    sources: list[str] | None = None,
    max_samples: int | None = None,
    expand_synonyms: bool | None = None,
    out_name: str | None = None,
) -> GenerateResult:
    """
    Build a regenerated subset under artifacts/datasets/generated/.

    Canonical processed files (van_train_leakage_safe, van_final, …) are never overwritten.
    """
    cfg = load_experiment_config(config_path) if config_path.is_file() else {}
    paths = artifacts or ArtifactPaths.from_env()
    gen_cfg = dict(cfg.get("generate", cfg))

    source_list = sources or list(gen_cfg.get("sources", ["crows"]))
    max_n = max_samples if max_samples is not None else gen_cfg.get("max_samples")
    max_n_i = int(max_n) if max_n is not None else None
    do_syn = (
        bool(expand_synonyms)
        if expand_synonyms is not None
        else bool(gen_cfg.get("expand_synonyms", False))
    )
    max_crows_rows = gen_cfg.get("max_crows_rows", max_n_i)
    max_crows_i = int(max_crows_rows) if max_crows_rows is not None else None

    groups: list[list[Example]] = []
    used: list[str] = []

    if "crows" in source_list:
        crows_csv = paths.raw("crows_pairs", "crows_pairs_anonymized.csv")
        if not crows_csv.is_file():
            raise FileNotFoundError(f"CrowS CSV missing: {crows_csv}")
        groups.append(
            list(
                iter_crows_examples(
                    crows_csv,
                    expand_synonyms=do_syn,
                    max_synonym_samples=int(gen_cfg.get("max_synonym_samples", 3)),
                    max_rows=max_crows_i,
                )
            )
        )
        used.append("crows")

    extra_globs = gen_cfg.get("extra_jsonl") or []
    if "jsonl" in source_list or extra_globs:
        extras: list[Path] = []
        for rel in extra_globs:
            extras.append(paths.root / rel if not Path(rel).is_absolute() else Path(rel))
        # optional thesis-processed siblings already imported under processed/
        for rel in gen_cfg.get("processed_jsonl", []):
            extras.append(paths.data_dir / rel)
        loaded = _load_extra_jsonl(extras, max_per_file=max_n_i)
        if loaded:
            groups.append(loaded)
            used.append("jsonl")

    if not groups:
        raise ValueError(f"No sources produced data from {source_list}")

    merged = merge_examples(*groups)
    merged = dedupe_examples(merged)

    removed = 0
    if bool(gen_cfg.get("filter_leakage", True)):
        splits = paths.splits("stereoset")
        forbidden = stereoset_forbidden_sentences(splits)
        merged, removed = filter_leakage(merged, forbidden)

    if max_n_i is not None and len(merged) > max_n_i:
        merged = sample_balanced(merged, max_samples=max_n_i, seed=int(gen_cfg.get("seed", 42)))

    issues = validate_examples(merged)
    if issues:
        # drop invalid
        valid: list[Example] = []
        bad_lines = {iss.line for iss in issues}
        for i, ex in enumerate(merged, start=1):
            if i not in bad_lines:
                valid.append(ex)
        merged = valid

    out_rel = out_name or str(gen_cfg.get("out", "generated/smoke_crows.jsonl"))
    out_path = paths.data_dir / out_rel
    n = write_examples(out_path, merged)
    meta = {
        "out": str(out_path),
        "n": n,
        "removed_leakage": removed,
        "schema_issues": len(issues),
        "sources": used,
        "canonical_note": (
            "Imported processed/*.jsonl remain the training source of truth; "
            "this file is a regenerated subset for validation only."
        ),
    }
    write_json(out_path.with_suffix(out_path.suffix + ".meta.json"), meta)

    return GenerateResult(
        out_path=str(out_path),
        n_written=n,
        n_removed_leakage=removed,
        n_schema_issues=len(issues),
        sources_used=used,
        note=meta["canonical_note"],
    )
