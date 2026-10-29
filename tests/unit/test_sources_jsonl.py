import json
from pathlib import Path

from fairai.infrastructure.generation.sources_jsonl import (
    load_named_source,
    resolve_source_files,
)


def test_load_winobias_style(tmp_path: Path) -> None:
    root = tmp_path / "wino_bias"
    root.mkdir()
    path = root / "type1_pro.jsonl"
    path.write_text(
        json.dumps({"sentence": "She is a nurse.", "label": 0, "bias_type": "gender"}) + "\n",
        encoding="utf-8",
    )
    files = resolve_source_files(tmp_path, "winobias")
    assert files == [path]
    exs = load_named_source(tmp_path, "winobias")
    assert len(exs) == 1 and exs[0].label == 0


def test_stereoset_synonym_expansion(tmp_path: Path) -> None:
    root = tmp_path / "stereoset"
    root.mkdir()
    path = root / "stereoset_dataset.jsonl"
    rec = {
        "sentence": "The chess player was hispanic.",
        "label": 0,
        "bias_type": "profession",
        "synonym_sentences": ["The chess participant was hispanic."],
    }
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    exs = load_named_source(tmp_path, "stereoset", max_samples=10)
    assert len(exs) == 2
