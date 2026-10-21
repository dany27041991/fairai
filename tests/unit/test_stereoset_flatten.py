from pathlib import Path

from fairai.infrastructure.data.stereoset_flatten import flatten_split_file

SPLITS = Path(__file__).resolve().parents[2] / "artifacts/datasets/splits/stereoset"


def test_flatten_intersentence_test_if_present() -> None:
    path = SPLITS / "intersentence_test.json"
    if not path.is_file():
        return
    flat = flatten_split_file(path, concatenate_context=True, sep_token="[SEP]")
    assert len(flat) > 0
    assert all(ex.label in (0, 1, 2) for ex in flat)
    assert "[SEP]" in flat[0].sentence
