from pathlib import Path

from fairai.application.audit_leakage import audit_train_leakage
from fairai.domain.example import Example
from fairai.domain.labels import Label
from fairai.infrastructure.generation.jsonl_io import write_examples
from fairai.infrastructure.generation.leakage import filter_leakage


def test_filter_leakage_removes_forbidden() -> None:
    exs = [
        Example(sentence="Hello world.", label=Label.STEREOTYPE, bias_type="race"),
        Example(sentence="Keep me please.", label=Label.ANTI_STEREOTYPE, bias_type="gender"),
    ]
    kept, n = filter_leakage(exs, {"hello world."})
    assert n == 1 and len(kept) == 1 and kept[0].sentence.startswith("Keep")


def test_audit_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    write_examples(
        path,
        [Example(sentence="Only train.", label=Label.NEUTRAL, bias_type=None)],
    )
    splits = tmp_path / "splits"
    splits.mkdir()
    # empty splits → 0 forbidden
    report = audit_train_leakage(path, splits_dir=splits, out_report=tmp_path / "r.json")
    assert report.n_total == 1 and report.n_hits == 0
    try:
        audit_train_leakage(path, splits_dir=splits, out_strict=path)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
