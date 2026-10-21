from pathlib import Path

from fairai.domain.example import Example
from fairai.domain.labels import Label
from fairai.infrastructure.generation.crows import iter_crows_examples
from fairai.infrastructure.generation.leakage import filter_leakage
from fairai.infrastructure.generation.merge import dedupe_examples, sample_balanced
from fairai.infrastructure.generation.schema import validate_examples, validate_record

ROOT = Path(__file__).resolve().parents[2]
CROWS = ROOT / "artifacts/datasets/raw/crows_pairs/crows_pairs_anonymized.csv"


def test_schema_ok() -> None:
    assert not validate_record({"sentence": "Hello.", "label": 0})
    assert validate_record({"sentence": "", "label": 0})


def test_dedupe_and_sample() -> None:
    examples = [
        Example("A.", Label.STEREOTYPE, "race"),
        Example("A.", Label.STEREOTYPE, "race"),
        Example("B.", Label.ANTI_STEREOTYPE, "gender"),
        Example("C.", Label.NEUTRAL, None),
        Example("D.", Label.STEREOTYPE, "race"),
        Example("E.", Label.ANTI_STEREOTYPE, "gender"),
        Example("F.", Label.NEUTRAL, None),
    ]
    uniq = dedupe_examples(examples)
    assert len(uniq) == 6
    sampled = sample_balanced(uniq, max_samples=3, seed=0)
    assert len(sampled) == 3


def test_leakage_filter() -> None:
    examples = [
        Example("Unique train sentence.", Label.NEUTRAL),
        Example("Forbidden.", Label.STEREOTYPE),
    ]
    kept, removed = filter_leakage(examples, {"forbidden."})
    assert removed == 1
    assert len(kept) == 1


def test_crows_smoke_rows() -> None:
    if not CROWS.is_file():
        return
    rows = list(iter_crows_examples(CROWS, expand_synonyms=False, max_rows=5))
    assert len(rows) == 10  # more + less per row
    assert not validate_examples(rows)
    assert all(ex.label in Label for ex in rows)
