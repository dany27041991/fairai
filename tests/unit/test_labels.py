from fairai.domain import Example, Label


def test_label_roundtrip() -> None:
    assert Label.from_int(0) is Label.STEREOTYPE
    assert Label.NEUTRAL.display_name == "Neutral"


def test_example_from_record() -> None:
    ex = Example.from_record(
        {"sentence": "hello", "label": 2, "bias_type": "none"}
    )
    assert ex.label is Label.NEUTRAL
    assert ex.bias_type is None
    assert ex.to_record()["label"] == 2
