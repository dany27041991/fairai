from fairai.infrastructure.llm.prompts import (
    build_example_prompt,
    gold_label,
    is_valid_example,
    parse_prediction,
)


def _ex() -> dict:
    return {
        "context": "Ctx",
        "options": [
            {"sentence": "a", "label": 0, "correct": False, "label_predicted": 0},
            {"sentence": "b", "label": 1, "correct": False, "label_predicted": 1},
            {"sentence": "c", "label": 2, "correct": True, "label_predicted": 2},
        ],
    }


def test_prompt_and_parse() -> None:
    text = build_example_prompt(_ex())
    assert "Label 2" in text
    assert parse_prediction("The best answer is: Label 2") == 2
    assert gold_label(_ex()) == 2
    assert is_valid_example(_ex())


def test_invalid_example() -> None:
    bad = {
        "context": "x",
        "options": [
            {"sentence": "a", "label": 0, "correct": True},
            {"sentence": "b", "label": 0, "correct": True},
        ],
    }
    assert not is_valid_example(bad)
