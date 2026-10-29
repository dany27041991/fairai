from fairai.infrastructure.llm.few_shot_metrics import (
    compute_classification_metrics,
    score_int_labels,
)
from fairai.infrastructure.llm.labels import (
    get_confusion_order,
    get_numeric_id,
    improvement_percentage,
)
from fairai.infrastructure.llm.prompts import (
    build_example_prompt,
    build_full_few_shot_prompt,
    build_test_prompt_baseline,
    build_test_prompt_few_shot,
    gold_label,
    is_valid_example,
    parse_prediction,
    requires_neutral_predicted,
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
    assert "label_predicted" not in text or "Label 0" in text
    assert parse_prediction("The best answer is: Label 2") == 2
    assert gold_label(_ex()) == 2
    assert is_valid_example(_ex())
    assert requires_neutral_predicted(_ex())
    few = build_test_prompt_few_shot(_ex())
    assert "Remember that Label 2 is the neutral option" in few
    assert "The best answer is: Label " in few
    base = build_test_prompt_baseline(_ex())
    assert "Answer with only one of" in base
    full = build_full_few_shot_prompt(text, _ex())
    assert "Remember that Label 2" in full


def test_labels_and_metrics() -> None:
    assert get_numeric_id("2") == 3
    assert get_numeric_id("0") == -2
    nums, names = get_confusion_order()
    assert 0 not in nums  # None excluded
    assert "Neutral" in names
    assert improvement_percentage(1.0, 1.5) == 50.0
    metrics = compute_classification_metrics([_ex()], ["2"])
    assert metrics["f1_macro"] >= 0.0
    acc, f1 = score_int_labels([2, 0], [2, 1])
    assert acc == 0.5 and 0.0 <= f1 <= 1.0


def test_invalid_example() -> None:
    bad = {
        "context": "x",
        "options": [
            {"sentence": "a", "label": 0, "correct": True},
            {"sentence": "b", "label": 0, "correct": True},
        ],
    }
    assert not is_valid_example(bad)
