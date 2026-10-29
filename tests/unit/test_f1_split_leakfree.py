from fairai.application.build_select_options_f1_split import (
    concat_text,
    leakage_key,
    prompt_has_leakage,
)


def test_leakage_key_normalizes() -> None:
    assert leakage_key("  Hello   World. ") == leakage_key("Hello World.")


def test_concat_text() -> None:
    assert concat_text("ctx", "opt") == "ctx [SEP] opt"
    assert concat_text("", "opt") == "opt"


def test_prompt_has_leakage() -> None:
    forbidden = {leakage_key("bad option")}
    dirty = {
        "context": "c",
        "options": [{"sentence": "bad option", "label": 2}],
    }
    clean = {
        "context": "c",
        "options": [{"sentence": "unique neutral here", "label": 2}],
    }
    assert prompt_has_leakage(dirty, forbidden)
    assert not prompt_has_leakage(clean, forbidden)
