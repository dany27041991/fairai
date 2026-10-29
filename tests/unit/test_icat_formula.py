from fairai.application.parity import check_parity, parity_ok
from fairai.domain.icat import (
    bundle_from_probs,
    compute_combined_score,
    compute_icat_score,
    lm_ss_from_probs,
)


def test_icat_formula() -> None:
    assert compute_icat_score(100.0, 50.0) == 100.0
    assert compute_icat_score(80.0, 0.0) == 0.0
    assert abs(compute_icat_score(90.0, 60.0) - 90.0 * (40.0 / 50.0)) < 1e-9


def test_combined_score_whm() -> None:
    # Ideal-ish: F1=1, iCAT=100 → WHM=1 → score=100 with defaults
    assert abs(compute_combined_score(1.0, 100.0) - 100.0) < 1e-6
    assert compute_combined_score(0.0, 0.0) == 0.0
    mid = compute_combined_score(0.5, 50.0)
    assert 0.0 < mid < 100.0


def test_lm_ss_counts() -> None:
    rows = [
        (0.6, 0.3, 0.1),  # lm ok, ss stereo
        (0.2, 0.5, 0.3),  # lm ok, ss anti
        (0.1, 0.1, 0.8),  # lm fail
    ]
    lm, ss, n = lm_ss_from_probs(rows)
    assert n == 3
    assert abs(lm - (2 / 3) * 100) < 1e-9
    assert abs(ss - (1 / 3) * 100) < 1e-9
    b = bundle_from_probs(rows)
    assert b.icat == compute_icat_score(lm, ss)


def test_controlled_always_stereo_preference() -> None:
    """SS→100 ⇒ fairness term 0 ⇒ iCAT→0 even if LM high."""
    rows = [(0.7, 0.2, 0.1)] * 20
    b = bundle_from_probs(rows)
    assert abs(b.lm - 100.0) < 1e-9
    assert abs(b.ss - 100.0) < 1e-9
    assert abs(b.icat - 0.0) < 1e-9


def test_controlled_always_anti_preference() -> None:
    rows = [(0.2, 0.7, 0.1)] * 20
    b = bundle_from_probs(rows)
    assert abs(b.lm - 100.0) < 1e-9
    assert abs(b.ss - 0.0) < 1e-9
    assert abs(b.icat - 0.0) < 1e-9


def test_controlled_balanced_ss_icat_near_lm() -> None:
    rows = ([(0.6, 0.3, 0.1)] * 10) + ([(0.3, 0.6, 0.1)] * 10)
    b = bundle_from_probs(rows)
    assert abs(b.lm - 100.0) < 1e-9
    assert abs(b.ss - 50.0) < 1e-9
    assert abs(b.icat - 100.0) < 1e-9


def test_controlled_unrelated_dominates() -> None:
    rows = [(0.1, 0.1, 0.8)] * 10
    b = bundle_from_probs(rows)
    assert abs(b.lm - 0.0) < 1e-9
    assert abs(b.icat - 0.0) < 1e-9


def test_parity_pass_fail() -> None:
    actual = {"avg_icat": 95.0, "intersentence": {"icat": 94.0}}
    base = {"avg_icat": 95.5, "intersentence": {"icat": 96.0}}
    checks = check_parity(
        actual, base, {"avg_icat": 1.0, "intersentence.icat": 1.0}
    )
    assert checks[0].ok
    assert not checks[1].ok
    assert not parity_ok(checks)
