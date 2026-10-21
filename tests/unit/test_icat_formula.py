from fairai.application.parity import check_parity, parity_ok
from fairai.domain.icat import bundle_from_probs, compute_icat_score, lm_ss_from_probs


def test_icat_formula() -> None:
    assert compute_icat_score(100.0, 50.0) == 100.0
    assert compute_icat_score(80.0, 0.0) == 0.0
    assert abs(compute_icat_score(90.0, 60.0) - 90.0 * (40.0 / 50.0)) < 1e-9


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


def test_parity_pass_fail() -> None:
    actual = {"avg_icat": 95.0, "intersentence": {"icat": 94.0}}
    base = {"avg_icat": 95.5, "intersentence": {"icat": 96.0}}
    checks = check_parity(
        actual, base, {"avg_icat": 1.0, "intersentence.icat": 1.0}
    )
    assert checks[0].ok
    assert not checks[1].ok
    assert not parity_ok(checks)
