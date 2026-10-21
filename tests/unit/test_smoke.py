from fairai.application.smoke import SmokeResult, smoke_passed


def test_smoke_passed_requires_pytest_and_budget() -> None:
    assert not smoke_passed(
        SmokeResult(pytest_ok=False, file_budget_ok=True, forward_ok=None, detail="")
    )
    assert smoke_passed(
        SmokeResult(pytest_ok=True, file_budget_ok=True, forward_ok=None, detail="")
    )
    assert not smoke_passed(
        SmokeResult(pytest_ok=True, file_budget_ok=True, forward_ok=False, detail="")
    )
    assert smoke_passed(
        SmokeResult(pytest_ok=True, file_budget_ok=True, forward_ok=True, detail="")
    )
