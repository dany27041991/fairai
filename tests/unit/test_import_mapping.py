from fairai.application.asis_mapping import CANONICAL_SPECS


def test_canonical_roles_unique() -> None:
    roles = [s.role for s in CANONICAL_SPECS]
    assert len(roles) == len(set(roles))


def test_v2_final_required() -> None:
    spec = next(s for s in CANONICAL_SPECS if s.role == "van_v2_final")
    assert spec.required
    assert spec.relative_dest.endswith("final_model.pt")
