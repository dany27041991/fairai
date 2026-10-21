from pathlib import Path

from fairai.infrastructure.file_budget import find_oversized_python_files


def test_src_under_line_budget() -> None:
    root = Path(__file__).resolve().parents[2] / "src"
    offenders = find_oversized_python_files(root, limit=500)
    assert offenders == [], f"files over 500 lines: {offenders}"
