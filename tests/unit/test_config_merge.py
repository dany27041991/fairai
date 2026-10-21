from pathlib import Path

from fairai.application.config_merge import load_experiment_config

ROOT = Path(__file__).resolve().parents[2]


def test_smoke_inherits_van_v2() -> None:
    cfg = load_experiment_config(ROOT / "configs/experiments/smoke_cpu.yaml")
    assert cfg["backbone"].startswith("sentence-transformers/")
    assert cfg["train"]["epochs"] == 1
    assert cfg["train"]["batch_size"] == 8
    assert cfg["loss"]["gamma"] > 0
    assert cfg.get("device") == "cpu"
