from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "artifacts/models/van/v2/final_model.pt"
CFG = ROOT / "configs/models/van_v2.yaml"


@pytest.mark.skipif(not CKPT.is_file(), reason="VAN v2 checkpoint not imported")
def test_load_van_v2_state_dict_strict() -> None:
    from fairai.infrastructure.torch_van.checkpoint import load_van_classifier

    model, thresholds, raw = load_van_classifier(CKPT, CFG, device="cpu", strict=True)
    assert raw["name"] == "van_v2"
    assert thresholds["temp"] > 0
    # forward smoke
    ids = torch.randint(0, 1000, (2, 16))
    mask = torch.ones(2, 16, dtype=torch.long)
    logits, pooled = model(ids, mask)
    assert logits.shape == (2, 3)
    assert pooled.shape[0] == 2
