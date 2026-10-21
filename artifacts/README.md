# Artifacts

Large scientific artifacts live here (datasets, checkpoints, metrics).  
**Not committed to git** (see `.gitignore`). Populate locally or from a GitHub Release.

## Populate from AS-IS thesis tree

```bash
cd /path/to/fairai
pip install -e ".[dev]"
make import-asis ASIS_ROOT="../TESI MAGISTRALE"
make manifest-verify
```

## Layout

```text
artifacts/
  datasets/
    raw/           # CrowS, StereoSet upstream
    processed/     # van_train_leakage_safe, van_final, …
    splits/        # StereoSet train/test json
  models/van/v1|v2/
  results/gold_scores.json + metrics/
  logs/
  hf_cache/
  manifest.sha256.json
```

## Canonical files (MVP)

| Role | Path |
|------|------|
| Train (leakage-safe) | `datasets/processed/van_train_leakage_safe.jsonl` |
| VAN v2 checkpoint | `models/van/v2/final_model.pt` |
| Gold ICAT scores | `results/gold_scores.json` |

See TO-BE docs under `../TESI MAGISTRALE/00_ASIS_FOR_REWRITE/TOBE/`.
