# M8 — Publication checklist

## Before first push

1. `git init` in `fairai/` (if not already a repo)
2. Confirm `.gitignore` excludes `artifacts/{datasets,models,results,hf_cache}` and `.env`
3. `make smoke` and `make lint` green
4. Create GitHub repo (empty), push `main`
5. Tag `v0.1.0`

## Release assets (large)

Upload separately (GitHub Release), do **not** commit:

- `artifacts/models/van/v2/final_model.pt` (+ optional `best_model.pt`)
- `artifacts/manifest.sha256.json`
- Optional: StereoSet splits + `gold_scores.json` zip

Document in the Release body:

```text
fairai import-asis   # or download Release assets into artifacts/
make smoke
docker compose --profile ci run --rm smoke
FAIRAI_DEVICE=cpu docker compose --profile eval run --rm eval
fairai icat --subset all --baseline configs/baselines/baseline_van_v2_icat.json
```

## Baseline ICAT (VAN v2, subset=all)

- avg_icat ≈ **96.10** (n=12687)
- Tracked copy: `configs/baselines/baseline_van_v2_icat.json`
