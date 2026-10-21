# Dataset generation (M5)

## Canonical vs regenerated

| Path | Role |
|------|------|
| `artifacts/datasets/processed/van_train_leakage_safe.jsonl` | **Canonical** train (imported AS-IS, 50409) |
| `artifacts/datasets/processed/van_final.jsonl` | Canonical pool |
| `artifacts/datasets/generated/*.jsonl` | **Regenerated** subsets for pipeline tests |

`fairai generate` **never** overwrites canonical `processed/` files. Retrain for parity still uses the imported leakage-safe set unless you explicitly point `--dataset` elsewhere.

## Smoke

```bash
fairai generate -c configs/data/generate.yaml
# writes artifacts/datasets/generated/smoke_crows.jsonl (+ .meta.json)
```

Sources today: CrowS-Pairs CSV (`artifacts/datasets/raw/crows_pairs/…`) with optional WordNet synonyms (`--synonyms`) and StereoSet-test leakage filter.

## Full rebuild (optional)

Import thesis per-source jsonl or extend `processed_jsonl` / `extra_jsonl` in the YAML, then:

```bash
fairai generate -c configs/data/generate.yaml --sources crows,jsonl --max-samples 5000
```

Ollama-based sentence evolution remains optional (Compose profile `gen` reserved); not required for M5 acceptance.
