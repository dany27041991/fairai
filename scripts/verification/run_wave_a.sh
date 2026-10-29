#!/usr/bin/env bash
# Wave A: artifact counts + manifest verify
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
EVID="$ROOT/docs/verification/EVIDENCE"
mkdir -p "$EVID"

{
  echo "=== line counts ==="
  wc -l artifacts/datasets/processed/van_train_leakage_safe.jsonl \
        artifacts/datasets/processed/van_final.jsonl \
        artifacts/datasets/processed/van_minimal_24k.jsonl
  echo "=== presence ==="
  for f in \
    artifacts/models/van/v2/final_model.pt \
    artifacts/models/van/v2/best_trial_params.json \
    artifacts/models/van/v2/best_threshold_params.json \
    artifacts/results/gold_scores.json \
    artifacts/datasets/splits/stereoset/intersentence_train.json \
    artifacts/datasets/splits/stereoset/intersentence_test.json \
    artifacts/datasets/splits/stereoset/intrasentence_train.json \
    artifacts/datasets/splits/stereoset/intrasentence_test.json \
    artifacts/datasets/raw/crows_pairs/crows_pairs_anonymized.csv \
    artifacts/datasets/processed/stereoset/stereoset_select_options_pred.json
  do
    if [[ -f "$f" ]]; then echo "OK $f"; else echo "MISSING $f"; fi
  done
  echo "=== manifest ==="
  fairai manifest --artifacts artifacts --verify || true
} | tee "$EVID/wave_a.txt"
