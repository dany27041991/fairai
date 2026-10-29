#!/usr/bin/env bash
# Wave C: ICAT parity vs committed baseline
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
EVID="$ROOT/docs/verification/EVIDENCE"
mkdir -p "$EVID"

export FAIRAI_DATA_DIR= FAIRAI_MODEL_DIR= FAIRAI_RESULTS_DIR=

{
  echo "=== icat test vs icat_test.json (if present) ==="
  if [[ -f artifacts/results/metrics/icat_test.json ]]; then
    fairai icat --subset test --device cpu --batch-size 64 \
      --baseline artifacts/results/metrics/icat_test.json 2>&1 | tail -30
  else
    echo "SKIP no icat_test.json — writing fresh"
    fairai icat --subset test --device cpu --batch-size 64 \
      --out artifacts/results/metrics/icat_test.json 2>&1 | tail -20
  fi
  echo "=== icat all vs configs/baselines/baseline_van_v2_icat.json ==="
  fairai icat --subset all --device cpu --batch-size 64 \
    --baseline configs/baselines/baseline_van_v2_icat.json 2>&1 | tail -40
} | tee "$EVID/wave_c.txt"
