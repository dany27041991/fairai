#!/usr/bin/env bash
# Sequential equal-budget Optuna fair compare: CE → −BPL → VAN → finalize suite.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/logs
LOG="artifacts/logs/optuna_fair_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== optuna fair pipeline start $(date -Is) log=$LOG ==="
make optuna-fair-ce
make optuna-fair-nobpl
make optuna-fair-van
# Finalize suite YAML on host (no GPU needed), then GPU final 3-seed trains.
python3 - <<'PY'
from pathlib import Path
from fairai.application.run_optuna_hpo import write_finalize_suite
from fairai.infrastructure.artifact_paths import ArtifactPaths
p = write_finalize_suite(
    artifacts=ArtifactPaths.from_env(root=Path("artifacts")),
    out_suite=Path("configs/experiments/suite_optuna_fair.yaml"),
)
print("wrote", p)
PY
FAIRAI_SUITE=configs/experiments/suite_optuna_fair.yaml \
  docker compose --profile experiment run --rm \
  -e FAIRAI_SUITE=configs/experiments/suite_optuna_fair.yaml experiment-gpu
echo "=== optuna fair pipeline done $(date -Is) ==="
ls -la artifacts/results/metrics/optuna_fair_compare*.json
