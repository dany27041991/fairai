#!/usr/bin/env bash
# Baselines under VAN Optuna policy (full leakage-safe data from van_final).
# Keeps released VAN; retunes CE then −BPL; 3-seed finals; product-plane eval.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/logs
LOG="artifacts/logs/optuna_vanpolicy_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=== optuna vanpolicy pipeline start $(date -Is) log=$LOG ==="

# Free GPU if vLLM holds VRAM (optional; ignore errors).
docker stop lexit-enrichment-vllm 2>/dev/null || true

make optuna-vanpolicy-ce
make optuna-vanpolicy-nobpl

python3 - <<'PY'
from pathlib import Path
from fairai.application.run_optuna_hpo import write_finalize_suite
from fairai.infrastructure.artifact_paths import ArtifactPaths

p = write_finalize_suite(
    artifacts=ArtifactPaths.from_env(root=Path("artifacts")),
    families=("ce", "no_bpl"),
    study_prefix="optuna_vanpolicy",
    config_prefix="optuna_vanpolicy",
    run_id_prefix="ovp",
    eval_dataset="processed/stereoset/f1_test_leakfree.jsonl",
    eval_out_dir="results/metrics/eval_reports_ovp",
    out_metrics="results/metrics/optuna_vanpolicy_baselines.json",
    out_suite=Path("configs/experiments/suite_optuna_vanpolicy_baselines.yaml"),
    suite_name="suite_optuna_vanpolicy_baselines",
)
print("wrote", p)
PY

# 3-seed final trains for CE / −BPL only
FAIRAI_SUITE=configs/experiments/suite_optuna_vanpolicy_baselines.yaml \
  docker compose --profile experiment run --rm \
  -e FAIRAI_SUITE=configs/experiments/suite_optuna_vanpolicy_baselines.yaml \
  experiment-gpu

# Product compare: reuse trained run_dirs (no second train) + released VAN + frozen
python3 - <<'PY'
from pathlib import Path
import yaml
suite = {
    "name": "suite_optuna_vanpolicy_product",
    "device": "cuda",
    "aggregate": True,
    "experiments": [
        {
            "config": "models/van_v2.yaml",
            "run_id": "released-van-v2",
            "run_dir": "artifacts/models/runs/released-van-v2",
        },
        *[
            {
                "config": f"experiments/optuna_vanpolicy_{fam}_best.yaml",
                "run_id": f"ovp-{fam}-s{seed}",
                "run_dir": f"artifacts/models/runs/ovp-{fam}-s{seed}",
            }
            for fam in ("ce", "no_bpl")
            for seed in (42, 123, 7)
        ],
        {
            "config": "experiments/rigorous_baseline_frozen_lr.yaml",
            "run_id": "rigc-baseline_frozen_lr-s42",
            "run_dir": "artifacts/models/runs/rigc-baseline_frozen_lr-s42",
        },
    ],
    "eval": {
        "dataset": "processed/stereoset/f1_test_leakfree.jsonl",
        "batch_size": 64,
        "subset": "test",
        "stereoset_splits": "splits/stereoset",
        "eval_out_dir": "results/metrics/eval_reports_ovp_product",
    },
    "out": "results/metrics/optuna_vanpolicy_product_compare.json",
}
path = Path("configs/experiments/suite_optuna_vanpolicy_product.yaml")
path.write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")
print("wrote", path)
PY

FAIRAI_SUITE=configs/experiments/suite_optuna_vanpolicy_product.yaml \
  docker compose --profile experiment run --rm \
  -e FAIRAI_SUITE=configs/experiments/suite_optuna_vanpolicy_product.yaml \
  experiment-gpu

echo "=== optuna vanpolicy pipeline done $(date -Is) ==="
ls -la artifacts/results/metrics/optuna_vanpolicy_*.json || true
