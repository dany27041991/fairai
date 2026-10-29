# Reproduce IEEE Software tables (unified protocol)

**Software copyright:** Danilo Giovannico, University of Salento — see root `LICENSE` (not MIT).

**Paper SoT:** numbers in `paper/ieee_software_stereotypes.tex` Table A come from this protocol.
Primary released artifact: `artifacts/results/metrics/f1_test_leakfree_released.json`
(F1 leak-free **0.962** + iCAT test from `icat_test_verify.json`).
Table B **SoT:** `artifacts/results/metrics/optuna_product_compare_agg.json`  
VAN = released Optuna checkpoint (**0.962 / 96.28 / 96.71**); CE/−BPL = equal-budget Optuna;  
**same** leak-free F1 + StereoSet-test iCAT. Suite: `suite_optuna_product.yaml`.

Archive (from-scratch VAN re-Optuna): `optuna_equal_product_compare_agg.json`.  
Do **not** use `select_options_test_f1.json` (~48% train overlap) for Table A.

Re-eval Table B only (existing checkpoints):

```bash
fairai experiment suite \
  -c configs/experiments/suite_rigorous.yaml \
  --skip-train --device cpu   # or cuda
# or suite_rigorous_clean.yaml for rigc-* after that retrain finishes
```

## Protocol

| Axis | Resource | Rule |
|------|----------|------|
| **F1** | `artifacts/datasets/processed/stereoset/f1_test_leakfree.jsonl` | select-options with **neutral** label 2; 0 exact overlap vs train/val corpus + select-options train |
| **iCAT** | StereoSet **`subset=test`** | adapted 3-class preference; `avg_icat=(inter+intra)/2` |

## One-command suite (eval only)

```bash
# 1) Build leak-free F1 test (once)
fairai analyze f1-split

# 2) Neutral audit sample (optional human labels)
fairai analyze neutral-audit

# 3) Re-eval released + rigorous runs
fairai experiment suite \
  -c configs/experiments/suite_unified.yaml \
  --skip-train \
  --device cuda   # or cpu
```

Outputs:

- `artifacts/results/metrics/unified_compare.json`
- `artifacts/results/metrics/unified_compare_agg.json` (mean±std by family)
- `artifacts/analysis/f1_test_leakfree_report.json` (must show `overlap_exact_vs_forbidden: 0`)
- `artifacts/results/metrics/f1_test_leakfree_released.json` (Table A SoT after released-only eval)

## Released-only smoke

```bash
fairai experiment eval-run \
  -r artifacts/models/runs/released-van-v2 \
  -d artifacts/datasets/processed/stereoset/f1_test_leakfree.jsonl \
  --splits-dir artifacts/datasets/splits/stereoset \
  --subset test \
  --device cpu
```

## Unit checks (adapted iCAT)

```bash
pytest -q tests/unit/test_icat_formula.py tests/unit/test_f1_split_leakfree.py
```
