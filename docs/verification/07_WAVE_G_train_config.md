# Wave G — Train config

**Status:** PASS  
**Agent:** train  
**Verified:** 2026-10-21

| ID | Check | Oracolo AS-IS | Comando / metodo TO-BE | Atteso | Esito | Evidenza | Note |
|----|-------|---------------|------------------------|--------|-------|----------|------|
| G1 | van_v2.yaml ↔ best_trial_params | van/v2 JSON | field compare | match | **PASS** | [EVIDENCE/wave_g_hparams.json](EVIDENCE/wave_g_hparams.json) | 23/23 fields match (same as wave_b) |
| G2 | thresholds ↔ best_threshold_params | van/v2 JSON | field compare | match | **PASS** | [EVIDENCE/wave_g_hparams.json](EVIDENCE/wave_g_hparams.json) | conf/margin/var/temp 4/4 match |
| G3 | Smoke metrics.json schema | train use-case | read runs/*/metrics.json | required keys | **PASS** | [EVIDENCE/wave_g_metrics_keys.json](EVIDENCE/wave_g_metrics_keys.json) | smoke-m2 has run_id, best_val_f1_macro, history, n_train |
| G4 | Optuna HPO CLI | Thesis Optuna | — | SKIP / GAP | **SKIP** | docs/TRAIN.md | accepted GAP-G4 |
