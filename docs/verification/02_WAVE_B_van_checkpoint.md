# Wave B — VAN architecture / checkpoint

**Status:** PASS  
**Agent:** van_arch  
**Verified:** 2026-10-21

| ID | Check | Oracolo AS-IS | Comando / metodo TO-BE | Atteso | Esito | Evidenza | Note |
|----|-------|---------------|------------------------|--------|-------|----------|------|
| B1 | load_state_dict strict | Thesis SingleSentenceModel | `python -m pytest tests/unit/test_van_load.py -q` | PASS | PASS | [wave_b_b1_pytest.txt](EVIDENCE/wave_b_b1_pytest.txt) | 1 passed in 6.34s; strict load + forward smoke |
| B2 | Arch YAML matches trial params | `best_trial_params.json` | compare to `van_v2.yaml` | match key hparams | PASS | [wave_b_hparams.json](EVIDENCE/wave_b_hparams.json) | 23/23 fields match (gen_num_layers, gen_hidden_size, bias_dim, alpha, gamma, learning_rate, …) |
| B3 | Forward shape (B,3) | Thesis forward | `test_van_load` forward assert | logits (2,3) | PASS | [wave_b_b1_pytest.txt](EVIDENCE/wave_b_b1_pytest.txt) | `assert logits.shape == (2, 3)` in test_van_load |
| B4 | Infer non-degenerate | Thesis classify | `fairai infer` 3 texts (FAIRAI_* unset) | ≥2 distinct labels or varied probs | PASS | [wave_b_infer.json.txt](EVIDENCE/wave_b_infer.json.txt) | labels {1,1,2}; probs vary across texts |
| B5 | fairai smoke | CI | `fairai smoke --device cpu` | smoke OK | PASS | [wave_b_smoke.txt](EVIDENCE/wave_b_smoke.txt) | 23 passed; `smoke OK` |

## Summary

All B1–B5 checks passed. VAN v2 checkpoint loads strictly, hyperparameters align with `van_v2.yaml`, forward shapes are correct, inference is non-degenerate, and smoke succeeds on CPU.
