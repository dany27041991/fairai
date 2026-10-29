# Wave C — ICAT parity

**Status:** PASS  
**Agent:** icat (evidence recovered after interrupt)  
**Verified:** 2026-10-21

| ID | Check | Oracolo AS-IS | Comando / metodo TO-BE | Atteso | Esito | Evidenza | Note |
|----|-------|---------------|------------------------|--------|-------|----------|------|
| C1 | Concat rule inter/intra | Thesis ICAT | code review `evaluate_icat.py` | inter concat, intra no | **PASS** | [EVIDENCE/wave_c_concat.txt](EVIDENCE/wave_c_concat.txt) | inter=True, intra=False |
| C2 | ICAT formula | `compute_icat_score` | `pytest tests/unit/test_icat_formula.py` | PASS | **PASS** | [EVIDENCE/wave_c_formula.txt](EVIDENCE/wave_c_formula.txt) | 3 passed |
| C3 | Parity subset=test | prior `icat_test.json` | `fairai icat --subset test --baseline …` | OK within tol | **PASS** | [EVIDENCE/wave_c_test_parity.txt](EVIDENCE/wave_c_test_parity.txt) | avg_icat=96.4941 exact; n=3816 |
| C4 | Parity subset=all | `configs/baselines/baseline_van_v2_icat.json` | `fairai icat --subset all --baseline …` | OK within tol | **PASS** | [EVIDENCE/wave_c_all_parity.txt](EVIDENCE/wave_c_all_parity.txt) | avg_icat=96.0973 exact match baseline |
| C5 | n_total all | baseline n=12687 | metrics json | 12687 | **PASS** | wave_c_all_parity.txt | inter 6369 + intra 6318 |

## Summary

Blocker wave C closed: AS-IS concat rule preserved; ICAT/LM parity vs committed baseline within (and at) CLI tolerances.
