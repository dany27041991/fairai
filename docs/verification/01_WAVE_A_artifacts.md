# Wave A — Artifacts / manifesto (gate)

**Status:** PASS  
**Agent:** local harness (`scripts/verification/run_wave_a.sh`, `compare_counts.py`)  
**Date:** 2026-10-21

| ID | Check | Oracolo AS-IS | Comando / metodo TO-BE | Atteso | Esito | Evidenza | Note |
|----|-------|---------------|------------------------|--------|-------|----------|------|
| A1 | Train leakage-safe line count | `09` / `11_addendum`: 50409 | `compare_counts.py` | 50409 | PASS | [EVIDENCE/wave_a_counts.json](EVIDENCE/wave_a_counts.json) | |
| A2 | van_final line count | catalog: 55817 | idem | 55817 | PASS | idem | |
| A3 | van_minimal line count | catalog: 24000 | idem | 24000 | PASS | idem | |
| A4 | Manifest verify | `manifest.sha256.json` | `fairai manifest --verify` | exit 0 all ok | PASS | [EVIDENCE/wave_a_full.txt](EVIDENCE/wave_a_full.txt) | |
| A5 | VAN v2 final_model.pt present | `09_preserve` | `test -f` | exists | PASS | wave_a_full.txt | |
| A6 | Gold scores present | `gold_scores.json` | `test -f` | exists | PASS | wave_a_full.txt | |
| A7 | StereoSet splits 4 files | DEVELOP/data | `ls` | 4 json | PASS | wave_a_full.txt | |
| A8 | CrowS CSV raw | DATASETS/CROWSPAIRS | `test -f` | exists | PASS | wave_a_full.txt | |
| A9 | select_options_pred | stereoset MC | `test -f` | exists | PASS | wave_a_full.txt | |
| A10 | asis_mapping covers checklist | `09_preserve` | review `asis_mapping.py` | required specs present | PASS | code review | v1/MC/CrowS optional flags OK |

**Gate:** OPEN — proceed to Wave B ∥ D.
