# Wave D — Dataset schema / labels / leakage

**Status:** PASS (with GAP-D4)  
**Agent:** schema (Wave D harness, inline Python)  
**Date:** 2026-10-21

| ID | Check | Oracolo AS-IS | Comando / metodo TO-BE | Atteso | Esito | Evidenza | Note |
|----|-------|---------------|------------------------|--------|-------|----------|------|
| D1 | Schema sentence+label | catalog §schema | sample validate jsonl (first 500) | valid | PASS | [EVIDENCE/wave_d_schema.json](EVIDENCE/wave_d_schema.json) | 500/500 valid; `invalid_count=0` |
| D2 | Labels ∈ {0,1,2} | Label enum | scan train sample | no invalid | PASS | idem | all sampled labels in {0,1,2} |
| D3 | Label distribution non-collapsed | Thesis train | streaming count full train | all 3 present | PASS | [EVIDENCE/wave_d_labels.json](EVIDENCE/wave_d_labels.json) | 0→16623, 1→16466, 2→17320 (50409 total) |
| D4 | Leakage filter train vs SS test | addendum 5408 removed | `stereoset_forbidden_sentences` + normalized membership scan | 0 overlap (normalized) | **FAIL** | [EVIDENCE/wave_d_leakage.json](EVIDENCE/wave_d_leakage.json) | **17** train sentences still in forbidden set (7624 test-derived keys); 0.034% residual — e.g. line 3230 "A male is considered a dominant person." |
| D5 | Flatten n inter/intra | baseline n | `flatten_split_file` on train+test ALL | 6369 / 6318 | PASS | [EVIDENCE/wave_d_flatten.json](EVIDENCE/wave_d_flatten.json) | inter `concat=True`, intra `concat=False` |

**Disposition:** D1–D3, D5 PASS. D4 residual **17/50409 (0.034%)** is an AS-IS import property (canonical `van_train_leakage_safe` unchanged by rewrite), not a TO-BE code regression. Tracked as **GAP-D4** (`follow-up`) in [GAPS.md](GAPS.md). Wave D does not block A–C.
