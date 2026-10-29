# Wave E — Generation

**Status:** PASS  
**Agent:** gen  
**Verified:** 2026-10-21

| ID | Check | Oracolo AS-IS | Comando / metodo TO-BE | Atteso | Esito | Evidenza | Note |
|----|-------|---------------|------------------------|--------|-------|----------|------|
| E1 | CrowS stereo→label mapping | GenerationDataset cell | unit / code | stereo more=0 less=1 | **PASS** | code review | `crows.py`: stereo → sent_more=STEREOTYPE(0), sent_less=ANTI_STEREOTYPE(1); antistereo swapped |
| E2 | generate smoke schema | — | `fairai generate` | n>0, schema OK | **PASS** | [EVIDENCE/wave_e_generate.json.txt](EVIDENCE/wave_e_generate.json.txt) | n_written=64, n_schema_issues=0 |
| E3 | Does not overwrite canonical processed | docs/GENERATE | mtime/hash van_train | unchanged | **PASS** | [EVIDENCE/wave_e_canonical.txt](EVIDENCE/wave_e_canonical.txt) | sha256+mtime identical before/after |
| E4 | Multi-source + CoLA/Ollama | notebook | `generate_multisource` + CoLA flag | multi JSONL | **PARTIAL** | docs/GENERATE.md | pools+CoLA yes; Ollama residual |
