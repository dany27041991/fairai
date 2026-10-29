# Wave F — LLM prompts / MC

**Status:** PASS  
**Agent:** llm  
**Verified:** 2026-10-21

| ID | Check | Oracolo AS-IS | Comando / metodo TO-BE | Atteso | Esito | Evidenza | Note |
|----|-------|---------------|------------------------|--------|-------|----------|------|
| F1 | Prompt builders match Thesis | Thesis few-shot cell | compare `prompts.py` | semantic match | **PASS** | docs/LLM.md + code review | Context/Options/Label format; few-shot teaches Label 2 neutral; test prompt ends "The best answer is: Label " |
| F2 | parse_prediction | Thesis | unit test | Label 0/1/2 | **PASS** | [EVIDENCE/wave_f_prompts.txt](EVIDENCE/wave_f_prompts.txt) | 2 passed |
| F3 | MC split counts | select_options_pred | load json | 1484/317/322 | **PASS** | [EVIDENCE/wave_f_mc_counts.json](EVIDENCE/wave_f_mc_counts.json) | exact match |
| F4 | few-shot smoke metrics exist | — | metrics json or re-run | file present | **PASS** | `artifacts/results/metrics/llm_fewshot_van.json` | n=16, accuracy=0.5625, backend=van_neutral_judge |
| F5 | Qwen PPO parity | Thesis PPO | — | SKIP | **SKIP** | docs/LLM.md | accepted gap |
