# FAIRAI-Thesis.ipynb → fairai parity map

Multi-agent audit (2026-10-22). SoT notebook: `TESI MAGISTRALE/DEVELOP/FAIRAI-Thesis.ipynb` (51 cells).

## Cell coverage

| Cells | Thesis content | fairai | Text | Images | Code |
|------:|----------------|--------|:----:|:------:|:----:|
| 0 | `# VAN CLASSIFIER` | [TRAIN.md](TRAIN.md) / [MODEL_VAN.md](MODEL_VAN.md) | ✓ | — | — |
| 1 | VAN train, losses, Optuna, CM | `fairai train`, `eval --plot-dir` | ✓ | ✓ CM | ✓ (Optuna → frozen YAML) |
| 2–3 | ICAT + gold + by bias | `fairai icat` | ✓¹ | — | ✓ |
| 4–5 | Threshold Optuna + MC dropout + label MC | `thresholds`, `infer --mc-samples`, `analyze label-mc`, frozen YAML | ✓ | — | ✓ (search frozen) |
| 6–7 | PROMPT TEST VAN | `fairai infer --gates [--mc-samples 10]` | ✓ | — | ✓ |
| 8 | Separator | — | — | — | SKIP |
| 9–10 | MC merge → `prompts` | `fairai analyze merge-mc` | ✓ | — | ✓ |
| 11–44 | 5× LLM PPO / RL-eval / few-shot | `fairai llm *` smoke + `analyze gen-metrics` / `metric-bars` / `confusion` | ✓ | ✓ from JSON | smoke² |
| 45–46 | ANALISYS paired t-test + bars | `analyze rl-compare` | ✓ | ✓ | ✓ |
| 47 | MC accuracy / CM / error-by-bias | `analyze mc` | ✓ | ✓ | ✓ |
| 48 | Dataset dist (6 plots) | `analyze dataset` | ✓ | ✓ | ✓ |
| 49 | Lexical / POS / VADER / TF-IDF / cosine / t-SNE | `analyze text` | ✓ | ✓ | ✓ |
| 50 | Empty | — | — | — | SKIP |

¹ ICAT printed Thesis numbers (historical AS-IS run):

| Split | LM | SS | iCAT |
|-------|-----:|-----:|-----:|
| Intersentence | 98.18 | 49.98 | 98.13 |
| Intrasentence | 97.23 | 49.91 | 97.05 |

Reproduce with `fairai icat`; baseline JSON may differ slightly by checkpoint/concat.

² Full 5-model CleanRL PPO + Qwen weights = **GAP-F5** (intentional smoke). Charts reproduced via `analyze` from CSV/JSON.

## Intentional non-ports (documented, not omissions)

| Item | Reason |
|------|--------|
| Live Optuna architecture (250) / threshold (1000) | Winner frozen in `van_v2.yaml` + threshold params (GAP-G4) |
| Early-stop on WHM(F1,iCAT) during retrain | Helper `compute_combined_score` ported; retrain loop uses val F1 |
| OneCycleLR scheduler | Winner uses cosine; linear+cosine wired |
| 5× duplicate LLM blocks | One smoke template + model-id docs |
| Interactive stdin REPL | CLI `infer` replaces REPL |
| WordNet train-time EDA augment | Optional; generation has synonym path |
| GRL / discriminator / `LAMBDA_ADV` | Unused in Thesis cell-1 forward |
| Cell 1 writes `models/van/v1` | Legacy path; SoT checkpoint is v2 |

## Commands cheat-sheet

```bash
pip install -e ".[viz,nlp,torch,dev]"
fairai train -c configs/experiments/smoke_cpu.yaml
fairai icat --subset all
fairai infer -t "..." --gates --mc-samples 10
fairai eval -d … --plot-dir artifacts/analysis/eval_cm
fairai analyze merge-mc -i …_pred.json -o …/stereoset_select_options.json
fairai analyze dataset -d … -o artifacts/analysis/dataset
fairai analyze text -d … -o artifacts/analysis/text
fairai analyze rl-compare -m configs/analysis/rl_compare_example.json
fairai llm few-shot --max-examples 64
fairai llm ppo-smoke --steps 64
```

## Engineering hygiene (post-parity)

Shared report/CLI JSON helpers, MC JSON parser, `VanBundle` / `load_van_and_tokenizer`, unified `has_neutral_label_predicted`, prompt option formatting. Public CLI + Thesis formulas unchanged.

## Regression gates

```bash
make test && make check-lines && make analyze-smoke
```
