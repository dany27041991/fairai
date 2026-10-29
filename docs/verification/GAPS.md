# Gaps — accepted / follow-up / blockers

| ID | Severity | Wave | Description | Disposition |
|----|----------|------|-------------|-------------|
| GAP-D4 | closed | D | 17/50409 residual StereoSet-test overlaps (0.034%) in imported `van_train_leakage_safe.jsonl`. | Tool: `fairai analyze leakage`; strict export `artifacts/datasets/generated/van_train_leakage_strict.jsonl` (50392). Canonical import **unchanged**. |
| GAP-E4 | follow-up | E | Local multi-source + optional CoLA wired; Ollama / MiniLM semantic rewrite still notebook-only | [docs/GENERATE.md](../GENERATE.md) |
| GAP-F5 | accepted | F | No Qwen/DeepSeek PPO parity (checkpoints absent in TESI MAGISTRALE) | [docs/LLM.md](../LLM.md) |
| GAP-G4 | closed | G | Optuna HPO wired: `fairai experiment optuna` + `make optuna-fair-*` (equal-budget fair compare) | [docs/TRAIN.md](../TRAIN.md) |

**Blockers:** none.

Severity: `blocker` | `accepted` | `follow-up`
