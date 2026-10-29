# FAIRAI / VAN

Open research software for **stereotype detection** in LLM-based services: a unified three-label corpus, a dual-axis evaluation protocol (macro-F1 + adapted StereoSet iCAT), and a compact released classifier (**VAN**) with Optuna-tuned hyperparameters and integration patterns for city / public-sector pipelines.

**Author:** Danilo Giovannico — University of Salento (Università del Salento), Lecce, Italy  
**Paper draft:** [`paper/ieee_software_stereotypes.tex`](paper/ieee_software_stereotypes.tex) (IEEE Software theme article)  
**Repositories:** [softengunisalento/van_bias_aware](https://github.com/softengunisalento/van_bias_aware) · [dany27041991/fairai](https://github.com/dany27041991/fairai)

> Training and evaluation text may contain stereotypical language used only for bias research. See [NOTICE.md](NOTICE.md).

---

## What you get

| Deliverable | Location |
|-------------|----------|
| Released VAN v2 hyperparameters | [`configs/models/van_v2.yaml`](configs/models/van_v2.yaml) |
| Optuna best configs (CE / −BPL / VAN fair + van-policy) | [`configs/experiments/optuna_*.yaml`](configs/experiments/) · [`artifacts/optuna/`](artifacts/optuna/) |
| Table A/B metric JSON | [`artifacts/results/metrics/`](artifacts/results/metrics/) |
| Provenance packs | [`configs/provenance/`](configs/provenance/) |
| Train / eval / Optuna / suite CLI | `fairai` (Typer) |
| Full documentation | [`docs/README.md`](docs/README.md) |

**Not shipped in git (large):** raw/processed datasets under `artifacts/datasets/`, model weights `*.pt`. Import or regenerate locally (see below).

---

## Quick start

```bash
git clone https://github.com/softengunisalento/van_bias_aware.git
cd van_bias_aware   # or fairai checkout
python -m venv .venv && source .venv/bin/activate
pip install -e ".[torch,dev]"

# Optional: import thesis AS-IS artifacts (datasets + VAN weights)
make import-asis ASIS_ROOT="/path/to/TESI MAGISTRALE"

# Unit tests (no GPU)
pytest -q

# Adapted iCAT on StereoSet (needs imported splits + checkpoint)
fairai icat --subset test --baseline configs/baselines/baseline_van_v2_icat.json
```

GPU Optuna / suite (Docker):

```bash
make optuna-vanpolicy-ce      # full-data Optuna, CE family
make optuna-vanpolicy-nobpl  # full-data Optuna, no bias projection
# Product compare suite after finals:
# FAIRAI_SUITE=configs/experiments/suite_optuna_product.yaml \
#   docker compose --profile experiment run --rm experiment-gpu
```

---

## Evaluation protocol (paper)

| Axis | Resource | Notes |
|------|----------|--------|
| **F1** | `artifacts/datasets/processed/stereoset/f1_test_leakfree.jsonl` | Neutral-substituted select-options; 0 exact overlap vs train/val |
| **iCAT** | Official StereoSet **`subset=test`** | Adapted 3-class preference; report inter / intra |

Released VAN product numbers (Table A): leak-free F1 **0.962**, iCAT test **96.28 / 96.71**.  
Table B SoT: [`artifacts/results/metrics/optuna_product_compare_agg.json`](artifacts/results/metrics/optuna_product_compare_agg.json).

Reproduce steps: [`docs/REPRODUCE_PAPER.md`](docs/REPRODUCE_PAPER.md).

---

## Documentation map

| Doc | Contents |
|-----|----------|
| [docs/README.md](docs/README.md) | Full index |
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | Pipeline, labels, Clean Architecture |
| [docs/DATASETS.md](docs/DATASETS.md) | Corpus, leakage, splits |
| [docs/MODEL_VAN.md](docs/MODEL_VAN.md) | Architecture & frozen hparams |
| [docs/TRAIN.md](docs/TRAIN.md) | Training + Optuna HPO |
| [docs/EVAL_ICAT.md](docs/EVAL_ICAT.md) | Adapted iCAT |
| [docs/INFERENCE.md](docs/INFERENCE.md) | Predict / thresholds / abstention |
| [docs/GENERATE.md](docs/GENERATE.md) | Dataset factory |
| [docs/LLM.md](docs/LLM.md) | Few-shot / PPO patterns |
| [docs/ARTIFACTS_AND_RESULTS.md](docs/ARTIFACTS_AND_RESULTS.md) | Artifact layout |
| [docs/REPRODUCE_PAPER.md](docs/REPRODUCE_PAPER.md) | IEEE Software tables |
| [docs/verification/00_INDEX.md](docs/verification/00_INDEX.md) | Anti-regression waves |

---

## Project layout

```
configs/          # models, experiments, provenance
src/fairai/       # domain / application / infrastructure / cli
paper/            # IEEE Software LaTeX draft
artifacts/        # metrics + Optuna studies (datasets/weights local)
docs/             # documentation
tests/            # unit tests
docker/           # GPU Optuna / suite entrypoints
```

Coding rules: max **500** lines per source file under `src/`; hyperparameters live in YAML, not hard-coded loops; domain layer stays free of PyTorch.

---

## Citation

If you use this software or the corpus/protocol in academic work, cite Danilo Giovannico (University of Salento) and the accompanying IEEE Software manuscript when published. Prefer linking the frozen release tag used for your experiments.

---

## License and third-party data

**Copyright (c) 2025–2026 Danilo Giovannico, University of Salento. All rights reserved.**  
See [LICENSE](LICENSE). This is **not** an MIT (or other OSI) license.

Upstream datasets and the Sentence-BERT backbone remain under their own terms — [NOTICE.md](NOTICE.md).
