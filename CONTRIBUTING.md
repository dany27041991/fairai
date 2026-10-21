# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[torch,dev]"
make smoke
```

## Rules

- Max **500** lines per `src/**/*.py` (`make check-lines`)
- `domain/` must not import `torch` / `transformers`
- Hyperparameters and thresholds only via `configs/**/*.yaml`
- Paths via CLI flags or `FAIRAI_*` env — no machine-absolute paths in code
- Prefer small modules over god-files; see `docs/` → TO-BE coding standards

## Checks before PR

```bash
ruff check src tests
make smoke
```

## Artifacts

Large datasets and `.pt` checkpoints are **not** in git. Import from the thesis tree (`make import-asis`) or download a GitHub Release (M8).
