.PHONY: install install-torch import-asis manifest-verify test check-lines smoke \
	compose-smoke compose-eval compose-eval-gpu lint typecheck

ASIS_ROOT ?= ../TESI MAGISTRALE
ARTIFACTS ?= artifacts
FAIRAI_DEVICE ?= cpu

install:
	pip install -e ".[dev]"

install-torch:
	pip install -e ".[torch,dev]"

import-asis:
	fairai import-asis --source "$(ASIS_ROOT)" --dest "$(ARTIFACTS)"

manifest-verify:
	fairai manifest --artifacts "$(ARTIFACTS)" --verify

test:
	pytest -q

check-lines:
	fairai check-file-length --root src --limit 500

smoke:
	fairai smoke --device "$(FAIRAI_DEVICE)"

lint:
	ruff check src tests

typecheck:
	mypy

compose-smoke:
	docker compose --profile ci run --rm smoke

compose-eval:
	FAIRAI_DEVICE="$(FAIRAI_DEVICE)" docker compose --profile eval run --rm eval

compose-eval-gpu:
	docker compose --profile eval-gpu run --rm eval-gpu
