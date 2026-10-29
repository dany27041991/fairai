.PHONY: install install-torch install-viz import-asis manifest-verify test check-lines smoke \
	compose-smoke compose-eval compose-eval-gpu compose-train compose-gen lint typecheck \
	train-smoke generate-smoke analyze-smoke experiment-smoke compose-experiment-gpu \
	compose-experiment-rigorous compose-experiment-rigorous-clean \
	optuna-smoke optuna-fair-ce optuna-fair-nobpl optuna-fair-van optuna-fair-finalize \
	optuna-vanpolicy-ce optuna-vanpolicy-nobpl

ASIS_ROOT ?= ../TESI MAGISTRALE
ARTIFACTS ?= artifacts
FAIRAI_DEVICE ?= cpu

install:
	pip install -e ".[dev]"

install-torch:
	pip install -e ".[torch,dev]"

install-viz:
	pip install -e ".[viz,dev]"

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

compose-train:
	FAIRAI_DEVICE="$(FAIRAI_DEVICE)" docker compose --profile train run --rm trainer

train-smoke:
	fairai train -c configs/experiments/smoke_cpu.yaml --device "$(FAIRAI_DEVICE)" --max-samples 256

experiment-smoke:
	fairai experiment suite -c configs/experiments/suite_smoke.yaml --device "$(FAIRAI_DEVICE)"

compose-experiment-gpu:
	docker compose --profile experiment run --rm experiment-gpu

compose-experiment-rigorous:
	FAIRAI_SUITE=configs/experiments/suite_rigorous.yaml \
		docker compose --profile experiment run --rm \
		-e FAIRAI_SUITE=configs/experiments/suite_rigorous.yaml experiment-gpu

# Clean WHM selection (icat_subset=train); new rigc-* run_ids; does not overwrite rig-*
compose-experiment-rigorous-clean:
	FAIRAI_SUITE=configs/experiments/suite_rigorous_clean.yaml \
		docker compose --profile experiment run --rm \
		-e FAIRAI_SUITE=configs/experiments/suite_rigorous_clean.yaml experiment-gpu

# Equal-budget Optuna fair compare on GPU image (host torch may be CPU-only).
# Overrides experiment-gpu command → docker/run_optuna_hpo.py (entrypoint stays python3 -u).
optuna-smoke:
	FAIRAI_OPTUNA_FAMILY=ce FAIRAI_OPTUNA_N_TRIALS=2 FAIRAI_OPTUNA_TRIAL_EPOCHS=2 \
	FAIRAI_OPTUNA_TRIAL_PATIENCE=1 FAIRAI_OPTUNA_MAX_SAMPLES=256 \
	FAIRAI_OPTUNA_STUDY=optuna_fair_ce_smoke \
		docker compose --profile experiment run --rm \
		-e FAIRAI_OPTUNA_FAMILY -e FAIRAI_OPTUNA_N_TRIALS -e FAIRAI_OPTUNA_TRIAL_EPOCHS \
		-e FAIRAI_OPTUNA_TRIAL_PATIENCE -e FAIRAI_OPTUNA_MAX_SAMPLES -e FAIRAI_OPTUNA_STUDY \
		experiment-gpu /app/docker/run_optuna_hpo.py
	FAIRAI_OPTUNA_FAMILY=no_bpl FAIRAI_OPTUNA_N_TRIALS=2 FAIRAI_OPTUNA_TRIAL_EPOCHS=2 \
	FAIRAI_OPTUNA_TRIAL_PATIENCE=1 FAIRAI_OPTUNA_MAX_SAMPLES=256 \
	FAIRAI_OPTUNA_STUDY=optuna_fair_no_bpl_smoke \
		docker compose --profile experiment run --rm \
		-e FAIRAI_OPTUNA_FAMILY -e FAIRAI_OPTUNA_N_TRIALS -e FAIRAI_OPTUNA_TRIAL_EPOCHS \
		-e FAIRAI_OPTUNA_TRIAL_PATIENCE -e FAIRAI_OPTUNA_MAX_SAMPLES -e FAIRAI_OPTUNA_STUDY \
		experiment-gpu /app/docker/run_optuna_hpo.py
	FAIRAI_OPTUNA_FAMILY=van FAIRAI_OPTUNA_N_TRIALS=2 FAIRAI_OPTUNA_TRIAL_EPOCHS=2 \
	FAIRAI_OPTUNA_TRIAL_PATIENCE=1 FAIRAI_OPTUNA_MAX_SAMPLES=256 \
	FAIRAI_OPTUNA_STUDY=optuna_fair_van_smoke \
		docker compose --profile experiment run --rm \
		-e FAIRAI_OPTUNA_FAMILY -e FAIRAI_OPTUNA_N_TRIALS -e FAIRAI_OPTUNA_TRIAL_EPOCHS \
		-e FAIRAI_OPTUNA_TRIAL_PATIENCE -e FAIRAI_OPTUNA_MAX_SAMPLES -e FAIRAI_OPTUNA_STUDY \
		experiment-gpu /app/docker/run_optuna_hpo.py

# HPO search: 100 trials × 10 epochs on 4096-train subsample (equal budget).
# Final 3-seed suite uses best_config on the full van_split (no max_samples).
optuna-fair-ce:
	FAIRAI_OPTUNA_FAMILY=ce FAIRAI_OPTUNA_N_TRIALS=100 FAIRAI_OPTUNA_TRIAL_EPOCHS=10 \
	FAIRAI_OPTUNA_TRIAL_PATIENCE=3 FAIRAI_OPTUNA_MAX_SAMPLES=4096 \
	FAIRAI_OPTUNA_STUDY=optuna_fair_ce \
		docker compose --profile experiment run --rm \
		-e FAIRAI_OPTUNA_FAMILY -e FAIRAI_OPTUNA_N_TRIALS -e FAIRAI_OPTUNA_TRIAL_EPOCHS \
		-e FAIRAI_OPTUNA_TRIAL_PATIENCE -e FAIRAI_OPTUNA_MAX_SAMPLES -e FAIRAI_OPTUNA_STUDY \
		experiment-gpu /app/docker/run_optuna_hpo.py

optuna-fair-nobpl:
	FAIRAI_OPTUNA_FAMILY=no_bpl FAIRAI_OPTUNA_N_TRIALS=100 FAIRAI_OPTUNA_TRIAL_EPOCHS=10 \
	FAIRAI_OPTUNA_TRIAL_PATIENCE=3 FAIRAI_OPTUNA_MAX_SAMPLES=4096 \
	FAIRAI_OPTUNA_STUDY=optuna_fair_no_bpl \
		docker compose --profile experiment run --rm \
		-e FAIRAI_OPTUNA_FAMILY -e FAIRAI_OPTUNA_N_TRIALS -e FAIRAI_OPTUNA_TRIAL_EPOCHS \
		-e FAIRAI_OPTUNA_TRIAL_PATIENCE -e FAIRAI_OPTUNA_MAX_SAMPLES -e FAIRAI_OPTUNA_STUDY \
		experiment-gpu /app/docker/run_optuna_hpo.py

optuna-fair-van:
	FAIRAI_OPTUNA_FAMILY=van FAIRAI_OPTUNA_N_TRIALS=100 FAIRAI_OPTUNA_TRIAL_EPOCHS=10 \
	FAIRAI_OPTUNA_TRIAL_PATIENCE=3 FAIRAI_OPTUNA_MAX_SAMPLES=4096 \
	FAIRAI_OPTUNA_STUDY=optuna_fair_van \
		docker compose --profile experiment run --rm \
		-e FAIRAI_OPTUNA_FAMILY -e FAIRAI_OPTUNA_N_TRIALS -e FAIRAI_OPTUNA_TRIAL_EPOCHS \
		-e FAIRAI_OPTUNA_TRIAL_PATIENCE -e FAIRAI_OPTUNA_MAX_SAMPLES -e FAIRAI_OPTUNA_STUDY \
		experiment-gpu /app/docker/run_optuna_hpo.py

optuna-fair-finalize:
	docker compose --profile experiment run --rm --entrypoint python3 experiment-gpu \
		-u -c "from pathlib import Path; from fairai.application.run_optuna_hpo import write_finalize_suite; from fairai.infrastructure.artifact_paths import ArtifactPaths; import subprocess,sys; subprocess.check_call([sys.executable,'-m','pip','install','-q','-e','.','optuna>=3.6','typer','pyyaml']); p=write_finalize_suite(artifacts=ArtifactPaths.from_env(root=Path('/app/artifacts')), out_suite=Path('/app/configs/experiments/suite_optuna_fair.yaml')); print(p)"
	FAIRAI_SUITE=configs/experiments/suite_optuna_fair.yaml \
		docker compose --profile experiment run --rm \
		-e FAIRAI_SUITE=configs/experiments/suite_optuna_fair.yaml experiment-gpu

# VAN-policy Optuna for baselines: full leakage-safe split (from van_final),
# no max_samples cap; trial epochs/patience match VAN early-stop recipe.
# Does NOT re-search the released VAN.
optuna-vanpolicy-ce:
	FAIRAI_OPTUNA_FAMILY=ce FAIRAI_OPTUNA_N_TRIALS=100 FAIRAI_OPTUNA_TRIAL_EPOCHS=100 \
	FAIRAI_OPTUNA_TRIAL_PATIENCE=5 \
	FAIRAI_OPTUNA_STUDY=optuna_vanpolicy_ce \
	FAIRAI_OPTUNA_CONFIG=configs/experiments/optuna_vanpolicy_base.yaml \
		docker compose --profile experiment run --rm \
		-e FAIRAI_OPTUNA_FAMILY -e FAIRAI_OPTUNA_N_TRIALS -e FAIRAI_OPTUNA_TRIAL_EPOCHS \
		-e FAIRAI_OPTUNA_TRIAL_PATIENCE -e FAIRAI_OPTUNA_STUDY -e FAIRAI_OPTUNA_CONFIG \
		experiment-gpu /app/docker/run_optuna_hpo.py

optuna-vanpolicy-nobpl:
	FAIRAI_OPTUNA_FAMILY=no_bpl FAIRAI_OPTUNA_N_TRIALS=100 FAIRAI_OPTUNA_TRIAL_EPOCHS=100 \
	FAIRAI_OPTUNA_TRIAL_PATIENCE=5 \
	FAIRAI_OPTUNA_STUDY=optuna_vanpolicy_no_bpl \
	FAIRAI_OPTUNA_CONFIG=configs/experiments/optuna_vanpolicy_base.yaml \
		docker compose --profile experiment run --rm \
		-e FAIRAI_OPTUNA_FAMILY -e FAIRAI_OPTUNA_N_TRIALS -e FAIRAI_OPTUNA_TRIAL_EPOCHS \
		-e FAIRAI_OPTUNA_TRIAL_PATIENCE -e FAIRAI_OPTUNA_STUDY -e FAIRAI_OPTUNA_CONFIG \
		experiment-gpu /app/docker/run_optuna_hpo.py

generate-smoke:
	fairai generate -c configs/data/generate.yaml

llm-fewshot:
	fairai llm few-shot --max-examples 64 --device cpu

llm-ppo-smoke:
	fairai llm ppo-smoke --steps 64

analyze-smoke:
	fairai analyze rl-compare -m configs/analysis/rl_compare_example.json \
		-o artifacts/analysis/smoke/rl_compare
	fairai analyze gen-metrics -m configs/analysis/gen_metrics_example.json \
		-o artifacts/analysis/smoke/gen_metrics
	fairai analyze metric-bars -m configs/analysis/metric_bars_example.json \
		-o artifacts/analysis/smoke/metric_bars
	@python -c "import json,tempfile,pathlib; from fairai.application.analyze_dataset_dist import analyze_dataset_dist; from fairai.application.analyze_mc_predictions import analyze_mc_predictions; d=pathlib.Path(tempfile.mkdtemp()); p=d/'x.jsonl'; p.write_text('{\"sentence\":\"a b c\",\"label\":0,\"bias_type\":\"race\"}\\n{\"sentence\":\"d e f g\",\"label\":1,\"bias_type\":\"race\"}\\n'); analyze_dataset_dist(p,out_dir=d/'dist',write_plots=True); mc=d/'mc.json'; mc.write_text(json.dumps({'prompts':[{'item_bias':'race','options':[{'sentence':'a','label':0,'label_predicted':0}]}]})); analyze_mc_predictions(mc,out_dir=d/'mc',write_plots=True); print('analyze-smoke dataset/mc OK')"

compose-gen:
	docker compose --profile gen run --rm gen
