# Training (M2)

## Smoke (CPU, ~minutes)

Uses a **tiny** head override in `configs/experiments/smoke_cpu.yaml` (inherits VAN v2 loss/optim, smaller MLP).

```bash
make train-smoke
# or
fairai train -c configs/experiments/smoke_cpu.yaml --device cpu --max-samples 256
```

Writes `artifacts/models/runs/<run_id>/{best_model.pt,final_model.pt,metrics.json}`.

## Full VAN v2 (optional, GPU)

```bash
fairai train -c configs/models/van_v2.yaml --device cuda
```

Expect: dataset `van_train_leakage_safe.jsonl` (50409), batch 64, grad_accum 8, up to 100 epochs with patience 5 on val F1. Needs NVIDIA + several GB VRAM; overnight-class runtime.

Optuna HPO is **not** wired yet (`--optuna` reserved); use frozen `configs/models/van_v2.yaml` hyperparameters from the thesis trial.

## Compose

```bash
docker compose --profile train run --rm trainer
```
