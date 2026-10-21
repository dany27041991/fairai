# LLM alignment (M6)

Optional phase after VAN-only MVP. Full thesis PPO on Qwen/DeepSeek needs multi-GB VRAM and checkpoints that are **not** in `TESI MAGISTRALE` (see sibling `saved_van_llm` if available).

## Few-shot StereoSet MC

Dataset: `artifacts/datasets/processed/stereoset/stereoset_select_options_pred.json`

```bash
fairai llm few-shot --max-examples 64 --n-shots 3 --device cpu
```

**Backend (smoke):** frozen **VAN v2** picks the option with highest P(neutral). Prompt builders match AS-IS Thesis (`build_*_prompt`). Metrics → `artifacts/results/metrics/llm_fewshot_van.json`.

To run a generative HF causal LM later, plug `AutoModelForCausalLM` behind the same prompt/parse helpers in `infrastructure/llm/prompts.py` (not required for M6 acceptance).

## PPO smoke

```bash
pip install 'fairai[llm]'   # gymnasium
fairai llm ppo-smoke --steps 64
```

Uses a tiny REINFORCE policy on a gymnasium-style MC env (action = option index). **Not** historical PPO parity with Qwen2.5-1.5B.

## Full Qwen PPO (manual)

1. GPU host + `Qwen/Qwen2.5-1.5B` (or thesis `saved_models/Qwen2.5-1.5B`)
2. Install `gymnasium`, `stable-baselines3`, `transformers`
3. Port Thesis PPO cells; keep env API compatible with `StereoSetMCEnv`
