"""Tiny REINFORCE smoke on StereoSet MC env (gymnasium-style API)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from fairai.infrastructure.artifact_paths import ArtifactPaths
from fairai.infrastructure.llm.mc_dataset import load_mc_file
from fairai.infrastructure.llm.ppo_env import StereoSetMCEnv
from fairai.infrastructure.torch_van.checkpointing import write_json


class _Policy(nn.Module):
    def __init__(self, obs_dim: int = 3, n_actions: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self.net(obs))


@dataclass(slots=True)
class PpoSmokeResult:
    steps: int
    mean_reward: float
    metrics_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_ppo_smoke(
    *,
    dataset_path: Path,
    steps: int = 64,
    seed: int = 42,
    artifacts: ArtifactPaths | None = None,
    out_name: str = "llm_ppo_smoke.json",
) -> PpoSmokeResult:
    paths = artifacts or ArtifactPaths.from_env()
    splits = load_mc_file(dataset_path)
    examples = (splits.get("train") or []) + (splits.get("validation") or [])
    env = StereoSetMCEnv(
        examples,
        seed=seed,
        reward_mode="thesis_relative",
        require_neutral_pred=True,
    )

    # Prefer gymnasium registration check (optional dependency)
    try:
        import gymnasium as gym  # noqa: F401
        backend = "gymnasium"
    except ImportError:
        backend = "builtin_env_only"

    torch.manual_seed(seed)
    policy = _Policy()
    optim = torch.optim.Adam(policy.parameters(), lr=1e-2)
    rewards: list[float] = []

    obs, _ = env.reset(seed=seed)
    for _ in range(steps):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        dist = policy(obs_t)
        action = dist.sample()
        logp = dist.log_prob(action)
        obs, reward, terminated, truncated, _ = env.step(int(action.item()))
        rewards.append(float(reward))
        loss = -logp * float(reward)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if terminated or truncated:
            obs, _ = env.reset()

    mean_r = float(np.mean(rewards)) if rewards else 0.0
    out = paths.metrics_dir() / out_name
    write_json(
        out,
        {
            "backend": backend,
            "algorithm": "reinforce_smoke",
            "steps": steps,
            "mean_reward": mean_r,
            "dataset": str(dataset_path),
            "reward_mode": "thesis_relative",
            "note": (
                "Env reward/filter match Thesis StereosetEnv; policy is REINFORCE MLP smoke "
                "(not CausalLM CleanRL PPO — GAP-F5). See docs/LLM.md."
            ),
        },
    )
    return PpoSmokeResult(steps=steps, mean_reward=mean_r, metrics_path=str(out))
