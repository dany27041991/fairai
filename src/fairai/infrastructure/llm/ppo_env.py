"""Gymnasium env: pick MC option index (Thesis StereosetEnv semantics)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from fairai.infrastructure.llm.ppo_reward import compute_reward
from fairai.infrastructure.llm.prompts import gold_label, has_neutral_label_predicted


RewardMode = Literal["correct", "thesis_relative"]


class StereoSetMCEnv:
    """Minimal gymnasium-compatible env (no heavy SB3 / CausalLM for smoke).

    - ``reward_mode="correct"``: +1 if chosen option has ``correct=True``
    - ``reward_mode="thesis_relative"``: Thesis ``compute_reward`` on
      ``label_predicted`` scores {0:−2, 1:−1, 2:+2}
    - ``require_neutral_pred=True``: Thesis filter (≥1 option with
      ``label_predicted == 2``)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        examples: list[dict[str, Any]],
        *,
        seed: int = 42,
        reward_mode: RewardMode = "thesis_relative",
        require_neutral_pred: bool = True,
    ) -> None:
        pool = list(examples)
        if require_neutral_pred:
            pool = [ex for ex in pool if has_neutral_label_predicted(ex)]
        else:
            pool = [ex for ex in pool if gold_label(ex) is not None]
        if not pool:
            raise ValueError("No valid MC examples for PPO env")
        self.examples = pool
        self.prompts = pool  # Thesis StereosetEnv attribute name
        self.reward_mode = reward_mode
        self._i = 0
        self._rng = np.random.default_rng(seed)
        self.action_space_n = 3
        self.observation_space_shape = (3,)  # bag-of-label_predicted features

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._i = int(self._rng.integers(0, len(self.examples)))
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        ex = self.examples[self._i]
        vec = np.zeros(3, dtype=np.float32)
        for opt in ex.get("options") or []:
            raw = opt.get("label_predicted")
            if raw is None:
                raw = opt.get("label", 0)
            if raw is None:
                continue
            lab = int(raw)
            if 0 <= lab < 3:
                vec[lab] += 1.0
        return vec

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        ex = self.examples[self._i]
        options = list(ex.get("options") or [])
        reward = 0.0
        a = int(action)
        if 0 <= a < len(options):
            if self.reward_mode == "thesis_relative":
                reward = compute_reward(options[a].get("label_predicted"), options, a)
            elif options[a].get("correct") is True:
                reward = 1.0
        self._i = int(self._rng.integers(0, len(self.examples)))
        return self._obs(), float(reward), True, False, {}
