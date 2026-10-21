"""Gymnasium env: pick MC option index; reward=1 if correct."""

from __future__ import annotations

from typing import Any

import numpy as np

from fairai.infrastructure.llm.prompts import gold_label


class StereoSetMCEnv:
    """Minimal gymnasium-compatible env (no heavy SB3 required for smoke)."""

    metadata = {"render_modes": []}

    def __init__(self, examples: list[dict[str, Any]], *, seed: int = 42) -> None:
        self.examples = [ex for ex in examples if gold_label(ex) is not None]
        if not self.examples:
            raise ValueError("No valid MC examples for PPO env")
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
        options = ex.get("options") or []
        reward = 0.0
        if 0 <= int(action) < len(options) and options[int(action)].get("correct") is True:
            reward = 1.0
        # next episode
        self._i = int(self._rng.integers(0, len(self.examples)))
        return self._obs(), reward, True, False, {}
