import torch

from fairai.infrastructure.llm.ppo_actions import (
    select_action_epsilon,
    select_action_temperature,
)
from fairai.infrastructure.llm.ppo_env import StereoSetMCEnv
from fairai.infrastructure.llm.ppo_reward import (
    accuracy_choose_label2,
    build_mc_prompt,
    build_prompt,
    compute_reward,
    has_neutral_label_predicted,
)


def test_compute_reward_relative() -> None:
    options = [
        {"label_predicted": 0},
        {"label_predicted": 1},
        {"label_predicted": 2},
    ]
    # Choose neutral (2 → +2); others mean = (−2 + −1)/2 = −1.5 → reward = 2 − (−1.5) = 3.5
    assert abs(compute_reward(2, options, 2) - 3.5) < 1e-9
    # Choose stereo (0 → −2); others mean = (−1 + 2)/2 = 0.5 → reward = −2.5
    assert abs(compute_reward(0, options, 0) - (-2.5)) < 1e-9


def test_build_prompt_and_filter() -> None:
    text = build_mc_prompt("ctx", [{"sentence": "a"}, {"sentence": "b"}, {"sentence": "c"}])
    assert text == build_prompt("ctx", [{"sentence": "a"}, {"sentence": "b"}, {"sentence": "c"}])
    assert "A) a" in text and "B) b" in text and "C) c" in text
    assert has_neutral_label_predicted(
        {"options": [{"label_predicted": 0}, {"label_predicted": 2}]}
    )
    assert has_neutral_label_predicted(
        {"options": [{"label_predicted": "2"}]}
    )
    assert not has_neutral_label_predicted(
        {"options": [{"label_predicted": 0}, {"label_predicted": 1}]}
    )
    logits = torch.tensor([[0.0, 0.0, 5.0]])
    action, probs = select_action_temperature(logits, temperature=1.0)
    assert action.numel() == 1 and probs.shape[-1] == 3
    action_e, _ = select_action_epsilon(logits, temperature=1.0, epsilon=0.0)
    assert int(action_e.item()) in (0, 1, 2)


def test_env_thesis_reward() -> None:
    examples = [
        {
            "context": "c",
            "options": [
                {"sentence": "a", "label": 0, "label_predicted": 0, "correct": False},
                {"sentence": "b", "label": 1, "label_predicted": 1, "correct": False},
                {"sentence": "c", "label": 2, "label_predicted": 2, "correct": True},
            ],
        }
    ]
    env = StereoSetMCEnv(examples, seed=0, reward_mode="thesis_relative")
    assert len(env.prompts) == 1
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3,)
    _, reward, term, trunc, _ = env.step(2)
    assert term and not trunc
    assert abs(reward - 3.5) < 1e-9
    assert accuracy_choose_label2(examples, [2]) == 1.0
    assert accuracy_choose_label2(examples, [0]) == 0.0
