"""Non-learning reference policies evaluated on exactly the same episodes as PPO.

Every policy sees only what the PPO agent may see: the action mask and the
(possibly noisy) preview of each action from ``env.preview_estimates()``,
plus — for semantic policies — the *believed* requirements.  None of them reads the true
requirements.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from task_offloading.domain import FailureReason, Task
from task_offloading.env import TaskOffloadingEnv

Policy = Callable[[TaskOffloadingEnv], int]


def fixed_target(label: str) -> Policy:
    """Always pick ``label`` if the mask allows it, otherwise the first valid."""

    def policy(env: TaskOffloadingEnv) -> int:
        mask = env.action_masks()
        labels = [action.label for action in env.scenario.actions]
        index = labels.index(label)
        if mask[index]:
            return index
        return int(np.flatnonzero(mask)[0])

    return policy


def random_valid(seed: int) -> Policy:
    rng = np.random.default_rng(seed)

    def policy(env: TaskOffloadingEnv) -> int:
        return int(rng.choice(np.flatnonzero(env.action_masks())))

    return policy


def greedy(*, semantic: bool, energy_weight: float = 0.0) -> Policy:
    """Lowest expected latency; semantic version prefers believed-feasible actions.

    Semantic greedy ranks masked actions by (number of believed deadline /
    reliability / accuracy requirements the preview would violate, latency).
    The non-semantic version ranks by latency only.
    """

    def policy(env: TaskOffloadingEnv) -> int:
        task: Task = env.current_task()
        mask = env.action_masks()
        previews = env.preview_estimates()
        believed = task.policy_requirements
        best_index = -1
        best_key: tuple[float, float] | None = None
        for index, outcome in enumerate(previews):
            if not mask[index]:
                continue
            if outcome.failure_reason is not FailureReason.NONE:
                key = (1e9, 1e9)
            else:
                misses = 0.0
                if semantic:
                    if (
                        believed.max_latency_s is not None
                        and outcome.total_latency_s > believed.max_latency_s
                    ):
                        misses += 1
                    if (
                        believed.min_success_probability is not None
                        and outcome.achieved_success_probability
                        < believed.min_success_probability
                    ):
                        misses += 1
                    if (
                        believed.min_accuracy is not None
                        and outcome.achieved_accuracy < believed.min_accuracy
                    ):
                        misses += 1
                key = (
                    misses,
                    outcome.total_latency_s + energy_weight * outcome.device_energy_j,
                )
            if best_key is None or key < best_key:
                best_key, best_index = key, index
        return best_index if best_index >= 0 else int(np.flatnonzero(mask)[0])

    return policy


__all__ = ["Policy", "fixed_target", "greedy", "random_valid"]
