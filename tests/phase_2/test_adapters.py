"""Gymnasium contract and exact Gym/event replay parity tests."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from task_offloading.env import TaskOffloadingEnv
from task_offloading.sim import EventSimulationRunner, PhysicsConfig

from .helpers import two_step_scenario


def test_gymnasium_environment_passes_official_checker() -> None:
    check_env(TaskOffloadingEnv(two_step_scenario()), skip_render_check=True)


def test_gym_and_event_adapters_have_exact_transition_parity() -> None:
    scenario = two_step_scenario()
    config = PhysicsConfig(latency_reference_s=2.0, energy_reference_j=2.0)
    action_indices = (1, 2)
    replay = EventSimulationRunner(scenario, config).run(action_indices)
    env = TaskOffloadingEnv(scenario, config)
    observation, info = env.reset(seed=2026)
    assert env.observation_space.contains(observation)
    assert info["step_index"] == 0
    gym_results = []
    for step, action in enumerate(action_indices):
        observation, reward, terminated, truncated, info = env.step(action)
        assert not truncated
        assert reward == replay.transitions[step].reward
        assert env.last_transition is not None
        gym_results.append(env.last_transition)
    assert terminated
    assert info["step_index"] == 2
    assert env.observation_space.contains(observation)
    assert tuple(gym_results) == replay.transitions
    assert env.current_state == replay.final_state


def test_reset_is_reproducible_and_returns_action_mask() -> None:
    env = TaskOffloadingEnv(two_step_scenario())
    first_observation, first_info = env.reset(seed=11)
    env.step(1)
    second_observation, second_info = env.reset(seed=11)
    for key in first_observation:
        np.testing.assert_array_equal(first_observation[key], second_observation[key])
    np.testing.assert_array_equal(first_info["action_mask"], second_info["action_mask"])


def test_step_after_termination_requires_reset() -> None:
    env = TaskOffloadingEnv(two_step_scenario())
    env.reset()
    env.step(0)
    env.step(0)
    with pytest.raises(RuntimeError, match="terminated"):
        env.step(0)
