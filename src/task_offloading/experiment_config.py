"""Single source of truth for the offloading experiment settings.

Scripts import these values and the thesis method chapter is generated from
them, so the documented and the executed configuration cannot drift apart.
"""

from __future__ import annotations

from typing import Any

PPO_ALGORITHM = "MaskablePPO (sb3-contrib), MultiInputPolicy"
PPO_HYPERPARAMETERS: dict[str, Any] = {
    "n_steps": 2048,
    "batch_size": 256,
    "learning_rate": 3e-4,
    "gamma": 0.95,
    "ent_coef": 0.01,
}
PPO_SEEDS = 5
PPO_TIMESTEPS = 150_000
TRAIN_EPISODES = 400
TEST_EPISODES = 60
EXPERIMENT_SEED = 20261120
LATENCY_REFERENCE_S = 0.2
ENERGY_REFERENCE_J = 0.1

__all__ = [
    "ENERGY_REFERENCE_J",
    "EXPERIMENT_SEED",
    "LATENCY_REFERENCE_S",
    "PPO_ALGORITHM",
    "PPO_HYPERPARAMETERS",
    "PPO_SEEDS",
    "PPO_TIMESTEPS",
    "TEST_EPISODES",
    "TRAIN_EPISODES",
]
