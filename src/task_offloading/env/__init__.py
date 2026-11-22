"""Gymnasium adapter around the single physics core.

The adapter contains no physical equation.  Per-action previews are obtained by
calling :func:`task_offloading.sim.transition` with a failure-free event and
the semantic mask disabled, so the observation, the greedy baselines and the
real step all share one implementation of latency and energy.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from task_offloading.domain import (
    ActionTarget,
    ExecutionTier,
    ExogenousEvent,
    FailureReason,
    OffloadOutcome,
    SystemState,
    Task,
    TaskRequirements,
    TransitionResult,
)
from task_offloading.sim import PhysicsConfig, Scenario, action_mask, transition

TASK_FEATURES = 3
REQUIREMENT_FEATURES = 10
TARGET_FEATURES = 8
DEVICE_FEATURES = 2

_UNKNOWN_REQUIREMENTS = TaskRequirements()


def _squash(value: float, reference: float) -> float:
    """Map a non-negative magnitude to [0, 1) without an arbitrary clip."""

    if not math.isfinite(value) or value <= 0:
        return 0.0
    return value / (value + reference)


def _log_scale(value: float, decades: float) -> float:
    return min(1.0, math.log10(1.0 + max(value, 0.0)) / decades)


def preview_outcomes(
    state: SystemState,
    task: Task,
    actions: tuple[ActionTarget, ...],
    config: PhysicsConfig,
) -> tuple[OffloadOutcome, ...]:
    """Expected outcome of every action if no surprise failure happens.

    Uses only information available before the decision.  Constraint flags in
    the returned outcomes are computed against ``task.requirements`` and must
    therefore not be shown to a policy; callers use latency, energy, success
    probability and accuracy only.
    """

    preview_config = replace(config, mask_semantic_permissions=False)
    calm = ExogenousEvent(0.0)
    return tuple(
        transition(state, task, action, calm, preview_config).outcome
        for action in actions
    )


class TaskOffloadingEnv(gym.Env[dict[str, NDArray[np.float32]], int]):
    """Finite-horizon, event-indexed offloading MDP over one fixed scenario.

    ``semantic_observation=False`` builds the semantic-off condition: the
    policy sees no requirement features.  Masks then follow ``config``; for a
    fair semantic-off run, pass a config with ``mask_semantic_permissions``
    disabled.  Rewards and violations are always judged against the true
    requirements stored in ``Task.requirements``.
    """

    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        scenario: Scenario,
        config: PhysicsConfig | None = None,
        *,
        semantic_observation: bool = True,
        preview_noise_sigma: float = 0.0,
        noise_seed: int = 0,
    ) -> None:
        super().__init__()
        if preview_noise_sigma < 0:
            raise ValueError("preview_noise_sigma must be >= 0")
        self.scenario = scenario
        self.config = config or PhysicsConfig()
        self.semantic_observation = semantic_observation
        self.preview_noise_sigma = preview_noise_sigma
        self.noise_seed = noise_seed
        self._preview_cache: tuple[int, tuple[OffloadOutcome, ...]] | None = None
        self._actions = scenario.actions
        self._initial_battery_j = {
            item.spec.device_id: max(item.battery_j, 1e-12)
            for item in scenario.initial_state.devices
        }
        count = len(self._actions)
        self.action_space = spaces.Discrete(count)
        self.observation_space = spaces.Dict(
            {
                "task": spaces.Box(0.0, 1.0, (TASK_FEATURES,), np.float32),
                "requirements": spaces.Box(
                    0.0, 1.0, (REQUIREMENT_FEATURES,), np.float32
                ),
                "targets": spaces.Box(0.0, 1.0, (count, TARGET_FEATURES), np.float32),
                "device": spaces.Box(0.0, 1.0, (DEVICE_FEATURES,), np.float32),
            }
        )
        self._state = scenario.initial_state
        self._index = 0
        self._terminated = False
        self.last_transition: TransitionResult | None = None

    # ------------------------------------------------------------------ state
    @property
    def current_state(self) -> SystemState:
        return self._state

    @property
    def step_index(self) -> int:
        return self._index

    def current_task(self) -> Task:
        if self._index >= len(self.scenario.tasks):
            raise RuntimeError("episode has no remaining task")
        return self.scenario.tasks[self._index]

    def action_masks(self) -> NDArray[np.bool_]:
        """Mask for sb3-contrib MaskablePPO; all-false is never returned."""

        if self._terminated:
            return np.ones(len(self._actions), dtype=bool)
        mask = np.asarray(
            action_mask(self._state, self.current_task(), self._actions, self.config),
            dtype=bool,
        )
        if not mask.any():
            # Every action is forbidden or down: keep the episode well defined;
            # whichever action is taken is recorded as a failure by the core.
            return np.ones(len(self._actions), dtype=bool)
        return mask

    # ------------------------------------------------------------- previews
    def _scenario_fingerprint(self) -> int:
        head = self.scenario.tasks[: min(4, len(self.scenario.tasks))]
        text = repr([(t.input_bits, t.compute_cycles, t.arrival_time_s) for t in head])
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")

    def preview_estimates(self) -> tuple[OffloadOutcome, ...]:
        """Per-action latency/energy estimates the decision maker may use.

        Without noise these are the exact failure-free previews.  With
        ``preview_noise_sigma > 0`` latency and energy are multiplied by
        independent mean-one log-normal factors that depend only on
        (noise_seed, scenario, step, action), so every policy evaluated on the
        same episode sees the same noisy information.  Physics is unaffected.
        """

        if self._preview_cache is not None and self._preview_cache[0] == self._index:
            return self._preview_cache[1]
        previews = preview_outcomes(
            self._state, self.current_task(), self._actions, self.config
        )
        sigma = self.preview_noise_sigma
        if sigma > 0:
            rng = np.random.default_rng(
                [self.noise_seed, self._scenario_fingerprint(), self._index]
            )
            factors = np.exp(
                sigma * rng.standard_normal((len(previews), 2)) - sigma**2 / 2
            )
            previews = tuple(
                replace(
                    outcome,
                    total_latency_s=outcome.total_latency_s * float(f[0]),
                    device_energy_j=outcome.device_energy_j * float(f[1]),
                )
                for outcome, f in zip(previews, factors, strict=True)
            )
        self._preview_cache = (self._index, previews)
        return previews

    # ----------------------------------------------------------- observation
    def _requirement_vector(self, requirements: TaskRequirements) -> list[float]:
        reference = self.config.latency_reference_s
        allowed = requirements.allowed_tiers
        latency = requirements.max_latency_s
        reliability = requirements.min_success_probability
        accuracy = requirements.min_accuracy
        return [
            float(ExecutionTier.DEVICE in allowed),
            float(ExecutionTier.EDGE in allowed),
            float(ExecutionTier.CLOUD in allowed),
            float(latency is not None),
            _squash(latency, reference) if latency is not None else 0.0,
            float(reliability is not None),
            reliability if reliability is not None else 0.0,
            float(accuracy is not None),
            accuracy if accuracy is not None else 0.0,
            requirements.semantic_confidence,
        ]

    def _observation(self) -> dict[str, NDArray[np.float32]]:
        count = len(self._actions)
        if self._terminated:
            return {
                "task": np.zeros(TASK_FEATURES, np.float32),
                "requirements": np.zeros(REQUIREMENT_FEATURES, np.float32),
                "targets": np.zeros((count, TARGET_FEATURES), np.float32),
                "device": np.zeros(DEVICE_FEATURES, np.float32),
            }
        task = self.current_task()
        state = self._state
        believed = (
            task.policy_requirements
            if self.semantic_observation
            else _UNKNOWN_REQUIREMENTS
        )
        mask = self.action_masks()
        previews = self.preview_estimates()
        latency_ref = self.config.latency_reference_s
        energy_ref = self.config.energy_reference_j
        rows: list[list[float]] = []
        for action, allowed, outcome in zip(self._actions, mask, previews, strict=True):
            if action.tier is ExecutionTier.DEVICE:
                available = True
                busy_until = state.device(task.source_device_id).available_at_s
            else:
                server = state.server(action.target_id)
                available = server.is_available
                busy_until = server.available_at_s
            usable = outcome.failure_reason is FailureReason.NONE
            meets_deadline = (
                1.0
                if self.semantic_observation
                and believed.max_latency_s is not None
                and usable
                and outcome.total_latency_s <= believed.max_latency_s
                else 0.0
            )
            rows.append(
                [
                    float(allowed),
                    float(available and usable),
                    _squash(outcome.total_latency_s, latency_ref) if usable else 1.0,
                    _squash(outcome.device_energy_j, energy_ref) if usable else 1.0,
                    outcome.achieved_success_probability,
                    outcome.achieved_accuracy,
                    _squash(max(busy_until - state.now_s, 0.0), latency_ref),
                    meets_deadline,
                ]
            )
        device = state.device(task.source_device_id)
        initial = self._initial_battery_j[device.spec.device_id]
        return {
            "task": np.asarray(
                [
                    _log_scale(task.input_bits, 10.0),
                    _log_scale(task.output_bits, 10.0),
                    _log_scale(task.compute_cycles, 12.0),
                ],
                np.float32,
            ),
            "requirements": np.asarray(self._requirement_vector(believed), np.float32),
            "targets": np.clip(np.asarray(rows, np.float32), 0.0, 1.0),
            "device": np.asarray(
                [
                    min(1.0, device.battery_j / initial),
                    self._index / max(len(self.scenario.tasks), 1),
                ],
                np.float32,
            ),
        }

    def _info(self) -> dict[str, Any]:
        return {
            "step_index": self._index,
            "action_mask": self.action_masks(),
        }

    # ------------------------------------------------------------- gym api
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
        super().reset(seed=seed)
        self._state = self.scenario.initial_state
        self._index = 0
        self._terminated = False
        self.last_transition = None
        self._preview_cache = None
        return self._observation(), self._info()

    def step(
        self, action: int
    ) -> tuple[dict[str, NDArray[np.float32]], float, bool, bool, dict[str, Any]]:
        if self._terminated:
            raise RuntimeError("episode terminated; call reset() before step()")
        index = int(action)
        if not 0 <= index < len(self._actions):
            raise ValueError(f"action index out of range: {action!r}")
        result = transition(
            self._state,
            self.scenario.tasks[self._index],
            self._actions[index],
            self.scenario.events[self._index],
            self.config,
        )
        self.last_transition = result
        self._state = result.next_state
        self._index += 1
        self._terminated = self._index >= len(self.scenario.tasks)
        info = self._info()
        info["outcome"] = result.outcome
        info["reward_components"] = result.reward_components
        return self._observation(), float(result.reward), self._terminated, False, info


class ScenarioPoolEnv(TaskOffloadingEnv):
    """Draw a new pre-built scenario at every reset (for RL training).

    All scenarios must share one action tuple so that the spaces are fixed.
    The draw uses the Gymnasium-seeded ``np_random`` stream.
    """

    def __init__(
        self,
        scenarios: tuple[Scenario, ...],
        config: PhysicsConfig | None = None,
        *,
        semantic_observation: bool = True,
        preview_noise_sigma: float = 0.0,
        noise_seed: int = 0,
    ) -> None:
        if not scenarios:
            raise ValueError("scenario pool must not be empty")
        actions = scenarios[0].actions
        if any(item.actions != actions for item in scenarios):
            raise ValueError("all pooled scenarios must share the same actions")
        self._pool = scenarios
        super().__init__(
            scenarios[0],
            config,
            semantic_observation=semantic_observation,
            preview_noise_sigma=preview_noise_sigma,
            noise_seed=noise_seed,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
        gym.Env.reset(self, seed=seed)
        self.scenario = self._pool[int(self.np_random.integers(len(self._pool)))]
        self._initial_battery_j = {
            item.spec.device_id: max(item.battery_j, 1e-12)
            for item in self.scenario.initial_state.devices
        }
        return super().reset(options=options)


__all__ = [
    "ScenarioPoolEnv",
    "TaskOffloadingEnv",
    "preview_outcomes",
]
