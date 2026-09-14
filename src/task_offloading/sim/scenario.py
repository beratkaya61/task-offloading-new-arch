"""Finite event scenarios and a non-Gym adapter around the physics core."""

from __future__ import annotations

import math
from dataclasses import dataclass

from task_offloading.domain import (
    ActionTarget,
    ExogenousEvent,
    SystemState,
    Task,
    TransitionResult,
)
from task_offloading.sim.config import PhysicsConfig
from task_offloading.sim.core import transition


@dataclass(frozen=True, slots=True)
class Scenario:
    """All policy-independent inputs for one finite-horizon episode."""

    initial_state: SystemState
    tasks: tuple[Task, ...]
    events: tuple[ExogenousEvent, ...]
    actions: tuple[ActionTarget, ...]

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("scenario must contain at least one task")
        if len(self.tasks) != len(self.events):
            raise ValueError("tasks and events must have equal length")
        if not self.actions:
            raise ValueError("scenario must contain at least one action")
        labels = [action.label for action in self.actions]
        if len(labels) != len(set(labels)):
            raise ValueError("action labels must be unique")
        expected_time_s = self.initial_state.now_s
        for index, (task, event) in enumerate(
            zip(self.tasks, self.events, strict=True)
        ):
            if not math.isclose(task.arrival_time_s, expected_time_s, abs_tol=1e-9):
                raise ValueError(
                    f"task {index} arrival does not follow the event timeline"
                )
            expected_time_s += event.next_interarrival_s


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Complete replay output from the event adapter."""

    transitions: tuple[TransitionResult, ...]
    final_state: SystemState

    @property
    def total_reward(self) -> float:
        return sum(item.reward for item in self.transitions)


class EventSimulationRunner:
    """Simple event adapter; it never reimplements physical equations."""

    def __init__(self, scenario: Scenario, config: PhysicsConfig | None = None) -> None:
        self.scenario = scenario
        self.config = config or PhysicsConfig()

    def run(self, action_indices: tuple[int, ...]) -> EpisodeResult:
        if len(action_indices) != len(self.scenario.tasks):
            raise ValueError("one action index is required for every task")
        state = self.scenario.initial_state
        results: list[TransitionResult] = []
        for index, action_index in enumerate(action_indices):
            if not 0 <= action_index < len(self.scenario.actions):
                raise ValueError(f"action index out of range at step {index}")
            result = transition(
                state,
                self.scenario.tasks[index],
                self.scenario.actions[action_index],
                self.scenario.events[index],
                self.config,
            )
            results.append(result)
            state = result.next_state
        return EpisodeResult(tuple(results), state)
