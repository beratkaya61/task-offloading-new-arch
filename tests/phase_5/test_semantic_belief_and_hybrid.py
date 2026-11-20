# ruff: noqa: E501, RUF001
"""Tests for believed-vs-true requirements, paraphrase pools and composition."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from task_offloading.domain import ExecutionTier, TaskRequirements
from task_offloading.env import ScenarioPoolEnv, TaskOffloadingEnv
from task_offloading.semantic.compositional import compose_tasks
from task_offloading.semantic.paraphrase import RULES, paraphrase_task, paraphrase_text
from task_offloading.sim import EventSimulationRunner, PhysicsConfig, transition
from task_offloading.sim.trace_hybrid import annotation_to_requirements
from tests.phase_2.helpers import actions, state, task, two_step_scenario

NO_CLOUD = TaskRequirements(
    allowed_tiers=frozenset({ExecutionTier.DEVICE, ExecutionTier.EDGE})
)
ANYWHERE = TaskRequirements()


def test_mask_follows_belief_but_violation_follows_truth() -> None:
    # Extractor missed the cloud ban: the mask allows cloud, the truth forbids it.
    item = dataclasses.replace(
        task(), requirements=NO_CLOUD, believed_requirements=ANYWHERE
    )
    current = state(edge_available_at_s=0.0)
    from task_offloading.domain import ExogenousEvent
    from task_offloading.sim import action_mask

    mask = action_mask(current, item, actions(), PhysicsConfig())
    assert mask == (True, True, True)
    result = transition(
        current, item, actions()[2], ExogenousEvent(0.0), PhysicsConfig()
    )
    assert result.outcome.hard_constraint_violated
    assert result.outcome.completed


def test_belief_can_mask_an_action_the_truth_allows() -> None:
    item = dataclasses.replace(
        task(), requirements=ANYWHERE, believed_requirements=NO_CLOUD
    )
    from task_offloading.sim import action_mask

    mask = action_mask(state(), item, actions(), PhysicsConfig())
    assert mask == (True, True, False)


def test_policy_requirements_default_to_truth() -> None:
    item = dataclasses.replace(task(), requirements=NO_CLOUD)
    assert item.policy_requirements is NO_CLOUD


def test_semantic_off_observation_hides_requirements() -> None:
    scenario = two_step_scenario()
    tasks = tuple(dataclasses.replace(t, requirements=NO_CLOUD) for t in scenario.tasks)
    scenario = dataclasses.replace(scenario, tasks=tasks)
    on, _ = TaskOffloadingEnv(scenario).reset(seed=1)
    off, _ = TaskOffloadingEnv(scenario, semantic_observation=False).reset(seed=1)
    assert on["requirements"][2] == 0.0  # cloud not allowed
    assert off["requirements"][2] == 1.0  # unknown requirements: all allowed


def test_scenario_pool_env_matches_single_env_physics() -> None:
    scenario = two_step_scenario()
    pool = ScenarioPoolEnv((scenario,), PhysicsConfig())
    pool.reset(seed=3)
    rewards = [pool.step(index)[1] for index in (1, 2)]
    replay = EventSimulationRunner(scenario, PhysicsConfig()).run((1, 2))
    assert rewards == [item.reward for item in replay.transitions]


def test_paraphrase_pools_are_disjoint_and_label_numbers_survive() -> None:
    for rule in RULES:
        assert not set(rule.train) & set(rule.test)
    text, applied = paraphrase_text(
        "Sonuç en geç 80 ms içinde hazır olmalıdır.", pool="test", key="k"
    )
    assert applied == ["latency_ms_deadline"]
    assert "80" in text
    assert text != "Sonuç en geç 80 ms içinde hazır olmalıdır."


def test_paraphrase_is_deterministic() -> None:
    item = {
        "task_id": "x",
        "task_text": "İşlenen kayıt hassas veridir.",
        "user_policy_text": None,
    }
    assert paraphrase_task(item, pool="test") == paraphrase_task(item, pool="test")


def test_compositional_labels_map_to_requirements() -> None:
    base = [
        {
            "task_id": "b1",
            "task_text": "Akıllı sayaç, enerji verisi üzerinde tüketimi tahmin etme görevini yürütür.",
            "design_strata": {"scenario_domain": "smart_city"},
        }
    ]
    tasks, labels = compose_tasks(
        base, variants_per_task=50, wording="canonical", seed=1, prefix="t"
    )
    assert len(tasks) == 50
    banned = 0
    for item in tasks:
        label = labels[item["task_id"]]
        requirements = annotation_to_requirements(label)
        policy = label["execution_policy"]
        if (
            policy["status"] in {"explicit", "partial"}
            and policy["cloud"] == "forbidden"
        ):
            banned += 1
            assert ExecutionTier.CLOUD not in requirements.allowed_tiers
        if label["latency"]["max_ms"] is not None:
            assert requirements.max_latency_s == pytest.approx(
                label["latency"]["max_ms"] / 1000
            )
    assert banned > 0


def test_conservative_fallback_only_removes_cloud() -> None:
    annotation = {
        "execution_policy": {"status": "not_stated", "confidence": 0.9},
        "privacy": {"status": "explicit", "data_class": "sensitive"},
        "latency": {"status": "not_stated"},
        "reliability": {"status": "not_stated"},
        "accuracy": {"status": "not_stated"},
    }
    plain = annotation_to_requirements(annotation)
    careful = annotation_to_requirements(annotation, conservative=True)
    assert plain.allowed_tiers == frozenset(ExecutionTier)
    assert careful.allowed_tiers == frozenset(
        {ExecutionTier.DEVICE, ExecutionTier.EDGE}
    )
    assert careful.allowed_tiers <= plain.allowed_tiers
    assert np.isclose(careful.semantic_confidence, 1.0)
