# ruff: noqa: E501
"""NEP layer and noisy-preview invariants."""

from __future__ import annotations

import dataclasses

import numpy as np

from task_offloading.domain import ExecutionTier, TaskRequirements
from task_offloading.env import TaskOffloadingEnv
from task_offloading.sim import PhysicsConfig
from task_offloading.sim.trace_hybrid import (
    HybridConfig,
    PhysicalPools,
    SemanticItem,
    WorkloadWindow,
    build_episode,
)
from tests.phase_2.helpers import two_step_scenario


def _pools() -> PhysicalPools:
    window = WorkloadWindow(
        source="w",
        interarrival_s=tuple([0.01] * 8),
        input_bits=tuple([4000.0] * 8),
        output_bits=tuple([2000.0] * 8),
    )
    return PhysicalPools(
        device_turnaround_s=np.asarray([1.0, 1.1, 0.9]),
        host_mean_turnaround_s={
            "raspberry_pi_4b": 1.0,
            "macbook_pro_1": 0.1,
            "ubuntu_virtualbox_vm": 0.4,
            "macbook_pro_2": 0.3,
        },
        train_windows=(window,),
        test_windows=(window,),
        user_edge_distances_m=np.asarray([[50.0, 300.0, 900.0]]),
    )


ITEMS = [SemanticItem("a", TaskRequirements(), TaskRequirements())]
NEP = {
    "nep_percentiles": (0.01, 0.5, 0.99),
    "nep_edge_load_quantiles": (0.2, 0.5, 0.8),
    "nep_rtt_ms_quantiles": (20.0, 40.0, 80.0),
}


def _episode(config: HybridConfig) -> object:
    return build_episode(
        rng=np.random.default_rng(1),
        pools=_pools(),
        windows=_pools().train_windows,
        semantic_items=ITEMS,
        config=dataclasses.replace(config, episode_tasks=8),
    )[0]


def test_nep_disabled_reproduces_base_episode_and_enabled_slows_edges() -> None:
    base = _episode(HybridConfig())
    again = _episode(HybridConfig())
    assert base == again
    with_nep = _episode(HybridConfig(**NEP))
    assert with_nep.tasks == base.tasks  # type: ignore[attr-defined]
    assert with_nep.events == base.events  # type: ignore[attr-defined]
    for plain, loaded in zip(
        base.initial_state.servers,  # type: ignore[attr-defined]
        with_nep.initial_state.servers,  # type: ignore[attr-defined]
        strict=True,
    ):
        if plain.spec.tier is ExecutionTier.EDGE:
            assert 0.19 * plain.spec.cpu_hz <= loaded.spec.cpu_hz < 0.81 * plain.spec.cpu_hz
        else:
            assert 0.01 <= loaded.spec.one_way_propagation_s <= 0.04


def test_preview_noise_is_shared_and_zero_noise_is_exact() -> None:
    scenario = two_step_scenario()
    exact = TaskOffloadingEnv(scenario, PhysicsConfig())
    exact.reset(seed=0)
    noisy_a = TaskOffloadingEnv(scenario, PhysicsConfig(), preview_noise_sigma=0.5)
    noisy_b = TaskOffloadingEnv(scenario, PhysicsConfig(), preview_noise_sigma=0.5)
    noisy_a.reset(seed=0)
    noisy_b.reset(seed=7)
    clean = [o.total_latency_s for o in exact.preview_estimates()]
    first = [o.total_latency_s for o in noisy_a.preview_estimates()]
    second = [o.total_latency_s for o in noisy_b.preview_estimates()]
    assert first == second  # same information for every policy
    assert first != clean
    # Physics is not affected by the noise.
    assert exact.step(1)[1] == noisy_a.step(1)[1]
