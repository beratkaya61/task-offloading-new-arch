# ruff: noqa: E501
"""NEP layer and noisy-preview invariants."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from task_offloading.domain import ExecutionTier, TaskRequirements
from task_offloading.env import TaskOffloadingEnv
from task_offloading.sim import PhysicsConfig
from task_offloading.sim.scenario import Scenario
from task_offloading.sim.trace_hybrid import (
    HybridConfig,
    PhysicalPools,
    SemanticItem,
    WorkloadWindow,
    ablate_physical_pools,
    build_episode,
    split_eua_entity_holdout,
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
NEP: dict[str, Any] = {
    "nep_percentiles": (0.01, 0.5, 0.99),
    "nep_edge_load_quantiles": (0.2, 0.5, 0.8),
    "nep_rtt_ms_quantiles": (20.0, 40.0, 80.0),
}


def _episode(config: HybridConfig) -> Scenario:
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
    assert with_nep.tasks == base.tasks
    assert with_nep.events == base.events
    for plain, loaded in zip(
        base.initial_state.servers,
        with_nep.initial_state.servers,
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


def test_step_exposes_reward_components_for_registered_decomposition() -> None:
    env = TaskOffloadingEnv(two_step_scenario(), PhysicsConfig())
    env.reset(seed=0)
    _, reward, _, _, info = env.step(1)
    components = info["reward_components"]
    assert reward == components.reward(env.config.reward_weights)


def test_entity_holdout_is_disjoint_deterministic_and_complete() -> None:
    pools = dataclasses.replace(
        _pools(),
        user_edge_distances_m=np.arange(90, dtype=float).reshape(30, 3) + 10.0,
    )
    train, test = split_eua_entity_holdout(pools, seed=17)
    train_again, test_again = split_eua_entity_holdout(pools, seed=17)
    assert np.array_equal(train.user_edge_distances_m, train_again.user_edge_distances_m)
    assert np.array_equal(test.user_edge_distances_m, test_again.user_edge_distances_m)
    assert len(train.user_edge_distances_m) == 24
    assert len(test.user_edge_distances_m) == 6
    train_rows = {tuple(row) for row in train.user_edge_distances_m}
    test_rows = {tuple(row) for row in test.user_edge_distances_m}
    assert train_rows.isdisjoint(test_rows)


def test_physical_source_ablations_remove_only_selected_variability() -> None:
    first = WorkloadWindow(
        source="a",
        interarrival_s=(0.01, 0.03),
        input_bits=(100.0, 300.0),
        output_bits=(10.0, 30.0),
    )
    second = WorkloadWindow(
        source="b",
        interarrival_s=(0.02, 0.04),
        input_bits=(200.0, 400.0),
        output_bits=(20.0, 40.0),
    )
    pools = dataclasses.replace(
        _pools(),
        train_windows=(first,),
        test_windows=(second,),
        user_edge_distances_m=np.arange(90, dtype=float).reshape(30, 3) + 10.0,
    )

    uci = ablate_physical_pools(
        pools, "without_uci_heterogeneity", fit_seed=1
    )
    assert len(set(uci.host_mean_turnaround_s.values())) == 1
    assert np.unique(uci.device_turnaround_s).size == 1

    bupt = ablate_physical_pools(pools, "without_bupt_variability", fit_seed=1)
    assert bupt.train_windows[0].input_bits == (200.0, 200.0)
    assert bupt.test_windows[0].input_bits == (200.0, 200.0)
    assert bupt.test_windows[0].output_bits == (20.0, 20.0)

    eua = ablate_physical_pools(pools, "without_eua_variability", fit_seed=1)
    assert len(eua.user_edge_distances_m) == len(pools.user_edge_distances_m)
    assert np.unique(eua.user_edge_distances_m, axis=0).shape[0] == 1
