"""Oracle, invariant, masking, failure, and determinism tests."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from task_offloading.domain import (
    ExecutionTier,
    ExogenousEvent,
    FailureReason,
    TaskRequirements,
)
from task_offloading.sim import PhysicsConfig, action_mask, transition

from .helpers import actions, state, task


def test_local_transition_matches_hand_calculated_oracle() -> None:
    result = transition(
        state(), task(), actions()[0], ExogenousEvent(0.5), PhysicsConfig()
    )
    # C/f = 4e9/2e9 = 2 s; kappa*C*f^2 = 1.6 J.
    assert result.outcome.compute_time_s == pytest.approx(2.0)
    assert result.outcome.total_latency_s == pytest.approx(2.0)
    assert result.outcome.device_energy_j == pytest.approx(1.6)
    assert result.next_state.device("device_0").available_at_s == pytest.approx(2.0)
    assert result.next_state.device("device_0").battery_j == pytest.approx(98.4)


def test_edge_transition_matches_hand_calculated_oracle() -> None:
    result = transition(
        state(), task(), actions()[1], ExogenousEvent(0.5), PhysicsConfig()
    )
    # Upload=2, arrival at server=2, FCFS wait=2, compute=1, download=1.
    assert result.outcome.upload_time_s == pytest.approx(2.0)
    assert result.outcome.queue_wait_time_s == pytest.approx(2.0)
    assert result.outcome.compute_time_s == pytest.approx(1.0)
    assert result.outcome.download_time_s == pytest.approx(1.0)
    assert result.outcome.total_latency_s == pytest.approx(6.0)
    # 2 W*2 s + .5 W*(2+1) s + 1 W*1 s = 6.5 J.
    assert result.outcome.device_energy_j == pytest.approx(6.5)
    assert result.next_state.server("edge_0").available_at_s == pytest.approx(5.0)


def test_cloud_transition_includes_both_backhaul_directions_and_propagation() -> None:
    result = transition(
        state(), task(), actions()[2], ExogenousEvent(0.5), PhysicsConfig()
    )
    assert result.outcome.upload_time_s == pytest.approx(1.0)
    assert result.outcome.backhaul_upload_time_s == pytest.approx(0.2)
    assert result.outcome.compute_time_s == pytest.approx(0.4)
    assert result.outcome.backhaul_download_time_s == pytest.approx(0.1)
    assert result.outcome.download_time_s == pytest.approx(0.5)
    assert result.outcome.total_latency_s == pytest.approx(2.24)


def test_cloud_forward_propagation_is_included_before_queue_admission() -> None:
    initial = state()
    cloud = replace(initial.server("cloud"), available_at_s=1.21)
    initial = replace(initial, servers=(initial.server("edge_0"), cloud))
    result = transition(
        initial, task(), actions()[2], ExogenousEvent(0.5), PhysicsConfig()
    )
    # Upload + backhaul + one-way propagation reaches cloud at t=1.22.
    assert result.outcome.queue_wait_time_s == 0.0


def test_reward_is_exact_negative_weighted_component_sum() -> None:
    config = PhysicsConfig(latency_reference_s=2.0, energy_reference_j=2.0)
    result = transition(state(), task(), actions()[0], ExogenousEvent(0.5), config)
    assert result.reward_components.latency_norm == pytest.approx(1.0)
    assert result.reward_components.energy_norm == pytest.approx(0.8)
    assert result.reward == pytest.approx(-1.8)


def test_explicit_execution_policy_masks_but_unknown_does_not() -> None:
    local_only = task(
        requirements=TaskRequirements(allowed_tiers=frozenset({ExecutionTier.DEVICE}))
    )
    unknown = task(
        requirements=TaskRequirements(
            allowed_tiers=frozenset(ExecutionTier),
            unknown_fields=frozenset({"execution_policy"}),
        )
    )
    config = PhysicsConfig()
    assert action_mask(state(), local_only, actions(), config) == (True, False, False)
    assert action_mask(state(), unknown, actions(), config) == (True, True, True)


def test_known_unavailability_is_masked_without_using_future_failures() -> None:
    assert action_mask(
        state(edge_online=False), task(), actions(), PhysicsConfig()
    ) == (
        True,
        False,
        True,
    )


def test_unavailable_action_is_a_hard_failure_without_queue_mutation() -> None:
    initial = state(edge_online=False)
    result = transition(
        initial, task(), actions()[1], ExogenousEvent(0.5), PhysicsConfig()
    )
    assert result.outcome.failure_reason is FailureReason.TARGET_UNAVAILABLE
    assert result.outcome.hard_constraint_violated
    assert not result.outcome.accepted
    assert result.next_state.server("edge_0").available_at_s == 4.0


def test_mask_ablation_keeps_hard_constraint_penalty() -> None:
    local_only = task(
        requirements=TaskRequirements(allowed_tiers=frozenset({ExecutionTier.DEVICE}))
    )
    config = PhysicsConfig(mask_semantic_permissions=False)
    assert action_mask(state(), local_only, actions(), config) == (True, True, True)
    result = transition(
        state(), local_only, actions()[1], ExogenousEvent(0.5), config
    )
    assert result.outcome.completed
    assert result.outcome.hard_constraint_violated
    assert result.reward_components.hard_constraint_violation == 1.0
    assert not result.outcome.success


def test_unavailability_mask_ablation_does_not_revive_target() -> None:
    initial = state(edge_online=False)
    config = PhysicsConfig(mask_known_unavailable_targets=False)
    assert action_mask(initial, task(), actions(), config) == (True, True, True)
    result = transition(
        initial, task(), actions()[1], ExogenousEvent(0.5), config
    )
    assert result.outcome.failure_reason is FailureReason.TARGET_UNAVAILABLE
    assert not result.outcome.completed


def test_exogenous_link_failure_is_not_leaked_into_action_mask() -> None:
    initial = state(edge_available_at_s=0.0)
    assert action_mask(initial, task(), actions(), PhysicsConfig())[1]
    result = transition(
        initial,
        task(),
        actions()[1],
        ExogenousEvent(0.5, failed_link_ids=frozenset({"edge_0"})),
        PhysicsConfig(failure_detection_s=0.1),
    )
    assert result.outcome.failure_reason is FailureReason.LINK_FAILURE
    assert result.next_state.server("edge_0").available_at_s == 0.0
    assert result.outcome.device_energy_j == pytest.approx(0.2)


def test_realized_failure_does_not_replace_expected_reliability_with_zero() -> None:
    work = task(requirements=TaskRequirements(min_success_probability=0.9))
    result = transition(
        state(),
        work,
        actions()[1],
        ExogenousEvent(0.5, failed_link_ids=frozenset({"edge_0"})),
        PhysicsConfig(failure_detection_s=0.1),
    )
    assert result.outcome.achieved_success_probability == pytest.approx(
        0.98 * 0.97**2
    )
    assert not result.outcome.reliability_violated
    assert result.reward_components.failure_or_drop == 1.0


def test_battery_depletion_cannot_create_negative_battery() -> None:
    initial = state()
    weak_device = replace(initial.devices[0], battery_j=0.25)
    initial = replace(initial, devices=(weak_device,))
    result = transition(
        initial, task(), actions()[0], ExogenousEvent(0.5), PhysicsConfig()
    )
    assert result.outcome.failure_reason is FailureReason.BATTERY_DEPLETED
    assert result.next_state.device("device_0").battery_j == 0.0
    assert result.outcome.device_energy_j == pytest.approx(0.25)


def test_fcfs_queue_never_moves_backwards_across_overlapping_tasks() -> None:
    config = PhysicsConfig()
    first = transition(
        state(edge_available_at_s=0.0),
        task(task_id="a"),
        actions()[1],
        ExogenousEvent(0.5),
        config,
    )
    second = transition(
        first.next_state,
        task(task_id="b", arrival_time_s=0.5),
        actions()[1],
        ExogenousEvent(0.5),
        config,
    )
    assert second.outcome.queue_wait_time_s > first.outcome.queue_wait_time_s
    assert (
        second.next_state.server("edge_0").available_at_s
        >= first.next_state.server("edge_0").available_at_s
    )


def test_queue_wait_clears_when_prior_compute_completed_before_next_upload() -> None:
    config = PhysicsConfig()
    first = transition(
        state(edge_available_at_s=0.0),
        task(task_id="a"),
        actions()[1],
        ExogenousEvent(5.0),
        config,
    )
    second = transition(
        first.next_state,
        task(task_id="b", arrival_time_s=5.0),
        actions()[1],
        ExogenousEvent(0.5),
        config,
    )
    assert first.next_state.server("edge_0").available_at_s == pytest.approx(3.0)
    assert second.outcome.queue_wait_time_s == 0.0


@given(
    input_bits=st.floats(min_value=0.0, max_value=1e8, allow_nan=False),
    output_bits=st.floats(min_value=0.0, max_value=1e8, allow_nan=False),
    cycles=st.floats(min_value=0.0, max_value=1e10, allow_nan=False),
)
def test_latency_energy_and_reward_are_finite_nonnegative_costs(
    input_bits: float, output_bits: float, cycles: float
) -> None:
    work = task(input_bits=input_bits, output_bits=output_bits, compute_cycles=cycles)
    result = transition(
        state(edge_available_at_s=0.0),
        work,
        actions()[1],
        ExogenousEvent(0.1),
        PhysicsConfig(),
    )
    assert math.isfinite(result.outcome.total_latency_s)
    assert result.outcome.total_latency_s >= 0.0
    assert math.isfinite(result.outcome.device_energy_j)
    assert result.outcome.device_energy_j >= 0.0
    assert math.isfinite(result.reward)
    assert result.reward <= 0.0


def test_identical_inputs_produce_identical_transition() -> None:
    inputs = (state(), task(), actions()[1], ExogenousEvent(0.5), PhysicsConfig())
    assert transition(*inputs) == transition(*inputs)


def test_decision_epoch_timeline_mismatch_fails_fast() -> None:
    with pytest.raises(ValueError, match="decision epoch"):
        transition(
            state(),
            task(arrival_time_s=1.0),
            actions()[0],
            ExogenousEvent(0.5),
            PhysicsConfig(),
        )
