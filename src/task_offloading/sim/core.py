"""The only source of truth for task-offloading physics and rewards."""

from __future__ import annotations

import math
from dataclasses import replace

from task_offloading.domain import (
    ActionTarget,
    DeviceState,
    ExecutionTier,
    ExogenousEvent,
    FailureReason,
    OffloadOutcome,
    RewardComponents,
    ServerState,
    SystemState,
    Task,
    TransitionResult,
)
from task_offloading.sim.config import PhysicsConfig


def _action_structure_is_valid(
    state: SystemState, task: Task, action: ActionTarget
) -> bool:
    try:
        state.device(task.source_device_id)
    except KeyError:
        return False
    if action.tier is ExecutionTier.DEVICE:
        return action.target_id == "local"
    try:
        server = state.server(action.target_id)
        state.link(action.target_id)
    except KeyError:
        return False
    return server.spec.tier is action.tier


def action_is_available(
    state: SystemState,
    task: Task,
    action: ActionTarget,
    config: PhysicsConfig,
) -> bool:
    """Return whether information known before the decision permits an action."""

    if not _action_structure_is_valid(state, task, action):
        return False

    if (
        config.mask_semantic_permissions
        and action.tier not in task.requirements.allowed_tiers
    ):
        return False
    if action.tier is ExecutionTier.DEVICE:
        return True
    server = state.server(action.target_id)
    return not (config.mask_known_unavailable_targets and not server.is_available)


def action_mask(
    state: SystemState,
    task: Task,
    actions: tuple[ActionTarget, ...],
    config: PhysicsConfig,
) -> tuple[bool, ...]:
    """Build a mask from facts observable before taking the action."""

    return tuple(action_is_available(state, task, item, config) for item in actions)


def _constraints(
    task: Task,
    *,
    total_latency_s: float,
    success_probability: float,
    accuracy: float,
    hard_constraint_violated: bool,
    completed: bool,
) -> tuple[bool, bool, bool, bool]:
    requirements = task.requirements
    deadline_missed = (
        requirements.max_latency_s is not None
        and total_latency_s > requirements.max_latency_s
    )
    reliability_violated = (
        requirements.min_success_probability is not None
        and success_probability < requirements.min_success_probability
    )
    accuracy_violated = (
        requirements.min_accuracy is not None and accuracy < requirements.min_accuracy
    )
    success = completed and not any(
        (
            deadline_missed,
            reliability_violated,
            accuracy_violated,
            hard_constraint_violated,
        )
    )
    return deadline_missed, reliability_violated, accuracy_violated, success


def _reward_components(
    outcome: OffloadOutcome, config: PhysicsConfig
) -> RewardComponents:
    return RewardComponents(
        latency_norm=outcome.total_latency_s / config.latency_reference_s,
        energy_norm=outcome.device_energy_j / config.energy_reference_j,
        deadline_miss=float(outcome.deadline_missed),
        soft_constraint_violation=float(outcome.reliability_violated)
        + float(outcome.accuracy_violated),
        failure_or_drop=float(not outcome.completed),
        hard_constraint_violation=float(outcome.hard_constraint_violated),
    )


def _advance_state(
    state: SystemState,
    action: ActionTarget,
    event: ExogenousEvent,
    *,
    device: DeviceState | None = None,
    server: ServerState | None = None,
) -> SystemState:
    devices = tuple(
        device
        if device is not None and item.spec.device_id == device.spec.device_id
        else item
        for item in state.devices
    )
    servers = tuple(
        server
        if server is not None and item.spec.server_id == server.spec.server_id
        else item
        for item in state.servers
    )
    return replace(
        state,
        now_s=state.now_s + event.next_interarrival_s,
        devices=devices,
        servers=servers,
        previous_action_label=action.label,
    )


def _failure_result(
    state: SystemState,
    task: Task,
    action: ActionTarget,
    event: ExogenousEvent,
    config: PhysicsConfig,
    *,
    reason: FailureReason,
    hard_constraint_violated: bool,
    accepted: bool,
    achieved_success_probability: float,
    achieved_accuracy: float,
    consumed_energy_j: float = 0.0,
    device: DeviceState | None = None,
) -> TransitionResult:
    latency_s = config.failure_detection_s
    deadline, reliability, accuracy, success = _constraints(
        task,
        total_latency_s=latency_s,
        success_probability=achieved_success_probability,
        accuracy=achieved_accuracy,
        hard_constraint_violated=hard_constraint_violated,
        completed=False,
    )
    outcome = OffloadOutcome(
        task_id=task.task_id,
        action_label=action.label,
        tier=action.tier,
        accepted=accepted,
        completed=False,
        success=success,
        failure_reason=reason,
        upload_time_s=0.0,
        backhaul_upload_time_s=0.0,
        queue_wait_time_s=0.0,
        compute_time_s=0.0,
        backhaul_download_time_s=0.0,
        download_time_s=0.0,
        total_latency_s=latency_s,
        device_energy_j=consumed_energy_j,
        achieved_success_probability=achieved_success_probability,
        achieved_accuracy=achieved_accuracy,
        deadline_missed=deadline,
        reliability_violated=reliability,
        accuracy_violated=accuracy,
        hard_constraint_violated=hard_constraint_violated,
    )
    components = _reward_components(outcome, config)
    return TransitionResult(
        next_state=_advance_state(state, action, event, device=device),
        outcome=outcome,
        reward_components=components,
        reward=components.reward(config.reward_weights),
    )


def transition(
    state: SystemState,
    task: Task,
    action: ActionTarget,
    event: ExogenousEvent,
    config: PhysicsConfig,
) -> TransitionResult:
    """Apply one whole-task decision without consulting any mutable RNG.

    The exogenous event is supplied by the scenario generator, so every policy
    can be evaluated against exactly the same task and failure realization.
    """

    if not math.isclose(state.now_s, task.arrival_time_s, abs_tol=1e-9):
        raise ValueError(
            "task arrival_time_s must equal state.now_s at the decision epoch"
        )
    try:
        source = state.device(task.source_device_id)
    except KeyError as error:
        raise ValueError("task source_device_id is absent from state") from error

    if not _action_structure_is_valid(state, task, action):
        return _failure_result(
            state,
            task,
            action,
            event,
            config,
            reason=FailureReason.INVALID_ACTION,
            hard_constraint_violated=True,
            accepted=False,
            achieved_success_probability=0.0,
            achieved_accuracy=0.0,
        )

    semantic_disallowed = action.tier not in task.requirements.allowed_tiers
    if config.mask_semantic_permissions and semantic_disallowed:
        return _failure_result(
            state,
            task,
            action,
            event,
            config,
            reason=FailureReason.INVALID_ACTION,
            hard_constraint_violated=True,
            accepted=False,
            achieved_success_probability=0.0,
            achieved_accuracy=0.0,
        )

    if action.tier is ExecutionTier.DEVICE:
        expected_success_probability = source.spec.success_probability
        expected_accuracy = source.spec.accuracy_score
    else:
        candidate_server = state.server(action.target_id)
        candidate_link = state.link(action.target_id)
        backhaul_probability = (
            candidate_server.spec.backhaul_success_probability**2
            if candidate_server.spec.tier is ExecutionTier.CLOUD
            else 1.0
        )
        expected_success_probability = (
            candidate_server.spec.success_probability
            * candidate_link.success_probability**2
            * backhaul_probability
        )
        expected_accuracy = candidate_server.spec.accuracy_score
        if not candidate_server.is_available:
            return _failure_result(
                state,
                task,
                action,
                event,
                config,
                reason=FailureReason.TARGET_UNAVAILABLE,
                hard_constraint_violated=True,
                accepted=False,
                achieved_success_probability=expected_success_probability,
                achieved_accuracy=expected_accuracy,
            )

    if action.target_id in event.failed_target_ids:
        return _failure_result(
            state,
            task,
            action,
            event,
            config,
            reason=FailureReason.TARGET_FAILURE,
            hard_constraint_violated=semantic_disallowed,
            accepted=True,
            achieved_success_probability=expected_success_probability,
            achieved_accuracy=expected_accuracy,
        )

    if action.tier is ExecutionTier.DEVICE:
        queue_start_s = max(state.now_s, source.available_at_s)
        queue_wait_s = queue_start_s - state.now_s
        compute_s = task.compute_cycles / source.spec.cpu_hz
        total_latency_s = queue_wait_s + compute_s
        required_energy_j = (
            source.spec.energy_coefficient_j_per_cycle_hz2
            * task.compute_cycles
            * source.spec.cpu_hz**2
            + source.spec.idle_power_w * queue_wait_s
        )
        success_probability = source.spec.success_probability
        accuracy_score = source.spec.accuracy_score
        if required_energy_j > source.battery_j:
            depleted = replace(source, battery_j=0.0)
            return _failure_result(
                state,
                task,
                action,
                event,
                config,
                reason=FailureReason.BATTERY_DEPLETED,
                hard_constraint_violated=semantic_disallowed,
                accepted=False,
                achieved_success_probability=success_probability,
                achieved_accuracy=accuracy_score,
                consumed_energy_j=source.battery_j,
                device=depleted,
            )
        updated_device = replace(
            source,
            battery_j=source.battery_j - required_energy_j,
            available_at_s=queue_start_s + compute_s,
        )
        deadline, reliability, accuracy, success = _constraints(
            task,
            total_latency_s=total_latency_s,
            success_probability=success_probability,
            accuracy=accuracy_score,
            hard_constraint_violated=semantic_disallowed,
            completed=True,
        )
        outcome = OffloadOutcome(
            task_id=task.task_id,
            action_label=action.label,
            tier=action.tier,
            accepted=True,
            completed=True,
            success=success,
            failure_reason=FailureReason.NONE,
            upload_time_s=0.0,
            backhaul_upload_time_s=0.0,
            queue_wait_time_s=queue_wait_s,
            compute_time_s=compute_s,
            backhaul_download_time_s=0.0,
            download_time_s=0.0,
            total_latency_s=total_latency_s,
            device_energy_j=required_energy_j,
            achieved_success_probability=success_probability,
            achieved_accuracy=accuracy_score,
            deadline_missed=deadline,
            reliability_violated=reliability,
            accuracy_violated=accuracy,
            hard_constraint_violated=semantic_disallowed,
        )
        components = _reward_components(outcome, config)
        return TransitionResult(
            next_state=_advance_state(state, action, event, device=updated_device),
            outcome=outcome,
            reward_components=components,
            reward=components.reward(config.reward_weights),
        )

    server = state.server(action.target_id)
    link = state.link(action.target_id)
    if action.target_id in event.failed_link_ids:
        consumed = min(
            source.battery_j, source.spec.tx_power_w * config.failure_detection_s
        )
        updated_device = replace(source, battery_j=source.battery_j - consumed)
        return _failure_result(
            state,
            task,
            action,
            event,
            config,
            reason=FailureReason.LINK_FAILURE,
            hard_constraint_violated=semantic_disallowed,
            accepted=True,
            achieved_success_probability=expected_success_probability,
            achieved_accuracy=server.spec.accuracy_score,
            consumed_energy_j=consumed,
            device=updated_device,
        )

    upload_s = task.input_bits / link.uplink_rate_bps
    download_s = task.output_bits / link.downlink_rate_bps
    backhaul_upload_s = 0.0
    backhaul_download_s = 0.0
    propagation_s = 0.0
    backhaul_probability = 1.0
    if server.spec.tier is ExecutionTier.CLOUD:
        if server.spec.backhaul_rate_bps is None:  # defensive; type narrowing
            raise RuntimeError("cloud server is missing backhaul_rate_bps")
        backhaul_upload_s = task.input_bits / server.spec.backhaul_rate_bps
        backhaul_download_s = task.output_bits / server.spec.backhaul_rate_bps
        propagation_s = 2.0 * server.spec.one_way_propagation_s
        backhaul_probability = server.spec.backhaul_success_probability**2

    compute_arrival_s = (
        state.now_s
        + upload_s
        + backhaul_upload_s
        + server.spec.one_way_propagation_s
    )
    queue_start_s = max(compute_arrival_s, server.available_at_s)
    queue_wait_s = queue_start_s - compute_arrival_s
    compute_s = task.compute_cycles / server.spec.cpu_hz
    total_latency_s = (
        upload_s
        + backhaul_upload_s
        + queue_wait_s
        + compute_s
        + backhaul_download_s
        + propagation_s
        + download_s
    )
    non_radio_wait_s = (
        backhaul_upload_s
        + queue_wait_s
        + compute_s
        + backhaul_download_s
        + propagation_s
    )
    required_energy_j = (
        source.spec.tx_power_w * upload_s
        + source.spec.idle_power_w * non_radio_wait_s
        + source.spec.rx_power_w * download_s
    )
    success_probability = (
        server.spec.success_probability
        * link.success_probability**2
        * backhaul_probability
    )
    accuracy_score = server.spec.accuracy_score
    if required_energy_j > source.battery_j:
        depleted = replace(source, battery_j=0.0)
        return _failure_result(
            state,
            task,
            action,
            event,
            config,
            reason=FailureReason.BATTERY_DEPLETED,
            hard_constraint_violated=semantic_disallowed,
            accepted=False,
            achieved_success_probability=success_probability,
            achieved_accuracy=accuracy_score,
            consumed_energy_j=source.battery_j,
            device=depleted,
        )

    updated_device = replace(source, battery_j=source.battery_j - required_energy_j)
    updated_server = replace(
        server,
        available_at_s=queue_start_s + compute_s,
    )
    deadline, reliability, accuracy, success = _constraints(
        task,
        total_latency_s=total_latency_s,
        success_probability=success_probability,
        accuracy=accuracy_score,
        hard_constraint_violated=semantic_disallowed,
        completed=True,
    )
    outcome = OffloadOutcome(
        task_id=task.task_id,
        action_label=action.label,
        tier=action.tier,
        accepted=True,
        completed=True,
        success=success,
        failure_reason=FailureReason.NONE,
        upload_time_s=upload_s,
        backhaul_upload_time_s=backhaul_upload_s,
        queue_wait_time_s=queue_wait_s,
        compute_time_s=compute_s,
        backhaul_download_time_s=backhaul_download_s,
        download_time_s=download_s,
        total_latency_s=total_latency_s,
        device_energy_j=required_energy_j,
        achieved_success_probability=success_probability,
        achieved_accuracy=accuracy_score,
        deadline_missed=deadline,
        reliability_violated=reliability,
        accuracy_violated=accuracy,
        hard_constraint_violated=semantic_disallowed,
    )
    components = _reward_components(outcome, config)
    return TransitionResult(
        next_state=_advance_state(
            state,
            action,
            event,
            device=updated_device,
            server=updated_server,
        ),
        outcome=outcome,
        reward_components=components,
        reward=components.reward(config.reward_weights),
    )
