"""Small deterministic fixtures shared by Phase 2 tests."""

from __future__ import annotations

from task_offloading.domain import (
    ActionTarget,
    DeviceSpec,
    DeviceState,
    ExecutionTier,
    ExogenousEvent,
    LinkMetrics,
    ServerSpec,
    ServerState,
    SystemState,
    TargetLink,
    Task,
    TaskRequirements,
)
from task_offloading.sim import Scenario


def actions() -> tuple[ActionTarget, ...]:
    return (
        ActionTarget(ExecutionTier.DEVICE, "local"),
        ActionTarget(ExecutionTier.EDGE, "edge_0"),
        ActionTarget(ExecutionTier.CLOUD, "cloud"),
    )


def state(*, edge_available_at_s: float = 4.0, edge_online: bool = True) -> SystemState:
    device = DeviceState(
        DeviceSpec(
            device_id="device_0",
            cpu_hz=2e9,
            energy_coefficient_j_per_cycle_hz2=1e-28,
            tx_power_w=2.0,
            rx_power_w=1.0,
            idle_power_w=0.5,
            success_probability=0.99,
            accuracy_score=0.80,
        ),
        battery_j=100.0,
    )
    edge = ServerState(
        ServerSpec(
            server_id="edge_0",
            tier=ExecutionTier.EDGE,
            cpu_hz=4e9,
            success_probability=0.98,
            accuracy_score=0.90,
        ),
        available_at_s=edge_available_at_s,
        is_available=edge_online,
    )
    cloud = ServerState(
        ServerSpec(
            server_id="cloud",
            tier=ExecutionTier.CLOUD,
            cpu_hz=10e9,
            success_probability=0.999,
            accuracy_score=0.99,
            backhaul_rate_bps=100e6,
            one_way_propagation_s=0.02,
            backhaul_success_probability=0.999,
        )
    )
    edge_link = TargetLink(
        "edge_0",
        LinkMetrics(
            uplink_rate_bps=10e6,
            downlink_rate_bps=10e6,
            snr_db=10.0,
            distance_m=100.0,
            path_loss_db=100.0,
            success_probability=0.97,
        ),
    )
    cloud_link = TargetLink(
        "cloud",
        LinkMetrics(
            uplink_rate_bps=20e6,
            downlink_rate_bps=20e6,
            snr_db=12.0,
            distance_m=80.0,
            path_loss_db=95.0,
            success_probability=0.995,
        ),
    )
    return SystemState(
        now_s=0.0,
        devices=(device,),
        servers=(edge, cloud),
        links=(edge_link, cloud_link),
    )


def task(
    *,
    task_id: str = "task_0",
    arrival_time_s: float = 0.0,
    input_bits: float = 20e6,
    output_bits: float = 10e6,
    compute_cycles: float = 4e9,
    requirements: TaskRequirements | None = None,
) -> Task:
    return Task(
        task_id=task_id,
        source_device_id="device_0",
        arrival_time_s=arrival_time_s,
        input_bits=input_bits,
        output_bits=output_bits,
        compute_cycles=compute_cycles,
        requirements=requirements or TaskRequirements(),
    )


def two_step_scenario() -> Scenario:
    return Scenario(
        initial_state=state(edge_available_at_s=0.0),
        tasks=(
            task(task_id="task_0", arrival_time_s=0.0),
            task(task_id="task_1", arrival_time_s=0.5),
        ),
        events=(ExogenousEvent(0.5), ExogenousEvent(1.0)),
        actions=actions(),
    )
