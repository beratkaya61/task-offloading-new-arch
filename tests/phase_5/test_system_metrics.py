from __future__ import annotations

import math
from dataclasses import replace

import pytest

from task_offloading.domain import ExecutionTier, FailureReason, OffloadOutcome
from task_offloading.system_metrics import system_metrics


def _outcome(
    label: str,
    *,
    latency: float,
    queue: float,
    energy: float,
    completed: bool = True,
    success: bool = True,
    accepted: bool = True,
) -> OffloadOutcome:
    return OffloadOutcome(
        task_id=label,
        action_label=label,
        tier=ExecutionTier.DEVICE,
        accepted=accepted,
        completed=completed,
        success=success,
        failure_reason=FailureReason.NONE,
        upload_time_s=0.0,
        backhaul_upload_time_s=0.0,
        queue_wait_time_s=queue,
        compute_time_s=latency - queue,
        backhaul_download_time_s=0.0,
        download_time_s=0.0,
        total_latency_s=latency,
        device_energy_j=energy,
        achieved_success_probability=1.0,
        achieved_accuracy=1.0,
        deadline_missed=False,
        reliability_violated=False,
        accuracy_violated=False,
        hard_constraint_violated=not success,
    )


def test_system_metrics_cover_registered_tail_and_resource_measures() -> None:
    outcomes = [
        _outcome("local", latency=0.1, queue=0.02, energy=0.1),
        _outcome("local", latency=0.2, queue=0.04, energy=0.2),
        _outcome(
            "edge_0",
            latency=0.3,
            queue=0.06,
            energy=0.3,
            completed=False,
            success=False,
            accepted=False,
        ),
        _outcome("edge_0", latency=0.4, queue=0.08, energy=0.4),
    ]
    metrics = system_metrics(
        outcomes,
        arrival_times_s=[0.0, 0.1, 0.2, 0.3],
        source_device_ids=["a", "a", "b", "b"],
        available_actions=4,
        decision_overhead_ms=[1.0, 2.0, 3.0, 4.0],
    )

    assert metrics["latency_p50_ms"] == pytest.approx(250.0)
    assert metrics["latency_p99_ms"] == pytest.approx(397.0)
    assert metrics["queue_wait_mean_ms"] == pytest.approx(50.0)
    assert metrics["drop_rate"] == pytest.approx(0.25)
    assert metrics["rejection_rate"] == pytest.approx(0.25)
    assert metrics["device_energy_episode_total_j"] == pytest.approx(1.0)
    assert metrics["device_energy_per_success_mj"] == pytest.approx(1000 / 3)
    assert metrics["completed_throughput_tps"] == pytest.approx(3 / 0.7)
    assert metrics["successful_throughput_tps"] == pytest.approx(3 / 0.7)
    assert metrics["action_entropy_normalized"] == pytest.approx(0.5)
    assert metrics["action_max_share"] == pytest.approx(0.5)
    assert metrics["action_load_jain_fairness"] == pytest.approx(0.5)
    assert metrics["source_success_jain_fairness"] == pytest.approx(0.9)
    assert metrics["decision_overhead_mean_ms"] == pytest.approx(2.5)
    assert metrics["decision_overhead_p95_ms"] == pytest.approx(3.85)


def test_per_success_energy_is_undefined_when_no_task_succeeds() -> None:
    failed = replace(
        _outcome("local", latency=0.1, queue=0.0, energy=0.2),
        success=False,
        hard_constraint_violated=True,
    )
    metrics = system_metrics([failed])
    assert math.isnan(metrics["device_energy_per_success_mj"])


def test_system_metrics_reject_misaligned_context() -> None:
    outcome = _outcome("local", latency=0.1, queue=0.0, energy=0.1)
    with pytest.raises(ValueError, match="arrival_times_s"):
        system_metrics([outcome], arrival_times_s=[])
