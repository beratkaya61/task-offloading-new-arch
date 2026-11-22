"""Publication-facing system metrics for task-offloading experiments.

The functions in this module contain no policy or simulator logic.  They turn
one completed episode into explicitly defined measurements so every policy is
scored with the same contract.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from task_offloading.domain import ExecutionTier, OffloadOutcome


def _require_aligned(
    name: str, values: Sequence[object] | None, expected: int
) -> None:
    if values is not None and len(values) != expected:
        raise ValueError(f"{name} must contain exactly {expected} values")


def _jain(values: Sequence[float]) -> float:
    """Return Jain's fairness index, with zero activity defined as zero."""

    array = np.asarray(values, dtype=float)
    denominator = float(len(array) * np.square(array).sum())
    if denominator == 0.0:
        return 0.0
    return float(array.sum() ** 2 / denominator)


def system_metrics(
    outcomes: Sequence[OffloadOutcome],
    *,
    arrival_times_s: Sequence[float] | None = None,
    source_device_ids: Sequence[str] | None = None,
    available_actions: int | None = None,
    decision_overhead_ms: Sequence[float] | None = None,
) -> dict[str, float]:
    """Compute the registered metrics for one episode or pooled outcome set.

    ``completed_throughput_tps`` measures physically completed tasks per
    simulated second. ``successful_throughput_tps`` additionally requires all
    semantic constraints to be satisfied. ``drop_rate`` is the fraction not
    completed, while ``rejection_rate`` is the narrower fraction not accepted.

    Action entropy and Jain load fairness use the complete action count, so an
    available-but-never-used target contributes a zero. Source fairness is Jain
    fairness over per-device success rates; ``active_source_count`` must be read
    with it because a one-device episode has a mechanically perfect value.
    """

    count = len(outcomes)
    if count == 0:
        raise ValueError("outcomes must not be empty")
    _require_aligned("arrival_times_s", arrival_times_s, count)
    _require_aligned("source_device_ids", source_device_ids, count)
    _require_aligned("decision_overhead_ms", decision_overhead_ms, count)
    if available_actions is not None and available_actions < 1:
        raise ValueError("available_actions must be positive")

    latency = np.asarray([item.total_latency_s for item in outcomes], dtype=float)
    queue_wait = np.asarray(
        [item.queue_wait_time_s for item in outcomes], dtype=float
    )
    energy = np.asarray([item.device_energy_j for item in outcomes], dtype=float)
    completed = np.asarray([item.completed for item in outcomes], dtype=bool)
    success = np.asarray([item.success for item in outcomes], dtype=bool)
    accepted = np.asarray([item.accepted for item in outcomes], dtype=bool)
    tiers = [item.tier for item in outcomes]

    energy_total = float(energy.sum())
    success_count = int(success.sum())
    per_success = energy_total / success_count if success_count else math.nan

    action_counts: dict[str, int] = defaultdict(int)
    for item in outcomes:
        action_counts[item.action_label] += 1
    action_slots = max(available_actions or len(action_counts), len(action_counts))
    action_probabilities = np.asarray(list(action_counts.values()), dtype=float) / count
    entropy = float(-np.sum(action_probabilities * np.log(action_probabilities)))
    normalized_entropy = entropy / math.log(action_slots) if action_slots > 1 else 0.0
    padded_action_counts = list(action_counts.values()) + [0] * (
        action_slots - len(action_counts)
    )

    result = {
        "decisions": float(count),
        "success_rate": float(success.mean()),
        "completion_rate": float(completed.mean()),
        "drop_rate": float((~completed).mean()),
        "rejection_rate": float((~accepted).mean()),
        "hard_violation_rate": float(
            np.mean([item.hard_constraint_violated for item in outcomes])
        ),
        "deadline_miss_rate": float(
            np.mean([item.deadline_missed for item in outcomes])
        ),
        "reliability_violation_rate": float(
            np.mean([item.reliability_violated for item in outcomes])
        ),
        "accuracy_violation_rate": float(
            np.mean([item.accuracy_violated for item in outcomes])
        ),
        # Kept for backward compatibility with the v1 result schema.
        "failure_rate": float((~completed).mean()),
        "latency_mean_ms": float(1000 * latency.mean()),
        "latency_p50_ms": float(1000 * np.percentile(latency, 50)),
        "latency_p95_ms": float(1000 * np.percentile(latency, 95)),
        "latency_p99_ms": float(1000 * np.percentile(latency, 99)),
        "queue_wait_mean_ms": float(1000 * queue_wait.mean()),
        "queue_wait_p95_ms": float(1000 * np.percentile(queue_wait, 95)),
        "device_energy_mean_mj": float(1000 * energy.mean()),
        "device_energy_episode_total_j": energy_total,
        "device_energy_per_success_mj": float(1000 * per_success),
        "share_local": float(np.mean([t is ExecutionTier.DEVICE for t in tiers])),
        "share_edge": float(np.mean([t is ExecutionTier.EDGE for t in tiers])),
        "share_cloud": float(np.mean([t is ExecutionTier.CLOUD for t in tiers])),
        "action_entropy_normalized": float(normalized_entropy),
        "action_max_share": float(max(action_counts.values()) / count),
        "action_load_jain_fairness": _jain(padded_action_counts),
        "action_unique_count": float(len(action_counts)),
    }

    if arrival_times_s is not None:
        arrivals = np.asarray(arrival_times_s, dtype=float)
        finish = arrivals + latency
        horizon = max(float(finish.max() - arrivals.min()), np.finfo(float).eps)
        result["completed_throughput_tps"] = float(completed.sum() / horizon)
        result["successful_throughput_tps"] = float(success.sum() / horizon)
        result["simulated_horizon_s"] = horizon
    else:
        result["completed_throughput_tps"] = math.nan
        result["successful_throughput_tps"] = math.nan
        result["simulated_horizon_s"] = math.nan

    if source_device_ids is not None:
        by_source: dict[str, list[bool]] = defaultdict(list)
        for source, successful in zip(source_device_ids, success, strict=True):
            by_source[source].append(bool(successful))
        source_rates = [float(np.mean(values)) for values in by_source.values()]
        result["source_success_jain_fairness"] = _jain(source_rates)
        result["active_source_count"] = float(len(source_rates))
    else:
        result["source_success_jain_fairness"] = math.nan
        result["active_source_count"] = math.nan

    if decision_overhead_ms is not None:
        overhead = np.asarray(decision_overhead_ms, dtype=float)
        if np.any(overhead < 0):
            raise ValueError("decision_overhead_ms must be nonnegative")
        result["decision_overhead_mean_ms"] = float(overhead.mean())
        result["decision_overhead_p95_ms"] = float(np.percentile(overhead, 95))
    else:
        result["decision_overhead_mean_ms"] = math.nan
        result["decision_overhead_p95_ms"] = math.nan
    return result


__all__ = ["system_metrics"]
