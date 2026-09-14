"""Typed, immutable domain values for the single physics core.

Every numeric field carries its SI unit in its name.  The simulator does not
accept ambiguous tuples such as ``(rate, x)``; link and outcome values are
named records so that distance cannot accidentally be consumed as SNR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _require_nonnegative(name: str, value: float) -> None:
    _require_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")


def _require_positive(name: str, value: float) -> None:
    _require_finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")


def _require_probability(name: str, value: float) -> None:
    _require_finite(name, value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


class ExecutionTier(StrEnum):
    """Execution locations visible to semantic requirements and the policy."""

    DEVICE = "device"
    EDGE = "edge"
    CLOUD = "cloud"


class FailureReason(StrEnum):
    """Mutually exclusive terminal reason for one attempted task."""

    NONE = "none"
    INVALID_ACTION = "invalid_action"
    TARGET_UNAVAILABLE = "target_unavailable"
    LINK_FAILURE = "link_failure"
    TARGET_FAILURE = "target_failure"
    BATTERY_DEPLETED = "battery_depleted"


@dataclass(frozen=True, slots=True)
class ActionTarget:
    """A policy action resolved to an execution tier and target identifier."""

    tier: ExecutionTier
    target_id: str

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("target_id must not be empty")
        if self.tier is ExecutionTier.DEVICE and self.target_id != "local":
            raise ValueError("device action must use the reserved target_id 'local'")

    @property
    def label(self) -> str:
        return "local" if self.tier is ExecutionTier.DEVICE else self.target_id


@dataclass(frozen=True, slots=True)
class TaskRequirements:
    """Human/LLM semantic requirements already converted to typed values.

    ``None`` means that the text did not provide a usable constraint.  Unknown
    execution permissions are represented by allowing every tier; unknown
    semantics never create a hard action mask.
    """

    allowed_tiers: frozenset[ExecutionTier] = field(
        default_factory=lambda: frozenset(ExecutionTier)
    )
    max_latency_s: float | None = None
    min_success_probability: float | None = None
    min_accuracy: float | None = None
    semantic_confidence: float = 1.0
    unknown_fields: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not self.allowed_tiers:
            raise ValueError("allowed_tiers must contain at least one tier")
        if self.max_latency_s is not None:
            _require_positive("max_latency_s", self.max_latency_s)
        if self.min_success_probability is not None:
            _require_probability(
                "min_success_probability", self.min_success_probability
            )
        if self.min_accuracy is not None:
            _require_probability("min_accuracy", self.min_accuracy)
        _require_probability("semantic_confidence", self.semantic_confidence)


@dataclass(frozen=True, slots=True)
class Task:
    """One indivisible task arriving at a decision epoch."""

    task_id: str
    source_device_id: str
    arrival_time_s: float
    input_bits: float
    output_bits: float
    compute_cycles: float
    requirements: TaskRequirements = field(default_factory=TaskRequirements)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must not be empty")
        if not self.source_device_id:
            raise ValueError("source_device_id must not be empty")
        _require_nonnegative("arrival_time_s", self.arrival_time_s)
        _require_nonnegative("input_bits", self.input_bits)
        _require_nonnegative("output_bits", self.output_bits)
        _require_nonnegative("compute_cycles", self.compute_cycles)


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """Static device parameters.

    ``energy_coefficient_j_per_cycle_hz2`` is the effective capacitance
    coefficient in the common dynamic CPU energy model ``kappa * C * f^2``.
    """

    device_id: str
    cpu_hz: float
    energy_coefficient_j_per_cycle_hz2: float
    tx_power_w: float
    rx_power_w: float
    idle_power_w: float
    success_probability: float = 1.0
    accuracy_score: float = 1.0

    def __post_init__(self) -> None:
        if not self.device_id:
            raise ValueError("device_id must not be empty")
        _require_positive("cpu_hz", self.cpu_hz)
        _require_nonnegative(
            "energy_coefficient_j_per_cycle_hz2",
            self.energy_coefficient_j_per_cycle_hz2,
        )
        _require_nonnegative("tx_power_w", self.tx_power_w)
        _require_nonnegative("rx_power_w", self.rx_power_w)
        _require_nonnegative("idle_power_w", self.idle_power_w)
        _require_probability("success_probability", self.success_probability)
        _require_probability("accuracy_score", self.accuracy_score)


@dataclass(frozen=True, slots=True)
class DeviceState:
    """Dynamic device state at an absolute simulation time."""

    spec: DeviceSpec
    battery_j: float
    available_at_s: float = 0.0

    def __post_init__(self) -> None:
        _require_nonnegative("battery_j", self.battery_j)
        _require_nonnegative("available_at_s", self.available_at_s)


@dataclass(frozen=True, slots=True)
class ServerSpec:
    """Static edge/cloud compute and backhaul parameters."""

    server_id: str
    tier: ExecutionTier
    cpu_hz: float
    success_probability: float = 1.0
    accuracy_score: float = 1.0
    backhaul_rate_bps: float | None = None
    one_way_propagation_s: float = 0.0
    backhaul_success_probability: float = 1.0

    def __post_init__(self) -> None:
        if not self.server_id:
            raise ValueError("server_id must not be empty")
        if self.tier is ExecutionTier.DEVICE:
            raise ValueError("ServerSpec tier must be edge or cloud")
        _require_positive("cpu_hz", self.cpu_hz)
        _require_probability("success_probability", self.success_probability)
        _require_probability("accuracy_score", self.accuracy_score)
        _require_nonnegative("one_way_propagation_s", self.one_way_propagation_s)
        _require_probability(
            "backhaul_success_probability", self.backhaul_success_probability
        )
        if self.backhaul_rate_bps is not None:
            _require_positive("backhaul_rate_bps", self.backhaul_rate_bps)
        if self.tier is ExecutionTier.CLOUD and self.backhaul_rate_bps is None:
            raise ValueError("cloud server requires a positive backhaul_rate_bps")


@dataclass(frozen=True, slots=True)
class ServerState:
    """Dynamic FCFS server state represented by its next free time."""

    spec: ServerSpec
    available_at_s: float = 0.0
    is_available: bool = True

    def __post_init__(self) -> None:
        _require_nonnegative("available_at_s", self.available_at_s)


@dataclass(frozen=True, slots=True)
class LinkMetrics:
    """Named link values; all rates and probabilities are explicit."""

    uplink_rate_bps: float
    downlink_rate_bps: float
    snr_db: float
    distance_m: float
    path_loss_db: float
    success_probability: float = 1.0

    def __post_init__(self) -> None:
        _require_positive("uplink_rate_bps", self.uplink_rate_bps)
        _require_positive("downlink_rate_bps", self.downlink_rate_bps)
        _require_finite("snr_db", self.snr_db)
        _require_positive("distance_m", self.distance_m)
        _require_finite("path_loss_db", self.path_loss_db)
        _require_probability("success_probability", self.success_probability)


@dataclass(frozen=True, slots=True)
class TargetLink:
    """Last-mile metrics from the current device to one remote target."""

    target_id: str
    metrics: LinkMetrics

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("target_id must not be empty")


@dataclass(frozen=True, slots=True)
class SystemState:
    """Complete observable physical state at one decision epoch."""

    now_s: float
    devices: tuple[DeviceState, ...]
    servers: tuple[ServerState, ...]
    links: tuple[TargetLink, ...]
    previous_action_label: str | None = None

    def __post_init__(self) -> None:
        _require_nonnegative("now_s", self.now_s)
        device_ids = [item.spec.device_id for item in self.devices]
        server_ids = [item.spec.server_id for item in self.servers]
        link_ids = [item.target_id for item in self.links]
        if not self.devices:
            raise ValueError("state must contain at least one device")
        if len(device_ids) != len(set(device_ids)):
            raise ValueError("device identifiers must be unique")
        if len(server_ids) != len(set(server_ids)):
            raise ValueError("server identifiers must be unique")
        if len(link_ids) != len(set(link_ids)):
            raise ValueError("link target identifiers must be unique")
        missing_links = set(server_ids) - set(link_ids)
        if missing_links:
            missing = ", ".join(sorted(missing_links))
            raise ValueError(f"remote servers require link metrics: {missing}")

    def device(self, device_id: str) -> DeviceState:
        for item in self.devices:
            if item.spec.device_id == device_id:
                return item
        raise KeyError(f"unknown device: {device_id}")

    def server(self, server_id: str) -> ServerState:
        for item in self.servers:
            if item.spec.server_id == server_id:
                return item
        raise KeyError(f"unknown server: {server_id}")

    def link(self, target_id: str) -> LinkMetrics:
        for item in self.links:
            if item.target_id == target_id:
                return item.metrics
        raise KeyError(f"no link metrics for target: {target_id}")


@dataclass(frozen=True, slots=True)
class ExogenousEvent:
    """Pre-sampled randomness applied to one transition.

    Failures are surprises to the policy.  Known unavailability belongs in
    ``SystemState`` and therefore participates in the action mask.
    """

    next_interarrival_s: float
    failed_target_ids: frozenset[str] = field(default_factory=frozenset)
    failed_link_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        _require_nonnegative("next_interarrival_s", self.next_interarrival_s)


@dataclass(frozen=True, slots=True)
class RewardWeights:
    latency_norm: float = 1.0
    energy_norm: float = 1.0
    deadline_miss: float = 5.0
    soft_constraint_violation: float = 5.0
    failure_or_drop: float = 10.0
    hard_constraint_violation: float = 20.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _require_nonnegative(name, float(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class RewardComponents:
    latency_norm: float
    energy_norm: float
    deadline_miss: float
    soft_constraint_violation: float
    failure_or_drop: float
    hard_constraint_violation: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _require_nonnegative(name, float(getattr(self, name)))

    def total_cost(self, weights: RewardWeights) -> float:
        return (
            weights.latency_norm * self.latency_norm
            + weights.energy_norm * self.energy_norm
            + weights.deadline_miss * self.deadline_miss
            + weights.soft_constraint_violation * self.soft_constraint_violation
            + weights.failure_or_drop * self.failure_or_drop
            + weights.hard_constraint_violation * self.hard_constraint_violation
        )

    def reward(self, weights: RewardWeights) -> float:
        return -self.total_cost(weights)


@dataclass(frozen=True, slots=True)
class OffloadOutcome:
    """Physical and constraint outcome for one selected action."""

    task_id: str
    action_label: str
    tier: ExecutionTier
    accepted: bool
    completed: bool
    success: bool
    failure_reason: FailureReason
    upload_time_s: float
    backhaul_upload_time_s: float
    queue_wait_time_s: float
    compute_time_s: float
    backhaul_download_time_s: float
    download_time_s: float
    total_latency_s: float
    device_energy_j: float
    achieved_success_probability: float
    achieved_accuracy: float
    deadline_missed: bool
    reliability_violated: bool
    accuracy_violated: bool
    hard_constraint_violated: bool

    def __post_init__(self) -> None:
        for name in (
            "upload_time_s",
            "backhaul_upload_time_s",
            "queue_wait_time_s",
            "compute_time_s",
            "backhaul_download_time_s",
            "download_time_s",
            "total_latency_s",
            "device_energy_j",
        ):
            _require_nonnegative(name, float(getattr(self, name)))
        _require_probability(
            "achieved_success_probability", self.achieved_success_probability
        )
        _require_probability("achieved_accuracy", self.achieved_accuracy)


@dataclass(frozen=True, slots=True)
class TransitionResult:
    next_state: SystemState
    outcome: OffloadOutcome
    reward_components: RewardComponents
    reward: float

    def __post_init__(self) -> None:
        _require_finite("reward", self.reward)
