"""Configuration values for the deterministic transition core."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from task_offloading.domain import RewardWeights


def _require_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")


def _require_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")


@dataclass(frozen=True, slots=True)
class PhysicsConfig:
    """Frozen assumptions shared by every simulator adapter.

    Reference values only make the reward dimensionless.  In later phases
    they must be fitted on the training split and then reused unchanged.
    """

    latency_reference_s: float = 1.0
    energy_reference_j: float = 1.0
    failure_detection_s: float = 0.0
    mask_semantic_permissions: bool = True
    mask_known_unavailable_targets: bool = True
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    def __post_init__(self) -> None:
        _require_positive("latency_reference_s", self.latency_reference_s)
        _require_positive("energy_reference_j", self.energy_reference_j)
        _require_nonnegative("failure_detection_s", self.failure_detection_s)
