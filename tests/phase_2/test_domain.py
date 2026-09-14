"""Validation tests for explicit, unit-bearing domain values."""

from __future__ import annotations

import pytest

from task_offloading.domain import (
    ActionTarget,
    DeviceSpec,
    ExecutionTier,
    LinkMetrics,
    ServerSpec,
    SystemState,
    TaskRequirements,
)

from .helpers import state


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: LinkMetrics(0.0, 1.0, 0.0, 1.0, 1.0),
            "uplink_rate_bps",
        ),
        (
            lambda: DeviceSpec("d", -1.0, 0.0, 0.0, 0.0, 0.0),
            "cpu_hz",
        ),
        (
            lambda: TaskRequirements(min_accuracy=1.1),
            "min_accuracy",
        ),
        (
            lambda: ServerSpec("cloud", ExecutionTier.CLOUD, 1.0),
            "backhaul_rate_bps",
        ),
    ],
)
def test_invalid_physical_values_fail_fast(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_device_action_has_reserved_identifier() -> None:
    with pytest.raises(ValueError, match="local"):
        ActionTarget(ExecutionTier.DEVICE, "device_0")


def test_state_requires_named_link_for_every_remote_server() -> None:
    valid = state()
    with pytest.raises(ValueError, match="remote servers require link metrics"):
        SystemState(
            now_s=valid.now_s,
            devices=valid.devices,
            servers=valid.servers,
            links=valid.links[:1],
        )


def test_domain_objects_are_immutable() -> None:
    valid = state()
    with pytest.raises(AttributeError):
        valid.now_s = 10.0  # type: ignore[misc]
