"""Property tests for physical monotonicity and random-stream contracts."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from task_offloading.sim import RadioConfig, RngFactory, build_link_metrics


@given(
    near=st.floats(min_value=1.0, max_value=500.0, allow_nan=False),
    extra=st.floats(min_value=0.01, max_value=500.0, allow_nan=False),
)
def test_rate_decreases_as_distance_increases(near: float, extra: float) -> None:
    radio = RadioConfig()
    near_link = build_link_metrics(near, radio)
    far_link = build_link_metrics(near + extra, radio)
    assert far_link.path_loss_db > near_link.path_loss_db
    assert far_link.uplink_rate_bps < near_link.uplink_rate_bps


@given(
    low_power=st.floats(min_value=-20.0, max_value=20.0, allow_nan=False),
    increment=st.floats(min_value=0.01, max_value=20.0, allow_nan=False),
)
def test_rate_increases_with_transmit_power(low_power: float, increment: float) -> None:
    low = build_link_metrics(100.0, RadioConfig(uplink_tx_power_dbm=low_power))
    high = build_link_metrics(
        100.0, RadioConfig(uplink_tx_power_dbm=low_power + increment)
    )
    assert high.uplink_rate_bps > low.uplink_rate_bps


@given(
    bandwidth=st.floats(min_value=1e5, max_value=50e6, allow_nan=False),
    factor=st.floats(min_value=1.01, max_value=4.0, allow_nan=False),
)
def test_capacity_is_monotonic_in_bandwidth(bandwidth: float, factor: float) -> None:
    low = build_link_metrics(50.0, RadioConfig(bandwidth_hz=bandwidth))
    high = build_link_metrics(50.0, RadioConfig(bandwidth_hz=bandwidth * factor))
    assert high.uplink_rate_bps > low.uplink_rate_bps


def test_named_rng_stream_is_order_independent_and_reproducible() -> None:
    first_factory = RngFactory(42)
    first_arrivals = first_factory.stream("arrivals").random(8)
    first_factory.stream("failures").random(100)
    second_factory = RngFactory(42)
    second_factory.stream("mobility").random(100)
    second_arrivals = second_factory.stream("arrivals").random(8)
    np.testing.assert_array_equal(first_arrivals, second_arrivals)


def test_named_rng_streams_are_separated() -> None:
    factory = RngFactory(42)
    assert not np.array_equal(
        factory.stream("arrivals").random(8), factory.stream("failures").random(8)
    )
