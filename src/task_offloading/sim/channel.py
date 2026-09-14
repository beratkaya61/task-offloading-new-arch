"""Small, explicit wireless-channel helpers used to build link metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

from task_offloading.domain import LinkMetrics

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
THERMAL_NOISE_DENSITY_DBM_PER_HZ = -174.0


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")


@dataclass(frozen=True, slots=True)
class RadioConfig:
    """Parameters for a log-distance path-loss and Shannon-rate model."""

    carrier_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 20e6
    path_loss_exponent: float = 3.0
    reference_distance_m: float = 1.0
    noise_figure_db: float = 7.0
    uplink_tx_power_dbm: float = 23.0
    downlink_tx_power_dbm: float = 30.0
    shadowing_db: float = 0.0
    link_success_probability: float = 1.0

    def __post_init__(self) -> None:
        _positive("carrier_frequency_hz", self.carrier_frequency_hz)
        _positive("bandwidth_hz", self.bandwidth_hz)
        _positive("path_loss_exponent", self.path_loss_exponent)
        _positive("reference_distance_m", self.reference_distance_m)
        for name in (
            "noise_figure_db",
            "uplink_tx_power_dbm",
            "downlink_tx_power_dbm",
            "shadowing_db",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.link_success_probability <= 1:
            raise ValueError("link_success_probability must be in [0, 1]")


def path_loss_db(distance_m: float, radio: RadioConfig) -> float:
    """Return log-distance path loss with free-space loss at the reference."""

    _positive("distance_m", distance_m)
    if distance_m < radio.reference_distance_m:
        raise ValueError("distance_m cannot be smaller than reference_distance_m")
    reference_loss_db = 20.0 * math.log10(
        4.0
        * math.pi
        * radio.carrier_frequency_hz
        * radio.reference_distance_m
        / SPEED_OF_LIGHT_M_PER_S
    )
    return (
        reference_loss_db
        + 10.0
        * radio.path_loss_exponent
        * math.log10(distance_m / radio.reference_distance_m)
        + radio.shadowing_db
    )


def _rate_bps(
    tx_power_dbm: float, loss_db: float, radio: RadioConfig
) -> tuple[float, float]:
    noise_dbm = (
        THERMAL_NOISE_DENSITY_DBM_PER_HZ
        + 10.0 * math.log10(radio.bandwidth_hz)
        + radio.noise_figure_db
    )
    snr_db = tx_power_dbm - loss_db - noise_dbm
    snr_linear = 10.0 ** (snr_db / 10.0)
    return radio.bandwidth_hz * math.log2(1.0 + snr_linear), snr_db


def build_link_metrics(distance_m: float, radio: RadioConfig) -> LinkMetrics:
    """Build named uplink/downlink values so tuple ordering cannot drift."""

    loss_db = path_loss_db(distance_m, radio)
    uplink_rate_bps, uplink_snr_db = _rate_bps(
        radio.uplink_tx_power_dbm, loss_db, radio
    )
    downlink_rate_bps, _ = _rate_bps(radio.downlink_tx_power_dbm, loss_db, radio)
    return LinkMetrics(
        uplink_rate_bps=uplink_rate_bps,
        downlink_rate_bps=downlink_rate_bps,
        snr_db=uplink_snr_db,
        distance_m=distance_m,
        path_loss_db=loss_db,
        success_probability=radio.link_success_probability,
    )
