"""Trace-driven *hybrid* scenario builder (UCI-859 + BUPT + EUA + semantics).

Provenance of every physical quantity (reported as ``trace_driven_hybrid``,
never as a fully real dataset):

* **Compute heterogeneity (measured, UCI-859).**  The four UCI hosts executed
  the same image-recognition workload.  The Raspberry Pi 4B is the local IoT
  device; MacBookPro1, Ubuntu VM and MacBookPro2 are ``edge_0..2``.  Relative
  speeds are the ratios of their measured mean turnaround times, and the
  per-task workload variability is drawn from the Raspberry Pi's measured
  turnaround distribution.
* **Task arrivals and payload sizes (observed, BUPT).**  Consecutive accepted
  rows of one city-day file give the inter-arrival pattern (time-compressed to
  a target mean, preserving burstiness) and the uplink/downlink bits.
  Chronological split: earlier days train, later days test.
* **Topology (observed, EUA).**  A real Melbourne user location and its three
  nearest real edge sites give the radio distances.
* **Semantics (controlled corpus).**  Each physical task is paired with one
  task text; its *true* requirements come from the reference labels and its
  *believed* requirements from a semantic extractor.

Calibration assumptions (explicit, not measured): the absolute workload
scale, cloud speed/propagation/backhaul, per-tier success probabilities and
accuracy scores, and the energy constants listed in ``HybridConfig``.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import fmean
from typing import Any
from zipfile import ZipFile

import numpy as np

from task_offloading.data.bupt import BuptParseError, parse_bupt_line
from task_offloading.data.eua import EuaEntityKind, iter_eua_archive
from task_offloading.data.uci_mec import iter_uci_mec_archive
from task_offloading.domain import (
    ActionTarget,
    DeviceSpec,
    DeviceState,
    ExecutionTier,
    ExogenousEvent,
    ServerSpec,
    ServerState,
    SystemState,
    TargetLink,
    Task,
    TaskRequirements,
)
from task_offloading.sim.channel import RadioConfig, build_link_metrics
from task_offloading.sim.scenario import Scenario

DEVICE_HOST = "raspberry_pi_4b"
EDGE_HOSTS = ("macbook_pro_1", "ubuntu_virtualbox_vm", "macbook_pro_2")
ACTIONS = (
    ActionTarget(ExecutionTier.DEVICE, "local"),
    ActionTarget(ExecutionTier.EDGE, "edge_0"),
    ActionTarget(ExecutionTier.EDGE, "edge_1"),
    ActionTarget(ExecutionTier.EDGE, "edge_2"),
    ActionTarget(ExecutionTier.CLOUD, "cloud"),
)

# Requirement mapping for qualitative classes (seconds / probabilities).
LATENCY_CLASS_S = {
    "hard_real_time": 0.05,
    "real_time": 0.2,
    "interactive": 1.0,
    "relaxed": 4.0,
}
RELIABILITY_TIER_P = {"mission_critical": 0.999, "high": 0.99, "standard": 0.95}
ACCURACY_TIER_SCORE = {"high": 0.95, "standard": 0.9}
CONSERVATIVE_PRIVACY = frozenset({"sensitive", "restricted"})


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """Calibration assumptions shared by every condition (frozen per study)."""

    device_cpu_hz: float = 1.5e9  # Raspberry Pi 4B nominal clock
    workload_scale: float = 0.25  # mean local service time ~= 0.25 s
    cloud_speedup_vs_fastest_edge: float = 2.0
    cloud_one_way_propagation_s: float = 0.025
    cloud_backhaul_bps: float = 1e9
    target_mean_interarrival_s: float = 0.03
    episode_tasks: int = 64
    energy_coefficient: float = 1e-28
    tx_power_w: float = 0.5
    rx_power_w: float = 0.3
    idle_power_w: float = 0.05
    battery_j: float = 5_000.0
    device_success: float = 0.999
    edge_success: tuple[float, float, float] = (0.995, 0.99, 0.995)
    cloud_success: float = 0.9999
    cloud_backhaul_success: float = 0.9999
    link_success: float = 0.9999
    device_accuracy: float = 0.85
    edge_accuracy: tuple[float, float, float] = (0.95, 0.93, 0.95)
    cloud_accuracy: float = 0.98
    radio: RadioConfig = field(
        default_factory=lambda: RadioConfig(
            bandwidth_hz=10e6, link_success_probability=0.9999
        )
    )
    # NEP layer (optional): percentile tables from nep_quantiles_v1.json.
    nep_percentiles: tuple[float, ...] | None = None
    nep_edge_load_quantiles: tuple[float, ...] | None = None
    nep_rtt_ms_quantiles: tuple[float, ...] | None = None
    max_edge_background_load: float = 0.9


def load_nep_layer(path: Path) -> dict[str, tuple[float, ...]]:
    """Keyword arguments for ``HybridConfig`` from the NEP quantile profile."""

    profile = json.loads(path.read_text(encoding="utf-8"))
    return {
        "nep_percentiles": tuple(float(v) for v in profile["percentiles"]),
        "nep_edge_load_quantiles": tuple(
            float(v) for v in profile["edge_site_hour_cpu_load"]["quantiles"]
        ),
        "nep_rtt_ms_quantiles": tuple(
            float(v) for v in profile["inter_site_rtt_ms"]["quantiles"]
        ),
    }


def _inverse_cdf(
    rng: np.random.Generator,
    percentiles: tuple[float, ...],
    quantiles: tuple[float, ...],
    size: int,
) -> np.ndarray:
    draws = rng.uniform(percentiles[0], percentiles[-1], size)
    return np.interp(draws, percentiles, quantiles)


@dataclass(frozen=True, slots=True)
class WorkloadWindow:
    source: str
    interarrival_s: tuple[float, ...]
    input_bits: tuple[float, ...]
    output_bits: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PhysicalPools:
    device_turnaround_s: np.ndarray
    host_mean_turnaround_s: Mapping[str, float]
    train_windows: tuple[WorkloadWindow, ...]
    test_windows: tuple[WorkloadWindow, ...]
    user_edge_distances_m: np.ndarray  # shape (users, 3), nearest first


# --------------------------------------------------------------------- loaders
def load_uci(path: Path) -> tuple[np.ndarray, dict[str, float]]:
    samples: dict[str, list[float]] = {}
    for record in iter_uci_mec_archive(path):
        samples.setdefault(record.server_id, []).append(record.turnaround_time_s)
    means = {host: fmean(values) for host, values in samples.items()}
    return np.asarray(samples[DEVICE_HOST], dtype=float), means


def load_bupt_windows(
    path: Path,
    *,
    window: int,
    key: bytes,
    test_from_day: int = 24,
) -> tuple[tuple[WorkloadWindow, ...], tuple[WorkloadWindow, ...]]:
    """Cut each city-day file into consecutive windows; split by calendar day."""

    train: list[WorkloadWindow] = []
    test: list[WorkloadWindow] = []
    with ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".csv"):
                continue
            relative = name.split("/", 1)[1]
            rows = []
            for number, raw in enumerate(
                archive.read(name).decode("utf-8", "replace").splitlines(), 1
            ):
                try:
                    rows.append(
                        parse_bupt_line(
                            raw,
                            source_path=relative,
                            source_row_number=number,
                            pseudonymization_key=key,
                        )
                    )
                except BuptParseError:
                    continue
            if len(rows) < window + 1:
                continue
            rows.sort(key=lambda item: item.arrival_time)
            day = int(name[-6:-4])
            target = test if day >= test_from_day else train
            for start in range(0, len(rows) - window, window):
                chunk = rows[start : start + window + 1]
                gaps = [
                    max(
                        (later.arrival_time - earlier.arrival_time).total_seconds(),
                        0.0,
                    )
                    for earlier, later in itertools.pairwise(chunk)
                ]
                target.append(
                    WorkloadWindow(
                        source=f"{relative}#{start}",
                        interarrival_s=tuple(gaps),
                        input_bits=tuple(float(r.input_bits) for r in chunk[:-1]),
                        output_bits=tuple(float(r.output_bits) for r in chunk[:-1]),
                    )
                )
    return tuple(train), tuple(test)


def _haversine_m(
    lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    radius = 6_371_000.0
    p1, p2 = math.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - math.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return np.asarray(2 * radius * np.arcsin(np.sqrt(a)), dtype=float)


def load_eua_distances(path: Path) -> np.ndarray:
    edges: list[tuple[float, float]] = []
    users: list[tuple[float, float]] = []
    for record in iter_eua_archive(path):
        target = edges if record.entity_kind is EuaEntityKind.EDGE_SERVER else users
        target.append((record.latitude_deg, record.longitude_deg))
    edge_lat = np.asarray([item[0] for item in edges])
    edge_lon = np.asarray([item[1] for item in edges])
    rows = []
    for lat, lon in users:
        distances = np.sort(_haversine_m(lat, lon, edge_lat, edge_lon))[:3]
        if distances[0] >= 10.0 and distances[2] <= 3_000.0:
            rows.append(np.maximum(distances, 10.0))
    return np.asarray(rows)


def load_physical_pools(
    *, uci_zip: Path, bupt_zip: Path, eua_zip: Path, window: int, key: bytes
) -> PhysicalPools:
    device_samples, means = load_uci(uci_zip)
    train, test = load_bupt_windows(bupt_zip, window=window, key=key)
    return PhysicalPools(
        device_turnaround_s=device_samples,
        host_mean_turnaround_s=means,
        train_windows=train,
        test_windows=test,
        user_edge_distances_m=load_eua_distances(eua_zip),
    )


# ------------------------------------------------------------------ semantics
def annotation_to_requirements(
    annotation: Mapping[str, Any],
    *,
    conservative: bool = False,
    confidence_floor: float = 0.6,
) -> TaskRequirements:
    """Typed requirements from an 8-field semantic annotation.

    ``conservative`` is the uncertainty-aware fallback: when the execution
    policy is not explicit and the data is sensitive/restricted, or when the
    policy confidence is below ``confidence_floor``, the public cloud is
    masked.  It never *adds* permissions.
    """

    policy = annotation["execution_policy"]
    allowed = set(ExecutionTier)
    if policy.get("status") in {"explicit", "partial"}:
        for tier, key in (
            (ExecutionTier.DEVICE, "device"),
            (ExecutionTier.EDGE, "edge"),
            (ExecutionTier.CLOUD, "cloud"),
        ):
            if policy.get(key) == "forbidden":
                allowed.discard(tier)
    if conservative:
        privacy = annotation["privacy"]
        policy_explicit = policy.get("status") in {"explicit", "partial"}
        risky_data = privacy.get("data_class") in CONSERVATIVE_PRIVACY or (
            privacy.get("status") == "ambiguous"
        )
        unsure = float(policy.get("confidence", 1.0)) < confidence_floor
        if (
            (risky_data and not policy_explicit)
            or unsure
            or (policy.get("status") == "ambiguous")
        ):
            allowed.discard(ExecutionTier.CLOUD)
    if not allowed:
        allowed = {ExecutionTier.DEVICE}

    latency = annotation["latency"]
    max_latency_s: float | None = None
    if latency.get("max_ms") is not None:
        max_latency_s = float(latency["max_ms"]) / 1000.0
    elif latency.get("class") in LATENCY_CLASS_S:
        max_latency_s = LATENCY_CLASS_S[str(latency["class"])]

    reliability = annotation["reliability"]
    min_success: float | None = None
    if reliability.get("min_success_probability") is not None:
        min_success = float(reliability["min_success_probability"])
    elif reliability.get("tier") in RELIABILITY_TIER_P:
        min_success = RELIABILITY_TIER_P[str(reliability["tier"])]

    accuracy = annotation["accuracy"]
    min_accuracy: float | None = None
    if accuracy.get("min_score") is not None:
        min_accuracy = float(accuracy["min_score"])
    elif accuracy.get("tier") in ACCURACY_TIER_SCORE:
        min_accuracy = ACCURACY_TIER_SCORE[str(accuracy["tier"])]

    confidence = annotation.get("overall_confidence")
    unknown = frozenset(
        name
        for name in (
            "latency",
            "privacy",
            "execution_policy",
            "reliability",
            "accuracy",
        )
        if annotation[name].get("status") in {"not_stated", "ambiguous"}
    )
    return TaskRequirements(
        allowed_tiers=frozenset(allowed),
        max_latency_s=max_latency_s,
        min_success_probability=min_success,
        min_accuracy=min_accuracy,
        semantic_confidence=float(confidence) if confidence is not None else 1.0,
        unknown_fields=unknown,
    )


# -------------------------------------------------------------------- builder
def _initial_state(
    pools: PhysicalPools, distances_m: np.ndarray, config: HybridConfig
) -> SystemState:
    work_cycles = pools.host_mean_turnaround_s[DEVICE_HOST] * config.device_cpu_hz
    edge_hz = [work_cycles / pools.host_mean_turnaround_s[host] for host in EDGE_HOSTS]
    device = DeviceState(
        DeviceSpec(
            device_id="device_0",
            cpu_hz=config.device_cpu_hz,
            energy_coefficient_j_per_cycle_hz2=config.energy_coefficient,
            tx_power_w=config.tx_power_w,
            rx_power_w=config.rx_power_w,
            idle_power_w=config.idle_power_w,
            success_probability=config.device_success,
            accuracy_score=config.device_accuracy,
        ),
        battery_j=config.battery_j,
    )
    servers = [
        ServerState(
            ServerSpec(
                server_id=f"edge_{index}",
                tier=ExecutionTier.EDGE,
                cpu_hz=edge_hz[index],
                success_probability=config.edge_success[index],
                accuracy_score=config.edge_accuracy[index],
            )
        )
        for index in range(3)
    ]
    servers.append(
        ServerState(
            ServerSpec(
                server_id="cloud",
                tier=ExecutionTier.CLOUD,
                cpu_hz=max(edge_hz) * config.cloud_speedup_vs_fastest_edge,
                success_probability=config.cloud_success,
                accuracy_score=config.cloud_accuracy,
                backhaul_rate_bps=config.cloud_backhaul_bps,
                one_way_propagation_s=config.cloud_one_way_propagation_s,
                backhaul_success_probability=config.cloud_backhaul_success,
            )
        )
    )
    links = [
        TargetLink(
            f"edge_{index}", build_link_metrics(float(distances_m[index]), config.radio)
        )
        for index in range(3)
    ]
    # The cloud is reached through the nearest site's radio link.
    links.append(
        TargetLink("cloud", build_link_metrics(float(distances_m[0]), config.radio))
    )
    return SystemState(
        now_s=0.0, devices=(device,), servers=tuple(servers), links=tuple(links)
    )


@dataclass(frozen=True, slots=True)
class SemanticItem:
    task_id: str
    true_requirements: TaskRequirements
    believed_requirements: TaskRequirements


def build_episode(
    *,
    rng: np.random.Generator,
    pools: PhysicalPools,
    windows: Sequence[WorkloadWindow],
    semantic_items: Sequence[SemanticItem],
    config: HybridConfig,
) -> tuple[Scenario, list[str]]:
    """One finite episode; returns the scenario and the semantic task ids used."""

    window = windows[int(rng.integers(len(windows)))]
    distances = pools.user_edge_distances_m[
        int(rng.integers(len(pools.user_edge_distances_m)))
    ]
    state = _initial_state(pools, distances, config)
    count = min(config.episode_tasks, len(window.input_bits))
    gaps = np.asarray(window.interarrival_s[:count], dtype=float)
    mean_gap = float(gaps.mean()) if gaps.size and gaps.mean() > 0 else 1.0
    gaps = gaps * (config.target_mean_interarrival_s / mean_gap)
    device_mean = float(pools.device_turnaround_s.mean())
    work_cycles = pools.host_mean_turnaround_s[DEVICE_HOST] * config.device_cpu_hz
    semantic_index = rng.integers(len(semantic_items), size=count)
    failure_draws = rng.random((count, 4))
    tasks: list[Task] = []
    events: list[ExogenousEvent] = []
    used: list[str] = []
    now = 0.0
    server_failure = (
        *(1.0 - p for p in config.edge_success),
        1.0 - config.cloud_success,
    )
    for index in range(count):
        item = semantic_items[int(semantic_index[index])]
        used.append(item.task_id)
        variability = (
            float(
                pools.device_turnaround_s[
                    int(rng.integers(len(pools.device_turnaround_s)))
                ]
            )
            / device_mean
        )
        tasks.append(
            Task(
                task_id=f"t{index}",
                source_device_id="device_0",
                arrival_time_s=now,
                input_bits=max(window.input_bits[index], 8.0),
                output_bits=max(window.output_bits[index], 8.0),
                compute_cycles=work_cycles * config.workload_scale * variability,
                requirements=item.true_requirements,
                believed_requirements=item.believed_requirements,
            )
        )
        failed = frozenset(
            target
            for target, probability, draw in zip(
                ("edge_0", "edge_1", "edge_2", "cloud"),
                server_failure,
                failure_draws[index],
                strict=True,
            )
            if draw < probability
        )
        gap = float(gaps[index])
        events.append(ExogenousEvent(gap, failed_target_ids=failed))
        now += gap
    if (
        config.nep_percentiles is not None
        and config.nep_edge_load_quantiles is not None
        and config.nep_rtt_ms_quantiles is not None
    ):
        # Drawn after every other random draw so that disabling the NEP layer
        # reproduces the episodes of the base configuration exactly.
        loads = np.minimum(
            _inverse_cdf(
                rng, config.nep_percentiles, config.nep_edge_load_quantiles, 3
            ),
            config.max_edge_background_load,
        )
        rtt_ms = float(
            _inverse_cdf(rng, config.nep_percentiles, config.nep_rtt_ms_quantiles, 1)[0]
        )
        servers = []
        for server in state.servers:
            spec = server.spec
            if spec.tier is ExecutionTier.EDGE:
                index = int(spec.server_id.split("_")[1])
                spec = replace(spec, cpu_hz=spec.cpu_hz * (1.0 - float(loads[index])))
            else:
                spec = replace(spec, one_way_propagation_s=rtt_ms / 2000.0)
            servers.append(replace(server, spec=spec))
        state = replace(state, servers=tuple(servers))
    return Scenario(state, tuple(tasks), tuple(events), ACTIONS), used


__all__ = [
    "ACTIONS",
    "HybridConfig",
    "PhysicalPools",
    "SemanticItem",
    "WorkloadWindow",
    "annotation_to_requirements",
    "build_episode",
    "load_nep_layer",
    "load_physical_pools",
]
