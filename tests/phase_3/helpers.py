"""Deterministic Phase 3 sample builders."""

from __future__ import annotations

from task_offloading.data import DataSample, content_sha256


def sample(
    index: int,
    *,
    timestamp_s: float | None = None,
    device_id: str | None = None,
    station_id: str | None = None,
    application_id: str | None = None,
    payload_label: str | None = None,
) -> DataSample:
    return DataSample(
        sample_id=f"sample_{index}",
        timestamp_s=float(index if timestamp_s is None else timestamp_s),
        device_id=device_id or f"device_{index}",
        station_id=station_id or f"station_{index}",
        application_id=application_id or f"app_{index}",
        content_sha256=content_sha256(
            {"task": payload_label or f"unique_payload_{index}"}
        ),
    )


def group_samples() -> tuple[DataSample, ...]:
    return tuple(
        sample(
            index,
            device_id=f"device_{index // 2}",
            station_id=f"station_{index // 2}",
            application_id=f"app_{index // 2}",
        )
        for index in range(12)
    )
