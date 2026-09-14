"""Deterministic chronological/group holdouts and leakage audits."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import accumulate
from typing import Any, Literal

from task_offloading.data.types import (
    DataSample,
    DataSplit,
    LeakageReport,
    SplitAssignment,
    SplitPlan,
    SplitStrategy,
)

GroupField = Literal["device_id", "station_id", "application_id"]


def content_sha256(payload: Mapping[str, Any]) -> str:
    """Hash canonical content, independent of dictionary insertion order."""

    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_samples(samples: Sequence[DataSample]) -> None:
    if not samples:
        raise ValueError("at least one sample is required")
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("sample_id values must be unique")


def _weighted_cuts(
    group_sizes: Sequence[int], train_fraction: float, validation_fraction: float
) -> tuple[int, int]:
    item_count = len(group_sizes)
    if item_count < 3:
        raise ValueError("at least three distinct time/group values are required")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must be < 1")
    if any(size <= 0 for size in group_sizes):
        raise ValueError("group sizes must be > 0")
    cumulative = tuple(accumulate(group_sizes))
    total = cumulative[-1]
    train_target = total * train_fraction
    validation_target = total * (train_fraction + validation_fraction)
    train_cut = min(
        range(1, item_count - 1),
        key=lambda index: (abs(cumulative[index - 1] - train_target), index),
    )
    validation_cut = min(
        range(train_cut + 1, item_count),
        key=lambda index: (abs(cumulative[index - 1] - validation_target), index),
    )
    return train_cut, validation_cut


def _partition_labels(
    item_count: int, train_cut: int, validation_cut: int
) -> tuple[DataSplit, ...]:
    return tuple(
        DataSplit.TRAIN
        if index < train_cut
        else DataSplit.VALIDATION
        if index < validation_cut
        else DataSplit.TEST
        for index in range(item_count)
    )


def chronological_split(
    samples: Sequence[DataSample],
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> SplitPlan:
    """Split whole timestamp groups so equal-time rows cannot cross boundaries."""

    _validate_samples(samples)
    counts_by_time: dict[float, int] = defaultdict(int)
    for sample in samples:
        counts_by_time[sample.timestamp_s] += 1
    timestamps = sorted(counts_by_time)
    train_cut, validation_cut = _weighted_cuts(
        [counts_by_time[value] for value in timestamps],
        train_fraction,
        validation_fraction,
    )
    labels = _partition_labels(len(timestamps), train_cut, validation_cut)
    split_by_time = dict(zip(timestamps, labels, strict=True))
    assignments = tuple(
        SplitAssignment(sample.sample_id, split_by_time[sample.timestamp_s])
        for sample in sorted(samples, key=lambda item: item.sample_id)
    )
    return SplitPlan(SplitStrategy.CHRONOLOGICAL, assignments)


def _stable_group_key(value: str, group_field: GroupField, seed: int) -> str:
    if seed < 0:
        raise ValueError("seed must be >= 0")
    return hashlib.sha256(f"{seed}:{group_field}:{value}".encode()).hexdigest()


def group_holdout_split(
    samples: Sequence[DataSample],
    group_field: GroupField,
    *,
    seed: int,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> SplitPlan:
    """Keep every selected device, station, or application in exactly one split."""

    _validate_samples(samples)
    if group_field not in {"device_id", "station_id", "application_id"}:
        raise ValueError(f"unsupported group_field: {group_field}")
    counts_by_group: dict[str, int] = defaultdict(int)
    for sample in samples:
        counts_by_group[str(getattr(sample, group_field))] += 1
    groups = sorted(
        counts_by_group,
        key=lambda value: _stable_group_key(value, group_field, seed),
    )
    train_cut, validation_cut = _weighted_cuts(
        [counts_by_group[value] for value in groups],
        train_fraction,
        validation_fraction,
    )
    labels = _partition_labels(len(groups), train_cut, validation_cut)
    split_by_group = dict(zip(groups, labels, strict=True))
    assignments = tuple(
        SplitAssignment(
            sample.sample_id, split_by_group[str(getattr(sample, group_field))]
        )
        for sample in sorted(samples, key=lambda item: item.sample_id)
    )
    return SplitPlan(
        SplitStrategy.GROUP_HOLDOUT,
        assignments,
        group_field=group_field,
        seed=seed,
    )


def audit_split(samples: Sequence[DataSample], plan: SplitPlan) -> LeakageReport:
    _validate_samples(samples)
    sample_by_id = {sample.sample_id: sample for sample in samples}
    assigned_ids = {item.sample_id for item in plan.assignments}
    if assigned_ids != set(sample_by_id):
        raise ValueError("split plan and sample identifiers must match exactly")

    splits_by_hash: dict[str, set[DataSplit]] = defaultdict(set)
    for assignment in plan.assignments:
        splits_by_hash[sample_by_id[assignment.sample_id].content_sha256].add(
            assignment.split
        )
    duplicate_hashes = tuple(
        sorted(digest for digest, splits in splits_by_hash.items() if len(splits) > 1)
    )

    overlaps: tuple[str, ...] = ()
    if plan.group_field is not None:
        splits_by_group: dict[str, set[DataSplit]] = defaultdict(set)
        for assignment in plan.assignments:
            value = str(getattr(sample_by_id[assignment.sample_id], plan.group_field))
            splits_by_group[value].add(assignment.split)
        overlaps = tuple(
            sorted(
                value for value, splits in splits_by_group.items() if len(splits) > 1
            )
        )

    chronological_valid = True
    if plan.strategy is SplitStrategy.CHRONOLOGICAL:
        times: dict[DataSplit, list[float]] = defaultdict(list)
        for assignment in plan.assignments:
            times[assignment.split].append(
                sample_by_id[assignment.sample_id].timestamp_s
            )
        chronological_valid = max(times[DataSplit.TRAIN]) < min(
            times[DataSplit.VALIDATION]
        ) and max(times[DataSplit.VALIDATION]) < min(times[DataSplit.TEST])

    return LeakageReport(duplicate_hashes, overlaps, chronological_valid)


def assert_no_leakage(samples: Sequence[DataSample], plan: SplitPlan) -> None:
    report = audit_split(samples, plan)
    if not report.is_clean:
        raise ValueError(
            "split leakage detected: "
            f"duplicate_hashes={report.cross_split_duplicate_hashes}, "
            f"overlapping_groups={report.overlapping_group_values}, "
            f"chronological_order_valid={report.chronological_order_valid}"
        )
