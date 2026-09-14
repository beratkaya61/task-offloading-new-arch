"""Chronological, entity holdout, duplicate, and leakage tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from task_offloading.data import (
    DataSplit,
    SplitAssignment,
    SplitPlan,
    SplitStrategy,
    assert_no_leakage,
    audit_split,
    chronological_split,
    content_sha256,
    group_holdout_split,
)

from .helpers import group_samples, sample


def test_canonical_content_hash_ignores_mapping_insertion_order() -> None:
    assert content_sha256({"a": 1, "b": 2}) == content_sha256({"b": 2, "a": 1})


def test_chronological_split_preserves_order_and_equal_timestamp_group() -> None:
    samples = (
        *(sample(index) for index in range(10)),
        sample(10, timestamp_s=5.0),
    )
    plan = chronological_split(samples)
    report = audit_split(samples, plan)
    assert report.is_clean
    assert plan.split_for("sample_5") == plan.split_for("sample_10")
    assert_no_leakage(samples, plan)


@pytest.mark.parametrize("group_field", ["device_id", "station_id", "application_id"])
def test_group_holdout_keeps_entities_in_one_split(group_field: str) -> None:
    samples = group_samples()
    plan = group_holdout_split(samples, group_field, seed=2026)  # type: ignore[arg-type]
    assert plan.group_field == group_field
    assert audit_split(samples, plan).is_clean


def test_group_holdout_is_deterministic_under_input_reordering() -> None:
    samples = group_samples()
    forward = group_holdout_split(samples, "device_id", seed=42)
    reverse = group_holdout_split(tuple(reversed(samples)), "device_id", seed=42)
    assert forward == reverse


def test_group_holdout_balances_rows_when_group_sizes_differ() -> None:
    samples = tuple(
        sample(
            index,
            device_id="large_group" if index < 6 else f"small_group_{index}",
        )
        for index in range(12)
    )
    plan = group_holdout_split(
        samples,
        "device_id",
        seed=12,
        train_fraction=0.5,
        validation_fraction=0.25,
    )
    counts = {
        split_name: sum(
            plan.split_for(item.sample_id) is split_name for item in samples
        )
        for split_name in DataSplit
    }
    assert all(count > 0 for count in counts.values())
    assert sum(counts.values()) == len(samples)


def test_cross_split_duplicate_content_is_reported() -> None:
    samples = tuple(
        sample(
            index,
            payload_label="duplicate" if index in {0, 5} else None,
        )
        for index in range(6)
    )
    plan = chronological_split(samples, train_fraction=0.5, validation_fraction=0.2)
    report = audit_split(samples, plan)
    assert len(report.cross_split_duplicate_hashes) == 1
    assert not report.is_clean
    with pytest.raises(ValueError, match="split leakage"):
        assert_no_leakage(samples, plan)


def test_audit_detects_group_value_manually_leaked_across_splits() -> None:
    samples = group_samples()
    clean = group_holdout_split(samples, "station_id", seed=7)
    assignments = list(clean.assignments)
    target = next(
        item
        for item in assignments
        if item.sample_id == "sample_1" and item.split == clean.split_for("sample_0")
    )
    other_split = next(
        split_name for split_name in DataSplit if split_name != target.split
    )
    assignments[assignments.index(target)] = SplitAssignment(
        target.sample_id, other_split
    )
    leaked = SplitPlan(
        SplitStrategy.GROUP_HOLDOUT,
        tuple(assignments),
        group_field="station_id",
        seed=7,
    )
    assert "station_0" in audit_split(samples, leaked).overlapping_group_values


def test_audit_requires_exact_sample_identity_set() -> None:
    samples = group_samples()
    plan = group_holdout_split(samples, "application_id", seed=7)
    with pytest.raises(ValueError, match="identifiers must match"):
        audit_split(samples[:-1], plan)


def test_data_sample_rejects_invalid_content_digest() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        replace(sample(0), content_sha256="not-a-sha")


def test_split_rejects_empty_or_duplicate_sample_ids() -> None:
    with pytest.raises(ValueError, match="at least one"):
        chronological_split(())
    duplicate = (sample(0), replace(sample(1), sample_id="sample_0"))
    with pytest.raises(ValueError, match="sample_id values must be unique"):
        chronological_split(duplicate)


@pytest.mark.parametrize(
    ("train_fraction", "validation_fraction"),
    [(0.0, 0.2), (0.6, 0.0), (0.8, 0.2)],
)
def test_invalid_split_fractions_are_rejected(
    train_fraction: float, validation_fraction: float
) -> None:
    with pytest.raises(ValueError, match="fractions"):
        chronological_split(
            tuple(sample(index) for index in range(5)),
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )


def test_split_requires_three_distinct_groups() -> None:
    samples = tuple(sample(index, device_id="same") for index in range(3))
    with pytest.raises(ValueError, match="three distinct"):
        group_holdout_split(samples, "device_id", seed=1)


def test_group_holdout_rejects_invalid_field_and_seed() -> None:
    samples = group_samples()
    with pytest.raises(ValueError, match="unsupported group_field"):
        group_holdout_split(samples, "bad_field", seed=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        group_holdout_split(samples, "device_id", seed=-1)


def test_split_plan_requires_all_splits_and_recorded_group_seed() -> None:
    assignments = (
        SplitAssignment("a", DataSplit.TRAIN),
        SplitAssignment("b", DataSplit.VALIDATION),
        SplitAssignment("c", DataSplit.TEST),
    )
    with pytest.raises(ValueError, match="recorded seed"):
        SplitPlan(
            SplitStrategy.GROUP_HOLDOUT,
            assignments,
            group_field="device_id",
        )
    with pytest.raises(ValueError, match="all be non-empty"):
        SplitPlan(
            SplitStrategy.CHRONOLOGICAL,
            (SplitAssignment("only", DataSplit.TRAIN),),
        )
