"""Typed records for provenance, benchmark identity, and leakage-safe splits."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


class ProvenanceKind(StrEnum):
    OBSERVED = "observed"
    MEASURED = "measured"
    DERIVED = "derived"
    MATCHED = "matched"
    GENERATED = "generated"
    HUMAN_LABELED = "human_labeled"
    SIMULATED = "simulated"


class BenchmarkKind(StrEnum):
    SYNTHETIC = "synthetic"
    TRACE_DRIVEN_HYBRID = "trace_driven_hybrid"


class DataSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class SplitStrategy(StrEnum):
    CHRONOLOGICAL = "chronological"
    GROUP_HOLDOUT = "group_holdout"


@dataclass(frozen=True, slots=True)
class DataSample:
    """Metadata needed to split one row without looking at model targets."""

    sample_id: str
    timestamp_s: float
    device_id: str
    station_id: str
    application_id: str
    content_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "sample_id",
            "device_id",
            "station_id",
            "application_id",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")
        if not math.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if SHA256_PATTERN.fullmatch(self.content_sha256) is None:
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    sample_id: str
    split: DataSplit

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """Immutable assignment plus the exact strategy that created it."""

    strategy: SplitStrategy
    assignments: tuple[SplitAssignment, ...]
    group_field: str | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.assignments:
            raise ValueError("split plan must contain assignments")
        sample_ids = [item.sample_id for item in self.assignments]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("a sample may be assigned only once")
        present = {item.split for item in self.assignments}
        if present != set(DataSplit):
            raise ValueError("train, validation, and test must all be non-empty")
        if self.strategy is SplitStrategy.GROUP_HOLDOUT and self.group_field is None:
            raise ValueError("group_holdout requires group_field")
        if self.strategy is SplitStrategy.GROUP_HOLDOUT and self.seed is None:
            raise ValueError("group_holdout requires a recorded seed")
        if (
            self.strategy is SplitStrategy.CHRONOLOGICAL
            and self.group_field is not None
        ):
            raise ValueError("chronological split cannot define group_field")

    def split_for(self, sample_id: str) -> DataSplit:
        for assignment in self.assignments:
            if assignment.sample_id == sample_id:
                return assignment.split
        raise KeyError(f"sample is absent from split plan: {sample_id}")


@dataclass(frozen=True, slots=True)
class LeakageReport:
    cross_split_duplicate_hashes: tuple[str, ...]
    overlapping_group_values: tuple[str, ...]
    chronological_order_valid: bool

    @property
    def is_clean(self) -> bool:
        return (
            not self.cross_split_duplicate_hashes
            and not self.overlapping_group_values
            and self.chronological_order_valid
        )
