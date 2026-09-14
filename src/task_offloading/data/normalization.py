"""A standardizer that can only learn parameters from the training split."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from task_offloading.data.types import DataSplit

FloatMatrix = NDArray[np.float64]


def _matrix(values: ArrayLike) -> FloatMatrix:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("values must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("normalizer values must all be finite")
    return matrix


@dataclass(frozen=True, slots=True)
class StandardizationStats:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    sample_count: int
    fitted_on: DataSplit = DataSplit.TRAIN

    def __post_init__(self) -> None:
        feature_count = len(self.feature_names)
        if not feature_count or len(set(self.feature_names)) != feature_count:
            raise ValueError("feature_names must be non-empty and unique")
        if len(self.mean) != feature_count or len(self.scale) != feature_count:
            raise ValueError("mean and scale must match feature_names")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be > 0")
        if self.fitted_on is not DataSplit.TRAIN:
            raise ValueError("normalization statistics must be fitted on train")
        if not all(np.isfinite(self.mean)):
            raise ValueError("mean values must be finite")
        if not all(np.isfinite(self.scale)) or any(value <= 0 for value in self.scale):
            raise ValueError("scale values must be finite and > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "sample_count": self.sample_count,
            "fitted_on": self.fitted_on.value,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> StandardizationStats:
        return cls(
            feature_names=tuple(str(item) for item in values["feature_names"]),
            mean=tuple(float(item) for item in values["mean"]),
            scale=tuple(float(item) for item in values["scale"]),
            sample_count=int(values["sample_count"]),
            fitted_on=DataSplit(str(values["fitted_on"])),
        )


class TrainOnlyStandardizer:
    """Prevent validation/test leakage by making the fit split explicit."""

    def __init__(self, stats: StandardizationStats | None = None) -> None:
        self._stats = stats

    @property
    def stats(self) -> StandardizationStats:
        if self._stats is None:
            raise RuntimeError("normalizer has not been fitted")
        return self._stats

    def fit(
        self,
        values: ArrayLike,
        feature_names: tuple[str, ...],
        *,
        split: DataSplit,
    ) -> TrainOnlyStandardizer:
        if split is not DataSplit.TRAIN:
            raise ValueError("normalizer fit is permitted only on the training split")
        if self._stats is not None:
            raise RuntimeError("normalizer is already fitted and cannot be refitted")
        matrix = _matrix(values)
        if len(feature_names) != matrix.shape[1]:
            raise ValueError("feature_names must match the matrix column count")
        scale = matrix.std(axis=0)
        scale[scale == 0.0] = 1.0
        self._stats = StandardizationStats(
            feature_names=feature_names,
            mean=tuple(float(item) for item in matrix.mean(axis=0)),
            scale=tuple(float(item) for item in scale),
            sample_count=matrix.shape[0],
        )
        return self

    def transform(self, values: ArrayLike) -> FloatMatrix:
        matrix = _matrix(values)
        stats = self.stats
        if matrix.shape[1] != len(stats.feature_names):
            raise ValueError("matrix columns do not match fitted feature count")
        mean = np.asarray(stats.mean, dtype=np.float64)
        scale = np.asarray(stats.scale, dtype=np.float64)
        return (matrix - mean) / scale
