"""Tests proving that preprocessing learns from training data only."""

from __future__ import annotations

import numpy as np
import pytest

from task_offloading.data import (
    DataSplit,
    StandardizationStats,
    TrainOnlyStandardizer,
)


def test_train_fit_centers_training_matrix() -> None:
    train = np.asarray([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
    normalizer = TrainOnlyStandardizer().fit(
        train, ("latency_s", "energy_j"), split=DataSplit.TRAIN
    )
    transformed = normalizer.transform(train)
    np.testing.assert_allclose(transformed.mean(axis=0), np.zeros(2), atol=1e-12)
    assert normalizer.stats.sample_count == 3
    assert normalizer.stats.fitted_on is DataSplit.TRAIN


def test_validation_transform_reuses_training_statistics() -> None:
    train = np.asarray([[0.0], [2.0]])
    validation = np.asarray([[100.0], [102.0]])
    normalizer = TrainOnlyStandardizer().fit(
        train, ("latency_s",), split=DataSplit.TRAIN
    )
    transformed = normalizer.transform(validation)
    np.testing.assert_allclose(transformed, np.asarray([[99.0], [101.0]]))
    assert transformed.mean() != pytest.approx(0.0)


@pytest.mark.parametrize("split", [DataSplit.VALIDATION, DataSplit.TEST])
def test_fit_rejects_non_training_splits(split: DataSplit) -> None:
    with pytest.raises(ValueError, match="training split"):
        TrainOnlyStandardizer().fit([[1.0]], ("feature",), split=split)


def test_fitted_normalizer_cannot_be_silently_refitted() -> None:
    normalizer = TrainOnlyStandardizer().fit(
        [[1.0], [2.0]], ("feature",), split=DataSplit.TRAIN
    )
    with pytest.raises(RuntimeError, match="cannot be refitted"):
        normalizer.fit([[100.0]], ("feature",), split=DataSplit.TRAIN)


def test_constant_training_feature_uses_safe_unit_scale() -> None:
    normalizer = TrainOnlyStandardizer().fit(
        [[5.0], [5.0]], ("constant",), split=DataSplit.TRAIN
    )
    assert normalizer.stats.scale == (1.0,)
    np.testing.assert_array_equal(normalizer.transform([[5.0]]), [[0.0]])


def test_stats_round_trip_preserves_frozen_parameters() -> None:
    original = (
        TrainOnlyStandardizer()
        .fit(
            [[1.0, 2.0], [3.0, 6.0]],
            ("first", "second"),
            split=DataSplit.TRAIN,
        )
        .stats
    )
    restored = StandardizationStats.from_dict(original.to_dict())
    assert restored == original
    np.testing.assert_array_equal(
        TrainOnlyStandardizer(restored).transform([[1.0, 2.0]]),
        [[-1.0, -1.0]],
    )


def test_nonfinite_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        TrainOnlyStandardizer().fit(
            [[1.0], [float("nan")]], ("feature",), split=DataSplit.TRAIN
        )


@pytest.mark.parametrize("values", [[], [1.0], [[]]])
def test_fit_requires_nonempty_matrix(values: list[object]) -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        TrainOnlyStandardizer().fit(values, ("feature",), split=DataSplit.TRAIN)


def test_unfitted_normalizer_cannot_transform() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        TrainOnlyStandardizer().transform([[1.0]])


def test_feature_count_must_match_during_fit_and_transform() -> None:
    with pytest.raises(ValueError, match="column count"):
        TrainOnlyStandardizer().fit([[1.0, 2.0]], ("one",), split=DataSplit.TRAIN)
    normalizer = TrainOnlyStandardizer().fit(
        [[1.0, 2.0]], ("one", "two"), split=DataSplit.TRAIN
    )
    with pytest.raises(ValueError, match="fitted feature count"):
        normalizer.transform([[1.0]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_names": (), "mean": (), "scale": (), "sample_count": 1},
        {
            "feature_names": ("a", "a"),
            "mean": (0.0, 0.0),
            "scale": (1.0, 1.0),
            "sample_count": 1,
        },
        {"feature_names": ("a",), "mean": (), "scale": (1.0,), "sample_count": 1},
        {"feature_names": ("a",), "mean": (0.0,), "scale": (1.0,), "sample_count": 0},
        {
            "feature_names": ("a",),
            "mean": (0.0,),
            "scale": (1.0,),
            "sample_count": 1,
            "fitted_on": DataSplit.TEST,
        },
        {
            "feature_names": ("a",),
            "mean": (float("nan"),),
            "scale": (1.0,),
            "sample_count": 1,
        },
        {"feature_names": ("a",), "mean": (0.0,), "scale": (0.0,), "sample_count": 1},
    ],
)
def test_invalid_saved_normalization_statistics_are_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        StandardizationStats(**kwargs)  # type: ignore[arg-type]
