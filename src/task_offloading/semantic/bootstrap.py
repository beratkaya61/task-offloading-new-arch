"""Task-level bootstrap confidence intervals for semantic-extraction metrics.

Macro-F1 follows ``evaluation.evaluate_semantic_predictions`` exactly: per
field, scikit-learn style macro-F1 over the union of labels present in the
(resampled) truth and prediction, ``zero_division=0``; then the mean over the
eight fields.  Paired differences resample the *same* task indices for both
models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from task_offloading.semantic.baselines import semantic_signature
from task_offloading.semantic.benchmark import ANNOTATION_FIELDS
from task_offloading.semantic.evaluation import _field_exact


def _macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    labels = np.union1d(truth, predicted)
    scores = []
    for label in labels:
        tp = float(np.sum((truth == label) & (predicted == label)))
        fp = float(np.sum((truth != label) & (predicted == label)))
        fn = float(np.sum((truth == label) & (predicted != label)))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores)) if scores else 0.0


class TaskLevelScores:
    """Pre-computed per-task outcomes for fast resampling."""

    def __init__(
        self,
        references: Sequence[Mapping[str, Any]],
        predictions: Sequence[Mapping[str, Any]],
    ) -> None:
        self.size = len(references)
        self.field_exact = np.asarray(
            [
                [_field_exact(ref, pred, field) for field in ANNOTATION_FIELDS]
                for ref, pred in zip(references, predictions, strict=True)
            ],
            dtype=bool,
        )
        self.truth: dict[str, np.ndarray] = {}
        self.predicted: dict[str, np.ndarray] = {}
        for field in ANNOTATION_FIELDS:
            t = [repr(semantic_signature(ref, field)) for ref in references]
            p = [repr(semantic_signature(pred, field)) for pred in predictions]
            vocabulary = {label: i for i, label in enumerate(sorted(set(t) | set(p)))}
            self.truth[field] = np.asarray([vocabulary[x] for x in t])
            self.predicted[field] = np.asarray([vocabulary[x] for x in p])

    def metrics(self, index: np.ndarray | None = None) -> dict[str, float]:
        idx = np.arange(self.size) if index is None else index
        exact = self.field_exact[idx]
        return {
            "exact_record_accuracy": float(exact.all(axis=1).mean()),
            "exact_field_accuracy": float(exact.mean()),
            "macro_f1_across_fields": float(
                np.mean(
                    [
                        _macro_f1(self.truth[f][idx], self.predicted[f][idx])
                        for f in ANNOTATION_FIELDS
                    ]
                )
            ),
        }


def bootstrap_intervals(
    scores: Mapping[str, TaskLevelScores],
    *,
    baseline: str,
    resamples: int = 1000,
    seed: int = 20261120,
    clusters: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Point estimate, 95% percentile CI, and paired difference vs ``baseline``.

    With ``clusters`` (one id per task) whole clusters are resampled, which is
    required when several rows share a scenario and are not independent.
    """

    sizes = {item.size for item in scores.values()}
    if len(sizes) != 1:
        raise ValueError("all models must be scored on the same tasks")
    size = sizes.pop()
    rng = np.random.default_rng(seed)
    if clusters is None:
        draws = [rng.integers(0, size, size) for _ in range(resamples)]
    else:
        if len(clusters) != size:
            raise ValueError("clusters must provide one id per task")
        members: dict[str, list[int]] = {}
        for index, cluster in enumerate(clusters):
            members.setdefault(str(cluster), []).append(index)
        groups = [np.asarray(v) for _, v in sorted(members.items())]
        draws = [
            np.concatenate(
                [groups[i] for i in rng.integers(0, len(groups), len(groups))]
            )
            for _ in range(resamples)
        ]
    samples = {
        name: [item.metrics(index) for index in draws] for name, item in scores.items()
    }
    output: dict[str, dict[str, Any]] = {}
    for name, item in scores.items():
        point = item.metrics()
        entry: dict[str, Any] = {}
        for metric, value in point.items():
            values = np.asarray([sample[metric] for sample in samples[name]])
            low, high = np.percentile(values, [2.5, 97.5])
            entry[metric] = {"value": value, "ci95": [float(low), float(high)]}
            if name != baseline and baseline in samples:
                diff = values - np.asarray(
                    [sample[metric] for sample in samples[baseline]]
                )
                d_low, d_high = np.percentile(diff, [2.5, 97.5])
                entry[metric]["diff_vs_baseline"] = {
                    "value": value - scores[baseline].metrics()[metric],
                    "ci95": [float(d_low), float(d_high)],
                    "p_value_two_sided": float(
                        min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0)))
                    ),
                }
        output[name] = entry
    return output


__all__ = ["TaskLevelScores", "bootstrap_intervals"]
