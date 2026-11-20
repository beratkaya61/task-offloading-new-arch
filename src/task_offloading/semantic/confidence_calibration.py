"""Post-hoc per-field confidence calibration for semantic extractors.

For every field an isotonic regression maps the raw confidence an extractor
reports to the empirical probability that the field is exactly correct.  It is
fitted on a dedicated *calibration set* (machine-generated, label-by-
construction tasks that no extractor was trained on) and then applied
unchanged to every evaluation set.  Labels are never changed; only the
``confidence`` values (and ``overall_confidence``) are replaced, so accuracy
and F1 are unaffected while ECE/Brier and the conservative fallback, which
reads confidences, see calibrated values.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

from task_offloading.semantic.benchmark import ANNOTATION_FIELDS
from task_offloading.semantic.evaluation import _field_exact


class FieldConfidenceCalibrator:
    def __init__(self) -> None:
        self._models: dict[str, IsotonicRegression] = {}
        self.fitted_on = 0

    def fit(
        self,
        references: Sequence[Mapping[str, Any]],
        predictions: Sequence[Mapping[str, Any]],
    ) -> FieldConfidenceCalibrator:
        if not references or len(references) != len(predictions):
            raise ValueError("calibration needs aligned, non-empty data")
        for field in ANNOTATION_FIELDS:
            confidence = np.asarray(
                [float(p[field]["confidence"]) for p in predictions]
            )
            correct = np.asarray(
                [
                    float(_field_exact(r, p, field))
                    for r, p in zip(references, predictions, strict=True)
                ]
            )
            model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            model.fit(confidence, correct)
            self._models[field] = model
        self.fitted_on = len(references)
        return self

    def apply(self, annotation: Mapping[str, Any]) -> dict[str, Any]:
        if set(self._models) != set(ANNOTATION_FIELDS):
            raise RuntimeError("calibrator must be fitted before use")
        calibrated = copy.deepcopy(dict(annotation))
        values = []
        for field in ANNOTATION_FIELDS:
            raw = float(calibrated[field]["confidence"])
            value = float(self._models[field].predict([raw])[0])
            calibrated[field]["confidence"] = value
            values.append(value)
        calibrated["overall_confidence"] = float(np.mean(values))
        return calibrated


__all__ = ["FieldConfidenceCalibrator"]
