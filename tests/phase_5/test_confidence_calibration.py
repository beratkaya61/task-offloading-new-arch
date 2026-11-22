from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from task_offloading.semantic import load_jsonl
from task_offloading.semantic.benchmark import ANNOTATION_FIELDS
from task_offloading.semantic.confidence_calibration import FieldConfidenceCalibrator

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = (
    ROOT
    / "data"
    / "semantic_benchmark"
    / "main"
    / "reference_variants"
    / "annotator_a.protocol_normalized.v1.3.1.jsonl"
)


def _records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = load_jsonl(REFERENCE)[0]["annotation"]
    references: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for index, confidence in enumerate((0.1, 0.35, 0.65, 0.9)):
        reference = copy.deepcopy(source)
        prediction = copy.deepcopy(source)
        for field in ANNOTATION_FIELDS:
            prediction[field]["confidence"] = confidence
        prediction["overall_confidence"] = confidence
        if index < 2:
            # Force every field to be inexact without depending on its value schema.
            for field in ANNOTATION_FIELDS:
                status = str(prediction[field]["status"])
                prediction[field]["status"] = (
                    "unknown" if status != "unknown" else "not_stated"
                )
        references.append(reference)
        predictions.append(prediction)
    return references, predictions


def _without_confidence(annotation: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(annotation)
    result.pop("overall_confidence", None)
    for field in ANNOTATION_FIELDS:
        result[field].pop("confidence", None)
    return result


def test_per_field_isotonic_calibration_changes_only_confidence() -> None:
    references, predictions = _records()
    calibrator = FieldConfidenceCalibrator().fit(references, predictions)

    calibrated = calibrator.apply(predictions[-1])

    assert calibrator.fitted_on == 4
    assert _without_confidence(calibrated) == _without_confidence(predictions[-1])
    assert 0.0 <= calibrated["overall_confidence"] <= 1.0
    assert all(
        0.0 <= float(calibrated[field]["confidence"]) <= 1.0
        for field in ANNOTATION_FIELDS
    )


@pytest.mark.parametrize(
    ("references", "predictions"),
    [([], []), ([{}], []), ([], [{}])],
)
def test_calibration_rejects_empty_or_misaligned_inputs(
    references: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> None:
    with pytest.raises(ValueError, match="aligned, non-empty"):
        FieldConfidenceCalibrator().fit(references, predictions)


def test_calibration_must_be_fitted_before_apply() -> None:
    _, predictions = _records()
    with pytest.raises(RuntimeError, match="fitted"):
        FieldConfidenceCalibrator().apply(predictions[0])
