"""Shared metrics for all rule, ML, encoder, and LLM semantic extractors."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import fmean, median
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator
from sklearn.metrics import f1_score, precision_recall_fscore_support

from task_offloading.semantic.baselines import (
    SemanticPrediction,
    semantic_signature,
    visible_text,
)
from task_offloading.semantic.benchmark import ANNOTATION_FIELDS

_EXACT_KEYS: dict[str, tuple[str, ...]] = {
    "domain": ("status", "value"),
    "latency": ("status", "class", "max_ms"),
    "privacy": ("status", "data_class"),
    "execution_policy": ("status", "device", "edge", "cloud"),
    "reliability": ("status", "tier", "min_success_probability"),
    "accuracy": ("status", "tier", "min_score", "metric"),
    "energy_priority": ("status", "value"),
    "divisibility": ("status", "value"),
}


def _field_values(annotation: Mapping[str, Any], field: str) -> tuple[object, ...]:
    raw = annotation.get(field)
    if not isinstance(raw, Mapping):
        raise ValueError(f"annotation lacks field: {field}")
    return tuple(raw.get(key) for key in _EXACT_KEYS[field])


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, int | float) and isinstance(right, int | float):
        return math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-9)
    return left == right


def _field_exact(
    reference: Mapping[str, Any], prediction: Mapping[str, Any], field: str
) -> bool:
    return all(
        _same_value(left, right)
        for left, right in zip(
            _field_values(reference, field),
            _field_values(prediction, field),
            strict=True,
        )
    )


def _safe_binary_metrics(
    truth: Sequence[bool], predicted: Sequence[bool]
) -> dict[str, float | int | None]:
    if not truth or not any(truth):
        return {
            "positive_reference_count": int(sum(truth)),
            "precision": None,
            "recall": None,
            "f1": None,
        }
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predicted,
        average="binary",
        zero_division=0,
    )
    return {
        "positive_reference_count": int(sum(truth)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _calibration(
    confidences: Sequence[float], outcomes: Sequence[bool], bins: int = 10
) -> dict[str, float | int]:
    if len(confidences) != len(outcomes) or not confidences:
        raise ValueError("calibration inputs must be non-empty and aligned")
    confidence_array = np.asarray(confidences, dtype=np.float64)
    outcome_array = np.asarray(outcomes, dtype=np.float64)
    brier = float(np.mean(np.square(confidence_array - outcome_array)))
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + (1 / bins)
        below_upper = (
            confidence_array <= upper
            if math.isclose(upper, 1.0)
            else confidence_array < upper
        )
        mask = (confidence_array >= lower) & (
            below_upper
        )
        count = int(np.sum(mask))
        if count:
            ece += (count / len(confidence_array)) * abs(
                float(np.mean(confidence_array[mask]))
                - float(np.mean(outcome_array[mask]))
            )
    return {"sample_count": len(confidences), "brier": brier, "ece_10_bin": ece}


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def evaluate_semantic_predictions(
    *,
    model_name: str,
    tasks: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    predictions: Sequence[SemanticPrediction],
    semantic_schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate aligned predictions without using evidence/confidence as labels."""

    if not tasks or len(tasks) != len(references) or len(tasks) != len(predictions):
        raise ValueError(
            "tasks, references, and predictions must be non-empty and aligned"
        )
    validator = Draft202012Validator(semantic_schema)
    field_outcomes: dict[str, list[bool]] = {
        field: [] for field in ANNOTATION_FIELDS
    }
    field_metrics: dict[str, dict[str, Any]] = {}
    schema_valid = 0
    grounded_evidence = 0
    evidence_count = 0
    abstention_truth: list[bool] = []
    abstention_predicted: list[bool] = []
    confidences: list[float] = []
    confidence_outcomes: list[bool] = []
    record_exact: list[bool] = []
    namespaced_truth: list[str] = []
    namespaced_predictions: list[str] = []

    for task, reference, prediction in zip(
        tasks, references, predictions, strict=True
    ):
        predicted = prediction.annotation
        if str(task["task_id"]) != reference.get("task_id"):
            raise ValueError("reference task order is not aligned")
        if str(task["task_id"]) != predicted.get("task_id"):
            raise ValueError("prediction task order is not aligned")
        if not tuple(validator.iter_errors(predicted)):
            schema_valid += 1
        visible = visible_text(task)
        task_field_outcomes: list[bool] = []
        for field in ANNOTATION_FIELDS:
            outcome = _field_exact(reference, predicted, field)
            field_outcomes[field].append(outcome)
            task_field_outcomes.append(outcome)
            truth_signature = semantic_signature(reference, field)
            predicted_signature = semantic_signature(predicted, field)
            namespaced_truth.append(f"{field}:{truth_signature!r}")
            namespaced_predictions.append(f"{field}:{predicted_signature!r}")

            reference_field = reference[field]
            predicted_field = predicted[field]
            if not isinstance(reference_field, Mapping) or not isinstance(
                predicted_field, Mapping
            ):
                raise ValueError("semantic fields must be objects")
            truth_abstains = reference_field.get("status") in {
                "not_stated",
                "ambiguous",
            }
            predicted_abstains = predicted_field.get("status") in {
                "not_stated",
                "ambiguous",
            }
            abstention_truth.append(truth_abstains)
            abstention_predicted.append(predicted_abstains)
            confidence = predicted_field.get("confidence")
            if not isinstance(confidence, int | float):
                raise ValueError("prediction confidence must be numeric")
            confidences.append(float(confidence))
            confidence_outcomes.append(outcome)
            evidence = predicted_field.get("evidence")
            if isinstance(evidence, list):
                for phrase in evidence:
                    if isinstance(phrase, str):
                        evidence_count += 1
                        grounded_evidence += int(phrase in visible)
        record_exact.append(all(task_field_outcomes))

    for field in ANNOTATION_FIELDS:
        truth = [
            repr(semantic_signature(reference, field)) for reference in references
        ]
        predicted_values = [
            repr(semantic_signature(prediction.annotation, field))
            for prediction in predictions
        ]
        field_metrics[field] = {
            "exact_semantic_accuracy": fmean(field_outcomes[field]),
            "macro_f1": float(
                f1_score(truth, predicted_values, average="macro", zero_division=0)
            ),
            "reference_class_count": len(set(truth)),
        }

    privacy_truth = [
        reference["privacy"]["status"] == "explicit"
        and reference["privacy"]["data_class"] in {"sensitive", "restricted"}
        for reference in references
    ]
    privacy_predicted = [
        prediction.annotation["privacy"]["status"] == "explicit"
        and prediction.annotation["privacy"]["data_class"]
        in {"sensitive", "restricted"}
        for prediction in predictions
    ]
    latency_truth = [
        reference["latency"]["status"] == "explicit"
        and reference["latency"]["class"] in {"real_time", "hard_real_time"}
        for reference in references
    ]
    latency_predicted = [
        prediction.annotation["latency"]["status"] == "explicit"
        and prediction.annotation["latency"]["class"]
        in {"real_time", "hard_real_time"}
        for prediction in predictions
    ]
    durations = [prediction.inference_ms for prediction in predictions]
    return {
        "model_name": model_name,
        "reference_claim": "single_human_reference_not_adjudicated_gold",
        "task_count": len(tasks),
        "schema_validity": schema_valid / len(tasks),
        "exact_record_accuracy": fmean(record_exact),
        "exact_field_accuracy": fmean(
            outcome for outcomes in field_outcomes.values() for outcome in outcomes
        ),
        "micro_f1": float(
            f1_score(
                namespaced_truth,
                namespaced_predictions,
                average="micro",
                zero_division=0,
            )
        ),
        "macro_f1_across_fields": fmean(
            float(metrics["macro_f1"]) for metrics in field_metrics.values()
        ),
        "fields": field_metrics,
        "abstention": {
            "accuracy": fmean(
                truth == predicted
                for truth, predicted in zip(
                    abstention_truth, abstention_predicted, strict=True
                )
            ),
            **_safe_binary_metrics(abstention_truth, abstention_predicted),
        },
        "critical_privacy": _safe_binary_metrics(
            privacy_truth, privacy_predicted
        ),
        "critical_latency": _safe_binary_metrics(
            latency_truth, latency_predicted
        ),
        "calibration": _calibration(confidences, confidence_outcomes),
        "evidence_grounding_rate": (
            grounded_evidence / evidence_count if evidence_count else 1.0
        ),
        "inference_latency_ms": {
            "mean": fmean(durations),
            "median": median(durations),
            "p95": _percentile(durations, 95),
            "total": sum(durations),
        },
        "cost": {
            "input_tokens": None,
            "output_tokens": None,
            "monetary_cost_usd": 0.0,
            "note": "local non-LLM baseline; tokenizer cost is not applicable",
        },
    }
