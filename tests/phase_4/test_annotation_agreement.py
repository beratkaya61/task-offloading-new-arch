"""Agreement calculations for the Phase 4 human-label pilot."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from task_offloading.semantic import (
    cohen_kappa,
    diagnose_third_annotator,
    disagreement_rows,
    pairwise_agreement,
    quadratic_weighted_kappa,
)


def annotation(task_id: str, *, domain: str) -> dict[str, Any]:
    base = {"status": "not_stated", "evidence": [], "confidence": 0.9}
    return {
        "schema_version": "1.0.0",
        "task_id": task_id,
        "language": "tr",
        "domain": {
            "status": "explicit",
            "value": domain,
            "evidence": ["domain evidence"],
            "confidence": 0.9,
        },
        "latency": {"class": None, "max_ms": None, **base},
        "privacy": {"data_class": None, **base},
        "execution_policy": {
            "device": "unknown",
            "edge": "unknown",
            "cloud": "unknown",
            **base,
        },
        "reliability": {
            "tier": None,
            "min_success_probability": None,
            **base,
        },
        "accuracy": {"tier": None, "min_score": None, "metric": None, **base},
        "energy_priority": {"value": None, **base},
        "divisibility": {"value": None, **base},
        "abstained_fields": [],
        "overall_confidence": 0.9,
    }


def records(annotator_id: str, domains: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "annotator_id": annotator_id,
            "task_id": f"task_{index}",
            "annotation": annotation(f"task_{index}", domain=domain),
        }
        for index, domain in enumerate(domains, start=1)
    ]


def test_cohen_kappa_known_cases_and_undefined_constant_case() -> None:
    assert cohen_kappa(("a", "a", "b", "b"), ("a", "b", "a", "b")) == 0.0
    assert cohen_kappa(("a", "b"), ("a", "b")) == 1.0
    assert cohen_kappa(("a", "a"), ("a", "a")) is None
    with pytest.raises(ValueError, match="same length"):
        cohen_kappa(("a",), ("a", "b"))


def test_quadratic_weighted_kappa_uses_predefined_order() -> None:
    levels = ("low", "medium", "high")
    assert quadratic_weighted_kappa(
        levels, levels, levels=levels
    ) == pytest.approx(1.0)
    assert quadratic_weighted_kappa(
        levels, tuple(reversed(levels)), levels=levels
    ) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="outside the ordinal scale"):
        quadratic_weighted_kappa(("unknown",), ("low",), levels=levels)


def test_pairwise_report_aligns_by_task_and_keeps_execution_separate() -> None:
    left = records("annotator_a", ("healthcare", "industrial"))
    right = list(reversed(records("annotator_b", ("healthcare", "industrial"))))
    report = pairwise_agreement(left, right)
    domain = next(field for field in report.fields if field.field == "domain")

    assert report.task_count == 2
    assert report.exact_record_agreement == 1.0
    assert report.exact_field_agreement == 1.0
    assert domain.value is not None
    assert domain.value.kappa == 1.0
    assert report.execution_policy.mean_allowed_target_jaccard == 1.0


def test_third_annotator_diagnostic_does_not_majority_vote() -> None:
    left = records("annotator_a", ("healthcare", "industrial"))
    right = records("annotator_b", ("consumer", "industrial"))
    diagnostic = records("annotator_c", ("healthcare", "industrial"))

    result = diagnose_third_annotator(left, right, diagnostic)
    domain = next(field for field in result if field.field == "domain")
    rows = disagreement_rows(left, right, diagnostic)

    assert domain.primary_agreements == 1
    assert domain.diagnostic_matches_primary_agreement == 1
    assert domain.primary_disagreements == 1
    assert domain.diagnostic_matches_left == 1
    assert domain.diagnostic_matches_right == 0
    assert domain.diagnostic_matches_neither == 0
    assert len(rows) == 1
    assert rows[0]["task_id"] == "task_1"
    assert rows[0]["field"] == "domain"
    assert rows[0]["diagnostic"]["value"] == "healthcare"


def test_pairwise_rejects_duplicate_tasks_and_same_annotator() -> None:
    left = records("annotator_a", ("healthcare", "industrial"))
    duplicate = [deepcopy(left[0]), deepcopy(left[0])]
    with pytest.raises(ValueError, match="different annotators"):
        pairwise_agreement(left, deepcopy(left))
    with pytest.raises(ValueError, match="duplicate task id"):
        pairwise_agreement(duplicate, records("annotator_b", ("healthcare",)))
