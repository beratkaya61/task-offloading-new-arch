"""Agreement metrics for independently validated semantic annotations.

The primary analysis is pairwise by design.  A diagnostic third annotator may
be compared with each primary annotator, but is never pooled into a majority
vote or into the pre-registered primary Cohen's kappa.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from task_offloading.semantic.benchmark import ANNOTATION_FIELDS

_VALUE_KEYS = {
    "domain": "value",
    "latency": "class",
    "privacy": "data_class",
    "reliability": "tier",
    "accuracy": "tier",
    "energy_priority": "value",
    "divisibility": "value",
}
_ORDINAL_LEVELS = {
    "latency": ("relaxed", "interactive", "real_time", "hard_real_time"),
    "reliability": ("best_effort", "standard", "high", "mission_critical"),
    "accuracy": ("relaxed", "standard", "high", "critical"),
    "energy_priority": ("low", "balanced", "high"),
}
_SIGNATURE_KEYS = {
    "domain": ("status", "value"),
    "latency": ("status", "class", "max_ms"),
    "privacy": ("status", "data_class"),
    "execution_policy": ("status", "device", "edge", "cloud"),
    "reliability": ("status", "tier", "min_success_probability"),
    "accuracy": ("status", "tier", "min_score", "metric"),
    "energy_priority": ("status", "value"),
    "divisibility": ("status", "value"),
}
_EXECUTION_TARGETS = ("device", "edge", "cloud")


@dataclass(frozen=True, slots=True)
class AgreementMetric:
    """One agreement estimate with its actual comparison count."""

    sample_size: int
    percent_agreement: float
    kappa: float | None
    kappa_kind: str


@dataclass(frozen=True, slots=True)
class FieldAgreement:
    """Status, value, and strict semantic agreement for one field."""

    field: str
    exact_semantic_agreement: float
    status: AgreementMetric
    value: AgreementMetric | None


@dataclass(frozen=True, slots=True)
class ExecutionAgreement:
    """Execution-policy status, target permission, and allowed-set agreement."""

    exact_semantic_agreement: float
    status: AgreementMetric
    targets: tuple[tuple[str, AgreementMetric], ...]
    mean_allowed_target_jaccard: float


@dataclass(frozen=True, slots=True)
class PairwiseAgreement:
    """Complete pairwise result without adjudication or majority voting."""

    left_annotator: str
    right_annotator: str
    task_count: int
    exact_record_agreement: float
    exact_field_agreement: float
    fields: tuple[FieldAgreement, ...]
    execution_policy: ExecutionAgreement

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ThirdAnnotatorFieldDiagnostic:
    """How C relates to A/B agreements and disagreements for one field."""

    field: str
    primary_agreements: int
    diagnostic_matches_primary_agreement: int
    diagnostic_differs_from_primary_agreement: int
    primary_disagreements: int
    diagnostic_matches_left: int
    diagnostic_matches_right: int
    diagnostic_matches_neither: int


def _round(value: float) -> float:
    return round(value, 6)


def _percent_agreement(left: Sequence[str], right: Sequence[str]) -> float:
    if len(left) != len(right):
        raise ValueError("agreement inputs must have the same length")
    if not left:
        return 0.0
    return _round(sum(a == b for a, b in zip(left, right, strict=True)) / len(left))


def cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float | None:
    """Compute unweighted Cohen's kappa; return None when it is undefined."""

    if len(left) != len(right):
        raise ValueError("kappa inputs must have the same length")
    if not left:
        return None
    observed = _percent_agreement(left, right)
    count_left = Counter(left)
    count_right = Counter(right)
    total = len(left)
    categories = set(count_left) | set(count_right)
    expected = sum(
        (count_left[value] / total) * (count_right[value] / total)
        for value in categories
    )
    if abs(1.0 - expected) < 1e-12:
        return None
    return _round((observed - expected) / (1.0 - expected))


def quadratic_weighted_kappa(
    left: Sequence[str],
    right: Sequence[str],
    *,
    levels: Sequence[str],
) -> float | None:
    """Compute quadratic-weighted kappa over a predefined ordinal scale."""

    if len(left) != len(right):
        raise ValueError("kappa inputs must have the same length")
    if not left:
        return None
    if len(levels) < 2 or len(set(levels)) != len(levels):
        raise ValueError("ordinal levels must contain at least two unique values")
    index = {value: position for position, value in enumerate(levels)}
    unknown = (set(left) | set(right)) - set(index)
    if unknown:
        raise ValueError(f"values outside the ordinal scale: {sorted(unknown)}")

    denominator = (len(levels) - 1) ** 2

    def weight(a: str, b: str) -> float:
        return ((index[a] - index[b]) ** 2) / denominator

    observed_disagreement = sum(
        weight(a, b) for a, b in zip(left, right, strict=True)
    ) / len(left)
    count_left = Counter(left)
    count_right = Counter(right)
    expected_disagreement = sum(
        (count_left[a] / len(left))
        * (count_right[b] / len(right))
        * weight(a, b)
        for a in levels
        for b in levels
    )
    if expected_disagreement < 1e-12:
        return None
    return _round(1.0 - observed_disagreement / expected_disagreement)


def _metric(
    left: Sequence[str],
    right: Sequence[str],
    *,
    levels: Sequence[str] | None = None,
) -> AgreementMetric:
    if levels is None:
        kappa = cohen_kappa(left, right)
        kind = "cohen_kappa"
    else:
        kappa = quadratic_weighted_kappa(left, right, levels=levels)
        kind = "quadratic_weighted_kappa"
    return AgreementMetric(len(left), _percent_agreement(left, right), kappa, kind)


def _records_by_task(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        task_id = record.get("task_id")
        if not isinstance(task_id, str):
            raise ValueError("every agreement record must have a string task_id")
        if task_id in result:
            raise ValueError(f"duplicate task id in agreement input: {task_id}")
        result[task_id] = record
    return result


def _annotation(record: Mapping[str, Any]) -> Mapping[str, Any]:
    annotation = record.get("annotation")
    if not isinstance(annotation, Mapping):
        raise ValueError("agreement records must contain validated annotations")
    return annotation


def _label(annotation: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    label = annotation.get(field)
    if not isinstance(label, Mapping):
        raise ValueError(f"validated annotation is missing field: {field}")
    return label


def _signature(annotation: Mapping[str, Any], field: str) -> tuple[Any, ...]:
    label = _label(annotation, field)
    return tuple(label.get(key) for key in _SIGNATURE_KEYS[field])


def _annotator_id(records: Sequence[Mapping[str, Any]]) -> str:
    annotators = {record.get("annotator_id") for record in records}
    if len(annotators) != 1:
        raise ValueError("agreement input must contain exactly one annotator")
    annotator = next(iter(annotators))
    if not isinstance(annotator, str):
        raise ValueError("agreement input has no valid annotator id")
    return annotator


def _aligned_annotations(
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, Mapping[str, Any], Mapping[str, Any]], ...]:
    left = _records_by_task(left_records)
    right = _records_by_task(right_records)
    if set(left) != set(right):
        raise ValueError("agreement inputs must contain the same task ids")
    return tuple(
        (task_id, _annotation(left[task_id]), _annotation(right[task_id]))
        for task_id in sorted(left)
    )


def _standard_field_agreement(
    field: str,
    aligned: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
) -> FieldAgreement:
    labels = [(_label(left, field), _label(right, field)) for _, left, right in aligned]
    left_status = [str(left["status"]) for left, _ in labels]
    right_status = [str(right["status"]) for _, right in labels]
    paired_explicit = [
        (str(left[_VALUE_KEYS[field]]), str(right[_VALUE_KEYS[field]]))
        for left, right in labels
        if left["status"] == "explicit" and right["status"] == "explicit"
    ]
    left_values = [left for left, _ in paired_explicit]
    right_values = [right for _, right in paired_explicit]
    levels = _ORDINAL_LEVELS.get(field)
    value_metric = _metric(left_values, right_values, levels=levels)
    exact = sum(
        _signature(left, field) == _signature(right, field)
        for _, left, right in aligned
    ) / len(aligned)
    return FieldAgreement(
        field=field,
        exact_semantic_agreement=_round(exact),
        status=_metric(left_status, right_status),
        value=value_metric,
    )


def _execution_agreement(
    aligned: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
) -> ExecutionAgreement:
    labels = [
        (_label(left, "execution_policy"), _label(right, "execution_policy"))
        for _, left, right in aligned
    ]
    status = _metric(
        [str(left["status"]) for left, _ in labels],
        [str(right["status"]) for _, right in labels],
    )
    target_metrics = tuple(
        (
            target,
            _metric(
                [str(left[target]) for left, _ in labels],
                [str(right[target]) for _, right in labels],
            ),
        )
        for target in _EXECUTION_TARGETS
    )
    jaccards: list[float] = []
    for left, right in labels:
        left_allowed = {
            target for target in _EXECUTION_TARGETS if left[target] == "allowed"
        }
        right_allowed = {
            target for target in _EXECUTION_TARGETS if right[target] == "allowed"
        }
        union = left_allowed | right_allowed
        jaccards.append(
            1.0
            if not union
            else len(left_allowed & right_allowed) / len(union)
        )
    exact = sum(
        _signature(left, "execution_policy")
        == _signature(right, "execution_policy")
        for _, left, right in aligned
    ) / len(aligned)
    return ExecutionAgreement(
        exact_semantic_agreement=_round(exact),
        status=status,
        targets=target_metrics,
        mean_allowed_target_jaccard=_round(sum(jaccards) / len(jaccards)),
    )


def pairwise_agreement(
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
) -> PairwiseAgreement:
    """Compare two validated submissions, aligning records by task id."""

    left_annotator = _annotator_id(left_records)
    right_annotator = _annotator_id(right_records)
    if left_annotator == right_annotator:
        raise ValueError("pairwise agreement requires two different annotators")
    aligned = _aligned_annotations(left_records, right_records)
    if not aligned:
        raise ValueError("agreement inputs cannot be empty")

    fields = tuple(
        _standard_field_agreement(field, aligned)
        for field in ANNOTATION_FIELDS
        if field != "execution_policy"
    )
    execution = _execution_agreement(aligned)
    exact_field_matches = sum(
        _signature(left, field) == _signature(right, field)
        for _, left, right in aligned
        for field in ANNOTATION_FIELDS
    )
    exact_record_matches = sum(
        all(
            _signature(left, field) == _signature(right, field)
            for field in ANNOTATION_FIELDS
        )
        for _, left, right in aligned
    )
    return PairwiseAgreement(
        left_annotator=left_annotator,
        right_annotator=right_annotator,
        task_count=len(aligned),
        exact_record_agreement=_round(exact_record_matches / len(aligned)),
        exact_field_agreement=_round(
            exact_field_matches / (len(aligned) * len(ANNOTATION_FIELDS))
        ),
        fields=fields,
        execution_policy=execution,
    )


def diagnose_third_annotator(
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
    diagnostic_records: Sequence[Mapping[str, Any]],
) -> tuple[ThirdAnnotatorFieldDiagnostic, ...]:
    """Describe C relative to A/B without changing the primary A/B analysis."""

    left = _records_by_task(left_records)
    right = _records_by_task(right_records)
    diagnostic = _records_by_task(diagnostic_records)
    if not (set(left) == set(right) == set(diagnostic)):
        raise ValueError("primary and diagnostic inputs must contain the same tasks")

    results: list[ThirdAnnotatorFieldDiagnostic] = []
    for field in ANNOTATION_FIELDS:
        counts: Counter[str] = Counter()
        for task_id in sorted(left):
            signature_left = _signature(_annotation(left[task_id]), field)
            signature_right = _signature(_annotation(right[task_id]), field)
            signature_diagnostic = _signature(_annotation(diagnostic[task_id]), field)
            if signature_left == signature_right:
                counts["primary_agreements"] += 1
                if signature_diagnostic == signature_left:
                    counts["diagnostic_matches_primary_agreement"] += 1
                else:
                    counts["diagnostic_differs_from_primary_agreement"] += 1
            else:
                counts["primary_disagreements"] += 1
                if signature_diagnostic == signature_left:
                    counts["diagnostic_matches_left"] += 1
                elif signature_diagnostic == signature_right:
                    counts["diagnostic_matches_right"] += 1
                else:
                    counts["diagnostic_matches_neither"] += 1
        results.append(
            ThirdAnnotatorFieldDiagnostic(
                field=field,
                primary_agreements=counts["primary_agreements"],
                diagnostic_matches_primary_agreement=counts[
                    "diagnostic_matches_primary_agreement"
                ],
                diagnostic_differs_from_primary_agreement=counts[
                    "diagnostic_differs_from_primary_agreement"
                ],
                primary_disagreements=counts["primary_disagreements"],
                diagnostic_matches_left=counts["diagnostic_matches_left"],
                diagnostic_matches_right=counts["diagnostic_matches_right"],
                diagnostic_matches_neither=counts["diagnostic_matches_neither"],
            )
        )
    return tuple(results)


def disagreement_rows(
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
    diagnostic_records: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return A/B disagreement cells for a later, separate adjudication UI."""

    left = _records_by_task(left_records)
    right = _records_by_task(right_records)
    if set(left) != set(right):
        raise ValueError("agreement inputs must contain the same task ids")
    diagnostic = (
        _records_by_task(diagnostic_records)
        if diagnostic_records is not None
        else None
    )
    if diagnostic is not None and set(diagnostic) != set(left):
        raise ValueError("diagnostic input must contain the same task ids")

    rows: list[dict[str, Any]] = []
    for task_id in sorted(left):
        annotation_left = _annotation(left[task_id])
        annotation_right = _annotation(right[task_id])
        for field in ANNOTATION_FIELDS:
            if _signature(annotation_left, field) == _signature(
                annotation_right, field
            ):
                continue
            row: dict[str, Any] = {
                "task_id": task_id,
                "field": field,
                "left": dict(_label(annotation_left, field)),
                "right": dict(_label(annotation_right, field)),
            }
            if diagnostic is not None:
                row["diagnostic"] = dict(
                    _label(_annotation(diagnostic[task_id]), field)
                )
            rows.append(row)
    return tuple(rows)
