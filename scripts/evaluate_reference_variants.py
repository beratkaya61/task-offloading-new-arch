"""Protocol-normalized reference and input-structure subsets (no relabeling).

The locked single-human reference (protocol 1.3.0) is never modified.  This
script derives a *protocol-normalized* copy by applying only deterministic
rules that are written verbatim in ``docs/ANNOTATION_PROTOCOL.md``:

* R1 (Section 5.5): an explicit minimum success probability ``>= 0.999``
  is ``mission_critical``.
* R2 (Section 5.6): the metric of ``F1 skoru`` is written ``F1``; metric
  names are compared in their canonical upper-case spelling.

No rule looks at any model prediction.  Judgement-dependent disagreements
(for example requirements stated only in ``user_policy_text``) are *not*
changed; instead the evaluation is additionally reported on the subsets with
and without a ``user_policy_text`` field, a split chosen from the input
structure alone.  Every existing prediction file is re-scored on every
variant; the original locked results are left untouched.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from task_offloading.semantic import load_jsonl, write_jsonl
from task_offloading.semantic.baselines import SemanticPrediction
from task_offloading.semantic.evaluation import evaluate_semantic_predictions
from task_offloading.semantic.experiment import (
    load_json_object,
    ordered_subset,
    sha256,
    write_json_atomic,
)

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data" / "semantic_benchmark" / "main"
TASKS = MAIN / "tasks.v1.jsonl"
SPLIT = MAIN / "split_manifest.v1.json"
REFERENCE = MAIN / "annotations" / "annotator_a.single_human_reference.v1.3.jsonl"
SCHEMA = ROOT / "schemas" / "semantic_requirements.schema.json"
OUTPUT = MAIN / "reference_variants"
NORMALIZED = OUTPUT / "annotator_a.protocol_normalized.v1.3.1.jsonl"
CORRECTIONS = OUTPUT / "protocol_normalization_log.v1.json"
RESULTS = OUTPUT / "results.v1.json"
PREDICTION_FILES = (
    *sorted((MAIN / "baselines" / "predictions").glob("*.jsonl")),
    MAIN / "baselines" / "encoder" / "predictions.v1.jsonl",
)

MISSION_CRITICAL_THRESHOLD = 0.999
CANONICAL_METRICS = {"f1": "F1"}


def normalize_annotation(
    annotation: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply R1/R2 only; return the copy and a per-change log."""

    fixed = copy.deepcopy(annotation)
    changes: list[dict[str, Any]] = []
    reliability = fixed["reliability"]
    probability = reliability.get("min_success_probability")
    if (
        reliability.get("status") == "explicit"
        and probability is not None
        and float(probability) >= MISSION_CRITICAL_THRESHOLD
        and reliability.get("tier") != "mission_critical"
    ):
        changes.append(
            {
                "rule": "R1_reliability_threshold",
                "field": "reliability.tier",
                "from": reliability.get("tier"),
                "to": "mission_critical",
            }
        )
        reliability["tier"] = "mission_critical"
    accuracy = fixed["accuracy"]
    metric = accuracy.get("metric")
    if isinstance(metric, str):
        canonical = CANONICAL_METRICS.get(metric.lower(), metric)
        if canonical != metric:
            changes.append(
                {
                    "rule": "R2_metric_spelling",
                    "field": "accuracy.metric",
                    "from": metric,
                    "to": canonical,
                }
            )
            accuracy["metric"] = canonical
    return fixed, changes


def _score(
    name: str,
    tasks: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    predicted: dict[str, dict[str, Any]],
    schema: dict[str, Any],
) -> dict[str, Any]:
    ids = [task["task_id"] for task in tasks]
    result = evaluate_semantic_predictions(
        model_name=name,
        tasks=tasks,
        references=[references[item] for item in ids],
        predictions=[SemanticPrediction(predicted[item], 0.0) for item in ids],
        semantic_schema=schema,
    )
    return {
        "task_count": len(ids),
        "exact_record_accuracy": result["exact_record_accuracy"],
        "exact_field_accuracy": result["exact_field_accuracy"],
        "macro_f1_across_fields": result["macro_f1_across_fields"],
        "critical_privacy_recall": result["critical_privacy"]["recall"],
        "fields": {
            field: values["exact_semantic_accuracy"]
            for field, values in result["fields"].items()
        },
    }


def main() -> None:
    split = load_json_object(SPLIT)
    evaluation_ids = list(split["single_human_evaluation"]["task_ids"])
    tasks = ordered_subset(load_jsonl(TASKS), evaluation_ids)
    schema = load_json_object(SCHEMA)
    reference_rows = {row["task_id"]: row for row in load_jsonl(REFERENCE)}
    if set(reference_rows) != set(evaluation_ids):
        raise ValueError("reference ids do not match the evaluation split")

    original = {key: row["annotation"] for key, row in reference_rows.items()}
    normalized: dict[str, dict[str, Any]] = {}
    log: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    for task_id in evaluation_ids:
        fixed, changes = normalize_annotation(original[task_id])
        normalized[task_id] = fixed
        log.extend({"task_id": task_id, **change} for change in changes)
        row = copy.deepcopy(reference_rows[task_id])
        row["annotation"] = fixed
        row["derived_from"] = {
            "path": REFERENCE.relative_to(ROOT).as_posix(),
            "sha256": sha256(REFERENCE),
            "transform": "deterministic protocol normalization R1/R2",
            "human_relabeling": False,
        }
        normalized_rows.append(row)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(NORMALIZED, normalized_rows)
    write_json_atomic(
        CORRECTIONS,
        {
            "log_version": "1.0.0",
            "source_reference_sha256": sha256(REFERENCE),
            "rules": {
                "R1_reliability_threshold": "docs/ANNOTATION_PROTOCOL.md 5.5: "
                ">= 0.999 is mission_critical",
                "R2_metric_spelling": "docs/ANNOTATION_PROTOCOL.md 5.6: metric=F1",
            },
            "model_predictions_consulted": False,
            "change_count": len(log),
            "changes": log,
        },
    )

    with_policy = [task for task in tasks if task.get("user_policy_text")]
    without_policy = [task for task in tasks if not task.get("user_policy_text")]
    variants = {
        "original_reference_all_60": (tasks, original),
        "protocol_normalized_all_60": (tasks, normalized),
        "protocol_normalized_without_policy_text": (without_policy, normalized),
        "protocol_normalized_with_policy_text": (with_policy, normalized),
    }

    models: dict[str, Any] = {}
    for path in PREDICTION_FILES:
        if not path.exists():
            continue
        rows = load_jsonl(path)
        name = str(rows[0].get("model_name") or path.name.split(".")[0])
        predicted = {row["task_id"]: row["annotation"] for row in rows}
        models[name] = {
            "prediction_path": path.relative_to(ROOT).as_posix(),
            "prediction_sha256": sha256(path),
            "variants": {
                variant: _score(name, list(subset), references, predicted, schema)
                for variant, (subset, references) in variants.items()
            },
        }

    write_json_atomic(
        RESULTS,
        {
            "results_version": "1.0.0",
            "reference_claim": "single_human_reference_not_adjudicated_gold",
            "normalized_reference_sha256": sha256(NORMALIZED),
            "subset_sizes": {
                variant: len(subset) for variant, (subset, _) in variants.items()
            },
            "evaluation_labels_used_for_training_or_tuning": False,
            "models": models,
        },
    )

    header = f"{'model':28s} " + " ".join(
        f"{variant[:26]:>27s}" for variant in variants
    )
    print(f"normalization changes: {len(log)}")
    print(header)
    for name, payload in models.items():
        cells = []
        for variant in variants:
            score = payload["variants"][variant]
            cells.append(
                f"exact={score['exact_record_accuracy']:.3f} "
                f"F1={score['macro_f1_across_fields']:.3f}".rjust(27)
            )
        print(f"{name:28s} " + " ".join(cells))


if __name__ == "__main__":
    main()
