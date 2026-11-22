"""Score frozen semantic prediction artifacts without re-fitting extractors.

This recovery-safe scorer is intentionally separate from model training. It
verifies task-id alignment, keeps calibrated and raw confidence provenance
explicit, and can finish the independent-set comparison after an interrupted
CPU training job without changing any labels or predictions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from task_offloading.semantic import load_jsonl
from task_offloading.semantic.baselines import SemanticPrediction
from task_offloading.semantic.bootstrap import TaskLevelScores, bootstrap_intervals
from task_offloading.semantic.evaluation import evaluate_semantic_predictions
from task_offloading.semantic.experiment import (
    load_json_object,
    sha256,
    write_json_atomic,
)

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data" / "semantic_benchmark" / "main"
V3 = MAIN / "benchmark_v3"
PREDICTIONS = V3 / "predictions"
RAW = V3 / "predictions_raw"
SCHEMA = ROOT / "schemas" / "semantic_requirements.schema.json"
REFERENCE = MAIN / "reference_variants" / "annotator_a.protocol_normalized.v1.3.1.jsonl"
OUTPUT = V3 / "results.cached_v3_2.json"

SETS = (
    "human60_original",
    "human60_paraphrase",
    "compositional600_canonical",
    "compositional600_paraphrase",
    "independent_paraphrase",
)
CALIBRATED_MODELS = (
    "constant_profile",
    "rule_keyword",
    "tfidf_lr",
    "tfidf_lr_augmented",
    "e5_frozen_lr",
    "e5_frozen_lr_augmented",
)
LEGACY_MODELS = {
    "e5_finetuned_augmented_v31_raw": "e5_finetuned_augmented",
    "llm_fewshot_Qwen2.5-1.5B-Instruct": "llm_fewshot_Qwen2.5-1.5B-Instruct",
}
TRAINING_COUNTS = {
    "constant_profile": 180,
    "rule_keyword": 180,
    "tfidf_lr": 180,
    "tfidf_lr_augmented": 1980,
    "e5_frozen_lr": 180,
    "e5_frozen_lr_augmented": 1980,
    "e5_finetuned_augmented_v31_raw": 1980,
    "llm_fewshot_Qwen2.5-1.5B-Instruct": 4,
}


def _labels(set_name: str) -> dict[str, dict[str, Any]]:
    if set_name.startswith("human60"):
        rows = load_jsonl(REFERENCE)
    else:
        rows = load_jsonl(V3 / f"eval_{set_name}.labels.jsonl")
    return {str(row["task_id"]): row["annotation"] for row in rows}


def _prediction_rows(
    path: Path, task_ids: Sequence[str]
) -> tuple[list[SemanticPrediction], bool]:
    rows = load_jsonl(path)
    by_id = {str(row["task_id"]): row for row in rows}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    extra = sorted(set(by_id) - set(task_ids))
    if missing or extra:
        raise ValueError(
            f"{path.name} is not aligned: missing={missing[:3]}, extra={extra[:3]}"
        )
    has_latency = all(
        isinstance(by_id[task_id].get("inference_ms"), int | float)
        for task_id in task_ids
    )
    return (
        [
            SemanticPrediction(
                by_id[task_id]["annotation"],
                float(by_id[task_id].get("inference_ms", 0.0)),
            )
            for task_id in task_ids
        ],
        has_latency,
    )


def _summary(
    evaluation: Mapping[str, Any], *, latency_recorded: bool
) -> dict[str, Any]:
    return {
        "exact_record_accuracy": evaluation["exact_record_accuracy"],
        "exact_field_accuracy": evaluation["exact_field_accuracy"],
        "macro_f1_across_fields": evaluation["macro_f1_across_fields"],
        "schema_validity": evaluation["schema_validity"],
        "abstention_f1": evaluation["abstention"]["f1"],
        "critical_privacy": evaluation["critical_privacy"],
        "calibration": evaluation["calibration"],
        "inference_latency_ms_mean": (
            evaluation["inference_latency_ms"]["mean"] if latency_recorded else None
        ),
        "fields_macro_f1": {
            field: value["macro_f1"] for field, value in evaluation["fields"].items()
        },
    }


def main() -> None:
    schema = load_json_object(SCHEMA)
    tasks_by_set = {
        name: load_jsonl(V3 / f"eval_{name}.tasks.jsonl") for name in SETS
    }
    labels_by_set = {name: _labels(name) for name in SETS}
    point: dict[str, Any] = {}
    scores: dict[str, dict[str, TaskLevelScores]] = {name: {} for name in SETS}

    for model in CALIBRATED_MODELS:
        point[model] = {
            "training_task_count": TRAINING_COUNTS[model],
            "calibration_task_count": 360,
            "confidence_provenance": "isotonic_on_development_only_calibration_set",
            "sets": {},
        }
        for set_name in SETS:
            tasks = tasks_by_set[set_name]
            task_ids = [str(task["task_id"]) for task in tasks]
            references = [labels_by_set[set_name][task_id] for task_id in task_ids]
            calibrated, calibrated_latency = _prediction_rows(
                PREDICTIONS / f"{model}.{set_name}.jsonl", task_ids
            )
            raw, raw_latency = _prediction_rows(
                RAW / f"{model}.{set_name}.jsonl", task_ids
            )
            calibrated_eval = evaluate_semantic_predictions(
                model_name=model,
                tasks=tasks,
                references=references,
                predictions=calibrated,
                semantic_schema=schema,
            )
            raw_eval = evaluate_semantic_predictions(
                model_name=model,
                tasks=tasks,
                references=references,
                predictions=raw,
                semantic_schema=schema,
            )
            entry = _summary(calibrated_eval, latency_recorded=calibrated_latency)
            entry["calibration_raw"] = raw_eval["calibration"]
            entry["raw_prediction_latency_recorded"] = raw_latency
            point[model]["sets"][set_name] = entry
            scores[set_name][model] = TaskLevelScores(
                references, [prediction.annotation for prediction in calibrated]
            )

    for output_name, file_stem in LEGACY_MODELS.items():
        point[output_name] = {
            "training_task_count": TRAINING_COUNTS[output_name],
            "calibration_task_count": 0,
            "confidence_provenance": (
                "constant_uncalibrated"
                if output_name.startswith("llm_")
                else "v3_1_uncalibrated_cache"
            ),
            "sets": {},
        }
        for set_name in SETS:
            path = PREDICTIONS / f"{file_stem}.{set_name}.jsonl"
            if not path.exists():
                continue
            tasks = tasks_by_set[set_name]
            task_ids = [str(task["task_id"]) for task in tasks]
            references = [labels_by_set[set_name][task_id] for task_id in task_ids]
            predictions, latency_recorded = _prediction_rows(path, task_ids)
            evaluation = evaluate_semantic_predictions(
                model_name=output_name,
                tasks=tasks,
                references=references,
                predictions=predictions,
                semantic_schema=schema,
            )
            entry = _summary(evaluation, latency_recorded=latency_recorded)
            entry["calibration_raw"] = evaluation["calibration"]
            point[output_name]["sets"][set_name] = entry
            scores[set_name][output_name] = TaskLevelScores(
                references, [prediction.annotation for prediction in predictions]
            )

    intervals = {
        set_name: bootstrap_intervals(set_scores, baseline="rule_keyword")
        for set_name, set_scores in scores.items()
    }
    write_json_atomic(
        OUTPUT,
        {
            "results_version": "3.2.0-cache-scored",
            "status": "complete_for_available_frozen_prediction_artifacts",
            "design": __doc__,
            "reference_sha256": sha256(REFERENCE),
            "evaluation_labels_used_for_training_or_tuning": False,
            "prediction_values_changed_by_this_script": False,
            "latency_note": (
                "The prediction cache did not retain per-task inference latency; "
                "missing latency values are null and are not estimated."
            ),
            "label_claims": {
                "human60_original": "single_human_reference_protocol_normalized",
                "human60_paraphrase": (
                    "machine_paraphrase_with_human_labels_transferred_under_"
                    "an_invariance_assumption_not_human_annotated"
                ),
                "compositional600_*": "machine_generated_label_by_construction",
                "independent_paraphrase": (
                    "machine_generated_paraphrase_with_labels_transferred_under_"
                    "an_invariance_assumption_not_human_annotated"
                ),
            },
            "set_sizes": {name: len(tasks) for name, tasks in tasks_by_set.items()},
            "bootstrap": {
                "resamples": 1000,
                "interval": "percentile 95%",
                "unit": "task",
            },
            "point_estimates": point,
            "intervals": intervals,
        },
    )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
