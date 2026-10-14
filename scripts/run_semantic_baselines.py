"""Fit and evaluate the frozen non-LLM semantic baseline suite."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from task_offloading.semantic import load_jsonl, write_jsonl
from task_offloading.semantic.baselines import (
    ConstantProfileBaseline,
    RuleBasedBaseline,
    SemanticPrediction,
    TfidfLogisticBaseline,
)
from task_offloading.semantic.evaluation import evaluate_semantic_predictions
from task_offloading.semantic.experiment import (
    load_json_object,
    ordered_subset,
    prediction_rows,
    sha256,
    task_ids,
    write_json_atomic,
)

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data" / "semantic_benchmark" / "main"
TASKS = MAIN / "tasks.v1.jsonl"
SPLIT = MAIN / "split_manifest.v1.json"
TRAINING_LABELS = MAIN / "preannotations" / "machine_development.v1.jsonl"
REFERENCES = (
    MAIN / "annotations" / "annotator_a.single_human_reference.v1.3.jsonl"
)
REFERENCE_MANIFEST = MAIN / "annotations" / "submission_manifest.v1.json"
SCHEMA = ROOT / "schemas" / "semantic_requirements.schema.json"
CONFIG = ROOT / "configs" / "semantic_baselines.v1.json"
SOURCE_BASELINES = ROOT / "src" / "task_offloading" / "semantic" / "baselines.py"
SOURCE_EVALUATION = ROOT / "src" / "task_offloading" / "semantic" / "evaluation.py"
SOURCE_EXPERIMENT = ROOT / "src" / "task_offloading" / "semantic" / "experiment.py"
SOURCE_RUNNER = ROOT / "scripts" / "run_semantic_baselines.py"
OUTPUT = MAIN / "baselines"
PREDICTIONS = OUTPUT / "predictions"
RESULTS = OUTPUT / "results.v1.json"
MANIFEST = OUTPUT / "manifest.v1.json"


def main() -> None:
    config = load_json_object(CONFIG)
    split = load_json_object(SPLIT)
    all_tasks = load_jsonl(TASKS)
    development_ids = list(split["machine_labeled_development"]["task_ids"])
    evaluation_ids = list(split["single_human_evaluation"]["task_ids"])
    development_id_set = set(development_ids)
    evaluation_id_set = set(evaluation_ids)
    if development_id_set & evaluation_id_set:
        raise ValueError("development and evaluation task ids overlap")
    if development_id_set | evaluation_id_set != task_ids(all_tasks):
        raise ValueError("split task ids do not exactly cover the corpus")
    development_tasks = ordered_subset(all_tasks, development_ids)
    evaluation_tasks = ordered_subset(all_tasks, evaluation_ids)

    raw_training_records = load_jsonl(TRAINING_LABELS)
    if task_ids(raw_training_records) != development_id_set:
        raise ValueError("training labels do not exactly match development ids")
    training_records = ordered_subset(raw_training_records, development_ids)
    training_annotations = tuple(record["annotation"] for record in training_records)
    if len(development_tasks) != int(config["training_task_count"]):
        raise ValueError("configured training count disagrees with the split")
    if len(evaluation_tasks) != int(config["evaluation_task_count"]):
        raise ValueError("configured evaluation count disagrees with the split")
    if config["evaluation_labels_may_be_used_for_training_or_tuning"] is not False:
        raise ValueError("evaluation-label isolation must remain enabled")

    seed = int(config["models"]["tfidf_logistic_regression"]["random_seed"])
    models = (
        ConstantProfileBaseline(),
        RuleBasedBaseline(),
        TfidfLogisticBaseline(random_seed=seed),
    )
    prediction_sets: list[tuple[str, list[SemanticPrediction]]] = []
    for model in models:
        model.fit(development_tasks, training_annotations)
        predictions = [model.predict(task) for task in evaluation_tasks]
        prediction_sets.append((model.name, predictions))

    reference_manifest = load_json_object(REFERENCE_MANIFEST)
    if reference_manifest["submission"]["sha256"] != sha256(REFERENCES):
        raise ValueError("locked reference checksum disagrees with its manifest")
    raw_reference_records = load_jsonl(REFERENCES)
    if task_ids(raw_reference_records) != evaluation_id_set:
        raise ValueError("references do not exactly match evaluation ids")
    reference_records = ordered_subset(raw_reference_records, evaluation_ids)
    references = tuple(record["annotation"] for record in reference_records)
    schema = load_json_object(SCHEMA)
    result_rows: list[dict[str, Any]] = []
    output_entries: dict[str, dict[str, Any]] = {}
    for model_name, predictions in prediction_sets:
        result_rows.append(
            evaluate_semantic_predictions(
                model_name=model_name,
                tasks=evaluation_tasks,
                references=references,
                predictions=predictions,
                semantic_schema=schema,
            )
        )
        if result_rows[-1]["schema_validity"] != 1.0:
            raise ValueError(f"{model_name} produced schema-invalid predictions")
        output_path = PREDICTIONS / f"{model_name}.v1.jsonl"
        write_jsonl(output_path, prediction_rows(model_name, predictions))
        output_entries[model_name] = {
            "path": output_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(output_path),
            "record_count": len(predictions),
        }

    results: dict[str, Any] = {
        "results_version": "1.0.0",
        "benchmark_version": config["benchmark_version"],
        "training_label_claim": config["training_label_claim"],
        "evaluation_label_claim": config["evaluation_label_claim"],
        "evaluation_labels_used_for_training_or_tuning": False,
        "model_count": len(result_rows),
        "models": result_rows,
    }
    write_json_atomic(RESULTS, results)
    manifest: dict[str, Any] = {
        "manifest_version": "1.0.0",
        "created_on": "2026-09-29",
        "status": "completed_non_llm_baselines",
        "inputs": {
            "tasks": {
                "path": TASKS.relative_to(ROOT).as_posix(),
                "sha256": sha256(TASKS),
            },
            "split": {
                "path": SPLIT.relative_to(ROOT).as_posix(),
                "sha256": sha256(SPLIT),
            },
            "training_labels": {
                "path": TRAINING_LABELS.relative_to(ROOT).as_posix(),
                "sha256": sha256(TRAINING_LABELS),
                "claim": "machine_generated_weak_label",
            },
            "evaluation_references": {
                "path": REFERENCES.relative_to(ROOT).as_posix(),
                "sha256": sha256(REFERENCES),
                "claim": "single_human_reference_not_adjudicated_gold",
            },
            "reference_manifest": {
                "path": REFERENCE_MANIFEST.relative_to(ROOT).as_posix(),
                "sha256": sha256(REFERENCE_MANIFEST),
            },
            "schema": {
                "path": SCHEMA.relative_to(ROOT).as_posix(),
                "sha256": sha256(SCHEMA),
            },
            "config": {
                "path": CONFIG.relative_to(ROOT).as_posix(),
                "sha256": sha256(CONFIG),
            },
        },
        "implementation": {
            "baselines": {
                "path": SOURCE_BASELINES.relative_to(ROOT).as_posix(),
                "sha256": sha256(SOURCE_BASELINES),
            },
            "evaluation": {
                "path": SOURCE_EVALUATION.relative_to(ROOT).as_posix(),
                "sha256": sha256(SOURCE_EVALUATION),
            },
            "experiment": {
                "path": SOURCE_EXPERIMENT.relative_to(ROOT).as_posix(),
                "sha256": sha256(SOURCE_EXPERIMENT),
            },
            "runner": {
                "path": SOURCE_RUNNER.relative_to(ROOT).as_posix(),
                "sha256": sha256(SOURCE_RUNNER),
            },
        },
        "outputs": {
            "results": {
                "path": RESULTS.relative_to(ROOT).as_posix(),
                "sha256": sha256(RESULTS),
            },
            "predictions": output_entries,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "leakage_controls": {
            "evaluation_labels_used_for_fit": False,
            "visible_inputs_only": ["task_text", "user_policy_text"],
            "authoring_metadata_used_by_rule_or_ml": False,
        },
    }
    write_json_atomic(MANIFEST, manifest)
    print(f"results={RESULTS}")
    print(f"results_sha256={sha256(RESULTS)}")
    print(f"manifest={MANIFEST}")
    print(f"manifest_sha256={sha256(MANIFEST)}")
    for result in result_rows:
        print(
            f"{result['model_name']}: "
            f"macro_f1={result['macro_f1_across_fields']:.6f} "
            f"exact_field={result['exact_field_accuracy']:.6f}"
        )


if __name__ == "__main__":
    main()
