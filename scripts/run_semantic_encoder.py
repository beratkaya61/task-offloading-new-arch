"""Run the revision-pinned frozen E5-small semantic baseline."""

from __future__ import annotations

import importlib.metadata
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from task_offloading.semantic import load_jsonl, write_jsonl
from task_offloading.semantic.encoder import (
    E5_SMALL_MODEL_ID,
    E5_SMALL_REVISION,
    FrozenEncoderLogisticBaseline,
    load_e5_small,
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
CONFIG = ROOT / "configs" / "semantic_encoder.v1.json"
SOURCE_BASELINES = ROOT / "src" / "task_offloading" / "semantic" / "baselines.py"
SOURCE_ENCODER = ROOT / "src" / "task_offloading" / "semantic" / "encoder.py"
SOURCE_EVALUATION = ROOT / "src" / "task_offloading" / "semantic" / "evaluation.py"
SOURCE_EXPERIMENT = ROOT / "src" / "task_offloading" / "semantic" / "experiment.py"
SOURCE_RUNNER = ROOT / "scripts" / "run_semantic_encoder.py"
OUTPUT = MAIN / "baselines" / "encoder"
PREDICTIONS = OUTPUT / "predictions.v1.jsonl"
RESULTS = OUTPUT / "results.v1.json"
MANIFEST = OUTPUT / "manifest.v1.json"


def _validate_model_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("encoder config lacks model settings")
    if model.get("model_id") != E5_SMALL_MODEL_ID:
        raise ValueError("encoder model id disagrees with pinned implementation")
    if model.get("revision") != E5_SMALL_REVISION:
        raise ValueError("encoder revision disagrees with pinned implementation")
    if model.get("encoder_frozen") is not True:
        raise ValueError("encoder must remain frozen for this low-data baseline")
    return model


def main() -> None:
    config = load_json_object(CONFIG)
    model_config = _validate_model_config(config)
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
    if config["evaluation_labels_may_be_used_for_training_or_tuning"] is not False:
        raise ValueError("evaluation-label isolation must remain enabled")

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

    cache_folder = ROOT / str(model_config["cache_folder"])
    encoder = load_e5_small(
        cache_folder=cache_folder,
        device=str(model_config["device"]),
    )
    model = FrozenEncoderLogisticBaseline(
        encoder,
        random_seed=int(model_config["random_seed"]),
        batch_size=int(model_config["batch_size"]),
    )
    fit_started = time.perf_counter()
    model.fit(development_tasks, training_annotations)
    fit_duration_seconds = time.perf_counter() - fit_started
    predictions = [model.predict(task) for task in evaluation_tasks]

    reference_manifest = load_json_object(REFERENCE_MANIFEST)
    if reference_manifest["submission"]["sha256"] != sha256(REFERENCES):
        raise ValueError("locked reference checksum disagrees with its manifest")
    raw_reference_records = load_jsonl(REFERENCES)
    if task_ids(raw_reference_records) != evaluation_id_set:
        raise ValueError("references do not exactly match evaluation ids")
    reference_records = ordered_subset(raw_reference_records, evaluation_ids)
    references = tuple(record["annotation"] for record in reference_records)
    schema = load_json_object(SCHEMA)
    result = evaluate_semantic_predictions(
        model_name=model.name,
        tasks=evaluation_tasks,
        references=references,
        predictions=predictions,
        semantic_schema=schema,
    )
    if result["schema_validity"] != 1.0:
        raise ValueError("encoder produced schema-invalid predictions")
    result["fit_duration_seconds"] = fit_duration_seconds
    result["encoder_frozen"] = True

    write_jsonl(PREDICTIONS, prediction_rows(model.name, predictions))
    results: dict[str, Any] = {
        "results_version": "1.0.0",
        "experiment_version": config["experiment_version"],
        "training_label_claim": config["training_label_claim"],
        "evaluation_label_claim": config["evaluation_label_claim"],
        "evaluation_labels_used_for_training_or_tuning": False,
        "model": result,
    }
    write_json_atomic(RESULTS, results)

    implementation_files = (
        SOURCE_BASELINES,
        SOURCE_ENCODER,
        SOURCE_EVALUATION,
        SOURCE_EXPERIMENT,
        SOURCE_RUNNER,
    )
    manifest: dict[str, Any] = {
        "manifest_version": "1.0.0",
        "created_on": "2026-09-29",
        "status": "completed_frozen_encoder_baseline",
        "model": {
            "model_id": E5_SMALL_MODEL_ID,
            "requested_revision": E5_SMALL_REVISION,
            "license": model_config["license"],
            "source": model_config["source"],
            "encoder_frozen": True,
            "cache_path": cache_folder.relative_to(ROOT).as_posix(),
            "cache_committed_to_git": False,
        },
        "inputs": {
            path.name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
            }
            for path in (
                TASKS,
                SPLIT,
                TRAINING_LABELS,
                REFERENCES,
                REFERENCE_MANIFEST,
                SCHEMA,
                CONFIG,
            )
        },
        "implementation": {
            path.name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
            }
            for path in implementation_files
        },
        "outputs": {
            "predictions": {
                "path": PREDICTIONS.relative_to(ROOT).as_posix(),
                "sha256": sha256(PREDICTIONS),
                "record_count": len(predictions),
            },
            "results": {
                "path": RESULTS.relative_to(ROOT).as_posix(),
                "sha256": sha256(RESULTS),
            },
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sentence_transformers": importlib.metadata.version(
                "sentence-transformers"
            ),
        },
        "leakage_controls": {
            "evaluation_labels_loaded_after_predictions": True,
            "visible_inputs_only": ["task_text", "user_policy_text"],
            "authoring_metadata_used": False,
        },
    }
    write_json_atomic(MANIFEST, manifest)
    print(f"results={RESULTS}")
    print(f"results_sha256={sha256(RESULTS)}")
    print(f"manifest={MANIFEST}")
    print(f"manifest_sha256={sha256(MANIFEST)}")
    print(
        f"{model.name}: macro_f1={result['macro_f1_across_fields']:.6f} "
        f"exact_field={result['exact_field_accuracy']:.6f}"
    )


if __name__ == "__main__":
    main()
