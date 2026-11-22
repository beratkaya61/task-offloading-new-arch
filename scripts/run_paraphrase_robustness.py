"""Paraphrase-robustness experiment for every local semantic extractor.

* Training: the 180 machine-labeled development tasks.  ``*_augmented`` models
  additionally see two paraphrased copies per task drawn from the *train*
  paraphrase pool, with the same weak labels (labels are invariant).
* Evaluation: the 60 locked tasks, original wording and *test*-pool
  paraphrases, scored against the protocol-normalized reference produced by
  ``scripts/evaluate_reference_variants.py``.  The test pool is disjoint from
  the train pool, and no evaluation label is used for fitting or selection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from task_offloading.semantic import load_jsonl, write_jsonl
from task_offloading.semantic.baselines import (
    ConstantProfileBaseline,
    RuleBasedBaseline,
    SemanticPrediction,
    TfidfLogisticBaseline,
)
from task_offloading.semantic.encoder import (
    FrozenEncoderLogisticBaseline,
    load_e5_small,
)
from task_offloading.semantic.evaluation import evaluate_semantic_predictions
from task_offloading.semantic.experiment import (
    load_json_object,
    ordered_subset,
    sha256,
    write_json_atomic,
)
from task_offloading.semantic.finetune import FineTunedEncoderBaseline
from task_offloading.semantic.paraphrase import paraphrase_task

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data" / "semantic_benchmark" / "main"
TASKS = MAIN / "tasks.v1.jsonl"
SPLIT = MAIN / "split_manifest.v1.json"
TRAINING_LABELS = MAIN / "preannotations" / "machine_development.v1.jsonl"
REFERENCE = MAIN / "reference_variants" / "annotator_a.protocol_normalized.v1.3.1.jsonl"
SCHEMA = ROOT / "schemas" / "semantic_requirements.schema.json"
OUTPUT = MAIN / "paraphrase_robustness"
PERTURBED_TASKS = OUTPUT / "evaluation_tasks.test_paraphrase.v1.jsonl"
RESULTS = OUTPUT / os.environ.get("RESULTS_NAME", "results.v1.json")
SEED = 20260929
AUGMENTED_VARIANTS = int(os.environ.get("AUGMENTED_VARIANTS", "4"))


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "exact_record_accuracy": result["exact_record_accuracy"],
        "exact_field_accuracy": result["exact_field_accuracy"],
        "micro_f1": result["micro_f1"],
        "macro_f1_across_fields": result["macro_f1_across_fields"],
        "schema_validity": result["schema_validity"],
        "abstention_f1": result["abstention"]["f1"],
        "critical_privacy_precision": result["critical_privacy"]["precision"],
        "critical_privacy_recall": result["critical_privacy"]["recall"],
        "critical_privacy_f1": result["critical_privacy"]["f1"],
        "calibration_ece_10_bin": result["calibration"]["ece_10_bin"],
        "calibration_brier": result["calibration"]["brier"],
        "inference_latency_ms_mean": result["inference_latency_ms"]["mean"],
        "fields_macro_f1": {
            field: values["macro_f1"] for field, values in result["fields"].items()
        },
    }


def main() -> None:
    split = load_json_object(SPLIT)
    all_tasks = load_jsonl(TASKS)
    development_ids = list(split["machine_labeled_development"]["task_ids"])
    evaluation_ids = list(split["single_human_evaluation"]["task_ids"])
    development = list(ordered_subset(all_tasks, development_ids))
    evaluation = list(ordered_subset(all_tasks, evaluation_ids))
    labels = {row["task_id"]: row["annotation"] for row in load_jsonl(TRAINING_LABELS)}
    train_annotations = [labels[task["task_id"]] for task in development]
    references = {row["task_id"]: row["annotation"] for row in load_jsonl(REFERENCE)}
    schema = load_json_object(SCHEMA)

    augmented_tasks = list(development)
    augmented_annotations = list(train_annotations)
    for variant in range(AUGMENTED_VARIANTS):
        for task, annotation in zip(development, train_annotations, strict=True):
            rewritten, applied = paraphrase_task(task, pool="train", variant=variant)
            if applied:
                augmented_tasks.append(rewritten)
                augmented_annotations.append(annotation)

    perturbed = [paraphrase_task(task, pool="test")[0] for task in evaluation]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(PERTURBED_TASKS, perturbed)

    encoder = load_e5_small(cache_folder=ROOT / "models" / "huggingface")
    models: list[tuple[str, Any, list[dict[str, Any]], list[dict[str, Any]]]] = [
        ("constant_profile", ConstantProfileBaseline(), development, train_annotations),
        ("rule_keyword", RuleBasedBaseline(), development, train_annotations),
        (
            "tfidf_logistic_regression",
            TfidfLogisticBaseline(random_seed=SEED),
            development,
            train_annotations,
        ),
        (
            "tfidf_logistic_regression_augmented",
            TfidfLogisticBaseline(random_seed=SEED),
            augmented_tasks,
            augmented_annotations,
        ),
        (
            "e5_small_logreg",
            FrozenEncoderLogisticBaseline(encoder, random_seed=SEED),
            development,
            train_annotations,
        ),
        (
            "e5_small_logreg_augmented",
            FrozenEncoderLogisticBaseline(encoder, random_seed=SEED),
            augmented_tasks,
            augmented_annotations,
        ),
        (
            "e5_small_finetuned_augmented",
            FineTunedEncoderBaseline(
                cache_folder=ROOT / "models" / "huggingface", random_seed=SEED
            ),
            augmented_tasks,
            augmented_annotations,
        ),
    ]

    eval_sets = {"original_wording": evaluation, "test_paraphrase": perturbed}
    results: dict[str, Any] = {}
    print(f"{'model':38s} {'set':18s} exact  macroF1  fieldAcc")
    for name, model, fit_tasks, fit_annotations in models:
        model.fit(fit_tasks, fit_annotations)
        results[name] = {"training_task_count": len(fit_tasks), "sets": {}}
        for set_name, tasks in eval_sets.items():
            predictions: list[SemanticPrediction] = [model.predict(t) for t in tasks]
            full = evaluate_semantic_predictions(
                model_name=name,
                tasks=tasks,
                references=[references[t["task_id"]] for t in tasks],
                predictions=predictions,
                semantic_schema=schema,
            )
            results[name]["sets"][set_name] = _summary(full)
            print(
                f"{name:38s} {set_name:18s} "
                f"{full['exact_record_accuracy']:.3f}  "
                f"{full['macro_f1_across_fields']:.3f}    "
                f"{full['exact_field_accuracy']:.3f}"
            )

    write_json_atomic(
        RESULTS,
        {
            "results_version": "1.0.0",
            "reference": REFERENCE.relative_to(ROOT).as_posix(),
            "reference_sha256": sha256(REFERENCE),
            "reference_claim": "single_human_reference_protocol_normalized",
            "perturbed_tasks_sha256": sha256(PERTURBED_TASKS),
            "perturbed_set_claim": "machine_generated_label_preserving_paraphrase",
            "train_and_test_paraphrase_pools_disjoint": True,
            "evaluation_labels_used_for_training_or_tuning": False,
            "augmented_variants_per_task": AUGMENTED_VARIANTS,
            "models": results,
        },
    )


if __name__ == "__main__":
    main()
