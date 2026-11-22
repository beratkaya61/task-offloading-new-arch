# ruff: noqa: E501
"""Paper-grade semantic extraction benchmark (v3) with bootstrap 95% CIs.

Training data (all machine-labeled; no evaluation label is ever used):
  * ``base``      — 180 development tasks with weak template labels.
  * ``augmented`` — base + 4 train-pool paraphrases per task + compositional
    tasks built from development domain sentences (train-pool and canonical
    wording, labels by construction).

Evaluation sets:
  * ``human60_original`` / ``human60_paraphrase`` — the 60 locked tasks vs the
    protocol-normalized single-human reference (original wording and
    held-out test-pool paraphrases).
  * ``compositional600_canonical`` / ``compositional600_paraphrase`` — 600
    machine-generated tasks composed from *evaluation-split* domain sentences,
    labels by construction, in template wording and in held-out wording.
  * ``independent_paraphrase`` — 188 independently generated and automatically
    filtered paraphrases with labels transferred under an invariance assumption;
    these texts are machine-generated and were not human re-annotated.

Every model is scored on the same tasks; CIs resample tasks (1000 draws) and
paired differences are reported against the rule baseline.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from task_offloading.semantic import load_jsonl, write_jsonl
from task_offloading.semantic.baselines import (
    ConstantProfileBaseline,
    RuleBasedBaseline,
    SemanticPrediction,
    TfidfLogisticBaseline,
)
from task_offloading.semantic.bootstrap import TaskLevelScores, bootstrap_intervals
from task_offloading.semantic.compositional import compose_tasks
from task_offloading.semantic.confidence_calibration import FieldConfidenceCalibrator
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
OUTPUT = MAIN / "benchmark_v3"
SCHEMA = ROOT / "schemas" / "semantic_requirements.schema.json"
REFERENCE = MAIN / "reference_variants" / "annotator_a.protocol_normalized.v1.3.1.jsonl"
SEED = 20261120
CACHE = ROOT / "models" / "huggingface"
SKIP_FINETUNE = os.environ.get("SKIP_FINETUNE") == "1"


def main() -> None:
    started = time.perf_counter()
    split = load_json_object(MAIN / "split_manifest.v1.json")
    all_tasks = load_jsonl(MAIN / "tasks.v1.jsonl")
    dev = ordered_subset(
        all_tasks, list(split["machine_labeled_development"]["task_ids"])
    )
    evaluation = ordered_subset(
        all_tasks, list(split["single_human_evaluation"]["task_ids"])
    )
    weak = {
        row["task_id"]: row["annotation"]
        for row in load_jsonl(MAIN / "preannotations" / "machine_development.v1.jsonl")
    }
    reference = {row["task_id"]: row["annotation"] for row in load_jsonl(REFERENCE)}
    schema = load_json_object(SCHEMA)

    # ------------------------------------------------------------- training
    base_tasks = list(dev)
    base_labels = [weak[t["task_id"]] for t in dev]
    aug_tasks = list(base_tasks)
    aug_labels = list(base_labels)
    for variant in range(4):
        for task, label in zip(dev, base_labels, strict=True):
            rewritten, applied = paraphrase_task(task, pool="train", variant=variant)
            if applied:
                aug_tasks.append(rewritten)
                aug_labels.append(label)
    for wording, variants, seed in (("train", 4, SEED), ("canonical", 2, SEED + 1)):
        comp_tasks, comp_labels = compose_tasks(
            dev,
            variants_per_task=variants,
            wording=wording,  # type: ignore[arg-type]
            seed=seed,
            prefix=f"ctrain_{wording}",
        )
        aug_tasks.extend(comp_tasks)
        aug_labels.extend(comp_labels[t["task_id"]] for t in comp_tasks)

    # ------------------------------------------------------------ evaluation
    para60 = [paraphrase_task(t, pool="test")[0] for t in evaluation]
    comp_canon, comp_canon_labels = compose_tasks(
        evaluation,
        variants_per_task=10,
        wording="canonical",
        seed=SEED + 7,
        prefix="ctest",
    )
    comp_para, comp_para_labels = compose_tasks(
        evaluation,
        variants_per_task=10,
        wording="test",
        seed=SEED + 7,
        prefix="ctest",
    )
    eval_sets: dict[str, tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]] = {
        "human60_original": (list(evaluation), reference),
        "human60_paraphrase": (para60, reference),
        "compositional600_canonical": (comp_canon, comp_canon_labels),
        "compositional600_paraphrase": (comp_para, comp_para_labels),
    }
    # Calibration set: development-split domain sentences, a fresh seed and
    # both train-pool and template wording; never used to fit any extractor.
    cal_tasks: list[dict[str, Any]] = []
    cal_labels: dict[str, dict[str, Any]] = {}
    for wording, seed in (("train", SEED + 500), ("canonical", SEED + 501)):
        tasks_w, labels_w = compose_tasks(
            dev,
            variants_per_task=1,
            wording=wording,  # type: ignore[arg-type]
            seed=seed,
            prefix=f"ccal_{wording}",
        )
        cal_tasks.extend(tasks_w)
        cal_labels.update(labels_w)
    independent = OUTPUT / "eval_independent_paraphrase.tasks.jsonl"
    if independent.exists():
        independent_labels = {
            r["task_id"]: r["annotation"]
            for r in load_jsonl(OUTPUT / "eval_independent_paraphrase.labels.jsonl")
        }
        eval_sets["independent_paraphrase"] = (
            list(load_jsonl(independent)),
            independent_labels,
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT / "train_augmented_tasks.jsonl", aug_tasks)
    write_jsonl(
        OUTPUT / "train_augmented_labels.jsonl",
        [
            {"row": i, "task_id": t["task_id"], "annotation": a}
            for i, (t, a) in enumerate(zip(aug_tasks, aug_labels, strict=True))
        ],
    )
    write_jsonl(
        OUTPUT / "calibration_labels.jsonl",
        [{"task_id": t["task_id"], "annotation": cal_labels[t["task_id"]]} for t in cal_tasks],
    )
    write_jsonl(OUTPUT / "calibration_tasks.jsonl", cal_tasks)
    for name, (tasks, labels) in eval_sets.items():
        write_jsonl(OUTPUT / f"eval_{name}.tasks.jsonl", tasks)
        if name.startswith(
            "compositional"
        ):  # independent set is written by its own script
            write_jsonl(
                OUTPUT / f"eval_{name}.labels.jsonl",
                [
                    {"task_id": t["task_id"], "annotation": labels[t["task_id"]]}
                    for t in tasks
                ],
            )
    print(
        f"train base={len(base_tasks)} augmented={len(aug_tasks)}; "
        f"setup {time.perf_counter() - started:.0f}s",
        flush=True,
    )

    encoder = load_e5_small(cache_folder=CACHE)
    models: list[tuple[str, Any, list[dict[str, Any]], list[dict[str, Any]]]] = [
        ("constant_profile", ConstantProfileBaseline(), base_tasks, base_labels),
        ("rule_keyword", RuleBasedBaseline(), base_tasks, base_labels),
        ("tfidf_lr", TfidfLogisticBaseline(random_seed=SEED), base_tasks, base_labels),
        (
            "tfidf_lr_augmented",
            TfidfLogisticBaseline(random_seed=SEED),
            aug_tasks,
            aug_labels,
        ),
        (
            "e5_frozen_lr",
            FrozenEncoderLogisticBaseline(encoder, random_seed=SEED),
            base_tasks,
            base_labels,
        ),
        (
            "e5_frozen_lr_augmented",
            FrozenEncoderLogisticBaseline(encoder, random_seed=SEED),
            aug_tasks,
            aug_labels,
        ),
    ]
    if not SKIP_FINETUNE:
        models.append(
            (
                "e5_finetuned_augmented",
                FineTunedEncoderBaseline(cache_folder=CACHE, random_seed=SEED),
                aug_tasks,
                aug_labels,
            )
        )

    point: dict[str, Any] = {}
    task_scores: dict[str, dict[str, TaskLevelScores]] = {
        name: {} for name in eval_sets
    }
    for model_name, model, fit_tasks, fit_labels in models:
        model.fit(fit_tasks, fit_labels)
        cal_predictions = [model.predict(task).annotation for task in cal_tasks]
        calibrator = FieldConfidenceCalibrator().fit(
            [cal_labels[t["task_id"]] for t in cal_tasks], cal_predictions
        )
        point[model_name] = {
            "training_task_count": len(fit_tasks),
            "calibration_task_count": calibrator.fitted_on,
            "sets": {},
        }
        for set_name, (tasks, labels) in eval_sets.items():
            raw_predictions = [model.predict(task) for task in tasks]
            raw_eval = evaluate_semantic_predictions(
                model_name=model_name,
                tasks=tasks,
                references=[labels[t["task_id"]] for t in tasks],
                predictions=raw_predictions,
                semantic_schema=schema,
            )
            write_jsonl(
                OUTPUT / "predictions_raw" / f"{model_name}.{set_name}.jsonl",
                [
                    {
                        "task_id": t["task_id"],
                        "model_name": model_name,
                        "annotation": p.annotation,
                    }
                    for t, p in zip(tasks, raw_predictions, strict=True)
                ],
            )
            predictions = [
                SemanticPrediction(calibrator.apply(p.annotation), p.inference_ms)
                for p in raw_predictions
            ]
            write_jsonl(
                OUTPUT / "predictions" / f"{model_name}.{set_name}.jsonl",
                [
                    {
                        "task_id": t["task_id"],
                        "model_name": model_name,
                        "annotation": p.annotation,
                    }
                    for t, p in zip(tasks, predictions, strict=True)
                ],
            )
            refs = [labels[t["task_id"]] for t in tasks]
            full = evaluate_semantic_predictions(
                model_name=model_name,
                tasks=tasks,
                references=refs,
                predictions=predictions,
                semantic_schema=schema,
            )
            point[model_name]["sets"][set_name] = {
                "exact_record_accuracy": full["exact_record_accuracy"],
                "exact_field_accuracy": full["exact_field_accuracy"],
                "macro_f1_across_fields": full["macro_f1_across_fields"],
                "schema_validity": full["schema_validity"],
                "abstention_f1": full["abstention"]["f1"],
                "critical_privacy": full["critical_privacy"],
                "calibration": full["calibration"],
                "calibration_raw": raw_eval["calibration"],
                "inference_latency_ms_mean": full["inference_latency_ms"]["mean"],
                "fields_macro_f1": {
                    f: v["macro_f1"] for f, v in full["fields"].items()
                },
            }
            task_scores[set_name][model_name] = TaskLevelScores(
                refs, [p.annotation for p in predictions]
            )
            print(
                f"{model_name:24s} {set_name:28s} exact={full['exact_record_accuracy']:.3f} "
                f"macroF1={full['macro_f1_across_fields']:.3f} "
                f"({time.perf_counter() - started:.0f}s)",
                flush=True,
            )

    intervals = {
        set_name: bootstrap_intervals(scores, baseline="rule_keyword")
        for set_name, scores in task_scores.items()
    }
    write_json_atomic(
        OUTPUT / "results.v3.json",
        {
            "results_version": "3.2.0",
            "revision_note": (
                "3.1 withdraws train-pool paraphrases that were written after "
                "the test pool and first results had been seen (leakage). "
                "3.2 adds per-field isotonic confidence calibration fitted on a "
                "separate machine-generated calibration set; labels unchanged."
            ),
            "design": __doc__,
            "reference_sha256": sha256(REFERENCE),
            "evaluation_labels_used_for_training_or_tuning": False,
            "label_claims": {
                "human60_original": "single_human_reference_protocol_normalized",
                "human60_paraphrase": (
                    "machine_paraphrased_text_with_human_labels_transferred_"
                    "under_invariance_assumption_not_human_annotated"
                ),
                "compositional600_*": "machine_generated_label_by_construction",
                "independent_paraphrase": (
                    "machine_generated_paraphrase_with_labels_transferred_under_"
                    "an_invariance_assumption_not_human_annotated"
                ),
                "training": "machine_generated_weak_and_constructed_labels",
            },
            "set_sizes": {name: len(tasks) for name, (tasks, _) in eval_sets.items()},
            "bootstrap": {
                "resamples": 1000,
                "interval": "percentile 95%",
                "unit": "task",
            },
            "point_estimates": point,
            "intervals": intervals,
        },
    )
    print(f"done in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
