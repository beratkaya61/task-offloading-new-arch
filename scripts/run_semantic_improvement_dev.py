"""Development-only ablation for post-v3 exploratory improvements.

No evaluation task or human-reference label is loaded.  The 180 machine-labeled
development tasks are split by scenario family; augmentation is built only from
the training families.  The held-out families are evaluated in original,
controlled-paraphrase and compositional wording.

Environment ``IMPROVEMENT_MODE`` may be ``e5``, ``llm`` or ``all`` (default).
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from task_offloading.semantic import load_jsonl
from task_offloading.semantic.compositional import compose_tasks
from task_offloading.semantic.evaluation import evaluate_semantic_predictions
from task_offloading.semantic.experiment import (
    load_json_object,
    ordered_subset,
    write_json_atomic,
)
from task_offloading.semantic.finetune import FineTunedEncoderBaseline
from task_offloading.semantic.llm import FewShotLlmBaseline
from task_offloading.semantic.paraphrase import paraphrase_task

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data" / "semantic_benchmark" / "main"
SCHEMA = ROOT / "schemas" / "semantic_requirements.schema.json"
CACHE = ROOT / "models" / "huggingface"
OUTPUT = ROOT / "artifacts" / "semantic_improvement_dev" / "results.json"
SEED = 20261120


def _split_by_family(
    tasks: list[dict[str, Any]], validation_fraction: float = 0.2
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    families = sorted({str(t["grouping"]["scenario_family_id"]) for t in tasks})
    rng = np.random.default_rng(SEED + 900)
    shuffled = [families[int(i)] for i in rng.permutation(len(families))]
    count = max(1, round(len(families) * validation_fraction))
    validation_families = set(shuffled[:count])
    train = [
        t
        for t in tasks
        if str(t["grouping"]["scenario_family_id"]) not in validation_families
    ]
    validation = [
        t
        for t in tasks
        if str(t["grouping"]["scenario_family_id"]) in validation_families
    ]
    return train, validation, sorted(validation_families)


def _augment(
    tasks: list[dict[str, Any]], labels: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    augmented_tasks = list(tasks)
    augmented_labels = [labels[t["task_id"]] for t in tasks]
    for variant in range(4):
        for task in tasks:
            rewritten, applied = paraphrase_task(
                task, pool="train", variant=variant
            )
            if applied:
                augmented_tasks.append(rewritten)
                augmented_labels.append(labels[task["task_id"]])
    for wording, variants, seed in (
        ("train", 4, SEED),
        ("canonical", 2, SEED + 1),
    ):
        composed, composed_labels = compose_tasks(
            tasks,
            variants_per_task=variants,
            wording=wording,  # type: ignore[arg-type]
            seed=seed,
            prefix=f"improve_train_{wording}",
        )
        augmented_tasks.extend(composed)
        augmented_labels.extend(composed_labels[t["task_id"]] for t in composed)
    return augmented_tasks, augmented_labels


def _validation_sets(
    tasks: list[dict[str, Any]], labels: dict[str, dict[str, Any]]
) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    original_labels = [labels[t["task_id"]] for t in tasks]
    paraphrased = [
        paraphrase_task(task, pool="train", variant=7)[0] for task in tasks
    ]
    composed, composed_labels = compose_tasks(
        tasks,
        variants_per_task=3,
        wording="train",
        seed=SEED + 901,
        prefix="improve_validation",
    )
    return {
        "development_original": (tasks, original_labels),
        "development_paraphrase": (paraphrased, original_labels),
        "development_compositional": (
            composed,
            [composed_labels[t["task_id"]] for t in composed],
        ),
    }


def _score(
    model: Any,
    sets: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    schema: dict[str, Any],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for set_name, (tasks, references) in sets.items():
        predictions = [model.predict(task) for task in tasks]
        full = evaluate_semantic_predictions(
            model_name=model.name,
            tasks=tasks,
            references=references,
            predictions=predictions,
            semantic_schema=schema,
        )
        results[set_name] = {
            "task_count": len(tasks),
            "macro_f1_across_fields": full["macro_f1_across_fields"],
            "exact_record_accuracy": full["exact_record_accuracy"],
            "critical_privacy": full["critical_privacy"],
            "fields_macro_f1": {
                field: value["macro_f1"]
                for field, value in full["fields"].items()
            },
            "inference_latency_ms_mean": full["inference_latency_ms"]["mean"],
        }
        print(
            f"{model.name:48s} {set_name:30s} "
            f"macroF1={full['macro_f1_across_fields']:.3f}",
            flush=True,
        )
    if hasattr(model, "invalid_field_count"):
        results["generation_diagnostics"] = {
            "invalid_field_count": model.invalid_field_count,
            "unparsable_count": model.unparsable_count,
        }
    return results


def main() -> None:
    mode = os.environ.get("IMPROVEMENT_MODE", "all").lower()
    if mode not in {"e5", "llm", "all"}:
        raise ValueError("IMPROVEMENT_MODE must be e5, llm or all")
    started = time.perf_counter()
    split = load_json_object(MAIN / "split_manifest.v1.json")
    tasks = list(
        ordered_subset(
            load_jsonl(MAIN / "tasks.v1.jsonl"),
            list(split["machine_labeled_development"]["task_ids"]),
        )
    )
    labels = {
        row["task_id"]: row["annotation"]
        for row in load_jsonl(MAIN / "preannotations" / "machine_development.v1.jsonl")
    }
    train, validation, validation_families = _split_by_family(tasks)
    augmented_tasks, augmented_labels = _augment(train, labels)
    validation_sets = _validation_sets(validation, labels)
    schema = load_json_object(SCHEMA)

    candidates: list[tuple[str, Any, list[dict[str, Any]], list[dict[str, Any]]]] = []
    if mode in {"e5", "all"}:
        candidates.extend(
            [
                (
                    "e5_standard_control",
                    FineTunedEncoderBaseline(cache_folder=CACHE, random_seed=SEED),
                    augmented_tasks,
                    augmented_labels,
                ),
                (
                    "e5_gradual",
                    FineTunedEncoderBaseline(
                        cache_folder=CACHE,
                        random_seed=SEED,
                        training_strategy="gradual",
                    ),
                    augmented_tasks,
                    augmented_labels,
                ),
            ]
        )
    if mode in {"llm", "all"}:
        base_labels = [labels[t["task_id"]] for t in train]
        candidates.extend(
            [
                (
                    "qwen_signature_4shot_control",
                    FewShotLlmBaseline(cache_folder=CACHE),
                    train,
                    base_labels,
                ),
                (
                    "qwen_codes_4shot",
                    FewShotLlmBaseline(
                        cache_folder=CACHE,
                        label_encoding="codes",
                        max_new_tokens=96,
                    ),
                    train,
                    base_labels,
                ),
                (
                    "qwen_codes_8shot",
                    FewShotLlmBaseline(
                        cache_folder=CACHE,
                        shots=8,
                        label_encoding="codes",
                        max_new_tokens=96,
                    ),
                    train,
                    base_labels,
                ),
            ]
        )

    results: dict[str, Any] = {}
    for candidate_name, model, fit_tasks, fit_labels in candidates:
        print(f"fitting {candidate_name}", flush=True)
        model.fit(fit_tasks, fit_labels)
        results[candidate_name] = _score(model, validation_sets, schema)
        del model
        gc.collect()

    write_json_atomic(
        OUTPUT,
        {
            "protocol": "docs/EXPLORATORY_IMPROVEMENT_PROTOCOL_V1.md",
            "evaluation_labels_used": False,
            "label_claim": "development machine weak and label-by-construction only",
            "selection_seed": SEED + 900,
            "training_tasks_base": len(train),
            "training_tasks_augmented": len(augmented_tasks),
            "validation_tasks_base": len(validation),
            "validation_scenario_families": validation_families,
            "mode": mode,
            "elapsed_s": time.perf_counter() - started,
            "results": results,
        },
    )
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
