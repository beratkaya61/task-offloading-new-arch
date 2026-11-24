# ruff: noqa: E501
"""Evaluate the few-shot local LLM extractor on the frozen v3 evaluation sets.

Few-shot examples come only from the 180 development tasks (machine weak
labels).  Results are scored exactly like ``run_semantic_benchmark_v3.py`` and
compared with the rule baseline on the same tasks (paired bootstrap).

Environment: ``LLM_SETS`` (comma list, default human60 sets), ``LLM_LIMIT``
(smoke test: first N tasks per set), ``LLM_MODEL`` (HF model id), optional
``LLM_REVISION`` (immutable Hugging Face commit hash), and
``LLM_QUANTIZATION`` (``none`` or ``4bit``).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from task_offloading.semantic import load_jsonl, write_jsonl
from task_offloading.semantic.bootstrap import TaskLevelScores, bootstrap_intervals
from task_offloading.semantic.evaluation import evaluate_semantic_predictions
from task_offloading.semantic.experiment import (
    load_json_object,
    ordered_subset,
    write_json_atomic,
)
from task_offloading.semantic.llm import DEFAULT_MODEL_ID, FewShotLlmBaseline

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data" / "semantic_benchmark" / "main"
V3 = MAIN / "benchmark_v3"
REFERENCE = MAIN / "reference_variants" / "annotator_a.protocol_normalized.v1.3.1.jsonl"


def _labels(set_name: str) -> dict[str, dict[str, Any]]:
    if set_name.startswith("human60"):
        return {r["task_id"]: r["annotation"] for r in load_jsonl(REFERENCE)}
    return {r["task_id"]: r["annotation"] for r in load_jsonl(V3 / f"eval_{set_name}.labels.jsonl")}


def main() -> None:
    sets = os.environ.get("LLM_SETS", "human60_original,human60_paraphrase").split(",")
    limit = int(os.environ.get("LLM_LIMIT", "0"))
    subset = os.environ.get("LLM_SUBSET") == "200"
    model_id = os.environ.get("LLM_MODEL", DEFAULT_MODEL_ID)
    model_revision = os.environ.get("LLM_REVISION") or None
    quantization = os.environ.get("LLM_QUANTIZATION", "none")
    split = load_json_object(MAIN / "split_manifest.v1.json")
    dev = ordered_subset(load_jsonl(MAIN / "tasks.v1.jsonl"), list(split["machine_labeled_development"]["task_ids"]))
    weak = {r["task_id"]: r["annotation"] for r in load_jsonl(MAIN / "preannotations" / "machine_development.v1.jsonl")}
    schema = load_json_object(ROOT / "schemas" / "semantic_requirements.schema.json")
    load_started = time.perf_counter()
    print(
        f"loading model={model_id} revision={model_revision or 'main'}",
        flush=True,
    )
    model = FewShotLlmBaseline(
        cache_folder=ROOT / "models" / "huggingface",
        model_id=model_id,
        revision=model_revision,
        quantization=quantization,
    )
    model.fit(dev, [weak[t["task_id"]] for t in dev])
    print(f"model and shared prefix ready ({time.perf_counter() - load_started:.1f}s)", flush=True)
    results: dict[str, Any] = {}
    for set_name in sets:
        tasks = list(load_jsonl(V3 / f"eval_{set_name}.tasks.jsonl"))
        if subset and set_name.startswith("compositional600"):
            # Pre-registered subset: same draw as build_independent_paraphrases.
            chosen = np.random.default_rng(20261120).choice(len(tasks), size=200, replace=False)
            tasks = [tasks[int(i)] for i in sorted(chosen.tolist())]
        if limit:
            tasks = tasks[:limit]
        labels = _labels(set_name)
        started = time.perf_counter()
        predictions = []
        for index, task in enumerate(tasks, 1):
            predictions.append(model.predict(task))
            if index % 10 == 0 or len(tasks) <= 10:
                print(f"{set_name}: {index}/{len(tasks)} ({time.perf_counter() - started:.0f}s)", flush=True)
        suffix = f".limit{limit}" if limit else (".subset200" if subset and set_name.startswith("compositional600") else "")
        write_jsonl(
            V3 / "predictions" / f"{model.name}.{set_name}{suffix}.jsonl",
            [{"task_id": t["task_id"], "model_name": model.name, "annotation": p.annotation} for t, p in zip(tasks, predictions, strict=True)],
        )
        refs = [labels[t["task_id"]] for t in tasks]
        full = evaluate_semantic_predictions(model_name=model.name, tasks=tasks, references=refs, predictions=predictions, semantic_schema=schema)
        rule_rows = {r["task_id"]: r["annotation"] for r in load_jsonl(V3 / "predictions" / f"rule_keyword.{set_name}.jsonl")}
        results_key = set_name + suffix
        intervals = bootstrap_intervals(
            {
                model.name: TaskLevelScores(refs, [p.annotation for p in predictions]),
                "rule_keyword": TaskLevelScores(refs, [rule_rows[t["task_id"]] for t in tasks]),
            },
            baseline="rule_keyword",
        )
        results[results_key] = {
            "task_count": len(tasks),
            "exact_record_accuracy": full["exact_record_accuracy"],
            "macro_f1_across_fields": full["macro_f1_across_fields"],
            "critical_privacy": full["critical_privacy"],
            "abstention_f1": full["abstention"]["f1"],
            "inference_latency_ms_mean": full["inference_latency_ms"]["mean"],
            "fields_macro_f1": {f: v["macro_f1"] for f, v in full["fields"].items()},
            "intervals": intervals,
        }
        ci = intervals[model.name]["macro_f1_across_fields"]
        print(f"{set_name}: macroF1={ci['value']:.3f} CI={ci['ci95']} diff_vs_rule={ci['diff_vs_baseline']['value']:+.3f} exact={full['exact_record_accuracy']:.3f}", flush=True)
    result_name = (
        f"results_{model.name}"
        f"{'_smoke' if limit else ''}_{'_'.join(sets)}"
        f"{'_subset200' if subset else ''}.json"
    )
    write_json_atomic(
        V3 / result_name,
        {
            "model_id": model_id,
            "requested_revision": model_revision,
            "extractor": model.name,
            "shots": 4,
            "decoding": "greedy",
            "few_shot_source": "development split, machine weak labels",
            "evaluation_labels_used_for_training_or_tuning": False,
            "calibration_note": "constant confidence; ECE/Brier not interpretable",
            "unparsable_outputs": model.unparsable_count,
            "out_of_vocabulary_fields": model.invalid_field_count,
            "sets": results,
        },
    )


if __name__ == "__main__":
    main()
