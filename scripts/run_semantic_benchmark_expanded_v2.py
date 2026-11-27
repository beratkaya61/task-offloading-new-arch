# ruff: noqa: E501
"""Provisional equal-supervision comparison on the context-disjoint expansion v2.

Every supervised model is fitted on exactly the same ``expanded_v2`` train rows
(construction labels) and scored on the same validation/test rows plus the
locked historical external sets.  Hyperparameters are frozen from benchmark v3;
validation is reported for transparency and is *not* used for any selection,
and test never influences anything.

Status: PROVISIONAL.  Expanded-v2 labels are construction labels pending blind
human annotation/adjudication; the final table is produced by re-running this
script on the adjudicated label file.

Environment: ``SKIP_FINETUNE=1`` skips the fine-tuned encoder (GPU-heavy).
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
    TfidfLogisticBaseline,
)
from task_offloading.semantic.bootstrap import TaskLevelScores, bootstrap_intervals
from task_offloading.semantic.encoder import (
    FrozenEncoderLogisticBaseline,
    load_e5_small,
)
from task_offloading.semantic.evaluation import evaluate_semantic_predictions
from task_offloading.semantic.experiment import (
    load_json_object,
    sha256,
    write_json_atomic,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "semantic_benchmark" / "expanded_v2"
LABELS = Path(os.environ.get("EXPANDED_LABELS", DATA / "candidates.construction_labels.jsonl"))
OUTPUT = DATA / "results"
V3 = ROOT / "data" / "semantic_benchmark" / "main" / "benchmark_v3"
REFERENCE = ROOT / "data" / "semantic_benchmark" / "main" / "reference_variants" / "annotator_a.protocol_normalized.v1.3.1.jsonl"
SCHEMA = ROOT / "schemas" / "semantic_requirements.schema.json"
CACHE = ROOT / "models" / "huggingface"
SEED = 20261120
SKIP_FINETUNE = os.environ.get("SKIP_FINETUNE") == "1"
EXTERNAL = (
    "human60_original",
    "human60_paraphrase",
    "compositional600_canonical",
    "compositional600_paraphrase",
    "independent_paraphrase",
)


def _external(name: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    tasks = list(load_jsonl(V3 / f"eval_{name}.tasks.jsonl"))
    if name.startswith("human60"):
        labels = {r["task_id"]: r["annotation"] for r in load_jsonl(REFERENCE)}
    else:
        labels = {r["task_id"]: r["annotation"] for r in load_jsonl(V3 / f"eval_{name}.labels.jsonl")}
    return tasks, labels


def main() -> None:
    started = time.perf_counter()
    schema = load_json_object(SCHEMA)
    rows = list(load_jsonl(DATA / "candidates.tasks.jsonl"))
    labels = {r["task_id"]: r["annotation"] for r in load_jsonl(LABELS)}
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        by_split[row["design_strata"]["split"]].append(row)
    train = by_split["train"]
    train_labels = [labels[t["task_id"]] for t in train]
    eval_sets: dict[str, tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str] | None]] = {
        f"v2_{split}": (
            by_split[split],
            labels,
            [str(t["grouping"]["subject_group_id"]) for t in by_split[split]],
        )
        for split in ("validation", "test")
    }
    # Test broken down by wording: template views vs LLM natural rewrites.
    for view_name, views in (("template", {"canonical", "paraphrase"}), ("natural", {"natural"})):
        subset = [t for t in by_split["test"] if t["design_strata"]["wording_view"] in views]
        if subset and len(subset) != len(by_split["test"]):
            eval_sets[f"v2_test_{view_name}"] = (
                subset,
                labels,
                [str(t["grouping"]["subject_group_id"]) for t in subset],
            )
    for name in EXTERNAL:
        tasks, ext_labels = _external(name)
        eval_sets[name] = (tasks, ext_labels, None)

    encoder = load_e5_small(cache_folder=CACHE)
    models: list[tuple[str, Any]] = [
        ("constant_profile", ConstantProfileBaseline()),
        ("rule_keyword", RuleBasedBaseline()),
        ("tfidf_lr_v2", TfidfLogisticBaseline(random_seed=SEED)),
        ("e5_frozen_lr_v2", FrozenEncoderLogisticBaseline(encoder, random_seed=SEED)),
    ]
    if not SKIP_FINETUNE:
        from task_offloading.semantic.finetune import FineTunedEncoderBaseline

        models.append(("e5_finetuned_v2", FineTunedEncoderBaseline(cache_folder=CACHE, random_seed=SEED)))

    point: dict[str, Any] = {}
    task_scores: dict[str, dict[str, TaskLevelScores]] = {name: {} for name in eval_sets}
    for model_name, model in models:
        fit_started = time.perf_counter()
        model.fit(train, train_labels)
        point[model_name] = {
            "training_rows": len(train),
            "fit_seconds": time.perf_counter() - fit_started,
            "sets": {},
        }
        for set_name, (tasks, set_labels, _) in eval_sets.items():
            predictions = [model.predict(task) for task in tasks]
            refs = [set_labels[t["task_id"]] for t in tasks]
            full = evaluate_semantic_predictions(
                model_name=model_name,
                tasks=tasks,
                references=refs,
                predictions=predictions,
                semantic_schema=schema,
            )
            write_jsonl(
                OUTPUT / "predictions" / f"{model_name}.{set_name}.jsonl",
                [
                    {"task_id": t["task_id"], "model_name": model_name, "annotation": p.annotation}
                    for t, p in zip(tasks, predictions, strict=True)
                ],
            )
            point[model_name]["sets"][set_name] = {
                "task_count": len(tasks),
                "exact_record_accuracy": full["exact_record_accuracy"],
                "macro_f1_across_fields": full["macro_f1_across_fields"],
                "schema_validity": full["schema_validity"],
                "abstention_f1": full["abstention"]["f1"],
                "critical_privacy": full["critical_privacy"],
                "inference_latency_ms_mean": full["inference_latency_ms"]["mean"],
                "fields_macro_f1": {f: v["macro_f1"] for f, v in full["fields"].items()},
            }
            task_scores[set_name][model_name] = TaskLevelScores(refs, [p.annotation for p in predictions])
            print(
                f"{model_name:18s} {set_name:28s} exact={full['exact_record_accuracy']:.3f} "
                f"macroF1={full['macro_f1_across_fields']:.3f} ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )

    intervals = {
        set_name: bootstrap_intervals(scores, baseline="rule_keyword", clusters=eval_sets[set_name][2])
        for set_name, scores in task_scores.items()
    }
    write_json_atomic(
        OUTPUT / "results.provisional_v2.json",
        {
            "status": "provisional_construction_labels_pending_human_adjudication",
            "design": __doc__,
            "labels_file": LABELS.name,
            "labels_sha256": sha256(LABELS),
            "training_rows": len(train),
            "hyperparameters": "frozen from benchmark v3; no validation or test tuning",
            "evaluation_labels_used_for_training_or_tuning": False,
            "bootstrap": {
                "resamples": 1000,
                "interval": "percentile 95%",
                "unit": {"v2_*": "subject_group (cluster bootstrap)", "external": "task"},
            },
            "set_sizes": {name: len(v[0]) for name, v in eval_sets.items()},
            "point_estimates": point,
            "intervals": intervals,
        },
    )
    print(f"done in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
