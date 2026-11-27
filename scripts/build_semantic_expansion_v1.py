"""Materialize the v1 semantic expansion candidate and annotation manifests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from task_offloading.semantic import write_jsonl
from task_offloading.semantic.expansion import (
    assign_family_splits,
    build_expanded_rows,
    build_scenario_families,
    select_training_audit_families,
)
from task_offloading.semantic.experiment import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "semantic_expansion_v1.json"
BLUEPRINT = ROOT / "configs" / "main_semantic_scenarios.v1.json"
OUTPUT = ROOT / "data" / "semantic_benchmark" / "expanded_v1"


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    config = _object(CONFIG)
    blueprint = _object(BLUEPRINT)
    seed = int(config["seed"])
    families = build_scenario_families(blueprint)
    assignments = assign_family_splits(
        families,
        seed=seed,
        per_domain=config["family_split_per_domain"],
    )
    tasks, labels = build_expanded_rows(
        families,
        assignments,
        profiles_per_family=int(config["semantic_profiles_per_family"]),
        seed=seed,
    )
    task_split = {
        str(row["task_id"]): str(row["design_strata"]["split"]) for row in tasks
    }
    if len(task_split) != len(tasks):
        raise ValueError("duplicate task ids in expansion candidate")
    hashes: set[str] = set()
    family_splits: dict[str, set[str]] = {}
    profile_splits: dict[str, set[str]] = {}
    for task in tasks:
        text = f"{task['task_text']}\n{task.get('user_policy_text') or ''}"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in hashes:
            raise ValueError("duplicate visible text in expansion candidate")
        hashes.add(digest)
        split = str(task["design_strata"]["split"])
        family_splits.setdefault(
            str(task["grouping"]["scenario_family_id"]), set()
        ).add(split)
        profile_splits.setdefault(
            str(task["grouping"]["semantic_profile_id"]), set()
        ).add(split)
    if any(len(values) != 1 for values in family_splits.values()):
        raise ValueError("scenario family leakage across splits")
    if any(len(values) != 1 for values in profile_splits.values()):
        raise ValueError("semantic profile leakage across splits")
    audit_families = select_training_audit_families(
        families,
        assignments,
        count=int(config["human_annotation"]["training_audit_families"]),
        seed=seed,
    )
    audit_set = set(audit_families)
    annotation_items = []
    for task in tasks:
        task_id = str(task["task_id"])
        split = task_split[task_id]
        family_id = str(task["grouping"]["scenario_family_id"])
        if split in {"validation", "test"} or family_id in audit_set:
            annotation_items.append(
                {
                    "task_id": task_id,
                    "scenario_family_id": family_id,
                    "split": split,
                    "annotation_role": (
                        f"{split}_reference" if split != "train" else "training_audit"
                    ),
                    "independent_annotators_required": int(
                        config["human_annotation"]["independent_annotators"]
                    ),
                    "adjudication_on_disagreement": True,
                }
            )

    family_counts = Counter(assignments.values())
    row_counts = Counter(task_split.values())
    manifest = {
        "protocol_version": config["protocol_version"],
        "status": config["status"],
        "seed": seed,
        "independent_unit": "scenario_family",
        "related_row_unit": "semantic_profile_x_wording_view",
        "family_counts": dict(sorted(family_counts.items())),
        "row_counts": dict(sorted(row_counts.items())),
        "total_families": len(families),
        "total_rows": len(tasks),
        "labels": "by_construction_pending_blind_human_annotation",
        "family_assignments": dict(sorted(assignments.items())),
        "comparison_contract": config["comparison_contract"],
    }
    annotation_manifest = {
        "protocol_version": config["protocol_version"],
        "blind_to_model_predictions": True,
        "independent_annotators": config["human_annotation"][
            "independent_annotators"
        ],
        "adjudicator_on_disagreement": config["human_annotation"][
            "adjudicator_on_disagreement"
        ],
        "training_audit_families": audit_families,
        "unique_tasks_to_annotate": len(annotation_items),
        "minimum_independent_assignments": len(annotation_items)
        * int(config["human_annotation"]["independent_annotators"]),
        "items": annotation_items,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT / "candidates.tasks.jsonl", tasks)
    write_jsonl(OUTPUT / "candidates.construction_labels.jsonl", labels)
    write_json_atomic(OUTPUT / "split_manifest.candidate.json", manifest)
    write_json_atomic(
        OUTPUT / "annotation_manifest.candidate.json", annotation_manifest
    )
    print(
        f"families={len(families)} rows={len(tasks)} "
        f"split_rows={dict(sorted(row_counts.items()))}"
    )
    print(
        f"human_annotation_tasks={len(annotation_items)} "
        f"minimum_assignments={annotation_manifest['minimum_independent_assignments']}"
    )


if __name__ == "__main__":
    main()
