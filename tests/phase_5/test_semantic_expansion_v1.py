from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from task_offloading.semantic import load_jsonl

ROOT = Path(__file__).resolve().parents[2]
EXPANDED = ROOT / "data" / "semantic_benchmark" / "expanded_v1"


def test_expansion_candidate_counts_and_family_disjointness() -> None:
    manifest = json.loads(
        (EXPANDED / "split_manifest.candidate.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == (
        "candidate_pool_requires_human_annotation_and_adjudication"
    )
    assert manifest["total_families"] == 600
    assert manifest["total_rows"] == 4800
    assert manifest["family_counts"] == {
        "test": 88,
        "train": 424,
        "validation": 88,
    }
    assert manifest["row_counts"] == {
        "test": 704,
        "train": 3392,
        "validation": 704,
    }


def test_each_profile_has_canonical_and_paraphrase_with_same_label() -> None:
    tasks = load_jsonl(EXPANDED / "candidates.tasks.jsonl")
    labels = {
        row["task_id"]: row["annotation"]
        for row in load_jsonl(
            EXPANDED / "candidates.construction_labels.jsonl"
        )
    }
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        profile = str(task["grouping"]["semantic_profile_id"])
        by_profile.setdefault(profile, []).append(task)
    assert len(by_profile) == 2400
    visible_hashes = {
        hashlib.sha256(
            f"{row['task_text']}\n{row.get('user_policy_text') or ''}".encode()
        ).hexdigest()
        for row in tasks
    }
    assert len(visible_hashes) == len(tasks)
    for rows in by_profile.values():
        assert {row["design_strata"]["wording_view"] for row in rows} == {
            "canonical",
            "paraphrase",
        }
        first, second = rows
        left = dict(labels[str(first["task_id"])])
        right = dict(labels[str(second["task_id"])])
        left.pop("task_id")
        right.pop("task_id")
        assert left == right


def test_annotation_manifest_covers_complete_validation_test_and_train_audit() -> None:
    annotation = json.loads(
        (EXPANDED / "annotation_manifest.candidate.json").read_text(
            encoding="utf-8"
        )
    )
    roles: dict[str, int] = {}
    for item in annotation["items"]:
        role = str(item["annotation_role"])
        roles[role] = roles.get(role, 0) + 1
    assert roles == {
        "test_reference": 704,
        "training_audit": 384,
        "validation_reference": 704,
    }
    assert annotation["unique_tasks_to_annotate"] == 1792
    assert annotation["minimum_independent_assignments"] == 3584
