# ruff: noqa: E501, RUF001
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from task_offloading.semantic import load_jsonl
from task_offloading.semantic.expansion import build_context_disjoint_families

ROOT = Path(__file__).resolve().parents[2]
EXPANDED = ROOT / "data" / "semantic_benchmark" / "expanded_v2"
MAIN = ROOT / "data" / "semantic_benchmark" / "main"
V3 = MAIN / "benchmark_v3"


def _manifest() -> dict[str, Any]:
    value = json.loads(
        (EXPANDED / "split_manifest.candidate.json").read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def _visible(row: dict[str, Any]) -> str:
    return f"{row['task_text']}\n{row.get('user_policy_text') or ''}".strip()


def test_build_reports_zero_leakage_and_expected_sizes() -> None:
    manifest = _manifest()
    assert manifest["leakage_audit"]
    assert all(value == 0 for value in manifest["leakage_audit"].values())
    assert manifest["subject_group_counts"] == {
        "test": 32,
        "train": 186,
        "validation": 32,
    }
    assert manifest["row_counts"] == {"test": 768, "train": 3024, "validation": 768}
    assert len(manifest["banned_evaluation_contexts"]["subjects"]) == 30


def test_subjects_and_operations_never_cross_splits() -> None:
    families = {r["task_id"]: r for r in load_jsonl(EXPANDED / "scenario_families.jsonl")}
    seen: dict[str, dict[str, set[str]]] = {
        "subject": defaultdict(set),
        "operation": defaultdict(set),
    }
    for task in load_jsonl(EXPANDED / "candidates.tasks.jsonl"):
        source = families[task["grouping"]["scenario_family_id"]]["provenance"]
        split = task["design_strata"]["split"]
        seen["subject"][source["subject"]].add(split)
        seen["operation"][source["operation"]].add(split)
    for kind in seen.values():
        assert all(len(splits) == 1 for splits in kind.values())


def test_no_locked_evaluation_text_or_context_is_reused() -> None:
    split = json.loads(
        (MAIN / "split_manifest.v1.json").read_text(encoding="utf-8")
    )
    evaluation_ids = set(split["single_human_evaluation"]["task_ids"])
    evaluation = {
        _visible(r)
        for r in load_jsonl(MAIN / "tasks.v1.jsonl")
        if r["task_id"] in evaluation_ids
    }
    for path in V3.glob("eval_*.tasks.jsonl"):
        evaluation |= {_visible(r) for r in load_jsonl(path)}
    banned = _manifest()["banned_evaluation_contexts"]
    terms = [t.casefold() for t in (*banned["subjects"], *banned["operations"])]
    for task in load_jsonl(EXPANDED / "candidates.tasks.jsonl"):
        text = _visible(task)
        assert text not in evaluation
        folded = text.casefold()
        assert not any(term in folded for term in terms)


def test_heldout_splits_contain_only_new_contexts() -> None:
    families = list(load_jsonl(EXPANDED / "scenario_families.jsonl"))
    assignments = _manifest()["family_assignments"]
    for family in families:
        if assignments[family["task_id"]] != "train":
            assert family["provenance"]["context_origin"] == "new_curated_context_v2"


def test_reused_historical_subject_is_rejected() -> None:
    blueprint = {
        "domains": {
            "healthcare": {
                "data": "klinik sensör verisi",
                "contexts": [["Sürekli glikoz ölçer", "hipoglisemi uyarısı üretme"]],
            }
        }
    }
    contexts = {
        "domains": {
            "healthcare": [
                ["Sürekli glikoz ölçer", ["yeni işlem a", "yeni işlem b"]],
                ["Yeni cihaz", ["yeni işlem c"]],
                ["Başka cihaz", ["yeni işlem d"]],
            ]
        }
    }
    with pytest.raises(ValueError, match="not new"):
        build_context_disjoint_families(
            blueprint,
            contexts,
            banned_subjects=set(),
            banned_operations=set(),
            subject_split_per_domain={"validation": 1, "test": 1},
            seed=1,
        )
