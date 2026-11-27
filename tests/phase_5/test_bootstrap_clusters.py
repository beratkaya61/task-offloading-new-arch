# ruff: noqa: E501
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from task_offloading.semantic import load_jsonl
from task_offloading.semantic.bootstrap import TaskLevelScores, bootstrap_intervals

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "semantic_benchmark" / "expanded_v2" / "candidates.construction_labels.jsonl"


def _annotations(count: int) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(load_jsonl(LABELS)):
        if index == count:
            break
        rows.append(row["annotation"])
    return rows


def _wrong(annotation: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(annotation)
    value["domain"] = {**value["domain"], "value": "other" if value["domain"]["value"] != "other" else "consumer"}
    return value


def test_cluster_bootstrap_is_wider_for_correlated_rows() -> None:
    refs = _annotations(40)
    # Errors are perfectly correlated within 4 clusters of 10 rows.
    preds = [_wrong(r) if (i // 10) % 2 else r for i, r in enumerate(refs)]
    scores = {"model": TaskLevelScores(refs, preds), "base": TaskLevelScores(refs, refs)}
    clusters = [str(i // 10) for i in range(40)]
    by_task = bootstrap_intervals(scores, baseline="base")
    by_cluster = bootstrap_intervals(scores, baseline="base", clusters=clusters)
    def width(r: dict[str, Any]) -> float:
        return float(r["model"]["exact_field_accuracy"]["ci95"][1] - r["model"]["exact_field_accuracy"]["ci95"][0])

    assert width(by_cluster) > width(by_task)
    assert by_cluster["model"]["exact_field_accuracy"]["value"] == by_task["model"]["exact_field_accuracy"]["value"]


def test_cluster_ids_must_match_task_count() -> None:
    refs = _annotations(4)
    with pytest.raises(ValueError, match="one id per task"):
        bootstrap_intervals({"a": TaskLevelScores(refs, refs)}, baseline="a", clusters=["x"])


def test_labels_file_is_json_lines() -> None:
    first = LABELS.read_text(encoding="utf-8").splitlines()[0]
    assert "annotation" in json.loads(first)
