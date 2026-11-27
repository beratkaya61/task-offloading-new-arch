# ruff: noqa: E501
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jsonschema

from task_offloading.semantic import load_jsonl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "semantic_benchmark" / "expanded_v2"
OUT = DATA / "annotation"
LEAKY_TOKENS = ("exp2_", "canonical", "paraphrase", "validation", "_test", "profile", "train")


def _plan() -> dict[str, Any]:
    value = json.loads((OUT / "annotation_plan.v2.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _key() -> list[dict[str, Any]]:
    return list(load_jsonl(OUT / "blinding_key.coordinator_only.jsonl"))


def test_lean_plan_counts() -> None:
    plan = _plan()
    assert plan["role_counts"] == {
        "hidden_repeat": 24,
        "pilot": 40,
        "test_reference": 384,
        "training_audit": 192,
        "validation_reference": 384,
    }
    assert plan["tasks_per_annotator_including_pilot_and_repeats"] == 1024
    assert plan["annotator_b_extra_human60_tasks"] == 60


def test_packets_are_blind() -> None:
    for path in (OUT / "packets").glob("*.jsonl"):
        text = path.read_text(encoding="utf-8")
        for token in LEAKY_TOKENS:
            assert token not in text, (path.name, token)
        for row in load_jsonl(path):
            assert set(row) == {
                "packet_version", "protocol_version", "annotator_id", "presentation_order",
                "task_id", "language", "task_text", "user_policy_text", "task_content_sha256",
            }


def test_every_heldout_subject_is_annotated_with_two_profiles_per_family() -> None:
    manifest = json.loads((DATA / "split_manifest.candidate.json").read_text(encoding="utf-8"))
    heldout = {f for f, s in manifest["family_assignments"].items() if s != "train"}
    rows: dict[str, Counter[str]] = defaultdict(Counter)
    for row in _key():
        if row["role"] in {"validation_reference", "test_reference"}:
            rows[row["scenario_family_id"]][row["wording_view"]] += 1
    assert set(rows) == heldout
    assert all(views == {"canonical": 2, "paraphrase": 2} for views in rows.values())


def test_pilot_is_train_only_and_disjoint_from_main() -> None:
    key = _key()
    pilot_families = {r["scenario_family_id"] for r in key if r["role"] == "pilot"}
    main_families = {r["scenario_family_id"] for r in key if r["role"] not in {"pilot", "hidden_repeat"}}
    assert {r["split"] for r in key if r["role"] == "pilot"} == {"train"}
    assert not pilot_families & main_families


def test_hidden_repeats_come_back_in_later_sessions() -> None:
    key = _key()
    session_of = {r["task_id"]: r["session"] for r in key if r["role"] not in {"pilot", "hidden_repeat"}}
    for row in key:
        if row["role"] == "hidden_repeat":
            assert session_of[row["repeat_of_task_id"]] in {"session_01", "session_02"}
            assert row["session"] in {"session_06", "session_07"}


def test_v2_record_schema_accepts_naturalness_and_rejects_real_ids() -> None:
    schema = json.loads((ROOT / "schemas" / "human_annotation_record.v2.schema.json").read_text(encoding="utf-8"))
    record = {
        "record_version": "1.0.0", "protocol_version": "2.0.0", "annotator_id": "annotator_a",
        "task_id": "v2x_0001", "task_content_sha256": "a" * 64,
        "started_at_utc": "2026-12-01T10:00:00Z", "submitted_at_utc": "2026-12-01T10:01:00Z",
        "annotation": {}, "naturalness": "natural",
    }
    jsonschema.validate(record, schema)
    leaked = {**record, "task_id": "exp2_healthcare_new_01_op1_profile_00_canonical"}
    try:
        jsonschema.validate(leaked, schema)
    except jsonschema.ValidationError:
        pass
    else:
        raise AssertionError("schema must reject non-opaque task ids")
