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
    for views in rows.values():
        assert views["canonical"] == 2
        assert views["natural"] + views["paraphrase"] == 2
        assert set(views) <= {"canonical", "natural", "paraphrase"}


def test_pilot_is_train_only_and_disjoint_from_main() -> None:
    key = _key()
    pilot_families = {r["scenario_family_id"] for r in key if r["role"] == "pilot"}
    main_families = {r["scenario_family_id"] for r in key if r["role"] not in {"pilot", "hidden_repeat"}}
    assert {r["split"] for r in key if r["role"] == "pilot"} == {"train"}
    assert not pilot_families & main_families
    pilot = [r for r in key if r["role"] == "pilot"]
    assert all(r["alias"].startswith("v2p_") for r in pilot)
    assert all(r["alias"].startswith("v2x_") for r in key if r["role"] != "pilot")
    assert {r["wording_view"] for r in pilot} <= {"canonical", "paraphrase"}


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
        "task_id": "v2p_0001", "task_content_sha256": "a" * 64,
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


FROZEN_PILOT_CONTENT_SHA256 = {
    # Released 2026-11-27; later wording additions must never change the pilot.
    "annotator_a": "0f6d4e3c37f28f36e43ee95729a6b14bcc07ce776d0e29f7900ef8607b2fe1a9",
    "annotator_b": "b85a5762b6e044896ce075a2423e5658930a90e1d27d2f0bf6b973e197fa94bb",
}


def test_released_pilot_packets_are_frozen() -> None:
    import hashlib

    for annotator, expected in FROZEN_PILOT_CONTENT_SHA256.items():
        rows = list(load_jsonl(OUT / "packets" / f"{annotator}.pilot.jsonl"))
        payload = json.dumps(
            [[r["presentation_order"], r["task_id"], r["task_text"], r["user_policy_text"]] for r in rows],
            ensure_ascii=False,
            sort_keys=True,
        )
        assert hashlib.sha256(payload.encode()).hexdigest() == expected
