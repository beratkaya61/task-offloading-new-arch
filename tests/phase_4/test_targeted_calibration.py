# ruff: noqa: RUF001
"""Protocol 1.2 workload and targeted-calibration regression tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from task_offloading.semantic import (
    ANNOTATION_FIELDS,
    audit_targeted_calibration,
    load_jsonl,
    render_targeted_calibration,
)

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = ROOT / "data" / "semantic_benchmark" / "calibration"
PILOT = ROOT / "data" / "semantic_benchmark" / "pilot"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    assert isinstance(value, dict)
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_targeted_calibration_is_schema_valid_novel_and_complete() -> None:
    package = load_json(CALIBRATION / "targeted_examples.v1.json")
    schema = load_json(ROOT / "schemas" / "targeted_calibration.schema.json")
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(package)

    audit = audit_targeted_calibration(
        package,
        load_jsonl(PILOT / "tasks.v1.jsonl"),
    )
    assert audit.is_valid, audit.issues
    assert audit.example_count == 20
    assert set(dict(audit.focus_field_counts)) == set(ANNOTATION_FIELDS)
    assert package["pilot_content_reused"] is False
    assert package["is_ground_truth"] is False


def test_targeted_calibration_html_is_offline_and_deterministic() -> None:
    package = load_json(CALIBRATION / "targeted_examples.v1.json")
    tracked = (CALIBRATION / "targeted_calibration.v1.html").read_text(
        encoding="utf-8"
    )
    assert tracked == render_targeted_calibration(package)
    assert "connect-src 'none'" in tracked
    assert "Bu bir anket veya sınav değildir" in tracked
    assert "insan ground truth'u sayılmaz" in tracked
    assert "pilot_tr_" not in tracked
    assert tracked.count("<article class=\"card\">") == 20


def test_calibration_manifest_binds_every_input_and_output() -> None:
    manifest = load_json(CALIBRATION / "manifest.v1.json")
    assert manifest["status"] == "prepared_not_completed_by_humans"
    assert manifest["pilot_exact_text_overlap_count"] == 0
    assert manifest["contains_human_labels"] is False
    assert manifest["is_ground_truth"] is False
    for section in (
        "source",
        "schema",
        "protocol",
        "rendered_html",
    ):
        info = manifest[section]
        assert file_sha256(ROOT / info["path"]) == info["sha256"]


def test_protocol_1_3_requires_no_new_secondary_annotator_tasks() -> None:
    design = load_json(ROOT / "configs" / "semantic_annotation_design.v1.json")
    partitions = design["partitions"]
    development = partitions["machine_labeled_development"]
    evaluation = partitions["single_human_evaluation"]
    evidence = design["existing_human_evidence"]

    assert design["protocol_version"] == "1.3.0"
    assert development["unique_task_count"] + evaluation["unique_task_count"] == 240
    assert development["unique_task_count"] == 180
    assert development["human_validated"] is False
    assert development["adjudicated_gold"] is False
    assert evaluation["unique_task_count"] == 60
    assert evaluation["annotators"] == ["annotator_a"]
    assert evaluation["llm_preannotation_visible_to_annotators"] is False
    assert evaluation["adjudicated_gold"] is False
    assert evidence["new_tasks_required_from_annotator_b"] == 0
    assert evidence["new_tasks_required_from_annotator_c"] == 0
    assert evidence["pilot_round_1"]["diagnostic_annotator"] == "annotator_c"
    assert evidence["pilot_round_2"]["quality_gate_status"] == "failed"
    assert design["evaluation_selection"]["performed_before_annotation"] is True
    assert design["evaluation_selection"]["label_values_may_be_used"] is False
    assert design["quality_gate"]["minimum_kappa"] == 0.7
    assert design["preannotation"]["model_id"] is None
    assert design["preannotation"]["model_must_be_frozen_before_use"] is True


def test_human_record_schema_accepts_protocol_1_3_main_ids() -> None:
    wrapper = load_json(ROOT / "schemas" / "human_annotation_record.schema.json")
    record = {
        "record_version": "1.0.0",
        "protocol_version": "1.3.0",
        "annotator_id": "annotator_a",
        "task_id": "main_tr_240",
        "task_content_sha256": "a" * 64,
        "started_at_utc": "2026-09-29T09:00:00Z",
        "submitted_at_utc": "2026-09-29T09:03:00Z",
        "annotation": {},
    }
    Draft202012Validator(wrapper).validate(record)
