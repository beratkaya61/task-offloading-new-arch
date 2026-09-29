"""Main semantic corpus balance, provenance, and frozen-partition tests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from task_offloading.semantic import (
    audit_annotation_release,
    audit_corpus,
    audit_submission,
    authoring_preannotation,
    load_jsonl,
    packet_issues,
    render_annotation_form,
)

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "data" / "semantic_benchmark" / "main"
PILOT = ROOT / "data" / "semantic_benchmark" / "pilot"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    assert isinstance(value, dict)
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_main_corpus_is_balanced_schema_valid_and_not_yet_released() -> None:
    tasks = load_jsonl(MAIN / "tasks.v1.jsonl")
    schema = load_json(ROOT / "schemas" / "main_semantic_task.schema.json")
    audit = audit_corpus(tasks, schema, expected_count=240)
    assert audit.is_valid, audit.issues
    assert dict(audit.domain_stratum_counts) == {
        "agriculture": 30,
        "consumer": 30,
        "healthcare": 30,
        "industrial": 30,
        "other": 30,
        "public_safety": 30,
        "smart_city": 30,
        "transportation": 30,
    }
    assert dict(audit.origin_counts) == {
        "adversarial": 60,
        "llm_assisted_synthetic": 60,
        "standard_derived": 60,
        "trace_conditioned": 60,
    }
    release = audit_annotation_release(tasks)
    assert not release.is_ready
    assert dict(release.review_status_counts) == {"pending": 240}


def test_main_split_is_label_blind_complete_and_family_disjoint() -> None:
    tasks = load_jsonl(MAIN / "tasks.v1.jsonl")
    split = load_json(MAIN / "split_manifest.v1.json")
    development = split["machine_labeled_development"]
    evaluation = split["single_human_evaluation"]
    development_ids = set(development["task_ids"])
    evaluation_ids = set(evaluation["task_ids"])

    assert split["selection_uses_human_or_model_labels"] is False
    assert len(development_ids) == development["task_count"] == 180
    assert len(evaluation_ids) == evaluation["task_count"] == 60
    assert not development_ids & evaluation_ids
    assert development_ids | evaluation_ids == {
        str(task["task_id"]) for task in tasks
    }
    assert not set(development["scenario_family_ids"]) & set(
        evaluation["scenario_family_ids"]
    )

    evaluation_rows = [
        task for task in tasks if str(task["task_id"]) in evaluation_ids
    ]
    domain_counts = Counter(
        str(task["design_strata"]["scenario_domain"])
        for task in evaluation_rows
    )
    origin_counts = Counter(
        str(task["provenance"]["origin"]) for task in evaluation_rows
    )
    phenomena = {
        str(phenomenon)
        for task in evaluation_rows
        for phenomenon in task["design_strata"]["semantic_phenomena"]
    }
    assert min(domain_counts.values()) >= 6
    assert min(origin_counts.values()) >= 14
    assert phenomena == {
        "ambiguity",
        "explicit_accuracy",
        "explicit_divisibility",
        "explicit_energy",
        "explicit_execution_policy",
        "explicit_latency",
        "explicit_privacy",
        "explicit_reliability",
        "missing_information",
        "negation",
        "numeric_threshold",
        "qualitative_requirement",
    }


def test_main_manifest_binds_sources_schema_split_and_protocol() -> None:
    manifest = load_json(MAIN / "corpus_manifest.v1.json")
    assert manifest["status"] == "generated_awaiting_human_text_review"
    assert manifest["annotation_release_allowed"] is False
    assert manifest["design_metadata_is_gold"] is False
    assert manifest["generation_model_may_define_gold"] is False
    assert manifest["pilot_exact_text_overlap_count"] == 0
    sections = (
        "tasks",
        "schema",
        "blueprint",
        "annotation_design",
        "protocol",
        "split",
    )
    for section in sections:
        info = manifest[section]
        assert file_sha256(ROOT / info["path"]) == info["sha256"]


def test_main_corpus_does_not_reuse_pilot_text_or_raw_trace_identifiers() -> None:
    pilot = load_jsonl(PILOT / "tasks.v1.jsonl")
    main = load_jsonl(MAIN / "tasks.v1.jsonl")

    def visible_text(row: dict[str, Any]) -> str:
        return (
            str(row["task_text"]) + "\n" + str(row.get("user_policy_text") or "")
        ).casefold()

    assert not {visible_text(row) for row in pilot} & {
        visible_text(row) for row in main
    }
    forbidden = ("http://", "https://", "imei", "imsi", "user-agent")
    assert all(
        token not in visible_text(row) for row in main for token in forbidden
    )


def test_each_main_scenario_family_has_two_paraphrases_in_one_partition() -> None:
    tasks = load_jsonl(MAIN / "tasks.v1.jsonl")
    split = load_json(MAIN / "split_manifest.v1.json")
    evaluation_ids = set(split["single_human_evaluation"]["task_ids"])
    family_partitions: dict[str, set[str]] = {}
    family_counts: Counter[str] = Counter()
    for task in tasks:
        family = str(task["grouping"]["scenario_family_id"])
        family_counts[family] += 1
        family_partitions.setdefault(family, set()).add(
            "evaluation" if task["task_id"] in evaluation_ids else "development"
        )
    assert len(family_counts) == 120
    assert set(family_counts.values()) == {2}
    assert all(len(partitions) == 1 for partitions in family_partitions.values())


def test_machine_preannotations_cover_only_development_and_pass_server_audit() -> None:
    tasks = load_jsonl(MAIN / "tasks.v1.jsonl")
    split = load_json(MAIN / "split_manifest.v1.json")
    development_ids = set(split["machine_labeled_development"]["task_ids"])
    evaluation_ids = set(split["single_human_evaluation"]["task_ids"])
    by_id = {str(task["task_id"]): task for task in tasks}
    development = tuple(by_id[task_id] for task_id in development_ids)
    stored = load_jsonl(
        MAIN / "preannotations" / "machine_development.v1.jsonl"
    )

    assert len(stored) == 180
    assert {str(record["task_id"]) for record in stored} == development_ids
    assert not {str(record["task_id"]) for record in stored} & evaluation_ids
    assert all(record["is_human_annotation"] is False for record in stored)
    assert all(record["is_ground_truth"] is False for record in stored)
    assert all(
        record["annotation"] == authoring_preannotation(by_id[str(record["task_id"])])
        for record in stored
    )

    wrappers = [
        {
            "record_version": "1.0.0",
            "protocol_version": "1.3.0",
            "annotator_id": "annotator_a",
            "task_id": record["task_id"],
            "task_content_sha256": record["task_content_sha256"],
            "started_at_utc": "2026-09-29T00:00:00Z",
            "submitted_at_utc": "2026-09-29T00:00:01Z",
            "annotation": record["annotation"],
        }
        for record in stored
    ]
    audit = audit_submission(
        wrappers,
        development,
        load_json(ROOT / "schemas" / "human_annotation_record.schema.json"),
        load_json(ROOT / "schemas" / "semantic_requirements.schema.json"),
        annotator_id="annotator_a",
        expected_protocol_version="1.3.0",
    )
    assert audit.is_valid, audit.issues


def test_main_a_packet_is_blind_complete_and_checksum_bound() -> None:
    tasks = load_jsonl(MAIN / "tasks.v1.jsonl")
    by_id = {str(task["task_id"]): task for task in tasks}
    split = load_json(MAIN / "split_manifest.v1.json")
    manifest = load_json(MAIN / "annotation_packets.v1.json")
    assert manifest["main_annotation_status"] == "forms_prepared_not_collected"
    assert manifest["additional_annotator_release_status"] == "not_planned"

    loaded_packets: dict[str, tuple[dict[str, Any], ...]] = {}
    for key, info in manifest["packets"].items():
        path = ROOT / info["path"]
        assert file_sha256(path) == info["sha256"]
        packet = load_jsonl(path)
        loaded_packets[key] = packet
        expected_ids = split["single_human_evaluation"]["task_ids"]
        corpus = tuple(by_id[str(task_id)] for task_id in expected_ids)
        annotator = "annotator_a"
        assert not packet_issues(
            packet,
            corpus,
            annotator_id=annotator,
            packet_version="1.3.0",
            protocol_version="1.3.0",
        )
        assert all(
            not ({"provenance", "design_strata", "grouping", "annotation"} & set(row))
            for row in packet
        )

    a_ids = tuple(
        row["task_id"] for row in loaded_packets["annotator_a_evaluation"]
    )
    assert len(a_ids) == 60
    assert set(loaded_packets) == {"annotator_a_evaluation"}


def test_human_forms_are_blind_and_machine_labels_are_not_human_forms() -> None:
    manifest = load_json(MAIN / "annotation_packets.v1.json")
    records = load_jsonl(
        MAIN / "preannotations" / "machine_development.v1.jsonl"
    )
    assert len(records) == 180
    assert all(record["is_human_annotation"] is False for record in records)
    assert "annotator_a_assisted_development" not in manifest["packets"]

    blind_path = ROOT / manifest["packets"]["annotator_a_evaluation"]["path"]
    blind_html = render_annotation_form(
        load_jsonl(blind_path),
        annotator_id="annotator_a",
        packet_sha256=file_sha256(blind_path),
    )
    assert '"form_mode":"blind_annotation"' in blind_html
    assert '"initial_responses":{}' in blind_html
    assert "deterministic_authoring_template_extractor" not in blind_html


def test_locked_a_submission_is_valid_bound_and_not_misrepresented_as_gold() -> None:
    annotation_dir = MAIN / "annotations"
    manifest = load_json(annotation_dir / "submission_manifest.v1.json")
    submission_info = manifest["submission"]
    submission_path = ROOT / submission_info["path"]
    records = load_jsonl(submission_path)
    tasks = load_jsonl(MAIN / "tasks.v1.jsonl")
    split = load_json(MAIN / "split_manifest.v1.json")
    by_id = {str(task["task_id"]): task for task in tasks}
    evaluation = tuple(
        by_id[str(task_id)]
        for task_id in split["single_human_evaluation"]["task_ids"]
    )

    assert manifest["status"] == "checksum_locked"
    assert manifest["annotation_claim"] == "single_human_reference"
    assert manifest["adjudicated_gold"] is False
    assert manifest["additional_human_surveys_required"] is False
    assert submission_info["record_count"] == len(records) == 60
    assert file_sha256(submission_path) == submission_info["sha256"]
    assert (
        manifest["existing_secondary_human_evidence"][
            "new_b_or_c_tasks_required"
        ]
        == 0
    )
    for binding in manifest["bindings"].values():
        assert file_sha256(ROOT / binding["path"]) == binding["sha256"]

    audit = audit_submission(
        records,
        evaluation,
        load_json(ROOT / "schemas" / "human_annotation_record.schema.json"),
        load_json(ROOT / "schemas" / "semantic_requirements.schema.json"),
        annotator_id="annotator_a",
        expected_protocol_version="1.3.0",
    )
    assert audit.is_valid, audit.issues
