"""Pilot balance, blindness, reproducibility, and submission-integrity tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from task_offloading.semantic import (
    ANNOTATION_FIELDS,
    audit_corpus,
    audit_submission,
    build_blind_packet,
    corpus_task_sha256,
    load_jsonl,
    packet_issues,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "data" / "semantic_benchmark" / "pilot"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    assert isinstance(value, dict)
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def corpus() -> tuple[dict[str, Any], ...]:
    return load_jsonl(PILOT / "tasks.v1.jsonl")


@pytest.fixture(scope="module")
def corpus_schema() -> dict[str, Any]:
    return load_json(ROOT / "schemas" / "semantic_task.schema.json")


def unknown_annotation(task_id: str) -> dict[str, Any]:
    base = {
        "status": "not_stated",
        "evidence": [],
        "confidence": 0.9,
    }
    return {
        "schema_version": "1.0.0",
        "task_id": task_id,
        "language": "tr",
        "domain": {"value": None, **base},
        "latency": {"class": None, "max_ms": None, **base},
        "privacy": {"data_class": None, **base},
        "execution_policy": {
            "device": "unknown",
            "edge": "unknown",
            "cloud": "unknown",
            **base,
        },
        "reliability": {
            "tier": None,
            "min_success_probability": None,
            **base,
        },
        "accuracy": {"tier": None, "min_score": None, "metric": None, **base},
        "energy_priority": {"value": None, **base},
        "divisibility": {"value": None, **base},
        "abstained_fields": list(ANNOTATION_FIELDS),
        "overall_confidence": 0.9,
    }


def submission_for(
    corpus: tuple[dict[str, Any], ...], annotator_id: str
) -> list[dict[str, Any]]:
    return [
        {
            "record_version": "1.0.0",
            "protocol_version": "1.0.0",
            "annotator_id": annotator_id,
            "task_id": task["task_id"],
            "task_content_sha256": corpus_task_sha256(task),
            "started_at_utc": "2026-09-18T09:00:00Z",
            "submitted_at_utc": "2026-09-18T09:03:00Z",
            "annotation": unknown_annotation(str(task["task_id"])),
        }
        for task in corpus
    ]


def test_pilot_corpus_passes_schema_balance_and_privacy_gates(
    corpus: tuple[dict[str, Any], ...], corpus_schema: dict[str, Any]
) -> None:
    audit = audit_corpus(corpus, corpus_schema)
    assert audit.is_valid, audit.issues
    assert audit.item_count == 20
    assert min(dict(audit.domain_stratum_counts).values()) >= 2
    assert dict(audit.origin_counts) == {
        "adversarial": 5,
        "llm_assisted_synthetic": 5,
        "standard_derived": 5,
        "trace_conditioned": 5,
    }


def test_design_metadata_is_not_present_in_blind_packets(
    corpus: tuple[dict[str, Any], ...]
) -> None:
    packet_a = load_jsonl(PILOT / "packets" / "annotator_a.v1.jsonl")
    packet_b = load_jsonl(PILOT / "packets" / "annotator_b.v1.jsonl")
    assert not packet_issues(packet_a, corpus, annotator_id="annotator_a")
    assert not packet_issues(packet_b, corpus, annotator_id="annotator_b")
    assert [row["task_id"] for row in packet_a] != [
        row["task_id"] for row in packet_b
    ]
    forbidden = {"provenance", "grouping", "design_strata", "annotation", "gold"}
    assert all(not (set(row) & forbidden) for row in (*packet_a, *packet_b))


def test_packet_generation_is_deterministic_and_matches_tracked_files(
    corpus: tuple[dict[str, Any], ...]
) -> None:
    for annotator_id, seed in (("annotator_a", 41041), ("annotator_b", 92093)):
        generated = build_blind_packet(
            corpus, annotator_id=annotator_id, seed=seed
        )
        assert generated == load_jsonl(
            PILOT / "packets" / f"{annotator_id}.v1.jsonl"
        )


def test_manifest_binds_corpus_and_packets_by_sha256() -> None:
    manifest = load_json(PILOT / "pilot_manifest.v1.json")
    assert manifest["gold_status"] == "not_collected"
    assert manifest["human_annotators_required"] == 2
    assert manifest["design_metadata_is_gold"] is False
    corpus_info = manifest["corpus"]
    assert file_sha256(ROOT / corpus_info["path"]) == corpus_info["sha256"]
    for packet_info in manifest["packets"].values():
        assert file_sha256(ROOT / packet_info["path"]) == packet_info["sha256"]


def test_trace_conditioning_uses_only_aggregate_profile(
    corpus: tuple[dict[str, Any], ...]
) -> None:
    trace_items = [
        item for item in corpus if item["provenance"]["origin"] == "trace_conditioned"
    ]
    assert len(trace_items) == 5
    assert all(
        item["provenance"]["trace_conditioning"]["profile_id"]
        == "bupt_04e664f_workload_quantiles_v1"
        for item in trace_items
    )
    profile = load_json(
        ROOT / "configs" / "data_profiles" / "bupt_04e664f_workload_quantiles.json"
    )
    assert profile["privacy"]["aggregate_only"] is True
    assert profile["privacy"]["raw_identifiers_included"] is False
    assert profile["method"]["row_level_values_retained"] is False


def test_privacy_schema_rejects_explicit_null_class() -> None:
    schema = load_json(ROOT / "schemas" / "semantic_requirements.schema.json")
    annotation = unknown_annotation("pilot_tr_001")
    annotation["privacy"] = {
        "data_class": None,
        "status": "explicit",
        "evidence": ["Ham EKG"],
        "confidence": 0.9,
    }
    annotation["abstained_fields"].remove("privacy")
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(annotation)


def test_structurally_complete_submission_passes_integrity_checks(
    corpus: tuple[dict[str, Any], ...]
) -> None:
    wrapper = load_json(ROOT / "schemas" / "human_annotation_record.schema.json")
    semantic = load_json(ROOT / "schemas" / "semantic_requirements.schema.json")
    audit = audit_submission(
        submission_for(corpus, "annotator_a"),
        corpus,
        wrapper,
        semantic,
        annotator_id="annotator_a",
    )
    assert audit.is_valid, audit.issues


def test_submission_rejects_nonverbatim_evidence_and_stale_hash(
    corpus: tuple[dict[str, Any], ...]
) -> None:
    wrapper = load_json(ROOT / "schemas" / "human_annotation_record.schema.json")
    semantic = load_json(ROOT / "schemas" / "semantic_requirements.schema.json")
    records = submission_for(corpus, "annotator_b")
    records[0]["task_content_sha256"] = "0" * 64
    annotation = records[0]["annotation"]
    annotation["domain"] = {
        "value": "healthcare",
        "status": "explicit",
        "evidence": ["metinde bulunmayan kanıt"],  # noqa: RUF001
        "confidence": 0.9,
    }
    annotation["abstained_fields"].remove("domain")
    audit = audit_submission(
        records,
        corpus,
        wrapper,
        semantic,
        annotator_id="annotator_b",
    )
    assert not audit.is_valid
    assert any("stale task hash" in issue for issue in audit.issues)
    assert any("exact visible substring" in issue for issue in audit.issues)


def test_submission_rejects_abstention_drift(
    corpus: tuple[dict[str, Any], ...]
) -> None:
    wrapper = load_json(ROOT / "schemas" / "human_annotation_record.schema.json")
    semantic = load_json(ROOT / "schemas" / "semantic_requirements.schema.json")
    records = submission_for(corpus, "annotator_a")
    records[0]["annotation"]["abstained_fields"] = []
    audit = audit_submission(
        records,
        corpus,
        wrapper,
        semantic,
        annotator_id="annotator_a",
    )
    assert any("must exactly match" in issue for issue in audit.issues)


def test_packet_audit_detects_metadata_leak(
    corpus: tuple[dict[str, Any], ...]
) -> None:
    packet = [
        dict(item)
        for item in build_blind_packet(corpus, annotator_id="annotator_a", seed=1)
    ]
    packet[0]["design_strata"] = {"scenario_domain": "healthcare"}
    issues = packet_issues(packet, corpus, annotator_id="annotator_a")
    assert any("leaks fields" in issue for issue in issues)


def test_jsonl_loader_and_writer_are_strict_and_stable(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    records = ({"b": 2, "a": "Türkçe"}, {"x": None})
    write_jsonl(path, records)
    assert load_jsonl(path) == records
    assert path.read_text(encoding="utf-8").startswith('{"a":"Türkçe","b":2}')
    path.write_text('{"ok":true}\n\n', encoding="utf-8")
    with pytest.raises(ValueError, match="blank JSONL line"):
        load_jsonl(path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "JSONL file is empty"),
        ("not-json\n", "invalid JSON"),
        ("[]\n", "record must be an object"),
    ],
)
def test_jsonl_loader_rejects_empty_invalid_and_non_object_records(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_jsonl(path)


def test_build_packet_rejects_fake_annotator_and_negative_seed(
    corpus: tuple[dict[str, Any], ...]
) -> None:
    with pytest.raises(ValueError, match="unsupported annotator"):
        build_blind_packet(corpus, annotator_id="llm_persona", seed=1)
    with pytest.raises(ValueError, match="seed"):
        build_blind_packet(corpus, annotator_id="annotator_a", seed=-1)


def test_corpus_audit_detects_duplicate_text_and_missing_domain(
    corpus: tuple[dict[str, Any], ...], corpus_schema: dict[str, Any]
) -> None:
    mutated = [copy.deepcopy(item) for item in corpus]
    mutated[1]["task_text"] = mutated[0]["task_text"]
    mutated[1]["user_policy_text"] = mutated[0]["user_policy_text"]
    mutated = [
        item
        for item in mutated
        if item["design_strata"]["scenario_domain"] != "agriculture"
    ]
    audit = audit_corpus(mutated, corpus_schema)
    assert not audit.is_valid
    assert any("normalized" in issue for issue in audit.issues)
    assert any("agriculture" in issue for issue in audit.issues)


def test_corpus_audit_detects_sensitive_text_and_coverage_gap(
    corpus: tuple[dict[str, Any], ...], corpus_schema: dict[str, Any]
) -> None:
    mutated = [copy.deepcopy(item) for item in corpus]
    mutated[0]["task_text"] += " Ayrinti icin https://example.test adresine bak."
    for item in mutated:
        item["design_strata"]["semantic_phenomena"] = []
    audit = audit_corpus(mutated, corpus_schema)
    assert any("forbidden url" in issue for issue in audit.issues)
    assert any(
        "pilot lacks required semantic phenomena" in issue for issue in audit.issues
    )


def test_packet_audit_detects_content_order_identity_and_coverage_drift(
    corpus: tuple[dict[str, Any], ...]
) -> None:
    packet = [
        dict(item)
        for item in build_blind_packet(corpus, annotator_id="annotator_a", seed=2)
    ]
    packet[0]["annotator_id"] = "annotator_b"
    packet[0]["presentation_order"] = 99
    packet[0]["task_text"] = "changed"
    packet[0]["task_content_sha256"] = "0" * 64
    packet[1]["task_id"] = "unknown-task"
    packet[2] = dict(packet[3])
    issues = packet_issues(packet, corpus, annotator_id="annotator_a")
    expected_fragments = (
        "wrong annotator id",
        "non-contiguous order",
        "changed task_text",
        "stale task hash",
        "unknown task",
        "duplicate task ids",
        "do not match the corpus",
    )
    assert all(
        any(fragment in issue for issue in issues)
        for fragment in expected_fragments
    )


def test_packet_audit_detects_missing_required_field(
    corpus: tuple[dict[str, Any], ...]
) -> None:
    packet = [
        dict(item)
        for item in build_blind_packet(corpus, annotator_id="annotator_b", seed=3)
    ]
    del packet[0]["task_text"]
    issues = packet_issues(packet, corpus, annotator_id="annotator_b")
    assert any("misses fields" in issue for issue in issues)


def test_submission_audit_detects_identity_language_and_coverage_drift(
    corpus: tuple[dict[str, Any], ...]
) -> None:
    wrapper = load_json(ROOT / "schemas" / "human_annotation_record.schema.json")
    semantic = load_json(ROOT / "schemas" / "semantic_requirements.schema.json")
    records = submission_for(corpus, "annotator_a")
    records[0]["annotator_id"] = "annotator_b"
    records[0]["annotation"]["task_id"] = "pilot_tr_999"
    records[0]["annotation"]["language"] = "en"
    records[1]["task_id"] = "unknown-task"
    records[2] = copy.deepcopy(records[3])
    audit = audit_submission(
        records,
        corpus,
        wrapper,
        semantic,
        annotator_id="annotator_a",
    )
    expected_fragments = (
        "wrong annotator id",
        "task ids disagree",
        "language disagrees",
        "unknown task",
        "duplicate task ids",
        "do not match the corpus",
    )
    assert all(
        any(fragment in issue for issue in audit.issues)
        for fragment in expected_fragments
    )
