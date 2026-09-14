"""Schema, provenance, checksum, license, and lifecycle gate tests."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema.exceptions import ValidationError

from task_offloading.data import (
    ProvenanceKind,
    assert_status_transition,
    checksum_matches,
    load_and_validate_manifest,
    sha256_file,
    source_is_analysis_ready,
    validate_manifest_semantics,
)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "configs" / "data_sources.v1.json"
SCHEMA_PATH = ROOT / "schemas" / "dataset_manifest.schema.json"


def manifest() -> dict[str, object]:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)  # type: ignore[no-any-return]


def source_by_id(values: dict[str, object], source_id: str) -> dict[str, object]:
    sources = values["sources"]
    assert isinstance(sources, list)
    return next(item for item in sources if item["id"] == source_id)


def test_manifest_schema_and_semantic_contract_are_valid() -> None:
    values = load_and_validate_manifest(MANIFEST_PATH, SCHEMA_PATH)
    assert len(values["sources"]) == 6
    assert {item["kind"] for item in values["benchmarks"]} == {
        "synthetic",
        "trace_driven_hybrid",
    }


def test_every_column_has_exactly_one_frozen_provenance_value() -> None:
    values = manifest()
    allowed = {item.value for item in ProvenanceKind}
    for source in values["sources"]:  # type: ignore[union-attr]
        for column in source["columns"]:
            assert column["provenance"] in allowed


def test_accessibility_is_not_misrepresented_as_license_permission() -> None:
    values = manifest()
    bupt = source_by_id(values, "bupt_edge_computing_dataset")
    assert bupt["access_status"] == "public"
    assert bupt["dataset_status"] == "license_pending"
    assert not source_is_analysis_ready(bupt)


def test_downloaded_source_is_not_automatically_analysis_ready() -> None:
    values = manifest()
    uci = copy.deepcopy(source_by_id(values, "uci_mec_execution_times_859"))
    uci["dataset_status"] = "downloaded"
    assert not source_is_analysis_ready(uci)


def test_schema_validated_claim_requires_complete_evidence() -> None:
    values = manifest()
    uci = copy.deepcopy(source_by_id(values, "uci_mec_execution_times_859"))
    uci["dataset_status"] = "schema_validated"
    with pytest.raises(ValueError, match="evidence"):
        validate_manifest_semantics(
            {
                "provenance_vocabulary": values["provenance_vocabulary"],
                "sources": [uci],
                "benchmarks": [
                    {
                        "id": "synthetic_v1",
                        "kind": "synthetic",
                        "source_ids": [],
                        "field_provenance": ["simulated"],
                    },
                    {
                        "id": "trace_driven_hybrid_v1",
                        "kind": "trace_driven_hybrid",
                        "source_ids": [uci["id"]],
                        "field_provenance": ["measured", "matched"],
                    },
                ],
            }
        )


def test_checksum_verified_claim_requires_local_verification() -> None:
    values = manifest()
    alibaba = source_by_id(values, "alibaba_cluster_trace_v2018")
    alibaba["dataset_status"] = "checksum_verified"
    with pytest.raises(ValueError, match="without local evidence"):
        validate_manifest_semantics(values)


def test_manifest_schema_rejects_non_sha256_checksum() -> None:
    values = manifest()
    source = source_by_id(values, "alibaba_cluster_trace_v2018")
    artifact = source["artifacts"][0]  # type: ignore[index]
    artifact["checksum"]["value"] = "not-a-digest"  # type: ignore[index]
    with SCHEMA_PATH.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    from jsonschema import Draft202012Validator

    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(values)


def test_streaming_checksum_matches_known_content(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"phase-3-checksum")
    expected = hashlib.sha256(b"phase-3-checksum").hexdigest()
    assert sha256_file(artifact, chunk_size=3) == expected
    assert checksum_matches(artifact, expected)
    assert not checksum_matches(artifact, "0" * 64)


def test_checksum_guards_reject_invalid_inputs(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"content")
    with pytest.raises(ValueError, match="chunk_size"):
        sha256_file(artifact, chunk_size=0)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        checksum_matches(artifact, "INVALID")


def test_analysis_ready_requires_and_accepts_all_evidence() -> None:
    values = manifest()
    uci = source_by_id(values, "uci_mec_execution_times_859")
    uci["dataset_status"] = "schema_validated"
    artifact = uci["artifacts"][0]  # type: ignore[index]
    artifact["checksum"]["value"] = "a" * 64  # type: ignore[index]
    artifact["checksum"]["verification"] = "verified"  # type: ignore[index]
    validate_manifest_semantics(values)
    assert source_is_analysis_ready(uci)


@pytest.mark.parametrize(
    ("current", "proposed"),
    [
        ("catalogued", "downloaded"),
        ("downloaded", "checksum_verified"),
        ("checksum_verified", "schema_validated"),
    ],
)
def test_valid_dataset_status_progression(current: str, proposed: str) -> None:
    assert_status_transition(current, proposed)


@pytest.mark.parametrize(
    ("current", "proposed"),
    [
        ("catalogued", "schema_validated"),
        ("downloaded", "schema_validated"),
        ("rejected", "catalogued"),
    ],
)
def test_status_cannot_skip_evidence_gates(current: str, proposed: str) -> None:
    with pytest.raises(ValueError, match="illegal"):
        assert_status_transition(current, proposed)


def test_unknown_current_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown current"):
        assert_status_transition("invented", "downloaded")


def test_benchmark_cannot_call_only_simulated_fields_trace_driven() -> None:
    values = manifest()
    hybrid = next(
        item
        for item in values["benchmarks"]  # type: ignore[union-attr]
        if item["kind"] == "trace_driven_hybrid"
    )
    hybrid["field_provenance"] = ["simulated"]
    with pytest.raises(ValueError, match="real and constructed"):
        validate_manifest_semantics(values)


def test_duplicate_source_identifier_is_rejected() -> None:
    values = manifest()
    sources = values["sources"]
    assert isinstance(sources, list)
    sources.append(copy.deepcopy(sources[0]))
    with pytest.raises(ValueError, match="source ids must be unique"):
        validate_manifest_semantics(values)


def test_mutable_reference_cannot_be_claimed_immutable() -> None:
    values = manifest()
    eua = source_by_id(values, "eua_dataset")
    eua["immutable_version"] = True
    with pytest.raises(ValueError, match="mutable ref immutable"):
        validate_manifest_semantics(values)


def test_license_status_fields_must_be_consistent() -> None:
    values = manifest()
    bupt = source_by_id(values, "bupt_edge_computing_dataset")
    license_info = bupt["license"]
    assert isinstance(license_info, dict)
    license_info["verification_status"] = "verified"
    with pytest.raises(ValueError, match="inconsistent license"):
        validate_manifest_semantics(values)


def test_benchmark_cannot_reference_unknown_source() -> None:
    values = manifest()
    benchmark = values["benchmarks"][1]  # type: ignore[index]
    benchmark["source_ids"].append("unknown_source")  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="unknown sources"):
        validate_manifest_semantics(values)


def test_synthetic_benchmark_cannot_reference_trace_source() -> None:
    values = manifest()
    benchmark = values["benchmarks"][0]  # type: ignore[index]
    benchmark["source_ids"] = ["eua_dataset"]  # type: ignore[index]
    with pytest.raises(ValueError, match="synthetic benchmark"):
        validate_manifest_semantics(values)
