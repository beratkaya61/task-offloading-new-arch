"""Manifest loading, semantic validation, checksums, and status gates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from task_offloading.data.types import BenchmarkKind, ProvenanceKind

JsonObject = dict[str, Any]

ALLOWED_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    "catalogued": frozenset(
        {"license_pending", "access_pending", "downloaded", "rejected"}
    ),
    "license_pending": frozenset(
        {"catalogued", "access_pending", "downloaded", "rejected"}
    ),
    "access_pending": frozenset({"catalogued", "downloaded", "rejected"}),
    "downloaded": frozenset({"checksum_verified", "rejected"}),
    "checksum_verified": frozenset({"schema_validated", "rejected"}),
    "schema_validated": frozenset({"rejected"}),
    "rejected": frozenset(),
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a local artifact without loading a large trace into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_matches(path: Path, expected_sha256: str) -> bool:
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
    return sha256_file(path) == expected_sha256


def assert_status_transition(current: str, proposed: str) -> None:
    try:
        allowed = ALLOWED_STATUS_TRANSITIONS[current]
    except KeyError as error:
        raise ValueError(f"unknown current dataset status: {current}") from error
    if proposed not in allowed:
        raise ValueError(f"illegal dataset status transition: {current} -> {proposed}")


def source_is_analysis_ready(source: Mapping[str, Any]) -> bool:
    """Downloaded alone is intentionally insufficient for scientific use."""

    artifacts = source.get("artifacts", [])
    license_info = source.get("license", {})
    return bool(
        source.get("dataset_status") == "schema_validated"
        and source.get("immutable_version") is True
        and license_info.get("verification_status") == "verified"
        and artifacts
        and all(
            artifact.get("checksum", {}).get("verification") == "verified"
            for artifact in artifacts
        )
    )


def _unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def validate_manifest_semantics(manifest: Mapping[str, Any]) -> None:
    expected_vocabulary = {item.value for item in ProvenanceKind}
    if set(manifest["provenance_vocabulary"]) != expected_vocabulary:
        raise ValueError("provenance_vocabulary must contain the frozen seven values")

    sources = manifest["sources"]
    source_ids = [str(source["id"]) for source in sources]
    _unique(source_ids, "source ids")
    source_id_set = set(source_ids)
    for source in sources:
        artifact_ids = [str(item["artifact_id"]) for item in source["artifacts"]]
        column_names = [str(item["name"]) for item in source["columns"]]
        _unique(artifact_ids, f"artifact ids for {source['id']}")
        _unique(column_names, f"column names for {source['id']}")
        if source["immutable_version"] and source["version_kind"] == "mutable_ref":
            raise ValueError(f"{source['id']} cannot call a mutable ref immutable")
        checksum_evidence_complete = all(
            artifact["checksum"]["verification"] == "verified"
            for artifact in source["artifacts"]
        )
        if (
            source["dataset_status"] in {"checksum_verified", "schema_validated"}
            and not checksum_evidence_complete
        ):
            raise ValueError(
                f"{source['id']} claims checksum verification without local evidence"
            )
        if (
            source["dataset_status"] == "license_pending"
            and source["license"]["verification_status"] == "verified"
        ):
            raise ValueError(f"{source['id']} has inconsistent license status")
        if source[
            "dataset_status"
        ] == "schema_validated" and not source_is_analysis_ready(source):
            raise ValueError(
                f"{source['id']} is schema_validated but has incomplete evidence"
            )

    benchmark_ids = [str(item["id"]) for item in manifest["benchmarks"]]
    _unique(benchmark_ids, "benchmark ids")
    kinds = {str(item["kind"]) for item in manifest["benchmarks"]}
    if kinds != {item.value for item in BenchmarkKind}:
        raise ValueError("manifest must define synthetic and trace_driven_hybrid")
    for benchmark in manifest["benchmarks"]:
        unknown_sources = set(benchmark["source_ids"]) - source_id_set
        if unknown_sources:
            raise ValueError(
                f"benchmark {benchmark['id']} references unknown sources: "
                f"{sorted(unknown_sources)}"
            )
        provenance = set(benchmark["field_provenance"])
        if benchmark["kind"] == BenchmarkKind.SYNTHETIC:
            if benchmark["source_ids"] or not provenance <= {
                ProvenanceKind.GENERATED,
                ProvenanceKind.SIMULATED,
            }:
                raise ValueError("synthetic benchmark cannot claim trace sources")
        else:
            has_real_layer = bool(
                provenance & {ProvenanceKind.OBSERVED, ProvenanceKind.MEASURED}
            )
            has_hybrid_layer = bool(
                provenance
                & {
                    ProvenanceKind.MATCHED,
                    ProvenanceKind.GENERATED,
                    ProvenanceKind.SIMULATED,
                }
            )
            if not benchmark["source_ids"] or not (has_real_layer and has_hybrid_layer):
                raise ValueError(
                    "trace_driven_hybrid requires real and constructed layers"
                )


def load_and_validate_manifest(manifest_path: Path, schema_path: Path) -> JsonObject:
    with schema_path.open(encoding="utf-8") as handle:
        schema: JsonObject = json.load(handle)
    Draft202012Validator.check_schema(schema)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest: JsonObject = json.load(handle)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    validator.validate(manifest)
    validate_manifest_semantics(manifest)
    return manifest
