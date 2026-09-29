"""Build fresh blind A/B packets for pilot Round 2 under protocol 1.1.0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from task_offloading.data import sha256_file
from task_offloading.semantic import (
    audit_annotation_release,
    audit_corpus,
    build_blind_packet,
    load_jsonl,
    packet_issues,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "semantic_benchmark" / "pilot"
ROUND_1 = PILOT / "annotations" / "round_1"
ROUND_2 = PILOT / "round_2"
PACKETS = ROUND_2 / "packets"
TASKS = PILOT / "tasks.v1.jsonl"
PROTOCOL = ROOT / "docs" / "ANNOTATION_PROTOCOL.md"
SCHEMA = ROOT / "schemas" / "semantic_task.schema.json"
MANIFEST = ROUND_2 / "pilot_manifest.v1.json"
PACKET_VERSION = "1.1.0"
PROTOCOL_VERSION = "1.1.0"
SEEDS = {"annotator_a": 17389, "annotator_b": 68443}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: MappingValue) -> None:
    content = (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(content)
    if temporary.read_bytes() != content:
        raise OSError(f"temporary write verification failed: {temporary}")
    temporary.replace(path)


MappingValue = dict[str, Any]


def main() -> None:
    corpus = load_jsonl(TASKS)
    audit = audit_corpus(corpus, _load_json(SCHEMA), expected_count=20)
    if not audit.is_valid:
        raise ValueError("pilot corpus audit failed:\n- " + "\n- ".join(audit.issues))
    release = audit_annotation_release(corpus)
    if not release.is_ready:
        raise ValueError(
            "pilot annotation release blocked:\n- " + "\n- ".join(release.issues)
        )

    packets: dict[str, tuple[dict[str, Any], ...]] = {}
    paths: dict[str, Path] = {}
    for annotator, seed in SEEDS.items():
        packet = build_blind_packet(
            corpus,
            annotator_id=annotator,
            seed=seed,
            packet_version=PACKET_VERSION,
            protocol_version=PROTOCOL_VERSION,
        )
        issues = packet_issues(
            packet,
            corpus,
            annotator_id=annotator,
            packet_version=PACKET_VERSION,
            protocol_version=PROTOCOL_VERSION,
        )
        if issues:
            raise ValueError("blind packet audit failed:\n- " + "\n- ".join(issues))
        path = PACKETS / f"{annotator}.v1.1.jsonl"
        write_jsonl(path, packet)
        packets[annotator] = packet
        paths[annotator] = path

    orders = {
        annotator: tuple(str(row["task_id"]) for row in packet)
        for annotator, packet in packets.items()
    }
    if len(set(orders.values())) != len(orders):
        raise ValueError("Round 2 A/B packets must use different orders")
    for annotator, order in orders.items():
        old_packet = load_jsonl(PILOT / "packets" / f"{annotator}.v1.jsonl")
        old_order = tuple(str(row["task_id"]) for row in old_packet)
        if order == old_order:
            raise ValueError(f"{annotator} Round 2 order repeats Round 1")

    agreement = ROUND_1 / "agreement.v1.json"
    manifest: MappingValue = {
        "manifest_version": "1.0.0",
        "round_id": "pilot_round_2",
        "created_on": "2026-09-29",
        "purpose": "blind repeat after pre-main-study protocol clarification",
        "corpus": {
            "path": TASKS.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(TASKS),
            "task_count": len(corpus),
        },
        "protocol": {
            "version": PROTOCOL_VERSION,
            "path": PROTOCOL.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(PROTOCOL),
        },
        "semantic_schema_version": "1.0.0",
        "primary_annotators": ["annotator_a", "annotator_b"],
        "diagnostic_annotators": [],
        "diagnostic_note": "annotator_c is not repeated in Round 2",
        "blindness": {
            "round_1_answers_visible": False,
            "other_annotator_answers_visible": False,
            "round_1_disagreement_list_visible": False,
        },
        "source_round_1_agreement": {
            "path": agreement.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(agreement),
            "used_only_for_protocol_and_form_quality_improvement": True,
        },
        "quality_gates": [
            "confidence_must_be_selected",
            "latency_unit_and_class_consistency",
            "numeric_evidence_consistency",
            "execution_status_completeness",
        ],
        "packets": {
            annotator: {
                "path": path.relative_to(ROOT).as_posix(),
                "seed": SEEDS[annotator],
                "sha256": sha256_file(path),
                "packet_version": PACKET_VERSION,
                "protocol_version": PROTOCOL_VERSION,
            }
            for annotator, path in paths.items()
        },
        "submission_status": "not_collected",
        "adjudication_status": "not_started",
    }
    _atomic_json(MANIFEST, manifest)
    print(f"manifest={MANIFEST}")
    print(f"manifest_sha256={sha256_file(MANIFEST)}")
    for annotator, path in paths.items():
        print(f"{annotator}_packet_sha256={sha256_file(path)}")


if __name__ == "__main__":
    main()
