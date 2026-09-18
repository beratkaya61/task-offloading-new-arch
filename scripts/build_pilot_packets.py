"""Validate the pilot corpus and reproducibly build blind A/B packets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from task_offloading.data import sha256_file
from task_offloading.semantic import (
    audit_corpus,
    build_blind_packet,
    load_jsonl,
    packet_issues,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = ROOT / "data" / "semantic_benchmark" / "pilot"
TASKS_PATH = PILOT_DIR / "tasks.v1.jsonl"
SCHEMA_PATH = ROOT / "schemas" / "semantic_task.schema.json"
PACKET_DIR = PILOT_DIR / "packets"
MANIFEST_PATH = PILOT_DIR / "pilot_manifest.v1.json"
SEEDS = {"annotator_a": 41041, "annotator_b": 92093}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def main() -> None:
    corpus = load_jsonl(TASKS_PATH)
    schema = _load_json(SCHEMA_PATH)
    audit = audit_corpus(corpus, schema, expected_count=20)
    if not audit.is_valid:
        raise ValueError("pilot corpus audit failed:\n- " + "\n- ".join(audit.issues))

    packets: dict[str, tuple[dict[str, Any], ...]] = {}
    packet_paths: dict[str, Path] = {}
    for annotator_id, seed in SEEDS.items():
        packet = build_blind_packet(corpus, annotator_id=annotator_id, seed=seed)
        issues = packet_issues(packet, corpus, annotator_id=annotator_id)
        if issues:
            raise ValueError("blind packet audit failed:\n- " + "\n- ".join(issues))
        path = PACKET_DIR / f"{annotator_id}.v1.jsonl"
        write_jsonl(path, packet)
        packets[annotator_id] = packet
        packet_paths[annotator_id] = path

    orders = {
        annotator_id: tuple(str(row["task_id"]) for row in packet)
        for annotator_id, packet in packets.items()
    }
    if orders["annotator_a"] == orders["annotator_b"]:
        raise ValueError("annotator packets must use different presentation orders")

    manifest = {
        "manifest_version": "1.0.0",
        "created_on": "2026-09-14",
        "corpus_version": "pilot-1.0.0",
        "protocol_version": "1.0.0",
        "annotation_schema_version": "1.0.0",
        "task_count": len(corpus),
        "gold_status": "not_collected",
        "human_annotators_required": 2,
        "design_metadata_is_gold": False,
        "corpus": {
            "path": TASKS_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(TASKS_PATH),
        },
        "packets": {
            annotator_id: {
                "path": path.relative_to(ROOT).as_posix(),
                "seed": SEEDS[annotator_id],
                "sha256": sha256_file(path),
            }
            for annotator_id, path in packet_paths.items()
        },
        "hidden_from_annotators": [
            "provenance",
            "grouping",
            "design_strata",
            "LLM output",
            "other annotator output",
            "network/server state",
            "oracle action",
            "split membership",
        ],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
