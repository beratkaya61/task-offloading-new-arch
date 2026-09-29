"""Validate and checksum-lock the final annotator-A main submission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from task_offloading.semantic import audit_submission, load_jsonl

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data" / "semantic_benchmark" / "main"
OUTPUT = MAIN / "annotations"
TARGET = OUTPUT / "annotator_a.single_human_reference.v1.3.jsonl"
MANIFEST = OUTPUT / "submission_manifest.v1.json"
TASKS = MAIN / "tasks.v1.jsonl"
SPLIT = MAIN / "split_manifest.v1.json"
PACKETS = MAIN / "annotation_packets.v1.json"
PROTOCOL = ROOT / "docs" / "ANNOTATION_PROTOCOL_AMENDMENT_1_3.md"
WRAPPER_SCHEMA = ROOT / "schemas" / "human_annotation_record.schema.json"
SEMANTIC_SCHEMA = ROOT / "schemas" / "semantic_requirements.schema.json"


def _load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(
                f"refusing to overwrite a different locked artifact: {path}"
            )
        return
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(content)
    if temporary.read_bytes() != content:
        raise OSError(f"temporary write verification failed: {temporary}")
    temporary.replace(path)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and lock the final 60-task annotator-A submission."
    )
    parser.add_argument("submission", type=Path)
    return parser.parse_args()


def main() -> None:
    source = _arguments().submission.resolve()
    all_tasks = load_jsonl(TASKS)
    split = _load_object(SPLIT)
    by_id = {str(task["task_id"]): task for task in all_tasks}
    evaluation_ids = list(split["single_human_evaluation"]["task_ids"])
    evaluation = tuple(by_id[str(task_id)] for task_id in evaluation_ids)
    records = load_jsonl(source)
    audit = audit_submission(
        records,
        evaluation,
        _load_object(WRAPPER_SCHEMA),
        _load_object(SEMANTIC_SCHEMA),
        annotator_id="annotator_a",
        expected_protocol_version="1.3.0",
    )
    if not audit.is_valid:
        raise ValueError(
            "annotator_a submission audit failed:\n- "
            + "\n- ".join(audit.issues)
        )

    content = source.read_bytes()
    _write_once(TARGET, content)
    manifest: dict[str, Any] = {
        "manifest_version": "1.0.0",
        "created_on": "2026-09-29",
        "status": "checksum_locked",
        "protocol_version": "1.3.0",
        "annotation_claim": "single_human_reference",
        "adjudicated_gold": False,
        "additional_human_surveys_required": False,
        "submission": {
            "path": TARGET.relative_to(ROOT).as_posix(),
            "source_filename": source.name,
            "sha256": _sha256_bytes(content),
            "byte_count": len(content),
            "record_count": audit.record_count,
            "annotator_id": audit.annotator_id,
            "validation_status": "schema_evidence_hash_and_protocol_valid",
        },
        "bindings": {
            "tasks": {
                "path": TASKS.relative_to(ROOT).as_posix(),
                "sha256": _sha256(TASKS),
            },
            "split": {
                "path": SPLIT.relative_to(ROOT).as_posix(),
                "sha256": _sha256(SPLIT),
            },
            "annotation_packets": {
                "path": PACKETS.relative_to(ROOT).as_posix(),
                "sha256": _sha256(PACKETS),
            },
            "protocol": {
                "path": PROTOCOL.relative_to(ROOT).as_posix(),
                "sha256": _sha256(PROTOCOL),
            },
        },
        "existing_secondary_human_evidence": {
            "pilot_round_1": "annotator_a_annotator_b_annotator_c",
            "pilot_round_2": "annotator_a_annotator_b",
            "new_b_or_c_tasks_required": 0,
        },
    }
    manifest_content = (
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_once(MANIFEST, manifest_content)
    print("status=valid_and_locked")
    print(f"record_count={audit.record_count}")
    print(f"submission_sha256={_sha256(TARGET)}")
    print(f"manifest={MANIFEST}")
    print(f"manifest_sha256={_sha256(MANIFEST)}")


if __name__ == "__main__":
    main()
