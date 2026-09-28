"""Validate and checksum-lock the three completed pilot submissions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from task_offloading.semantic import audit_submission, load_jsonl

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "semantic_benchmark" / "pilot"
DEFAULT_OUTPUT = PILOT / "annotations" / "round_1"


def _load_json(path: Path) -> dict[str, Any]:
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
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(
                f"refusing to overwrite a different locked submission: {path}"
            )
        return
    path.write_bytes(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and checksum-lock pilot round-1 submissions."
    )
    parser.add_argument("--annotator-a", required=True, type=Path)
    parser.add_argument("--annotator-b", required=True, type=Path)
    parser.add_argument("--annotator-c", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = {
        "annotator_a": args.annotator_a.resolve(),
        "annotator_b": args.annotator_b.resolve(),
        "annotator_c": args.annotator_c.resolve(),
    }
    corpus_path = PILOT / "tasks.v1.jsonl"
    corpus = load_jsonl(corpus_path)
    wrapper = _load_json(ROOT / "schemas" / "human_annotation_record.schema.json")
    semantic = _load_json(ROOT / "schemas" / "semantic_requirements.schema.json")
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: dict[str, dict[str, Any]] = {}
    for annotator, source in sources.items():
        records = load_jsonl(source)
        audit = audit_submission(
            records,
            corpus,
            wrapper,
            semantic,
            annotator_id=annotator,
        )
        if not audit.is_valid:
            raise ValueError(
                f"{annotator} submission audit failed:\n- "
                + "\n- ".join(audit.issues)
            )
        content = source.read_bytes()
        target = output_dir / f"{annotator}.jsonl"
        _write_once(target, content)
        entries[annotator] = {
            "path": target.relative_to(ROOT).as_posix(),
            "sha256": _sha256_bytes(content),
            "byte_count": len(content),
            "record_count": len(records),
            "validation_status": "schema_and_evidence_valid",
            "lock_status": "checksum_locked",
        }

    manifest = {
        "manifest_version": "1.0.0",
        "round_id": "pilot_round_1",
        "corpus": {
            "path": corpus_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(corpus_path),
            "task_count": len(corpus),
        },
        "protocol_version": "1.0.0",
        "primary_annotators": ["annotator_a", "annotator_b"],
        "diagnostic_annotators": ["annotator_c"],
        "diagnostic_excluded_from_primary_kappa": True,
        "automatic_majority_vote": False,
        "adjudication_status": "not_started",
        "submissions": entries,
    }
    manifest_path = output_dir / "submission_manifest.v1.json"
    manifest_content = (
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode()
    _write_once(manifest_path, manifest_content)
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={_sha256(manifest_path)}")
    for annotator, entry in entries.items():
        print(f"{annotator}_sha256={entry['sha256']}")
    print("adjudication_status=not_started")


if __name__ == "__main__":
    main()
