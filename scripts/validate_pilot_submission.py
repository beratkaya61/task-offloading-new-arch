"""Validate one completed pilot annotation file before it can be locked."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from task_offloading.semantic import audit_submission, load_jsonl

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "semantic_benchmark" / "pilot"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tamamlanan kör pilot etiketlerini şema ve metin kanıtıyla doğrula."  # noqa: RUF001
        )
    )
    parser.add_argument(
        "--annotator",
        required=True,
        choices=("annotator_a", "annotator_b", "annotator_c"),
    )
    parser.add_argument("--submission", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission_path: Path = args.submission.resolve()
    records = load_jsonl(submission_path)
    corpus = load_jsonl(PILOT / "tasks.v1.jsonl")
    wrapper = _load_json(ROOT / "schemas" / "human_annotation_record.schema.json")
    semantic = _load_json(ROOT / "schemas" / "semantic_requirements.schema.json")
    audit = audit_submission(
        records,
        corpus,
        wrapper,
        semantic,
        annotator_id=args.annotator,
    )
    if not audit.is_valid:
        raise ValueError("submission audit failed:\n- " + "\n- ".join(audit.issues))

    checksum = hashlib.sha256(submission_path.read_bytes()).hexdigest()
    print(f"annotator_id={audit.annotator_id}")
    print(f"record_count={audit.record_count}")
    print(f"submission_sha256={checksum}")
    print("status=valid_unlocked")
    print("The raw file is valid but is not yet an immutable adjudication input.")


if __name__ == "__main__":
    main()
