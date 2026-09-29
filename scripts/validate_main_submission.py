"""Validate one protocol-1.3 main-corpus human submission without writing it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from task_offloading.semantic import audit_submission, load_jsonl

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data" / "semantic_benchmark" / "main"


def _load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an A/B main semantic annotation JSONL file."
    )
    parser.add_argument("submission", type=Path)
    parser.add_argument(
        "--annotator",
        required=True,
        choices=("annotator_a",),
    )
    parser.add_argument(
        "--partition",
        required=True,
        choices=("single_human_evaluation",),
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    all_tasks = load_jsonl(MAIN / "tasks.v1.jsonl")
    by_id = {str(task["task_id"]): task for task in all_tasks}
    split = _load_object(MAIN / "split_manifest.v1.json")
    task_ids = list(split[args.partition]["task_ids"])
    corpus = tuple(by_id[str(task_id)] for task_id in task_ids)
    audit = audit_submission(
        load_jsonl(args.submission.resolve()),
        corpus,
        _load_object(ROOT / "schemas" / "human_annotation_record.schema.json"),
        _load_object(ROOT / "schemas" / "semantic_requirements.schema.json"),
        annotator_id=args.annotator,
        expected_protocol_version="1.3.0",
    )
    if not audit.is_valid:
        print("status=invalid")
        for issue in audit.issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("status=valid")
    print(f"annotator={audit.annotator_id}")
    print(f"partition={args.partition}")
    print(f"record_count={audit.record_count}")


if __name__ == "__main__":
    main()
