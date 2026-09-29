"""Validate and compare the two primary pilot annotators plus diagnostic C."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from task_offloading.semantic import (
    audit_submission,
    diagnose_third_annotator,
    disagreement_rows,
    load_jsonl,
    pairwise_agreement,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "semantic_benchmark" / "pilot"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate A/B/C pilot labels and compute pairwise agreement."
    )
    parser.add_argument("--annotator-a", required=True, type=Path)
    parser.add_argument("--annotator-b", required=True, type=Path)
    parser.add_argument("--annotator-c", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "semantic_annotation" / "agreement.v1.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "annotator_a": args.annotator_a.resolve(),
        "annotator_b": args.annotator_b.resolve(),
    }
    if args.annotator_c is not None:
        paths["annotator_c"] = args.annotator_c.resolve()
    corpus = load_jsonl(PILOT / "tasks.v1.jsonl")
    wrapper = _load_json(ROOT / "schemas" / "human_annotation_record.schema.json")
    semantic = _load_json(ROOT / "schemas" / "semantic_requirements.schema.json")
    records = {annotator: load_jsonl(path) for annotator, path in paths.items()}

    for annotator, submission in records.items():
        audit = audit_submission(
            submission,
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

    primary = pairwise_agreement(records["annotator_a"], records["annotator_b"])
    diagnostic_records = records.get("annotator_c")
    if diagnostic_records is None:
        diagnostic_pairwise: list[dict[str, Any]] = []
        diagnostic: list[dict[str, Any]] = []
        disagreements = disagreement_rows(
            records["annotator_a"], records["annotator_b"]
        )
    else:
        diagnostic_pairwise = [
            pairwise_agreement(
                records["annotator_a"], diagnostic_records
            ).to_dict(),
            pairwise_agreement(
                records["annotator_b"], diagnostic_records
            ).to_dict(),
        ]
        diagnostic = [
            asdict(item)
            for item in diagnose_third_annotator(
                records["annotator_a"],
                records["annotator_b"],
                diagnostic_records,
            )
        ]
        disagreements = disagreement_rows(
            records["annotator_a"],
            records["annotator_b"],
            diagnostic_records,
        )

    result = {
        "analysis_version": "1.0.0",
        "policy": {
            "primary_pair": ["annotator_a", "annotator_b"],
            "diagnostic_annotator": (
                "annotator_c" if diagnostic_records is not None else None
            ),
            "diagnostic_excluded_from_primary_kappa": True,
            "automatic_majority_vote": False,
            "adjudication_performed": False,
        },
        "inputs": {
            annotator: {
                "sha256": _sha256(path),
                "byte_count": path.stat().st_size,
                "record_count": len(records[annotator]),
            }
            for annotator, path in paths.items()
        },
        "primary_pairwise": primary.to_dict(),
        "diagnostic_pairwise": diagnostic_pairwise,
        "third_annotator_diagnostic": diagnostic,
        "primary_disagreement_count": len(disagreements),
        "primary_disagreements": disagreements,
    }
    output: Path = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    content = (
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode()
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_bytes(content)
    if temporary.read_bytes() != content:
        raise OSError(f"temporary write verification failed: {temporary}")
    temporary.replace(output)

    print(f"output={output}")
    print(f"output_sha256={_sha256(output)}")
    print(f"primary_task_count={primary.task_count}")
    print(f"primary_exact_field_agreement={primary.exact_field_agreement:.3f}")
    print(f"primary_exact_record_agreement={primary.exact_record_agreement:.3f}")
    print(f"primary_disagreement_cells={len(disagreements)}")
    print("diagnostic_c_in_primary_kappa=false")
    print("adjudication_performed=false")


if __name__ == "__main__":
    main()
