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
    parser.add_argument("--annotator-c", required=True, type=Path)
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
        "annotator_c": args.annotator_c.resolve(),
    }
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
    a_c = pairwise_agreement(records["annotator_a"], records["annotator_c"])
    b_c = pairwise_agreement(records["annotator_b"], records["annotator_c"])
    diagnostic = diagnose_third_annotator(
        records["annotator_a"],
        records["annotator_b"],
        records["annotator_c"],
    )
    disagreements = disagreement_rows(
        records["annotator_a"],
        records["annotator_b"],
        records["annotator_c"],
    )

    result = {
        "analysis_version": "1.0.0",
        "policy": {
            "primary_pair": ["annotator_a", "annotator_b"],
            "diagnostic_annotator": "annotator_c",
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
        "diagnostic_pairwise": [a_c.to_dict(), b_c.to_dict()],
        "third_annotator_diagnostic": [asdict(item) for item in diagnostic],
        "primary_disagreement_count": len(disagreements),
        "primary_disagreements": disagreements,
    }
    output: Path = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

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
