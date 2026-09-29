"""Evaluate the pre-main-study pilot agreement quality gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROUND_1 = (
    ROOT
    / "data"
    / "semantic_benchmark"
    / "pilot"
    / "annotations"
    / "round_1"
    / "agreement.v1.json"
)
DEFAULT_ROUND_2 = (
    ROOT
    / "data"
    / "semantic_benchmark"
    / "pilot"
    / "round_2"
    / "annotations"
    / "agreement.v1.json"
)
DEFAULT_OUTPUT = DEFAULT_ROUND_2.parent / "quality_gate.v1.json"
MINIMUM_KAPPA = 0.70
MINIMUM_EXACT_FIELD_AGREEMENT = 0.80
MINIMUM_VALUE_SAMPLE = 5


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    content = (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(content)
    if temporary.read_bytes() != content:
        raise OSError(f"temporary write verification failed: {temporary}")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Round 2 against the pre-main pilot agreement gate."
    )
    parser.add_argument("--round-1", type=Path, default=DEFAULT_ROUND_1)
    parser.add_argument("--round-2", type=Path, default=DEFAULT_ROUND_2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    round_1_path = args.round_1.resolve()
    round_2_path = args.round_2.resolve()
    output = args.output.resolve()
    round_1 = _load_json(round_1_path)["primary_pairwise"]
    round_2 = _load_json(round_2_path)["primary_pairwise"]

    checks: list[dict[str, Any]] = []

    def add_check(name: str, value: float | None, threshold: float) -> None:
        checks.append(
            {
                "name": name,
                "value": value,
                "threshold": threshold,
                "passed": value is not None and value >= threshold,
            }
        )

    add_check(
        "overall.exact_field_agreement",
        round_2["exact_field_agreement"],
        MINIMUM_EXACT_FIELD_AGREEMENT,
    )
    insufficient: list[dict[str, Any]] = []
    for field in round_2["fields"]:
        name = str(field["field"])
        add_check(f"{name}.status_kappa", field["status"]["kappa"], MINIMUM_KAPPA)
        value = field["value"]
        if value["sample_size"] < MINIMUM_VALUE_SAMPLE:
            insufficient.append(
                {
                    "name": f"{name}.value_kappa",
                    "sample_size": value["sample_size"],
                    "minimum_sample_size": MINIMUM_VALUE_SAMPLE,
                }
            )
        else:
            add_check(f"{name}.value_kappa", value["kappa"], MINIMUM_KAPPA)

    execution = round_2["execution_policy"]
    add_check(
        "execution_policy.status_kappa",
        execution["status"]["kappa"],
        MINIMUM_KAPPA,
    )
    for target, metric in execution["targets"]:
        add_check(
            f"execution_policy.{target}_kappa",
            metric["kappa"],
            MINIMUM_KAPPA,
        )

    failed = [check for check in checks if not check["passed"]]
    passed = not failed and not insufficient
    result = {
        "gate_version": "1.0.0",
        "status": "passed" if passed else "failed",
        "main_240_annotation_allowed": passed,
        "thresholds": {
            "minimum_kappa": MINIMUM_KAPPA,
            "minimum_exact_field_agreement": MINIMUM_EXACT_FIELD_AGREEMENT,
            "minimum_value_sample_size": MINIMUM_VALUE_SAMPLE,
        },
        "inputs": {
            "round_1": {
                "path": round_1_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(round_1_path),
            },
            "round_2": {
                "path": round_2_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(round_2_path),
            },
        },
        "round_comparison": {
            "exact_field_agreement": {
                "round_1": round_1["exact_field_agreement"],
                "round_2": round_2["exact_field_agreement"],
                "delta": round(
                    round_2["exact_field_agreement"]
                    - round_1["exact_field_agreement"],
                    6,
                ),
            },
            "exact_record_agreement": {
                "round_1": round_1["exact_record_agreement"],
                "round_2": round_2["exact_record_agreement"],
                "delta": round(
                    round_2["exact_record_agreement"]
                    - round_1["exact_record_agreement"],
                    6,
                ),
            },
        },
        "checks": checks,
        "failed_checks": failed,
        "insufficient_value_samples": insufficient,
        "required_next_action": (
            "targeted annotator calibration with novel examples; do not start "
            "the main 240-task annotation"
            if not passed
            else "proceed to separately recorded pilot adjudication"
        ),
    }
    _atomic_json(output, result)
    print(f"output={output}")
    print(f"output_sha256={_sha256(output)}")
    print(f"status={result['status']}")
    print(f"failed_checks={len(failed)}")
    print(f"insufficient_value_samples={len(insufficient)}")
    print(f"main_240_annotation_allowed={str(passed).lower()}")


if __name__ == "__main__":
    main()
