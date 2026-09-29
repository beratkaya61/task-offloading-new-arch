"""Validate and render the protocol-1.2 targeted calibration package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from task_offloading.data import sha256_file
from task_offloading.semantic import (
    audit_targeted_calibration,
    load_jsonl,
    render_targeted_calibration,
)

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "data" / "semantic_benchmark" / "calibration"
SOURCE = CALIBRATION / "targeted_examples.v1.json"
SCHEMA = ROOT / "schemas" / "targeted_calibration.schema.json"
PILOT = ROOT / "data" / "semantic_benchmark" / "pilot" / "tasks.v1.jsonl"
PROTOCOL = ROOT / "docs" / "ANNOTATION_PROTOCOL_AMENDMENT_1_2.md"
HTML = CALIBRATION / "targeted_calibration.v1.html"
MANIFEST = CALIBRATION / "manifest.v1.json"


def _load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(content)
    if temporary.read_bytes() != content:
        raise OSError(f"temporary write verification failed: {temporary}")
    temporary.replace(path)


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    package = _load_object(SOURCE)
    schema = _load_object(SCHEMA)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(package)

    audit = audit_targeted_calibration(package, load_jsonl(PILOT))
    if not audit.is_valid:
        raise ValueError(
            "targeted calibration audit failed:\n- " + "\n- ".join(audit.issues)
        )

    rendered = render_targeted_calibration(package).encode("utf-8")
    _atomic_write(HTML, rendered)
    manifest: dict[str, Any] = {
        "manifest_version": "1.0.0",
        "created_on": "2026-09-29",
        "status": "prepared_not_completed_by_humans",
        "purpose": "novel worked examples before the one-time stratified human audit",
        "source": {
            "path": _relative(SOURCE),
            "sha256": sha256_file(SOURCE),
            "example_count": audit.example_count,
        },
        "schema": {"path": _relative(SCHEMA), "sha256": sha256_file(SCHEMA)},
        "protocol": {
            "version": "1.2.0",
            "path": _relative(PROTOCOL),
            "sha256": sha256_file(PROTOCOL),
        },
        "rendered_html": {
            "path": _relative(HTML),
            "sha256": sha256_file(HTML),
            "network_access": False,
        },
        "pilot_exact_text_overlap_count": 0,
        "contains_human_labels": False,
        "is_ground_truth": False,
        "focus_field_counts": dict(audit.focus_field_counts),
    }
    encoded = (
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write(MANIFEST, encoded)
    print(f"html={HTML}")
    print(f"html_sha256={sha256_file(HTML)}")
    print(f"manifest={MANIFEST}")
    print(f"manifest_sha256={sha256_file(MANIFEST)}")


if __name__ == "__main__":
    main()
