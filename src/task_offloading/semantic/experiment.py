"""Shared, deterministic file contracts for semantic benchmark runners."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from task_offloading.semantic.baselines import SemanticPrediction


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object and reject other root types."""

    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Write stable UTF-8 JSON through a verified temporary file."""

    content = (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(content)
    if temporary.read_bytes() != content:
        raise OSError(f"temporary write verification failed: {temporary}")
    temporary.replace(path)


def task_ids(rows: Sequence[dict[str, Any]]) -> set[str]:
    """Return unique task IDs or reject duplicates."""

    identifiers = [str(row["task_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("rows contain duplicate task ids")
    return set(identifiers)


def ordered_subset(
    rows: Sequence[dict[str, Any]], identifiers: Sequence[str]
) -> tuple[dict[str, Any], ...]:
    """Select rows in a declared ID order without silently losing IDs."""

    by_id = {str(row["task_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("rows contain duplicate task ids")
    missing = set(identifiers) - set(by_id)
    if missing:
        raise ValueError("rows lack task ids: " + ", ".join(sorted(missing)))
    return tuple(by_id[identifier] for identifier in identifiers)


def prediction_rows(
    model_name: str, predictions: Sequence[SemanticPrediction]
) -> tuple[dict[str, Any], ...]:
    """Wrap schema annotations in the versioned prediction record format."""

    return tuple(
        {
            "prediction_version": "1.0.0",
            "model_name": model_name,
            "task_id": prediction.annotation["task_id"],
            "annotation": prediction.annotation,
        }
        for prediction in predictions
    )
