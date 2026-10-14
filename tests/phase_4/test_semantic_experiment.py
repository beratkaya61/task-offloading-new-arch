"""Direct tests for reproducible semantic experiment file contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from task_offloading.semantic.baselines import SemanticPrediction
from task_offloading.semantic.experiment import (
    load_json_object,
    ordered_subset,
    prediction_rows,
    sha256,
    task_ids,
    write_json_atomic,
)


def test_json_hash_and_atomic_write_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    write_json_atomic(path, {"value": "Türkçe", "count": 2})

    assert load_json_object(path) == {"value": "Türkçe", "count": 2}
    assert sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not path.with_suffix(".json.tmp").exists()


def test_load_json_object_rejects_non_object_root(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON root must be an object"):
        load_json_object(path)


def test_task_id_contract_orders_rows_and_rejects_missing_or_duplicates() -> None:
    rows = ({"task_id": "task_b"}, {"task_id": "task_a"})

    assert task_ids(rows) == {"task_a", "task_b"}
    assert ordered_subset(rows, ("task_a", "task_b")) == (rows[1], rows[0])
    with pytest.raises(ValueError, match="lack task ids"):
        ordered_subset(rows, ("task_missing",))
    with pytest.raises(ValueError, match="duplicate task ids"):
        task_ids((rows[0], rows[0]))
    with pytest.raises(ValueError, match="duplicate task ids"):
        ordered_subset((rows[0], rows[0]), ("task_b",))


def test_prediction_rows_wrap_annotations_without_timing_noise() -> None:
    prediction = SemanticPrediction(
        annotation={"task_id": "task_001", "payload": "kept"},
        inference_ms=123.0,
    )

    assert prediction_rows("test_model", (prediction,)) == (
        {
            "prediction_version": "1.0.0",
            "model_name": "test_model",
            "task_id": "task_001",
            "annotation": {"task_id": "task_001", "payload": "kept"},
        },
    )
