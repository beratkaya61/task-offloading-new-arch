"""Frozen encoder linear-probe tests that do not download model weights."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from task_offloading.semantic.benchmark import ANNOTATION_FIELDS
from task_offloading.semantic.encoder import (
    E5_SMALL_MODEL_ID,
    E5_SMALL_REVISION,
    FrozenEncoderLogisticBaseline,
    load_e5_small,
)

ROOT = Path(__file__).resolve().parents[2]


class RecordingEncoder:
    """Small deterministic stand-in for the external E5 model."""

    def __init__(self) -> None:
        self.inputs: list[str] = []

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        del batch_size, show_progress_bar, convert_to_numpy
        assert normalize_embeddings
        self.inputs.extend(sentences)
        rows = []
        for text in sentences:
            normalized = text.casefold()
            vector = np.asarray(
                [
                    float("hastane" in normalized),
                    float("fabrika" in normalized),
                    float(len(text) % 7),
                ],
                dtype=np.float64,
            )
            norm = np.linalg.norm(vector)
            rows.append(vector / norm if norm else vector)
        return np.asarray(rows, dtype=np.float64)


class InvalidEncoder(RecordingEncoder):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        del batch_size, show_progress_bar, convert_to_numpy, normalize_embeddings
        return np.asarray([float("nan") for _ in sentences])


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    assert isinstance(value, dict)
    return value


def task(task_id: str, text: str, domain: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "language": "tr",
        "task_text": text,
        "user_policy_text": None,
        "design_strata": {
            "scenario_domain": domain,
            "semantic_phenomena": [],
        },
    }


def annotation(task_id: str, domain: str) -> dict[str, Any]:
    base = {"status": "not_stated", "evidence": [], "confidence": 0.9}
    return {
        "schema_version": "1.0.0",
        "task_id": task_id,
        "language": "tr",
        "domain": {
            "value": domain,
            "status": "explicit",
            "evidence": [domain],
            "confidence": 0.9,
        },
        "latency": {"class": None, "max_ms": None, **base},
        "privacy": {"data_class": None, **base},
        "execution_policy": {
            "device": "unknown",
            "edge": "unknown",
            "cloud": "unknown",
            **base,
        },
        "reliability": {
            "tier": None,
            "min_success_probability": None,
            **base,
        },
        "accuracy": {
            "tier": None,
            "min_score": None,
            "metric": None,
            **base,
        },
        "energy_priority": {"value": None, **base},
        "divisibility": {"value": None, **base},
        "abstained_fields": list(ANNOTATION_FIELDS[1:]),
        "overall_confidence": 0.9,
    }


def test_encoder_linear_probe_is_deterministic_schema_valid_and_text_only() -> None:
    tasks = (
        task("task_001", "Hastane ortak görevi bir", "healthcare"),
        task("task_002", "Hastane ortak görevi iki", "healthcare"),
        task("task_003", "Fabrika ortak görevi bir", "industrial"),
        task("task_004", "Fabrika ortak görevi iki", "industrial"),
    )
    annotations = tuple(
        annotation(str(row["task_id"]), str(row["design_strata"]["scenario_domain"]))
        for row in tasks
    )
    encoder = RecordingEncoder()
    model = FrozenEncoderLogisticBaseline(encoder, batch_size=2)
    model.fit(tasks, annotations)

    evaluation = task("task_999", "Hastane ortak görevi yeni", "industrial")
    first = model.predict(evaluation).annotation
    second = model.predict(evaluation).annotation
    schema = load_json(ROOT / "schemas" / "semantic_requirements.schema.json")

    assert first == second
    assert first["domain"]["value"] == "healthcare"
    assert all(value.startswith("query: ") for value in encoder.inputs)
    assert all("industrial" not in value for value in encoder.inputs)
    assert not tuple(Draft202012Validator(schema).iter_errors(first))


def test_encoder_rejects_invalid_embedding_shape_and_non_finite_values() -> None:
    rows = (task("task_001", "Hastane görevi", "healthcare"),)
    labels = (annotation("task_001", "healthcare"),)

    with pytest.raises(ValueError, match="invalid embedding matrix"):
        FrozenEncoderLogisticBaseline(InvalidEncoder()).fit(rows, labels)


def test_encoder_config_pins_identity_revision_license_and_isolation() -> None:
    config = load_json(ROOT / "configs" / "semantic_encoder.v1.json")
    model = config["model"]

    assert model["model_id"] == E5_SMALL_MODEL_ID
    assert model["revision"] == E5_SMALL_REVISION
    assert model["license"] == "MIT"
    assert model["encoder_frozen"] is True
    assert model["input_prefix"] == "query: "
    assert model["cache_folder"].startswith("models/")
    assert config["evaluation_labels_may_be_used_for_training_or_tuning"] is False


def test_encoder_loader_pins_revision_and_uses_ignored_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    expected = RecordingEncoder()

    def factory(*args: Any, **kwargs: Any) -> RecordingEncoder:
        calls.append((args, kwargs))
        return expected

    module = SimpleNamespace(SentenceTransformer=factory)
    monkeypatch.setattr(
        "task_offloading.semantic.encoder.importlib.import_module",
        lambda name: module,
    )
    cache = tmp_path / "models" / "huggingface"

    loaded = load_e5_small(cache_folder=cache, device="cpu")

    assert loaded is expected
    assert cache.is_dir()
    assert calls == [
        (
            (E5_SMALL_MODEL_ID,),
            {
                "revision": E5_SMALL_REVISION,
                "cache_folder": str(cache),
                "device": "cpu",
            },
        )
    ]


def test_encoder_loader_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def missing(_name: str) -> None:
        raise ModuleNotFoundError("not installed")

    monkeypatch.setattr(
        "task_offloading.semantic.encoder.importlib.import_module", missing
    )

    with pytest.raises(RuntimeError, match=r"install requirements\.txt"):
        load_e5_small(cache_folder=tmp_path)
