"""Frozen multilingual sentence encoder with deterministic linear probes."""

from __future__ import annotations

import importlib
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from sklearn.linear_model import LogisticRegression

from task_offloading.semantic.baselines import (
    SemanticPrediction,
    _build_annotation,
    _label_signature,
    _signature_label,
    semantic_signature,
    visible_text,
)
from task_offloading.semantic.benchmark import ANNOTATION_FIELDS

E5_SMALL_MODEL_ID = "intfloat/multilingual-e5-small"
E5_SMALL_REVISION = "0e60b8d9d2166d80387f86e3b48ec9ced55f4d15"


class SentenceEncoder(Protocol):
    """Minimum embedding interface required by the linear probe."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        """Encode text as one dense row per sentence."""


def load_e5_small(
    *, cache_folder: Path, device: str = "cpu"
) -> SentenceEncoder:
    """Load the exact, revision-pinned encoder without hiding downloads."""

    try:
        module = importlib.import_module("sentence_transformers")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "sentence-transformers is required; install requirements.txt first"
        ) from error
    cache_folder.mkdir(parents=True, exist_ok=True)
    sentence_transformer: Any = vars(module)["SentenceTransformer"]
    model: Any = sentence_transformer(
        E5_SMALL_MODEL_ID,
        revision=E5_SMALL_REVISION,
        cache_folder=str(cache_folder),
        device=device,
    )
    return cast(SentenceEncoder, model)


def _encoder_input(task: Mapping[str, Any]) -> str:
    return f"query: {visible_text(task)}"


class FrozenEncoderLogisticBaseline:
    """Use fixed E5 embeddings and one balanced linear head per field."""

    name = "frozen_multilingual_e5_small_logreg"

    def __init__(
        self,
        encoder: SentenceEncoder,
        *,
        random_seed: int = 20260929,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._encoder = encoder
        self._random_seed = random_seed
        self._batch_size = batch_size
        self._models: dict[str, LogisticRegression | None] = {}
        self._constant_labels: dict[str, str] = {}

    def _encode(self, tasks: Sequence[Mapping[str, Any]]) -> np.ndarray:
        embeddings = self._encoder.encode(
            [_encoder_input(task) for task in tasks],
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        matrix = np.asarray(embeddings, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(tasks):
            raise ValueError("encoder returned an invalid embedding matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("encoder returned non-finite embeddings")
        return matrix

    def fit(
        self,
        tasks: Sequence[Mapping[str, Any]],
        annotations: Sequence[Mapping[str, Any]],
    ) -> None:
        if not tasks or len(tasks) != len(annotations):
            raise ValueError("tasks and annotations must have the same non-zero size")
        matrix = self._encode(tasks)
        for field in ANNOTATION_FIELDS:
            labels = np.asarray(
                [
                    _signature_label(semantic_signature(annotation, field))
                    for annotation in annotations
                ]
            )
            classes = np.unique(labels)
            if len(classes) == 1:
                self._models[field] = None
                self._constant_labels[field] = str(classes[0])
                continue
            model = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2_000,
                random_state=self._random_seed,
                solver="lbfgs",
            )
            model.fit(matrix, labels)
            self._models[field] = model

    def predict(self, task: Mapping[str, Any]) -> SemanticPrediction:
        if set(self._models) != set(ANNOTATION_FIELDS):
            raise RuntimeError("encoder baseline must be fitted before prediction")
        started = time.perf_counter_ns()
        matrix = self._encode((task,))
        signatures: dict[str, tuple[object, ...]] = {}
        confidences: dict[str, float] = {}
        for field in ANNOTATION_FIELDS:
            model = self._models[field]
            if model is None:
                label = self._constant_labels[field]
                confidence = 1.0
            else:
                probabilities = model.predict_proba(matrix)[0]
                best = int(np.argmax(probabilities))
                label = str(model.classes_[best])
                confidence = float(probabilities[best])
            signatures[field] = _label_signature(label)
            confidences[field] = confidence
        annotation = _build_annotation(task, signatures, confidences)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        return SemanticPrediction(annotation, elapsed)
