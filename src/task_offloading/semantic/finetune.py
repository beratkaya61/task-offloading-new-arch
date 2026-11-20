"""Fine-tuned small multilingual encoder with one classification head per field.

Hyper-parameters are fixed before any evaluation and never tuned on the
locked evaluation labels: mean-pooled ``intfloat/multilingual-e5-small``, eight
linear heads, AdamW (lr 3e-5, weight decay 0.01), batch 16, 4 epochs (compute
budget fixed on 2026-11-20 before any v3 result of this model), max 128 tokens
with per-batch dynamic padding, class-balanced cross-entropy per field, fixed
seed.  The class set of
each head is the set of weak-label signatures seen in training, exactly as for
the TF-IDF and frozen-encoder baselines, so every model has the same output
vocabulary.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from task_offloading.semantic.baselines import (
    SemanticPrediction,
    _build_annotation,
    _label_signature,
    _signature_label,
    semantic_signature,
    visible_text,
)
from task_offloading.semantic.benchmark import ANNOTATION_FIELDS
from task_offloading.semantic.encoder import E5_SMALL_MODEL_ID, E5_SMALL_REVISION


class _MultiHead(nn.Module):
    def __init__(self, backbone: nn.Module, hidden: int, sizes: dict[str, int]):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict(
            {field: nn.Linear(hidden, size) for field, size in sizes.items()}
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = self.dropout(pooled)
        return {field: head(pooled) for field, head in self.heads.items()}


class FineTunedEncoderBaseline:
    name = "e5_small_finetuned"

    def __init__(
        self,
        *,
        cache_folder: Path,
        random_seed: int = 20260929,
        epochs: int = 4,
        batch_size: int = 16,
        learning_rate: float = 3e-5,
        max_length: int = 128,
    ) -> None:
        self._cache = str(cache_folder)
        self._seed = random_seed
        self._epochs = epochs
        self._batch = batch_size
        self._lr = learning_rate
        self._max_length = max_length
        self._classes: dict[str, list[str]] = {}
        self._model: _MultiHead | None = None
        self._tokenizer: Any = None

    def _tokens(self, tasks: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [f"query: {visible_text(task)}" for task in tasks]
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def fit(
        self,
        tasks: Sequence[Mapping[str, Any]],
        annotations: Sequence[Mapping[str, Any]],
    ) -> None:
        if not tasks or len(tasks) != len(annotations):
            raise ValueError("tasks and annotations must have the same non-zero size")
        torch.manual_seed(self._seed)
        np.random.seed(self._seed)
        self._tokenizer = AutoTokenizer.from_pretrained(
            E5_SMALL_MODEL_ID, revision=E5_SMALL_REVISION, cache_dir=self._cache
        )
        backbone = AutoModel.from_pretrained(
            E5_SMALL_MODEL_ID, revision=E5_SMALL_REVISION, cache_dir=self._cache
        )
        targets: dict[str, torch.Tensor] = {}
        weights: dict[str, torch.Tensor] = {}
        for field in ANNOTATION_FIELDS:
            labels = [
                _signature_label(semantic_signature(annotation, field))
                for annotation in annotations
            ]
            classes = sorted(set(labels))
            self._classes[field] = classes
            index = {label: i for i, label in enumerate(classes)}
            target = torch.tensor([index[label] for label in labels])
            counts = torch.bincount(target, minlength=len(classes)).float()
            targets[field] = target
            weights[field] = len(labels) / (len(classes) * counts.clamp(min=1.0))
        model = _MultiHead(
            backbone,
            int(backbone.config.hidden_size),
            {field: len(self._classes[field]) for field in ANNOTATION_FIELDS},
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self._lr, weight_decay=0.01
        )
        losses = {
            field: nn.CrossEntropyLoss(weight=weights[field])
            for field in ANNOTATION_FIELDS
        }
        generator = torch.Generator().manual_seed(self._seed)
        model.train()
        for epoch in range(self._epochs):
            order = torch.randperm(len(tasks), generator=generator)
            total = 0.0
            started = time.perf_counter()
            for start in range(0, len(tasks), self._batch):
                batch = order[start : start + self._batch]
                tokens = self._tokens([tasks[int(i)] for i in batch])
                logits = model(tokens["input_ids"], tokens["attention_mask"])
                loss = sum(
                    losses[field](logits[field], targets[field][batch])
                    for field in ANNOTATION_FIELDS
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss)
            print(
                f"  epoch {epoch + 1}/{self._epochs} loss={total:.3f} "
                f"({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
        model.eval()
        self._model = model

    def predict(self, task: Mapping[str, Any]) -> SemanticPrediction:
        if self._model is None:
            raise RuntimeError("fine-tuned encoder must be fitted before prediction")
        started = time.perf_counter_ns()
        tokens = self._tokens([task])
        with torch.no_grad():
            logits = self._model(tokens["input_ids"], tokens["attention_mask"])
        signatures: dict[str, tuple[object, ...]] = {}
        confidences: dict[str, float] = {}
        for field in ANNOTATION_FIELDS:
            probabilities = torch.softmax(logits[field][0], dim=-1)
            best = int(torch.argmax(probabilities))
            signatures[field] = _label_signature(self._classes[field][best])
            confidences[field] = float(probabilities[best])
        annotation = _build_annotation(task, signatures, confidences)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        return SemanticPrediction(annotation, elapsed)


__all__ = ["FineTunedEncoderBaseline"]
