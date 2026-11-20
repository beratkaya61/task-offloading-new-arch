# ruff: noqa: E501, RUF001
"""Few-shot LLM semantic extractor (local, open-weight, CPU-capable).

The LLM only reads the visible task text and returns one categorical label per
field; it never sees the network state, the evaluation labels or any oracle.
Design choices fixed before evaluation:

* Few-shot examples are drawn deterministically from the *training* tasks
  (development split, machine weak labels): the ``k`` examples that state the
  most requirement fields, ties broken by task id.
* The allowed label values per field are the label vocabulary observed in the
  training annotations, exactly the output vocabulary of the other learned
  extractors.  Numeric thresholds (ms, %, F1) are filled by the same shared
  number parser that every extractor uses.
* Greedy decoding; an unparsable or out-of-vocabulary field falls back to
  ``not_stated`` and is counted.
* Per-field confidences are not available from a single JSON generation pass,
  so a constant confidence is reported; calibration metrics (ECE/Brier) of this
  extractor are therefore not interpretable and must be reported as such.
"""

from __future__ import annotations

import copy
import json
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from task_offloading.semantic.baselines import (
    SemanticPrediction,
    _build_annotation,
    _label_signature,
    _signature_label,
    semantic_signature,
    visible_text,
)
from task_offloading.semantic.benchmark import ANNOTATION_FIELDS

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
CONSTANT_CONFIDENCE = 0.7

_FIELD_HELP = {
    "domain": "uygulama alanı",
    "latency": "gecikme gereksinimi (status ve class)",
    "privacy": "veri gizliliği sınıfı (status ve data_class)",
    "execution_policy": "çalıştırma izni: device/edge/cloud için allowed/forbidden/unknown",
    "reliability": "güvenilirlik katmanı (status ve tier)",
    "accuracy": "sonuç doğruluğu (status, tier, metric)",
    "energy_priority": "enerji önceliği (status ve value)",
    "divisibility": "bölünebilirlik (status ve value)",
}


def _compact(annotation: Mapping[str, Any]) -> dict[str, list[Any]]:
    return {f: list(semantic_signature(annotation, f)) for f in ANNOTATION_FIELDS}


class FewShotLlmBaseline:
    def __init__(
        self,
        *,
        cache_folder: Path,
        model_id: str = DEFAULT_MODEL_ID,
        shots: int = 4,
        max_new_tokens: int = 260,
        threads: int = 16,
    ) -> None:
        self.name = f"llm_fewshot_{model_id.split('/')[-1]}"
        self._cache = str(cache_folder)
        self._model_id = model_id
        self._shots = shots
        self._max_new_tokens = max_new_tokens
        self._threads = threads
        self._vocabulary: dict[str, list[str]] = {}
        self._examples: list[tuple[str, dict[str, list[Any]]]] = []
        self._model: Any = None
        self._tokenizer: Any = None
        self._prefix_ids: Any = None
        self._prefix_cache: Any = None
        self.invalid_field_count = 0
        self.unparsable_count = 0

    # ----------------------------------------------------------------- fit
    def fit(
        self,
        tasks: Sequence[Mapping[str, Any]],
        annotations: Sequence[Mapping[str, Any]],
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        for field in ANNOTATION_FIELDS:
            counts = Counter(
                _signature_label(semantic_signature(a, field)) for a in annotations
            )
            self._vocabulary[field] = sorted(counts)
        ranked = sorted(
            zip(tasks, annotations, strict=True),
            key=lambda pair: (
                -sum(
                    pair[1][f]["status"] not in {"not_stated"}
                    for f in ANNOTATION_FIELDS
                    if f != "domain"
                ),
                str(pair[0]["task_id"]),
            ),
        )
        seen_domains: set[str] = set()
        for task, annotation in ranked:
            domain = str(annotation["domain"].get("value"))
            if domain in seen_domains:
                continue
            seen_domains.add(domain)
            self._examples.append((visible_text(task), _compact(annotation)))
            if len(self._examples) == self._shots:
                break
        torch.set_num_threads(self._threads)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, cache_dir=self._cache
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id, cache_dir=self._cache, dtype=torch.bfloat16
        )
        self._model.eval()
        # The system prompt and few-shot turns are identical for every task;
        # encode them once and reuse the key/value cache (exact, not approximate).
        prefix = self._tokenizer.apply_chat_template(
            self._messages(""), add_generation_prompt=False, tokenize=False
        )
        marker = "Metin: "
        prefix = prefix[: prefix.rfind(marker)]
        self._prefix_ids = self._tokenizer(prefix, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            self._prefix_cache = self._model(
                self._prefix_ids, use_cache=True
            ).past_key_values

    # -------------------------------------------------------------- prompt
    def _system_prompt(self) -> str:
        lines = [
            "Sen bir gereksinim çıkarıcısısın. Türkçe IoT/MEC görev metnini okuyup "
            "8 alan için etiket üretirsin. Yalnız metinde açıkça yazanı etiketle; "
            "yazmayan alan için status 'not_stated' kullan; çelişkili ifade için "
            "'ambiguous' kullan. Sadece tek satır JSON döndür, açıklama yazma.",
            "Her alanın izin verilen değerleri (JSON listesi olarak, birebir kopyala):",
        ]
        for field in ANNOTATION_FIELDS:
            options = "; ".join(self._vocabulary[field])
            lines.append(f"- {field} ({_FIELD_HELP[field]}): {options}")
        return "\n".join(lines)

    def _messages(self, text: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self._system_prompt()}]
        for example_text, labels in self._examples:
            messages.append({"role": "user", "content": f"Metin: {example_text}"})
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(labels, ensure_ascii=False),
                }
            )
        messages.append({"role": "user", "content": f"Metin: {text}"})
        return messages

    # ------------------------------------------------------------- predict
    def _parse(self, output: str) -> dict[str, tuple[object, ...]]:
        signatures: dict[str, tuple[object, ...]] = {}
        match = re.search(r"\{.*\}", output, flags=re.DOTALL)
        parsed: Any = None
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
        if not isinstance(parsed, dict):
            self.unparsable_count += 1
            parsed = {}
        for field in ANNOTATION_FIELDS:
            value = parsed.get(field)
            label = (
                json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if isinstance(value, list)
                else None
            )
            if label is not None and label in self._vocabulary[field]:
                signatures[field] = _label_signature(label)
                continue
            self.invalid_field_count += 1
            fallback = next(
                (
                    candidate
                    for candidate in self._vocabulary[field]
                    if _label_signature(candidate)[0] == "not_stated"
                ),
                self._vocabulary[field][0],
            )
            signatures[field] = _label_signature(fallback)
        return signatures

    def predict(self, task: Mapping[str, Any]) -> SemanticPrediction:
        import torch

        if self._model is None:
            raise RuntimeError("LLM baseline must be fitted before prediction")
        started = time.perf_counter_ns()
        prompt = self._tokenizer.apply_chat_template(
            self._messages(visible_text(task)),
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        ids = inputs["input_ids"]
        prefix_length = self._prefix_ids.shape[1]
        cache = None
        if torch.equal(ids[0, :prefix_length], self._prefix_ids[0]):
            cache = copy.deepcopy(self._prefix_cache)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                past_key_values=cache,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        text = self._tokenizer.decode(
            output[0, ids.shape[1] :], skip_special_tokens=True
        )
        signatures = self._parse(text)
        confidences = dict.fromkeys(ANNOTATION_FIELDS, CONSTANT_CONFIDENCE)
        annotation = _build_annotation(task, signatures, confidences)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        return SemanticPrediction(annotation, elapsed)


__all__ = ["DEFAULT_MODEL_ID", "FewShotLlmBaseline"]
