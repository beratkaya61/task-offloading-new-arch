# ruff: noqa: RUF001
"""Task-specific LoRA/QLoRA utilities for the bounded semantic extractor."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch.utils.data import Dataset

from task_offloading.semantic.baselines import visible_text
from task_offloading.semantic.llm import FewShotLlmBaseline

LoadMode = Literal["auto", "fp16", "4bit"]
MINIMUM_FP16_VRAM_BYTES = 18 * 2**30
LORA_SYSTEM_PROMPT = (
    "Türkçe IoT/MEC görev metninden domain, latency, privacy, execution_policy, "
    "reliability, accuracy, energy_priority ve divisibility alanlarını çıkar. "
    "Her alan için eğitimde öğrendiğin cN sınıf kodunu kullan. Yalnız tek satır "
    "JSON döndür; açıklama veya yeni alan ekleme."
)


@dataclass(frozen=True)
class LoadDecision:
    requested: LoadMode
    selected: Literal["fp16", "4bit"]
    reason: str


def resolve_load_mode(requested: LoadMode, total_vram_bytes: int) -> LoadDecision:
    """Select FP16 when feasible; otherwise make the QLoRA fallback explicit."""

    if requested == "4bit":
        return LoadDecision(requested, "4bit", "4-bit loading explicitly requested")
    if requested == "fp16":
        if total_vram_bytes < MINIMUM_FP16_VRAM_BYTES:
            raise RuntimeError(
                "FP16 Qwen3-8B LoRA requires at least 18 GiB device memory; "
                f"detected {total_vram_bytes / 2**30:.1f} GiB"
            )
        return LoadDecision(requested, "fp16", "FP16 memory gate passed")
    if requested != "auto":
        raise ValueError("load mode must be auto, fp16 or 4bit")
    if total_vram_bytes >= MINIMUM_FP16_VRAM_BYTES:
        return LoadDecision(requested, "fp16", "FP16 memory gate passed")
    return LoadDecision(
        requested,
        "4bit",
        (
            "FP16 Qwen3-8B base weights plus training activations exceed "
            f"the detected {total_vram_bytes / 2**30:.1f} GiB device; using QLoRA"
        ),
    )


class SemanticSftDataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenized chat-SFT examples with prompt tokens masked from the loss."""

    def __init__(
        self,
        *,
        tasks: list[dict[str, Any]],
        annotations: list[dict[str, Any]],
        tokenizer: Any,
        codec: FewShotLlmBaseline,
        max_length: int,
    ) -> None:
        if len(tasks) != len(annotations) or not tasks:
            raise ValueError("tasks and annotations must have the same non-zero size")
        self._rows: list[dict[str, torch.Tensor]] = []
        for task, annotation in zip(tasks, annotations, strict=True):
            user_text = visible_text(task)
            prompt_messages = [
                {"role": "system", "content": LORA_SYSTEM_PROMPT},
                {"role": "user", "content": f"Metin: {user_text}"},
            ]
            answer = json.dumps(
                codec._encoded_example(
                    {
                        field: list(value)
                        for field, value in _compact_annotation(
                            annotation, codec
                        ).items()
                    }
                ),
                ensure_ascii=False,
            )
            full_messages = [
                *prompt_messages,
                {"role": "assistant", "content": answer},
            ]
            template_options = codec._chat_template_options
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                add_generation_prompt=True,
                tokenize=False,
                **template_options,
            )
            full_text = tokenizer.apply_chat_template(
                full_messages,
                add_generation_prompt=False,
                tokenize=False,
                **template_options,
            )
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            if len(full_ids) > max_length:
                # Truncation would cut the JSON answer and teach malformed output.
                raise ValueError(
                    f"example {task.get('task_id')} needs {len(full_ids)} tokens; "
                    f"max_length={max_length} would truncate the assistant target"
                )
            prompt_length = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prompt_length + full_ids[prompt_length:]
            if all(value == -100 for value in labels):
                raise ValueError("example has no assistant target tokens")
            self._rows.append(
                {
                    "input_ids": torch.tensor(full_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._rows[index]


def _compact_annotation(
    annotation: dict[str, Any], codec: FewShotLlmBaseline
) -> dict[str, tuple[object, ...]]:
    from task_offloading.semantic.baselines import semantic_signature
    from task_offloading.semantic.benchmark import ANNOTATION_FIELDS

    del codec
    return {
        field: semantic_signature(annotation, field) for field in ANNOTATION_FIELDS
    }


def make_collator(pad_token_id: int) -> Any:
    """Return a dynamic-padding collator for causal-LM SFT."""

    def collate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        width = max(int(row["input_ids"].shape[0]) for row in rows)
        result: dict[str, list[torch.Tensor]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for row in rows:
            padding = width - int(row["input_ids"].shape[0])
            result["input_ids"].append(
                torch.nn.functional.pad(
                    row["input_ids"], (0, padding), value=pad_token_id
                )
            )
            result["attention_mask"].append(
                torch.nn.functional.pad(
                    row["attention_mask"], (0, padding), value=0
                )
            )
            result["labels"].append(
                torch.nn.functional.pad(row["labels"], (0, padding), value=-100)
            )
        return {key: torch.stack(values) for key, values in result.items()}

    return collate


__all__ = [
    "LORA_SYSTEM_PROMPT",
    "LoadDecision",
    "SemanticSftDataset",
    "make_collator",
    "resolve_load_mode",
]
