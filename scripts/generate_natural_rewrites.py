# ruff: noqa: E501
"""Generate one constraint-preserving natural rewrite per expansion-v2 profile (GPU).

To avoid self-preference bias, the model that writes a split's natural view
must not be the model evaluated on it:

* train + validation: ``Qwen/Qwen3-8B``
* test: ``ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1`` (Llama-3 family)

Each profile's canonical text is rewritten with sampling; a rewrite is kept
only if ``rewrite_issues`` accepts it (up to ``NAT_ATTEMPTS`` tries with new
seeds).  Output is appended per batch, so an interrupted run resumes.

Environment: ``NAT_SPLITS`` (comma list, default ``train,validation``),
``NAT_MODEL``, ``NAT_REVISION``, ``NAT_BATCH`` (default 8), ``NAT_ATTEMPTS``
(default 3), ``NAT_LIMIT`` (smoke: first N profiles).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from task_offloading.semantic import load_jsonl
from task_offloading.semantic.baselines import visible_text
from task_offloading.semantic.llm import _chat_template_options
from task_offloading.semantic.natural_rewrite import (
    clean_output,
    rewrite_issues,
    rewrite_messages,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "semantic_benchmark" / "expanded_v2"
CACHE = ROOT / "models" / "huggingface"
OUT_DIR = DATA / "natural_rewrites"
DEFAULT_MODELS = {
    "train": ("Qwen/Qwen3-8B", None),
    "validation": ("Qwen/Qwen3-8B", None),
    "test": ("ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1", "1417b3d1e453cd91772919568f0bba5dbff9473b"),
}


def _cache_for(model_id: str) -> Path:
    """Models live either directly in CACHE or in CACHE/hub (snapshot_download)."""

    folder = "models--" + model_id.replace("/", "--")
    return CACHE if (CACHE / folder).exists() else CACHE / "hub"


def _load(model_id: str, revision: str | None) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cache = _cache_for(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache,
        revision=revision,
        dtype=torch.float16,
        device_map={"": torch.cuda.current_device()},
        quantization_config=BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
    )
    model.eval()
    return model, tokenizer


def _generate(model: Any, tokenizer: Any, model_id: str, sources: list[str], seed: int) -> list[str]:
    prompts = [
        tokenizer.apply_chat_template(
            rewrite_messages(text), add_generation_prompt=True, tokenize=False, **_chat_template_options(model_id)
        )
        for text in sources
    ]
    batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    # Llama-3 chat models end a turn with <|eot_id|>, not the base eos token;
    # without it they keep generating "assistant ..." continuations.
    eos_ids = [int(tokenizer.eos_token_id)]
    for marker in ("<|eot_id|>", "<|im_end|>"):
        token_id = tokenizer.convert_tokens_to_ids(marker)
        if isinstance(token_id, int) and token_id != tokenizer.unk_token_id and token_id not in eos_ids:
            eos_ids.append(token_id)
    torch.manual_seed(seed)
    with torch.inference_mode():
        output = model.generate(
            **batch,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=220,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )
    new_tokens = output[:, batch["input_ids"].shape[1] :]
    return [tokenizer.decode(row, skip_special_tokens=True) for row in new_tokens]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("natural rewrite generation requires the CUDA machine")
    splits = os.environ.get("NAT_SPLITS", "train,validation").split(",")
    model_id, revision = DEFAULT_MODELS[splits[0]]
    model_id = os.environ.get("NAT_MODEL", model_id)
    revision = os.environ.get("NAT_REVISION") or revision
    if any(DEFAULT_MODELS[s][0] != model_id for s in splits) and "NAT_MODEL" not in os.environ:
        raise ValueError("splits with different default generators must run separately")
    batch_size = int(os.environ.get("NAT_BATCH", "8"))
    attempts = int(os.environ.get("NAT_ATTEMPTS", "3"))
    limit = int(os.environ.get("NAT_LIMIT", "0"))

    labels = {r["task_id"]: r["annotation"] for r in load_jsonl(DATA / "candidates.construction_labels.jsonl")}
    canonical = [
        t for t in load_jsonl(DATA / "candidates.tasks.jsonl")
        if t["design_strata"]["wording_view"] == "canonical" and t["design_strata"]["split"] in splits
    ]
    canonical.sort(key=lambda t: t["task_id"])
    if limit:
        canonical = canonical[:limit]
    tag = model_id.split("/")[-1]
    out_path = OUT_DIR / f"{tag}.{'_'.join(splits)}{'.smoke' if limit else ''}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["source_task_id"] for line in out_path.open(encoding="utf-8")} if out_path.exists() else set()
    todo = [t for t in canonical if t["task_id"] not in done]
    print(f"model={model_id} splits={splits} profiles={len(canonical)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return
    model, tokenizer = _load(model_id, revision)
    started = time.perf_counter()
    accepted = rejected = 0
    for start in range(0, len(todo), batch_size):
        chunk = todo[start : start + batch_size]
        results: dict[str, dict[str, Any]] = {}
        pending = list(chunk)
        for attempt in range(1, attempts + 1):
            if not pending:
                break
            sources = [visible_text(t).replace("\n", " ") for t in pending]
            raws = _generate(model, tokenizer, model_id, sources, seed=20261127 + 1000 * attempt + start)
            still: list[dict[str, Any]] = []
            for task, source, raw in zip(pending, sources, raws, strict=True):
                text = clean_output(raw)
                issues = rewrite_issues(source, text, labels[task["task_id"]])
                if not issues:
                    results[task["task_id"]] = {"text": text, "attempt": attempt, "issues": []}
                else:
                    results[task["task_id"]] = {"text": text, "attempt": attempt, "issues": issues}
                    still.append(task)
            pending = still
        with out_path.open("a", encoding="utf-8") as handle:
            for task in chunk:
                result = results[task["task_id"]]
                ok = not result["issues"]
                accepted += ok
                rejected += not ok
                handle.write(
                    json.dumps(
                        {
                            "source_task_id": task["task_id"],
                            "semantic_profile_id": task["grouping"]["semantic_profile_id"],
                            "split": task["design_strata"]["split"],
                            "source_text": visible_text(task).replace("\n", " "),
                            "rewrite": result["text"],
                            "accepted": ok,
                            "issues": result["issues"],
                            "attempts_used": result["attempt"],
                            "generator_model": model_id,
                            "generator_revision": revision,
                            "decoding": {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "max_new_tokens": 220},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        elapsed = time.perf_counter() - started
        processed = start + len(chunk)
        print(
            f"{processed}/{len(todo)} accepted={accepted} rejected={rejected} "
            f"({elapsed:.0f}s, eta {elapsed / processed * (len(todo) - processed) / 60:.0f} min)",
            flush=True,
        )


if __name__ == "__main__":
    main()
