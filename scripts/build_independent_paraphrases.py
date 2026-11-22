# ruff: noqa: E501, RUF001
"""Build the independent, LLM-written paraphrase test set (pre-registered).

Source: the compositional test tasks in *template* wording (labels by
construction).  A fixed-seed sample of 200 tasks is rewritten by an LLM that
never sees the project's paraphrase pools.  Each visible text field is
rewritten separately with a zero-shot instruction to keep every requirement
and number.  Automatic filters (all original numbers kept, no new numbers,
length ratio 0.5-2.0, text changed) reject unreliable rewrites; a rejected
task is dropped, and the rejection count is reported.  The resulting set is
machine-generated; its labels are transferred under the invariance
assumption and are not human annotations.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from task_offloading.semantic import load_jsonl, write_jsonl
from task_offloading.semantic.experiment import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "data" / "semantic_benchmark" / "main" / "benchmark_v3"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SAMPLE_SEED = 20261120
SAMPLE_SIZE = 200
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
INSTRUCTION = (
    "Aşağıdaki Türkçe metni, anlamını ve içindeki bütün gereksinimleri aynen "
    "koruyarak farklı kelimelerle yeniden yaz. Bütün sayıları aynen koru, yeni "
    "bilgi veya gereksinim ekleme, hiçbir gereksinimi çıkarma. Yalnız yeniden "
    "yazılmış metni döndür.\n\nMetin: "
)


def _numbers(text: str) -> list[str]:
    return sorted(n.replace(",", ".") for n in NUMBER.findall(text))


def _accept(original: str, rewritten: str) -> bool:
    if not rewritten or rewritten.strip() == original.strip():
        return False
    ratio = len(rewritten) / max(len(original), 1)
    return 0.5 <= ratio <= 2.0 and _numbers(original) == _numbers(rewritten)


def main() -> None:
    torch.set_num_threads(16)
    tasks = load_jsonl(V3 / "eval_compositional600_canonical.tasks.jsonl")
    labels = {r["task_id"]: r["annotation"] for r in load_jsonl(V3 / "eval_compositional600_canonical.labels.jsonl")}
    rng = np.random.default_rng(SAMPLE_SEED)
    chosen = sorted(rng.choice(len(tasks), size=SAMPLE_SIZE, replace=False).tolist())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(ROOT / "models" / "huggingface"))
    model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=str(ROOT / "models" / "huggingface"),
            dtype=torch.bfloat16,
        ),
    )
    model.eval()

    def rewrite(text: str) -> str:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": INSTRUCTION + text}], add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=220, do_sample=False, temperature=None, top_p=None, top_k=None)
        return str(tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)).strip().strip('"')

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    started = time.perf_counter()
    for count, index in enumerate(chosen, 1):
        task = tasks[index]
        new = dict(task)
        ok = True
        for field in ("task_text", "user_policy_text"):
            value = task.get(field)
            if isinstance(value, str):
                rewritten = rewrite(value)
                if not _accept(value, rewritten):
                    ok = False
                    rejected.append({"task_id": task["task_id"], "field": field, "original": value, "rewritten": rewritten})
                    break
                new[field] = rewritten
        if ok:
            new["task_id"] = f"indep_{task['task_id']}"
            new["provenance"] = {
                **task["provenance"],
                "wording": "independent_llm_paraphrase",
                "paraphrase_model": MODEL_ID,
                "source_task_id": task["task_id"],
                "label_claim": "label_by_construction_transferred_not_human",
            }
            kept.append(new)
        if count % 10 == 0:
            print(f"{count}/{SAMPLE_SIZE} kept={len(kept)} ({time.perf_counter() - started:.0f}s)", flush=True)

    write_jsonl(V3 / "eval_independent_paraphrase.tasks.jsonl", kept)
    write_jsonl(
        V3 / "eval_independent_paraphrase.labels.jsonl",
        [{"task_id": t["task_id"], "annotation": {**labels[t["provenance"]["source_task_id"]], "task_id": t["task_id"]}} for t in kept],
    )
    write_json_atomic(
        V3 / "independent_paraphrase_log.json",
        {
            "model_id": MODEL_ID,
            "sample_seed": SAMPLE_SEED,
            "sampled": SAMPLE_SIZE,
            "kept": len(kept),
            "rejected": len(rejected),
            "filters": ["numbers preserved exactly", "length ratio 0.5-2.0", "text changed"],
            "rejected_examples": rejected[:30],
        },
    )
    print(json.dumps({"kept": len(kept), "rejected": len(rejected)}))


if __name__ == "__main__":
    main()
