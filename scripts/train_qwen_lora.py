# ruff: noqa: E501
"""Train a Qwen3-8B task-specific LoRA adapter on a leakage-safe train split.

The default ``auto`` loading policy prefers FP16 but selects QLoRA when the GPU
cannot hold the FP16 base plus training state.  The model remains Qwen3-8B in
both cases.  Construction labels are exploratory; final reporting requires the
separately adjudicated label artifact described by the expansion protocol.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from task_offloading.semantic import load_jsonl
from task_offloading.semantic.experiment import write_json_atomic
from task_offloading.semantic.llm import FewShotLlmBaseline
from task_offloading.semantic.llm_lora import (
    SemanticSftDataset,
    make_collator,
    resolve_load_mode,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "semantic_benchmark" / os.environ.get(
    "QWEN_DATASET", "expanded_v2"
)
CACHE = ROOT / "models" / "huggingface"
OUTPUT = ROOT / "models" / "adapters"
MODEL_ID = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-8B")
os.environ.setdefault("HF_HOME", str(CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(CACHE / "xet"))


def _training_rows(limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = {
        row["task_id"]: row
        for row in load_jsonl(DATA / "candidates.tasks.jsonl")
        if row["design_strata"]["split"] == "train"
    }
    labels = {
        row["task_id"]: row["annotation"]
        for row in load_jsonl(DATA / "candidates.construction_labels.jsonl")
        if row["split"] == "train"
    }
    ids = sorted(tasks)
    if limit:
        ids = ids[:limit]
    if set(ids) - set(labels):
        raise ValueError("training task without matching label")
    return [tasks[key] for key in ids], [labels[key] for key in ids]


def main() -> None:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-8B LoRA training requires CUDA")
    requested = os.environ.get("QWEN_LOAD_MODE", "auto")
    decision = resolve_load_mode(
        requested, torch.cuda.get_device_properties(0).total_memory  # type: ignore[arg-type]
    )
    print(
        f"model={MODEL_ID} requested={decision.requested} "
        f"selected={decision.selected}: {decision.reason}",
        flush=True,
    )
    limit = int(os.environ.get("QWEN_TRAIN_LIMIT", "0"))
    epochs = int(os.environ.get("QWEN_EPOCHS", "2"))
    max_steps = int(os.environ.get("QWEN_MAX_STEPS", "0"))
    max_length = int(os.environ.get("QWEN_MAX_LENGTH", "320"))
    gradient_accumulation = int(os.environ.get("QWEN_GRAD_ACCUM", "8"))
    learning_rate = float(os.environ.get("QWEN_LR", "2e-4"))
    seed = int(os.environ.get("QWEN_SEED", "20261127"))
    optimizer_name = os.environ.get("QWEN_OPTIMIZER", "paged_adamw_8bit")
    kbit_preparation = os.environ.get("QWEN_KBIT_PREP", "light")
    if kbit_preparation not in {"light", "peft"}:
        raise ValueError("QWEN_KBIT_PREP must be light or peft")
    torch.manual_seed(seed)
    tasks, annotations = _training_rows(limit)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    codec = FewShotLlmBaseline(
        cache_folder=CACHE,
        model_id=MODEL_ID,
        shots=0,
        label_encoding="codes",
    )
    codec.prepare_label_space(annotations)
    dataset = SemanticSftDataset(
        tasks=tasks,
        annotations=annotations,
        tokenizer=tokenizer,
        codec=codec,
        max_length=max_length,
    )
    load_options: dict[str, Any] = {
        "cache_dir": CACHE,
        "dtype": torch.float16,
        "device_map": {"": torch.cuda.current_device()},
    }
    if decision.selected == "4bit":
        load_options["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    model: Any = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_options)
    model.config.use_cache = False
    if decision.selected == "4bit" and kbit_preparation == "peft":
        # PEFT's helper upcasts every non-quantized tensor (incl. the 151k-row
        # embedding and lm_head) to FP32, which costs ~2.5 GiB on Qwen3-8B.
        model = prepare_model_for_kbit_training(  # type: ignore[no-untyped-call]
            model, use_gradient_checkpointing=True
        )
    else:
        # Light preparation: base stays FP16/NF4; only LoRA weights are FP32.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.print_trainable_parameters()
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=make_collator(int(tokenizer.pad_token_id)),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer: Any
    if optimizer_name == "paged_adamw_8bit":
        import bitsandbytes as bnb

        optimizer = bnb.optim.PagedAdamW8bit(  # type: ignore[attr-defined,no-untyped-call]
            trainable, lr=learning_rate, weight_decay=0.01
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    else:
        raise ValueError("QWEN_OPTIMIZER must be paged_adamw_8bit or adamw")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    update_step = 0
    micro_step = 0
    losses: list[float] = []
    stop = False
    for epoch in range(epochs):
        for batch in loader:
            micro_step += 1
            batch = {key: value.to("cuda") for key, value in batch.items()}
            loss = model(**batch).loss / gradient_accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at micro step {micro_step}; lower QWEN_LR"
                )
            loss.backward()
            losses.append(float(loss.detach()) * gradient_accumulation)
            if micro_step % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                print(
                    f"epoch={epoch + 1} update={update_step} "
                    f"loss={sum(losses[-gradient_accumulation:]) / gradient_accumulation:.4f} "
                    f"vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
                    flush=True,
                )
                if max_steps and update_step >= max_steps:
                    stop = True
                    break
        if stop:
            break
    adapter_name = f"Qwen3-8B_semantic_{DATA.name}_{decision.selected}"
    adapter_path = OUTPUT / adapter_name
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path, save_embedding_layers=False)
    tokenizer.save_pretrained(adapter_path)
    metadata = {
        "model_id": MODEL_ID,
        "adapter": "LoRA",
        "load_decision": {
            "requested": decision.requested,
            "selected": decision.selected,
            "reason": decision.reason,
        },
        "dataset": f"semantic_{DATA.name}_candidate",
        "label_status": "construction_labels_pending_human_adjudication",
        "training_rows": len(dataset),
        "epochs_requested": epochs,
        "optimizer_updates": update_step,
        "micro_steps": micro_step,
        "max_length": max_length,
        "gradient_accumulation": gradient_accumulation,
        "learning_rate": learning_rate,
        "seed": seed,
        "optimizer": optimizer_name,
        "kbit_preparation": kbit_preparation,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "final_recent_loss": (
            sum(losses[-gradient_accumulation:])
            / min(len(losses), gradient_accumulation)
            if losses
            else None
        ),
        "evaluation_labels_used_for_training": False,
    }
    write_json_atomic(adapter_path / "training_metadata.json", metadata)
    print(f"saved adapter={adapter_path}", flush=True)


if __name__ == "__main__":
    main()
