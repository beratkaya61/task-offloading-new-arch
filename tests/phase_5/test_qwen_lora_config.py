from __future__ import annotations

import pytest

from task_offloading.semantic.llm_lora import resolve_load_mode


def test_auto_load_mode_falls_back_to_qlora_on_eight_gib() -> None:
    decision = resolve_load_mode("auto", 8 * 2**30)
    assert decision.selected == "4bit"
    assert "QLoRA" in decision.reason


def test_auto_load_mode_prefers_fp16_when_memory_is_sufficient() -> None:
    decision = resolve_load_mode("auto", 24 * 2**30)
    assert decision.selected == "fp16"


def test_forced_fp16_rejects_insufficient_device_memory() -> None:
    with pytest.raises(RuntimeError, match="18 GiB"):
        resolve_load_mode("fp16", 8 * 2**30)
