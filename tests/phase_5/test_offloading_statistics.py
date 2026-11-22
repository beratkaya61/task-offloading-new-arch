from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _summary_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "summarize_offloading_results.py"
    )
    spec = importlib.util.spec_from_file_location("offloading_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase7_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "summarize_phase7_generalization.py"
    )
    spec = importlib.util.spec_from_file_location("phase7_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_holm_adjustment_matches_step_down_definition() -> None:
    adjusted = _summary_module().holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_holm_adjustment_is_order_invariant() -> None:
    adjusted = _summary_module().holm_adjust([0.4, 0.001, 0.02, 0.8])
    assert adjusted == pytest.approx([0.8, 0.004, 0.06, 0.8])


def test_phase7_paired_ci_averages_training_seeds_before_bootstrap() -> None:
    module = _phase7_module()
    left = [
        {"per_episode": {"success_rate": [0.8, 0.9]}},
        {"per_episode": {"success_rate": [0.6, 0.7]}},
    ]
    right = [
        {"per_episode": {"success_rate": [0.7, 0.8]}},
        {"per_episode": {"success_rate": [0.5, 0.6]}},
    ]
    mean, low, high = module._paired_ci(left, right, "success_rate")
    assert (mean, low, high) == pytest.approx((0.1, 0.1, 0.1))
