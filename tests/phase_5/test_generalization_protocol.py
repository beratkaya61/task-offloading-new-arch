"""Frozen Phase 7 generalization decisions must stay executable and honest."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_generalization_protocol_freezes_holdouts_before_results() -> None:
    protocol = json.loads(
        (ROOT / "configs" / "generalization_protocol.v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["protocol_version"] == "1.0.0"
    assert protocol["experiment_seed"] == 20261120
    assert protocol["temporal_holdout"]["train_rule"] == "calendar day < 24"
    assert protocol["temporal_holdout"]["test_rule"] == "calendar day >= 24"
    assert protocol["entity_holdout"]["requires_policy_retraining"] is True
    assert protocol["domain_holdout"]["heldout_domains"] == [
        "public_safety",
        "transportation",
    ]
    assert (
        "not a real deployed application-id"
        in protocol["application_proxy_holdout"]["claim_limit"]
    )
    assert protocol["physical_source_ablations"]["policy"].startswith(
        "load the same v2_nep checkpoints"
    )
