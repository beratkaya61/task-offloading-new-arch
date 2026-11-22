from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from task_offloading.experiment_config import (
    EXPERIMENT_SEED,
    PPO_SEEDS,
    PPO_TIMESTEPS,
    TEST_EPISODES,
    TRAIN_EPISODES,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "publication_suite.v1.json"
EVIDENCE = ROOT / "configs" / "publication_evidence.v1.json"


def _suite() -> dict[str, Any]:
    value: Any = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def test_publication_contract_matches_executed_constants() -> None:
    contract = _suite()["experiment_contract"]
    assert contract == {
        "experiment_seed": EXPERIMENT_SEED,
        "ppo_seeds": PPO_SEEDS,
        "ppo_timesteps": PPO_TIMESTEPS,
        "train_episodes": TRAIN_EPISODES,
        "test_episodes": TEST_EPISODES,
        "tasks_per_episode": 64,
        "bootstrap_resamples": 2_000,
    }


def test_publication_stages_are_unique_safe_and_explicit() -> None:
    stages = _suite()["stages"]
    ids = [stage["id"] for stage in stages]
    assert len(ids) == len(set(ids))
    assert {
        "semantic_score_v3_2",
        "offloading_v2_nep_exact",
        "offloading_v2_nep_noisy",
        "offloading_v2_nep_independent",
        "offloading_entity_holdout",
        "offloading_domain_holdout",
        "source_without_nep",
        "source_without_uci",
        "source_without_bupt",
        "source_without_eua",
        "publication_reports",
        "advisor_document",
        "quadro_qwen3_scale_extension",
    } == set(ids)

    for stage in stages:
        if stage["kind"] == "manual_optional":
            assert stage["scope"] == "quadro_optional"
            assert not stage["commands"]
            continue
        assert stage["scope"] == "current_pc"
        assert stage["commands"]
        for command in stage["commands"]:
            assert command[0] == "{python}"
        for output in stage["outputs"]:
            relative = Path(output["path"])
            assert not relative.is_absolute()
            assert ".." not in relative.parts


def test_training_stages_are_resume_safe() -> None:
    for stage in _suite()["stages"]:
        if stage["kind"] != "training":
            continue
        runner_commands = [
            command
            for command in stage["commands"]
            if "scripts/run_offloading_experiment.py" in command
        ]
        assert runner_commands
        assert all("--resume" in command for command in runner_commands)


def test_publication_evidence_paths_and_hashes_are_well_formed() -> None:
    value: Any = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    entries = value["files"]
    paths = [entry["path"] for entry in entries]
    assert len(paths) == len(set(paths))
    for entry in entries:
        relative = Path(entry["path"])
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        digest = entry["sha256"]
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)
