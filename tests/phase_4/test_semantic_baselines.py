"""Leakage, schema, metric, and artifact tests for semantic baselines."""

# Ruff's homoglyph check is intentionally disabled for Turkish fixture text.
# ruff: noqa: RUF001

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from task_offloading.semantic import ANNOTATION_FIELDS, load_jsonl
from task_offloading.semantic.baselines import (
    ConstantProfileBaseline,
    RuleBasedBaseline,
    SemanticPrediction,
    TfidfLogisticBaseline,
    visible_text,
)
from task_offloading.semantic.evaluation import evaluate_semantic_predictions

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "data" / "semantic_benchmark" / "main"
OUTPUT = MAIN / "baselines"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value: Any = json.load(handle)
    assert isinstance(value, dict)
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task(task_id: str, text: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "language": "tr",
        "task_text": text,
        "user_policy_text": None,
    }


def annotation(task_id: str, domain: str) -> dict[str, Any]:
    base = {"status": "not_stated", "evidence": [], "confidence": 0.9}
    return {
        "schema_version": "1.0.0",
        "task_id": task_id,
        "language": "tr",
        "domain": {
            "value": domain,
            "status": "explicit",
            "evidence": [domain],
            "confidence": 0.9,
        },
        "latency": {"class": None, "max_ms": None, **base},
        "privacy": {"data_class": None, **base},
        "execution_policy": {
            "device": "unknown",
            "edge": "unknown",
            "cloud": "unknown",
            **base,
        },
        "reliability": {
            "tier": None,
            "min_success_probability": None,
            **base,
        },
        "accuracy": {
            "tier": None,
            "min_score": None,
            "metric": None,
            **base,
        },
        "energy_priority": {"value": None, **base},
        "divisibility": {"value": None, **base},
        "abstained_fields": list(ANNOTATION_FIELDS[1:]),
        "overall_confidence": 0.9,
    }


def test_visible_text_excludes_all_authoring_metadata() -> None:
    row = task("task_001", "Görev metni")
    row["user_policy_text"] = "Kullanıcı politikası"
    row["design_strata"] = {"scenario_domain": "industrial"}
    row["provenance"] = {"origin": "adversarial"}
    row["grouping"] = {"scenario_family_id": "secret_family"}

    assert visible_text(row) == "Görev metni\nKullanıcı politikası"
    assert "industrial" not in visible_text(row)
    assert "secret_family" not in visible_text(row)


def test_constant_profile_is_deterministic_and_schema_valid() -> None:
    tasks = (
        task("task_001", "Ortak görev bir"),
        task("task_002", "Ortak görev iki"),
        task("task_003", "Ortak görev üç"),
    )
    annotations = (
        annotation("task_001", "healthcare"),
        annotation("task_002", "healthcare"),
        annotation("task_003", "industrial"),
    )
    model = ConstantProfileBaseline()
    model.fit(tasks, annotations)

    first = model.predict(task("task_004", "Yeni ortak görev")).annotation
    second = model.predict(task("task_004", "Yeni ortak görev")).annotation
    schema = load_json(ROOT / "schemas" / "semantic_requirements.schema.json")

    assert first == second
    assert first["domain"]["value"] == "healthcare"
    assert not tuple(Draft202012Validator(schema).iter_errors(first))


def test_rule_parser_extracts_only_visible_requirements() -> None:
    text = (
        "Hastanedeki görev en geç 25 ms içinde tamamlanmalıdır. "
        "Bu kısıtlı veridir. Sonuçların en az %99,95'i başarıyla "
        "tamamlanması gerekir. F1 skoru en az 0,97 olmalıdır. "
        "İşlem cihaz veya yakın edge üzerinde yapılabilir; genel bulut "
        "kullanılamaz. Pil birinci önceliktir. Parçaları bağımsız ve "
        "paralel işlenebilir."
    )
    row = task("task_101", text)
    row["design_strata"] = {"scenario_domain": "consumer"}
    prediction = RuleBasedBaseline().predict(row).annotation
    schema = load_json(ROOT / "schemas" / "semantic_requirements.schema.json")

    assert prediction["domain"]["value"] == "healthcare"
    assert prediction["latency"] == {
        "class": "hard_real_time",
        "max_ms": 25.0,
        "status": "explicit",
        "evidence": [text],
        "confidence": 0.8,
    }
    assert prediction["privacy"]["data_class"] == "restricted"
    assert prediction["execution_policy"]["device"] == "allowed"
    assert prediction["execution_policy"]["edge"] == "allowed"
    assert prediction["execution_policy"]["cloud"] == "forbidden"
    assert prediction["reliability"]["tier"] == "mission_critical"
    assert prediction["reliability"]["min_success_probability"] == 0.9995
    assert prediction["accuracy"]["metric"] == "F1"
    assert prediction["accuracy"]["min_score"] == 0.97
    assert prediction["energy_priority"]["value"] == "high"
    assert prediction["divisibility"]["value"] == "divisible"
    assert not tuple(Draft202012Validator(schema).iter_errors(prediction))


def test_accuracy_percentage_does_not_create_false_reliability() -> None:
    row = task(
        "task_102",
        "Artırılmış gerçeklik sonucu en az %95 doğruluk sağlamalıdır.",
    )
    prediction = RuleBasedBaseline().predict(row).annotation

    assert prediction["accuracy"]["status"] == "explicit"
    assert prediction["accuracy"]["min_score"] == 0.95
    assert prediction["accuracy"]["metric"] is None
    assert prediction["reliability"]["status"] == "not_stated"


def test_tfidf_logistic_model_is_deterministic_and_schema_valid() -> None:
    tasks = tuple(
        task(f"task_{index:03d}", f"Ortak görev metni {domain} {index}")
        for index, domain in enumerate(
            ("sağlık", "sağlık", "fabrika", "fabrika"), start=1
        )
    )
    annotations = tuple(
        annotation(
            str(row["task_id"]),
            "healthcare" if index < 2 else "industrial",
        )
        for index, row in enumerate(tasks)
    )
    model = TfidfLogisticBaseline(random_seed=20260929)
    model.fit(tasks, annotations)

    evaluation_task = task("task_999", "Ortak görev metni sağlık yeni")
    first = model.predict(evaluation_task).annotation
    second = model.predict(evaluation_task).annotation
    schema = load_json(ROOT / "schemas" / "semantic_requirements.schema.json")

    assert first == second
    assert not tuple(Draft202012Validator(schema).iter_errors(first))


def test_evaluation_metrics_have_a_perfect_prediction_golden_case() -> None:
    row = task(
        "task_201",
        "Hastane verisi 25 ms içinde işlenmelidir. Bu kısıtlı veridir.",
    )
    predicted = RuleBasedBaseline().predict(row).annotation
    predicted["privacy"] = {
        "data_class": "restricted",
        "status": "explicit",
        "evidence": ["Bu kısıtlı veridir."],
        "confidence": 1.0,
    }
    predicted["latency"]["confidence"] = 1.0
    for field in ANNOTATION_FIELDS:
        predicted[field]["confidence"] = 1.0
    predicted["overall_confidence"] = 1.0
    schema = load_json(ROOT / "schemas" / "semantic_requirements.schema.json")

    result = evaluate_semantic_predictions(
        model_name="golden_case",
        tasks=(row,),
        references=(copy.deepcopy(predicted),),
        predictions=(SemanticPrediction(predicted, 2.5),),
        semantic_schema=schema,
    )

    assert result["schema_validity"] == 1.0
    assert result["exact_record_accuracy"] == 1.0
    assert result["exact_field_accuracy"] == 1.0
    assert result["micro_f1"] == 1.0
    assert result["macro_f1_across_fields"] == 1.0
    assert result["calibration"]["brier"] == 0.0
    assert result["critical_privacy"]["recall"] == 1.0
    assert result["critical_latency"]["recall"] == 1.0
    assert result["evidence_grounding_rate"] == 1.0
    assert result["inference_latency_ms"]["mean"] == 2.5


def test_tracked_baseline_artifacts_are_bound_and_schema_valid() -> None:
    config = load_json(ROOT / "configs" / "semantic_baselines.v1.json")
    split = load_json(MAIN / "split_manifest.v1.json")
    results = load_json(OUTPUT / "results.v1.json")
    manifest = load_json(OUTPUT / "manifest.v1.json")
    schema = load_json(ROOT / "schemas" / "semantic_requirements.schema.json")
    validator = Draft202012Validator(schema)
    evaluation_ids = split["single_human_evaluation"]["task_ids"]

    assert config["evaluation_labels_may_be_used_for_training_or_tuning"] is False
    assert results["evaluation_labels_used_for_training_or_tuning"] is False
    assert results["evaluation_label_claim"] == (
        "single_human_reference_not_adjudicated_gold"
    )
    assert {row["model_name"] for row in results["models"]} == {
        "constant_profile",
        "rule_keyword",
        "tfidf_logistic_regression",
    }
    assert all(row["schema_validity"] == 1.0 for row in results["models"])

    for section in manifest["inputs"].values():
        assert file_sha256(ROOT / section["path"]) == section["sha256"]
    for section in manifest["implementation"].values():
        assert file_sha256(ROOT / section["path"]) == section["sha256"]
    results_info = manifest["outputs"]["results"]
    assert file_sha256(ROOT / results_info["path"]) == results_info["sha256"]

    for prediction_info in manifest["outputs"]["predictions"].values():
        path = ROOT / prediction_info["path"]
        rows = load_jsonl(path)
        assert file_sha256(path) == prediction_info["sha256"]
        assert len(rows) == prediction_info["record_count"] == 60
        assert [row["task_id"] for row in rows] == evaluation_ids
        assert all(
            not tuple(validator.iter_errors(row["annotation"])) for row in rows
        )
