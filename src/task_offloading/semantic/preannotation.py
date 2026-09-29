# ruff: noqa: RUF001
"""Non-human seed labels for the assisted-development annotation form."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from task_offloading.semantic.benchmark import ANNOTATION_FIELDS

_LATENCY_SENTENCE = re.compile(
    r"Sonuç en geç (?P<value>\d+(?:[.,]\d+)?) ms içinde hazır olmalıdır\."
)


def _visible_text(task: Mapping[str, Any]) -> str:
    policy = task.get("user_policy_text")
    return f"{task['task_text']}\n{policy if isinstance(policy, str) else ''}"


def _evidence(task: Mapping[str, Any], phrase: str) -> str:
    if phrase not in _visible_text(task):
        raise ValueError(f"preannotation evidence is not visible: {phrase}")
    return phrase


def _unknown() -> dict[str, Any]:
    return {"status": "not_stated", "evidence": [], "confidence": 0.9}


def _domain_evidence(task: Mapping[str, Any]) -> str:
    text = str(task["task_text"])
    if " tarafından işlenir." in text:
        prefix = text.split(" tarafından işlenir.", maxsplit=1)[0]
        return prefix.rsplit(", ", maxsplit=1)[-1]
    return text.split(",", maxsplit=1)[0]


def _latency_class(max_ms: float) -> str:
    if max_ms < 50:
        return "hard_real_time"
    if max_ms <= 200:
        return "real_time"
    if max_ms <= 1000:
        return "interactive"
    return "relaxed"


def authoring_preannotation(task: Mapping[str, Any]) -> dict[str, Any]:
    """Suggest labels from authoring rules; never return or imply human gold."""

    grouping = task.get("grouping")
    strata = task.get("design_strata")
    if not isinstance(grouping, Mapping) or not isinstance(strata, Mapping):
        raise ValueError("task lacks authoring metadata")
    template_id = str(grouping.get("template_family_id", ""))
    match = re.fullmatch(r"main_tf_(\d{3})", template_id)
    if match is None:
        raise ValueError(f"unsupported template family: {template_id}")
    pattern = int(match.group(1)) - 1
    if pattern not in range(15):
        raise ValueError(f"unsupported template pattern: {pattern}")

    annotation: dict[str, Any] = {
        "schema_version": "1.0.0",
        "task_id": str(task["task_id"]),
        "language": str(task["language"]),
        "domain": {
            "value": str(strata["scenario_domain"]),
            "status": "explicit",
            "evidence": [_domain_evidence(task)],
            "confidence": 0.9,
        },
        "latency": {"class": None, "max_ms": None, **_unknown()},
        "privacy": {"data_class": None, **_unknown()},
        "execution_policy": {
            "device": "unknown",
            "edge": "unknown",
            "cloud": "unknown",
            **_unknown(),
        },
        "reliability": {
            "tier": None,
            "min_success_probability": None,
            **_unknown(),
        },
        "accuracy": {
            "tier": None,
            "min_score": None,
            "metric": None,
            **_unknown(),
        },
        "energy_priority": {"value": None, **_unknown()},
        "divisibility": {"value": None, **_unknown()},
    }

    if pattern == 0:
        sentence_match = _LATENCY_SENTENCE.search(_visible_text(task))
        if sentence_match is None:
            raise ValueError("numeric latency template has no visible threshold")
        value = float(sentence_match.group("value").replace(",", "."))
        annotation["latency"] = {
            "class": _latency_class(value),
            "max_ms": value,
            "status": "explicit",
            "evidence": [_evidence(task, sentence_match.group(0))],
            "confidence": 0.9,
        }
    elif pattern == 1:
        privacy = "İşlenen kayıt hassas veridir."
        execution = "Cihaz veya yakın edge kullanılabilir; genel bulut kullanılamaz."
        annotation["privacy"] = {
            "data_class": "sensitive",
            "status": "explicit",
            "evidence": [_evidence(task, privacy)],
            "confidence": 0.9,
        }
        annotation["execution_policy"] = {
            "device": "allowed",
            "edge": "allowed",
            "cloud": "forbidden",
            "status": "explicit",
            "evidence": [_evidence(task, execution)],
            "confidence": 0.9,
        }
    elif pattern == 2:
        reliability = "İsteklerin en az %99,95'i başarıyla tamamlanmalıdır."
        accuracy = "Sonucun F1 skoru en az 0,97 olmalıdır."
        annotation["reliability"] = {
            "tier": "mission_critical",
            "min_success_probability": 0.9995,
            "status": "explicit",
            "evidence": [_evidence(task, reliability)],
            "confidence": 0.9,
        }
        annotation["accuracy"] = {
            "tier": "high",
            "min_score": 0.97,
            "metric": "F1",
            "status": "explicit",
            "evidence": [_evidence(task, accuracy)],
            "confidence": 0.9,
        }
    elif pattern == 3:
        energy = "Pil tüketimini azaltmak birinci önceliktir."
        divisibility = "Veri parçaları bağımsız ve paralel işlenebilir."
        annotation["energy_priority"] = {
            "value": "high",
            "status": "explicit",
            "evidence": [_evidence(task, energy)],
            "confidence": 0.9,
        }
        annotation["divisibility"] = {
            "value": "divisible",
            "status": "explicit",
            "evidence": [_evidence(task, divisibility)],
            "confidence": 0.9,
        }
    elif pattern == 5:
        immediate = "Yanıt anlık verilmelidir."
        relaxed = "Dört saniyelik gecikme tamamen kabul edilebilir."
        annotation["latency"] = {
            "class": None,
            "max_ms": None,
            "status": "ambiguous",
            "evidence": [_evidence(task, immediate), _evidence(task, relaxed)],
            "confidence": 0.9,
        }
    elif pattern == 6:
        required = "Analiz yalnız bulutta yapılmalıdır."
        forbidden = "Aynı kaydın buluta gönderilmesi kesinlikle yasaktır."
        annotation["execution_policy"] = {
            "device": "unknown",
            "edge": "unknown",
            "cloud": "unknown",
            "status": "ambiguous",
            "evidence": [_evidence(task, required), _evidence(task, forbidden)],
            "confidence": 0.9,
        }
    elif pattern == 7:
        high = "Sonuç yüksek doğrulukta olmalıdır."
        low = "Yaklaşık ve düşük doğruluklu bir sonuç da yeterlidir."
        annotation["accuracy"] = {
            "tier": None,
            "min_score": None,
            "metric": None,
            "status": "ambiguous",
            "evidence": [_evidence(task, high), _evidence(task, low)],
            "confidence": 0.9,
        }
    elif pattern == 8:
        privacy = "Kullanılan veri kamuya açıktır."
        execution = "Görev cihaz, yakın edge veya bulutta çalıştırılabilir."
        annotation["privacy"] = {
            "data_class": "public",
            "status": "explicit",
            "evidence": [_evidence(task, privacy)],
            "confidence": 0.9,
        }
        annotation["execution_policy"] = {
            "device": "allowed",
            "edge": "allowed",
            "cloud": "allowed",
            "status": "explicit",
            "evidence": [_evidence(task, execution)],
            "confidence": 0.9,
        }
    elif pattern == 9:
        phrase = "Bağlantı kesintilerine karşı yüksek güvenilirlik sağlanmalıdır."
        annotation["reliability"] = {
            "tier": "high",
            "min_success_probability": None,
            "status": "explicit",
            "evidence": [_evidence(task, phrase)],
            "confidence": 0.9,
        }
    elif pattern == 10:
        phrase = "İşlenen ham kayıt kısıtlı veridir."
        annotation["privacy"] = {
            "data_class": "restricted",
            "status": "explicit",
            "evidence": [_evidence(task, phrase)],
            "confidence": 0.9,
        }
    elif pattern == 11:
        phrase = (
            "Yakın edge sunucusunda işlem yapılabilir ve genel bulut kullanılamaz."
        )
        annotation["execution_policy"] = {
            "device": "unknown",
            "edge": "allowed",
            "cloud": "forbidden",
            "status": "partial",
            "evidence": [_evidence(task, phrase)],
            "confidence": 0.9,
        }
    elif pattern == 12:
        latency = "Sonuç 500 ms içinde hazırlanmalıdır."
        energy = "Enerji tüketimi ile yanıt süresi dengelenmelidir."
        annotation["latency"] = {
            "class": "interactive",
            "max_ms": 500,
            "status": "explicit",
            "evidence": [_evidence(task, latency)],
            "confidence": 0.9,
        }
        annotation["energy_priority"] = {
            "value": "balanced",
            "status": "explicit",
            "evidence": [_evidence(task, energy)],
            "confidence": 0.9,
        }
    elif pattern == 13:
        phrase = "Sonucun F1 skoru en az 0,95 olmalıdır."
        annotation["accuracy"] = {
            "tier": "high",
            "min_score": 0.95,
            "metric": "F1",
            "status": "explicit",
            "evidence": [_evidence(task, phrase)],
            "confidence": 0.9,
        }
    elif pattern == 14:
        phrase = "Görev parçaları birbirinden bağımsız işlenebilir."
        annotation["divisibility"] = {
            "value": "divisible",
            "status": "explicit",
            "evidence": [_evidence(task, phrase)],
            "confidence": 0.9,
        }

    annotation["abstained_fields"] = [
        field
        for field in ANNOTATION_FIELDS
        if annotation[field]["status"] in {"not_stated", "ambiguous"}
    ]
    annotation["overall_confidence"] = 0.9
    return annotation


def annotation_form_seed(annotation: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a semantic annotation into editable HTML-form field values."""

    result: dict[str, Any] = {}
    for field_name in ANNOTATION_FIELDS:
        raw = annotation.get(field_name)
        if not isinstance(raw, Mapping):
            raise ValueError(f"annotation lacks field: {field_name}")
        field = dict(raw)
        evidence = field.get("evidence")
        if not isinstance(evidence, list):
            raise ValueError(f"annotation evidence must be a list: {field_name}")
        field["evidence"] = "\n".join(str(value) for value in evidence)
        confidence = field.get("confidence")
        if not isinstance(confidence, int | float):
            raise ValueError(f"annotation confidence must be numeric: {field_name}")
        field["confidence"] = f"{float(confidence):.2f}"
        for numeric_key in ("max_ms", "min_success_probability", "min_score"):
            if numeric_key in field:
                value = field[numeric_key]
                field[numeric_key] = "" if value is None else str(value)
        for nullable_key in ("value", "class", "data_class", "tier", "metric"):
            if nullable_key in field and field[nullable_key] is None:
                field[nullable_key] = ""
        result[field_name] = field
    return result
