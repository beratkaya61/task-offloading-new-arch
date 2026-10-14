"""Deterministic semantic baselines that consume only visible task text."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from task_offloading.semantic.benchmark import ANNOTATION_FIELDS

_SIGNATURE_KEYS: dict[str, tuple[str, ...]] = {
    "domain": ("status", "value"),
    "latency": ("status", "class"),
    "privacy": ("status", "data_class"),
    "execution_policy": ("status", "device", "edge", "cloud"),
    "reliability": ("status", "tier"),
    "accuracy": ("status", "tier", "metric"),
    "energy_priority": ("status", "value"),
    "divisibility": ("status", "value"),
}
_LATENCY_NUMBER = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(ms|milisaniye|saniye)", re.IGNORECASE
)
_PERCENT_PREFIX = re.compile(r"%\s*(\d+(?:[.,]\d+)?)")
_PERCENT_SUFFIX = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
_F1_SCORE = re.compile(
    r"F1\s+skoru\s+en\s+az\s+(0(?:[.,]\d+)|1(?:[.,]0+)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SemanticPrediction:
    """One schema-shaped prediction and its measured inference duration."""

    annotation: dict[str, Any]
    inference_ms: float


def visible_text(task: Mapping[str, Any]) -> str:
    """Return exactly the text that a semantic extractor is allowed to see."""

    task_text = str(task["task_text"])
    policy = task.get("user_policy_text")
    return f"{task_text}\n{policy}" if isinstance(policy, str) else task_text


def semantic_signature(
    annotation: Mapping[str, Any], field: str
) -> tuple[object, ...]:
    """Project a field onto the categorical values used by model metrics."""

    keys = _SIGNATURE_KEYS[field]
    raw = annotation.get(field)
    if not isinstance(raw, Mapping):
        raise ValueError(f"annotation lacks field: {field}")
    return tuple(raw.get(key) for key in keys)


def _signature_label(signature: Sequence[object]) -> str:
    return json.dumps(
        list(signature), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def _label_signature(label: str) -> tuple[object, ...]:
    value: Any = json.loads(label)
    if not isinstance(value, list):
        raise ValueError("signature label must decode to a list")
    return tuple(value)


def _normalize(text: str) -> str:
    turkish_casefolded = text.casefold().replace("\u0131", "i")
    decomposed = unicodedata.normalize("NFKD", turkish_casefolded)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _first_number(match: re.Match[str] | None) -> float | None:
    if match is None:
        return None
    return float(match.group(1).replace(",", "."))


def _latency_ms(text: str) -> float | None:
    match = _LATENCY_NUMBER.search(text)
    value = _first_number(match)
    if match is None or value is None:
        return None
    return value * 1000 if match.group(2).casefold() == "saniye" else value


def _percentage(text: str) -> float | None:
    for pattern in (_PERCENT_PREFIX, _PERCENT_SUFFIX):
        value = _first_number(pattern.search(text))
        if value is not None:
            return value / 100
    return None


def _reliability_probability(text: str) -> float | None:
    normalized = _normalize(text)
    reliability_terms = ("basari", "guvenilir", "kaybol")
    if not any(term in normalized for term in reliability_terms):
        return None
    return _percentage(text)


def _accuracy_score(text: str) -> float | None:
    f1_value = _first_number(_F1_SCORE.search(text))
    if f1_value is not None:
        return f1_value
    if "dogruluk" in _normalize(text):
        return _percentage(text)
    return None


def _accuracy_metric(text: str) -> str | None:
    return "F1" if _F1_SCORE.search(text) is not None else None


def _evidence(text: str) -> list[str]:
    compact = text.strip()
    if not compact:
        raise ValueError("visible task text cannot be empty")
    return [compact[:500]]


def _build_annotation(
    task: Mapping[str, Any],
    signatures: Mapping[str, tuple[object, ...]],
    confidences: Mapping[str, float],
) -> dict[str, Any]:
    text = visible_text(task)
    fields: dict[str, dict[str, Any]] = {}
    for field in ANNOTATION_FIELDS:
        signature = signatures[field]
        keys = _SIGNATURE_KEYS[field]
        values = dict(zip(keys, signature, strict=True))
        status = str(values["status"])
        evidence = [] if status == "not_stated" else _evidence(text)
        confidence = float(min(1.0, max(0.0, confidences[field])))

        if field == "domain":
            fields[field] = {
                "value": values["value"],
                "status": status,
                "evidence": evidence,
                "confidence": confidence,
            }
        elif field == "latency":
            fields[field] = {
                "class": values["class"],
                "max_ms": _latency_ms(text) if status == "explicit" else None,
                "status": status,
                "evidence": evidence,
                "confidence": confidence,
            }
        elif field == "privacy":
            fields[field] = {
                "data_class": values["data_class"],
                "status": status,
                "evidence": evidence,
                "confidence": confidence,
            }
        elif field == "execution_policy":
            fields[field] = {
                "device": values["device"],
                "edge": values["edge"],
                "cloud": values["cloud"],
                "status": status,
                "evidence": evidence,
                "confidence": confidence,
            }
        elif field == "reliability":
            fields[field] = {
                "tier": values["tier"],
                "min_success_probability": (
                    _reliability_probability(text)
                    if status == "explicit"
                    else None
                ),
                "status": status,
                "evidence": evidence,
                "confidence": confidence,
            }
        elif field == "accuracy":
            fields[field] = {
                "tier": values["tier"],
                "min_score": _accuracy_score(text) if status == "explicit" else None,
                "metric": values["metric"],
                "status": status,
                "evidence": evidence,
                "confidence": confidence,
            }
        else:
            fields[field] = {
                "value": values["value"],
                "status": status,
                "evidence": evidence,
                "confidence": confidence,
            }

    abstained = [
        field
        for field in ANNOTATION_FIELDS
        if fields[field]["status"] in {"not_stated", "ambiguous"}
    ]
    return {
        "schema_version": "1.0.0",
        "task_id": str(task["task_id"]),
        "language": str(task["language"]),
        **fields,
        "abstained_fields": abstained,
        "overall_confidence": float(
            np.mean([confidences[field] for field in ANNOTATION_FIELDS])
        ),
    }


class ConstantProfileBaseline:
    """Predict the most frequent development signature for every field."""

    name = "constant_profile"

    def __init__(self) -> None:
        self._signatures: dict[str, tuple[object, ...]] = {}
        self._confidences: dict[str, float] = {}

    def fit(
        self,
        tasks: Sequence[Mapping[str, Any]],
        annotations: Sequence[Mapping[str, Any]],
    ) -> None:
        if not tasks or len(tasks) != len(annotations):
            raise ValueError("tasks and annotations must have the same non-zero size")
        for field in ANNOTATION_FIELDS:
            counts = Counter(
                semantic_signature(annotation, field) for annotation in annotations
            )
            signature, count = min(
                counts.items(),
                key=lambda item: (-item[1], _signature_label(item[0])),
            )
            self._signatures[field] = signature
            self._confidences[field] = count / len(annotations)

    def predict(self, task: Mapping[str, Any]) -> SemanticPrediction:
        if set(self._signatures) != set(ANNOTATION_FIELDS):
            raise RuntimeError("constant baseline must be fitted before prediction")
        started = time.perf_counter_ns()
        annotation = _build_annotation(task, self._signatures, self._confidences)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        return SemanticPrediction(annotation, elapsed)


_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "healthcare": (
        "hasta",
        "klinik",
        "hastane",
        "saglik",
        "ekg",
        "glikoz",
        "patoloji",
        "radyoloji",
        "ambulans",
        "yogun bakim",
    ),
    "industrial": (
        "uretim",
        "fabrika",
        "endustri",
        "robot kolu",
        "cnc",
        "montaj",
        "konveyor",
        "paketleme hatti",
    ),
    "transportation": (
        "ulasim",
        "trafik",
        "arac",
        "otobus",
        "demiryolu",
        "metro",
        "havalimani",
        "otopark",
        "feribot",
    ),
    "public_safety": (
        "acil durum",
        "afet",
        "yangin",
        "deprem",
        "arama kurtarma",
        "kolluk",
        "tahliye",
        "heyelan",
    ),
    "smart_city": (
        "belediye",
        "kent",
        "sokak lambasi",
        "hava kalitesi",
        "atik kutusu",
        "kanalizasyon",
        "kamusal",
    ),
    "agriculture": (
        "tarimsal",
        "tarla",
        "sera",
        "sulama",
        "surü",
        "suru",
        "ekin",
        "koyun",
        "hasat",
        "traktor",
    ),
    "consumer": (
        "kisisel cihaz",
        "akilli ev",
        "oyun",
        "sanal gerceklik",
        "artirilmis gerceklik",
        "robot supurge",
        "akilli televizyon",
        "muzik hoparloru",
    ),
}


class RuleBasedBaseline:
    """Parse visible Turkish phrases without authoring metadata or learned weights."""

    name = "rule_keyword"

    def fit(
        self,
        tasks: Sequence[Mapping[str, Any]],
        annotations: Sequence[Mapping[str, Any]],
    ) -> None:
        if len(tasks) != len(annotations):
            raise ValueError("tasks and annotations must have the same size")

    def _domain(self, normalized: str) -> tuple[object, ...]:
        scores = {
            domain: sum(keyword in normalized for keyword in keywords)
            for domain, keywords in _DOMAIN_KEYWORDS.items()
        }
        best_domain, best_score = min(
            scores.items(), key=lambda item: (-item[1], item[0])
        )
        return ("explicit", best_domain if best_score else "other")

    def _signatures(self, task: Mapping[str, Any]) -> dict[str, tuple[object, ...]]:
        text = visible_text(task)
        normalized = _normalize(text)
        signatures: dict[str, tuple[object, ...]] = {
            "domain": self._domain(normalized),
            "latency": ("not_stated", None),
            "privacy": ("not_stated", None),
            "execution_policy": (
                "not_stated",
                "unknown",
                "unknown",
                "unknown",
            ),
            "reliability": ("not_stated", None),
            "accuracy": ("not_stated", None, None),
            "energy_priority": ("not_stated", None),
            "divisibility": ("not_stated", None),
        }

        if "anit" in normalized and "dort saniye" in normalized:
            signatures["latency"] = ("ambiguous", None)
        else:
            maximum = _latency_ms(text)
            if maximum is not None:
                latency_class = (
                    "hard_real_time"
                    if maximum < 50
                    else "real_time"
                    if maximum <= 200
                    else "interactive"
                    if maximum <= 1000
                    else "relaxed"
                )
                signatures["latency"] = ("explicit", latency_class)

        if "kamuya acik" in normalized:
            signatures["privacy"] = ("explicit", "public")
        elif "kisitli veri" in normalized:
            signatures["privacy"] = ("explicit", "restricted")
        elif "hassas veri" in normalized or "kisisel veri" in normalized:
            signatures["privacy"] = ("explicit", "sensitive")
        elif "kurum disina" in normalized:
            signatures["privacy"] = ("explicit", "internal")

        cloud_required = "yalniz bulutta" in normalized
        cloud_forbidden = any(
            phrase in normalized
            for phrase in (
                "bulut kullanilamaz",
                "buluta gonderilmesi kesinlikle yasaktir",
                "genel bulut kullanilamaz",
            )
        )
        if cloud_required and cloud_forbidden:
            signatures["execution_policy"] = (
                "ambiguous",
                "unknown",
                "unknown",
                "unknown",
            )
        elif "cihaz, yakin edge veya bulutta" in normalized:
            signatures["execution_policy"] = (
                "explicit",
                "allowed",
                "allowed",
                "allowed",
            )
        elif "cihaz veya yakin edge" in normalized and cloud_forbidden:
            signatures["execution_policy"] = (
                "explicit",
                "allowed",
                "allowed",
                "forbidden",
            )
        elif "yakin edge sunucusunda" in normalized and cloud_forbidden:
            signatures["execution_policy"] = (
                "partial",
                "unknown",
                "allowed",
                "forbidden",
            )

        probability = _reliability_probability(text)
        if probability is not None:
            tier = (
                "mission_critical"
                if probability >= 0.999
                else "high"
                if probability >= 0.99
                else "standard"
            )
            signatures["reliability"] = ("explicit", tier)
        elif "yuksek guvenilirlik" in normalized:
            signatures["reliability"] = ("explicit", "high")
        elif "basarisiz istek kabul edilebilir" in normalized:
            signatures["reliability"] = ("explicit", "best_effort")

        if "yuksek dogrulukta" in normalized and "dusuk dogruluklu" in normalized:
            signatures["accuracy"] = ("ambiguous", None, None)
        elif _accuracy_score(text) is not None:
            signatures["accuracy"] = (
                "explicit",
                "high",
                _accuracy_metric(text),
            )
        elif "yuksek dogruluk" in normalized:
            signatures["accuracy"] = ("explicit", "high", None)

        if "birinci oncelik" in normalized and "pil" in normalized:
            signatures["energy_priority"] = ("explicit", "high")
        elif "enerji tuketimi" in normalized and "dengelenmelidir" in normalized:
            signatures["energy_priority"] = ("explicit", "balanced")

        if "tek parca" in normalized and "bagimsiz parca" in normalized:
            signatures["divisibility"] = ("ambiguous", None)
        elif "bagimsiz" in normalized and any(
            phrase in normalized for phrase in ("paralel", "parcalari")
        ):
            signatures["divisibility"] = ("explicit", "divisible")
        elif "tek parca" in normalized:
            signatures["divisibility"] = ("explicit", "indivisible")
        return signatures

    def predict(self, task: Mapping[str, Any]) -> SemanticPrediction:
        started = time.perf_counter_ns()
        signatures = self._signatures(task)
        confidences = {
            field: (0.8 if signature[0] != "not_stated" else 0.7)
            for field, signature in signatures.items()
        }
        annotation = _build_annotation(task, signatures, confidences)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        return SemanticPrediction(annotation, elapsed)


class TfidfLogisticBaseline:
    """One text-only multiclass logistic model per coherent semantic field."""

    name = "tfidf_logistic_regression"

    def __init__(self, *, random_seed: int = 20260929) -> None:
        self._random_seed = random_seed
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=20_000,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            dtype=np.float64,
        )
        self._models: dict[str, LogisticRegression | None] = {}
        self._constant_labels: dict[str, str] = {}

    def fit(
        self,
        tasks: Sequence[Mapping[str, Any]],
        annotations: Sequence[Mapping[str, Any]],
    ) -> None:
        if not tasks or len(tasks) != len(annotations):
            raise ValueError("tasks and annotations must have the same non-zero size")
        matrix = self._vectorizer.fit_transform([visible_text(task) for task in tasks])
        for field in ANNOTATION_FIELDS:
            labels = np.asarray(
                [
                    _signature_label(semantic_signature(annotation, field))
                    for annotation in annotations
                ]
            )
            classes = np.unique(labels)
            if len(classes) == 1:
                self._models[field] = None
                self._constant_labels[field] = str(classes[0])
                continue
            model = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2_000,
                random_state=self._random_seed,
                solver="lbfgs",
            )
            model.fit(matrix, labels)
            self._models[field] = model

    def predict(self, task: Mapping[str, Any]) -> SemanticPrediction:
        if set(self._models) != set(ANNOTATION_FIELDS):
            raise RuntimeError("TF-IDF baseline must be fitted before prediction")
        started = time.perf_counter_ns()
        matrix = self._vectorizer.transform([visible_text(task)])
        signatures: dict[str, tuple[object, ...]] = {}
        confidences: dict[str, float] = {}
        for field in ANNOTATION_FIELDS:
            model = self._models[field]
            if model is None:
                label = self._constant_labels[field]
                confidence = 1.0
            else:
                probabilities = model.predict_proba(matrix)[0]
                best = int(np.argmax(probabilities))
                label = str(model.classes_[best])
                confidence = float(probabilities[best])
            signatures[field] = _label_signature(label)
            confidences[field] = confidence
        annotation = _build_annotation(task, signatures, confidences)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        return SemanticPrediction(annotation, elapsed)
