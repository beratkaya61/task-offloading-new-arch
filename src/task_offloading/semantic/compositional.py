# ruff: noqa: E501, RUF001
"""Compositional, label-by-construction semantic tasks (machine-generated).

The locked corpus states at most one template pattern per text, so most texts
carry one or two requirements and execution restrictions are rare.  This
generator keeps the *domain sentence* of an existing task and appends 0-4
independently drawn requirement slots (latency, privacy, execution policy,
reliability, accuracy, energy, divisibility) with varied numeric thresholds.

* Each slot is first written in the canonical template wording, whose label
  mapping is exactly the one used by ``authoring_preannotation`` and the
  annotation protocol (Sections 5.1-5.8), and then optionally rewritten with
  the ``train`` or the held-out ``test`` paraphrase pool.
* Labels therefore hold *by construction*; they are machine-generated design
  labels and must be reported as such, never as human annotation.
* Test sets are composed only from evaluation-split domain sentences and the
  test paraphrase pool; training sets only from development-split sentences
  and the train pool.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from task_offloading.semantic.benchmark import ANNOTATION_FIELDS
from task_offloading.semantic.paraphrase import paraphrase_text

Wording = Literal["canonical", "train", "test"]

LATENCY_MS = (30, 50, 80, 120, 200, 300, 500, 800, 1500, 3000)
RELIABILITY_PERCENT = ("95", "98", "99", "99,5", "99,9", "99,95", "99,99")
ACCURACY_F1 = ("0,93", "0,95", "0,97")
SLOT_COUNT_PROBABILITIES = (0.1, 0.25, 0.3, 0.25, 0.1)  # 0..4 slots


def _latency_class(max_ms: float) -> str:
    if max_ms < 50:
        return "hard_real_time"
    if max_ms <= 200:
        return "real_time"
    if max_ms <= 1000:
        return "interactive"
    return "relaxed"


def _reliability_tier(probability: float) -> str:
    if probability >= 0.999:
        return "mission_critical"
    if probability >= 0.98:
        return "high"
    return "standard"


def _percent_suffix(text: str) -> str:
    return {"5": "i", "8": "i", "9": "u"}[text[-1]]


def _unknown(**extra: Any) -> dict[str, Any]:
    return {**extra, "status": "not_stated", "evidence": [], "confidence": 1.0}


def _explicit(status: Any = "explicit", **values: Any) -> dict[str, Any]:
    return {**values, "status": status, "evidence": [], "confidence": 1.0}


def _slot_latency(rng: np.random.Generator) -> tuple[list[str], dict[str, Any]]:
    if rng.random() < 0.15:
        return (
            [
                "Yanıt anlık verilmelidir.",
                "Dört saniyelik gecikme tamamen kabul edilebilir.",
            ],
            _explicit("ambiguous", **{"class": None, "max_ms": None}),
        )
    value = int(rng.choice(LATENCY_MS))
    return (
        [f"Sonuç en geç {value} ms içinde hazır olmalıdır."],
        _explicit(**{"class": _latency_class(value), "max_ms": float(value)}),
    )


def _slot_privacy(rng: np.random.Generator) -> tuple[list[str], dict[str, Any]]:
    choice = int(rng.integers(3))
    sentence, data_class = (
        ("İşlenen kayıt hassas veridir.", "sensitive"),
        ("İşlenen ham kayıt kısıtlı veridir.", "restricted"),
        ("Kullanılan veri kamuya açıktır.", "public"),
    )[choice]
    return [sentence], _explicit(data_class=data_class)


def _slot_execution(rng: np.random.Generator) -> tuple[list[str], dict[str, Any]]:
    choice = int(rng.choice(4, p=(0.35, 0.25, 0.25, 0.15)))
    if choice == 0:
        return (
            ["Cihaz veya yakın edge kullanılabilir; genel bulut kullanılamaz."],
            _explicit(device="allowed", edge="allowed", cloud="forbidden"),
        )
    if choice == 1:
        return (
            ["Yakın edge sunucusunda işlem yapılabilir ve genel bulut kullanılamaz."],
            _explicit("partial", device="unknown", edge="allowed", cloud="forbidden"),
        )
    if choice == 2:
        return (
            ["Görev cihaz, yakın edge veya bulutta çalıştırılabilir."],
            _explicit(device="allowed", edge="allowed", cloud="allowed"),
        )
    return (
        [
            "Analiz yalnız bulutta yapılmalıdır.",
            "Aynı kaydın buluta gönderilmesi kesinlikle yasaktır.",
        ],
        _explicit("ambiguous", device="unknown", edge="unknown", cloud="unknown"),
    )


def _slot_reliability(rng: np.random.Generator) -> tuple[list[str], dict[str, Any]]:
    if rng.random() < 0.25:
        return (
            ["Bağlantı kesintilerine karşı yüksek güvenilirlik sağlanmalıdır."],
            _explicit(tier="high", min_success_probability=None),
        )
    text = str(rng.choice(RELIABILITY_PERCENT))
    probability = round(float(text.replace(",", ".")) / 100.0, 6)
    return (
        [
            f"İsteklerin en az %{text}'{_percent_suffix(text)} başarıyla tamamlanmalıdır."
        ],
        _explicit(
            tier=_reliability_tier(probability), min_success_probability=probability
        ),
    )


def _slot_accuracy(rng: np.random.Generator) -> tuple[list[str], dict[str, Any]]:
    draw = rng.random()
    if draw < 0.15:
        return (
            [
                "Sonuç yüksek doğrulukta olmalıdır.",
                "Yaklaşık ve düşük doğruluklu bir sonuç da yeterlidir.",
            ],
            _explicit("ambiguous", tier=None, min_score=None, metric=None),
        )
    if draw < 0.4:
        return (
            ["Sonuç yüksek doğrulukta olmalıdır."],
            _explicit(tier="high", min_score=None, metric=None),
        )
    text = str(rng.choice(ACCURACY_F1))
    return (
        [f"Sonucun F1 skoru en az {text} olmalıdır."],
        _explicit(tier="high", min_score=float(text.replace(",", ".")), metric="F1"),
    )


def _slot_energy(rng: np.random.Generator) -> tuple[list[str], dict[str, Any]]:
    if rng.random() < 0.5:
        return ["Pil tüketimini azaltmak birinci önceliktir."], _explicit(value="high")
    return (
        ["Enerji tüketimi ile yanıt süresi dengelenmelidir."],
        _explicit(value="balanced"),
    )


def _slot_divisibility(rng: np.random.Generator) -> tuple[list[str], dict[str, Any]]:
    sentence = (
        "Veri parçaları bağımsız ve paralel işlenebilir."
        if rng.random() < 0.5
        else "Görev parçaları birbirinden bağımsız işlenebilir."
    )
    return [sentence], _explicit(value="divisible")


SLOTS = {
    "latency": _slot_latency,
    "privacy": _slot_privacy,
    "execution_policy": _slot_execution,
    "reliability": _slot_reliability,
    "accuracy": _slot_accuracy,
    "energy_priority": _slot_energy,
    "divisibility": _slot_divisibility,
}


def _domain_sentence(task_text: str) -> str:
    return task_text.split(". ", 1)[0].rstrip(".") + "."


def _empty_annotation(task_id: str, domain: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "task_id": task_id,
        "language": "tr",
        "domain": _explicit(value=domain),
        "latency": _unknown(**{"class": None, "max_ms": None}),
        "privacy": _unknown(data_class=None),
        "execution_policy": _unknown(device="unknown", edge="unknown", cloud="unknown"),
        "reliability": _unknown(tier=None, min_success_probability=None),
        "accuracy": _unknown(tier=None, min_score=None, metric=None),
        "energy_priority": _unknown(value=None),
        "divisibility": _unknown(value=None),
    }


def compose_tasks(
    base_tasks: Sequence[Mapping[str, Any]],
    *,
    variants_per_task: int,
    wording: Wording,
    seed: int,
    prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return (tasks, labels-by-task-id) for ``variants_per_task`` per base."""

    tasks: list[dict[str, Any]] = []
    labels: dict[str, dict[str, Any]] = {}
    for base in base_tasks:
        domain = str(base["design_strata"]["scenario_domain"])
        domain_sentence = _domain_sentence(str(base["task_text"]))
        for variant in range(variants_per_task):
            digest = hashlib.sha256(
                f"{seed}|{base['task_id']}|{variant}".encode()
            ).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            task_id = f"{prefix}_{base['task_id']}_{variant:02d}"
            annotation = _empty_annotation(task_id, domain)
            count = int(rng.choice(5, p=SLOT_COUNT_PROBABILITIES))
            names = list(rng.choice(list(SLOTS), size=count, replace=False))
            sentences: list[str] = []
            for name in names:
                slot_sentences, label = SLOTS[name](rng)
                sentences.extend(slot_sentences)
                annotation[name] = label
            order = rng.permutation(len(sentences))
            sentences = [sentences[i] for i in order]
            split_at = (
                int(rng.integers(0, len(sentences) + 1))
                if rng.random() < 0.5
                else len(sentences)
            )
            task_text = " ".join([domain_sentence, *sentences[:split_at]])
            policy_text = " ".join(sentences[split_at:]) or None
            if wording != "canonical":
                task_text, _ = paraphrase_text(
                    task_text, pool=wording, key=f"{task_id}|t"
                )
                if policy_text is not None:
                    policy_text, _ = paraphrase_text(
                        policy_text, pool=wording, key=f"{task_id}|p"
                    )
            annotation["abstained_fields"] = [
                field
                for field in ANNOTATION_FIELDS
                if annotation[field]["status"] in {"not_stated", "ambiguous"}
            ]
            annotation["overall_confidence"] = 1.0
            tasks.append(
                {
                    "task_id": task_id,
                    "language": "tr",
                    "task_text": task_text,
                    "user_policy_text": policy_text,
                    "design_strata": {"scenario_domain": domain},
                    "provenance": {
                        "origin": "compositional_machine_generated",
                        "base_task_id": base["task_id"],
                        "wording": wording,
                        "label_claim": "label_by_construction_not_human",
                    },
                }
            )
            labels[task_id] = annotation
    return tasks, labels


__all__ = ["compose_tasks"]
