# ruff: noqa: E501, RUF001
"""Audit and render targeted human-annotator calibration material."""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from task_offloading.semantic.benchmark import ANNOTATION_FIELDS

_SPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class CalibrationAudit:
    """Structural and leakage checks for a worked-example package."""

    example_count: int
    focus_field_counts: tuple[tuple[str, int], ...]
    issues: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


def _normalized_text(value: str) -> str:
    return _SPACE_PATTERN.sub(" ", value.casefold()).strip()


def audit_targeted_calibration(
    package: Mapping[str, Any],
    pilot_corpus: Sequence[Mapping[str, Any]],
) -> CalibrationAudit:
    """Reject duplicate, pilot-reused, or internally inconsistent examples."""

    issues: list[str] = []
    examples_value = package.get("examples")
    if not isinstance(examples_value, list):
        return CalibrationAudit(0, (), ("examples must be a list",))

    pilot_texts = {
        _normalized_text(str(row.get("task_text", ""))) for row in pilot_corpus
    }
    identifiers: set[str] = set()
    texts: set[str] = set()
    focus_counts: Counter[str] = Counter()

    for index, raw_example in enumerate(examples_value, start=1):
        if not isinstance(raw_example, Mapping):
            issues.append(f"example {index} must be an object")
            continue
        identifier = str(raw_example.get("example_id", ""))
        if identifier in identifiers:
            issues.append(f"duplicate example_id: {identifier}")
        identifiers.add(identifier)

        text = _normalized_text(str(raw_example.get("text", "")))
        if text in texts:
            issues.append(f"duplicate normalized calibration text: {identifier}")
        texts.add(text)
        if text in pilot_texts:
            issues.append(f"calibration text reuses a pilot task: {identifier}")

        focus_value = raw_example.get("focus_fields")
        decisions_value = raw_example.get("decisions")
        if not isinstance(focus_value, list) or not isinstance(decisions_value, list):
            issues.append(f"{identifier} has invalid focus_fields or decisions")
            continue
        focus = [str(field) for field in focus_value]
        decision_fields = [
            str(decision.get("field", ""))
            for decision in decisions_value
            if isinstance(decision, Mapping)
        ]
        if len(decision_fields) != len(decisions_value):
            issues.append(f"{identifier} contains a non-object decision")
        if len(decision_fields) != len(set(decision_fields)):
            issues.append(f"{identifier} repeats a decision field")
        if set(focus) != set(decision_fields):
            issues.append(
                f"{identifier} focus_fields must exactly match decision fields"
            )
        focus_counts.update(focus)

    missing_fields = set(ANNOTATION_FIELDS) - set(focus_counts)
    if missing_fields:
        issues.append(
            "calibration misses annotation fields: " + ", ".join(sorted(missing_fields))
        )
    if package.get("pilot_content_reused") is not False:
        issues.append("pilot_content_reused must be false")
    if package.get("is_ground_truth") is not False:
        issues.append("worked examples cannot be declared ground truth")

    return CalibrationAudit(
        example_count=len(examples_value),
        focus_field_counts=tuple(sorted(focus_counts.items())),
        issues=tuple(issues),
    )


def _decision_summary(decision: Mapping[str, Any]) -> str:
    parts = [
        f"alan={decision['field']}",
        f"durum={decision['status']}",
    ]
    if "value" in decision:
        value = decision["value"]
        parts.append(f"değer={'null' if value is None else value}")
    if "numeric_value" in decision:
        value = decision["numeric_value"]
        parts.append(f"sayısal değer={'null' if value is None else value}")
    if decision.get("metric") is not None:
        parts.append(f"metrik={decision['metric']}")
    targets = decision.get("targets")
    if isinstance(targets, Mapping):
        parts.extend(
            f"{target}={targets[target]}" for target in ("device", "edge", "cloud")
        )
    return ", ".join(str(part) for part in parts)


def render_targeted_calibration(package: Mapping[str, Any]) -> str:
    """Render a deterministic, network-free worked-example booklet."""

    examples = package.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("calibration package must contain examples")

    cards: list[str] = []
    for index, raw_example in enumerate(examples, start=1):
        if not isinstance(raw_example, Mapping):
            raise ValueError(f"example {index} must be an object")
        decisions = raw_example.get("decisions")
        if not isinstance(decisions, list):
            raise ValueError(f"example {index} decisions must be a list")
        decision_items: list[str] = []
        for decision in decisions:
            if not isinstance(decision, Mapping):
                raise ValueError(f"example {index} contains a non-object decision")
            summary = html.escape(_decision_summary(decision))
            explanation = html.escape(str(decision["explanation"]))
            decision_items.append(
                f"<li><code>{summary}</code><p>{explanation}</p></li>"
            )
        task_text = html.escape(str(raw_example["text"]))
        example_id = html.escape(str(raw_example["example_id"]))
        cards.append(
            "<article class=\"card\">"
            f"<div class=\"eyebrow\">Örnek {index} · {example_id}</div>"
            f"<p class=\"task\">{task_text}</p>"
            "<details><summary>Doğru kararı ve gerekçeyi göster</summary>"
            f"<ul>{''.join(decision_items)}</ul></details></article>"
        )

    protocol_version = html.escape(str(package.get("protocol_version", "")))
    payload = json.dumps(package, ensure_ascii=False, sort_keys=True)
    return f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; connect-src 'none'">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hedefli annotator kalibrasyonu</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; line-height: 1.55; }}
    body {{ margin: 0; background: #f4f6f8; color: #17202a; }}
    main {{ max-width: 920px; margin: auto; padding: 32px 18px 64px; }}
    header, .card {{ background: white; border: 1px solid #d9e0e6; border-radius: 14px; }}
    header {{ padding: 24px; margin-bottom: 18px; }}
    .notice {{ border-left: 5px solid #1769aa; padding-left: 14px; }}
    .card {{ padding: 20px; margin: 14px 0; }}
    .eyebrow {{ color: #546e7a; font-size: .86rem; font-weight: 700; text-transform: uppercase; }}
    .task {{ font-size: 1.08rem; }}
    summary {{ cursor: pointer; color: #0d47a1; font-weight: 700; }}
    code {{ white-space: normal; color: #263238; }}
    li p {{ margin-top: 5px; }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Hedefli annotator kalibrasyonu</h1>
    <p class="notice"><strong>Bu bir anket veya sınav değildir.</strong> Bunlar yeni,
    cevaplı öğretim örnekleridir. Pilot A/B cevapları kullanılmamıştır ve buradaki
    kararlar insan ground truth'u sayılmaz.</p>
    <p>Önce metni okuyup kendi kararınızı düşünün; sonra ayrıntıyı açarak kuralı
    kontrol edin. Protokol sürümü: <strong>{protocol_version}</strong>.</p>
  </header>
  {''.join(cards)}
</main>
<!-- package_sha_source={html.escape(payload)} -->
</body>
</html>
"""
