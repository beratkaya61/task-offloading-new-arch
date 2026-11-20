# ruff: noqa: RUF001
"""Render every semantic extractor's metrics as Markdown tables.

Reads ``data/semantic_benchmark/main/paraphrase_robustness/results.v2.json``
and writes ``phase_reports/SEMANTIC_RESULTS_SUMMARY.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    ROOT
    / "data"
    / "semantic_benchmark"
    / "main"
    / "paraphrase_robustness"
    / "results.v2.json"
)
OUTPUT = ROOT / "phase_reports" / "SEMANTIC_RESULTS_SUMMARY.md"

LABELS = {
    "constant_profile": "Sabit profil",
    "rule_keyword": "Kural/anahtar kelime",
    "tfidf_logistic_regression": "TF-IDF + LR",
    "tfidf_logistic_regression_augmented": "TF-IDF + LR + çoğaltma",
    "e5_small_logreg": "E5-small (donuk) + LR",
    "e5_small_logreg_augmented": "E5-small (donuk) + LR + çoğaltma",
    "e5_small_finetuned_augmented": "E5-small ince ayar + çoğaltma",
}
SETS = {"original_wording": "Orijinal ifade", "test_paraphrase": "Yeniden ifade"}
FIELDS = (
    "domain",
    "latency",
    "privacy",
    "execution_policy",
    "reliability",
    "accuracy",
    "energy_priority",
    "divisibility",
)


def _pct(value: Any) -> str:
    return "–" if value is None else f"%{100 * float(value):.1f}"


def _num(value: Any, digits: int = 3) -> str:
    return "–" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    models: dict[str, Any] = payload["models"]
    lines: list[str] = [
        "# Semantik Çıkarım Sonuçları — Özet Tablo",
        "",
        "Değerlendirme: 60 kilitli görev, protokol-normalize tek-insan referansı "
        "(`annotator_a.protocol_normalized.v1.3.1`). *Yeniden ifade* seti, aynı 60 "
        "görevin anlamı koruyan otomatik (makine üretimi) yeniden yazımıdır; "
        "eğitim çoğaltmasında kullanılan ifadelerden ayrık bir havuzdan gelir. "
        "Değerlendirme etiketleri eğitim veya ayar için kullanılmamıştır.",
        "",
        f"Çoğaltma: görev başına {payload['augmented_variants_per_task']} "
        "yeniden ifade varyantı (yalnız eğitim).",
        "",
    ]
    for set_key, set_name in SETS.items():
        lines += [
            f"## {set_name} — genel metrikler",
            "",
            "| Yöntem | Eğitim örneği | Tam kayıt | Alan doğruluğu | Macro-F1 | "
            "Kritik gizlilik P/R/F1 | Abstention F1 | ECE ↓ | Brier ↓ | "
            "Süre (ms/görev) |",
            "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
        for name, model in models.items():
            row = model["sets"][set_key]
            lines.append(
                f"| {LABELS.get(name, name)} | {model['training_task_count']} | "
                f"{_pct(row['exact_record_accuracy'])} | "
                f"{_pct(row['exact_field_accuracy'])} | "
                f"**{_pct(row['macro_f1_across_fields'])}** | "
                f"{_pct(row['critical_privacy_precision'])} / "
                f"{_pct(row['critical_privacy_recall'])} / "
                f"{_pct(row['critical_privacy_f1'])} | "
                f"{_pct(row['abstention_f1'])} | "
                f"{_num(row['calibration_ece_10_bin'])} | "
                f"{_num(row['calibration_brier'])} | "
                f"{_num(row['inference_latency_ms_mean'], 2)} |"
            )
        lines += [
            "",
            f"## {set_name} — alan bazında macro-F1",
            "",
            "| Yöntem | " + " | ".join(FIELDS) + " |",
            "|---|" + "---:|" * len(FIELDS),
        ]
        for name, model in models.items():
            fields = model["sets"][set_key]["fields_macro_f1"]
            lines.append(
                f"| {LABELS.get(name, name)} | "
                + " | ".join(_pct(fields[field]) for field in FIELDS)
                + " |"
            )
        lines.append("")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
