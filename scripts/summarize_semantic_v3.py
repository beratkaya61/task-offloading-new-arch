# ruff: noqa: E501, RUF001
"""Publication-style tables for the v3 semantic benchmark (with 95% CIs)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    ROOT / "data" / "semantic_benchmark" / "main" / "benchmark_v3" / "results.v3.json"
)
OUTPUT = ROOT / "phase_reports" / "SEMANTIC_RESULTS_V3.md"

MODELS = {
    "constant_profile": "Sabit profil",
    "rule_keyword": "Kural/anahtar kelime",
    "tfidf_lr": "TF-IDF + LR",
    "tfidf_lr_augmented": "TF-IDF + LR (+veri)",
    "e5_frozen_lr": "E5-small donuk + LR",
    "e5_frozen_lr_augmented": "E5-small donuk + LR (+veri)",
    "e5_finetuned_augmented": "E5-small ince ayar (+veri)",
}
SETS = {
    "human60_original": "İnsan referansı (n=60), orijinal ifade",
    "human60_paraphrase": (
        "Makine yeniden yazımı, insan etiketi aktarılmış (n=60) — metinler "
        "otomatik yeniden ifade edildi; insan bu metinleri görmedi, etiketler "
        "anlam korunduğu varsayımıyla orijinalden aktarıldı"
    ),
    "compositional600_canonical": (
        "Kompozisyonel (n=600), şablon ifadesi — yalnız doğruluk kontrolü; "
        "test cümleleri kural ve eğitim verisiyle aynı şablondan, kanıt değildir"
    ),
    "compositional600_paraphrase": "Kompozisyonel (n=600), görülmemiş ifade",
}
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


def _ci(entry: dict[str, Any]) -> str:
    low, high = entry["ci95"]
    return f"{100 * entry['value']:.1f} [{100 * low:.1f}, {100 * high:.1f}]"


def _diff(entry: dict[str, Any]) -> str:
    diff = entry.get("diff_vs_baseline")
    if diff is None:
        return "—"
    low, high = diff["ci95"]
    p = diff["p_value_two_sided"]
    p_text = "<0.001" if p < 0.001 else f"{p:.3f}"
    return (
        f"{100 * diff['value']:+.1f} [{100 * low:+.1f}, {100 * high:+.1f}]; p={p_text}"
    )


def main() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    point = data["point_estimates"]
    intervals = data["intervals"]
    lines = [
        "# Semantik Çıkarım Benchmark'ı v3 — Yayın Tabloları",
        "",
        "Değerler yüzde olarak verilmiştir; köşeli parantez içinde görev düzeyinde "
        "1000 tekrarlı bootstrap ile %95 güven aralığı (GA) yer alır. *Fark* "
        "sütunu, aynı bootstrap örnekleri üzerinde kural tabanlı yönteme göre "
        "eşleştirilmiş farkı, %95 GA'sını ve iki yönlü bootstrap p-değerini "
        "gösterir. Değerlendirme etiketleri eğitim veya ayar için kullanılmamıştır.",
        "",
        "Etiket kaynakları: insan referansı = protokol-normalize tek-insan "
        "referansı (yalnız orijinal 60 metin); yeniden ifade seti = makine "
        "üretimi metin + aktarılmış insan etiketi; kompozisyonel = makine "
        "üretimi, inşa yoluyla etiketli. Toplam gerçek insan etiketli metin: "
        "60 ana + 20 pilot. "
        "*(+veri)* = 180 görev + eğitim havuzundan yeniden ifade + kompozisyonel "
        f"eğitim görevleri (toplam {point['tfidf_lr_augmented']['training_task_count']}).",
        "",
    ]
    for set_key, set_name in SETS.items():
        lines += [
            f"## {set_name}",
            "",
            "| Yöntem | Eğitim n | Tam kayıt doğruluğu | Macro-F1 | Macro-F1 farkı (vs kural) "
            "| Kritik gizlilik P / R / F1 (n+) | Abstention F1 | ECE ↓ | ms/görev |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
        for name, label in MODELS.items():
            if name not in point:
                continue
            p = point[name]["sets"][set_key]
            ci = intervals[set_key][name]
            cp = p["critical_privacy"]

            def pct(value: Any) -> str:
                return "—" if value is None else f"{100 * value:.1f}"

            critical = (
                f"{pct(cp['precision'])} / {pct(cp['recall'])} / {pct(cp['f1'])} "
                f"({cp['positive_reference_count']})"
            )
            lines.append(
                f"| {label} | {point[name]['training_task_count']} | "
                f"{_ci(ci['exact_record_accuracy'])} | "
                f"**{_ci(ci['macro_f1_across_fields'])}** | "
                f"{_diff(ci['macro_f1_across_fields'])} | "
                f"{critical} | "
                f"{100 * p['abstention_f1']:.1f} | "
                f"{p['calibration']['ece_10_bin']:.3f} | "
                f"{p['inference_latency_ms_mean']:.2f} |"
            )
        lines += [
            "",
            "Alan bazında macro-F1 (%):",
            "",
            "| Yöntem | " + " | ".join(FIELDS) + " |",
            "|---|" + "---:|" * len(FIELDS),
        ]
        for name, label in MODELS.items():
            if name not in point:
                continue
            fields = point[name]["sets"][set_key]["fields_macro_f1"]
            lines.append(
                f"| {label} | "
                + " | ".join(f"{100 * fields[f]:.1f}" for f in FIELDS)
                + " |"
            )
        lines.append("")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
