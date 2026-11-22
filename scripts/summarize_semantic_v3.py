# ruff: noqa: E501, RUF001
"""Publication-style tables for the v3 semantic benchmark (with 95% CIs)."""

from __future__ import annotations

import argparse
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
    "e5_finetuned_augmented_v31_raw": "E5-small ince ayar v3.1 (raw cache)",
    "llm_fewshot_Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Instruct 4-shot",
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
    "independent_paraphrase": (
        "Bağımsız makine paraphrase (n=188), etiket değişmezliği varsayımı — "
        "metinler insan tarafından yeniden etiketlenmedi"
    ),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    point = data["point_estimates"]
    intervals = data["intervals"]
    lines = [
        f"# Semantik Çıkarım Benchmark'ı — {data['results_version']}",
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
        if set_key not in intervals:
            continue
        lines += [
            f"## {set_name}",
            "",
            "| Yöntem | Eğitim n | Tam kayıt doğruluğu | Macro-F1 | Macro-F1 farkı (vs kural) "
            "| Kritik gizlilik P / R / F1 (n+) | Abstention F1 | ECE raw→cal ↓ "
            "| Brier raw→cal ↓ | ms/görev |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
        for name, label in MODELS.items():
            if name not in point or set_key not in point[name]["sets"]:
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
            raw_calibration = p.get("calibration_raw", p["calibration"])
            latency = p.get("inference_latency_ms_mean")
            latency_text = "not recorded" if latency is None else f"{latency:.2f}"
            lines.append(
                f"| {label} | {point[name]['training_task_count']} | "
                f"{_ci(ci['exact_record_accuracy'])} | "
                f"**{_ci(ci['macro_f1_across_fields'])}** | "
                f"{_diff(ci['macro_f1_across_fields'])} | "
                f"{critical} | "
                f"{100 * p['abstention_f1']:.1f} | "
                f"{raw_calibration['ece_10_bin']:.3f}→"
                f"{p['calibration']['ece_10_bin']:.3f} | "
                f"{raw_calibration['brier']:.3f}→"
                f"{p['calibration']['brier']:.3f} | "
                f"{latency_text} |"
            )
        lines += [
            "",
            "Alan bazında macro-F1 (%):",
            "",
            "| Yöntem | " + " | ".join(FIELDS) + " |",
            "|---|" + "---:|" * len(FIELDS),
        ]
        for name, label in MODELS.items():
            if name not in point or set_key not in point[name]["sets"]:
                continue
            fields = point[name]["sets"][set_key]["fields_macro_f1"]
            lines.append(
                f"| {label} | "
                + " | ".join(f"{100 * fields[f]:.1f}" for f in FIELDS)
                + " |"
            )
        lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
