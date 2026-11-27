# ruff: noqa: E501, RUF001
"""Build the evidence-backed semantic method/data audit for the advisor report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "data" / "semantic_benchmark" / "main" / "benchmark_v3"
EXPANDED = ROOT / "data" / "semantic_benchmark" / "expanded_v2"
LORA = ROOT / "models" / "adapters" / "Qwen3-8B_semantic_exp1_4bit"
OUTPUT = ROOT / "phase_reports" / "SEMANTIC_DATA_AND_METHOD_AUDIT_2026_11_27.md"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _pct(value: Any) -> str:
    return "—" if value is None else f"%{100 * float(value):.1f}".replace(".", ",")


def _ms(value: Any) -> str:
    return "—" if value is None else f"{float(value):,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _metric(point: dict[str, Any], model: str, set_name: str, key: str) -> Any:
    try:
        return point[model]["sets"][set_name][key]
    except KeyError:
        return None


def main() -> None:
    cached = _json(V3 / "results.cached_v3_2.json")
    authoritative = _json(V3 / "results.v3.json")
    old_llm = _json(V3 / "results_llm.json")
    expansion = _json(EXPANDED / "split_manifest.candidate.json")
    annotation = _json(EXPANDED / "annotation" / "annotation_plan.v2.json")
    provisional = _json(EXPANDED / "results" / "results.provisional_v2.json")
    qwen_shots = {
        shots: _json(
            V3
            / f"results_llm_prompt_Qwen3-8B_{shots}shot_human60_original_human60_paraphrase_independent_paraphrase.json"
        )
        for shots in (0, 4, 8)
    }
    lora_path = LORA / "training_metadata.json"
    lora = _json(lora_path) if lora_path.exists() else None
    point = cached["point_estimates"]
    methods = [
        ("constant_profile", "Constant profile"),
        ("rule_keyword", "Rule/keyword"),
        ("tfidf_lr", "TF-IDF + LR (180)"),
        ("tfidf_lr_augmented", "TF-IDF + LR (1.924)"),
        ("e5_frozen_lr", "Frozen E5 + LR (180)"),
        ("e5_frozen_lr_augmented", "Frozen E5 + LR (1.924)"),
        ("e5_finetuned_augmented_v31_raw", "Fine-tuned E5 (1.924)"),
        ("llm_fewshot_Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B 4-shot"),
    ]
    authoritative_names = {
        "constant_profile": "constant_profile",
        "rule_keyword": "rule_keyword",
        "tfidf_lr": "tfidf_lr",
        "tfidf_lr_augmented": "tfidf_lr_augmented",
        "e5_frozen_lr": "e5_frozen_lr",
        "e5_frozen_lr_augmented": "e5_frozen_lr_augmented",
        "e5_finetuned_augmented_v31_raw": "e5_finetuned_augmented",
    }
    sets = [
        ("human60_original", "Human-60 original"),
        ("human60_paraphrase", "Human-60 rewrite"),
        ("compositional600_canonical", "Comp-600 canonical"),
        ("compositional600_paraphrase", "Comp-600 unseen wording"),
        ("independent_paraphrase", "Independent-188"),
    ]
    lines = [
        "# Semantik Veri, Yöntem ve Karşılaştırma Denetimi — 27.11.2026",
        "",
        "## 1. Karar özeti",
        "",
        "- Eski confirmatory v3.2 sonuçları korunur; 180 temel development görevi ve "
        "bunlardan türetilmiş 1.924 training satırı tarihsel baseline'dır.",
        "- Cached sonuç dosyasındaki `1.980` değeri sabit yazılmış metadata hatasıdır. "
        "Güncel training dosyaları ve authoritative `results.v3.json` değeri `1.924`'tür.",
        "- Qwen2.5-1.5B koşulu supervised training değildir: 180 development etiketi "
        "çıktı sözlüğünü tanımlar, yalnız dört metin–etiket çifti prompt demonstrasyonudur.",
        "- Expanded-v1 iki sızıntı yolu nedeniyle geri çekildi: 600 ailenin 244'ü kilitli "
        "değerlendirme bağlamlarını kullanıyordu ve 120 cihazın 98'i split'ler arasında geçiyordu.",
        f"- Yeni ortak koşul Expanded-v2'dir: `{expansion['row_counts']['train']}` train, "
        f"`{expansion['row_counts']['validation']}` validation, `{expansion['row_counts']['test']}` "
        "test satırı; cihazlar ve işlemler split'ler arasında geçmez. Prompt-only Qwen "
        "0/4/8-shot ayrı track'tir.",
        "- Expanded-v2 henüz final veri seti değildir; insan annotation ve adjudication "
        "tamamlanana kadar yalnız candidate/engineering artifact statüsündedir.",
        "",
        "## 2. Veri envanteri",
        "",
        "| Artifact | Bağımsız birim | İlişkili satır | Label durumu | Rol |",
        "|---|---:|---:|---|---|",
        "| v3 development | 90 scenario family / 180 metin | 180 | Machine weak | Eski D180 training |",
        "| v3 augmented | Aynı 180 temel metin | 1.924 | Weak + construction | Eski augmented training |",
        "| Human-60 original | 30 scenario family / 60 metin | 60 | Single-human reference | Eski primary test |",
        "| Human-60 rewrite | Human-60 ile aynı görevler | 60 | Label transfer | Wording robustness |",
        "| Compositional | Evaluation domain cümleleri | 600 + 600 | Construction | Diagnostic test |",
        "| Independent paraphrase | 188 kabul edilen metin | 188 | Label transfer | External wording robustness |",
        f"| Expanded-v2 candidate | {expansion['total_subject_groups']} cihaz grubu / "
        f"{expansion['total_families']} senaryo ailesi | {expansion['total_rows']} | "
        "Construction, human review pending | Yeni ortak benchmark |",
        "",
        "Expanded-v2 split'i cihaz-grubu düzeyindedir: "
        f"`{expansion['row_counts']['train']}` train, "
        f"`{expansion['row_counts']['validation']}` validation, "
        f"`{expansion['row_counts']['test']}` test satırı. Validation/test yalnız yeni cihazlardan "
        "oluşur. Zorunlu sızıntı denetimleri: "
        + ", ".join(f"`{k}={v}`" for k, v in expansion["leakage_audit"].items())
        + ".",
        "",
        "İnsan annotation (yalın tasarım, 2 etiketleyici): rol sayıları "
        + ", ".join(f"`{k}={v}`" for k, v in annotation["role_counts"].items())
        + f"; etiketleyici başına `{annotation['tasks_per_annotator_including_pilot_and_repeats']}` "
        f"görev; B'ye ek `{annotation['annotator_b_extra_human60_tasks']}` Human-60 görevi.",
        "",
        "## 3. Yöntem × task-specific supervision matrisi",
        "",
        "| Yöntem | Eski task-specific exposure | Parametre öğrenimi | Yeni ortak koşul |",
        "|---|---|---|---|",
        "| Constant profile | 180 label | Basit profil | Expanded train |",
        "| Rule/keyword | Training yok; kurallar önceden yazılı | Yok | Aynı test; ayrı non-learning baseline |",
        "| TF-IDF + LR | D180 ve D1924 | Evet | Expanded train/validation/test |",
        "| Frozen multilingual-E5 + LR | D180 ve D1924 | Yalnız LR head | Expanded train/validation/test |",
        "| Fine-tuned multilingual-E5 | D1924 | Backbone + heads | Expanded train/validation/test |",
        "| Qwen2.5-1.5B 4-shot | 180 label vocabulary + 4 demo | Yok | Tarihsel prompt baseline |",
        "| Qwen3-8B 0/4/8-shot | Label vocabulary + k demo | Yok | Ayrı in-context track |",
        "| Qwen3-8B QLoRA | Henüz final eğitim yok | LoRA adapter (8 GB nedeniyle 4-bit taban) | Expanded-v2 train/validation/test |",
        "",
        "## 4. Mevcut Macro-F1 sonuçlarının tamamı",
        "",
        "| Yöntem | " + " | ".join(label for _, label in sets) + " |",
        "|---|" + "---:|" * len(sets),
    ]
    for model, label in methods:
        values = [
            _pct(_metric(point, model, set_name, "macro_f1_across_fields"))
            for set_name, _ in sets
        ]
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## 5. Human-60 original ayrıntılı metrikler",
            "",
            "| Yöntem | Exact-record | Macro-F1 | Privacy F1 | Abstention F1 | Schema validity | Latency ms/task |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model, label in methods:
        row = point[model]["sets"].get("human60_original")
        if row is None:
            continue
        privacy = row.get("critical_privacy", {}).get("f1")
        if model == "llm_fewshot_Qwen2.5-1.5B-Instruct":
            latency = old_llm["sets"]["human60_original"].get(
                "inference_latency_ms_mean"
            )
        else:
            authoritative_name = authoritative_names[model]
            latency = authoritative["point_estimates"][authoritative_name]["sets"][
                "human60_original"
            ].get("inference_latency_ms_mean")
        lines.append(
            f"| {label} | {_pct(row.get('exact_record_accuracy'))} | "
            f"{_pct(row.get('macro_f1_across_fields'))} | {_pct(privacy)} | "
            f"{_pct(row.get('abstention_f1'))} | {_pct(row.get('schema_validity'))} | "
            f"{_ms(latency)} |"
        )

    lines.extend(
        [
            "",
            "## 6. Veri artırmanın ölçülen etkisi",
            "",
            "| Model ailesi | Set | D180 Macro-F1 | D1924 Macro-F1 | Fark |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for base, augmented, label in (
        ("tfidf_lr", "tfidf_lr_augmented", "TF-IDF + LR"),
        ("e5_frozen_lr", "e5_frozen_lr_augmented", "Frozen E5 + LR"),
    ):
        for set_name, set_label in sets:
            left = _metric(point, base, set_name, "macro_f1_across_fields")
            right = _metric(point, augmented, set_name, "macro_f1_across_fields")
            if left is None or right is None:
                continue
            delta = 100 * (float(right) - float(left))
            lines.append(
                f"| {label} | {set_label} | {_pct(left)} | {_pct(right)} | "
                f"{delta:+.1f} puan |".replace(".", ",")
            )

    qwen_set_names = [name for name, _ in sets if name in qwen_shots[0]["sets"]]
    lines.extend(
        [
            "",
            "## 7. Qwen3-8B prompt-only 0/4/8-shot",
            "",
            "Aynı Qwen3-8B (4-bit NF4, 8 GB GPU), aynı label encoding ve greedy decoding; "
            "yalnız paired demonstrasyon sayısı değişir. Ağırlıklar güncellenmez.",
            "",
            "| Koşul | "
            + " | ".join(label for name, label in sets if name in qwen_set_names)
            + " | Ortalama gecikme | OOV alan |",
            "|---|" + "---:|" * (len(qwen_set_names) + 2),
        ]
    )
    for shots, result in qwen_shots.items():
        cells = [_pct(result["sets"][name]["macro_f1_across_fields"]) for name in qwen_set_names]
        latencies = [float(v["inference_latency_ms_mean"]) for v in result["sets"].values()]
        lines.append(
            f"| Qwen3-8B {shots}-shot | "
            + " | ".join(cells)
            + f" | {_ms(min(latencies) / 1000)}–{_ms(max(latencies) / 1000)} s"
            + f" | {result['out_of_vocabulary_fields']} |"
        )
    lines.extend(
        [
            "",
            "## 8. Eşit supervision geçici karşılaştırması (Expanded-v2, construction label)",
            "",
            "Bütün supervised yöntemler aynı Expanded-v2 train ile eğitildi; hiperparametreler "
            "v3'ten sabit; validation/test hiçbir seçimde kullanılmadı. v2 setlerinde CI "
            "cihaz-grubu cluster bootstrap'ıdır.",
            "",
        ]
    )
    v2_sets = list(provisional["set_sizes"])
    lines.append("| Yöntem | " + " | ".join(v2_sets) + " |")
    lines.append("|---|" + "---:|" * len(v2_sets))
    for model, values in provisional["point_estimates"].items():
        lines.append(
            f"| {model} | "
            + " | ".join(_pct(values["sets"][name]["macro_f1_across_fields"]) for name in v2_sets)
            + " |"
        )
    test_ci = provisional["intervals"]["v2_test"]
    lines.append("")
    for model in ("tfidf_lr_v2", "e5_frozen_lr_v2"):
        if model in test_ci:
            diff = test_ci[model]["macro_f1_across_fields"]["diff_vs_baseline"]
            text = (
                f"- `{model}` − rule (v2 test): {100 * diff['value']:+.1f} puan "
                f"[95% CI {100 * diff['ci95'][0]:+.1f}; {100 * diff['ci95'][1]:+.1f}]"
            )
            lines.append(text.replace(".", ","))
    lines.extend(["", "## 9. Qwen3-8B task-specific eğitim teknik kapısı", ""])
    if lora is not None:
        lines.append(
            f"- Tarihsel v1 smoke: `{lora['load_decision']['reason']}`; seçilen yükleme "
            f"`{lora['load_decision']['selected']}`; `{lora['optimizer_updates']}` update; "
            f"peak CUDA allocation `{float(lora['peak_vram_gib']):.2f}` GiB (8 GiB kartı aştı)."
        )
    lines.extend(
        [
            "- Düzeltmeler: hafif k-bit hazırlık (embedding/lm_head FP32'ye çevrilmez), 8-bit "
            "paged AdamW, reentrant olmayan gradient checkpointing; max length 320 ve cevap "
            "kesilirse eğitim durur (256 token v2'de 157 satırın cevabını kesiyordu).",
            "- Final eğitim insan etiketlemesi ve validation'da ayar dondurma sonrasıdır.",
        ]
    )
    lines.extend(
        [
            "",
            "## 10. Final karşılaştırma kapıları",
            "",
            "1. Candidate metinler naturalness ve label agreement açısından iki kör "
            "annotator tarafından etiketlenmeden benchmark dondurulmaz.",
            "2. Aile-gruplu train/validation/test ayrımı değiştirilemez; test hiçbir "
            "prompt, hyperparameter, epoch veya threshold seçiminde kullanılmaz.",
            "3. TF-IDF, frozen E5, fine-tuned E5 ve Qwen3-8B QLoRA aynı adjudicated "
            "training satırları ve aynı validation/test üzerinde çalıştırılır.",
            "4. Qwen3-8B 0/4/8-shot ayrı track olarak verilir; LoRA sonucuyla "
            "birleştirilmez.",
            "5. Macro-F1, exact-record, per-field F1, privacy/execution recall, schema "
            "validity, OOV/parse error, latency, VRAM ve training compute birlikte raporlanır.",
        ]
    )
    authoritative_count = authoritative["point_estimates"]["tfidf_lr_augmented"][
        "training_task_count"
    ]
    if authoritative_count != 1924:
        raise ValueError("authoritative augmented count changed unexpectedly")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"built {OUTPUT}")


if __name__ == "__main__":
    main()
