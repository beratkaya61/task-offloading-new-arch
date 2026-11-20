# ruff: noqa: E501, RUF001
"""Package the semantic benchmark (train/test) for the advisor and the paper.

Output: ``release/semantic_offloading_benchmark_v1/`` plus a ZIP next to it.
Every record carries its label provenance.  Raw physical traces are NOT
included (NEP licence forbids redistribution; BUPT has no redistribution
licence); only aggregate profiles and their sources are listed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from task_offloading.semantic import load_jsonl
from task_offloading.semantic.baselines import semantic_signature
from task_offloading.semantic.benchmark import ANNOTATION_FIELDS
from task_offloading.semantic.experiment import load_json_object, ordered_subset

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "data" / "semantic_benchmark" / "main"
PILOT = ROOT / "data" / "semantic_benchmark" / "pilot"
V3 = MAIN / "benchmark_v3"
OUT = ROOT / "release" / "semantic_offloading_benchmark_v1"


def _row(task: dict[str, Any], annotation: dict[str, Any], split: str, claim: str) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "split": split,
        "label_provenance": claim,
        "task_text": task["task_text"],
        "user_policy_text": task.get("user_policy_text"),
        "labels": {f: annotation[f] for f in ANNOTATION_FIELDS},
    }


def _flat(row: dict[str, Any]) -> dict[str, Any]:
    flat = {k: row[k] for k in ("task_id", "split", "label_provenance", "task_text", "user_policy_text")}
    for field in ANNOTATION_FIELDS:
        signature = semantic_signature(row["labels"], field)
        flat[field] = "|".join("" if v is None else str(v) for v in signature)
    for key, field, attr in (
        ("latency_max_ms", "latency", "max_ms"),
        ("reliability_min_success", "reliability", "min_success_probability"),
        ("accuracy_min_score", "accuracy", "min_score"),
    ):
        flat[key] = row["labels"][field].get(attr)
    return flat


def _write(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    jsonl = OUT / f"{name}.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    csv_path = OUT / f"{name}.csv"
    flat_rows = [_flat(r) for r in rows]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    return {
        "file": f"{name}.jsonl",
        "csv": f"{name}.csv",
        "records": len(rows),
        "sha256": hashlib.sha256(jsonl.read_bytes()).hexdigest(),
        "label_provenance": sorted({r["label_provenance"] for r in rows}),
    }


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    split = load_json_object(MAIN / "split_manifest.v1.json")
    tasks = load_jsonl(MAIN / "tasks.v1.jsonl")
    dev = ordered_subset(tasks, list(split["machine_labeled_development"]["task_ids"]))
    evaluation = ordered_subset(tasks, list(split["single_human_evaluation"]["task_ids"]))
    weak = {r["task_id"]: r["annotation"] for r in load_jsonl(MAIN / "preannotations" / "machine_development.v1.jsonl")}
    human = {r["task_id"]: r["annotation"] for r in load_jsonl(MAIN / "annotations" / "annotator_a.single_human_reference.v1.3.jsonl")}
    normalized = {r["task_id"]: r["annotation"] for r in load_jsonl(MAIN / "reference_variants" / "annotator_a.protocol_normalized.v1.3.1.jsonl")}
    manifest: dict[str, Any] = {"release": "semantic_offloading_benchmark_v1", "files": {}}

    manifest["files"]["train/development_180"] = _write(
        "train/development_180",
        [_row(t, weak[t["task_id"]], "train", "machine_generated_weak_label") for t in dev],
    )
    aug_tasks = load_jsonl(V3 / "train_augmented_tasks.jsonl")
    aug_labels = load_jsonl(V3 / "train_augmented_labels.jsonl")
    aug_rows = []
    for index, (task, label) in enumerate(zip(aug_tasks, aug_labels, strict=True)):
        origin = task.get("provenance", {}).get("origin", "")
        claim = (
            "machine_generated_label_by_construction"
            if origin == "compositional_machine_generated"
            else "machine_generated_weak_label"
        )
        row = _row(task, label["annotation"], "train", claim)
        row["task_id"] = f"{task['task_id']}#r{index}"
        aug_rows.append(row)
    manifest["files"]["train/augmented_1924"] = _write("train/augmented_1924", aug_rows)
    cal_tasks = load_jsonl(V3 / "calibration_tasks.jsonl")
    cal_labels = {r["task_id"]: r["annotation"] for r in load_jsonl(V3 / "calibration_labels.jsonl")}
    manifest["files"]["calibration/calibration_set"] = _write(
        "calibration/calibration_set",
        [_row(t, cal_labels[t["task_id"]], "calibration", "machine_generated_label_by_construction") for t in cal_tasks],
    )
    manifest["files"]["test/human60_original_protocol_normalized"] = _write(
        "test/human60_original_protocol_normalized",
        [_row(t, normalized[t["task_id"]], "test", "single_human_reference_protocol_normalized") for t in evaluation],
    )
    manifest["files"]["test/human60_original_raw_human_reference"] = _write(
        "test/human60_original_raw_human_reference",
        [_row(t, human[t["task_id"]], "test", "single_human_reference_raw_protocol_1_3") for t in evaluation],
    )
    para60 = load_jsonl(V3 / "eval_human60_paraphrase.tasks.jsonl")
    manifest["files"]["test/human60_machine_paraphrase"] = _write(
        "test/human60_machine_paraphrase",
        [_row(t, normalized[t["task_id"]], "test", "machine_paraphrase_human_label_transferred") for t in para60],
    )
    for name in ("compositional600_canonical", "compositional600_paraphrase", "independent_paraphrase"):
        path = V3 / f"eval_{name}.tasks.jsonl"
        if not path.exists():
            continue
        labels = {r["task_id"]: r["annotation"] for r in load_jsonl(V3 / f"eval_{name}.labels.jsonl")}
        claim = (
            "machine_llm_paraphrase_label_by_construction_transferred"
            if name == "independent_paraphrase"
            else "machine_generated_label_by_construction"
        )
        manifest["files"][f"test/{name}"] = _write(
            f"test/{name}", [_row(t, labels[t["task_id"]], "test", claim) for t in load_jsonl(path)]
        )
    pilot_tasks = load_jsonl(PILOT / "tasks.v1.jsonl")
    for annotator in ("a", "b", "c"):
        rows = {r["task_id"]: r["annotation"] for r in load_jsonl(PILOT / "annotations" / "round_1" / f"annotator_{annotator}.jsonl")}
        manifest["files"][f"pilot/round1_annotator_{annotator}"] = _write(
            f"pilot/round1_annotator_{annotator}",
            [_row(t, rows[t["task_id"]], "pilot", f"human_annotator_{annotator}_round1") for t in pilot_tasks if t["task_id"] in rows],
        )
    shutil.copy(ROOT / "schemas" / "semantic_requirements.schema.json", OUT / "semantic_requirements.schema.json")
    profiles = OUT / "physical_trace_profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    for name in ("bupt_04e664f_workload_quantiles.json", "uci_mec_859_23716d69.json", "nep_quantiles_v1.json", "eua_61238e00_22c07483.json"):
        source = ROOT / "configs" / "data_profiles" / name
        if source.exists():
            shutil.copy(source, profiles / name)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(DATASHEET, encoding="utf-8")
    archive = shutil.make_archive(str(OUT), "zip", OUT.parent, OUT.name)
    print(f"wrote {OUT} and {archive}")
    for key, value in manifest["files"].items():
        print(f"{key:48s} {value['records']:5d}  {', '.join(value['label_provenance'])}")


DATASHEET = """# Semantic Offloading Benchmark v1 — Veri Kartı (Datasheet)

## Amaç
Türkçe doğal dil IoT/MEC görev metinlerinden sekiz alanlı gereksinim sözleşmesi
(domain, latency, privacy, execution_policy, reliability, accuracy,
energy_priority, divisibility) çıkarımını ve bu çıkarımın offloading
kararlarına etkisini değerlendirmek.

## Klasörler
- `train/development_180` — 180 kontrollü metin; makine zayıf etiketi.
- `train/augmented_1924` — 180 metin + makine yeniden ifadesi + kompozisyonel
  görevler; etiketler makine zayıf etiketi veya inşa yoluyla.
- `calibration/calibration_set` — yalnız güven kalibrasyonu için; inşa yoluyla etiketli.
- `test/human60_original_protocol_normalized` — **ana insan referansı** (tek
  etiketleyici, kör; protokolde yazılı 5 deterministik düzeltme uygulanmış).
- `test/human60_original_raw_human_reference` — aynı 60 metnin düzeltilmemiş insan etiketi.
- `test/human60_machine_paraphrase` — 60 metnin makine yeniden yazımı; insan
  etiketi değişmezlik varsayımıyla aktarılmıştır (insan bu metinleri görmemiştir).
- `test/compositional600_*` — makine üretimi, inşa yoluyla etiketli test görevleri
  (şablon ifadesi ve eğitimde görülmemiş ifade).
- `test/independent_paraphrase` — bağımsız bir LLM ile yeniden yazılmış
  kompozisyonel görevler; etiketler aktarılmıştır.
- `pilot/` — 20 pilot metnin üç insan etiketleyicisi (A, B, C) tarafından yapılan etiketleri.
- `physical_trace_profiles/` — fiziksel izlerin yalnız özet istatistikleri.

## Etiket kaynağı (`label_provenance`)
Her kayıtta açıkça yazılıdır. Gerçek insan etiketli metinler yalnızca
`test/human60_*_reference` (60) ve `pilot/` (20) kayıtlarıdır; diğer bütün
etiketler makine üretimidir ve insan etiketi olarak sunulmamalıdır.

## Biçim
Her bölüm JSONL (tam yapı) ve CSV (Excel ile açılabilir, alan başına
`status|değer` biçiminde) olarak verilir. Şema: `semantic_requirements.schema.json`.

## Sınırlamalar
Metinler kontrollü olarak üretilmiştir; gerçek kullanıcı kayıtları değildir.
İnsan referansı tek etiketleyicilidir. Ham fiziksel izler (NEP, BUPT) lisans
nedeniyle dahil değildir; kaynakları ve SHA-256 değerleri proje deposundaki
`configs/data_sources.v1.json` dosyasındadır.
"""


if __name__ == "__main__":
    main()
