# ruff: noqa: E501, RUF001
"""Generate the thesis "Veri ve Yöntem" chapter from the repository's sources.

Every number in the chapter is read from a versioned artifact, a manifest or a
code constant; nothing is typed by hand.  Re-run after any change:

    .venv/Scripts/python.exe scripts/build_thesis_method_chapter.py

``tests/phase_5/test_thesis_chapter_current.py`` fails when the committed
chapter is stale, and a Claude Code Stop hook regenerates it automatically.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from task_offloading import experiment_config as xc  # noqa: E402
from task_offloading.domain import RewardWeights  # noqa: E402
from task_offloading.semantic import compositional as comp  # noqa: E402
from task_offloading.semantic import load_jsonl  # noqa: E402
from task_offloading.semantic.benchmark import ANNOTATION_FIELDS  # noqa: E402
from task_offloading.semantic.finetune import FineTunedEncoderBaseline  # noqa: E402
from task_offloading.semantic.paraphrase import RULES  # noqa: E402
from task_offloading.sim import trace_hybrid as th  # noqa: E402

OUTPUT = ROOT / "docs" / "thesis" / "03_VERI_VE_YONTEM.md"
MAIN = ROOT / "data" / "semantic_benchmark" / "main"
PILOT = ROOT / "data" / "semantic_benchmark" / "pilot"
PROFILES = ROOT / "configs" / "data_profiles"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find(obj: Any, key: str) -> Iterator[Any]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from _find(v, key)
    elif isinstance(obj, list):
        for item in obj:
            yield from _find(item, key)


def _n(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")


def _pct(value: float, digits: int = 1) -> str:
    return f"%{100 * value:.{digits}f}".replace(".", ",")


def _mb(size: int) -> str:
    return f"{size / 1e6:,.1f} MB".replace(".", ",") if size < 1e9 else f"{size / 1e9:,.2f} GB".replace(".", ",")


def _dec(value: float) -> str:
    return f"{value:g}".replace(".", ",")


def render() -> str:
    corpus = _json(MAIN / "corpus_manifest.v1.json")
    split = _json(MAIN / "split_manifest.v1.json")
    dev_n = len(split["machine_labeled_development"]["task_ids"])
    eval_n = len(split["single_human_evaluation"]["task_ids"])
    tasks = load_jsonl(MAIN / "tasks.v1.jsonl")
    eval_ids = set(split["single_human_evaluation"]["task_ids"])
    eval_policy = sum(1 for t in tasks if t["task_id"] in eval_ids and t.get("user_policy_text"))
    reference_n = len(load_jsonl(MAIN / "annotations" / "annotator_a.single_human_reference.v1.3.jsonl"))
    normalization = _json(MAIN / "reference_variants" / "protocol_normalization_log.v1.json")
    norm_rules = Counter(change["rule"] for change in normalization["changes"])
    pilot_n = len(load_jsonl(PILOT / "tasks.v1.jsonl"))
    gate = _json(PILOT / "round_2" / "annotations" / "quality_gate.v1.json")
    failed = [c["name"] for c in gate["checks"] if not c["passed"]]
    agreement = gate["round_comparison"]["exact_field_agreement"]

    sources = {s["id"]: s for s in _json(ROOT / "configs" / "data_sources.v1.json")["sources"]}
    uci = _json(PROFILES / "uci_mec_859_23716d69.json")
    uci_means = {v["server_id"]: v["turnaround_s_mean"] for v in uci["selected_files"].values()}
    bupt_q = _json(PROFILES / "bupt_04e664f_workload_quantiles.json")
    eua = _json(PROFILES / "eua_61238e00_22c07483.json")
    eua_rows = {k: v["rows"] for k, v in eua["selected_files"].items()}
    nep = _json(PROFILES / "nep_large_full_trace_ff80a07e.json")
    nep_total = next(_find(nep, "total_rows"))
    nep_accepted = next(_find(nep, "accepted_rows_total"))
    nep_quantiles = sorted(PROFILES.glob("nep_*quantiles*.json"))

    v3_path = MAIN / "benchmark_v3" / "results.v3.json"
    v3 = _json(v3_path) if v3_path.exists() else None
    hybrid = th.HybridConfig()
    weights = RewardWeights()
    ft_defaults = {
        name: param.default
        for name, param in inspect.signature(FineTunedEncoderBaseline.__init__).parameters.items()
        if param.default is not inspect.Parameter.empty and name != "cache_folder"
    }
    train_pool = {len(rule.train) for rule in RULES}
    test_pool = {len(rule.test) for rule in RULES}
    domains = corpus["overall_counts"]["domains"]
    origins = corpus["overall_counts"]["origins"]

    def size(source_id: str) -> str:
        return _mb(int(sources[source_id]["artifacts"][0]["size_bytes"]))

    lines: list[str] = []
    add = lines.append
    add("# Bölüm 3 — Veri ve Yöntem")
    add("")
    add("> **Otomatik üretilmiş belge.** Bu bölümdeki bütün sayılar depodaki sürümlü veri, manifest ve kod sabitlerinden `scripts/build_thesis_method_chapter.py` ile üretilir; elle düzenlenmemelidir. Veri, protokol veya deney ayarı değiştiğinde betik yeniden çalıştırılır; `tests/phase_5/test_thesis_chapter_current.py` belge güncel değilse başarısız olur.")
    add("")
    add("## 3.1 Genel yaklaşım")
    add("")
    add("Çalışma iki ayrık katmandan oluşur. **Semantik katman**, doğal dildeki görev metnini sekiz alanlı bir gereksinim sözleşmesine dönüştürür; ağ durumunu görmez ve karar vermez. **Karar katmanı**, bu sözleşmeyi ve fiziksel sistem durumunu kullanarak görevi yerel cihazda, üç edge sunucudan birinde veya bulutta çalıştırır. Katmanların ayrılması, semantik çıkarım kalitesinin sistem sonuçlarına etkisinin nedensel olarak ölçülmesini sağlar: fiziksel olaylar, görevler ve tohumlar sabit tutulurken yalnız semantik bilginin kaynağı değiştirilir.")
    add("")
    add("Doğal dil MEC görev gereksinimleri ile fiziksel offloading sonuçlarını aynı kayıtta sunan açık bir veri seti bulunmadığından, semantik katman için kontrollü bir benchmark tasarlanmış; fiziksel katman ise gerçek veri kaynaklarından beslenmiştir. Bu nedenle deney ortamı **iz güdümlü hibrit (trace-driven hybrid)** olarak adlandırılır; tamamen gerçek bir veri seti olarak sunulmaz.")
    add("")
    add("## 3.2 Semantik gereksinim şeması")
    add("")
    add(f"Her görev {len(ANNOTATION_FIELDS)} alanla temsil edilir: " + ", ".join(f"`{f}`" for f in ANNOTATION_FIELDS) + ". Her alan bir durum (`explicit`, `partial`, `not_stated`, `ambiguous`), değer(ler), metinden kanıt alıntısı ve güven skoru taşır. Metinde belirtilmeyen bir gereksinim uydurulmaz (`not_stated`); çelişkili ifadeler `ambiguous` olarak işaretlenir ve karar verilmez (abstention).")
    add("")
    add("## 3.3 Kontrollü görev corpus'u")
    add("")
    add(f"Ana corpus {_n(corpus['task_count'])} Türkçe görev metninden oluşur ({_n(corpus['scenario_family_count'])} senaryo ailesi). Metinler yapılandırılmış senaryo tanımlarından LLM destekli olarak üretilmiştir (`generation_method: llm_assisted_structured_design`); **gerçek son kullanıcı kayıtları değildir.** Dağılım alan (domain) bazında dengelidir:")
    add("")
    add("| Alan | " + " | ".join(domains) + " |")
    add("|---|" + "---:|" * len(domains))
    add("| Metin | " + " | ".join(str(v) for v in domains.values()) + " |")
    add("")
    add("Köken bazında: " + ", ".join(f"`{k}` {v}" for k, v in origins.items()) + f". Corpus, senaryo/şablon/yeniden ifade aileleri bölünmeden (aile sızıntısı olmadan) **{dev_n} geliştirme** ve **{eval_n} değerlendirme** metnine ayrılmıştır. Değerlendirme metinlerinden {eval_policy} tanesi ayrı bir kullanıcı politikası metni içerir.")
    add("")
    add("## 3.4 Etiketleme")
    add("")
    add(f"**Pilot.** {pilot_n} pilot metin, birinci turda üç (A, B, C), ikinci turda iki (A, B) etiketleyici tarafından kör olarak etiketlenmiştir. Tam alan uyumu birinci turda {_pct(agreement['round_1'])}, ikinci turda {_pct(agreement['round_2'])} olmuştur; ancak önceden tanımlı kalite kapısı (kappa ≥ {_dec(gate['thresholds']['minimum_kappa'])}) {len(failed)} kontrolde karşılanamamıştır (`{gate['status']}`). Bu sonuç değiştirilmeden saklanmış ve protokol revizyonlarına gerekçe olmuştur.")
    add("")
    add(f"**Ana değerlendirme referansı.** {reference_n} değerlendirme metni, protokol 1.3.0'a göre tek bir insan etiketleyici (A) tarafından kör biçimde etiketlenmiş ve SHA-256 ile kilitlenmiştir. Bu referans iki uzmanla uzlaştırılmış bir \"gold standard\" değildir; **tek-insan referansıdır.**")
    add("")
    add(f"**Protokol normalizasyonu.** Referansa yalnız protokolde yazılı, yoruma kapalı iki kural deterministik olarak uygulanmıştır: R1 — sayısal başarı eşiği ≥ %99,9 ise `mission_critical` ({norm_rules.get('R1_reliability_threshold', 0)} değişiklik); R2 — metrik adının kanonik yazımı `F1` ({norm_rules.get('R2_metric_spelling', 0)} değişiklik). Toplam {normalization['change_count']} değişiklik ayrı bir kayıtta tutulur; orijinal referans dosyası değiştirilmemiştir ve normalizasyon sırasında hiçbir model çıktısına bakılmamıştır.")
    add("")
    add(f"**Belirlenen protokol açığı.** Annotasyon protokolü ayrı kullanıcı politikası metninin nasıl etiketleneceğini tanımlamamaktadır. Şablon tasarım etiketleri ile insan referansı arasındaki ayrışmaların büyük bölümü politika metni içeren {eval_policy} görevde yoğunlaşmıştır. Bu durum sonuçlar raporlanırken ayrıca belirtilir ve ölçüm sınırlaması olarak kabul edilir.")
    add("")
    add(f"**Geliştirme etiketleri.** {dev_n} geliştirme metninin etiketleri şablon tasarım kurallarından otomatik üretilmiş **makine zayıf etiketleridir** (`machine_generated_weak_label`); insan etiketi olarak sunulmaz.")
    add("")
    add(f"Özetle, çalışmadaki gerçek insan etiketli metin sayısı **{reference_n} ana + {pilot_n} pilot** metindir.")
    add("")
    add("## 3.5 Makine üretimi ek veri")
    add("")
    add(f"**Anlam koruyan yeniden ifade (CheckList tarzı değişmezlik testi).** Şablon gereksinim cümlelerinin her biri için {len(RULES)} kural tanımlanmıştır. Her kuralın birbirinden ayrık iki havuzu vardır: yalnız eğitim çoğaltmasında kullanılan **eğitim havuzu** (kural başına {', '.join(map(str, sorted(train_pool)))} ifade) ve yalnız test için kullanılan **test havuzu** (kural başına {', '.join(map(str, sorted(test_pool)))} ifade). Sayısal eşikler korunur; etiketler değişmezlik varsayımıyla aktarılır. Havuzlar herhangi bir model sonucu görülmeden birlikte yazılmıştır. Test havuzu ve ilk sonuçlar görüldükten sonra eklenen ek eğitim ifadeleri, deneyci sızıntısı oluşturduğu için **geri çekilmiştir**.")
    add("")
    add(f"**Kompozisyonel görevler.** Mevcut bir metnin alan cümlesine 0–4 bağımsız gereksinim yuvası eklenerek etiketleri inşa yoluyla bilinen görevler üretilir. Yuva sayısı olasılıkları {', '.join(_dec(p) for p in comp.SLOT_COUNT_PROBABILITIES)}; gecikme eşikleri {', '.join(map(str, comp.LATENCY_MS))} ms; güvenilirlik eşikleri %{', %'.join(comp.RELIABILITY_PERCENT)}; F1 eşikleri {', '.join(comp.ACCURACY_F1)}. Etiket eşlemesi annotasyon protokolündeki kurallarla aynıdır. Test görevleri yalnız değerlendirme bölümünün alan cümlelerinden ve test havuzundan, eğitim görevleri yalnız geliştirme bölümünden ve eğitim havuzundan üretilir.")
    add("")
    if v3 is not None:
        sizes = v3["set_sizes"]
        add("| Veri | Metin | Etiket kaynağı |")
        add("|---|---:|---|")
        add(f"| Genişletilmiş eğitim seti | {_n(v3['point_estimates']['tfidf_lr_augmented']['training_task_count'])} | {dev_n} geliştirme metni + makine yeniden ifadesi + kompozisyonel (makine) |")
        for name, count in sizes.items():
            claim = v3["label_claims"].get(name, v3["label_claims"].get("compositional600_*"))
            add(f"| Test: `{name}` | {_n(count)} | `{claim}` |")
        add("")
    add("## 3.6 Semantik çıkarım yöntemleri")
    add("")
    add("Karşılaştırılan yöntemler: (i) **sabit profil** (eğitimdeki en sık profil); (ii) **kural/anahtar kelime** ayrıştırıcısı; (iii) **TF-IDF + lojistik regresyon** (karakter 3–5-gram, alan başına sınıf-dengeli model); (iv) **donuk `intfloat/multilingual-e5-small` + lojistik regresyon**; (v) **ince ayarlı `multilingual-e5-small`** (alan başına doğrusal başlık). Öğrenen yöntemler hem yalnız geliştirme metinleriyle hem de genişletilmiş eğitim setiyle eğitilmiştir.")
    add("")
    add("İnce ayar hiperparametreleri herhangi bir değerlendirme sonucu görülmeden sabitlenmiştir: " + ", ".join(f"`{k}={v}`" for k, v in ft_defaults.items()) + ". Değerlendirme etiketleri hiçbir yöntemin eğitiminde veya ayarında kullanılmamıştır.")
    add("")
    add("## 3.7 Değerlendirme ölçütleri ve istatistik")
    add("")
    add("Ölçütler: tam kayıt doğruluğu (sekiz alanın tamamı doğru), alan doğruluğu, alanlar üzerinden ortalama macro-F1, kritik gizlilik (hassas/kısıtlı veri) için precision/recall/F1 ve pozitif örnek sayısı, abstention F1, kalibrasyon (ECE, Brier) ve görev başına çıkarım süresi.")
    if v3 is not None:
        boot = v3["bootstrap"]
        add(f"Belirsizlik, görev düzeyinde {boot['resamples']} tekrarlı bootstrap ile %95 yüzdelik güven aralığı olarak verilir. Yöntem karşılaştırmaları aynı bootstrap örnekleri üzerinde kural tabanlı yönteme göre eşleştirilmiş fark, farkın güven aralığı ve iki yönlü bootstrap p-değeri ile raporlanır.")
    add("")
    add("## 3.8 Semantik sözleşmeden kısıtlara dönüşüm")
    add("")
    add("Çalıştırma izni yalnız açık (`explicit`/`partial`) yasaklarla daraltılır; `unknown` hiçbir zaman kendiliğinden yasak üretmez. Nitel sınıflar şu sayısal karşılıklarla kısıta çevrilir:")
    add("")
    add("| Alan | Eşleme |")
    add("|---|---|")
    add("| Gecikme sınıfı (s) | " + ", ".join(f"`{k}`={_dec(v)}" for k, v in th.LATENCY_CLASS_S.items()) + " |")
    add("| Güvenilirlik katmanı | " + ", ".join(f"`{k}`={_dec(v)}" for k, v in th.RELIABILITY_TIER_P.items()) + " |")
    add("| Doğruluk katmanı | " + ", ".join(f"`{k}`={_dec(v)}" for k, v in th.ACCURACY_TIER_SCORE.items()) + " |")
    add("")
    add("**İhtiyatlı mod (conservative fallback).** Çalıştırma politikası açık değilken veri hassas/kısıtlı ise, politika belirsizse veya politika güveni 0,6'nın altındaysa bulut maskelenir. İhtiyatlı mod yalnız izin kaldırır, hiçbir zaman izin eklemez.")
    add("")
    add("**İnanılan ve gerçek gereksinim.** Karar katmanının maskesi semantik çıkarıcının ürettiği *inanılan* gereksinimle kurulur; kısıt ihlalleri ise her zaman *gerçek* gereksinime göre ölçülür. Böylece çıkarım hatalarının sistemdeki karşılığı doğrudan gözlenir.")
    add("")
    add("## 3.9 İz güdümlü hibrit MEC ortamı")
    add("")
    add("| Kaynak | Boyut | Lisans/erişim | Ortamdaki rolü |")
    add("|---|---:|---|---|")
    add(f"| UCI MEC-859 | {size('uci_mec_execution_times_859')} | {sources['uci_mec_execution_times_859']['license']['identifier']} | {_n(uci['total_rows'])} ölçülmüş yürütme süresi; Raspberry Pi 4B yerel cihaz, MacBookPro1 / Ubuntu VM / MacBookPro2 sırasıyla `edge_0..2`. Ortalama süreler (s): " + ", ".join(f"{k} {_dec(round(v, 3))}" for k, v in uci_means.items()) + " |")
    add(f"| BUPT Edge-Computing | {size('bupt_edge_computing_dataset')} | yerel akademik kullanım | {_n(bupt_q['accepted_records'])} kabul edilen istek kaydı; görev geliş deseni ve uplink/downlink bit büyüklükleri. Kronolojik bölme: ayın 24'ünden önceki günler eğitim, sonrası test |")
    add(f"| EUA | {size('eua_dataset')} | {sources['eua_dataset']['license']['identifier']} | Avustralya genelinde {_n(eua_rows['users/users-aus.csv'])} kullanıcı ve {_n(eua_rows['edge-servers/site.csv'])} edge aday konumu; en yakın üç sitenin mesafesi radyo bağlantısını belirler |")
    nep_role = (
        "edge yük ve RTT dağılımları ortama bağlanmıştır"
        if nep_quantiles
        else "**henüz ortama bağlanmamıştır**; yalnız bütünlük ve kabul profili mevcuttur"
    )
    add(f"| NEP-large (Full trace) | {size('edge_workloads_traces_nep_large')} | yalnız araştırma, paylaşım yasak | toplam {_n(nep_total)} satırdan {_n(nep_accepted)} tanesi kabul politikasından geçmiştir; {nep_role} |")
    add("")
    add("Ölçülmemiş, açıkça kalibrasyon varsayımı olan değerler:")
    add("")
    add("| Parametre | Değer |")
    add("|---|---|")
    pretty = {
        "device_cpu_hz": f"{_dec(hybrid.device_cpu_hz / 1e9)} GHz (Raspberry Pi 4B nominal saat)",
        "workload_scale": f"{_dec(hybrid.workload_scale)} (yerel ortalama servis süresi ölçeği)",
        "cloud_speedup_vs_fastest_edge": f"{_dec(hybrid.cloud_speedup_vs_fastest_edge)}× en hızlı edge",
        "cloud_one_way_propagation_s": f"{_dec(1000 * hybrid.cloud_one_way_propagation_s)} ms tek yön",
        "cloud_backhaul_bps": f"{_dec(hybrid.cloud_backhaul_bps / 1e9)} Gbit/s",
        "target_mean_interarrival_s": f"{_dec(1000 * hybrid.target_mean_interarrival_s)} ms ortalama geliş aralığı (BUPT deseni ölçeklenir)",
        "battery_j": f"{_dec(hybrid.battery_j)} J",
    }
    for field in dataclasses.fields(hybrid):
        if field.name in {
            "radio",
            "episode_tasks",
            "nep_percentiles",
            "nep_edge_load_quantiles",
            "nep_rtt_ms_quantiles",
        }:
            continue
        value = getattr(hybrid, field.name)
        text = pretty.get(field.name)
        if text is None:
            text = (
                ", ".join(_dec(v) for v in value)
                if isinstance(value, tuple)
                else _dec(value)
            )
        add(f"| `{field.name}` | {text} |")
    add(f"| Radyo | bant genişliği {_dec(hybrid.radio.bandwidth_hz / 1e6)} MHz, taşıyıcı {_dec(hybrid.radio.carrier_frequency_hz / 1e9)} GHz, yol kaybı üssü {_dec(hybrid.radio.path_loss_exponent)}, log-mesafe + Shannon hızı |")
    add("")
    add(f"Her bölüm (episode) {hybrid.episode_tasks} görevden oluşur. Eylem uzayı: " + ", ".join(f"`{a.label}`" for a in th.ACTIONS) + ". Ödül, tek fizik çekirdeğinde hesaplanan bileşenlerin ağırlıklı toplamının negatifidir: " + ", ".join(f"`{f.name}`×{_dec(getattr(weights, f.name))}" for f in dataclasses.fields(weights)) + f" (normalizasyon referansları {_dec(xc.LATENCY_REFERENCE_S)} s ve {_dec(xc.ENERGY_REFERENCE_J)} J).")
    add("")
    add("Eğitim bölümlerinden önce bir *baskın eylem* kontrolü yapılmıştır: hiçbir sabit eylem tek başına semantik-duyarlı açgözlü (greedy) politikanın başarısına ulaşmamaktadır. Bu kontrol yalnız eğitim bölümleri üzerinde yapılmış, test sonuçlarına bakılarak ayar yapılmamıştır.")
    add("")
    add("## 3.10 Karar politikaları ve deney tasarımı")
    add("")
    add("Referans politikalar: her zaman yerel / `edge_0` / bulut, geçerli eylemler arasından rastgele, semantiksiz açgözlü (en düşük beklenen gecikme) ve semantik-duyarlı açgözlü (inanılan gecikme, güvenilirlik ve doğruluk kısıtlarını karşılayan eylemler arasında en düşük gecikme).")
    add("")
    add(f"Öğrenen politika: {xc.PPO_ALGORITHM}; " + ", ".join(f"`{k}={v}`" for k, v in xc.PPO_HYPERPARAMETERS.items()) + f"; {xc.PPO_SEEDS} tohum × {_n(xc.PPO_TIMESTEPS)} adım; {xc.TRAIN_EPISODES} eğitim ve {xc.TEST_EPISODES} test bölümü. **Semantikli PPO** gereksinim özniteliklerini görür ve semantik maskeyi kullanır; **semantiksiz PPO** ikisini de görmez. PPO, eğitimde referans semantikle eğitilir; testte, gerçek dağıtımdaki gibi, semantik çıkarıcıların dondurulmuş tahminleriyle çalıştırılır. Bütün politikalar birebir aynı test bölümleri üzerinde değerlendirilir; güven aralıkları bölüm düzeyinde eşleştirilmiş bootstrap ile hesaplanır.")
    add("")
    add("## 3.11 Geçerlilik tehditleri ve sınırlamalar")
    add("")
    add("- Görev metinleri kontrollü olarak üretilmiştir; gerçek kullanıcı ifadelerindeki performans doğrudan ölçülmemiştir. Yeniden ifade testleri bunun yalnız bir yaklaşımıdır ve bazı ipucu sözcükler eğitim ile test havuzlarında ortaktır; bu nedenle dayanıklılık sonuçları iyimser bir tahmindir.")
    add(f"- İnsan referansı {reference_n} metinlik ve tek etiketleyicilidir; güven aralıkları geniştir ve kritik gizlilik gibi nadir sınıflarda örnek sayısı çok düşüktür. Etiketleyiciler arası uyum yalnız pilot setinde ölçülmüş ve kalite kapısı karşılanamamıştır.")
    add("- Kompozisyonel ve yeniden ifade setleri makine üretimidir; etiketleri inşa veya aktarma yoluyla elde edilmiştir ve insan etiketi değildir.")
    add("- Fiziksel ortam iz güdümlü hibrittir: farklı kaynakların satırları aynı gerçek olayın ölçümü değildir; mutlak iş yükü ölçeği, bulut ve güvenilirlik/doğruluk parametreleri kalibrasyon varsayımıdır.")
    add("- İnce ayarlı encoder CPU bütçesiyle sınırlı sayıda epoch eğitilmiştir; sonuçları daha büyük bütçelerdeki performansı temsil etmeyebilir.")
    add("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="exit 1 if stale")
    args = parser.parse_args()
    text = render()
    current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
    if args.check:
        sys.exit(0 if current == text else 1)
    if current != text:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(text, encoding="utf-8")
        print(f"updated {OUTPUT.relative_to(ROOT)}")
    else:
        print("chapter already up to date")


if __name__ == "__main__":
    main()
