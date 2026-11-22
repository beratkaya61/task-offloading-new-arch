# Faz 6 Kapanış Raporu — PPO ve Semantic Ablation

**Kapanış tarihi:** 21 Kasım 2026

**Kapsam:** NEP-v2 exact/noisy preview, semantic source, confidence fallback,
action-mask/penalty ve independent-188 LLM ablation'ları

**Karar:** Ön kayıtlı H1–H6 kararları üretildi; H1, H2 ve H6 desteklendi, H3–H5
desteklenmedi. Negatif sonuçlar gizlenmeden ana yayın bulgusu olarak korunacaktır.

## 1. Araştırma sorusu

Bu faz şu soruyu test eder: Natural-language task intent'ten önceden çıkarılmış
structured semantic requirements, aynı MEC fiziği ve aynı evaluation events
altında no-semantic policy'ye göre success ve hard-constraint satisfaction'ı
iyileştiriyor mu?

LLM doğrudan local/edge/cloud action üretmez. Extractor yalnız görev ve kullanıcı
politika metnini görür; network state, reward, oracle ve correct action görmez.
Tahmin cache'i PPO'dan önce dondurulur. Böylece uçtan uca fark, serbest biçimli
LLM scheduler iddiasından ziyade semantic requirement layer'ın katkısıdır.

## 2. Mimari ve ablation tasarımı

```text
Task text + user policy
          |
          v
Frozen semantic extractor/cache
(rule / TF-IDF / E5 / Qwen / single-human reference)
          |
          v
Structured requirements + confidence/abstention
          |
          +--> action mask veya penalty ablation
          |
          v
PPO-MLP / greedy policy --> shared deterministic transition core
```

Karşılaştırılan ana koşullar:

- no-semantic;
- rule parser;
- augmented TF-IDF + logistic regression;
- frozen multilingual-E5 + logistic heads;
- Qwen2.5-1.5B 4-shot raw cache;
- `single_human_reference` erişilebilir semantic upper bound;
- calibrated confidence + fixed fallback;
- hard action mask ve penalty-only ablation;
- exact-preview ve σ=0,5 noisy-preview.

## 3. Confidence ve veri izolasyonu

Calibration yalnız development/calibration kayıtlarında isotonic regression ile
fit edilmiştir. Human-60, independent-188 ve compositional evaluation labels
calibrator veya threshold seçimine açılmamıştır. Fallback threshold `0,6` olarak
önceden sabittir; evaluation set üzerinde optimize edilmemiştir.

Calibrated ve raw prediction provenance ayrı tutulmuştur. Eski fine-tuned E5 ve
Qwen cache'leri calibrated gibi adlandırılmamış; `legacy_raw`/`raw` olarak
işaretlenmiştir. Bu ayrım özellikle H3 değerlendirmesi için zorunludur.

## 4. Zorunlu metric seti

| Grup | Metrikler | Neden gerekli? |
|---|---|---|
| Task outcome | success, completion, drop, rejection | Görevin gerçekten tamamlanıp gereksinimleri karşılayıp karşılamadığını ayırır. |
| Constraints | hard, deadline, privacy, reliability, accuracy violation | Ortalama reward altında gizlenebilecek güvenlik ihlallerini gösterir. |
| Latency/queue | mean, p50, p95, p99, queue mean/p95 | Tail latency ve congestion etkisini görünür kılar. |
| Capacity | completed/s, successful/s | Sadece oran değil faydalı iş miktarını ölçer. |
| Energy | mean task, episode total, run total, per-success | Başarısız task'ların enerji maliyetini saklamaz. |
| Policy | action histogram, normalized entropy, max-action share/collapse | Degenerate single-action policy'yi yakalar. |
| Fairness | action-load Jain, source-success Jain, active sources | Load dağılımını ölçer; tek kaynak sınırlamasını açık eder. |
| Cost/compute | reward decomposition, decision overhead mean/p95 | Kazancın hangi bileşenden geldiğini ve runtime maliyetini açıklar. |

## 5. Exact-preview ana sonuçları

| Semantic source | Greedy success | Greedy hard | PPO success | PPO hard |
|---|---:|---:|---:|---:|
| No semantic | %78,8 | %7,45 | %76,2 | %0,01 |
| Rule paraphrase | %81,5 | %7,84 | %79,5 | %14,18 |
| TF-IDF + data paraphrase | %85,3 | %2,19 | %85,7 | %4,08 |
| Frozen E5 + data paraphrase | %86,9 | %3,26 | %86,8 | %5,03 |
| Single-human reference | **%91,7** | **%0,00** | **%91,2** | **%0,00** |

Reference semantics'in no-semantic'e success etkisi PPO için `+15,03` puan,
%95 CI `[+13,85; +16,27]`; greedy için `+12,94` puan, `%95 CI
[+11,77; +14,12]` olmuştur. Bu, semantic state'in iki farklı policy family'de
aynı yönde fayda verdiğini gösterir.

Rule yerine TF-IDF(+data) kullanıldığında hard violation PPO'da `-10,10` puan,
greedy'de `-5,65` puan düşmüştür. Dolayısıyla yalnız “semantic var/yok” değil,
extractor quality de downstream safety'yi belirler.

## 6. Independent-188 ve LLM'in uçtan uca etkisi

Independent-188, özgün anlam/sayısal gereksinimi koruyan fakat yeni ifade
biçimleri kullanan son robustness setidir. Etiketler yeni iki-insan adjudication
değil, parent intent'ten aktarılmış reference'tır; bu sınır korunur.

| Semantic source | Greedy success | Greedy hard | PPO success | PPO hard |
|---|---:|---:|---:|---:|
| TF-IDF + data | **%89,1** | **%0,29** | %89,2 | **%0,98** |
| Frozen E5 + data | **%89,2** | %0,44 | **%89,7** | %1,19 |
| Qwen2.5 4-shot raw | %83,8 | %4,48 | %82,1 | %7,36 |
| Qwen2.5 raw + fallback | %83,9 | %4,32 | %82,4 | %6,80 |
| Single-human reference | %92,3 | %0,00 | %91,5 | %0,00 |

Bu sonuçta küçük Qwen2.5 LLM, TF-IDF/E5'i geçmemiştir. Semantic macro-F1 sırası
downstream system sırasına taşınmıştır: güçlü ve daha ucuz küçük classifier'lar
daha yüksek success ve daha düşük hard violation üretmiştir. Çalışmanın katkısı
“LLM her baseline'ı geçti” değildir; extractor error'ın MEC karar güvenliğine
ölçülebilir biçimde yayıldığını göstermesidir.

## 7. Exact/noisy greedy–PPO karşılaştırması

Exact-preview reference koşulunda PPO − greedy success farkı `-0,5` puandır,
%95 CI `[-1,0; -0,1]`; raw p=0,018 fakat family-wise Holm p=0,090'dır. Hard
violation iki yöntemde de sıfırdır. PPO üstünlüğü gösterilmemiştir.

σ=0,5 noisy-preview koşulunda reference PPO success `%90,7`, greedy reference
success `%91,7` olmuştur. PPO − greedy farkı `-0,98` puan, %95 CI
`[-1,44; -0,55]` olduğundan ön kayıtlı `-1` puan non-inferiority sınırı
geçilmemiştir. Gürültü altında da PPO'nun greedy'yi geçtiği söylenemez.

Noisy no-semantic PPO'nun max-action share değeri yaklaşık `%99,5`; reference
semantics ile `%73` çevresine düşmüştür. Semantic input collapse'ı azaltır fakat
öğrenen politikanın güçlü greedy baseline karşısında üstünlüğünü garanti etmez.

## 8. Fallback ve action-mask ablation

Fallback etkisi model ve policy'ye bağlıdır:

| Policy/source | Success farkı, fallback − raw | Hard violation farkı | H3 geçti mi? |
|---|---:|---:|---|
| PPO / TF-IDF | -0,52 `[-0,92; -0,13]` | -0,40 `[-0,59; -0,20]` | Evet |
| PPO / frozen E5 | -1,99 `[-2,41; -1,56]` | +0,10 `[+0,05; +0,17]` | Hayır |
| Greedy / TF-IDF | -0,36 `[-0,78; +0,08]` | -0,05 `[-0,29; +0,16]` | Hayır |
| Greedy / frozen E5 | -1,61 `[-2,14; -1,12]` | +0,26 `[+0,00; +0,52]` | Hayır |

Bu nedenle “calibration/fallback her zaman daha güvenlidir” iddiası
desteklenmemiştir. Calibration ECE'yi düşürebilir; fakat downstream threshold
policy ayrıca doğrulanmadan safety garantisi vermez.

Mask-PPO ile penalty-PPO reference karşılaştırmasında hard violation farkı
`0,00 [0,00; 0,00]`, success farkı `-0,18 [-0,36; -0,02]` olmuştur. Her iki
yöntem de hard violation'ı sıfırladığı için mask'in penalty'den daha düşük ihlal
üreteceği yön hipotezi desteklenmemiştir.

## 9. Ön kayıtlı H1–H6 kararları

| Hipotez | Karar | Kanıt özeti |
|---|---|---|
| H1: reference semantic > no-semantic success | **Desteklendi** | PPO +15,03; greedy +12,94 puan, iki CI da sıfırın üzerinde. |
| H2: rule hard violation > TF-IDF(+data) | **Desteklendi** | PPO +10,10; greedy +5,65 puan. |
| H3: calibrated fallback güvenliği artırır | **Desteklenmedi** | Yalnız PPO/TF-IDF iki kapıyı birlikte geçti. |
| H4: mask penalty'den daha düşük hard violation üretir | **Desteklenmedi** | İki yöntem de %0 hard violation. |
| H5: noisy PPO, greedy'ye -1 pp sınırında non-inferior | **Desteklenmedi** | Alt CI -1,44 puan. |
| H6: H1/H2 yönü NEP katmanında korunur | **Desteklendi** | v1 ve NEP-v2 aynı yönü verdi. |

Çoklu karşılaştırmalar metric family içinde Holm step-down ile düzeltilmiştir.
Effect sizes episode-paired bootstrap ile ve PPO için ayrıca training-seed
standart sapmasıyla verilmiştir.

## 10. Üretilen kanıtlar

- `artifacts/offloading_experiment/results_v2_nep.json`
- `artifacts/offloading_experiment/results_v2_nep_noise.json`
- `artifacts/offloading_experiment/results_v2_nep_independent.json`
- `phase_reports/OFFLOADING_RESULTS_V2_NEP_EXACT.md`
- `phase_reports/OFFLOADING_RESULTS_V2_NEP_NOISY.md`
- `phase_reports/OFFLOADING_RESULTS_V2_NEP_INDEPENDENT.md`
- `phase_reports/HYPOTHESIS_DECISIONS_V2.md`
- `phase_reports/SEMANTIC_RESULTS_V3_2_CACHE.md`
- `scripts/summarize_offloading_results.py`
- `scripts/summarize_hypotheses_v2.py`

## 11. Test ve kabul denetimi

- Exact/noisy/independent result manifest doğrulaması: **0 issue**;
- her ana PPO koşulunda 5/5 seed, 60/60 episode ve finite metrics: **geçti**;
- paired bootstrap ve Holm correction unit testleri: **geçti**;
- system metric golden/unit testleri: **geçti**;
- cache task-ID/provenance ve raw/calibrated ayrımı: **geçti**;
- repository-wide `pytest`: **264 passed**;
- repository-wide Ruff: **geçti**;
- strict mypy `src tests scripts`: **111 source file, 0 issue**.

## 12. Sınırlamalar ve açık genişletmeler

1. Independent-188 labels parent intent'ten aktarılmıştır; yeni adjudicated human
   gold değildir.
2. Fine-tuned E5'in independent calibrated checkpoint'i yoktur. H3 genel kararı
   mevcut koşullarda zaten başarısızdır; eksik koşul ayrıca raporlanır.
3. Qwen independent cache'i constant/raw confidence taşır; calibrated LLM diye
   sunulmaz. Cache generation latency'si bu artifact'ta yoktur; Human-60 ölçümü
   yaklaşık 14,7–16,0 s/task olarak ayrı raporlanmıştır.
4. Decision overhead host wall-clock ölçümüdür; fiziksel task latency'ye
   eklenmez ve farklı donanımlar arası benchmark gibi yorumlanmaz.
5. Reward sensitivity policy retraining değildir.
6. Source fairness tek-source episode nedeniyle tanımlayıcı değildir.
7. Time/entity/application/domain holdout ve raw source-layer ablation'ları Faz
   7'de kalır. Quadro Qwen3-4B/8B yalnız opsiyonel model-scale genişletmesidir.
