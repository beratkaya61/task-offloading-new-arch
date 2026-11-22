# Bağımsız Paraphrase Semantic Sonuçları v1

**Tarih:** 2026-11-21

**Durum:** Faz 7 için doğrulanmış ara kanıt; Faz 7 kapanış raporu değildir.

**Makine-okunur sonuç:**
`data/semantic_benchmark/main/benchmark_v3/results.cached_v3_2.json`

**Tam tablo:** `phase_reports/SEMANTIC_RESULTS_V3_2_CACHE.md`

## 1. Neden bu deney yapıldı?

Önceki human-60 ve compositional-600 setleri, sırasıyla küçük tek-insan referans
bütçesi ve kontrollü template üretimi nedeniyle dış geçerliliği tek başına
gösteremez. Bu deney, eğitim paraphrase havuzunu görmeyen bir üretici tarafından
yeniden yazılan ve otomatik filtrelerden geçen 188 metinde extractor sıralamasının
korunup korunmadığını ölçer. Etiketler yeniden yazılan metinlere anlam değişmezliği
varsayımıyla aktarılmıştır; 188 metin insan tarafından yeniden etiketlenmemiştir.

## 2. Veri ve değerlendirme sözleşmesi

- Aday üretimler: 240.
- Otomatik kontrollerden sonra tutulan: 188.
- Birincil metric: sekiz semantic field'ın ortalama macro-F1 değeri.
- İkincil metrics: exact-record accuracy, critical-privacy P/R/F1, abstention F1,
  ECE, Brier score ve field-level macro-F1.
- Belirsizlik: task-level, paired, 1,000 tekrarlı percentile bootstrap %95 CI.
- Farklar aynı bootstrap task indices ile `rule_keyword` yöntemine göre ölçüldü.
- Isotonic confidence calibration yalnız 360 örneklik development-split
  calibration setinde fit edildi. Bu işlem label/signature tahminini değiştirmedi.
- Prediction cache per-task inference latency saklamadığı için bu sette latency
  uydurulmadı ve `not recorded` olarak bırakıldı. Human-60 Qwen latency sonucu ayrı
  ölçümde yaklaşık 14.7--16.0 s/task'tır.

## 3. Ana sonuçlar

| Yöntem | Macro-F1 (%) | Exact record (%) | Critical privacy F1 (%) | Abstention F1 (%) | ECE raw→cal |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LR (+data) | **80.91** | 36.17 | 64.71 | 94.37 | 0.151→0.117 |
| E5-small frozen + LR (+data) | **73.81** | 34.04 | **68.57** | 94.38 | 0.271→0.071 |
| Rule/keyword | 54.64 | 18.09 | 38.60 | 89.86 | 0.086→0.138 |
| E5-small frozen + LR | 51.70 | 18.09 | 32.79 | 88.56 | 0.340→0.054 |
| TF-IDF + LR | 44.01 | 13.30 | 11.76 | 87.22 | 0.123→0.073 |
| Qwen2.5-1.5B-Instruct, 4-shot | 42.03 | 7.45 | 33.33 | 84.22 | 0.010→0.010* |
| Constant profile | 20.93 | 1.06 | 0.00 | 85.66 | 0.085→0.005 |

\* Qwen bütün field'lar için sabit 0.7 confidence ürettiği için bu ECE gerçek bir
calibration modelinin kalitesini göstermez ve isotonic calibration uygulanmamıştır.

### Paired contrasts against rule

| Contrast | Macro-F1 farkı (puan) | %95 CI | İki yönlü bootstrap p |
|---|---:|---:|---:|
| TF-IDF(+data) − rule | **+26.27** | **[+22.24, +29.98]** | <0.001 |
| E5 frozen(+data) − rule | **+19.16** | **[+14.40, +23.33]** | <0.001 |
| Qwen2.5-1.5B − rule | **−12.62** | **[−16.37, −8.37]** | <0.001 |

## 4. Hipotez kararı ve yorum

- **S1 desteklendi:** TF-IDF(+data), bağımsız paraphrase setinde rule baseline'ı
  belirgin biçimde geçti; CI sıfırın tamamen üzerindedir.
- Data augmentation hem sparse TF-IDF hem frozen E5 için kritik bileşendir. Aynı
  model ailesinin augmentation'sız sürümleri sırasıyla %44.01 ve %51.70'te kaldı.
- Qwen2.5-1.5B few-shot bu bounded Turkish extraction görevinde güçlü bir baseline
  olmadı. Daha büyük language model'in otomatik olarak daha iyi extractor olduğu
  iddiası bu kanıtla desteklenmez.
- Isotonic calibration E5 modellerinde ECE'yi güçlü biçimde düşürdü; fakat rule
  ECE'sini 0.086'dan 0.138'e, TF-IDF(+data) Brier değerini 0.117'den 0.121'e
  kötüleştirdi. Calibration sonucu yöntem bazında raporlanmalı, yalnız iyileşen
  yöntemler seçilmemelidir.

## 5. Eksik kalan ve yanlış yorumlanmaması gerekenler

1. E5 fine-tuned modelinin v3.1 dört-set prediction cache'i vardır; fakat model
   checkpoint'i kaydedilmediği için 188 bağımsız metin üzerinde prediction yoktur.
   CPU retraining denemesi frozen tahminler bittikten sonra kaynak baskısı altında
   sonlanmıştır. Bu yöntem bağımsız ortak tablodan sessizce çıkarılmamış, eksik
   koşul olarak kaydedilmiştir.
2. 188 label insan tarafından yeniden doğrulanmadı. Sonuç paraphrase robustness
   kanıtıdır; yeni human-gold benchmark iddiası değildir.
3. Semantic macro-F1 tek başına MEC yararını göstermez. Aynı belief'lerin greedy ve
   PPO üzerindeki success/hard-violation etkisi NEP-v2 bağımsız-pool koşusunda ayrıca
   ölçülmelidir.
4. Cache inference latency saklamadığı için bağımsız sette model latency sıralaması
   yeniden ölçülmedi. Daha önce kaydedilmiş human-60 latency ile karıştırılmamalıdır.

## 6. Faz 7'yi kapatmak için sonraki kontroller

- Bağımsız-188 pool üzerinde reference/rule/TF-IDF/E5/Qwen belief'leriyle aynı
  greedy ve PPO checkpoints değerlendirmesi.
- Task time/entity/application-domain ve source-layer ablation tabloları.
- Synthetic/compositional ile NEP trace-layer yön tutarlılığı.
- Yukarıdaki sonuçlar tamamlandıktan sonra resmi Faz 7 kapanış raporu.
