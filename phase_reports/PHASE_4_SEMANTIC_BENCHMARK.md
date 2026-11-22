# Faz 4 Kapanış Raporu — Semantik Gereksinim Benchmark'ı

**Kapanış tarihi:** 21 Kasım 2026

**Kapsam:** Ana semantik extractor benchmark'ı

**Karar:** Ana Faz 4 deneyleri tamamlandı; bağımsız 188 paraphrase Faz 7 sağlamlık genişletmesine taşındı. Quadro Qwen3-4B/8B koşusu opsiyoneldir.

## 1. Amaç ve araştırma sınırı

Bu fazın amacı doğal dil görev metnini doğrudan offloading eylemine çevirmek değil;
metinden sekiz alanlı denetlenebilir gereksinim sözleşmesi çıkarmak ve bu çıkarımı
RL eğitiminden bağımsız ölçmektir. Extractor yalnız `task_text` ve
`user_policy_text` görür. Ağ durumu, eylem, reward, oracle sonucu ve değerlendirme
etiketi modele verilmez. Tahminler offloading deneyinden önce dosyaya yazılır ve
cache'lenir.

Sekiz alan `domain`, `latency`, `privacy`, `execution_policy`, `reliability`,
`accuracy`, `energy_priority` ve `divisibility`'dir. Ana ölçüt sekiz alan ortalama
macro-F1; yardımcı ölçütler tam kayıt doğruluğu, tam alan doğruluğu, alan bazlı F1,
kritik gizlilik precision/recall/F1, abstention F1, ECE ve görev başına çıkarım
süresidir.

## 2. İnsan protokolü ve dürüstlük sınırı

Pilot Round 1 A/B/C ve Round 2 A/B verileri protokol iyileştirme ve anlaşma
teşhisi için korunmuştur. Round 2 kalite kapısı geçmemiştir. İkinci annotator ana
60 görevi tamamlayamadığı için protokol, ana etiketleme başlamadan `1.3.0` olarak
amend edilmiştir. Son veri rolleri şöyledir:

- 180 geliştirme kaydı: `machine_generated_weak_label`;
- 60 kilitli değerlendirme kaydı: `single_human_reference`;
- pilot turlar: yalnız anlaşma/protokol kanıtı;
- hiçbir ana kayıt `adjudicated_gold` olarak sunulmaz.

A'nın 60/60 teslimi şema ve kanıt denetiminden geçmiş,
`7eecd3d8efd42b1606911729cbd96706cc9b4b55560fb78fabb8a1f37dae3b08`
SHA-256 ile kilitlenmiştir. Bu sınırlama sonuçları geçersiz kılmaz; fakat dış
geçerlilik ve etikete bağlı belirsizlik tezde açık tutulmalıdır.

## 3. Veri bölümleri ve sızıntı kontrolü

| Bölüm | Boyut | Etiket rolü | Model seçimine açık mı? |
|---|---:|---|---|
| Geliştirme | 180 | Makine zayıf etiketi | Evet, yalnız geliştirme içinde |
| Human-60 özgün | 60 | Tek-insan referansı | Hayır; kilitli test |
| Human-60 paraphrase | 60 | Özgün etiketi aktarıldı | Hayır; sağlamlık testi |
| Kompozisyonel kanonik | 600 | İnşa yoluyla etiket | Hayır; test |
| Kompozisyonel görülmemiş ifade | 600 | İnşa yoluyla etiket | Hayır; test |
| Bağımsız paraphrase | 188 kabul | Otomatik anlam/sayı koruma filtresi | Faz 7 son sağlamlık testi |

Scenario/paraphrase aileleri split sınırını geçmez. Önceki ara koşuda saptanan test
ifadesi sızıntısı kaldırılmış; bu rapor yalnız `results.v3.json` ve dondurulmuş
tahminleri kullanmıştır. Test etiketleri prompt, threshold, model veya eğitim
seçiminde kullanılmamıştır.

## 4. Karşılaştırılan yöntemler

1. Sabit profil: metni okumayan alt sınır.
2. Kural/anahtar kelime: hızlı ve şeffaf LLM'siz baseline.
3. TF-IDF + lojistik regresyon: artırımsız ve ifade artırımlı sürümler.
4. Donuk multilingual-E5-small + lojistik regresyon: artırımsız ve artırımlı.
5. İnce ayarlı multilingual-E5-small: artırılmış geliştirme verisi.
6. Qwen2.5-1.5B-Instruct 4-shot: greedy decoding ve dondurulmuş prompt/parser.
7. Qwen3-0.6B non-thinking: üç örnekli yalnız teknik CPU smoke; kalite sonucu değil.

İnsan referansı extractor modeli değildir; uçtan uca deneyde erişilebilir semantik
üst sınır olarak kullanılır. Quadro Qwen3-4B/8B karşılaştırması bu fazın kabul
koşulu değildir.

## 5. Ana sonuçlar

### 5.1 Human-60

| Yöntem | Özgün macro-F1 | Özgün exact | Paraphrase macro-F1 | Paraphrase exact | Özgün süre |
|---|---:|---:|---:|---:|---:|
| Sabit | %25,3 | %3,3 | %25,3 | %3,3 | 0,01 ms |
| Kural | %78,0 | %38,3 | %40,3 | %23,3 | 0,16 ms |
| TF-IDF | %76,6 | %30,0 | %48,1 | %25,0 | 1,58 ms |
| TF-IDF + veri | **%78,2** | **%38,3** | **%67,9** | %23,3 | 1,58 ms |
| E5 donuk | %75,8 | %33,3 | %51,8 | %20,0 | 18,97 ms |
| E5 donuk + veri | %76,3 | %36,7 | %60,4 | **%26,7** | 17,32 ms |
| E5 ince ayar + veri | %70,2 | %10,0 | %62,3 | %8,3 | 13,45 ms |
| Qwen2.5 4-shot | %48,3 | %5,0 | %43,7 | %8,3 | 14.696 ms |

### 5.2 Kompozisyonel görülmemiş ifade

| Yöntem | Macro-F1 | Tam kayıt |
|---|---:|---:|
| Sabit | %20,7 | %0,8 |
| Kural | %47,5 | %16,5 |
| TF-IDF | %36,7 | %15,0 |
| TF-IDF + veri | **%73,6** | **%37,0** |
| E5 donuk | %44,9 | %17,7 |
| E5 donuk + veri | %70,9 | %33,0 |
| E5 ince ayar + veri | %73,0 | %20,5 |

Kural ile veri artırılmış TF-IDF özgün human-60'da pratik olarak eşittir:
macro-F1 farkı +0,2 puan, %95 GA [0,0; 0,5], p=0,254. Buna karşılık human-60
paraphrase'te fark +27,6 puan, %95 GA [16,6; 32,0], p<0,001; kompozisyonel
görülmemiş ifadede +26,0 puan, %95 GA [23,9; 27,8], p<0,001'dir. Fazın ana
pozitif sonucu model büyüklüğü değil, ifade çeşitliliğiyle kazanılan paraphrase
dayanıklılığıdır.

## 6. Önceden tanımlı kalite kararları

| Karar | Sonuç | Gerekçe |
|---|---|---|
| Metin kullanan baseline sabiti geçiyor mu? | **Evet** | Kural/TF-IDF/E5 özgün macro-F1'de sabitin çok üzerindedir. |
| Artırma paraphrase sağlamlığını artırıyor mu? | **Evet** | TF-IDF %48,1→%67,9; donuk E5 %51,8→%60,4. |
| Encoder küçük ML'yi zorunlu olarak geçiyor mu? | **Hayır** | TF-IDF + veri bütün ana sağlamlık tablolarında en az E5 kadar güçlü ve daha hızlıdır. |
| Fine-tuning güvenilir üstünlük sağlıyor mu? | **Hayır** | Özgün exact %10'a düştü; domain forgetting/overfit işareti vardır. |
| Küçük few-shot LLM kuralı paraphrase'te geçiyor mu? | **Desteklenmedi** | Qwen2.5–kural farkı +3,4 puan, p=0,24; gecikme yaklaşık 16 s/görev. |
| Ham confidence güvenli fallback için yeterli mi? | **Hayır** | Qwen constant confidence üretir; E5 ECE ve v1 fallback davranışı model-bağımlıdır. |

Qwen3-0.6B smoke'ta parse/OOV hatası olmamıştır; ancak yalnız üç özgün örnekte
ortalama yaklaşık 289 saniye/görev ölçüldüğü için tam CPU koşusu durdurulmuştur.
Bu, model kalite karşılaştırması değil uygulanabilirlik açısından negatif bir
sonuçtur.

## 7. Üretilen kanıtlar

- `data/semantic_benchmark/main/benchmark_v3/results.v3.json`
- `data/semantic_benchmark/main/benchmark_v3/results_llm.json`
- `data/semantic_benchmark/main/benchmark_v3/predictions/`
- `phase_reports/SEMANTIC_RESULTS_V3.md`
- `phase_reports/SEMANTIC_RESULTS_SUMMARY.md`
- `docs/EXPLORATORY_IMPROVEMENT_PROTOCOL_V1.md`
- `TEZ_DANISMANI_MEVCUT_DURUM_VE_LITERATUR_OZETI.md`

## 8. Test ve kabul denetimi

21 Kasım 2026 kapanış denetimi:

- `pytest -q`: **245 passed**;
- `ruff check .`: **all checks passed**;
- `mypy --strict src tests`: **69 source file, no issue**;
- danışman raporu MD–DOCX görünür metin/paragraf/tablo hücresi parity: **geçti**;
- DOCX embedded figure count: **3/3**.

Semantik sonuçların satır sayıları, izinli sınıf sözlüğü, split aile ayrıklığı ve
tahmin–etiket görev kimliği eşleşmesi otomatik testlerle korunmaktadır.

Faz 4 kabul kararı ana benchmark için verilmiştir. Şunlar Faz 4'ü yeniden açmaz:
bağımsız 188 sağlamlık tablosu (Faz 7), Quadro büyük-LLM karşılaştırması (opsiyonel)
ve yeni ikinci uzman çalışması (danışman kararı/ek geçerlilik). Test veya hash
uyuşmazlığı bulunursa faz yeniden açılır.

## 9. Sonraki faza taşınan riskler

1. Human-60 tek annotator referansıdır; gold iddiası kurulamaz.
2. Bağımsız 188 paraphrase için yalnız Qwen cache'i hazırdır; bütün extractor'ların
   ortak skoru ve offloading etkisi henüz yoktur.
3. Confidence calibration yalnız development/calibration verisinde yapılmalıdır.
4. LLM'in offloading v2 etkisi ayrıca ölçülmelidir; semantik F1 tablosu sistem
   katkısının yerine geçmez.
5. Nihai çoklu kontrastlarda Holm düzeltmesi uygulanmalıdır.
