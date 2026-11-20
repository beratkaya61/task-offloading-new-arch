# Bölüm 3 — Veri ve Yöntem

> **Otomatik üretilmiş belge.** Bu bölümdeki bütün sayılar depodaki sürümlü veri, manifest ve kod sabitlerinden `scripts/build_thesis_method_chapter.py` ile üretilir; elle düzenlenmemelidir. Veri, protokol veya deney ayarı değiştiğinde betik yeniden çalıştırılır; `tests/phase_5/test_thesis_chapter_current.py` belge güncel değilse başarısız olur.

## 3.1 Genel yaklaşım

Çalışma iki ayrık katmandan oluşur. **Semantik katman**, doğal dildeki görev metnini sekiz alanlı bir gereksinim sözleşmesine dönüştürür; ağ durumunu görmez ve karar vermez. **Karar katmanı**, bu sözleşmeyi ve fiziksel sistem durumunu kullanarak görevi yerel cihazda, üç edge sunucudan birinde veya bulutta çalıştırır. Katmanların ayrılması, semantik çıkarım kalitesinin sistem sonuçlarına etkisinin nedensel olarak ölçülmesini sağlar: fiziksel olaylar, görevler ve tohumlar sabit tutulurken yalnız semantik bilginin kaynağı değiştirilir.

Doğal dil MEC görev gereksinimleri ile fiziksel offloading sonuçlarını aynı kayıtta sunan açık bir veri seti bulunmadığından, semantik katman için kontrollü bir benchmark tasarlanmış; fiziksel katman ise gerçek veri kaynaklarından beslenmiştir. Bu nedenle deney ortamı **iz güdümlü hibrit (trace-driven hybrid)** olarak adlandırılır; tamamen gerçek bir veri seti olarak sunulmaz.

## 3.2 Semantik gereksinim şeması

Her görev 8 alanla temsil edilir: `domain`, `latency`, `privacy`, `execution_policy`, `reliability`, `accuracy`, `energy_priority`, `divisibility`. Her alan bir durum (`explicit`, `partial`, `not_stated`, `ambiguous`), değer(ler), metinden kanıt alıntısı ve güven skoru taşır. Metinde belirtilmeyen bir gereksinim uydurulmaz (`not_stated`); çelişkili ifadeler `ambiguous` olarak işaretlenir ve karar verilmez (abstention).

## 3.3 Kontrollü görev corpus'u

Ana corpus 240 Türkçe görev metninden oluşur (120 senaryo ailesi). Metinler yapılandırılmış senaryo tanımlarından LLM destekli olarak üretilmiştir (`generation_method: llm_assisted_structured_design`); **gerçek son kullanıcı kayıtları değildir.** Dağılım alan (domain) bazında dengelidir:

| Alan | agriculture | consumer | healthcare | industrial | other | public_safety | smart_city | transportation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Metin | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |

Köken bazında: `adversarial` 60, `llm_assisted_synthetic` 60, `standard_derived` 60, `trace_conditioned` 60. Corpus, senaryo/şablon/yeniden ifade aileleri bölünmeden (aile sızıntısı olmadan) **180 geliştirme** ve **60 değerlendirme** metnine ayrılmıştır. Değerlendirme metinlerinden 29 tanesi ayrı bir kullanıcı politikası metni içerir.

## 3.4 Etiketleme

**Pilot.** 20 pilot metin, birinci turda üç (A, B, C), ikinci turda iki (A, B) etiketleyici tarafından kör olarak etiketlenmiştir. Tam alan uyumu birinci turda %75,6, ikinci turda %83,8 olmuştur; ancak önceden tanımlı kalite kapısı (kappa ≥ 0,7) 7 kontrolde karşılanamamıştır (`failed`). Bu sonuç değiştirilmeden saklanmış ve protokol revizyonlarına gerekçe olmuştur.

**Ana değerlendirme referansı.** 60 değerlendirme metni, protokol 1.3.0'a göre tek bir insan etiketleyici (A) tarafından kör biçimde etiketlenmiş ve SHA-256 ile kilitlenmiştir. Bu referans iki uzmanla uzlaştırılmış bir "gold standard" değildir; **tek-insan referansıdır.**

**Protokol normalizasyonu.** Referansa yalnız protokolde yazılı, yoruma kapalı iki kural deterministik olarak uygulanmıştır: R1 — sayısal başarı eşiği ≥ %99,9 ise `mission_critical` (4 değişiklik); R2 — metrik adının kanonik yazımı `F1` (1 değişiklik). Toplam 5 değişiklik ayrı bir kayıtta tutulur; orijinal referans dosyası değiştirilmemiştir ve normalizasyon sırasında hiçbir model çıktısına bakılmamıştır.

**Belirlenen protokol açığı.** Annotasyon protokolü ayrı kullanıcı politikası metninin nasıl etiketleneceğini tanımlamamaktadır. Şablon tasarım etiketleri ile insan referansı arasındaki ayrışmaların büyük bölümü politika metni içeren 29 görevde yoğunlaşmıştır. Bu durum sonuçlar raporlanırken ayrıca belirtilir ve ölçüm sınırlaması olarak kabul edilir.

**Geliştirme etiketleri.** 180 geliştirme metninin etiketleri şablon tasarım kurallarından otomatik üretilmiş **makine zayıf etiketleridir** (`machine_generated_weak_label`); insan etiketi olarak sunulmaz.

Özetle, çalışmadaki gerçek insan etiketli metin sayısı **60 ana + 20 pilot** metindir.

## 3.5 Makine üretimi ek veri

**Anlam koruyan yeniden ifade (CheckList tarzı değişmezlik testi).** Şablon gereksinim cümlelerinin her biri için 19 kural tanımlanmıştır. Her kuralın birbirinden ayrık iki havuzu vardır: yalnız eğitim çoğaltmasında kullanılan **eğitim havuzu** (kural başına 2 ifade) ve yalnız test için kullanılan **test havuzu** (kural başına 2 ifade). Sayısal eşikler korunur; etiketler değişmezlik varsayımıyla aktarılır. Havuzlar herhangi bir model sonucu görülmeden birlikte yazılmıştır. Test havuzu ve ilk sonuçlar görüldükten sonra eklenen ek eğitim ifadeleri, deneyci sızıntısı oluşturduğu için **geri çekilmiştir**.

**Kompozisyonel görevler.** Mevcut bir metnin alan cümlesine 0–4 bağımsız gereksinim yuvası eklenerek etiketleri inşa yoluyla bilinen görevler üretilir. Yuva sayısı olasılıkları 0,1, 0,25, 0,3, 0,25, 0,1; gecikme eşikleri 30, 50, 80, 120, 200, 300, 500, 800, 1500, 3000 ms; güvenilirlik eşikleri %95, %98, %99, %99,5, %99,9, %99,95, %99,99; F1 eşikleri 0,93, 0,95, 0,97. Etiket eşlemesi annotasyon protokolündeki kurallarla aynıdır. Test görevleri yalnız değerlendirme bölümünün alan cümlelerinden ve test havuzundan, eğitim görevleri yalnız geliştirme bölümünden ve eğitim havuzundan üretilir.

| Veri | Metin | Etiket kaynağı |
|---|---:|---|
| Genişletilmiş eğitim seti | 1.924 | 180 geliştirme metni + makine yeniden ifadesi + kompozisyonel (makine) |
| Test: `human60_original` | 60 | `single_human_reference_protocol_normalized` |
| Test: `human60_paraphrase` | 60 | `machine_paraphrased_text_with_human_labels_transferred_under_invariance_assumption_not_human_annotated` |
| Test: `compositional600_canonical` | 600 | `machine_generated_label_by_construction` |
| Test: `compositional600_paraphrase` | 600 | `machine_generated_label_by_construction` |

## 3.6 Semantik çıkarım yöntemleri

Karşılaştırılan yöntemler: (i) **sabit profil** (eğitimdeki en sık profil); (ii) **kural/anahtar kelime** ayrıştırıcısı; (iii) **TF-IDF + lojistik regresyon** (karakter 3–5-gram, alan başına sınıf-dengeli model); (iv) **donuk `intfloat/multilingual-e5-small` + lojistik regresyon**; (v) **ince ayarlı `multilingual-e5-small`** (alan başına doğrusal başlık). Öğrenen yöntemler hem yalnız geliştirme metinleriyle hem de genişletilmiş eğitim setiyle eğitilmiştir.

İnce ayar hiperparametreleri herhangi bir değerlendirme sonucu görülmeden sabitlenmiştir: `random_seed=20260929`, `epochs=4`, `batch_size=16`, `learning_rate=3e-05`, `max_length=128`. Değerlendirme etiketleri hiçbir yöntemin eğitiminde veya ayarında kullanılmamıştır.

## 3.7 Değerlendirme ölçütleri ve istatistik

Ölçütler: tam kayıt doğruluğu (sekiz alanın tamamı doğru), alan doğruluğu, alanlar üzerinden ortalama macro-F1, kritik gizlilik (hassas/kısıtlı veri) için precision/recall/F1 ve pozitif örnek sayısı, abstention F1, kalibrasyon (ECE, Brier) ve görev başına çıkarım süresi.
Belirsizlik, görev düzeyinde 1000 tekrarlı bootstrap ile %95 yüzdelik güven aralığı olarak verilir. Yöntem karşılaştırmaları aynı bootstrap örnekleri üzerinde kural tabanlı yönteme göre eşleştirilmiş fark, farkın güven aralığı ve iki yönlü bootstrap p-değeri ile raporlanır.

## 3.8 Semantik sözleşmeden kısıtlara dönüşüm

Çalıştırma izni yalnız açık (`explicit`/`partial`) yasaklarla daraltılır; `unknown` hiçbir zaman kendiliğinden yasak üretmez. Nitel sınıflar şu sayısal karşılıklarla kısıta çevrilir:

| Alan | Eşleme |
|---|---|
| Gecikme sınıfı (s) | `hard_real_time`=0,05, `real_time`=0,2, `interactive`=1, `relaxed`=4 |
| Güvenilirlik katmanı | `mission_critical`=0,999, `high`=0,99, `standard`=0,95 |
| Doğruluk katmanı | `high`=0,95, `standard`=0,9 |

**İhtiyatlı mod (conservative fallback).** Çalıştırma politikası açık değilken veri hassas/kısıtlı ise, politika belirsizse veya politika güveni 0,6'nın altındaysa bulut maskelenir. İhtiyatlı mod yalnız izin kaldırır, hiçbir zaman izin eklemez.

**İnanılan ve gerçek gereksinim.** Karar katmanının maskesi semantik çıkarıcının ürettiği *inanılan* gereksinimle kurulur; kısıt ihlalleri ise her zaman *gerçek* gereksinime göre ölçülür. Böylece çıkarım hatalarının sistemdeki karşılığı doğrudan gözlenir.

## 3.9 İz güdümlü hibrit MEC ortamı

| Kaynak | Boyut | Lisans/erişim | Ortamdaki rolü |
|---|---:|---|---|
| UCI MEC-859 | 0,1 MB | CC-BY-4.0 | 4.000 ölçülmüş yürütme süresi; Raspberry Pi 4B yerel cihaz, MacBookPro1 / Ubuntu VM / MacBookPro2 sırasıyla `edge_0..2`. Ortalama süreler (s): macbook_pro_1 0,108, macbook_pro_2 0,348, raspberry_pi_4b 1,017, ubuntu_virtualbox_vm 0,427 |
| BUPT Edge-Computing | 29,0 MB | yerel akademik kullanım | 476.419 kabul edilen istek kaydı; görev geliş deseni ve uplink/downlink bit büyüklükleri. Kronolojik bölme: ayın 24'ünden önceki günler eğitim, sonrası test |
| EUA | 34,0 MB | MIT | Avustralya genelinde 4.748 kullanıcı ve 95.562 edge aday konumu; en yakın üç sitenin mesafesi radyo bağlantısını belirler |
| NEP-large (Full trace) | 5,89 GB | yalnız araştırma, paylaşım yasak | toplam 685.506.639 satırdan 592.724.139 tanesi kabul politikasından geçmiştir; **henüz ortama bağlanmamıştır**; yalnız bütünlük ve kabul profili mevcuttur |

Ölçülmemiş, açıkça kalibrasyon varsayımı olan değerler:

| Parametre | Değer |
|---|---|
| `device_cpu_hz` | 1,5 GHz (Raspberry Pi 4B nominal saat) |
| `workload_scale` | 0,25 (yerel ortalama servis süresi ölçeği) |
| `cloud_speedup_vs_fastest_edge` | 2× en hızlı edge |
| `cloud_one_way_propagation_s` | 25 ms tek yön |
| `cloud_backhaul_bps` | 1 Gbit/s |
| `target_mean_interarrival_s` | 30 ms ortalama geliş aralığı (BUPT deseni ölçeklenir) |
| `energy_coefficient` | 1e-28 |
| `tx_power_w` | 0,5 |
| `rx_power_w` | 0,3 |
| `idle_power_w` | 0,05 |
| `battery_j` | 5000 J |
| `device_success` | 0,999 |
| `edge_success` | 0,995, 0,99, 0,995 |
| `cloud_success` | 0,9999 |
| `cloud_backhaul_success` | 0,9999 |
| `link_success` | 0,9999 |
| `device_accuracy` | 0,85 |
| `edge_accuracy` | 0,95, 0,93, 0,95 |
| `cloud_accuracy` | 0,98 |
| `max_edge_background_load` | 0,9 |
| Radyo | bant genişliği 10 MHz, taşıyıcı 3,5 GHz, yol kaybı üssü 3, log-mesafe + Shannon hızı |

Her bölüm (episode) 64 görevden oluşur. Eylem uzayı: `local`, `edge_0`, `edge_1`, `edge_2`, `cloud`. Ödül, tek fizik çekirdeğinde hesaplanan bileşenlerin ağırlıklı toplamının negatifidir: `latency_norm`×1, `energy_norm`×1, `deadline_miss`×5, `soft_constraint_violation`×5, `failure_or_drop`×10, `hard_constraint_violation`×20 (normalizasyon referansları 0,2 s ve 0,1 J).

Eğitim bölümlerinden önce bir *baskın eylem* kontrolü yapılmıştır: hiçbir sabit eylem tek başına semantik-duyarlı açgözlü (greedy) politikanın başarısına ulaşmamaktadır. Bu kontrol yalnız eğitim bölümleri üzerinde yapılmış, test sonuçlarına bakılarak ayar yapılmamıştır.

## 3.10 Karar politikaları ve deney tasarımı

Referans politikalar: her zaman yerel / `edge_0` / bulut, geçerli eylemler arasından rastgele, semantiksiz açgözlü (en düşük beklenen gecikme) ve semantik-duyarlı açgözlü (inanılan gecikme, güvenilirlik ve doğruluk kısıtlarını karşılayan eylemler arasında en düşük gecikme).

Öğrenen politika: MaskablePPO (sb3-contrib), MultiInputPolicy; `n_steps=2048`, `batch_size=256`, `learning_rate=0.0003`, `gamma=0.95`, `ent_coef=0.01`; 5 tohum × 150.000 adım; 400 eğitim ve 60 test bölümü. **Semantikli PPO** gereksinim özniteliklerini görür ve semantik maskeyi kullanır; **semantiksiz PPO** ikisini de görmez. PPO, eğitimde referans semantikle eğitilir; testte, gerçek dağıtımdaki gibi, semantik çıkarıcıların dondurulmuş tahminleriyle çalıştırılır. Bütün politikalar birebir aynı test bölümleri üzerinde değerlendirilir; güven aralıkları bölüm düzeyinde eşleştirilmiş bootstrap ile hesaplanır.

## 3.11 Geçerlilik tehditleri ve sınırlamalar

- Görev metinleri kontrollü olarak üretilmiştir; gerçek kullanıcı ifadelerindeki performans doğrudan ölçülmemiştir. Yeniden ifade testleri bunun yalnız bir yaklaşımıdır ve bazı ipucu sözcükler eğitim ile test havuzlarında ortaktır; bu nedenle dayanıklılık sonuçları iyimser bir tahmindir.
- İnsan referansı 60 metinlik ve tek etiketleyicilidir; güven aralıkları geniştir ve kritik gizlilik gibi nadir sınıflarda örnek sayısı çok düşüktür. Etiketleyiciler arası uyum yalnız pilot setinde ölçülmüş ve kalite kapısı karşılanamamıştır.
- Kompozisyonel ve yeniden ifade setleri makine üretimidir; etiketleri inşa veya aktarma yoluyla elde edilmiştir ve insan etiketi değildir.
- Fiziksel ortam iz güdümlü hibrittir: farklı kaynakların satırları aynı gerçek olayın ölçümü değildir; mutlak iş yükü ölçeği, bulut ve güvenilirlik/doğruluk parametreleri kalibrasyon varsayımıdır.
- İnce ayarlı encoder CPU bütçesiyle sınırlı sayıda epoch eğitilmiştir; sonuçları daha büyük bütçelerdeki performansı temsil etmeyebilir.

