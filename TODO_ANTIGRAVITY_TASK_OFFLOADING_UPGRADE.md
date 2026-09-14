# Task Offloading New Architecture — Bilimsel ve Mimari Sözleşme

Bu belge projenin “kuzey yıldızı”dır. `task.md` ilerlemeyi, bu dosya ise bir işin kabul edilmesi için gereken kaliteyi tanımlar.

## 1. Ana araştırma hedefi

İlk hedef, LLM’nin doğrudan local/edge/cloud kararı vermesi değildir. Hedef; doğal dil görev niyetinden yapılandırılmış gereksinim çıkaran, insan ground truth’una göre doğrulanmış ve dondurulmuş bir LLM katmanının DRL offloading politikasına nedensel katkısını ölçmektir.

Önerilen soru:

> Bağımsız doğrulanmış LLM semantik gereksinimleri, aynı MEC fiziği ve aynı değerlendirme olayları altında, semantiksiz ve LLM’siz semantik baselines’a göre constraint satisfaction ve genellemeyi iyileştiriyor mu?

## 2. Değiştirilemez ilkeler

### 2.1 Tek fizik kaynağı

- Kanal, gecikme, enerji, queue ve başarısızlık hesabı tek geçiş çekirdeğinde yaşar.
- Gym, event simulation, baseline ve oracle aynı çekirdeği çağırır.
- Birimler tiplerde ve dokümantasyonda görünürdür.
- Tuple ile belirsiz fizik değeri taşınmaz; isimli veri tipleri kullanılır.

### 2.2 Semantik izolasyonu

- LLM yalnız görev metni/kullanıcı politikasını görür.
- Ağ durumu, doğru eylem, ödül ve oracle sonucu LLM girdisine verilmez.
- Çıktı eylem değil, gereksinim JSON’udur.
- Bilinmeyen alan uydurulmaz; `unknown/null` ve confidence kullanılır.
- LLM çıktıları RL’den önce cache’lenir ve sürümlenir.
- Ödülde “LLM önerisine uyum” bonusu bulunmaz.

### 2.3 Veri dürüstlüğü

- Her dataset için kaynak, sürüm, lisans ve checksum manifestte bulunur.
- `downloaded`, `validated` anlamına gelmez.
- Her kolon observed/measured/derived/matched/generated/labeled olarak işaretlenir.
- İlişkisiz kaynakların satırları doğal ortak gözlem gibi sunulmaz.
- Normalizer yalnız training split’ine fit edilir.
- Zaman ve entity sızıntısı otomatik test edilir.
- Hibrit benchmark açıkça hibrit diye adlandırılır.

### 2.4 Deneysel adalet

- Karşılaştırmalar aynı görevler, exogenous events ve episode seed’lerini kullanır.
- Baseline’ların bilgi erişimi eşittir; oracle ayrı etiketlenir.
- En az 5, tercihen 10 bağımsız eğitim seed’i kullanılır.
- Güven aralığı, eşleştirilmiş test ve etki büyüklüğü raporlanır.
- Tek eylem oranı, eylem entropisi ve durum duyarlılığı zorunlu metriktir.
- Ana baseline sezgisel/oracle’dan kötü ise neden çözülmeden karmaşıklık eklenmez.

### 2.5 Aşamalı karmaşıklık

- İlk eylem alanı local/edge_i/cloud’dur.
- Partial offloading yalnız görev bölünebilirliği ve ek yük modeli doğrulanınca eklenir.
- Resource allocation gerçekten eylem alanına girmeden başlıkta iddia edilmez.
- GNN yalnız değişken topoloji/ilişki yapısı gerektiriyorsa kullanılır.
- GNN-PPO iddiası için grafik politika PPO ile uçtan uca eğitilir.
- GUI ve demo bilimsel çekirdeği değiştirmez, son fazda yapılır.

## 3. Zorunlu baseline ve ablation ailesi

### Fiziksel karar baselines

- Always-local, always-edge, always-cloud.
- Uniform random ve valid-action random.
- Latency-aware, energy-aware, deadline-aware greedy.
- Küçük exhaustive/MILP oracle.
- PPO-MLP ana öğrenen politika; DQN/A2C ikincil.

### Semantik baselines

- Semantiksiz.
- Sabit profil.
- Rule/keyword parser.
- TF-IDF + lojistik regresyon veya benzer küçük ML.
- Küçük encoder.
- Zero-shot ve few-shot LLM.
- İnsan oracle etiketi.

### Ana ablation’lar

- Semantik özellikleri çıkar.
- Action mask’i çıkar.
- LLM yerine rule/ML/human oracle koy.
- Confidence/abstention kullanımını çıkar.
- Ödül bileşenlerini ayrı çıkar.
- Veri kaynak katmanlarını ayrı çıkar.
- Ön-eğitim varsa scratch karşılaştır.
- İleri mimaride GNN/recurrent/partial bileşenini ayrı çıkar.

## 4. Zorunlu metrikler

### Sistem

- Success/acceptance, throughput, drop.
- End-to-end latency p50/p95/p99 ve deadline miss ratio.
- Total/per-task/per-success energy.
- Queue waiting.
- Privacy/reliability/accuracy ve toplam constraint violations.
- User success/slowdown ve server load temelli fairness.

### Politika

- Action histogram, entropy, collapse ratio.
- Durum dilimleri ve karşı-olgusal semantik çiftlerde eylem duyarlılığı.
- Reward component decomposition.
- Training stability ve generalization gap.

### Semantik katman

- Schema validity ve exact match.
- Macro/micro F1 ve alan bazlı precision/recall.
- Calibration: Brier/ECE.
- Abstention/unknown doğruluğu.
- Model, prompt, parser ve toplam latency/token/cost.

## 5. Test matrisi

- Birim ve boyut/birim testleri.
- Fiziksel monotonicity/property testleri.
- Determinizm ve seed testi.
- Elle hesap/exhaustive oracle testi.
- Gym-event parity testi.
- Action-mask ve hard constraint testi.
- Split/duplicate/leakage testi.
- Semantic JSON ve parser robustness testi.
- LLM cache hash/reproducibility testi.
- Metric definition golden testleri.
- End-to-end küçük deney smoke testi.

## 6. Faz kabul kuralları

Bir faz yalnız şu koşullarda tamamlanır:

1. `task.md` içindeki çıktılar üretilmiştir.
2. O fazın testleri geçmektedir.
3. `phase_reports/` altında girişler, yöntem, sonuç, başarısızlık ve sınırlamalar yazılmıştır.
4. Veri/config/kod/seed bağı izlenebilir durumdadır.
5. Sonraki fazın riskleri ve açık işleri yazılmıştır.

Kutunun işaretlenmesi akademik iddia değildir; rapordaki kanıt akademik iddiadır.

## 7. Definition of Done

- [ ] Tez sorusu ve hipotezleri dondurulmuş.
- [ ] Semantik ground truth iki etiketçi ve anlaşma analiziyle mevcut.
- [ ] LLM semantik çıkarıcı bağımsız benchmark’ta ölçülmüş ve cache’lenmiş.
- [x] Tek fizik çekirdeği oracle/parity/invariant testlerini geçiyor.
- [ ] Veri manifesti, lisans, checksum, alan kökeni ve sızıntısız split var.
- [ ] Güçlü sabit/sezgisel/oracle baselines tamam.
- [ ] Semantiksiz/rule/ML/LLM/human-oracle politika karşılaştırmaları tamam.
- [ ] En az 5 seed, %95 CI, planlı test ve effect size mevcut.
- [ ] p95/p99, DMR, enerji, constraint, fairness, overhead ve collapse raporlu.
- [ ] Sentetik ve hibrit trace sonuçlarının sınırları ayrı yazılmış; testbed yapılırsa ayrı ek doğrulama olarak raporlanmış.
- [ ] GNN/partial/resource allocation iddiaları yalnız gerçekten uygulanmışsa başlıkta.
- [ ] Bütün deneyler tek komut/config ve sürümlü manifestlerle yeniden üretilebilir.
- [ ] Negative results ve threats to validity tezde açıkça bulunuyor.

## 8. Faz 0 kararı

Eski repo doğrudan taşınmayacaktır. Fizik hataları regresyon testine, eski veri üretim sorunları leakage/provenance testine, çökmüş politikalar collapse testine dönüştürülecektir. Yeni kod başlamadan önce Faz 1 araştırma sözleşmesi tamamlanmalıdır.

## 9. Faz 1 dondurulan kararları

- Araştırma sözleşmesi: `docs/RESEARCH_CONTRACT.md` ve `configs/research_contract.json`.
- İlk sistem: 20 cihaz, 3 edge, 1 cloud; olay indeksli 512 karar.
- İlk eylem uzayı: `local / edge_0 / edge_1 / edge_2 / cloud`.
- Semantik schema: `schemas/semantic_requirements.schema.json`.
- Ground truth protokolü: 20 pilot + 240 ana Türkçe görev, iki gerçek insan, tam çift-kör etiketleme ve adjudication.
- Trace kaynakları doğal ortak olay gibi birleştirilmeyecek; final benchmark `trace-driven-hybrid` diye adlandırılacak.
- BUPT lisans doğrulamasına, NEP-small erişim başvurusuna bağlıdır; UCI/EUA doğrulanmış başlangıç kaynaklarıdır.
- Qwen3-4B/Qwen3-8B nihai seçimi Faz 4 validation kapılarında yapılacaktır.
- Fiziksel testbed çekirdek Definition of Done koşulu değil, isteğe bağlı ek doğrulamadır.
- Faz 1 kanıtı: `phase_reports/PHASE_1_RESEARCH_CONTRACT.md`, 16/16 test geçti.

## 10. Faz 2 dondurulan kararları

- `src/task_offloading/sim/core.py::transition` kanal sonucu, gecikme, FCFS
  kuyruk, cihaz enerjisi, failure, constraint ve reward hesabının tek kaynağıdır.
- Domain nesneleri immutable ve SI birimleri alan adlarında açıktır; belirsiz
  fizik tuple'ları kullanılmaz.
- Bütün rastgelelik core dışında önceden örneklenmiş `ExogenousEvent` olarak
  verilir; adlandırılmış NumPy `SeedSequence` akışları çağrı sırasından bağımsızdır.
- Gymnasium ve event replay adaptörleri aynı geçiş fonksiyonunu çağırır; parity
  testi transition kayıtlarının ve final state'in birebir eşitliğini doğrular.
- Bilinen unavailability ve açık execution policy action mask kaynağıdır;
  gelecekteki failure maskeye sızmaz, unknown semantik hard mask üretmez.
- Mask-ablation eylem görünürlüğünü değiştirir; fiziksel failure ve hard
  constraint cezasını silmez.
- Reward Faz 1'de dondurulan negatif normalize bileşen toplamıdır; queue wait
  latency içindedir ve ikinci kez cezalandırılmaz.
- 34 Faz 2 testi ile unit/property/invariant/oracle/determinism/mask/parity/API
  kanıtları geçti; tüm 50 test, Ruff ve strict mypy temizdir; branch coverage
  yüzde 88'dir.
- Fizik denklemleri ve sınırlar `docs/SIMULATION_CORE.md`, faz kanıtı
  `phase_reports/PHASE_2_DETERMINISTIC_SIMULATION_CORE.md` içindedir.
- NVIDIA Quadro RTX 4000 (8 GB GDDR6) ikinci makine bilgisi Faz 4 planlama
  girdisi olarak `docs/COMPUTE_INVENTORY.md` içinde kaydedildi; model seçimi
  henüz yapılmadı.

## 11. Faz 3 dondurulan kararları

- Manifest sözleşmesi `schemas/dataset_manifest.schema.json`, kaynak kataloğu
  `configs/data_sources.v1.json` ve kullanım kuralları `docs/DATA_GOVERNANCE.md`
  içindedir.
- Alan kökeni sözlüğü tam olarak `observed / measured / derived / matched /
  generated / human_labeled / simulated` değerlerinden oluşur; her kullanılan
  kolon bu değerlerden biriyle işaretlenir.
- `downloaded` analize hazır demek değildir. Bir kaynağın analize açılması için
  immutable sürüm, doğrulanmış lisans, yerel SHA-256 kanıtı ve
  `schema_validated` statüsü birlikte gerekir.
- Kaynak yaşam döngüsü katalog/lisans/erişim bekleme durumlarından `downloaded`,
  `checksum_verified`, `schema_validated` veya `rejected` durumlarına kontrollü
  geçer; ileri statü yerel kanıt olmadan ilan edilemez.
- Split stratejileri kronolojik, cihaz, istasyon ve uygulama holdout'tur.
  Duplicate içerik hash'i, entity kesişimi, kronoloji ve örnek bütünlüğü ayrı
  denetlenir; normalizer yalnız train split'ine fit edilir.
- `synthetic_v1` ile `trace_driven_hybrid_v1` ayrı benchmark'lardır. Hibrit ad,
  farklı kaynakların doğal olarak ortak ölçülmüş olaylar olduğu iddiasını taşımaz.
- UCI MEC ve EUA başlangıç adayıdır; BUPT lisans, NEP-small erişim koşulu
  çözülmeden kullanılmaz. T-Drive ve Alibaba yalnız açık kısıtlarıyla ikincil
  adaydır.
- Faz 3 yazılım/sözleşme kapsamı 61 Faz 3 testiyle tamamlandı; tüm 111 test,
  Ruff ve strict mypy geçti, toplam branch coverage yüzde 91'dir. Henüz hiçbir
  ham kaynak indirilmediği veya `schema_validated` olmadığı için genel
  Definition of Done içindeki veri manifesti maddesi açık kalır.
- Faz 3 kanıtı `phase_reports/PHASE_3_DATA_PROVENANCE_AND_SPLITS.md` içindedir.
