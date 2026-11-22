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

- [x] Tez sorusu ve hipotezleri dondurulmuş.
- [ ] Ana 60 semantik kayıt iki etiketçiyle adjudicated gold değildir. Tek-insan
  referansı kilitli, pilot A/B/C anlaşma analizleri mevcuttur; bu madde ancak
  ikinci uzman çalışması yapılırsa tamamlanabilir ve mevcut ana yolu bloke etmez.
- [x] LLM semantik çıkarıcı kilitli human-60 üzerinde ölçülmüş/cache'lenmiş;
  bağımsız 188 için ortak semantic ve greedy/PPO offloading değerlendirmesi
  tamamlanmıştır. Fine-tuned E5 independent checkpoint eksikliği ayrı sınırlamadır.
- [x] Tek fizik çekirdeği oracle/parity/invariant testlerini geçiyor.
- [x] Veri manifesti, lisans, checksum, alan kökeni ve sızıntısız split var.
- [x] Güçlü sabit/sezgisel/oracle baselines v1'de ve NEP-v2 nihai iz katmanında
  aynı fizik çekirdeğiyle tamamlandı.
- [x] Semantiksiz/rule/ML/LLM/single-human-reference politika karşılaştırmaları
  tamam; insan referansı `oracle` değil erişilebilir semantic upper bound'dır.
- [x] v1 ve NEP-v2 için 5 seed, %95 bootstrap GA, eşleştirilmiş test ve effect
  size mevcut; nihai ana kontrastların Holm düzeltmesi tamamlandı.
- [x] p50/p95/p99, queue, throughput/drop, total/per-success energy, constraints,
  action entropy/collapse, load fairness ve decision overhead nihai NEP-v2
  tablolarında raporlandı. Tek-source fairness'in yorumlanamazlığı açıklandı.
- [x] Sentetik ve hibrit trace sonuçlarının sınırları ayrı yazılmış; testbed
  yapılırsa ayrı ek doğrulama olarak raporlanacak.
- [x] GNN/partial/resource allocation uygulanmadığı için başlıkta veya katkı
  iddiasında kullanılmıyor; bunlar koşullu genişletmedir.
- [x] Bütün deneyler tek komut/config ve sürümlü manifestlerle yeniden üretilebilir.
- [x] Negative results ve threats to validity tezde açıkça bulunuyor.

## 8. Faz 0 kararı

Eski repo doğrudan taşınmayacaktır. Fizik hataları regresyon testine, eski veri üretim sorunları leakage/provenance testine, çökmüş politikalar collapse testine dönüştürülecektir. Yeni kod başlamadan önce Faz 1 araştırma sözleşmesi tamamlanmalıdır.

## 9. Faz 1 dondurulan kararları

- Araştırma sözleşmesi: `docs/RESEARCH_CONTRACT.md` ve `configs/research_contract.json`.
- İlk sistem: 20 cihaz, 3 edge, 1 cloud; olay indeksli 512 karar.
- İlk eylem uzayı: `local / edge_0 / edge_1 / edge_2 / cloud`.
- Semantik schema: `schemas/semantic_requirements.schema.json`.
- Ground truth için ilk plan: 20 pilot + 240 ana Türkçe görev, iki gerçek insan,
  tam çift-kör etiketleme ve adjudication. Bu tarihsel plan, annotator B'nin ana
  turu tamamlayamaması nedeniyle protokol `1.3.0` ile yürürlüğe girmeden önce
  değiştirilmiştir: ana kanıt 60 `single_human_reference`, 180 makine-zayıf
  development etiketi ve ayrı pilot anlaşma analizidir; gold iddiası yapılmaz.
- Trace kaynakları doğal ortak olay gibi birleştirilmeyecek; final benchmark `trace-driven-hybrid` diye adlandırılacak.
- BUPT yalnız resmî research-use beyanı ve no-redistribution sınırıyla yerel
  deneylere açılır. İlk kararda erişime bağlı olan NEP daha sonra NEP-large/full
  olarak alınmıştır; UCI/EUA açık kaynaklardır.
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
- UCI MEC, EUA, BUPT ve NEP-large/full doğrulanmış kaynaklardır. NEP'in
  685.506.639 satırının tamamı taranmış; değer ve PM-VM-site soyunu birlikte
  uygulayan kabul politikası v1.0.0 ile 592.724.139 satır kabul, 92.782.500 satır
  onarılmadan deterministik ret edilmiştir. T-Drive ve Alibaba yalnız açık
  kısıtlarıyla ikincil adaydır.
- Faz 3 yazılım/sözleşme kapsamı 61 Faz 3 testiyle tamamlandı; tüm 111 test,
  Ruff ve strict mypy geçti, toplam branch coverage yüzde 91'dir. Faz 3'ün ilk
  kapanışında ham kaynak yoktu; aşağıdaki Faz 3A kararıyla BUPT sonradan
  `schema_validated` oldu. Final hibrit veri/split artifact'ı tamamlanmadığı için
  genel Definition of Done içindeki veri manifesti maddesi açık kalır.
- Faz 3 kanıtı `phase_reports/PHASE_3_DATA_PROVENANCE_AND_SPLITS.md` içindedir.

## 12. Faz 3A BUPT edinim kararları

- NEP talebi 2026-09-14'te gönderildi ve NEP-large/full erişimi 2026-09-18'de
  alındı.
- BUPT kaynak commit'i `04e664fab9cdb2a58d04ebc615cd74405e6062e2`
  olarak sabitlendi; ham ZIP repo dışındaki merkezi veri kökünde tutulur.
- Yerel SHA-256
  `5d89bd853a5300207892dad38070957b935a95ea02bfe1ec063e0c4829e21822`
  olarak doğrulandı; ham artifact Git'e girmez.
- Resmî research-use beyanı yerel, ticari olmayan tez kullanımı için dayanak;
  ham ve satır seviyesinde türetilmiş verinin yeniden dağıtımı yasak kabul edilir.
- 482.687 satırın 476.419'u (%98,70) güvenli prefix şemasını geçti; 6.268 bozuk
  satır onarılmadan deterministik reddedilir.
- Ham URL, IP, User-Agent ve cihaz kimliği modele girmez. Cihaz/istasyon
  kimlikleri proje-yerel anahtarlı HMAC-SHA256 ile pseudonymize edilir.
- Profil `configs/data_profiles/bupt_04e664f.json`, erişim takibi
  `docs/DATA_ACCESS_REQUESTS.md`, kanıt raporu
  `phase_reports/PHASE_3A_BUPT_ACQUISITION.md` içindedir.

## 13. Faz 3B NEP-large/full edinim kararları

- 2026-09-14'te NEP için gönderilen erişim talebinin ardından proje sahibi
  2026-09-18'de erişim iznini ve `Full_trace.7z` artifact'ını aldı.
- Dosya adı “BUPT” klasöründe olsa da artifact, altı CSV üyesi, 7.410 VM,
  139 VM sitesi ve `VM_BW_THREE_MONTHS.csv` varlığıyla BUPT mobil izi değil,
  **EdgeWorkloadsTraces NEP-large/full** olarak tanımlandı.
- 5.885.170.081 byte arşivin yerel SHA-256 değeri
  `ff80a07e25f8055fed1a64439194a47721d9bc2330c79c9a8201649072852816`
  olarak doğrulandı. Raw artifact ve erişim yazışmaları Git'e girmez.
- Resmî research-only ve offline paylaşmama koşulu değişmez. Erişim alınmış
  olması yeniden dağıtım izni anlamına gelmez.
- NEP-large/full; CPU, bant genişliği, site RTT ve PM/VM kapasite replay'i için
  ana gerçek iz kaynağıdır. BUPT mobil iziyle satır numarasına göre birleştirilmez;
  yalnız açık zaman-penceresi/eşleme kurallarıyla hibrit senaryo kurulur.
- SHA-256 ve tam üye CRC taraması geçmiştir. DuckDB 1.5.5 ile altı üyenin
  685.506.639 satırı taranmış; kabul politikası v1.0.0 yalnız değerleri geçerli
  ve PM-VM-site soyu eksiksiz 592.724.139 satırı analize açmıştır. Kaynak
  `schema_validated` durumundadır; 92.782.500 reddedilen satır onarılmaz.
- Kabul sayacı kesinti sonrası dosya bazında devam edebilir; 17 NEP testi dahil
  tüm 180 test, Ruff ve strict mypy geçmiştir; toplam branch coverage yüzde 92'dir.

## 14. Faz 4 pilot Round 1 ve Round 2 kararları

- Pilot Round 1'de A, B ve tanısal C'nin 20'şer kaydı şema, görev hash'i ve
  metin-içi kanıtla doğrulanıp checksum ile donduruldu. C birincil A/B kappa'ya
  veya otomatik çoğunluk kararına katılmadı.
- A/B, 160 semantik alanın 121'inde aynı, 39'unda farklı karar verdi; strict alan
  uyumu yüzde 75,625, tam kayıt uyumu 2/20 oldu. Round 1 gold değildir ve
  adjudication yapılmadı.
- Round 1 protokolü `1.0.0` anlık görüntüsüyle birlikte korundu. Ana veri toplama
  başlamadan protokol `1.1.0` olarak açıkça amend edildi.
- `1.1.0`; confidence varsayılanını kaldırır, sayısal kanıt/değer tutarlılığını,
  latency eşik sınıfını ve execution `explicit/partial` bütünlüğünü formda zorlar;
  domain, privacy, reliability ve accuracy ayrımlarını kılavuzda kesinleştirir.
- Round 2 yalnız A/B ile, yeni ve birbirinden farklı kör sıralarda yürütülür.
  Round 1 cevapları ve anlaşmazlık listesi gösterilmez; C tekrar edilmez.
- Round 2 kalite kapısı geçmeden adjudication'a veya 240 görevlik ana annotasyona
  başlanmaz.
- Round 2 A/B gönderimleri protokol `1.1.0` sunucu kalite kapılarıyla doğrulanıp
  checksum ile kilitlendi. Strict alan uyumu yüzde 75,625'ten 83,75'e, tam kayıt
  uyumu 2/20'den 6/20'ye çıktı; A/B anlaşmazlığı 39'dan 26'ya düştü.
- İyileşmeye rağmen Round 2 kapısı geçmedi: domain status, privacy value,
  reliability status, accuracy status, energy value, execution status ve device
  permission kappaları `0.70` altındadır. Reliability/accuracy value kappaları
  için ortak explicit örnek sayısı da yetersizdir.
- Bu sonuç adjudication veya gold değildir. Ana 240 görev, gerçek pilot
  cevaplarını göstermeyen hedefli eğitim/kalibrasyon ve yeni doğrulama kanıtı
  tamamlanana kadar bloke kalır.

## 15. Faz 4 annotasyon protokolü 1.2.0 ve ana corpus hazırlığı

- İkinci annotatorın sürekli bulunamaması nedeniyle, ana etiketleme başlamadan
  protokol `docs/ANNOTATION_PROTOCOL_AMENDMENT_1_2.md` ile amend edildi. Round 1
  ve Round 2 ham verileri ve başarısız Round 2 kapısı tarihsel kanıt olarak aynen
  korunur; geçmiş sonuç başarıya çevrilmez.
- Ana 240 görevin 180'i makine ön etiketli ve A tarafından tek-insan doğrulamalı
  development verisidir; gold diye adlandırılmaz. Etiketlerden önce seçilmiş 60
  görev A/B tarafından bağımsız ve kör etiketlenip uzlaştırılır; yalnız bu bölüm
  `adjudicated_gold` olabilir.
- 20 yeni cevaplı eğitim örneği sekiz semantik alanı kapsar; pilot metinleri veya
  A/B cevapları yeniden kullanılmaz. Paket eğitim materyalidir, insan etiketi ya
  da ground truth değildir.
- Ana corpus taslağı 240 görev, 120 ikili senaryo/paraphrase ailesi, domain başına
  30 ve `standard_derived / trace_conditioned / llm_assisted_synthetic /
  adversarial` origin başına 60 görev olarak üretildi.
- 180/60 split, insan veya model etiketi kullanılmadan, scenario/paraphrase
  ailelerini iki tarafa sızdırmadan donduruldu. BUPT yalnız güvenli aggregate
  input-bit quantile'larıyla görev büyüklüğünü koşullar; satır verisi ve semantik
  etiket sağlamaz.
- Bütün 240 metin `human_text_review=pending` durumundadır. A'nın 60 kör ve 180
  assisted formu hazırdır; A'nın geçerli gönderimleri metin inceleme kanıtı
  olmadan B taslak formu dağıtılamaz.
- B için yalnız tek 60 görevlik kör paket planlanmıştır. İş yükünü azaltmak için
  gizli tekrar yoktur; annotator-içi tekrar güvenilirliğinin ölçülememesi nihai
  sınırlılıklarda açıkça yazılacaktır.

## 16. Faz 4 annotasyon protokolü 1.3.0 iş yükü düzeltmesi

- İnsan ana-corpus etiketlemesi başlamadan annotator B'nin 60 görevi
  tamamlayamayacağı kesinleşti. Protokol `docs/ANNOTATION_PROTOCOL_AMENDMENT_1_3.md`
  ile tekrar amend edildi; geçmiş pilot ve 1.2 hazırlık kanıtları değiştirilmedi.
- A yalnız 60 kör değerlendirme görevini etiketler; B veya C'den yeni görev
  istenmez. Ana 60 kayıt `single_human_reference`, 180 geliştirme kaydı ise
  `machine_generated_weak_label` olarak açıkça ayrılır; ikisi de adjudicated gold
  değildir.
- Round 1 A/B/C ve Round 2 A/B kayıtları korunur. C'nin 20 etiketi pairwise
  sensitivity ve A/B uyuşmazlık teşhisinde kullanılır; yeni 60 görevin etiketiymiş
  gibi taşınmaz ve otomatik çoğunluk oyu yapılmaz.
- İki insan 60 kaydı etiketlemiş gibi bir iddia yapılamaz. Uyum yalnız mevcut pilot
  turlarında ölçülür; küçük örneklem, protokol farkı, prevalans ve ana corpus'un
  tek-insan olması nihai sınırlılıklarda raporlanır.
- Annotator A'nın 60/60 protokol `1.3.0` kaydı sunucu denetiminden geçti ve
  `7eecd3d8efd42b1606911729cbd96706cc9b4b55560fb78fabb8a1f37dae3b08`
  SHA-256 ile `single_human_reference` olarak kilitlendi. Bu kayıt gold diye
  sunulmaz; ek A/B/C anketi planlanmaz.

## 17. Faz 4 yerel semantik baseline kararları

- Sabit profil, görünür metinle çalışan Türkçe rule/keyword ayrıştırıcı ve
  karakter TF-IDF + lojistik regresyon baseline'ları aynı sözleşmeyle kuruldu.
  Yalnız `task_text` ve `user_policy_text` modele açıktır; design metadata,
  provenance, grouping, ağ durumu ve insan referansı model girdisi değildir.
- Modeller yalnız 180 `machine_generated_weak_label` geliştirme kaydıyla fit
  edilir. Kilitli 60 `single_human_reference` kaydı modellerin tahminleri
  üretildikten sonra yüklenir ve yalnız değerlendirmede kullanılır.
- Dondurulan ilk sonuçlarda schema validity üç modelde de `1.0` oldu. Sabit
  profil/rule/TF-IDF için sırasıyla macro-F1 `0.266678 / 0.712914 / 0.698611`,
  tam alan doğruluğu `0.758333 / 0.897917 / 0.879167` ve tam kayıt doğruluğu
  `0.033333 / 0.350000 / 0.266667` ölçüldü.
- Yüksek tam-alan doğruluğu tek başına başarı kanıtı değildir: referansta
  `not_stated` alanları yaygındır. Bu nedenle macro-F1, alan bazlı sonuçlar,
  abstention ve kritik gereksinim recall'u birlikte okunur. 60 referansta
  kritik latency pozitif örneği bulunmadığından bu recall tanımsızdır; Faz 4
  sınırlılığı olarak korunur.
- Config, girdi, şema, uygulama, tahmin ve sonuç SHA-256 bağları
  `data/semantic_benchmark/main/baselines/manifest.v1.json` içindedir. Çalıştırma
  `scripts/run_semantic_baselines.py` ile tekrarlanır.
- Qwen bu bilgisayarda çalıştırılmaz. Sıradaki işler aynı değerlendirme
  sözleşmesiyle küçük encoder baseline'ı ve Quadro RTX 4000'e taşınabilir,
  cache'li Qwen3-4B/8B zero-shot/few-shot deney paketidir.
- Küçük encoder için MIT lisanslı `intfloat/multilingual-e5-small`, commit
  `0e60b8d9d2166d80387f86e3b48ec9ced55f4d15` ile seçildi. Encoder dondurulur,
  model kartının doğrusal sınıflandırma önerisine uygun `query:` önekli 384 boyutlu
  normalize embedding üzerinde dengeli lojistik başlıklar fit edilir. Kod, config
  ve model indirmeyen testler hazırdır; mevcut makinede paket indeksi kurulumu
  tamamlanmadığı için gerçek encoder sonucu henüz üretilmiş sayılmaz.

## 18. 20–21 Kasım 2026 kanıt eşitlemesi

Bu bölüm, yukarıdaki tarihsel Faz 4 durumunu yeni commit ve dondurulmuş sonuçlarla
günceller. Bölüm 17'deki “encoder henüz üretilmedi” ve “bu bilgisayarda Qwen
çalıştırılmaz” ifadeleri artık güncel iş durumunu temsil etmez; tarihsel karar
kaydı olarak korunur.

- Sızıntısı giderilmiş semantik benchmark v3 tamamlandı. İnsan-60 özgün sette
  rule, TF-IDF(+veri), E5-donuk(+veri) ve E5-ince-ayar(+veri) macro-F1 değerleri
  sırasıyla `%78,0 / %78,2 / %76,3 / %70,2`; görülmemiş kompozisyonel ifadede
  `%47,5 / %73,6 / %70,9 / %73,0` oldu. Kanıt:
  `phase_reports/SEMANTIC_RESULTS_V3.md`.
- Qwen2.5-1.5B 4-shot CPU baseline tamamlandı. İnsan-60 özgün/paraphrase
  macro-F1 `%48,3 / %43,7`; ortalama süre yaklaşık `14,7 / 16,0` saniye/görev.
  Sonuç S3(a)'yı desteklemedi, S3(b) gecikme hipotezini destekledi.
- Offloading v1, 60 bölüm × 64 görev ve 5 PPO tohumu ile tamamlandı. İnsan
  referanslı PPO, semantiksiz PPO'ya göre başarıyı `+15,2` yüzde puanı artırdı;
  TF-IDF paraphrase + fallback `%87,7` başarı ve `%0` hard ihlal verdi. PPO'nun
  greedy'yi geçmesi şimdilik desteklenmedi. Kanıt:
  `phase_reports/OFFLOADING_RESULTS_V1.md`.
- NEP full-trace aggregate quantile profili üretildi:
  `configs/data_profiles/nep_quantiles_v1.json`. Profil 276.084.449 geçerli CPU
  satırı ve 330.710.309 geçerli RTT satırından türetildi; ham satır içermez.
- Sonuç sonrası tüm yeni iyileştirmeler confirmatory v2'den ayrıldı ve
  `docs/EXPLORATORY_IMPROVEMENT_PROTOCOL_V1.md` altında önceden donduruldu.
- Yerel LLM genişletmesinde pinned `Qwen/Qwen3-0.6B` non-thinking smoke teknik
  olarak geçti: 3 özgün örnekte parse/OOV hatası yok, fakat ortalama yaklaşık
  289 saniye/görev. Küçük örnek kalite kanıtı değildir; tam CPU koşusu
  uygulanabilir bulunmadı. Quadro Qwen3-4B/8B deneyi opsiyonel genişletmedir.

## 19. Danışman raporu ve güncel kapanış hedefi

- Danışmana sunulacak güncel yöntem/literatür/sonuç belgesi
  `TEZ_DANISMANI_MEVCUT_DURUM_VE_LITERATUR_OZETI.md` olarak 21 Kasım 2026'da
  yeniden yazıldı. Yöntem ve metrik gerekçeleri numaralı birincil literatür
  atıflarıyla, sonuçlar tablo ve yeniden üretilebilir şekillerle açıklandı.
- Faz 4 ana semantik benchmark'ı, bağımsız 188 paraphrase sağlamlık genişletmesi,
  Faz 5/6 NEP-v2 ve Faz 7 trace/entity/domain/source-layer deneyleri resmî
  raporlarıyla tamamlanmıştır. Danışman MD/DOCX belgesi bu kanıtla eşitlenmiş,
  Faz 10 reproducibility/nihai kalite kapısı kapatılmıştır.
- Ana yayın iddiası “LLM ile ilk edge scheduling” değildir. Güncel ve savunulabilir
  çerçeve, paraphrase dayanıklılığı ve kalibrasyonu ölçülen semantik kısıt
  katmanının aynı MEC olaylarında başarı/hard-ihlal etkisini nedensel olarak
  ayırmasıdır.

## 20. 21 Kasım 2026 NEP-v2 Faz 5/6 kapanışı

- Bağımsız 188 için constant/rule/TF-IDF/E5/Qwen2.5 ortak semantic tablosu ve
  Qwen dahil greedy/PPO downstream etkisi üretildi. TF-IDF(+data) independent
  macro-F1 `%80,91`, Qwen2.5 4-shot `%42,03` oldu; küçük LLM güçlü ML
  baseline'ını geçmedi.
- Development-only isotonic calibration uygulandı; raw ve calibrated prediction
  provenance ayrıldı. Fallback threshold evaluation set üzerinde ayarlanmadı.
- NEP-v2 exact-preview ve `sigma=0.5` noisy-preview koşuları 5 PPO seed × 150.000
  step ile tamamlandı. H1, H2 ve H6 desteklendi; H3, H4 ve H5 desteklenmedi.
- Reference semantics exact-preview'da success'i no-semantic'e göre PPO'da
  `+15,03`, greedy'de `+12,94` yüzde puan artırdı. PPO'nun güçlü greedy
  baseline'ı geçtiği iddiası desteklenmedi.
- p50/p99, queue wait, throughput/drop, total/per-success energy, reward
  decomposition, action entropy/collapse, Jain load fairness ve decision
  overhead ortak tabloları tamamlandı. Tek source-device nedeniyle source-success
  fairness mekanik olarak 1'dir ve user-fairness kanıtı değildir.
- Reward ağırlıkları için dondurulmuş trajectories üzerinde `0,5×/2×` post-hoc
  sensitivity tamamlandı; bunun PPO retraining sensitivity olmadığı belirtildi.
- Resmî kapanış kanıtları
  `phase_reports/PHASE_5_BASELINES_ORACLE_NEP_V2.md` ve
  `phase_reports/PHASE_6_PPO_SEMANTIC_ABLATION_NEP_V2.md` içindedir.

## 21. 22 Kasım 2026 Faz 7 genelleme kapanışı

- Dondurulmuş protokol `configs/generalization_protocol.v1.json` ile BUPT
  kronolojik split, EUA disjoint 80/20 entity split, disjoint
  scenario/paraphrase-family proxy ve iki-domain holdout değerlendirildi.
- Entity ve domain holdout yeni 5 PPO seed × 150.000 step retraining kullanır.
  PPO-reference − PPO-no-semantic success farkı entity'de `+15,03` puan
  [%95 CI `+13,82; +16,25`], domain'de `+13,41` puandır
  [%95 CI `+12,37; +14,51`]; iki durumda da reference hard violation `%0,00`dır.
- Domain holdout'ta PPO-reference, güçlü greedy-reference'dan `−0,52` puan
  düşüktür [%95 CI `−0,98; −0,08`, Holm `p=0,023`]; PPO üstünlüğü iddiası yine
  desteklenmez.
- Frozen `v2_nep` checkpoint'leriyle NEP, UCI compute heterogeneity, BUPT
  workload variability ve EUA topology variability source-layer
  counterfactual'ları tamamlandı. En büyük shift UCI heterogeneity çıkarıldığında
  PPO-reference success'te `−8,92` puandır; bunlar retraining sensitivity veya
  nedensel kaynak katkısı değildir.
- Application kanıtı gerçek deployed application-ID holdout değil, disjoint
  scenario/paraphrase-family proxy'sidir. Domain sonucu frozen extractor yerine
  reference semantics koşullu policy generalization kanıtıdır.
- Resmî kanıt `phase_reports/PHASE_7_GENERALIZATION_RESULTS.md`; ayrıntılı entity
  ve domain tabloları ilgili `OFFLOADING_RESULTS_V2_NEP_*_HOLDOUT.md`
  raporlarındadır. Phase 7 kalite kapısı 9 test, Ruff ve strict mypy ile geçti.

## 22. 22 Kasım 2026 Faz 10 reproducibility ve kalite kapanışı

- `configs/publication_suite.v1.json`, current-PC zorunlu stages ile
  `quadro_optional` genişletmeyi ayırır. Varsayılan
  `scripts/reproduce_publication.py` read-only doğrulama yapar; training yalnız
  explicit `--execute`, frozen overwrite ise ayrıca `--allow-frozen-overwrite`
  gerektirir.
- `configs/publication_evidence.v1.json`; runtime versions ile config, code, data
  manifests, results, reports, figures ve advisor document SHA-256 bağlarını
  dondurur. Varsayılan publication verification `issues=[]` ile geçti.
- Repository-wide final gate: `264 passed`; Ruff `All checks passed`; strict mypy
  `src tests scripts` için `111 source file, 0 issue`.
- Canonical advisor Markdown ile DOCX visible-text parity ve dört embedded figure
  doğrulandı. Negative results, threats to validity ve claim sınırları belgede
  açıkça korunur.
- Resmî kanıt ve clean-checkout rehberi sırasıyla
  `phase_reports/PHASE_10_REPRODUCIBILITY_AND_QUALITY.md` ve
  `REPRODUCIBILITY.md` içindedir.
- Bu bilgisayardaki zorunlu publication core tamamlandı. Kullanıcı manuel commit
  atacaktır; Quadro Qwen3-4B/8B, 10-seed extension ve testbed opsiyoneldir.
