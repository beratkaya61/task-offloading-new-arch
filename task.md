# Task Offloading New Architecture — Faz Bazlı Yol Haritası

Bu dosya günlük ilerleme listesidir. Bilimsel kurallar için `TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md`, denetim gerekçeleri için `phase_reports/PHASE_0_LEGACY_AUDIT_AND_LITERATURE_REVIEW.md` birlikte okunur.

## Çalışma kuralı

- Her işe başlamadan önce bu dosya ve bilimsel TODO kontrol edilir.
- Her faz sonunda ilgili testler çalıştırılır ve `phase_reports/` altında rapor yazılır.
- Test/kanıt tamamlanmadan kutu işaretlenmez.
- Eski depodan kod yalnız satır satır yeniden doğrulanarak taşınabilir; toplu kopyalama yapılmaz.
- Raw data, model checkpoint ve büyük deney çıktıları Git’e eklenmez.
- Kullanıcı istemeden commit atılmaz.

## Faz 0 — Denetim ve araştırma

- [x] Eski `task.md` ve TODO denetlendi.
- [x] `11_09_2026__chat.md` talimat değil, geçmiş bağlam olarak ayrıştırıldı.
- [x] Eski kod, ortamlar, veri hattı, semantik katman ve sonuçlar incelendi.
- [x] Eski testler çalıştırıldı: 23/23 geçti; eksik LLM/GUI bağımlılıkları kaydedildi.
- [x] AgentVNE ve yakın dönem MEC/LLM/semantic/DRL literatürü tarandı.
- [x] Açık veri setleri amaç ve eksik alanlarıyla karşılaştırıldı.
- [x] Faz 0 raporu yazıldı.

## Faz 1 — Araştırma sözleşmesi

- [x] Tez başlığını ve tek ana araştırma sorusunu dondur.
- [x] Sistem sınırını yaz: tek/çok cihaz, edge sayısı, cloud, zaman modeli.
- [x] MDP/POMDP tanımını yaz: durum, eylem, geçiş, ödül, horizon.
- [x] Birim ve sembol tablosu oluştur.
- [x] Semantik JSON Schema ve bilinmeyen/abstention politikasını oluştur.
- [x] İnsan annotasyon kılavuzu ve anlaşmazlık çözümünü yaz.
- [x] H1–H4 hipotezlerini, ana metrikleri ve istatistik planını önceden kaydet.
- [x] Trace-driven-hybrid veri rolleri, model seçim kapısı ve testbed sınırını dondur.
- [x] Faz 1 testlerini/şema doğrulamasını çalıştır ve raporla: 16/16 geçti.

## Faz 2 — Deterministik simülasyon çekirdeği

- [x] Paketleme, minimal bağımlılıklar, lint/type/test araçlarını kur.
- [x] Birimli domain tiplerini oluştur.
- [x] Kanal, gecikme, kuyruk, enerji ve başarısızlık modellerini tek çekirdekte yaz.
- [x] Gymnasium adaptörü ile event adaptörünü aynı çekirdeğe bağla.
- [x] Seed/RNG sözleşmesini uygula.
- [x] Unit, property/invariant, oracle ve parity testlerini geçir: 34/34 Faz 2 testi geçti.
- [x] Faz 2 raporunu yaz.

## Faz 3 — Veri kökeni ve split sistemi

- [x] Veri manifest şeması: URL, sürüm, lisans, checksum, statü.
- [x] Her kolon için observed/measured/derived/matched/generated/labeled etiketi.
- [x] Train-only normalizer.
- [x] Kronolojik, cihaz/istasyon ve uygulama holdout split’leri.
- [x] Duplicate ve leakage testleri.
- [x] Sentetik ve trace-driven-hybrid benchmark’ları ayrı adlandır.
- [x] Faz 3 raporunu yaz.

## Faz 4 — Semantik benchmark

- [x] Pilot Round 1 A/B/C gönderimlerini doğrula, checksum ile dondur ve A/B
  anlaşmasını C'den bağımsız hesapla: 20 görev, 160 alan, 39 A/B anlaşmazlığı.
- [x] Round 1 bulgularıyla protokolü `1.1.0` yap; confidence, sayısal kanıt,
  latency sınıfı ve execution status kalite kapılarını ekle.
- [x] A/B için eski sırayı ve birbirini tekrar etmeyen kör Round 2 paket/formlarını
  üret; Round 1 cevaplarını gizli tut.
- [x] A/B Round 2 gönderimlerini topla ve kilitle; strict alan uyumu yüzde
  83,75'e çıktı, anlaşmazlık 39'dan 26'ya düştü.
- [x] Round 2 kalite kapısını yeniden ölç: 7 kappa kontrolü başarısız ve 2 sınıf
  metriğinde örnek yetersiz; ana 240 görev bloke edildi.
- [x] Pilot cevaplarını göstermeyen yeni örneklerle hedefli annotator kalibrasyonu
  planını ve materyalini hazırla. İnsan doğrulama turu protokol `1.3.0` ile
  **bilinçli olarak iptal edildi/yerine geçti**; B/C'den yeni görev istenmeyecek,
  ana 60 kayıt tek-insan referansı olarak sınırlı raporlanacak.
- [x] Protokol `1.2.0` hazırlığını tarihsel olarak koru; insan toplama başlamadan
  `1.3.0` iş yükü düzeltmesini yap: A 60 kör görevi etiketler; B/C'den yeni görev
  istenmez, mevcut A/B/C pilot kanıtı ayrı kullanılır ve ana 60 gold sayılmaz.
- [x] Pilot cevaplarını içermeyen 20 örnekli hedefli kalibrasyon materyalini hazırla
  ve checksum ile bağla; insanın materyali okuması ve odaklı doğrulama henüz açık.
- [x] Dengeli görev-niyet corpus’unu hazırla: 240 görev, 8 domain × 30, 4 origin ×
  60 ve 180/60 aile-sızıntısız split doğrulandı. Ana 60 A tarafından etiketlenip
  checksum ile kilitlendi; kalan 180 kayıt insan doğrulamalı gösterilmeden makine
  zayıf etiketi olarak sınırlandı.
- [x] A için 60 görevlik tek kör form üret. 180 geliştirme kaydını insan doğrulamalı
  göstermeden makine zayıf etiketi olarak ayır. B/C için yeni ana-corpus formu
  üretme; mevcut pilot etiketlerini tanımlı rolleriyle koru.
- [x] Ana 60 görev için A'nın protokol 1.3.0 teslimini doğrula ve
  `single_human_reference` olarak kilitle; ek insan anketi isteme ve gold iddiası
  yapma.
- [x] Mevcut etiketçi anlaşmasını hesapla: Round 1 A/B ve A/B/C tanısal çiftleri,
  Round 2 A/B kalite kapısı sonuçları ayrı artifact'larda mevcut.
- [x] Sabit profil, rule/keyword ve TF-IDF + lojistik regresyon baselines'larını
  180 makine-zayıf geliştirme etiketiyle eğitip 60 kilitli tek-insan referansında
  ölç; değerlendirme etiketlerini eğitim/ayar amacıyla kullanma.
- [x] Küçük encoder baseline'ını aynı split ve metrik sözleşmesiyle ölç:
  donuk ve ince ayarlı multilingual-E5-small, sızıntısı giderilmiş v3'te tamamlandı.
- **Opsiyonel GPU genişletmesi:** Quadro RTX 4000 üzerinde Qwen3-4B/8B
  zero/few-shot ölçülebilir; ana kanıtı ve faz kapanışını bloke etmez.
- [x] Üç yerel baseline için schema validity, exact match, F1, calibration,
  abstention, latency/cost ölç ve tahmin/config/kod SHA-256 bağlarını üret.
- [x] Küçük encoder ve Qwen2.5-1.5B few-shot baseline için aynı metrikleri
  tamamla; Qwen3-0.6B yerel smoke yalnız teknik/uygulanabilirlik kanıtıdır.
- [x] Qwen2.5 baseline model/prompt/parser/cache sürümünü dondur.
- [x] Önceden tanımlı kalite kararlarını v3/LLM sonuçlarıyla raporla ve Faz 4
  kapanış raporunu yaz: `phase_reports/PHASE_4_SEMANTIC_BENCHMARK.md`. Bağımsız
  188 paraphrase, kapanış sonrası Faz 7 sağlamlık genişletmesidir.

## Paralel gerçek veri edinimi

- [x] BUPT commit'ini sabitle, indir, SHA-256 ve ZIP bütünlüğünü doğrula.
- [x] BUPT için privacy-minimum safe-prefix ayrıştırıcısı ve veri profili üret.
- [x] NEP erişim talebini gönder: 2026-09-14.
- [x] NEP-large/full erişimini al, artifact kimliğini ve SHA-256'yı doğrula: 2026-09-18.
- [x] NEP-large/full için tam üye CRC taramasını tamamla.
- [x] NEP-large/full için tam satır-şema/range profilini tamamla: 685.506.639
  satır tarandı; kabul politikası v1.0.0 ile 592.724.139 satır kabul, 92.782.500
  satır onarılmadan deterministik ret.
- [x] UCI MEC 859 artifact'ını indirip checksum/şema doğrula.
- [x] EUA reposunu commit SHA ile sabitleyip checksum/şema doğrula.

## Faz 5 — Baselines ve oracle

- [x] Always-local/edge/cloud, random ve valid-random; v1'de 3.840 kararda ölçüldü.
- [x] Latency/energy/deadline-aware greedy; semantiksiz ve semantik koşulları ölçüldü.
- [x] Küçük exhaustive oracle ve parity doğrulaması simülasyon çekirdeğinde mevcut.
- [x] Baskın eylem/collapse ve oracle çeşitlilik testleri mevcut; semantiksiz PPO'nun
  `edge_0` çöküşü v1 raporunda açıkça gösterildi.
- [x] Nihai NEP-v2 üzerinde reward bileşeni ve sensitivity tablosunu tamamla.
- [x] NEP-v2 sonrası resmî Faz 5 kapanış raporunu yaz. v1 ara kanıtı
  `phase_reports/OFFLOADING_RESULTS_V1.md` içinde tamamdır.

## Faz 6 — PPO ve semantik ablation

- [x] PPO-MLP semantiksiz ana baseline; v1'de 5 tohumla tamamlandı.
- [x] PPO + rule semantiği; kanonik ve paraphrase koşulları tamamlandı.
- [x] PPO + küçük sınıflandırıcı semantiği; TF-IDF ve E5 koşulları tamamlandı.
- [x] PPO + LLM semantiği. Qwen2.5'in 188 bağımsız paraphrase tahmin cache'i
  NEP-v2 içinde aynı olaylarda greedy/PPO ile değerlendirildi; küçük Qwen
  TF-IDF/E5'i geçmedi ve bu negatif sonuç açıkça raporlandı.
- [x] PPO + tek-insan referansı semantiği; `oracle` değil erişilebilir semantik
  üst sınır olarak adlandırıldı.
- [x] Aynı episode seed'leriyle 5 eğitim tohumu; nihai ana koşullar için 10 tohum
  kaynak bütçesine bağlı genişletmedir.
- [x] Collapse ve karar overhead testleri; ayrıntılı reward sensitivity NEP-v2'de açık.
- [x] v1 için CI, paired bootstrap ve etki farkları üretildi.
- [x] Confidence calibration/fallback eşiklerini yalnız development/calibration
  verisinde dondur; ECE/Brier ve güvenlik–başarı trade-off'unu yeniden ölç.
- [x] Exact-preview ve `sigma=0.5` noisy-preview altında greedy–PPO karşılaştırmasını
  aynı bilgi/seed sözleşmesiyle tamamla.
- [x] Nihai sistem tablosuna p50/p99, queue wait, throughput/drop, total ve
  per-success energy, fairness, action entropy/collapse ve decision overhead ekle.
- [x] Holm düzeltmeli NEP-v2 sonuçlarıyla resmî Faz 6 kapanış raporunu yaz:
  `phase_reports/PHASE_6_PPO_SEMANTIC_ABLATION_NEP_V2.md`.

## Faz 7 — İz güdümlü genelleme

- [x] Bağımsız 188 paraphrase üzerinde constant/rule/TF-IDF/E5/Qwen2.5 ortak
  semantik tablosunu ve uçtan uca etkiyi üret. Fine-tuned E5'in independent
  checkpoint'i bulunmadığı için bu eksik hücre sınırlama olarak raporlandı.
- [x] Zaman/entity/application-proxy/domain-shift değerlendirmeleri: BUPT
  kronolojik split bütün koşullarda; EUA 80/20 disjoint entity split ve iki-domain
  holdout yeni 5-seed retraining ile tamamlandı. Application sonucu yalnız
  disjoint scenario/paraphrase-family proxy'sidir, deployed application-ID değildir.
- [x] Kaynak katmanı ablation’ları: frozen `v2_nep` checkpoint'leriyle NEP, UCI
  compute heterogeneity, BUPT workload variability ve EUA topology variability
  evaluation counterfactual'ları tamamlandı; retraining sensitivity değildir.
- [x] Synthetic ↔ trace-driven yön tutarlılığı: H1/H2 yönü NEP-v2 katmanında
  korundu (H6 desteklendi).
- [x] Gerçek/sentetik/trace-driven-hybrid terminoloji kontrolü danışman belgesi ve
  v1 raporunda tamamlandı; simülasyon testbed olarak sunulmuyor.
- [x] Faz 7 raporu: `phase_reports/PHASE_7_GENERALIZATION_RESULTS.md`; entity ve
  domain ayrıntı tabloları ayrı raporlarda verilmiştir.

## Faz 8 — Gerekçeli ileri mimari (koşullu; aktif backlog değil)

- Değişken topoloji gerçekten gerekirse adil GNN-PPO açılır.
- Gizli durum kanıtlanırsa recurrent PPO açılır.
- Bölünebilir görevler doğrulanırsa partial offloading ve overhead açılır.
- Resource allocation gerçekten eylem alanına girerse inner solver/hibrit eylem açılır.
- Hiçbiri mevcut makale/tez çekirdeğinin tamamlanma koşulu değildir; açılırsa her
  özellik için ayrı ablation ve Faz 8 raporu gerekir.

## Faz 9 — İsteğe bağlı küçük testbed/emulation (aktif backlog değil)

Bu faz çekirdek tez kabul koşulu değildir. Yapılmazsa sonuçlar sentetik ve trace-driven-hybrid simülasyon olarak sınırlandırılır.

- Yapılmasına karar verilirse donanım/ağ envanteri, ortak zamanlı telemetri,
  ölçülmüş service/queue/network/energy, sim-to-real/OOD, ham log/manifest ve Faz 9
  raporu birlikte üretilir.

## Faz 10 — Tez ve yeniden üretilebilirlik

- [x] Sonuç tabloları/figürleri dondur.
- [x] Config, code version, data checksum ve seed bağını doğrula.
- [x] Advisor belgesinde yazılmış threats to validity, negative results ve
  limitations bölümünü nihai makale/tez metnine taşı ve son v2 kanıtıyla dondur.
- [x] Reproducibility paketi ve çalıştırma rehberi.
- **Opsiyonel demo:** GUI yalnız danışman/yayın demosu istenirse bilimsel çekirdekten
  ayrı tamamlanır; tez sonuçlarını bloke etmez.
- [x] Nihai test matrisi ve Faz 10 raporu:
  `phase_reports/PHASE_10_REPRODUCIBILITY_AND_QUALITY.md`.

## Kalan işler

**Current-PC publication core tamamlandı:** Faz 4–7 kanıtları, danışman MD/DOCX,
sonuç/figür/config–code–data–seed dondurması, read-only publication verification ve
nihai kalite kapısı tamamlandı; publication core `03dee8e` commit'iyle version
control altında donduruldu. Quadro Qwen3-4B/8B, 10-seed
genişletmesi, GNN/recurrent/partial offloading ve testbed ana yolu bloke etmeyen
opsiyonel veya koşullu işlerdir.
