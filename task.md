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
- [ ] Pilot cevaplarını göstermeyen yeni örneklerle hedefli annotator kalibrasyonu
  ve odaklı doğrulama turu yap.
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
- [ ] Sabit, rule, TF-IDF/logreg, küçük encoder, zero/few-shot LLM baselines.
- [ ] Schema validity, exact match, F1, calibration, abstention, latency/cost ölç.
- [ ] Seçilen LLM/prompt/parser/cache sürümünü dondur.
- [ ] Önceden tanımlı kalite kapısını geçir ve Faz 4 raporunu yaz.

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

- [ ] Always-local/edge/cloud, random ve valid-random.
- [ ] Latency/energy/deadline-aware greedy.
- [ ] Küçük exhaustive veya MILP oracle.
- [ ] Baskın eylem ve oracle çeşitlilik testi.
- [ ] Reward bileşeni ve sensitivity testi.
- [ ] Faz 5 raporunu yaz.

## Faz 6 — PPO ve semantik ablation

- [ ] PPO-MLP semantiksiz ana baseline.
- [ ] PPO + rule semantiği.
- [ ] PPO + küçük sınıflandırıcı semantiği.
- [ ] PPO + LLM semantiği.
- [ ] PPO + insan-oracle semantiği.
- [ ] Aynı episode seed’leriyle en az 5, tercihen 10 eğitim seed’i.
- [ ] Collapse, reward decomposition ve karar overhead testleri.
- [ ] CI, paired test, effect size ve Faz 6 raporu.

## Faz 7 — İz güdümlü genelleme

- [ ] Zaman, entity, uygulama ve domain-shift değerlendirmeleri.
- [ ] Kaynak katmanı ablation’ları.
- [ ] Synthetic ↔ trace-driven yön tutarlılığı.
- [ ] Tam gerçek/hibrit terminoloji kontrolü.
- [ ] Faz 7 raporu.

## Faz 8 — Gerekçeli ileri mimari

- [ ] Değişken topoloji gerçekten gerekiyorsa adil GNN-PPO.
- [ ] Gizli durum varsa recurrent PPO.
- [ ] Bölünebilir görevler doğrulanırsa partial offloading ve overhead.
- [ ] Gerçek resource allocation gerekiyorsa inner solver/hibrit eylem.
- [ ] Her özellik için ayrı ablation ve Faz 8 raporu.

## Faz 9 — İsteğe bağlı küçük testbed/emulation

Bu faz çekirdek tez kabul koşulu değildir. Yapılmazsa sonuçlar sentetik ve trace-driven-hybrid simülasyon olarak sınırlandırılır.

- [ ] Donanım ve ağ envanteri.
- [ ] Ortak zamanlı görev-niyet-sistem-sonuç telemetrisi.
- [ ] Ölçülmüş hizmet süresi, queue, ağ ve mümkünse enerji.
- [ ] Sim-to-real kalibrasyon ve OOD testi.
- [ ] Ham log, manifest ve Faz 9 raporu.

## Faz 10 — Tez ve yeniden üretilebilirlik

- [ ] Sonuç tabloları/figürleri dondur.
- [ ] Config, code version, data checksum ve seed bağını doğrula.
- [ ] Threats to validity, negative results ve limitations yaz.
- [ ] Reproducibility paketi ve çalıştırma rehberi.
- [ ] GUI/gösterimi bilimsel çekirdekten ayrı tamamla.
- [ ] Nihai test matrisi ve Faz 10 raporu.

## Sıradaki tek iş

**Faz 4:** İnsan etiketleme kapandı; yeni anket istenmeyecek. Şimdi sabit profil,
rule/keyword, TF-IDF + lojistik regresyon ve küçük encoder semantik baseline'larını
aynı split ve metriklerle kur. Ardından Quadro RTX 4000 üzerinde Qwen3-4B/8B
zero-shot/few-shot karşılaştırma paketini hazırla. A'nın 180 geliştirme görevini
doldurması gerekmez.
