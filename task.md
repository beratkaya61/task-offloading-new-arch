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

- [ ] Tez başlığını ve tek ana araştırma sorusunu dondur.
- [ ] Sistem sınırını yaz: tek/çok cihaz, edge sayısı, cloud, zaman modeli.
- [ ] MDP/POMDP tanımını yaz: durum, eylem, geçiş, ödül, horizon.
- [ ] Birim ve sembol tablosu oluştur.
- [ ] Semantik JSON Schema ve bilinmeyen/abstention politikasını oluştur.
- [ ] İnsan annotasyon kılavuzu ve anlaşmazlık çözümünü yaz.
- [ ] H1–H4 hipotezlerini, ana metrikleri ve istatistik planını önceden kaydet.
- [ ] Faz 1 testlerini/şema doğrulamasını çalıştır ve raporla.

## Faz 2 — Deterministik simülasyon çekirdeği

- [ ] Paketleme, minimal bağımlılıklar, lint/type/test araçlarını kur.
- [ ] Birimli domain tiplerini oluştur.
- [ ] Kanal, gecikme, kuyruk, enerji ve başarısızlık modellerini tek çekirdekte yaz.
- [ ] Gymnasium adaptörü ile event adaptörünü aynı çekirdeğe bağla.
- [ ] Seed/RNG sözleşmesini uygula.
- [ ] Unit, property/invariant, oracle ve parity testlerini geçir.
- [ ] Faz 2 raporunu yaz.

## Faz 3 — Veri kökeni ve split sistemi

- [ ] Veri manifest şeması: URL, sürüm, lisans, checksum, statü.
- [ ] Her kolon için observed/measured/derived/matched/generated/labeled etiketi.
- [ ] Train-only normalizer.
- [ ] Kronolojik, cihaz/istasyon ve uygulama holdout split’leri.
- [ ] Duplicate ve leakage testleri.
- [ ] Sentetik ve trace-driven-hybrid benchmark’ları ayrı adlandır.
- [ ] Faz 3 raporunu yaz.

## Faz 4 — Semantik benchmark

- [ ] Dengeli görev-niyet corpus’u hazırla.
- [ ] En az iki bağımsız insan etiketi ve adjudication yap.
- [ ] Etiketçi anlaşmasını hesapla.
- [ ] Sabit, rule, TF-IDF/logreg, küçük encoder, zero/few-shot LLM baselines.
- [ ] Schema validity, exact match, F1, calibration, abstention, latency/cost ölç.
- [ ] Seçilen LLM/prompt/parser/cache sürümünü dondur.
- [ ] Önceden tanımlı kalite kapısını geçir ve Faz 4 raporunu yaz.

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

## Faz 9 — Küçük testbed

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

**Faz 1:** Araştırma sözleşmesi, MDP ve semantik JSON Schema. Kod mimarisi ancak bunlar netleşince başlatılacak.

