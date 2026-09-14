# Faz 1 — Araştırma Sözleşmesi, MDP ve Semantik Şema Raporu

**Tarih:** 2026-09-13

**Faz durumu:** Tamamlandı

**Sürüm kontrolü:** Faz kapanış commit'i kullanıcı talebiyle oluşturulacak; push yapılmayacak

## 1. Amaç

Faz 1'in amacı kod mimarisine başlamadan önce tez iddiasını, başlangıç sistem sınırını, karar modelini, semantik çıktı biçimini, iki-insan annotasyon yöntemini ve deneysel başarı kapılarını dondurmaktı.

Faz 0 denetiminde eski çalışmanın ana sorununun model eksikliği değil; semantik doğruluk, fizik doğruluğu ve veri kökeni arasındaki zincirin kanıtlanmamış olması olduğu belirlenmişti. Bu faz, aynı belirsizliklerin yeni mimariye taşınmasını önleyen makinece test edilebilir bir sözleşme oluşturdu.

## 2. Kontrol edilen girdiler

- `TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md`
- `task.md`
- `phase_reports/PHASE_0_LEGACY_AUDIT_AND_LITERATURE_REVIEW.md`
- Kullanıcının güncel kararları:
  - fiziksel testbed çekirdek tez için zorunlu olmayacak,
  - deney trace-driven-hybrid olarak adlandırılacak,
  - BUPT yararlıysa kullanılacak fakat lisansı doğrulanacak,
  - semantik gold corpus iki gerçek insan tarafından bütünüyle bağımsız etiketlenecek,
  - ana LLM Qwen3-4B/Qwen3-8B pilotundan sonra seçilecek,
  - bütün geliştirme yalnız yeni repoda yapılacak,
  - kullanıcı izni olmadan commit veya push yapılmayacak.

## 3. Üretilen çıktılar

### 3.1 Araştırma sözleşmesi

`docs/RESEARCH_CONTRACT.md` içinde aşağıdakiler donduruldu:

- Türkçe ve İngilizce tez başlığı,
- tek ana araştırma sorusu,
- H1–H4 hipotezleri,
- izin verilen ve verilmeyen bilimsel iddialar,
- 20 cihaz, 3 edge ve 1 cloud başlangıç topolojisi,
- 512 görevlik olay indeksli episode,
- ilk ayrık eylem uzayı,
- durum, eylem maskesi, geçiş ve reward sözleşmesi,
- sembol ve SI birim tablosu,
- veri kökeni ve trace-driven-hybrid terminolojisi,
- model seçim kapıları,
- baseline, metrik ve istatistik planı,
- testbed'in isteğe bağlı uzantı olması,
- değişiklik/amendment prosedürü.

Aynı kararlar testlerin doğrudan okuyabilmesi için `configs/research_contract.json` dosyasına yazıldı.

### 3.2 Semantik JSON Schema

`schemas/semantic_requirements.schema.json`, JSON Schema Draft 2020-12 olarak oluşturuldu. Şema:

- LLM'nin eylem değil gereksinim döndürmesini,
- bütün nesnelerde beklenmeyen alanların reddedilmesini,
- `explicit`, `not_stated` ve `ambiguous` durumlarını,
- execution policy için `explicit`, `partial`, `not_stated`, `ambiguous` ayrımını,
- evidence ve confidence değerlerini,
- metinde olmayan alanların `null/unknown` kalmasını,
- deadline, olasılık ve skor aralıklarını

zorunlu kılar.

İki pozitif fixture eklendi:

- tüm gereksinimleri açık örnek,
- bütün semantik alanları doğru biçimde `not_stated` olan örnek.

### 3.3 İki-insan annotasyon protokolü

`docs/ANNOTATION_PROTOCOL.md` içinde şu süreç donduruldu:

- 20 pilot + 240 benzersiz ana Türkçe görev,
- bütün ana görevlerin iki gerçek insan tarafından bağımsız etiketlenmesi,
- kör sıra ve LLM önerisi göstermeme,
- yüzde 5 gizli tekrar,
- ham A/B cevaplarını kilitleme,
- yalnız anlaşmazlıkları uzlaştırma,
- çözülemeyen alanı `ambiguous` bırakma,
- nominal/ordinal/multi-label alanlara uygun anlaşma ölçümleri,
- minimum 0.70 ve hedef 0.80 alan bazlı kappa,
- scenario/paraphrase/entity grup sızıntısını engelleyen split.

İnsan emeğini düşürmek için yalnız metindeki semantik gereksinimler etiketlenecek; CPU, ağ, enerji veya oracle eylemi insanlar tarafından tahmin edilmeyecektir.

### 3.4 Veri kaynağı kararı

Kaynak rolleri sözleşmeye işlendi:

| Kaynak | Rol | Risk/koşul |
|---|---|---|
| BUPT | Geliş, byte, RAT, hücre geçişi, hizmet metadata | Açık lisans doğrulanmalı |
| NEP-small | Edge CPU/bant genişliği/RTT/kapasite | Başvuru ve yeniden dağıtım yasağı |
| UCI 859 | Ölçülmüş image-recognition turnaround | CC BY 4.0 |
| EUA | Gerçek kaynaklı kullanıcı/edge konumu | MIT |
| T-Drive veya DiDi | Sürekli mobilite | Erişim/lisans sonrası biri seçilecek |
| Alibaba v2018 | Cluster yükü/OOD stresi | Edge değil; yalnız ikincil |

İlişkisiz kaynakların satır bazında doğal ortak olay gibi birleştirilmesi yasaklandı. Her alan `observed/measured/derived/matched/generated/human_labeled/simulated` kökenlerinden biriyle işaretlenecektir.

## 4. Test yöntemi

Yerel ve Git tarafından yok sayılan `.venv` oluşturuldu. `requirements.txt` içindeki `jsonschema>=4.23,<5` bağımlılığı kuruldu.

Çalıştırılan komut:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 5. Test sonucu

**Sonuç: 16/16 test geçti.**

Doğrulanan ana özellikler:

- JSON Schema Draft 2020-12 yapısal olarak geçerli.
- Tam açık gereksinim fixture'ı geçerli.
- Tam `not_stated` fixture'ı geçerli.
- Beklenmeyen/recommended-action alanı reddediliyor.
- `explicit` etikette evidence zorunlu.
- `not_stated` latency ile sayısal deadline birlikte kabul edilmiyor.
- `not_stated` execution policy ile cloud yasağı birlikte kabul edilmiyor.
- `[0,1]` dışı confidence reddediliyor.
- LLM girdi izolasyonu sözleşmede korunuyor.
- İlk eylem uzayı yalnız local/üç edge/cloud.
- Annotasyon iki gerçek insan, tüm örnekler ve kör geçiş olarak tanımlı.
- BUPT `license_pending`, benchmark `trace_driven_hybrid`.
- Reward'da LLM uyum bileşeni yok.
- Minimum seed, yüzde 95 CI ve collapse alarmı önceden tanımlı.
- Testbed çekirdek iddia için zorunlu değil.

## 6. Başarısızlıklar ve yapılan düzeltmeler

- Sistem Python ortamında `jsonschema` bulunmadı. Bağımlılık `requirements.txt` dosyasına eklendi ve repo içindeki `.venv` ortamına kuruldu.
- Başlangıçta iki dilli 120 Türkçe + 120 İngilizce corpus düşünülmüştü. Bu, araştırma sorusuna zorunlu olmayan dil değişkeni ve annotator yükü eklediği için ana corpus 240 Türkçe görev olarak daraltıldı. İngilizce set isteğe bağlı robustness uzantısına taşındı.
- BUPT public görünmesine rağmen açık lisans dosyası saptanmadı. Kaynak reddedilmedi; `license_pending` kapısıyla koşullu tutuldu.
- Alibaba yerine gerçek edge yükü için NEP-small'ın ana aday olması kararlaştırıldı; Alibaba yalnız dağılım kayması/stres kaynağı olarak bırakıldı.

## 7. Sınırlamalar

- Henüz 240 görev üretilmedi ve insan annotasyonu yapılmadı; bu Faz 4 işidir.
- Annotatorların kim olacağı henüz kaydedilmedi. İki gerçek insan bulunmadan gold-dataset maddesi tamamlanamaz.
- BUPT lisansı ve NEP erişimi çözülmeden bu kaynaklar indirildi/doğrulandı sayılamaz.
- Reward ağırlıkları teorik başlangıç değerleridir; ana sonuçta sensitivity analizi zorunludur.
- 20 cihaz/3 edge ölçeklenebilirlik iddiası değildir; kontrollü başlangıç topolojisidir.
- Qwen3-4B veya Qwen3-8B henüz seçilmedi. Seçim yalnız Faz 4 validation sonuçlarıyla yapılacaktır.
- Fiziksel testbed yoktur; tez “gerçek deployment” iddiasında bulunamaz.
- Semantik sınıf eşikleri operasyonel tanımlardır. Pilot yalnız anlaşılabilirlik için kullanılabilir; ana/test sonuçlarına bakılarak değiştirilemez.

## 8. Faz kabul kontrolü

- [x] Tez başlığı ve tek araştırma sorusu donduruldu.
- [x] Sistem sınırı yazıldı.
- [x] MDP, durum, eylem, geçiş, reward ve horizon tanımlandı.
- [x] Birim/sembol tablosu oluşturuldu.
- [x] Semantik JSON Schema ve abstention politikası oluşturuldu.
- [x] İki-insan annotasyon ve anlaşmazlık çözüm protokolü yazıldı.
- [x] H1–H4, ana metrikler ve istatistik planı önceden kaydedildi.
- [x] Faz 1 testleri geçti: 16/16.
- [x] Girdi, yöntem, sonuç, başarısızlık, sınırlama ve sonraki riskler raporlandı.

## 9. Sonraki faz ve riskler

Sıradaki tek iş **Faz 2 — deterministik simülasyon çekirdeği**dir.

Öncelik sırası:

1. Python paket iskeleti ve kalite araçları,
2. birimli domain tipleri,
3. isimli `LinkMetrics`, `Task`, `Device`, `Server`, `Outcome` tipleri,
4. tek saf geçiş çekirdeği,
5. kanal, kuyruk, gecikme, enerji ve arıza modelleri,
6. Gym/event adaptör parity'si,
7. unit, property, oracle, determinism ve dimension testleri,
8. Faz 2 raporu.

Faz 2'de henüz gerçek veri veya LLM bağlanmayacaktır. Önce fizik çekirdeği elle hesaplanabilir örneklerde doğrulanacaktır.
