# Faz 3 — Veri Kökeni ve Sızıntısız Split Sistemi Raporu

**Tarih:** 2026-09-14

**Faz durumu:** Yazılım/sözleşme kapsamı tamamlandı; kullanıcı incelemesi ve
manuel commit bekleniyor

**Sürüm kontrolü:** Faz 2 commit'i `a1f98af`; bu fazda commit veya push yapılmadı

## 1. Amaç

Faz 3'ün amacı, ileride kullanılacak sentetik ve gerçek-kaynaklı izlerin nereden
geldiğini, ne kadarının ölçülmüş veya üretilmiş olduğunu ve train/validation/test
arasında bilgi sızıntısı olmadığını makinece kanıtlayan katmanı kurmaktı.

Bu faz “dosyayı indirebildik” ile “bilimsel analize hazır” kavramlarını ayırdı.
Faz kapanış anında hiçbir ham dataset hazır değildi; sonraki Faz 3A ediniminde
BUPT artifact'ı sabitlendi ve doğrulandı. Güncel kanıt ayrı Faz 3A raporundadır.

## 2. Kontrol edilen girdiler

- `TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md`
- `task.md`
- `configs/research_contract.json`
- `docs/RESEARCH_CONTRACT.md`
- `docs/SIMULATION_CORE.md`
- Faz 0–2 raporları
- UCI, EUA, BUPT, NEP-small, Microsoft T-Drive ve Alibaba resmî sayfaları

Faz 1'de dondurulan kurallar korundu: unrelated kaynaklar doğal ortak ölçüm gibi
birleştirilmeyecek, normalizer yalnız train'e fit edilecek, ham veri Git'e
girmeyecek ve ana benchmark `trace-driven-hybrid` diye adlandırılacaktır.

## 3. Üretilen çıktılar

### 3.1 Manifest şeması ve kaynak kataloğu

- `schemas/dataset_manifest.schema.json`
- `configs/data_sources.v1.json`

Şema her kaynak için resmî URL, citation, sürüm, immutable/mutable bilgisi,
erişim statüsü, dataset statüsü, lisans kanıtı, redistribution, artifact,
SHA-256, kolon kökeni, dönüşüm ve raw-Git yasağını zorunlu kılar.

Katalog altı kaynak içerir:

| Kaynak | Güncel statü | Ana karar |
|---|---|---|
| UCI MEC 859 | `catalogued` | CC BY 4.0; küçük ilk calibration kaynağı |
| EUA | `catalogued` | MIT; commit SHA ile pinlenecek |
| BUPT | `schema_validated` (Faz 3A) | Yerel research-only kullanım; yeniden dağıtım yok |
| NEP-small | `access_pending` | Talep 2026-09-14'te gönderildi; yanıt bekleniyor |
| T-Drive | `catalogued` | Non-commercial; yeniden dağıtım yok |
| Alibaba v2018 | `access_pending` | İkincil/OOD; MEC değil; yaklaşık 49 GB |

Alibaba için resmî yayımlanmış archive checksum kaydedildi fakat yerel dosya
olmadığı için statü `published`, `verified` değildir.

### 3.2 Provenance sözlüğü

Yedi değer kod ve JSON Schema'da aynı şekilde donduruldu:

```text
observed, measured, derived, matched,
generated, human_labeled, simulated
```

Mevcut katalogda her normalize hedef kolonuna tam bir provenance, kaynak alanı,
birim ve dönüşüm açıklaması yazıldı. BUPT byte→bit dönüşümü `derived`, UCI
turnaround değeri `measured`, kaynaklar arası senaryo birleşimi `matched` olarak
ayrıldı.

### 3.3 Checksum ve yaşam döngüsü

`data/manifest.py` şunları uygular:

- büyük dosyaları parça parça SHA-256 hesaplama,
- beklenen checksum ile eşitlik denetimi,
- statü kapılarını atlamayı reddetme,
- kaynak/kolon/artifact/benchmark kimliği benzersizliği,
- mutable ref'in immutable diye sunulmasını reddetme,
- `schema_validated` iddiası için immutable sürüm + lisans + yerel checksum,
- synthetic ve trace-driven-hybrid katmanlarının semantik doğrulaması.

### 3.4 Train-only normalizer

`TrainOnlyStandardizer`:

- yalnız `DataSplit.TRAIN` ile fit olur,
- validation/test ile fit çağrısını reddeder,
- sessiz refit'i reddeder,
- NaN/sonsuz ve şekil hatalarını reddeder,
- sabit feature için scale 1 kullanır,
- feature isimleri, mean, scale, sample count ve `fitted_on=train` bilgisini
  serileştirilebilir istatistikte saklar.

### 3.5 Split ve leakage sistemi

Uygulanan değerlendirme görünümleri:

- kronolojik holdout,
- cihaz holdout,
- istasyon/site holdout,
- uygulama holdout.

Aynı timestamp grubu bölünmez. Grup holdout seed'i plan içinde kaydedilir ve
girdi sırası değişse bile aynı atama üretilir. Farklı büyüklükteki gruplarda
satır sayısı hedef orana yaklaştırılır, fakat grup bütünlüğü bozulmaz.

Leakage audit:

- canonical içerik SHA-256'sının splitler arası tekrarını,
- seçilen grup alanının splitler arası kesişimini,
- kronolojik zaman sınırı ihlalini,
- sample ID plan/veri uyuşmazlığını

tespit edip fazı başarısız kılar.

### 3.6 Ham veri dizinleri

`data/raw`, `data/interim` ve `data/processed` dizinleri `.gitkeep` ile
oluşturuldu. `.gitignore` ham/işlenmiş içerikleri ve büyük artifact'ları dışarıda
tutar; yalnız dizin iskeleti sürümlenir.

## 4. Test matrisi ve sonuç

Faz 3 test sınıfları:

| Sınıf | Kanıt |
|---|---|
| Manifest | Draft 2020-12 + semantik doğrulama, altı kaynak, iki benchmark |
| Lisans/statü | Public≠licensed; downloaded≠validated; kapı atlama reddi |
| Checksum | Streaming SHA-256, eşleşme/mismatch, biçim ve chunk guard'ları |
| Provenance | Her kolonda dondurulmuş tek değer, hybrid katman zorunluluğu |
| Normalizer | Train fit, validation reuse, validation/test fit ve refit reddi |
| Chronology | Eş timestamp bölünmez, train zamanı validation/testten önce |
| Group holdout | Device/station/application ayrık, seed deterministik |
| Duplicate/leakage | Cross-split içerik ve manuel grup sızıntısı yakalanır |
| Regression | Faz 1 ve Faz 2 testleri aynen geçer |

Son temiz sonuç:

- **61/61 Faz 3 testi geçti.**
- **111/111 toplam test geçti.**
- **Ruff:** temiz.
- **Strict mypy:** 16 source dosyasında hata yok.
- **Toplam branch coverage:** yüzde 91.
- Yeni veri modülleri: manifest yüzde 95, normalizer yüzde 100, splitting yüzde
  98 branch coverage.

## 5. Geliştirme sırasında bulunan sorunlar

1. Strict mypy, `jsonschema` tip bilgisini bulamadı. Güncel
   `types-jsonschema>=4.26,<5` geliştirme bağımlılığına eklendi.
2. İlk yeni test turu 32/32 geçti; incelemede grup sayısına göre bölmenin büyük
   cihaz gruplarında oranları bozabileceği görüldü. Split sınırları grup
   bütünlüğünü koruyarak satır sayısına yakın dengeleyecek şekilde düzeltildi.
3. Checksum statüsü için yalnız JSON biçim kontrolünün yeterli olmadığı görüldü.
   `checksum_verified/schema_validated` iddiaları yerel `verified` kanıtına
   bağlandı.
4. Yeni checksum gate'i mevcut negatif testte daha erken ve daha doğru hata
   verdi; test beklenen genel `evidence` sınıfını kontrol edecek biçimde
   düzeltildi.
5. İlk kod biçimi Ruff tarafından mekanik olarak düzenlendi; nihai lint temizdir.

## 6. Sınırlamalar ve dürüstlük sınırı

- Bu fazın ilk kapanışı veri yönetim sistemini tamamladı; BUPT edinimi daha sonra
  `PHASE_3A_BUPT_ACQUISITION.md` ile kanıtlandı.
- Şu anda yalnız BUPT `schema_validated` statüsündedir; bu statü yalnız yerel,
  ticari olmayan araştırma kullanımı içindir.
- UCI/EUA kullanılmadan önce artifact indirimi, checksum ve ham şema doğrulaması
  gereklidir.
- BUPT resmî research-use beyanı altında ana benchmark'a girebilir; ham veya
  satır seviyesinde türetilmiş veri yeniden dağıtılamaz.
- NEP-small erişim talebi kullanıcı/kurum tarafından tamamlanmalıdır; raw veri
  paylaşılamaz ve Git'e giremez.
- T-Drive yalnız non-commercial kullanım ve no-redistribution koşuluyla adaydır.
- Alibaba v2018 edge/MEC ölçümü değildir; birincil gerçeklik kaynağı olamaz.
- Kaynaklar arası eşleştirme henüz yapılmadı; yapılınca zaman penceresi, seed ve
  eşleştirme kuralı artifact'a yazılmalıdır.
- Semantik family split alanları Faz 4 corpus tipleri eklendiğinde aynı denetim
  yapısına bağlanacaktır.

Genel Definition of Done içindeki “veri manifesti, lisans, checksum, alan kökeni
ve sızıntısız split var” maddesi henüz kapatılmadı. BUPT artifact kanıtı artık
mevcuttur; ancak final hibrit benchmark'ın UCI/EUA/NEP katmanları ve gerçek split
artifact'ı henüz üretilmedi.

## 7. Faz kabul kontrolü

- [x] URL, sürüm, lisans, checksum ve statü içeren manifest şeması oluşturuldu.
- [x] Altı kaynağın güncel ve dürüst başlangıç kataloğu oluşturuldu.
- [x] Her katalog kolonuna frozen provenance yazıldı.
- [x] Train-only normalizer ve serializable istatistikleri uygulandı.
- [x] Kronolojik, device, station ve application holdout uygulandı.
- [x] Duplicate, group leakage, time leakage ve ID bütünlük denetimleri uygulandı.
- [x] Synthetic ve trace-driven-hybrid benchmark adları/kısıtları ayrıldı.
- [x] Faz 3 testleri geçti ve raporlandı.

## 8. Sonraki faz ve riskler

Yol haritasındaki sıradaki faz **Faz 4 — semantik benchmark**tır. Başlangıç
sırası:

1. 20 görevlik pilot corpus veri modelini ve family-ID yapısını kurmak,
2. annotator A/B için kör ve bağımsız form/export şemasını üretmek,
3. rule ve sabit profil baselinelarını kurmak,
4. TF-IDF/logreg ve küçük encoder baseline hattını kurmak,
5. Qwen3-4B/8B zero/few-shot çıktısını frozen cache'e almak,
6. schema validity, F1, critical recall, calibration, abstention ve latency/VRAM
   kapılarını validation üzerinde değerlendirmek.

Gerçek veri katmanı için paralel açık işler unutulmayacaktır: BUPT yerel
research-only kullanıma açılmıştır; UCI/EUA edinimi ve NEP erişim yanıtı hâlâ
beklemektedir.
