# Faz 3 Veri Kökeni, Doğrulama ve Split Sözleşmesi

## 1. Bu katman neden gerekli?

Bir CSV dosyasının internette bulunması, onun güvenilir, lisanslı veya model
eğitimine hazır olduğu anlamına gelmez. Faz 3 veri katmanı her sayı için şu
zinciri görünür kılar:

```text
resmî kaynak
  → lisans ve sürüm
  → indirilen dosyanın SHA-256 değeri
  → ham şema doğrulaması
  → kolonun kökeni ve dönüşümü
  → leakage-safe split
  → yalnız train üzerinde fit edilen normalizer
```

Bu zincirin bir halkası eksikse kaynak “analize hazır” sayılamaz.

## 2. Yedi kolon-kökeni etiketi

Her normalize edilmiş kolon tam olarak bir etikete sahip olur:

| Etiket | Anlamı | Örnek |
|---|---|---|
| `observed` | Kaynak dosyada doğrudan bulunan kayıt | BUPT start time |
| `measured` | Fiziksel sistem/ölçüm sürecinden gelen değer | UCI turnaround süresi |
| `derived` | Aynı kaynaktan deterministik hesaplanan değer | uplink byte × 8 = input bit |
| `matched` | Ayrı kaynaklardan açık kuralla eşleştirilen değer | BUPT geliş penceresi + NEP yük penceresi |
| `generated` | İnsan/LLM/kural ile oluşturulan içerik | görev metni paraphrase'i |
| `human_labeled` | İnsan annotator tarafından verilen gold etiket | privacy sınıfı |
| `simulated` | Fizik çekirdeğinin ürettiği değer | sentetik kanal/failure sonucu |

`matched`, “aynı gerçek olay ölçüldü” demek değildir. Yalnız ayrı kayıtların
senaryo kurmak için belirli bir yöntemle eşleştirildiğini söyler.

## 3. Manifest ve statü kapıları

Makinece doğrulanan şema `schemas/dataset_manifest.schema.json`, güncel kaynak
kataloğu `configs/data_sources.v1.json` içindedir. Her kaynakta URL, citation,
sürüm, sürümün immutable olup olmadığı, erişim, lisans, artifact, SHA-256,
kolon kökeni ve Git politikası zorunludur.

Kaynak yaşam döngüsü kanıt atlanmadan ilerler:

```text
catalogued / license_pending / access_pending
  → downloaded
  → checksum_verified
  → schema_validated
```

`downloaded`, yalnız dosyanın diskte olduğunu söyler. `schema_validated` olmak
için immutable sürüm, doğrulanmış lisans ve her artifact için yerelde doğrulanmış
SHA-256 gerekir. Resmî sayfada yazan checksum `published` statüsündedir; dosya
indirildikten sonra aynı değer yerelde üretilmeden `verified` olmaz.

## 4. Kaynakların 2026-09-14 durumu

| Kaynak | Rol | Durum | Neden hazır değil? |
|---|---|---|---|
| UCI 859 | Ölçülmüş edge turnaround kalibrasyonu | `catalogued` | Dosya henüz indirilip hash/şema doğrulanmadı |
| EUA | Avustralya kullanıcı/edge konumu | `catalogued` | Mutable `master`, commit SHA ve artifact hash gerekli |
| BUPT | Oturum, byte, RAT, hücre, servis metadata | `schema_validated` | Yerel research-only; ham/türev satır dağıtımı yok |
| NEP-small | Edge CPU/bant/RTT/kapasite replay | `access_pending` | Talep 2026-09-14'te gönderildi; yanıt bekleniyor |
| T-Drive | Mobility stress | `catalogued` | Non-commercial ve yeniden dağıtım yasaklı |
| Alibaba 2018 | İkincil cluster/OOD stresi | `access_pending` | Survey/erişim ve lisans kapsamı; MEC değil |

UCI 859 resmî sayfası 4.000 ölçüm ve CC BY 4.0 bildirir:
<https://archive.ics.uci.edu/dataset/859/image+recognition+task+execution+times+in+mobile+edge+computing>

EUA resmî reposu gerçek dünya kaynaklı Avustralya konumlarını ve MIT lisansını
gösterir: <https://github.com/PhuLai/eua-dataset>

BUPT resmî reposu 480 binden fazla mobil kayıt ve 22 alan tarif eder. Sabit README
veriyi Edge Computing araştırmaları için herkese açık yayımladığını ve ilgili
makaleye atıf istediğini belirtir. Genel yeniden dağıtım izni bulunmadığı için
proje bunu yalnız yerel research-only kullanım olarak yorumlar:
<https://github.com/BuptMecMigration/Edge-Computing-Dataset>

2026-09-14'te commit `04e664f...` indirildi. Yerel SHA-256 doğrulaması ve ham
şema profili `configs/data_profiles/bupt_04e664f.json` içinde kayıtlıdır. 482.687
satırın 476.419'u güvenli ilk 18 alanı geçti; kalan 6.268 satır deterministik
olarak reddedilir. Ham URL/IP/User-Agent/kimlik değerleri işlenmiş kayda girmez.

NEP-small resmî açıklaması 14 edge site, Haziran 2020, beş dakikalık kayıtları;
yalnız araştırma kullanımını ve offline paylaşım yasağını bildirir:
<https://github.com/xumengwei/EdgeWorkloadsTraces>

T-Drive sample 10.357 taksinin bir haftalık izidir ve MSR lisansı
non-commercial kullanımla sınırlayıp yeniden dağıtımı yasaklar:
<https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/User_guide_T-drive.pdf>

Alibaba 2018 yaklaşık 4.000 makinenin sekiz günlük cluster izidir. Resmî sayfa
yaklaşık 49 GB sıkıştırılmış boyutu ve SHA-256 değerini yayımlar; bu çalışma için
yalnız ikincil stres kaynağıdır:
<https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md>

## 5. Split yöntemleri

### 5.1 Kronolojik split

Eski zamanlar train, daha yeni zamanlar validation, en yeni zamanlar test olur.
Aynı timestamp'e sahip satırlar iki tarafa bölünmez. Böylece geleceğin dağılımı
geçmiş eğitime sızmaz.

### 5.2 Device holdout

Bir cihaz yalnız train, validation veya testten birinde bulunur. Modelin aynı
cihazı ezberlemek yerine yeni cihaza genellemesi ölçülür.

### 5.3 Station holdout

Aynı baz istasyonu/edge site tek splitte tutulur. Yeni lokasyon veya siteye
genelleme sınanır.

### 5.4 Application holdout

Aynı uygulama ailesi tek splitte tutulur. Modelin gördüğü uygulama adını ezberleyip
semantik başarı iddia etmesi engellenir.

Grup holdout'ları seed ile deterministik sıralanır. Sınırlar yalnız grup sayısını
değil grup içindeki satır sayılarını da dikkate alarak hedef yüzde 60/20/20'ye
yaklaşır. Hiçbir grup parçalanmaz; bu nedenle oranların tam eşit olması garanti
edilmez ve gerçekleşen oran raporlanmalıdır.

## 6. Duplicate ve leakage denetimi

Her örnek, canonical JSON içeriğinin SHA-256 değeriyle izlenir. Aynı içerik farklı
sample ID ile train ve testte bulunursa cross-split duplicate olarak reddedilir.
Grup stratejisinde aynı cihaz/istasyon/uygulamanın iki splitte bulunması;
kronolojik stratejide zaman aralıklarının üst üste binmesi de reddedilir.

Semantik corpus geldiğinde `scenario_family_id`, `paraphrase_family_id` ve
`source_entity_id` ayrı grup alanları olarak aynı ilkeyle genişletilecektir.

## 7. Train-only normalizer

Normalizer ortalama ve standart sapmayı yalnız train satırlarından öğrenir:

```text
train mean = 1, train std = 1
validation values = 100, 102
validation transformed = 99, 101
```

Validation kendi ortalamasıyla tekrar sıfıra merkezlenirse validation dağılımı
modele sızmış olur. `TrainOnlyStandardizer` bu nedenle validation/test ile fit
etmeyi ve fit edilmiş nesneyi sessizce yeniden fit etmeyi reddeder. Sıfır
varyanslı train kolonu güvenli biçimde scale `1` kullanır.

## 8. Benchmark adlandırma kuralı

- `synthetic_v1`: yalnız `generated/simulated`; gerçek veya trace-driven iddiası
  yoktur.
- `trace_driven_hybrid_v1`: en az bir `observed/measured` katman ve en az bir
  `matched/generated/simulated` katman içermek zorundadır.

İkinci benchmark hiçbir zaman “tamamen gerçek aynı olay kayıtları” diye
sunulmayacaktır. Kaynak katmanı ablation'ları hangi sonucun hangi gerçek veri
katmanına bağlı olduğunu ayrıca gösterecektir.

## 9. Ham veri edinildiğinde uygulanacak sıra

1. Lisans ve erişim koşulunu yeniden kontrol et.
2. Mutable Git kaynağını commit SHA ile dondur.
3. Dosyayı yalnız `data/raw/` altına indir; Git'e ekleme.
4. Boyut ve SHA-256 hesapla, manifestteki yayımlanmış değer varsa karşılaştır.
5. Ham kolon, tip, birim, eksik ve sentinel değerlerini doğrula.
6. Statüyü kapıları atlamadan ilerlet.
7. Normalize edilmiş her kolona tek provenance ve açık transform yaz.
8. Önce split üret; sonra normalizer'ı yalnız train üzerinde fit et.
9. Split planı, normalizer istatistikleri, kaynak checksum ve kod commit'ini
   birlikte deney artifact'ına yaz.
