# Annotasyon Protokolü 1.2.0 Eki

**Yürürlük tarihi:** 2026-09-29

**Temel protokol:** `docs/ANNOTATION_PROTOCOL.md` (`1.1.0`)

**Kapsam:** Ana 240 görevlik corpus; geçmiş pilot kayıtları değişmez.

## 1. Neden bir ek gerekiyor?

Round 2, A/B strict alan uyumunu yüzde 75,625'ten yüzde 83,75'e çıkardı; ancak
7 kappa kapısı geçilemedi ve reliability/accuracy değerleri için ortak explicit
örnek sayısı yetersiz kaldı. Aynı 20 görevi tekrar tekrar etiketletmek öğrenme ve
hatırlama etkisi üretir. Ayrıca ikinci annotatorın 240 görevin tamamına sürekli
erişimi pratik değildir.

Bu ek, insan emeğini azaltırken final değerlendirme kümesinin bağımsız çift
etiket ve uzlaştırma güvencesini korur. Küçük ama önceden seçilmiş bir alt kümeyi
çift etiketlemek; anlaşmayı açıkça ölçmek, zor sınıfları ayrıca denetlemek ve
tek-annotator verisini gold diye sunmamak koşuluyla kabul edilen bir maliyet
azaltma stratejisidir.

## 2. Değişmeyen ilkeler

- A ve B iki gerçek insandır; yapay zekâ personası insan gibi raporlanamaz.
- Pilot Round 1 `1.0.0`, Round 2 `1.1.0` ile yorumlanır; ham cevaplar değiştirilmez.
- LLM çıktısı final test alt kümesindeki insan kararlarına gösterilmez.
- `not_stated` doğru bir etikettir; eksik bilgi tahmin edilmez.
- Alan kappa'ları, yüzde uyum, örnek sayısı ve anlaşmazlıklar raporlanır.
- Uzlaştırılmamış veya tek insan tarafından doğrulanmış kayıtlar
  `adjudicated_gold` olarak adlandırılmaz.

## 3. Ana corpus için insan emeği tasarımı

Ana corpus 240 benzersiz Türkçe görevden oluşur ve etiketleme başlamadan önce
iki role ayrılır:

1. **Assisted development — 180 görev:** Sürüm ve yöntem kimliği kaydedilmiş bir
   makine ön etiket önerir; annotator A her alanı okuyup kabul eder veya düzeltir.
   İlk sürüm, yapılandırılmış authoring şablonlarından deterministik öneri üretir;
   ileride LLM kullanılırsa model/prompt/cache ayrıca dondurulur. Bu kayıtların
   provenance etiketi `machine_preannotated_single_human_verified` olur. Eğitim
   ve geliştirmede kullanılabilir, fakat adjudicated gold değildir.
2. **Double-blind evaluation — 60 görev:** Annotator A ve B aynı görevleri farklı
   kör sıralarda, birbirinin ve LLM'nin cevabını görmeden etiketler. Ham dosyalar
   kilitlenir, anlaşma ölçülür ve yalnız anlaşmazlıklar uzlaştırılır. Yalnız bu
   alt kümenin uzlaştırılmış hali `adjudicated_gold` olabilir.

60 görev etiketlerden önce, sabit seed ile ve yalnız corpus tasarım metadata'sı
kullanılarak seçilir. Seçim; sekiz domain'i, kaynak ailesini, negation,
contradiction, missing-information, numeric-threshold ve explicit-policy gibi
semantik olguları kapsayacak şekilde tabakalıdır. A veya LLM etiketi seçimde
kullanılamaz.

Annotator B için iş akışı tek planlı paket olarak hazırlanır: kısa kalibrasyon ve
60 benzersiz kör görev. Ham cevap kilitlendikten sonra mümkünse aynı oturumda
yalnız anlaşmazlıkların kısa uzlaştırması yapılır. Yeni bir pilot turu otomatik
olarak istenmez. İkinci annotator emeğini en aza indirmek için gizli tekrar
eklenmez; bu nedenle annotator-içi tekrar güvenilirliğinin ölçülememesi açık bir
sınırlılık olarak raporlanır.

## 4. Veri bölümü ve iddia sınırı

- 180 assisted-development görevi model/prompt geliştirmesinde kullanılabilir.
- 60 double-blind görev, model ve prompt seçimi tamamlanana kadar kapalı final
  değerlendirme kümesidir.
- Final 60 görev için üretilmiş LLM ön etiketleri annotatorlara gösterilmez ve
  insan ground truth yerine geçmez.
- Model karşılaştırmalarının ana semantik sonuçları 60 görevlik adjudicated gold
  üzerinde raporlanır.
- 180 görev için yalnız `single-human-verified` kalite iddiası yapılır.

Bu ayrım, daha çok fakat zayıf etiketli geliştirme verisi ile daha küçük fakat
bağımsız doğrulanmış değerlendirme verisini birbirine karıştırmaz.

## 5. Round 2 sonrasında kesinleştirilen karar kuralları

### 5.1 Domain

- Görev nesnesini açıkça belirten sektör sözcüğü domain için yeterli kanıttır.
  “Yoğun bakım monitörü”, “üretim hattı” veya “sürüdeki hayvan” denmişse domain
  `not_stated` değildir.
- Otobüs, tren, araç, yol ve trafik öncelikle `transportation`; belediye çevre
  sensörü ve sokak altyapısı `smart_city`; ev asistanı ve kişisel uygulama
  `consumer`; hayvan/tarla/sulama `agriculture` olur.
- Bir alan adı açıkça bulunuyorsa annotatorın etiketten emin olmaması tek başına
  `ambiguous` gerekçesi değildir. `ambiguous`, metnin iki etiketi gerçekten
  desteklemesini gerektirir.

### 5.2 Privacy

- “Kişisel veri” ve kişi konumu → `sensitive`.
- Ham klinik kayıt, biyometri veya açıkça “kısıtlı sağlık verisi” → `restricted`.
- Kişisel/klinik içerik söylenmeden yalnız kurum içinde kalan işletme kaydı →
  `internal`.
- Şifreleme veya cloud izni privacy sınıfını düşürmez. Execution policy ayrı
  alandır.

### 5.3 Reliability ve accuracy

- İsteğin tamamlanması, paket kaybı, bağlantı kopması ve hizmet başarısızlığı →
  reliability.
- Yanlış sınıflandırma, yanlış çeviri, yanlış negatif ve F1/accuracy skoru →
  accuracy.
- Aynı cümlede hem “yanlış sonuç” hem “başarısız istek” geçiyorsa iki alan da
  ayrı ayrı explicit olabilir.
- “Tek bir kayıp kabul edilemez” mission-critical olabilir fakat sayı yoksa
  `min_success_probability=1.0` uydurulmaz.
- “Yüksek doğruluk” çelişki yoksa `high` ve `explicit`tir; sayısal metrik
  verilmemesi onu ambiguous yapmaz.

### 5.4 Energy

- Pil ömrünü en çok koruma veya enerji tasarrufunu öncelik yapma → `high`.
- `balanced` yalnız enerji ile gecikme/doğruluk/deneyim arasında açık bir denge
  istendiğinde kullanılır.

### 5.5 Execution policy

- Bir hedef hakkında bilgi yoksa `unknown` kalır; başka hedefin izninden sonuç
  çıkarılmaz.
- “Edge izinli, cloud yasak” → device `unknown`, edge `allowed`, cloud
  `forbidden`, status `partial`.
- “Veri cihazdan çıkamaz” → device `allowed`, edge/cloud `forbidden`, status
  `explicit`.
- “Yerel sistemler” tek başına device ile kurum içi edge'i ayırmaz. Cloud yasağı
  açık ise cloud `forbidden`, diğer iki hedef `unknown`, status `partial` olur.
- Aynı hedef hem zorunlu/izinli hem yasaksa status `ambiguous` ve üç hedef
  `unknown` olur; cümlelerden biri görmezden gelinmez.

### 5.6 Doğrudan çelişki

Sayısal değer normalde sınıfı belirler; fakat aynı alan için açıkça bağdaşmayan
iki gereksinim varsa çelişki önceliklidir. Örneğin “anlık olmalı” ile “4 saniye
tamamen kabul edilir” birlikteyse latency `ambiguous` olur. Aynı kural accuracy
ve divisibility için de geçerlidir.

## 6. Kalite kapısı

60 görevlik ham A/B alt kümesinde:

- overall strict field agreement en az `0.80`,
- ana status/değer ve execution hedef kappaları en az `0.70`,
- sınıf/değer kappası için iki annotatorın da explicit dediği en az 5 örnek,
- sınıf prevalansı ve alan başına gerçek örnek sayısı ayrıca raporlanır.

Kappa prevalence nedeniyle tanımsız veya yanıltıcıysa sınıf dağılımı, yüzde uyum
ve bootstrap güven aralığı birlikte verilir. Kapı geçmezse sonuç saklanmaz:
problemli alanlar sınırlılık olarak raporlanır ve tek-insan verisi gold'a
yükseltilmez.

## 7. Yöntem dayanağı

- Artstein ve Poesio, chance-corrected agreement katsayılarının varsayımlarını ve
  corpus annotasyonunda uygun raporlamayı açıklar:
  https://aclanthology.org/J08-4004/
- Grouin ve arkadaşları, küçük bir alt kümede çift annotasyonla yönerge
  anlaşılabilirliğini kurup kalan emeği hedefli kullanmanın maliyet azaltan bir
  strateji olduğunu deneysel olarak gösterir:
  https://aclanthology.org/W14-4907/
- Klie, Eckart de Castilho ve Gurevych; annotator yönetimi, agreement,
  adjudication ve veri doğrulamasının birlikte raporlanmasını önerir:
  https://aclanthology.org/2024.cl-3.1/
- Oortwijn, Ossenkoppele ve Betti, gold üretiminde anlaşmazlıkların sistematik
  biçimde çözülmesini savunur:
  https://aclanthology.org/2021.humeval-1.15/

## 8. Sürümleme

Bu ek ana corpus başlamadan önce yayımlanmıştır. `1.1.0` ile çelişen ana-corpus
iş yükü ve split hükümlerinde bu `1.2.0` eki önceliklidir; diğer bütün hükümler
`1.1.0` metnindeki haliyle yürürlüktedir.
