# İki İnsanlı Semantik Annotasyon Protokolü

**Sürüm:** 1.0.0

**Bağlı şema:** `schemas/semantic_requirements.schema.json`

**Ana corpus:** 240 benzersiz Türkçe görev metni

**Annotator:** 2 gerçek insan; bütün ana örnekler çift etiketlenecek

## 1. Amaç

Bu çalışma, bir görev metninde açıkça bulunan MEC gereksinimlerini çıkarmak için güvenilir bir insan ground truth'u oluşturur. Annotator bir offloading eylemi seçmez. Annotator yalnızca metindeki gereksinimi kaydeder.

Örneğin annotator şu soruyu cevaplamaz:

> Bu görev edge'e mi gönderilmeli?

Onun yerine şunları cevaplar:

> Metinde bir gecikme sınırı var mı? Cloud yasak mı? Veri hassas mı? Başarı olasılığı istenmiş mi?

Bu ayrım, insan ve LLM etiketlerini fiziksel “en iyi eylem”den bağımsız tutar.

## 2. Annotator koşulları

- İki annotator gerçek ve birbirinden bağımsız kişilerdir.
- Her ikisi de Türkçe görev metinlerini anlayabilmelidir.
- Yapay zekâ veya MEC uzmanlığı gerekmez; bu kılavuzdaki kavram eğitimi yeterlidir.
- Annotator kimliği yayımlanan veride `annotator_a` ve `annotator_b` olarak takma kimlikle tutulur.
- Arka plan bilgisi yalnızca toplu biçimde raporlanır: eğitim seviyesi, teknik alan deneyimi ve dil yeterliği.
- Kurumsal etik/insan katılımcı prosedürü gerekiyorsa veri toplamadan önce danışman veya etik kuruldan teyit alınır.

Bir yapay zekâ, iki farklı persona veya iki farklı model “iki insan” olarak raporlanamaz.

## 3. Annotator ne görecek?

Annotator yalnızca şunları görür:

1. görev açıklaması,
2. varsa ayrı kullanıcı/kurum politikası,
3. alanlara ait kısa tanım ve seçenekler.

Annotator şunları görmez:

- LLM cevabı,
- diğer annotatorın cevabı,
- ağ ve sunucu durumu,
- simülatör sonucu,
- oracle eylemi,
- train/validation/test üyeliği,
- metni üreten şablon veya model.

## 4. Temel etiketleme ilkesi

Yalnızca metnin desteklediği bilgi etiketlenir.

- Açık veya doğrudan dilsel kanıt varsa `explicit`.
- Hiç bilgi yoksa `not_stated`.
- Metin kendi içinde çelişkili veya birden fazla makul etikete izin veriyorsa `ambiguous`.
- Execution policy'nin yalnız bir bölümü biliniyorsa `partial`.
- `not_stated`, düşük kalite veya hata değildir; doğru abstention etiketidir.

Annotator genel dünya bilgisinden gizli kısıt üretmez. “Sağlık görevi” sözü tek başına “cloud yasak” anlamına gelmez. “Acil” sözü latency için kanıt olabilir, fakat sayısal bir `max_ms` değeri uydurulamaz.

## 5. Alan tanımları

### 5.1 Domain

| Etiket | Operasyonel anlam |
|---|---|
| `healthcare` | Hasta, klinik, tıbbi cihaz veya sağlık hizmeti |
| `industrial` | Fabrika, üretim, robot, kalite kontrol veya endüstriyel bakım |
| `transportation` | Araç, yol, trafik, lojistik veya toplu taşıma |
| `public_safety` | Afet, yangın, kolluk, acil müdahale veya güvenlik |
| `smart_city` | Belediye altyapısı, aydınlatma, çevre veya kent hizmeti |
| `agriculture` | Tarla, sera, hayvan, sulama veya tarımsal izleme |
| `consumer` | Kişisel eğlence, ev veya genel tüketici uygulaması |
| `other` | Yukarıdaki alanların dışında açıkça tanımlanan görev |
| `null` | Alan belirtilmemiş veya belirsiz |

Domain, offloading izni üretmez.

### 5.2 Latency

Önce metindeki açık `max_ms` değeri çıkarılır. Saniye verilmişse arayüz milisaniyeye dönüştürür. Sayısal sınır yoksa yalnızca nitel ifade etiketlenir.

| Etiket | Kılavuz |
|---|---|
| `relaxed` | Sonucun bir saniyeden daha geç gelmesi açıkça kabul edilebilir; batch/deferred iş |
| `interactive` | Yaklaşık 200–1000 ms veya kullanıcı etkileşimli fakat güvenlik-kritik olmayan iş |
| `real_time` | Yaklaşık 50–200 ms veya hızlı gerçek zamanlı geri bildirim |
| `hard_real_time` | 50 ms altı açık sınır ya da kaçırılması güvenliği/işlevi bozan sert süre |
| `null` | Gecikme gereksinimi yok veya çelişkili |

Sayısal sınır varsa sınıf arayüz tarafından bu eşiklerle önerilebilir; ancak öneri kaydedilmeden önce annotator tarafından onaylanır. LLM cevabı hiçbir zaman gösterilmez.

### 5.3 Privacy veri sınıfı

| Etiket | Operasyonel anlam |
|---|---|
| `public` | Açıkça kamusal veya hassas olmayan veri |
| `internal` | Kurum/ev içinde kalması tercih edilen fakat kişisel sır içermeyen veri |
| `sensitive` | Kişisel, ticari veya konum gibi korunması gereken veri |
| `restricted` | Sağlık kaydı, biyometri, güvenlik görüntüsü, ticari sır gibi açıkça çok kısıtlı veri |
| `null` | Veri hassasiyeti belirtilmemiş veya çelişkili |

Privacy sınıfı ve execution policy ayrıdır. Hassas veri şifreli biçimde cloud'a gönderilebilir; metin cloud'u yasaklamıyorsa annotator yasağı tahmin etmez.

### 5.4 Execution policy

Her hedef için ayrı değer seçilir:

```text
allowed, forbidden, unknown
```

Hedefler:

- `device`: görev kaynağındaki cihaz,
- `edge`: yakın MEC/edge sunucu,
- `cloud`: uzak merkezi cloud.

Örnekler:

- “Veri cihazdan çıkmamalı” → device `allowed`, edge/cloud `forbidden`, status `explicit`.
- “Buluta gönderilmemeli” → cloud `forbidden`, device/edge `unknown`, status `partial`.
- Hedef hakkında cümle yok → üçü de `unknown`, status `not_stated`.
- “Yalnız edge'de çalışsın fakat gerekirse cloud kullanılabilir” bağlama göre çelişkiliyse üçü `unknown`, status `ambiguous`.

### 5.5 Reliability

| Etiket | Operasyonel anlam |
|---|---|
| `best_effort` | Başarısızlık tolere edilebilir |
| `standard` | Normal hizmet beklentisi; yaklaşık `%90–98` açık hedef |
| `high` | Düşük hata toleransı; yaklaşık `%98–99.9` açık hedef |
| `mission_critical` | Çok düşük hata toleransı; `>= %99.9` veya açık kritik görev |
| `null` | Güvenilirlik belirtilmemiş veya çelişkili |

Yüzde verilmişse `min_success_probability` 0–1 aralığına dönüştürülür. “Hasta izleme” ifadesi tek başına sayısal güvenilirlik üretmez.

### 5.6 Accuracy

| Etiket | Operasyonel anlam |
|---|---|
| `relaxed` | Yaklaşık/doğruluğu düşük sonuç açıkça kabul edilebilir |
| `standard` | Normal doğruluk beklentisi |
| `high` | Yüksek doğruluk açıkça istenir |
| `critical` | Yanlış sonucun ciddi zarar verdiği veya çok yüksek eşik istendiği durum |
| `null` | Doğruluk gereksinimi belirtilmemiş veya çelişkili |

Sayısal değer varsa `min_score` 0–1 aralığında, metrik adı varsa `metric` alanına yazılır. Metrik yoksa uydurulmaz.

### 5.7 Energy priority

| Etiket | Operasyonel anlam |
|---|---|
| `low` | Enerji açıkça önemsiz veya cihaz şebeke beslemeli |
| `balanced` | Enerji ile performans arasında açık denge istenir |
| `high` | Pil ömrü/enerji tasarrufu açık önceliktir |
| `null` | Enerji tercihi belirtilmemiş veya çelişkili |

### 5.8 Divisibility

| Etiket | Operasyonel anlam |
|---|---|
| `divisible` | Görev parçalarının bağımsız/paralel işlenebildiği açıkça belirtilir |
| `indivisible` | Görevin bütün halinde işlenmesi gerektiği açıkça belirtilir |
| `null` | Bölünebilirlik belirtilmemiş veya çelişkili |

Bu alan Faz 2–7 eylem uzayını genişletmez; Faz 8 partial-offloading kararının ön koşuludur.

## 6. Evidence ve confidence

`explicit`, `partial` veya `ambiguous` etikette annotator kararı destekleyen kısa metin parçasını işaretler. `not_stated` için evidence boş kalır.

Confidence arayüzünde üç seçenek bulunur:

| Arayüz | Kaydedilen değer |
|---|---:|
| Eminim | 0.90 |
| Orta | 0.70 |
| Emin değilim | 0.50 |

`not_stated` kararından emin olunabilir; bu nedenle confidence ile bilginin varlığı karıştırılmaz. Genel confidence, alan değerlerinin ortalaması olarak önerilir ve annotator tarafından onaylanır.

## 7. Corpus tasarımı

Ana corpus 240 benzersiz Türkçe metindir. İngilizce robustness seti, ancak ana deney tamamlandıktan ve iki annotatorın yeterliği doğrulandıktan sonra ayrı uzantı olarak eklenebilir.

Hedef kaynak aileleri:

- ETSI/3GPP ve akademik kullanım senaryolarından kaynak gösterilerek türetilen metinler,
- BUPT gibi gerçek trafik kayıtlarının veri büyüklüğü/hizmet türüyle koşullanan metinler,
- insan tarafından yazılan doğal varyasyonlar,
- test edilen Qwen modelinden farklı bir araçla üretilmiş paraphrase adayları,
- negation, eksik bilgi, çelişki ve domain-shift örnekleri.

Metni üreten şablon/model gold etiketi belirlemez. Gold yalnızca iki insanın kilitlenmiş ham kararları ve adjudication sonucudur.

## 8. Pilot

1. Her iki annotator aynı 20 pilot örneği bağımsız etiketler.
2. Pilot sırasında birbirlerinin cevaplarını görmezler.
3. Alan bazlı anlaşma ve anlaşmazlık nedenleri çıkarılır.
4. Kılavuzdaki belirsiz tanımlar düzeltilir.
5. Değişen alanlar için pilot yeniden uygulanır.
6. Pilot örnekleri ana test setine girmez.

Ana annotasyon başladıktan sonra etiket tanımları değiştirilmez. Zorunlu değişiklik olursa ilgili alan bütün 240 örnekte yeniden etiketlenir ve amendment kaydı tutulur.

## 9. Ana annotasyon

- 240 görev için iki ayrı random sıra üretilir.
- İki annotator bütün görevleri etiketler.
- Yaklaşık 12 örnek, aynı annotatora farklı yerde gizli tekrar olarak sunulur.
- Kısa oturumlar ve düzenli mola önerilir.
- Gönderim zamanı ve kılavuz sürümü otomatik kaydedilir.
- Eksik kayıtlar anlaşma hesabından önce tamamlatılır; etiket tahmin edilmez.

Beklenen minimum emek kişi başına yaklaşık 2–3 saattir. Gerçek süre pilotta ölçülür ve raporlanır.

## 10. Anlaşma hesabı

Adjudication öncesi ham cevaplarda:

- domain/privacy gibi nominal alanlarda Cohen's kappa,
- latency/reliability/accuracy/energy gibi sıralı alanlarda quadratic weighted kappa,
- execution policy için hedef başına Cohen's kappa ve kayıt başına Jaccard,
- yüzde uyum ve confidence dağılımı,
- gizli tekrarlarda annotator içi uyum

raporlanır.

Her ana alanda minimum hedef `κ >= 0.70`, tercih edilen hedef `κ >= 0.80`'dir. Nadir sınıflarda kappa'nın prevalence etkisi nedeniyle sınıf dağılımı, yüzde uyum ve güven aralığı birlikte verilir.

## 11. Adjudication

1. İki ham annotasyon değiştirilemez biçimde kilitlenir ve checksum alınır.
2. Yalnız anlaşmazlıklar ayrı ekranda açılır.
3. Annotatorlar kılavuz ve evidence üzerinden ortak karar arar.
4. Uzlaşma varsa final etiket ve kısa gerekçe kaydedilir.
5. Uzlaşma yoksa alan `ambiguous/null` kalır; üçüncü kişi varmış gibi karar uydurulmaz.
6. Nihai gold dosyası ham A/B dosyalarından ayrı üretilir.

## 12. Split ve sızıntı önleme

Nihai 240 benzersiz görev yaklaşık `60/20/20` oranında train/validation/test olarak ayrılır. Kesin sayı, bütün sınıflar ve scenario family grupları görüldükten sonra deterministik split aracıyla belirlenir.

Aynı:

- scenario family,
- template family,
- paraphrase/translation family,
- trace entity

birden fazla split'e sızamaz. Normalizasyon yalnız train üzerinde fit edilir. Test etiketleri model/prompt seçiminden önce kilitlenir.

## 13. Saklanacak artefaktlar

```text
data/semantic_benchmark/
├── tasks.jsonl
├── pilot/
├── annotations/
│   ├── annotator_a.jsonl
│   ├── annotator_b.jsonl
│   └── adjudicated_gold.jsonl
├── splits/
├── manifests/
└── agreement_report.md
```

Repo yalnız küçük, lisansı uygun ve mahremiyet içermeyen işlenmiş artefaktları tutabilir. Ham üçüncü taraf izleri, kişisel veri, model checkpoint'leri ve yeniden dağıtımı yasak içerikler Git'e eklenmez.
