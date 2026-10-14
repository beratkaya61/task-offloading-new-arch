# MEC Görev Tanımı Veri Setleri ve Semantik Çıkarım Literatür İncelemesi

**Hazırlanma tarihi:** 30 Eylül 2026  
**Çalışma:** LLM destekli semantik gereksinim çıkarımı ve DRL tabanlı MEC task offloading  
**Belgenin amacı:** Tez çalışmasında neden kontrollü görev metinleri ile gerçek MEC izlerinin ayrı katmanlarda ele alındığını, literatürde hangi veri kaynaklarının bulunduğunu ve MEC ekosisteminde “semantik çıkarım” kavramının hangi anlamlarda kullanıldığını açıklamak.

## 1. Yönetici özeti

Bu incelemenin temel sonucu şudur:

> Literatürde doğal dil ağ/servis niyetleri içeren veri kümeleri de gerçek MEC ölçümleri içeren veri kümeleri de bulunmaktadır. Ancak yapılan taramada; doğal dil görev açıklamasını, doğrulanmış semantik gereksinim etiketlerini, aynı göreve ait fiziksel MEC durumunu ve gerçekleşen offloading sonucunu aynı kayıtta birlikte sunan açık bir veri kümesine rastlanmamıştır.

Dolayısıyla “MEC alanında doğal dil içeren hiçbir veri seti yoktur” ifadesi doğru değildir. Daha kesin ifade şöyledir:

- BUPT, NEP, UCI MEC ve EUA gibi veri kümeleri gerçek ağ, kaynak, konum veya çalışma süresi bilgileri sağlar; fakat doğal dil görev gereksinimi sağlamaz.
- BINS ve NIT gibi veri kümeleri doğal dil ağ niyetleri ve bunların yapılandırılmış karşılıklarını sağlar; fakat görevlerin gerçek MEC ortamındaki CPU, kuyruk, bant genişliği, enerji ve offloading sonuçlarını sağlamaz.
- TopoSense-Bench gibi yeni IoT veri kümeleri doğal dil isteklerini fiziksel topolojiyle ilişkilendirir; fakat çözmeye çalıştığı problem computation offloading değildir.
- STEPS, AgentVNE ve IGAA gibi yakın dönem çalışmalar, doğal dil/semantik katman ile fiziksel simülasyon veya trace katmanını ayrı üretip kontrollü şekilde birleştirmektedir.

Bu bulgular, tezde benimsenen **trace-driven-hybrid** yaklaşımın bilimsel olarak savunulabilir olduğunu göstermektedir. Bununla birlikte, 2026 tarihli STEPS çalışması tez fikrine oldukça yakındır. Bu nedenle tezin yenilik iddiası yalnızca “LLM doğal dili okuyarak edge scheduling’e yardımcı olur” şeklinde kurulamaz. Tezin özgün katkısı; daha kapsamlı semantik sözleşme, hard constraint uygulaması, belirsizlik/abstention, semantik çıkarıcının DRL’den izolasyonu, çoklu gerçek iz kaynağı ve nedensel ablation deneyleri üzerinden tanımlanmalıdır.

## 2. İncelemenin kapsamı ve değerlendirme ölçütü

Literatür taramasında “görev tanımı içeren veri seti” ifadesi dört farklı veri türüne ayrılmıştır:

1. **Görev adı veya türü:** `image_classification`, `video_analytics` veya `task_type=2` gibi sembolik bilgi.
2. **Sayısal görev tanımı:** Girdi boyutu, gerekli CPU cycle, deadline, enerji bütçesi ve bağımlılık grafiği gibi değerler.
3. **Doğal dil görev gereksinimi:** “Bu görüntü 100 ms içinde işlenmeli ve veri buluta gönderilmemeli” gibi insan tarafından okunabilen istek.
4. **Görev içeriği:** İşlenecek görüntü, video, ses, metin veya LLM promptu.

Bu tez üçüncü türle ilgilenmektedir. Amaç, doğal dilde ifade edilen bir görevin aşağıdaki gibi yapılandırılmış gereksinimlere dönüştürülmesidir:

```json
{
  "latency_max_ms": 100,
  "privacy": "sensitive",
  "cloud_allowed": false,
  "reliability": "high"
}
```

Bu nedenle yalnızca `task_type` veya `cpu_cycles` alanı bulunan bir veri kümesi, tez açısından “doğal dil görev tanımı içeren veri seti” sayılmamıştır.

## 3. İstenen ideal veri kümesi

Tez açısından ideal bir gerçek dünya kaydı şu dört katmanı aynı gözlemde içermelidir:

```json
{
  "task_text": "Ambulanstan gelen görüntüler 100 ms içinde işlenmeli ve buluta çıkmamalı.",
  "gold_semantics": {
    "deadline_ms": 100,
    "privacy": "sensitive",
    "cloud_allowed": false,
    "reliability": "high"
  },
  "physical_state": {
    "input_bytes": 5242880,
    "cpu_cycles": 2000000000,
    "bandwidth_mbps": 18.4,
    "edge_queue": 6,
    "rtt_ms": 21
  },
  "decision": "edge_2",
  "outcome": {
    "latency_ms": 83,
    "energy_j": 1.7,
    "success": true
  }
}
```

Yapılan taramada bu dört parçayı aynı görev kaydı altında sunan açık bir MEC benchmark’ı bulunamamıştır. 2024 tarihli kapsamlı bir MEC task-offloading incelemesi de çalışmaların çoğunda deney koşullarına göre özel/sentetik görev verisi üretildiğini, açık veri setlerinin sınırlı olduğunu ve EUA gibi gerçek konum verilerinin üzerine üretilmiş hesaplama görevleri bindirildiğini bildirmektedir [1].

Bu sonuç mutlak bir “dünyada yoktur” kanıtı değildir. Literatür sürekli değişmektedir ve kapalı kurumsal veri kümeleri bulunabilir. Buradaki akademik olarak savunulabilir iddia, **incelenen açık literatür ve resmî veri kaynaklarında tam eşlenmiş bir benchmark bulunamadığıdır**.

## 4. Veri kümelerinin karşılaştırılması

| Kaynak | Doğal dil görev/niyet metni | Gerçek fiziksel veri | Semantik etiket | Tez açısından rolü |
|---|---:|---:|---:|---|
| BUPT Edge Computing Dataset | Hayır | Evet | Hayır | İstek zamanı ve trafik büyüklüğü izi |
| NEP Edge Workloads Traces | Hayır | Evet | Hayır | CPU, bant genişliği, RTT ve kapasite izi |
| UCI MEC Execution Times | Hayır | Evet | Hayır | Ölçülmüş görüntü tanıma çalışma süresi |
| EUA Dataset | Hayır | Evet | Hayır | Gerçek kullanıcı ve edge konumları |
| BINS | Evet | Kısmen | Evet | Ağ niyeti ve performans ifadesi çıkarımı |
| NIT | Evet | Hayır | Evet | Doğal dil ağ niyetinden cihaz komutu üretimi |
| IW-IB-5GNET | Ham doğal dil yok | Evet | Politika/intent ilişkisi var | Niyetten türetilmiş 5G politika telemetrisi |
| TopoSense-Bench | Evet | Gerçek topolojiden türetilmiş | Evet | Doğal dil IoT isteğinden sensör/topoloji seçimi |
| OCE-CRS | Evet; LLM promptu | Evet | İş süresi etiketi | Prompt içeriğinden inference süresi tahmini |

### 4.1 BUPT Edge Computing Dataset

BUPT veri kümesi Çin’deki bir internet servis sağlayıcısından elde edilen 480 binden fazla mobil bağlantı kaydı içerir. Kayıtlarda baz istasyonu, erişim zamanı, bağlantı süresi, uplink/downlink bayt miktarı, radyo erişim türü, URL, port, IP, durum kodu, User-Agent ve Content-Type gibi alanlar bulunur [2].

BUPT aşağıdaki sorulara yardımcı olabilir:

- İstekler hangi zamanlarda ve hangi yoğunlukta geliyor?
- Girdi/çıktı trafiği hangi büyüklüklerde?
- Trafik hangi istasyonlarda oluşuyor?
- İstek süreleri ve servis kullanım örüntüleri nasıl değişiyor?

Fakat aşağıdaki gereksinimler BUPT kayıtlarında bulunmaz:

- Görevin doğal dil açıklaması
- Deadline’ın kullanıcı tarafından nasıl ifade edildiği
- Verinin gizlilik seviyesi
- Local/edge/cloud çalıştırma izinleri
- Beklenen doğruluk ve güvenilirlik
- Görevin bölünebilirliği

URL alanı kullanılan servise ilişkin zayıf bir ipucu verebilir; ancak URL’den gizlilik, deadline veya çalışma izni gibi gereksinimleri kesin olarak çıkarmak bilimsel olarak savunulamaz. Ayrıca gizlilik nedeniyle ham URL, IP, User-Agent ve cihaz kimlikleri tez modeline doğrudan verilmemelidir.

**Sonuç:** BUPT doğal dil veri seti değil, fiziksel istek ve trafik izi kaynağıdır.

### 4.2 NEP Edge Workloads Traces

NEP-small ve NEP-large/full, Çin’deki geniş ölçekli bir edge platformundan VM düzeyinde CPU, bellek, depolama ve bant genişliği kullanım izleri sağlar. Veri setinde edge siteleri arasındaki RTT, VM–fiziksel makine ilişkisi ve fiziksel kapasite bilgileri de bulunmaktadır [3].

NEP şu amaçlarla değerlidir:

- Edge sunucu yüklerinin zaman içindeki değişimini modellemek
- Gerçek CPU ve bant genişliği dalgalanmalarını replay etmek
- Edge siteleri arasındaki RTT dağılımını kullanmak
- Heterojen fiziksel kaynak kapasitesi oluşturmak

Ancak VM’nin hangi insan isteği veya doğal dil görevi için çalıştığı açıklanmaz.

**Sonuç:** NEP fiziksel kaynak durumunu sağlar; semantik görev gereksinimi sağlamaz.

### 4.3 UCI Image Recognition Task Execution Times in MEC

UCI veri seti, görüntü tanıma görevlerinin dört farklı edge sunucuda ölçülmüş turnaround sürelerini içerir. Veri setinde 4.000 gözlem bulunmaktadır ve görevlerin gerçek donanımlardaki çalışma süreleri kaydedilmiştir [4].

Burada görev türü bilinmektedir: görüntü tanıma. Ancak kullanıcının “hız kritik”, “yüksek doğruluk gerekli” veya “veri cihazdan çıkamaz” gibi doğal dil gereksinimleri bulunmamaktadır.

**Sonuç:** Ölçülmüş hizmet süresi kalibrasyonu için değerlidir; doğal dil semantik benchmark’ı değildir.

### 4.4 EUA Dataset

EUA, Avustralya’daki gerçek veri kaynaklarından türetilmiş edge sunucu ve kullanıcı konumlarını sağlar [5]. Literatürde sıklıkla kullanıcı–edge mesafesi, kapsama ve yerleştirme deneyleri için kullanılmaktadır.

EUA’da doğal dil görev açıklaması, görev boyutu veya deadline bulunmaz. İlgili MEC çalışmalarında bu konumların üzerine araştırmacı tarafından üretilen görevler bindirilir [1]. 2026 tarihli STEPS çalışması da gerçek dünya deneyi olarak EUA’dan Melbourne merkezindeki 300 kullanıcı ve 12 edge konumunu kullanmış, görev ve doğal dil gereksinimlerini ayrıca üretmiştir [10].

**Sonuç:** EUA topoloji/konum gerçekliği sağlar; görev semantiği sağlamaz.

### 4.5 BINS: Business Intent and Network Slicing Correlation Dataset

BINS, 2025 yılında *Scientific Data* dergisinde yayımlanmış doğal dil ağ niyeti veri kümesidir. Veri kümesinde kullanıcı iş niyeti açıklamaları, varlık/ilişki etiketleri ve niyetlerin eMBB, URLLC veya mMTC ağ dilimleriyle ilişkileri bulunur [6].

Veri üç ana kaynaktan oluşturulmuştur:

1. Sichuan Telecom’dan alınan 10.001 gerçek operasyon kaydı
2. Ağ mühendisleri ve gönüllüler tarafından oluşturulan niyetler
3. Akademik yayın, standart ve resmî ağ belgelerinden çıkarılan örnekler

Önemli metodolojik ayrıntı şudur: Gerçek operasyon kayıtlarındaki servis adı, tarih ve performans gereksinimleri doğal dil ifadelerine dönüştürülmüştür. Örneğin `web browsing` ve `latency ≤ 200 ms` alanları “Web browsing gecikmesini 200 ms’nin altında tut” benzeri bir niyet cümlesine çevrilmiştir. Bu nedenle metinlerin bir bölümü gerçek operasyon verisine dayansa da doğrudan ham son kullanıcı konuşması değildir.

BINS’te başlıca şu etiketler bulunmaktadır:

- Niyet kullanıcısı
- Ağ eylemi
- Niyet hedefi
- Hedef cihaz/fonksiyon
- Ağ performansı
- Yaşam döngüsü
- Slice türü
- BIO token etiketleri

Veri setinin teknik doğrulamasında BERT-CRF, BERT-BiLSTM-CRF ve DeBERTa-v3-large kullanılmış; ilgili deneylerde yaklaşık %90–94 F1 bildirilmiştir [6].

BINS, tezimizin semantik katmanına en yakın açık veri kaynaklarından biridir. Bununla birlikte:

- Computation offloading kararı içermez.
- Local/edge/cloud sonuçlarını içermez.
- CPU kuyruğu, enerji, kanal veya RTT ile aynı kayıtta eşlenmemiştir.
- Gizlilik, doğruluk ve bölünebilirlik alanları tez şemamızdaki ayrıntıda değildir.
- Temel problem network slicing ve intent-based networking’dir.

**Sonuç:** BINS, semantik çıkarım için dış doğrulama veya kısmi transfer kaynağı olabilir; fiziksel MEC trace’lerinin yerine geçemez.

### 4.6 NIT: Network Intent Translations

NIT, doğal dil ağ yapılandırma isteklerini Juniper EX3300 cihazları için uygulanabilir komutlara dönüştürmek üzere hazırlanmıştır. Her kayıtta doğal dil `question`, genel komut biçimini sağlayan `context` ve uygulanabilir komutları içeren `answer` alanları bulunur [7].

Veri kümesi yapılandırma kılavuzları ve ağ mühendisi uzmanlığı kullanılarak elle oluşturulmuştur. Kullanıcının bütün parametreleri verdiği, yalnızca bazılarını verdiği veya hiç vermediği farklı prompt türleri bulunmaktadır.

NIT’in problemi şöyledir:

```text
doğal dil ağ niyeti → cihaz yapılandırma komutu
```

Tezimizin problemi ise şöyledir:

```text
doğal dil görev gereksinimi → semantic contract → MEC offloading kararı
```

**Sonuç:** NIT semantik ayrıştırma yöntemleri için yararlı bir komşu literatür kaynağıdır; MEC workload/offloading veri seti değildir.

### 4.7 IW-IB-5GNET

IW-IB-5GNET, intent tabanlı kapalı kontrol döngüsü çalışan deneysel bir 5G altyapısından toplanmış 64.290 gözlem ve 107 özellik içerir. Cihaz, ağ arayüzü, kuyruk, trafik akışı ve kontrol kurallarına ilişkin telemetri bulunmaktadır [8].

Her deney dosyası bir niyetten çevrilmiş ağ politikasıyla ilişkilidir. Ancak veri kümesinin doğrudan model girdisi olarak kullanılabilecek ham doğal dil kullanıcı cümlelerini fiziksel telemetriyle birlikte sunduğu gösterilmemektedir.

**Sonuç:** Niyet/politika ile gerçek ağ durumunu ilişkilendiren önemli bir kaynaktır; fakat tezde aranan doğal dil MEC görev kaydı değildir.

### 4.8 TopoSense-Bench

TopoSense-Bench, 2026 MobiCom çalışması IoT-Brain için hazırlanmış semantik-mekânsal sensör scheduling benchmark’ıdır. Veri kümesinde:

- 5.250 doğal dil sorgusu
- 33 bina
- 161 kat planı
- 2.510 kamera
- Her sorgu için doğru sensör/topoloji cevabı

bulunmaktadır [9].

Veri üretim süreci özellikle önemlidir:

```text
uzman şablonları
      ↓
LLM ile kontrollü çeşitlendirme
      ↓
manuel uzman doğrulaması ve ground-truth annotation
```

Topoloji gerçek OpenStreetMap verisinden ve yarı otomatik anotasyondan türetilmiştir; fakat doğal dil sorguları uzman şablonlarından LLM ile sentezlenmiştir. Gerçek video veya görüntü bulunmamaktadır.

Bu yöntem, tezimizde kullanılan kontrollü metin üretiminin literatürde karşılığı olduğunu göstermektedir. Ancak problem, hesaplamanın local/edge/cloud’a atanması değil, doğal dil isteğine göre sensörlerin seçilmesidir.

**Sonuç:** Veri oluşturma metodolojisi açısından çok değerlidir; doğrudan computation offloading benchmark’ı değildir.

### 4.9 OCE-CRS

OCE-CRS, cloud-edge ortamında LLM isteklerini schedule etmektedir. Burada metin yalnızca bir gereksinim açıklaması değildir; çalıştırılacak işin kendisi LLM promptudur. Sistem, promptun semantik içeriğinden processing time tahmini yapar ve kısa sürmesi beklenen işleri edge tarafında önceliklendirir [15].

Resmî kod deposundaki veri üretim hattı, Alpaca promptlarını seçilen bir LLM ve donanım üzerinde çalıştırarak her prompt için gerçek inference süresi, token sayısı, yanıt ve cihaz bilgisini kaydetmektedir [16].

Bu kaynak, “metin + ölçülmüş çalışma süresi” ikilisini gerçekten sağlar. Ancak cevapladığı soru farklıdır:

- OCE-CRS: “Bu promptun çalışması ne kadar sürer?”
- Tezimiz: “Bu görev açıklamasının gizlilik, deadline, güvenilirlik ve çalıştırma izinleri nedir?”

**Sonuç:** Metnin fiziksel maliyet tahmininde kullanılabileceğine dair güçlü bir örnektir; genel IoT/MEC görev gereksinimi veri seti değildir.

## 5. MEC literatüründe “semantik” kavramının farklı anlamları

“Semantic-aware MEC” ifadesi tek bir araştırma problemini göstermemektedir. Literatürde en az beş farklı kullanım vardır.

### 5.1 Klasik MEC: Semantik çıkarım yapılmaz

Klasik task-offloading çalışmalarında görev çoğunlukla doğrudan sayısal bir vektör olarak tanımlanır:

```text
Task = (
    input_data_size,
    required_cpu_cycles,
    deadline,
    output_size,
    energy_budget,
    dependency_graph
)
```

Bu değerler kullanıcı cümlesinden çıkarılmaz; araştırmacı tarafından deney parametresi olarak belirlenir veya bir dağılımdan örneklenir. DRL ajanı metin okumaz ve doğrudan sayısal state üzerinden karar verir.

Bu nedenle bir makalede “delay-sensitive task” ifadesinin bulunması, deadline’ın doğal dilden çıkarıldığı anlamına gelmez.

### 5.2 Semantic communication: Görev gereksinimi değil, veri içeriği

Semantic communication çalışmalarında amaç, ham verinin tamamı yerine görev için gerekli anlamlı özellikleri iletmektir:

```text
görüntü / sinyal
      ↓
DNN semantic encoder
      ↓
göreve yararlı latent özellikler
      ↓
edge sunucuya sıkıştırılmış aktarım
```

Ji ve Qin, cloud-edge-end ortamında task-oriented semantic communication kullanarak iletilecek görev bilgisini azaltmakta ve Lyapunov destekli DRL ile enerji/offloading optimizasyonu yapmaktadır [12]. Zheng ve arkadaşları ise task accuracy ile gecikme arasında semantic compression ve offloading optimizasyonu kurmaktadır [13].

Buradaki “semantik”, kullanıcının doğal dilde belirttiği gizlilik veya deadline gereksinimi değildir. Görüntü, sinyal veya görev verisinin DNN tarafından çıkarılan görevle ilişkili temsilidir.

### 5.3 Intent-based networking: Doğal dil → yapılandırılmış niyet

Intent-based networking çalışmalarında kullanıcı veya operatör şöyle bir istek verebilir:

> Video konferans trafiğinde gecikmeyi 100 ms’nin altında tut.

Sistem bu metni yapılandırılmış hedefe dönüştürür:

```json
{
  "service": "video_conference",
  "latency_max_ms": 100,
  "action": "provide_service"
}
```

Bu alanda kullanılan başlıca yöntemler şunlardır:

- Kural/anahtar kelime tabanlı ayrıştırma
- Intent classification
- Named Entity Recognition (NER)
- BERT-CRF
- BERT-BiLSTM-CRF
- DeBERTa
- LLM + retrieval-augmented generation
- LLM ile JSON, DSL veya Network Service Descriptor üretimi

BINS ve NIT bu araştırma hattındadır. Yöntem açısından tezimizin semantik ayrıştırıcısına en yakın yerleşik literatür budur.

### 5.4 LLM destekli doğal dil edge scheduling

Bu grup tezimizin çekirdeğine en yakın çalışmaları içerir.

#### STEPS

STEPS, doğal dil servis beklentisini Qwen3.5-4B ile aşağıdaki semantik vektöre dönüştürür:

- Gecikme hassasiyeti
- Enerji hassasiyeti
- Parasal maliyet hassasiyeti
- Çalıştırma güveni/T4E
- Parser confidence

Ardından bu seviyeler yorumlanabilir eşleme kurallarıyla preference, bound ve uncertainty içeren semantic contract’a dönüştürülür. LLM doğrudan kaynak tahsisi veya node seçimi yapmaz; scheduling potansiyel oyun/asenkron best-response yöntemiyle çözülür [10].

Deneylerdeki doğal dil istekleri gerçek kullanıcı veri setinden gelmez. Latency, enerji, maliyet ve T4E şablonlarından üretilir; ayrıca belirsiz, çelişkili ve kapsam dışı ifadeler eklenir. Fiziksel görev boyutları ile CPU workload değerleri simüle edilir. “Gerçek dünya” değerlendirmesinde EUA yalnızca kullanıcı ve edge konumları için kullanılır.

Bu çalışma iki önemli sonuç doğurur:

1. Tezimizin doğal dil ve fiziksel trace katmanlarını ayrı ele alan hibrit yöntemi literatürle uyumludur.
2. “LLM ile doğal dil gereksinimi çıkarıp edge scheduling yapmak” tek başına artık özgünlük iddiası olamaz.

#### AgentVNE

AgentVNE, Qwen3-30B ile multi-agent workflow’larındaki TEE veya kamera bağlantısı gibi örtük donanım/affinity kısıtlarını belirler. LLM çıktısı fiziksel düğümler için resource bias üretir; ardından GNN ve PPO tabanlı yerleştirme yapılır [11].

Deney ortamı tamamen kontrollüdür:

- Fiziksel ağ Waxman modeliyle üretilmiştir.
- Düğümlerin yaklaşık %10’u özel affinity niteliğine sahiptir.
- Dört sentetik workflow türü kullanılır.
- Workflow’lar text search ve image processing kategorilerindedir.
- Arrival süreci Poisson, lifetime üstel dağılımdır.

AgentVNE açık bir doğal dil görev benchmark’ı kullanmaz. LLM tabanlı semantik algı ile sentetik fiziksel simülasyonu birleştirir.

#### IGAA

IGAA’nın akışı şöyledir:

```text
doğal dil kullanıcı niyeti
          ↓
LLM ile önceden tanımlı intent vector
          ↓
Network–Service–Intent matrisi
          ↓
sayısal CPU/GPU/DPU gereksinimleri
          ↓
service scheduling modeli
```

LLM yalnızca niyeti anlamakla kalmaz; yeni servis senaryoları ve eğitim veri setleri de üretir. Üretilen senaryoların fiziksel olarak geçersiz olabileceği kabul edilmiş ve bunları denetleyen evaluation-correction mekanizması eklenmiştir [14].

Bu çalışma, semantik–fizik köprüsünün doğrudan serbest LLM çıktısına bırakılamayacağını; uzman bilgisi, tarihsel ölçüm veya açık fiziksel kurallarla sınırlandırılması gerektiğini göstermektedir.

### 5.5 LLM kullanan fakat doğal dil semantiği çıkarmayan çalışmalar

Bir çalışmada LLM kullanılması, o çalışmanın doğal dil semantik çıkarımı yaptığı anlamına gelmez.

2026 tarihli vehicular offloading çalışması Llama 3.2-1B’yi offloading karar vericisi olarak kullanmaktadır. Ancak model ham doğal dil görev açıklaması almaz. Araç hızı, CPU, bant genişliği, batarya, mesafe ve görev özellikleri yapılandırılmış JSON olarak verilir. Hareketlilik SUMO/OMNeT++ ile simüle edilir ve görev gereksinimleri rastgele oluşturulur [17].

Bu yaklaşımın akışı şöyledir:

```text
sayısal fiziksel state → JSON → LLM → V2V/V2E/local kararı
```

Tezimizin yaklaşımı ise şöyledir:

```text
doğal dil → gereksinim JSON'u
fiziksel state + gereksinim JSON'u → DRL kararı
```

Bu ayrım tezde açıkça korunmalıdır.

## 6. Literatürde kullanılan temel semantik çıkarım yöntemleri

### 6.1 Kural tabanlı yöntemler

Belirli anahtar kelime ve sayı kalıpları eşleştirilir:

```text
"100 ms içinde"  → latency_max_ms = 100
"buluta çıkamaz" → cloud_allowed = false
"pil önemli"     → energy_priority = high
```

Avantajları yorumlanabilir ve düşük maliyetli olmalarıdır. Dezavantajları farklı ifade biçimlerine, örtük gereksinimlere ve çelişkilere karşı zayıf olmalarıdır.

### 6.2 Klasik denetimli NLP

TF-IDF, lojistik regresyon, SVM veya benzer küçük modellerle her semantik alan ayrı sınıflandırılabilir. Bu modeller iyi baseline’lardır; fakat metindeki sayısal span’leri ve alanlar arası ilişkileri yakalamakta sınırlı kalabilirler.

### 6.3 NER ve sequence labeling

BERT-CRF ve BERT-BiLSTM-CRF gibi modeller, metindeki ilgili token aralıklarını bulur:

```text
"video konferans" → SERVICE
"100 ms"          → LATENCY_BOUND
"6 saat"          → LIFECYCLE
```

BINS’in teknik doğrulaması bu yaklaşımı kullanmaktadır [6].

### 6.4 Encoder tabanlı sınıflandırma

Çok dilli encoder modeli metni sabit boyutlu bir embedding’e dönüştürür. Bu embedding üzerinde her semantik alan için küçük sınıflandırma başlıkları eğitilir. LLM’den daha düşük maliyetlidir ve güçlü bir ara baseline sağlar.

### 6.5 LLM tabanlı yapılandırılmış çıkarım

LLM’den doğrudan eylem değil, şema uyumlu gereksinim çıktısı istenir:

```text
task_text
    ↓
LLM semantic parser
    ↓
JSON Schema validation
    ↓
confidence / unknown / conflict
```

Bu tasarım STEPS’in yaklaşımıyla uyumludur. LLM’nin bilmediği alanı uydurması yerine `unknown/null` üretmesi, confidence değerinin kalibre edilmesi ve çıktının scheduling’den önce cache’lenmesi gerekir.

## 7. Tez mimarisine etkisi

Literatür sonucunda tezde korunması gereken mimari aşağıdadır:

```text
Kontrollü görev metni / kullanıcı politikası
                    ↓
          Semantik çıkarım modeli
                    ↓
      Yapılandırılmış semantic contract
                    ↓
  ┌─────────────────┴──────────────────┐
  │ BUPT / NEP / UCI / EUA fiziksel izi│
  └─────────────────┬──────────────────┘
                    ↓
      Açık ve tekrarlanabilir eşleme
                    ↓
        DRL local/edge/cloud kararı
                    ↓
 latency / energy / constraint / success
```

Bu yapıda iki ayrı gerçeklik katmanı vardır:

1. **Semantik katman:** Görevin ne istediğini açıklar.
2. **Fiziksel katman:** Ağın ve hesaplama kaynaklarının o anda ne durumda olduğunu açıklar.

Bu katmanlar aynı gerçek dünya gözleminden gelmiyorsa, birleşik olay “tam gerçek veri” olarak sunulmamalıdır. Projedeki doğru adlandırma **trace-driven-hybrid benchmark**’tır.

### 7.1 Mevcut 240 görev metninin doğru konumu

Mevcut metinler gerçek son kullanıcı trafiği olarak gösterilmemelidir. Bunlar:

- `standard_derived`
- `trace_conditioned`
- `llm_assisted_synthetic`
- `adversarial`

kategorilerinde kontrollü araştırma metinleridir. Fiziksel izlerden yararlanılması, metinleri otomatik olarak gerçek kullanıcı cümlesine dönüştürmez.

“Kafamızdan yazılmış görevler” yerine akademik olarak daha doğru ifade şudur:

> Önceden tanımlanmış semantik şemayı, uygulama alanlarını ve fiziksel trace dağılımlarını kapsayacak şekilde kontrollü olarak üretilmiş, adversarial örneklerle zenginleştirilmiş Türkçe görev-niyet corpus’u.

### 7.2 Metin ile trace arasında kurulabilecek ve kurulamayacak bağ

BUPT veya NEP satırlarının üzerine doğrulanmamış anlam yüklenmemelidir. Örneğin bir URL veya trafik büyüklüğünden “bu sağlık görevidir ve buluta çıkamaz” sonucu çıkarılamaz.

Doğru yöntem şudur:

```text
semantic scenario sampler
           +
physical trace sampler
           ↓
açık eşleme kuralları ve provenance
           ↓
birleşik deney olayı
```

Her alanın kaynağı `observed`, `measured`, `derived`, `matched`, `generated`, `human_labeled` veya `simulated` olarak kaydedilmelidir.

## 8. Yenilik iddiasının yeniden konumlandırılması

STEPS nedeniyle aşağıdaki genel iddia yeterince güçlü değildir:

> LLM doğal dil gereksinimlerini çıkaracak ve edge scheduling yapılacaktır.

Tezin daha güçlü ve savunulabilir özgün katkıları şunlar olmalıdır:

1. **Daha geniş semantic contract:** Domain, latency, privacy, execution permission, reliability, accuracy, energy priority ve divisibility alanları.
2. **Hard constraint uygulaması:** Özellikle gizlilik ve local/edge/cloud izinleri yalnızca reward tercihi değil, action mask veya fiziksel ihlal olarak modellenmelidir.
3. **Semantik izolasyonu:** LLM ağ durumunu, ödülü, oracle kararını veya doğru eylemi görmemelidir.
4. **Belirsizlik yönetimi:** `unknown`, confidence, çelişki tespiti ve gerektiğinde abstention.
5. **Model ailesi karşılaştırması:** Semantiksiz, sabit profil, rule, TF-IDF, küçük encoder, zero-shot/few-shot LLM ve insan referansı.
6. **Çoklu gerçek iz:** BUPT, NEP, UCI MEC ve EUA’nın açık rollerde kullanılması.
7. **Nedensel ablation:** Aynı fiziksel olaylar ve aynı seed’ler altında yalnızca semantik katmanın değiştirilmesi.
8. **DRL katkısı:** STEPS’in potential-game scheduler’ından farklı olarak PPO tabanlı politikanın değerlendirilmesi.
9. **Uçtan uca hata analizi:** Semantik çıkarım hatasının latency, deadline miss, enerji ve constraint violation sonuçlarına etkisinin ölçülmesi.
10. **Dürüst gerçeklik sınırı:** Kontrollü sentetik metin, gerçek trace ve hibrit eşlemenin birbirinden açıkça ayrılması.

## 9. BINS’in projede kullanılmasına ilişkin değerlendirme

BINS tamamen göz ardı edilmemelidir. Ancak doğrudan mevcut Türkçe corpus’un yerine geçirilmesi de doğru değildir.

Olası kullanım biçimleri:

- Network performance ve servis intent alanlarında dış-domain değerlendirme
- Doğal dilde latency/bandwidth ifadelerinin çıkarımı için ek benchmark
- BERT/DeBERTa tabanlı semantik modellerle yöntem karşılaştırması
- Ağ niyeti literatürüyle tez şemasının kavramsal karşılaştırması

Sınırlamalar:

- BINS esas olarak İngilizce ağ niyeti verisidir.
- Tezimizin sekiz alanının tamamını kapsamaz.
- Türkçeye otomatik çeviri, yeni bir yapay veri katmanı oluşturur; çevrilmiş veri gerçek Türkçe kullanıcı verisi sayılamaz.
- Veri lisansı ve türetilmiş veri paylaşım koşulları kullanılmadan önce artifact düzeyinde ayrıca doğrulanmalıdır.
- BINS ile BUPT/NEP kayıtları arasında doğal bire bir eşleşme yoktur.

Bu nedenle önerilen rol, BINS’i ana fiziksel veri kaynağı değil, **komşu alan semantik benchmark’ı** olarak değerlendirmektir.

## 10. Sonuç

Literatür taraması sonunda ulaşılan dengeli ve kanıta dayalı sonuç şöyledir:

- MEC alanında görev türü ve sayısal görev tanımı içeren çok sayıda çalışma vardır.
- Doğal dil ağ ve servis niyeti içeren açık veri kümeleri vardır.
- Gerçek MEC fiziksel ölçümleri içeren açık veya izinli veri kümeleri vardır.
- Ancak doğal dil MEC görev gereksinimini, uzman doğrulamalı semantik etiketleri, fiziksel çalışma koşullarını ve offloading sonucunu aynı kayıtta birleştiren açık bir benchmark bulunamamıştır.
- Literatürde bu eksiklik çoğunlukla kontrollü şablonlar, LLM destekli veri üretimi, uzman doğrulaması ve ayrı gerçek trace/topoloji kaynaklarının hibrit kullanımıyla giderilmektedir.
- Bu nedenle tezde izlenen kontrollü semantik corpus + gerçek trace + açık eşleme yaklaşımı geçerlidir; fakat tam gerçek veri olarak sunulmamalıdır.
- STEPS çalışması tezin temel fikrine yakın olduğundan, yenilik iddiası daha kapsamlı semantik sözleşme, hard constraint, belirsizlik, çoklu trace, DRL ve güçlü ablation deneyleri üzerinden kurulmalıdır.

Tezin bilimsel iddiası veri setinin tamamen gerçek olduğu iddiasına değil, aşağıdaki ölçülebilir araştırma sorusuna dayanmalıdır:

> Bağımsız ve dondurulmuş bir semantik çıkarım katmanının ürettiği gereksinimler, aynı MEC fiziği ve aynı değerlendirme olayları altında, semantiksiz ve LLM’siz semantik baseline’lara kıyasla constraint satisfaction, güvenlik ve genelleme performansını anlamlı şekilde iyileştiriyor mu?

## 11. Kaynaklar

1. S. A. H. Mohsan ve diğerleri, “A comprehensive review on internet of things task offloading in multi-access edge computing,” *Heliyon*, 2024. [Makale](https://pmc.ncbi.nlm.nih.gov/articles/PMC11064154/)
2. BUPT MEC Migration, “Edge Computing Dataset.” [Resmî veri deposu](https://github.com/BuptMecMigration/Edge-Computing-Dataset)
3. M. Xu ve diğerleri, “From Cloud to Edge: A First Look at Public Edge Platforms,” IMC 2021; EdgeWorkloadsTraces. [Resmî veri deposu](https://github.com/xumengwei/EdgeWorkloadsTraces/blob/main/README.md)
4. I. A. Alghamdi, C. Anagnostopoulos ve D. Pezaros, “Image Recognition Task Execution Times in Mobile Edge Computing.” [UCI veri sayfası](https://archive.ics.uci.edu/dataset/859/image%2Brecognition%2Btask%2Bexecution%2Btimes%2Bin%2Bmobile%2Bedge%2Bcompu)
5. P. Lai ve diğerleri, “EUA Datasets.” [Resmî veri deposu](https://github.com/PhuLai/eua-dataset)
6. J. Li, S. Zou, Y. Sun, H. Gao ve W. Ni, “Business Intent and Network Slicing Correlation Dataset from Data-Driven Perspective,” *Scientific Data*, 2025. [Makale](https://www.nature.com/articles/s41597-025-04736-z) · [Veri DOI](https://doi.org/10.6084/m9.figshare.27311538)
7. A. A. AlSamarneh, O. Y. Al-Jarrah ve A. T. Al-Hammouri, “NIT: A dataset for network intent translation,” *Data in Brief*, 2025. [Makale](https://pmc.ncbi.nlm.nih.gov/articles/PMC12272756/) · [Veri kartı](https://huggingface.co/datasets/Smarneh/NIT)
8. J. Andrade-Hoz, Q. Wang ve J. M. Alcaraz-Calero, “Infrastructure-Wide and Intent-Based Networking Dataset for 5G-and-beyond AI-Driven Autonomous Networks,” *Sensors*, 2024. [Makale](https://www.mdpi.com/1424-8220/24/3/783) · [Veri deposu](https://github.com/jimenaandrade/iw-ib-5gnet)
9. “IoT-Brain: Grounding LLMs for Semantic-Spatial Sensor Scheduling,” MobiCom 2026; TopoSense-Bench. [Makale](https://arxiv.org/abs/2604.08033) · [Veri seti](https://huggingface.co/datasets/IoT-Brain/TopoSense-Bench)
10. H. Qi, M. Liwang, X. Wang, X. Yi ve S. Hosseinalipour, “STEPS: Semantic Contract-Guided Scheduling for LLM-Assisted Natural-Language-Driven Edge AI Services,” 2026. [Makale](https://arxiv.org/abs/2606.09537)
11. R. Zheng ve diğerleri, “AgentVNE: LLM-Augmented Graph Reinforcement Learning for Affinity-Aware Multi-Agent Placement in Edge Agentic AI,” 2026. [Makale](https://arxiv.org/abs/2601.02021)
12. Z. Ji ve Z. Qin, “Computational Offloading in Semantic-Aware Cloud-Edge-End Collaborative Networks,” *IEEE Journal of Selected Topics in Signal Processing*, 2024. [Makale](https://arxiv.org/abs/2402.18183)
13. Y. Zheng, T. Zhang, R. Huang ve Y. Wang, “Computing Offloading and Semantic Compression for Intelligent Computing Tasks in MEC Systems,” IEEE WCNC, 2023. [Makale](https://arxiv.org/abs/2307.02747)
14. Y. Sun ve diğerleri, “IGAA: Intent-Driven General Agentic AI for Edge Services Scheduling using Generative Meta Learning,” 2026. [Makale](https://arxiv.org/abs/2601.13702)
15. Y. Li ve diğerleri, “Cloud-Edge System for Scheduling Unpredictable LLM Requests With Combinatorial Bandit,” *IEEE Transactions on Services Computing*, 2025. [Makale DOI](https://doi.org/10.1109/TSC.2025.3611379)
16. OCE-CRS, “Cloud-Edge System for Scheduling Unpredictable LLM Requests with Combinatorial Bandit.” [Resmî kod ve veri üretim deposu](https://github.com/myfilezone/OCE-CRS)
17. Z. Trabelsi, M. Ali, T. Qayyum ve A. Tariq, “Dynamic task offloading in vehicular networks using large language models for adaptive low latency decision making,” *Scientific Reports*, 2026. [Makale](https://www.nature.com/articles/s41598-026-39791-y)

## 12. Akademik kullanım notu

Bu belge bir sistematik derleme protokolüyle hazırlanmış yayımlanabilir survey makalesi değildir. Tez mimarisini ve veri stratejisini gerekçelendirmek için yapılan hedefli bir literatür ve veri kaynağı incelemesidir. “Tam eşlenmiş açık veri seti bulunamadı” sonucu bu taramanın kapsamı içinde yorumlanmalı; tez yazımı sırasında yeni yayınlar için son bir güncellik taraması yapılmalıdır.
