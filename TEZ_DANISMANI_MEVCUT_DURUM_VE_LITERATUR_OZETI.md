# Tez Çalışmasının Amacı, Mevcut Durumu ve Literatürdeki Konumu

**Güncellik tarihi:** 14 Ekim 2026  
**Belgenin amacı:** Çalışmada neyi çözmeye çalıştığımızı, “sekiz semantik alan” ile neyi kastettiğimizi, şu ana kadar ne yaptığımızı ve çalışmanın güncel literatürde nerede durduğunu kısa ve açık biçimde anlatmak.

## 1. Çalışmanın tek cümlelik özeti

Bu tez, kullanıcının doğal dille anlattığı bir IoT/MEC görevinin gereksinimlerini önce yapılandırılmış bir **semantik sözleşmeye** dönüştürmeyi, ardından bu sözleşmeyi gerçek sistem izleriyle beslenen bir MEC ortamında PPO/DRL ajanına vererek görevin cihazda, edge sunucusunda veya bulutta çalıştırılmasına karar vermeyi amaçlamaktadır.

LLM'nin görevi doğrudan “edge'i seç” veya “cloud'u seç” demek değildir. LLM yalnızca metnin ne istediğini çıkaracaktır. Nihai çalıştırma kararını ağın ve cihazların fiziksel durumunu gören DRL politikası verecektir.

```text
Doğal dil görev metni
        ↓
Semantik çıkarıcı: görev ne istiyor?
        ↓
Sekiz alanlı semantik sözleşme
        +
Gerçek izlerden gelen fiziksel durum
        ↓
PPO/DRL: local / edge / cloud kararı
        ↓
Gecikme, enerji, başarı ve kısıt ihlali sonuçları
```

## 2. Çözmeye çalıştığımız problem

Klasik MEC offloading çalışmalarında görevin gereksinimleri çoğunlukla baştan sayısal ve eksiksiz kabul edilir. Örneğin modele doğrudan “deadline = 100 ms, veri boyutu = 5 MB” verilir. Gerçek bir kullanıcı ise çoğu zaman “Bu görüntü çok hassas, buluta gitmesin ve sonuç hızlı gelsin” gibi doğal dil kullanır.

Burada iki farklı bilgi katmanı vardır:

1. **Semantik katman:** Kullanıcı/görev ne istiyor?
2. **Fiziksel katman:** Ağ, cihaz, edge ve cloud o anda ne durumda?

Tezin temel araştırma sorusu şudur:

> Doğal dilden bağımsız olarak çıkarılmış ve önceden dondurulmuş semantik gereksinimler, aynı fiziksel MEC olayları altında semantiksiz veya daha basit semantik yöntemlere kıyasla gecikme, enerji, başarı ve kısıt ihlali sonuçlarını anlamlı biçimde iyileştiriyor mu?

## 3. “Sekiz semantik alan” tam olarak nedir?

Sekiz alan, görev metninin sekiz parçaya bölünmesi değildir. Bunlar, bir görevin **nasıl çalıştırılması gerektiğini etkileyen sekiz farklı gereksinim türüdür**.

| Alan | Sorduğu soru | Örnek | Offloading kararına etkisi |
|---|---|---|---|
| **1. Uygulama alanı (domain)** | Görev hangi bağlamda kullanılıyor? | Sağlık, endüstri, ulaşım, kamu güvenliği, akıllı şehir, tarım, tüketici | Modelin görevin bağlamını ve farklı alanlara genellemesini ölçmemizi sağlar. Tek başına çalıştırma yeri emri değildir. |
| **2. Gecikme (latency)** | Sonuç en geç ne zaman hazır olmalı? | “80 ms içinde”, “ertesi sabaha kadar” | Çok kısa süre isteyen görevler uzak cloud yerine local veya yakın edge'e yönelmeye daha yatkındır. Süre aşımı deadline ihlalidir. |
| **3. Veri gizliliği (privacy)** | İşlenen veri ne kadar hassas? | Kamuya açık, kurum içi, hassas, kısıtlı | Hassasiyetin raporlanmasını ve güvenlik politikasının uygulanmasını sağlar. Açık çalıştırma yasağıyla birlikte cloud seçimi hard constraint olabilir. |
| **4. Çalıştırma izni (execution policy)** | Görev hangi katmanlarda çalışabilir? | Cihaz izinli, edge izinli, cloud yasak | Doğrudan **hard constraint** üretir. Örneğin cloud yasaksa DRL ajanı cloud'u seçemez; seçenek action mask ile kapatılır. |
| **5. Güvenilirlik (reliability)** | Görevin başarıyla tamamlanması ne kadar kritik? | Best effort, yüksek, görev-kritik; en az %99,95 başarı | Başarısızlık ihtimali yüksek düğümlerin kritik görevler için seçilmesini engeller veya ağır ihlal olarak sayar. |
| **6. Sonuç doğruluğu (accuracy)** | Üretilen sonucun kalitesi ne kadar yüksek olmalı? | “F1 en az 0,97”, “yaklaşık sonuç yeterli” | Gerekli doğruluğu sağlayamayan hızlı fakat düşük kaliteli yürütme seçeneğinin uygun olmadığını belirtir. |
| **7. Enerji önceliği (energy priority)** | Pil/enerji tasarrufu ne kadar önemli? | Düşük, dengeli, yüksek öncelik | DRL ödülünde enerji tüketiminin ağırlığını ve enerji-duyarlı tercihleri etkiler. Genellikle soft preference'tır; açık bütçe tanımlanırsa kısıta dönüşebilir. |
| **8. Bölünebilirlik (divisibility)** | Görev parçalara ayrılıp farklı yerlerde çalıştırılabilir mi? | Bölünebilir veya bölünemez | Bölünemez görev tek yerde tamamlanmalıdır. Partial offloading ancak bölünebilir görevler ve ek iletişim maliyeti güvenilir biçimde modellendiğinde açılacaktır. |

Her alan ayrıca şu bilgileri taşır:

- Gereksinim metinde **açıkça belirtilmiş mi**, hiç söylenmemiş mi, yoksa çelişkili/belirsiz mi?
- Karara dayanak olan metin parçası nedir?
- Model bu çıkarımdan ne kadar emindir?
- Model yeterince emin değilse bu alanda karar vermekten kaçınmalı mı (**abstention**)?

Örneğin:

> “Yoğun bakım alarmı 40 ms içinde değerlendirilmelidir; veri cihazdan çıkamaz ve sonuç kritik doğrulukta olmalıdır.”

Bu metinden sağlık alanı, 40 ms hard-real-time gecikme, kısıtlı veri, yalnız cihazda çalıştırma izni ve kritik doğruluk gereksinimleri çıkarılır. Metinde enerji veya bölünebilirlik söylenmemişse model bunları uydurmamalı, `not_stated/unknown` olarak bırakmalıdır.

## 4. Tez kapsamında tam olarak ne yapmayı planlıyoruz?

Tez kapsamında iki ayrı modeli art arda kullanmayı planlıyoruz:

- **Semantik çıkarım modeli**, yalnız görev metnini okuyacak ve sekiz alanlı gereksinim JSON'u üretecek.
- **PPO/DRL ajanı**, bu gereksinim JSON'u ile o andaki fiziksel sistem durumunu birlikte kullanarak görevi `local`, `edge_0`, `edge_1`, `edge_2` veya `cloud` seçeneklerinden birine atayacak.

Başka bir ifadeyle LLM ağ durumunu görmeyecek ve offloading kararı vermeyecek; PPO da doğal dil metnini doğrudan okumayacaktır. Böylece semantik anlama başarısı ile fiziksel karar başarısını ayrı ayrı ölçebileceğiz.

### Planlanan uygulama ve deney adımları

| Adım | Yapılacak çalışma | Üretilecek sonuç | Durum |
|---|---|---|---|
| **1. Semantik sözlüğü tanımlama** | Görevlerin domain, latency, privacy, execution policy, reliability, accuracy, energy ve divisibility alanlarıyla temsil edilmesi | Geçerli ve denetlenebilir sekiz alanlı JSON şeması | Tamamlandı |
| **2. Kontrollü görev corpus'u oluşturma** | Farklı alanları, açık/eksik/çelişkili ifadeleri ve paraphrase'leri kapsayan Türkçe görev metinlerinin hazırlanması | 240 görev metni; 180 geliştirme ve 60 kilitli değerlendirme kaydı | Tamamlandı |
| **3. Semantik yöntemleri karşılaştırma** | Sabit profil, rule, TF-IDF, küçük encoder, Qwen zero-shot ve Qwen few-shot yöntemlerinin aynı 60 kayıtta ölçülmesi | Her yöntem için alan bazlı F1, tam görev doğruluğu, confidence/calibration, abstention, gecikme ve maliyet | Devam ediyor |
| **4. Semantik katmanı dondurma** | Seçilen modelin ağırlık/model kimliği, prompt'u, parser'ı ve tüm görev çıktılarının cache'lenmesi | PPO eğitiminden bağımsız, tekrarlanabilir semantik girdiler | Yapılacak |
| **5. Trace-driven hybrid ortamı kurma** | BUPT, NEP, UCI ve EUA'nın trafik, kaynak, RTT, yürütme süresi ve konum rollerinde açık eşleme kurallarıyla kullanılması | Aynı fizik çekirdeğinde yeniden oynatılabilen MEC olayları | Yapılacak |
| **6. Karar baselines ve oracle** | Always-local/edge/cloud, random, valid-random, latency/energy/deadline greedy ve küçük exhaustive/MILP oracle çalıştırılması | PPO'nun gerçekten öğrenip öğrenmediğini gösterecek alt ve üst karşılaştırma çizgileri | Yapılacak |
| **7. PPO politikalarını eğitme** | Semantiksiz PPO, PPO+rule, PPO+küçük model, PPO+LLM ve PPO+insan referansı koşullarının eğitilmesi | Local/edge/cloud offloading politikaları | Yapılacak |
| **8. Adil semantic-on/off deneyi** | Fiziksel olay, görev, episode seed'i ve PPO ayarları aynı tutulup yalnız semantik bilgi kaynağının değiştirilmesi | Semantik bilginin nedensel katkısını ölçen karşılaştırma | Yapılacak |
| **9. Uçtan uca etki analizi** | Semantik alanlardaki doğru ve yanlış çıkarımların sistem sonuçlarına etkisinin incelenmesi | Latency p50/p95/p99, deadline miss, enerji, başarı/drop, privacy/reliability/accuracy ihlali ve fairness sonuçları | Yapılacak |
| **10. Genelleme ve istatistik** | Farklı zaman, cihaz, uygulama ve domain koşullarında en az 5, tercihen 10 seed ile deney | %95 güven aralığı, eşleştirilmiş test, etki büyüklüğü ve genelleme analizi | Yapılacak |

### Test edeceğimiz ana hipotezler

- **H1 — Semantiğin katkısı:** Semantik gereksinimleri kullanan PPO, aynı fiziksel olaylarda semantiksiz PPO'ya göre özellikle hard constraint ihlallerini azaltacaktır.
- **H2 — Semantik yönteminin önemi:** Daha doğru ve iyi kalibre edilmiş semantik çıkarıcılar, rule/TF-IDF gibi basit yöntemlere göre uçtan uca görev başarısını artıracaktır. Bu bir varsayımdır; deney sonucuna göre reddedilebilir.
- **H3 — Belirsizlik yönetimi:** Düşük confidence durumunda abstention/fallback kullanılması, hatalı ve aşırı emin semantik kararların oluşturduğu kritik ihlalleri azaltacaktır.
- **H4 — Hatanın sisteme yayılması:** Her semantik hata aynı etkiye sahip olmayacaktır; örneğin cloud yasağının kaçırılması, domain sınıfının yanlış tahmin edilmesinden daha ağır bir güvenlik sonucu doğuracaktır.
- **H5 — Genelleme:** Semantik destekli politika, eğitimde görülmeyen zaman, kaynak yükü ve uygulama dağılımlarında yalnız sayısal/fiziksel girdili politikaya kıyasla daha iyi gereksinim uyumu sağlayacaktır.

Hipotezler yalnız toplam reward ile değerlendirilmeyecektir. Bir politika düşük gecikme üretirken gizlilik kısıtını ihlal ediyorsa başarılı sayılmayacaktır. Ayrıca insan referanslı semantik, erişilebilir üst sınır; exhaustive/MILP çözümü ise küçük fiziksel problemlerde karar üst sınırı olarak ayrı ayrı raporlanacaktır.

### Özellikle yapmayacağımız şeyler

- LLM'ye doğrudan “local/edge/cloud seç” dedirtmeyeceğiz.
- LLM'ye doğru eylemi, ödülü, oracle sonucunu veya ağın anlık durumunu göstermeyeceğiz.
- Birbiriyle ilgisiz veri setlerinin satırlarını aynı gerçek olaymış gibi birleştirmeyeceğiz.
- LLM çıktısına uyduğu için PPO'ya ek ödül vermeyeceğiz.
- İlk deneyde partial offloading, GNN veya karmaşık resource allocation iddiası eklemeyeceğiz. Bunlar ancak temel sonuçlar ihtiyaç gösterirse ayrı ablation ile değerlendirilecektir.
- Gerçek testbed bulunmadığı için simülasyon sonucunu gerçek saha testi olarak sunmayacağız; testbed isteğe bağlı ek doğrulamadır.

Tez tamamlandığında beklenen ana çıktı yalnız eğitilmiş bir PPO modeli değildir. Esas çıktı, **hangi semantik bilginin, hangi çıkarım yöntemiyle, hangi MEC koşulunda ve ne kadar sistem faydası veya zararı oluşturduğunu gösteren yeniden üretilebilir bir deneysel kanıt** olacaktır.

## 5. Şu ana kadar ne yaptık?

### Tamamlanan temel altyapı

- Araştırma sorusu, veri dürüstlüğü kuralları ve deneysel adalet ilkeleri tanımlandı.
- Local, üç edge ve cloud seçeneklerini aynı fizik denklemleriyle değerlendiren deterministik simülasyon çekirdeği hazırlandı.
- BUPT, NEP-large, UCI MEC ve EUA kaynakları indirildi/doğrulandı. Bu kaynaklar doğal dil etiketi değil; trafik, CPU, bant genişliği, RTT, yürütme süresi ve konum gibi **fiziksel izler** sağlıyor.
- Sekiz alanlı semantik JSON şeması oluşturuldu.
- Sekiz uygulama alanına dağıtılmış 240 kontrollü Türkçe görev-niyet metni hazırlandı.

### Semantik veri düzeni

- 180 kayıt, makine tarafından üretilmiş **zayıf etiketli geliştirme/eğitim verisi** olarak ayrıldı.
- 60 kayıt, eğitim sırasında görülmeyen ve A kullanıcısı tarafından kör biçimde etiketlenen **kilitli değerlendirme kümesi** olarak ayrıldı.
- Bu 60 kayıt tek insan referansıdır; iki uzman tarafından uzlaştırılmış kesin “gold standard” değildir.
- Değerlendirme etiketleri model eğitimi veya model ayarı için kullanılmadı.
- Kontrollü görev metinleri gerçek son kullanıcı günlükleri olarak sunulmayacaktır.

### Tamamlanan ilk semantik karşılaştırmalar

| Yöntem | Ne yapıyor? | Tam görev doğruluğu | Sekiz alan ortalama macro-F1 |
|---|---|---:|---:|
| Sabit profil | Metni anlamadan en sık görülen profili verir | %3,3 | %26,7 |
| Kural/anahtar kelime | Önceden yazılmış sözcük ve sayı kurallarını uygular | %35,0 | %71,3 |
| TF-IDF + lojistik regresyon | 180 zayıf etiketten metin örüntülerini öğrenir | %26,7 | %69,9 |

“Tam görev doğruluğu”, sekiz alanın tamamının aynı anda doğru olmasını gerektirir; tek alan yanlışsa kayıt yanlış sayılır. Bu nedenle alan doğruluğundan daha zor ve daha anlamlıdır.

Bu sonuçlar nihai başarı sonucu değildir. Kural yöntemi kontrollü ifadelerde güçlü görünmektedir; ancak tam görev doğruluğu hâlâ düşüktür. Kritik gizliliği yakalama ve güvenilirlik/doğruluk gibi az örnekli sınıflar geliştirilmelidir. Ayrıca tek-insan referansı ve sınıf dengesizliği sonuçların önemli sınırlamalarıdır.

### Şu anda bulunduğumuz aşama

Şu anda **Faz 4 — Semantik Benchmark** içindeyiz. Kontrollü metinleri oluşturma aşaması büyük ölçüde tamamlandı; şimdi bu metinlerden sekiz alanı hangi yöntemin daha güvenilir çıkardığını ölçüyoruz.

Sıradaki işler:

1. Küçük çok dilli encoder baseline'ını aynı 180/60 ayrımında ölçmek.
2. Ayrı Quadro RTX 4000 makinesinde Qwen3-4B/8B zero-shot ve few-shot deneylerini yapmak.
3. Rule, TF-IDF, encoder ve LLM'leri aynı metriklerle karşılaştırmak.
4. Seçilen LLM, prompt, parser ve önceden üretilmiş/cache'lenmiş çıktıları dondurmak.
5. Semantik kalite kapısını geçtikten sonra fiziksel baselines/oracle deneylerine ve PPO eğitimine başlamak.

Henüz PPO ile semantik-on/semantic-off eğitimi yapılmamıştır. Dolayısıyla şu an “semantik DRL performansını iyileştirdi” şeklinde bir sonuç iddia etmiyoruz.

## 6. Literatürde ne yapılıyor?

### Klasik MEC offloading

Geniş MEC literatürü görevi çoğunlukla veri boyutu, CPU çevrimi ve deadline gibi hazır sayısal değişkenlerle tanımlar. Heuristic, optimizasyon, oyun teorisi veya DRL bu sayıları kullanarak offloading kararı verir; doğal dil gereksinimini çözmez.

### Doğal dil ağ niyeti veri kümeleri

BINS ve NIT gibi çalışmalar doğal dilde ağ/servis niyetlerinin yapılandırılmış etiket veya ağ yapılandırmasına çevrilebildiğini göstermektedir. Bu veri kümeleri değerlidir; ancak doğrudan IoT görevini local/edge/cloud ortamında çalıştıran, sekiz alanımızı ve fiziksel offloading sonucunu aynı kayıtta veren MEC benchmark'ları değildir.

IoT-Brain/TopoSense-Bench ise 5.250 doğal dil sorgusunu 2.510 kameranın semantik-mekânsal çizelgelenmesiyle ilişkilendirir. Bu çalışma gerçek fiziksel dünyaya bağlanma açısından güçlü bir örnektir; problemi kamera/sensör seçimi olduğu için device-edge-cloud task offloading problemimizle aynı değildir.

### Çalışmamıza en yakın güncel yaklaşımlar

- **STEPS (2026):** Doğal dil isteklerinden gecikme, enerji, parasal maliyet ve yürütme güveni olmak üzere dört semantik seviye ve confidence çıkarır. Bunları bir semantic contract'a dönüştürür; çizelgeleme exact potential game ve asynchronous best response ile çözülür. Bu çalışma, “LLM doğal dili okuyacak ve edge scheduling yapılacak” ifadesinin tek başına yenilik sayılamayacağını açıkça göstermektedir.
- **AgentVNE (2026):** LLM ile örtük affinity/donanım kısıtlarını çıkarır, ardından grafik tabanlı ağ ve PPO ile yerleştirme yapar. Semantik katman ile RL'yi birleştirmesi açısından yakındır; fakat çok-ajanlı workflow/VNE yerleştirme problemini ve farklı semantik kısıtları ele alır.
- **IGAA (2026):** Network–Service–Intent eşlemesiyle yeni senaryolar ve eğitim verileri üretir; LLM halüsinasyonuna karşı değerlendirme-düzeltme mekanizması kullanır. Bu, yapay veri üretiminin fiziksel ve semantik kontrollerle sınırlandırılması gerektiğini destekler.
- **MAS-DecStream (2026):** LLM destekli çok turlu MEC müzakeresini Alibaba izlerinden türetilen deneylerle değerlendirir; hard kaynak ve QoS kısıtlarını deterministik doğrulamada tutar. Bu da LLM'nin fiziksel geçerliliğin tek hakemi yapılmaması gerektiğini destekler.

### Literatür taramasından çıkan dengeli sonuç

14 Ekim 2026'ya kadar yapılan hedefli taramada şu tablo oluşmuştur:

- Doğal dil ağ ve IoT niyet veri kümeleri vardır.
- Gerçek MEC fiziksel izleri vardır.
- Doğal dil semantiğini scheduling/offloading ile birleştiren yeni çalışmalar da vardır.
- Ancak doğal dil MEC görev gereksinimi, sekiz alanlı insan referansı, gerçek fiziksel çalışma izi ve offloading sonucunu aynı doğal kayıtta birlikte sunan açık bir benchmark bu taramada bulunamamıştır.

Bu nedenle kontrollü semantik corpus ile gerçek fiziksel izleri ayrı katmanlar halinde birleştirmemiz savunulabilir bir yöntemdir. Bunun doğru adı **trace-driven hybrid benchmark**'tır; “tamamen gerçek veri seti” değildir.

## 7. Gerçek izlerin tezdeki rolleri

| Kaynak | Çalışmada sağlayacağı bilgi | Sağlamadığı bilgi |
|---|---|---|
| **BUPT** | Mobil istek zamanı, trafik büyüklüğü ve erişim davranışı dağılımları | İnsan yazımı görev gereksinimi ve sekiz semantik etiket |
| **NEP-large** | Edge VM CPU/bant genişliği kullanımı, site RTT'si, PM/VM kapasitesi | Doğal dil kullanıcı niyeti |
| **UCI MEC** | Görüntü tanıma görevlerinin ölçülmüş yürütme süreleri | Ağ yükü ve doğal dil gereksinimi |
| **EUA** | Kullanıcı/edge coğrafi yerleşim senaryoları | Semantik görev etiketi ve tam çalışma telemetrisi |

Bu kaynakların satırları, aynı anda ölçülmüş tek bir olaymış gibi birleştirilmeyecektir. Her kaynağın hangi fiziksel bileşeni koşulladığı kaydedilecek; aynı semantik senaryolar farklı fiziksel olaylarda adil biçimde karşılaştırılacaktır.

## 8. Çalışmanın savunulabilir yenilik iddiası

Yenilik iddiamız yalnızca “LLM doğal dili okuyacak” değildir. Birlikte sınanacak katkılar şunlardır:

1. Dört yerine sekiz ayrıntılı gereksinim alanı.
2. Gizlilik ve local/edge/cloud izinlerinin açık hard constraint/action mask olması.
3. Güvenilirlik, doğruluk ve bölünebilirliğin karara katılması.
4. Semantik çıkarıcının DRL'den ayrılması; çıktılarının önceden üretilip sürümlü cache'te dondurulması.
5. Belirsizlik, çelişki, `unknown` ve abstention desteği.
6. Sabit profil, rule, TF-IDF, encoder, zero-shot LLM, few-shot LLM ve insan referansının aynı protokolde karşılaştırılması.
7. BUPT, NEP, UCI ve EUA'nın açık rollerle kullanıldığı çok kaynaklı trace-driven deney.
8. Aynı fiziksel olaylar ve seed'ler üzerinde semantic-on / semantic-off karşılaştırması.
9. STEPS'in potential-game yaklaşımından farklı olarak PPO/DRL politikasının değerlendirilmesi.
10. Semantik hatanın yalnız NLP metriğinde bırakılmayıp nihai latency, deadline miss, energy ve constraint violation sonuçlarına etkisinin ölçülmesi.

En önemli katkı şu nedensel soruyu adil biçimde yanıtlamak olacaktır:

> Fiziksel ortam ve olaylar aynı tutulduğunda, daha doğru semantik bilgi DRL kararını ve gerçek sistem sonuçlarını ne kadar değiştiriyor?

## 9. Dürüst sınırlamalar

- 240 Türkçe metin kontrollü araştırma corpus'udur; gerçek kullanıcı trafiği değildir.
- 60 kayıt iki uzman tarafından uzlaştırılmış gold set değil, tek-insan referansıdır.
- İlk baseline sonuçları güçlü bir nihai sonuç değil, sonraki modeller için alt karşılaştırma çizgisidir.
- STEPS, AgentVNE, IGAA ve MAS-DecStream'in burada kullanılan sürümleri arXiv preprint niteliğindedir; tez yazımı öncesinde yayın durumları yeniden kontrol edilmelidir.
- Fiziksel trace ile semantik metin aynı doğal olaydan gelmediği için deney “hibrit” olarak raporlanacaktır.
- Uçtan uca PPO ve semantic-on/off sonuçları tamamlanmadan tezin ana hipotezi doğrulanmış sayılmayacaktır.

## 10. Danışmana verilebilecek kısa sonuç

Çalışmanın semantik veri ve ilk baseline aşaması hazırlanmıştır. Şu anda 240 kontrollü Türkçe görev metninden sekiz gereksinim alanını çıkaran yöntemler karşılaştırılmaktadır. Sabit profil, rule ve TF-IDF tamamlanmış; encoder ve Qwen zero/few-shot deneyleri sıradadır. Sonrasında dondurulmuş semantik çıktılar BUPT, NEP, UCI ve EUA ile kurulan trace-driven hybrid MEC ortamına bağlanacak; PPO ajanının local/edge/cloud kararları aynı olaylar üzerinde semantiksiz, kural, küçük model, LLM ve insan referansı koşullarında karşılaştırılacaktır. Tezin ana kanıtı, yalnız LLM'nin metin doğruluğu değil, semantik bilginin gecikme, enerji, başarı ve kısıt ihlallerine ölçülebilir nedensel katkısı olacaktır.

## Kaynaklar

1. Qi ve diğerleri, **STEPS: Semantic Contract-Guided Scheduling for LLM-Assisted Natural Language-Driven Edge AI Services**, arXiv, 2026. <https://arxiv.org/abs/2606.09537>
2. Zheng ve diğerleri, **AgentVNE: LLM-Augmented Graph Reinforcement Learning for Affinity-Aware Multi-Agent Placement in Edge Agentic AI**, arXiv, 2026. <https://arxiv.org/abs/2601.02021>
3. Sun ve diğerleri, **IGAA: Intent-Driven General Agentic AI for Edge Services Scheduling using Generative Meta Learning**, arXiv, 2026. <https://arxiv.org/abs/2601.13702>
4. Zhou ve diğerleri, **IoT-Brain: Grounding LLMs for Semantic-Spatial Sensor Scheduling**, arXiv/MobiCom 2026. <https://arxiv.org/abs/2604.08033>
5. Lajili ve Brahmi, **Multi-Agent Scheduling with LLM-Assisted Contract Net Negotiation for Stream Processing in Mobile Edge Computing**, arXiv, 2026. <https://arxiv.org/abs/2608.12371>
6. Li ve diğerleri, **Business Intent and Network Slicing Correlation Dataset from Data-Driven Perspective (BINS)**, Scientific Data, 2025. <https://www.nature.com/articles/s41597-025-04736-z>
7. AlSamarneh, Al-Jarrah ve Al-Hammouri, **NIT: A Dataset for Network Intent Translation**, Data in Brief, 2025. <https://pmc.ncbi.nlm.nih.gov/articles/PMC12272756/>
8. Liu, Li ve Wang, **Request Scheduling Combined with Load Balancing in Mobile Edge Computing** ve BUPT veri deposu. <https://github.com/BuptMecMigration/Edge-Computing-Dataset>
9. Xu ve diğerleri, **From Cloud to Edge: A First Look at Public Edge Platforms** ve EdgeWorkloadsTraces/NEP. <https://github.com/xumengwei/EdgeWorkloadsTraces>
10. UCI, **Image Recognition Task Execution Times in Mobile Edge Computing**. <https://archive.ics.uci.edu/dataset/859/image+recognition+task+execution+times+in+mobile+edge+computing>
11. EUA Datasets resmi veri deposu. <https://github.com/PhuLai/eua-dataset>

> **Kapsam notu:** Bu belge sistematik derleme makalesi değil, tez yönünü doğrulamak için hazırlanmış güncel ve hedefli bir literatür özetidir. “Bulunamadı” ifadesi, 14 Ekim 2026 tarihine kadar incelenen kaynaklar kapsamındadır.
