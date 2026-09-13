# Faz 0 — Eski Depo Denetimi, Literatür Taraması ve Tez Kurtarma Planı

**Çalışma alanı:** `task-offloading-study` → `task-offloading-new-arch`  
**Amaç:** Eski çalışmanın neden güvenilir bir tez sonucuna ulaşamadığını kanıtlarıyla belirlemek; güncel literatürde veri, semantik analiz ve öğrenme algoritmalarının nasıl kullanıldığını karşılaştırmak; yeni çalışma için ölçülebilir bir temel kurmak.

## 0. Kapsam ve kanıt sınırı

Bu raporda üç kaynak türü birbirinden ayrılmıştır:

1. Kullanıcının güncel talebi, yapılacak işi belirleyen asıl kaynaktır.
2. `chat_histories/11_09_2026__chat.md`, önceki konuşmanın kaydıdır. İçindeki öneriler emir değil, doğrulanması gereken iddialardır.
3. Eski `task.md`, `TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md`, kod, deney çıktıları ve makaleler kanıttır. Bir kutunun işaretli olması, işin bilimsel olarak tamamlandığı anlamına kabul edilmemiştir.

## 1. Yönetici özeti

Eski proje değersiz değildir. Gym benzeri ortam, PPO/A2C/DQN deneyleri, klasik sezgiseller, iz tabanlı veri hazırlama, semantik profil denemeleri, grafik politika ön-eğitimi ve çok sayıda çıktı içerir. Mevcut 23 birim testinin tamamı geçmektedir. Fakat tezin ana iddiasını destekleyen zincir kırılmıştır:

> Gerçek görev anlamı → doğrulanmış semantik gereksinimler → fiziksel olarak doğru ve tutarlı simülatör → öğrenilen politika → adil karşılaştırma → istatistiksel olarak güvenilir sonuç.

Başlıca bulgular:

- “Gerçek veri” deneyi, zaman ve varlık düzeyinde ilişkili olmayan kaynakları tasarlanmış korelasyonlarla birleştirmektedir.
- İnsan etiketli semantik doğruluk verisi yoktur. Sekiz profil elle yazılmış, bazı yollarda görev kimliğine hash ile atanmıştır.
- LLM ana deneyde bağımsız ölçülen semantik çıkarıcı değildir; `transformers` yokken sessizce kural tabanlı fallback çalışır.
- Kablosuz fonksiyonun ikinci dönüş değeri mesafe olmasına rağmen SNR olarak kullanılmaktadır. Uzak cihaz daha iyi bağlantılı görünebilir.
- Eğitim ve SimPy gösterim ortamları aynı geçiş çekirdeğini kullanmaz; fizik ve enerji formülleri ayrışmıştır. SimPy tarafında çift kaynak edinme kaynaklı deadlock yolu vardır.
- Gerçek-veri deneylerinde PPO/A2C tek eyleme çökmüştür. Semantik açık/kapalı PPO sonuçları neredeyse aynıdır; semantik katkı tanımlanamamıştır.
- Grafik model PPO ile uçtan uca eğitilmemiştir; denetimli grafik sınıflandırıcısı ile vektör PPO yan yana konmuştur. GNN-PPO iddiası karşılanmaz.
- “Fairness” kullanıcı adaleti yerine eylem sayılarının Jain indeksidir; karar süresi semantik/state maliyetini dışarıda bırakır.
- Üç seed, güven aralığı ve planlı istatistik için yetersizdir; eski bitiş kriterindeki en az beş seed ve ileri metrikler tamamlanmamıştır.

Literatür taramasında tek başına bütün ihtiyacı karşılayan açık bir veri seti bulunmadı. Doğal dilli IoT görev niyeti, insan etiketli gecikme/gizlilik/güvenilirlik gereksinimi, zaman eşleşmiş radyo-mobilite-kuyruk-sunucu durumu, CPU/enerji ve gerçek offloading sonucu aynı veri setinde yoktur. Güncel çalışmalar çoğunlukla:

- Anlamsal iletişim için BERT/ViT/DeepSC gibi kodlayıcılarla içeriği sıkıştırır.
- Sayısal durumu metne çevirip LLM’den eylem veya prior ister.
- Mobilite, sunucu yükü ya da görev içeriğinin yalnızca bir eksenini gerçek veriden alıp kalanı simüle eder.

En savunulabilir tez sorusu şudur:

> Ağ durumunu görmeyen, dondurulmuş ve insan etiketlerine karşı bağımsız doğrulanmış bir LLM semantik gereksinim çıkarıcısı; aynı fizik, aynı görevler ve aynı seed’ler altında, semantiksiz ve LLM’siz semantik taban çizgilerine göre DRL politikasının kısıt ihlallerini ve genellemesini iyileştiriyor mu?

Bu soru ölçülebilir ve yanlışlanabilirdir. Sonucun “iyileştirmiyor” çıkması da bilimsel olarak değerlidir.

## 2. Yeni başlayanlar için kavramlar

### 2.1 MEC ve task offloading

Bir IoT görevi yerel cihazda, yakındaki edge/MEC sunucusunda veya uzaktaki cloud’da çalıştırılabilir. Yerel işlem ağ gecikmesini önler ama pil ve zayıf CPU sınırlıdır. Edge hızlı olabilir fakat aktarım ve kuyruk ekler. Cloud güçlüdür; uzak ağ, gizlilik ve bağlantı belirsizliği artabilir. Ajan, her görevde veri boyutu, CPU ihtiyacı, kanal, kuyruk, enerji, deadline ve gizliliğe göre hedef seçer.

DRL ajanı bir **durum** görür, **eylem** seçer ve **ödül** alır. Ortamın fiziği yanlışsa ajan yanlış dünyayı iyi öğrenir. Bu nedenle simülatör doğruluğu model karmaşıklığından önce gelir.

### 2.2 “Semantik” iki farklı anlama geliyor

1. **Anlamsal iletişim:** Ham resim/metin bitleri yerine görev için gerekli anlam temsilini göndermek.
2. **Görev niyeti analizi:** “Bu sağlık verisi cihazdan çıkmasın ve 50 ms içinde cevap gelsin” ifadesini yapılandırılmış gizlilik/gecikme kısıtlarına dönüştürmek.

Eski çalışma bunları yer yer karıştırmıştır. Yeni tez ikinci anlamı ana katkı yapmalıdır. Birinci anlam ancak ayrı bir deney faktörü olarak eklenmelidir.

### 2.3 “Gerçek veri” nasıl adlandırılmalı?

Her alanın statüsü açık olmalıdır:

- **Ölçülmüş:** Gerçek cihaz/testbed üzerinde ölçüldü.
- **İzden gözlenmiş:** Açık kaynaktan doğrudan alındı.
- **Türetilmiş:** Gözlenen alanlardan formülle hesaplandı.
- **Eşlenmiş:** Başka kaynaktan kontrollü eşleştirildi.
- **Üretilmiş:** Dağılım/simülatörden oluşturuldu.
- **Etiketlenmiş:** İnsan ya da model sınıflandırdı.

Tek bir sütunun gerçek izden gelmesi bütün benchmark’ı “tam gerçek” yapmaz.

## 3. Eski deponun nesnel envanteri

Yaklaşık 559 izlenen dosya görüldü: 77 Python dosyası, 30 yapılandırma ve yüzlerce veri/model/deney artefaktı. Sanal ortam yaklaşık 2.6 GB, ham veri yaklaşık 10.6 GB, ayrıca yaklaşık 1.85 GB’lık arşiv vardır. Çok sayıda CSV/model çıktısının kaynak kodla aynı depoda bulunması, hangi sonucun hangi kod-veri-config ile üretildiğini zorlaştırmaktadır.

Eski `task.md` üzerinde Faz 1–5, sentetik Faz 6 ve bazı 6R adımları işaretlidir. Buna rağmen eski TODO’nun bitiş ölçütlerindeki şu işler tamamlanmamıştır:

- En az sekiz anlamlı baseline ve altı kontrollü ablation.
- Sentetik ve iz tabanlı deneylerde hizalı fizik.
- En az beş seed, güven aralıkları ve istatistiksel testler.
- Gerçek GNN-PPO ve adil MLP karşılaştırması.
- Tam LLM/karar maliyeti.
- İleri metrikler, metodoloji belgesi ve son GUI.

Dolayısıyla işaretler “kod/çıktı üretildi” anlamındadır; “tez iddiası doğrulandı” anlamında değildir.

## 4. Çalıştırılan kontroller

Eski sanal ortamda `python -m unittest discover -v` ile **23/23 test geçti**; bağımlılık tutarlılığı kontrolü hata vermedi. Ancak şu uyarılar alındı:

```text
[LLM] Transformers not available. Using rule-based analyzer.
gui.py or pygame not found. Running in headless mode.
```

README/uygulamanın kullandığı `transformers`, `accelerate` ve `pygame` eski `requirements.txt` içinde yoktur. Testler; gerçek LLM yolunu, uçtan uca yeniden üretimi, dataset kökenini, eğitim-değerlendirme ortam eşitliğini ve seed istatistiğini kapsamamaktadır. Testlerin geçmesi yararlıdır fakat tez geçerliliği kanıtı değildir.

## 5. Eski depodaki eksiklikler

### 5.1 P0 — Sonucu geçersiz kılabilecek hatalar

#### P0.1 — Mesafe, SNR sanılmış

`src/env/simulation_env.py` içindeki `WirelessChannel.calculate_datarate`, veri hızı ve mesafe döndürür. `src/env/rl_env.py` ile `src/agents/baselines.py` ikinci değeri `snr` değişkenine atayıp kalite hesabında kullanır. Bu, sonuçları sistematik bozabilecek fizik hatasıdır. Yeni yapıda tuple yerine `LinkMetrics(rate_bps, snr_db, distance_m, path_loss_db)` gibi isimli/birimli tip kullanılmalıdır.

#### P0.2 — Eğitim ve gösterim iki ayrı fizik

RL ortamı ve SimPy gösterimi farklı gecikme, enerji ve cloud formülleri kullanır. “Simülatör destekli eğitim” yolu SimPy kaynaklarını oluştursa da eğitim geçişleri olay simülasyonunu ilerletmez. Model bir dünyada eğitilip başka dünyada anlatılmaktadır. Yeni sistemde Gym, olay koşucusu, baseline ve oracle tek saf geçiş çekirdeğini çağırmalıdır.

#### P0.3 — SimPy deadlock yolu

Kısmi edge yolu sunucu kaynağını edinir, ardından `EdgeServer.process_task` kapasitesi bir olan aynı kaynağı yeniden ister. İlk istek bırakılmadan ikincisi karşılanamayabilir.

#### P0.4 — Benchmark’a hedef kararın gömülmesi

Birleşik iz oluşturucu farklı kaynaklardan alanları yapay olarak ilişkilendirir. “Correlation aware” yolu veri boyutu, CPU, öncelik ve hizmet sınıfı arasında istenen korelasyonları kurar. Deadline, referans fiziksel eylemin gecikmesine göre üretilebilmektedir. Mevcut 5.000 satırda veri boyutu–CPU Spearman korelasyonu yaklaşık 0.93’tür. Ajan doğal ilişki keşfetmek yerine tasarımcının gömdüğü hedefi öğrenebilir.

#### P0.5 — Eğitim/test sızıntısı

Normalizasyon istatistikleri veri ayrılmadan önce bütün kayıt üzerinde hesaplanır. Kayıtlar zaman ve cihaz/istasyon kimliği korunmadan karıştırılır. Yeni hatta normalizatör yalnız eğitim split’ine fit edilmeli; kronolojik ve entity/application holdout’ları ayrı sınanmalıdır.

### 5.2 P1 — Ana iddiayı desteklemeyen tasarımlar

#### P1.1 — Semantik gerçek görev anlamından gelmiyor

Sekiz YAML profili şeffaf ama elle yazılmıştır; bazı deneylerde profil görev kimliğinin hash’ine göre atanır. Sağlayıcı yokken öncelik/deadline sıkılığı semantik vekili olur. Bunlar doğal dil görev analizi değildir ve fiziksel hedefle döngüsel ilişki oluşturabilir.

#### P1.2 — LLM bağımsız bir gereksinim çıkarıcı değil

Eski analizör isteğe bağlı küçük model kullanır; bağımlılık yoksa kural tabanlı yola düşer. Ağ/cihaz durumunu da görebilir ve gereksinim yerine doğrudan eylem hedefi üretir. Böylece LLM semantik başarısı fiziksel karar başarısından ayrı ölçülemez.

#### P1.3 — Semantik ödülde iki kez sayılıyor

Semantik hem durum/prior’ı etkiler hem de “semantik uyum” bonusuyla ödüle tekrar girer. Ajan fiziksel sonucu iyileştirmek yerine elle yazılmış öneriye katılmayı öğrenebilir. Yeni ödül gecikme, enerji, başarısızlık ve kısıt ihlalinden oluşmalı; semantik yalnız gereksinim ve action mask belirlemelidir.

#### P1.4 — MDP tam gözlenebilir değil

`snr_norm` gerçekte veri hızıdır. Mesafe/hız/mobilite, açık kuyruk ve bazı geçiş değişkenleri eksiktir. Switching cezası önceki eyleme bağlıyken önceki eylem gözlemde yoktur. Ya durum tamamlanmalı ya problem POMDP olarak tanımlanmalıdır.

#### P1.5 — Başlıktaki eylem alanı uygulanmamış

Kısmi offloading lineer, paralel ve masrafsız varsayılır; bölünebilirlik, bağımlılık, birleştirme, dönüş verisi ve kod taşıma modellenmez. Ajan edge sunucu seçmez, bant genişliği/CPU tahsis etmez. Bu nedenle mevcut deney “dynamic resource allocation” iddiasını karşılamaz.

#### P1.6 — GNN-PPO eğitilmemiş

Grafik politika cross-entropy ile hedef eylem sınıflandırıcısı olarak eğitilmiş, sonra vektör PPO ile karşılaştırılmıştır. Aynı PPO kaybıyla uçtan uca grafik politika optimizasyonu yoktur. GNN ancak değişken topoloji gerektiğinde, aynı bilgi ve eğitim bütçesiyle denenmelidir.

### 5.3 Sonuçların dürüst okuması

İz tabanlı raporda oracle başarı tavanı yaklaşık %91.8 ve oracle’ın yaklaşık %70’i `edge_75` seçimidir. DeadlineAwareGreedy yaklaşık %76.5, Greedy %76.3, GA %75.5 iken A2C/PPO yaklaşık %66, DQN %46.5, Random %34’tür.

Yeniden eğitilen PPO/A2C neredeyse her kayıtta `edge_75` seçmiştir. Semantik açık/kapalı PPO yaklaşık %66.2–%66.4 aralığında aynı çökmüş dağılımdadır. Sentetikte cloud/greedy yaklaşık %78–%80 ile PPO/A2C’nin yaklaşık %65’ini geçer. Dolayısıyla:

- DRL en güçlü baseline değildir.
- Politika duruma anlamlı tepki vermemektedir.
- Semantik katkı gösterilememiştir.
- Küçük farklar seed, ödül veya veri tasarımından kaynaklanabilir.

Eski depo bundan sonra başarı kanıtı değil, hata kataloğu ve karşılaştırmalı arşiv olarak korunmalıdır.

### 5.4 P2 — Ölçüm ve mühendislik sorunları

- Jain fairness, kullanıcı/sunucu adaleti yerine eylem sayılarından hesaplanır.
- Birbiriyle uyumsuz iki QoE formülü vardır.
- Decision overhead yalnız `model.predict` ölçer; semantik/state/iletişim dışarıdadır.
- `battery_depletion`, gerçek cihaz ömrü yerine görev bayrağı ortalamasına yakındır.
- Global RNG kullanımı vektör ortamlarda tekrarlanabilirliği zayıflatır.
- Gösterim ortamı görsellik için PPO eylemini rastgele değiştirebilir.
- Veri, checkpoint ve rapor çıktıları kaynak ağacına karışmıştır.

## 6. Literatür taraması: kim ne yapmış?

Tarama; yayıncı sayfaları, arXiv tam metinleri, resmi veri depoları ve standart kuruluşlarına dayanır. Odak 2023–2026 semantik-aware MEC, LLM destekli offloading, DRL, gerçek iz ve testbed çalışmalarıdır. Çok yeni ön baskılar hakemli sonuç gibi sunulmamıştır.

| Çalışma | “Semantik” ne demek? | Yöntem/algoritma | Veri ve deney | Bu tez için ders |
|---|---|---|---|---|
| Cang vd., 2023 [1] | Görev verisinin semantik bölünmesi/sıkıştırılması | Değişimli optimizasyon, konveks/geometrik alt problemler | 10 kullanıcılı sentetik parametreler | Semantik oran ile iletişim/hesap kaynağı birlikte optimize edilebilir; LLM/DRL yok |
| Ji & Qin, 2024 [2] | DeepSC ile metin anlam temsilinin iletimi | DNN aktör + model tabanlı optimize edici eleştirmen, Lyapunov/imitation | Makine çevirisi cümle varsayımı; 8 cihaz, 15 bin sentetik slot | Model tabanlı oracle’dan ön-eğitim ve kuyruk/kısıt modeli değerlidir |
| Chen vd., 2025 [3] | Metin/görüntü/multimodal özellik sıkıştırması | MAPPO; D3QN karşılaştırması | SST-2, CIFAR-10, VQAv2 içerikleri; ağ/gelişler simüle | Gerçek içerik dataset’i gerçek ağ izi değildir; sıkıştırma-doğruluk maliyeti ölçülür |
| Song vd., ICTC 2024 [4] | Sayısal MEC durumundan LLM ile offloading | Açık metadata mevcut, tam yöntem ayrıntısı sınırlı | Gerçek semantik etiket kanıtı yok | Kapalı ayrıntı üzerine yeniden üretilebilirlik iddiası kurulmamalı |
| Ren vd., 2024/25 [5] | Sistem durumuna RAG bağlamı ekleyip eylem üretmek | BGE + LlamaIndex, GPT-4o/Qwen; DQN/DDPG/PPO baselines | DSCC/DUSD/DUP/DUCC adlı küçük sentetik parametre taramaları | RAG bilgi getirir; bunlar gerçek görev dili veya gerçek MEC izi değildir |
| Zhu vd., 2025 [6] | LLM ile çok ajanlı problem bölgeleme/işbirliği | LLM-QTRAN, GCN, self-attention | Sentetik UAV/MEC alanı ve kullanıcılar | LLM yapısal prior üretebilir; başarı tek başına semantik anlama kanıtı değildir |
| AgentVNE, 2026 ön baskı [7] | Doğal dilde sert yerleştirme gereksinimleri | Qwen3-30B; çift akış GCN/Transformer; NodeRank ön-eğitim + PPO | Waxman sentetik topoloji/workflow | Sert kısıtı bias/action mask’e çevirme ve aşamalı eğitim aktarılabilir; gerçek dataset yok |
| LeDRL, 2026 ön baskı [8] | Görev/topoloji metninden bağlam, hafıza/reflection | Qwen3-4B + attention füzyonu + PPO | Sentetik 10–20 düğüm; Jetson testbed, YOLOv8/COCO | Donanım ölçümü değerlidir; geliş/arıza yine üretilebilir; LLM maliyeti ayrılmalı |
| COMLLM, 2026 ön baskı [9] | Sayısal durumu değişken uzunlukta metne dönüştürme | Qwen 1.5B/7B, oracle SFT + GRPO, look-ahead ödül | 1k SFT, 2k GRPO, 1k test; sentetik görev/sunucu | Değişken topolojiye metinle genelleme ilginç; insan niyeti analizi değildir |
| Vehicular LLM Offloading, 2026 [10] | Araç durumundan offloading sınıflandırması | Quantized/fine-tuned Llama 3.2 1B; RF/XGBoost | Kaggle/SUMO/OMNeT++ ve araç izlerinin karışımı | Gerçek/sentetik alanlar ayrılmalı; çekirdek yöntem DRL değildir |
| O2O-DRL, 2024 [11] | Semantik değil; sim-to-real aktarım | Sezgisel loglarla offline ısınma + on-policy DRL | Simülasyon ve Kubernetes testbed | Oracle/sezgisel ön-eğitim ve gerçek sistemde düzeltme iyi yöntemdir |
| ADPRL, 2024 [12] | Semantik değil; bağımlı görev DAG’ları | Asynchronous deep progressive RL | Alibaba cluster trace + sentetik DAG/ağ | Gerçek sunucu yükü MEC radyo/niyet verisi değildir |

### 6.1 Soruların doğrudan cevapları

**Semantik analiz yapan var mı?** Evet; fakat çoğu çalışma semantik iletişim yapar: BERT, ViT veya DeepSC ile içeriği görev odaklı temsile sıkıştırır. Doğal dil kullanıcı niyetinden gizlilik, gecikme ve güvenilirlik çıkaran çalışma azdır. AgentVNE doğal dil sert yerleştirme koşullarını ayrıştırmasıyla en yakın örnektir; problem VNE/service placement ve verisi sentetiktir.

**LLM nasıl kullanılıyor?** Dört örüntü görülür:

1. Sayısal sistem durumunu metne çevirip doğrudan eylem istemek.
2. LLM’den eylem prior’ı veya yapılandırılmış kısıt çıkarmak; son kararı DRL’ye bırakmak.
3. RAG ile geçmiş/uzman bilgisini prompt’a getirmek.
4. Oracle çözümlerle SFT/behavior cloning, ardından PPO/GRPO ile iyileştirmek.

Bu tez için ikinci örüntü en güvenlisidir. LLM fiziksel hedef seçmemeli; insan dilini denetlenebilir gereksinimlere çevirmelidir. DRL anlık ağı görerek fiziksel karar verir.

**Hangi algoritmalar eğitiliyor?** PPO/MAPPO, DQN/D3QN, DDPG, QTRAN, QMIX, MASAC ve GRPO görülür. Algoritma sayısını artırmak tek başına katkı değildir. Sabit boyutlu ilk problem için PPO ana model, DQN ikincil model; sezgiseller ve küçük exhaustive/MILP oracle güçlü temel için yeterlidir. MAPPO/GNN-PPO ancak problem gerçekten çok ajanlı veya değişken topolojili olduğunda eklenmelidir.

## 7. Dataset denetimi

| Kaynak | Gerçekte ne içeriyor? | İçermediği kritik alanlar | Uygun kullanım |
|---|---|---|---|
| Glasgow MEC [13] | Roma taksi hareketi ile Alibaba kullanımının birleştirilmiş küçük izi | Ortak gerçek zaman/varlık, görev dili, CPU/enerji, offload sonucu | Birleşik veri metodolojisi; tam gerçek diye değil |
| UCI Image Recognition Execution Times [14] | Dört heterojen sunucuda 4.000 görüntü tanıma turnaround ölçümü | Mobilite, radyo, görev niyeti, enerji, tam yük telemetrisi | Hizmet süresi kalibrasyonu |
| Alibaba Cluster Trace v2018 [15] | Büyük veri merkezi iş/kaynak günlükleri | Edge radyo, IoT mobilitesi, kullanıcı anlamı | Sunucu yükü stres senaryosu |
| Google Cluster Data [16] | Borg görev/makine olayları ve kullanım | MEC ağı, IoT görevi, semantik QoS | Cloud yükü/arrival araştırması |
| BUPT Edge Dataset [17] | 480 binden fazla mobil kullanıcı, 16 bin baz istasyonu; erişim, byte, URL/port/içerik türü | CPU cycle, deadline, enerji, açık niyet | Trafik/service-category ve kaba mobilite; lisans/mahremiyet kontrolü şart |
| EUA [18] | Avustralya’dan gerçek kullanıcı ve edge konumları | Görev, zaman serisi, kanal, kuyruk, semantik | Topoloji/yerleşim senaryosu |
| EdgeDroid [19] | İnsan etkileşim izlerini gerçek backend’de replay | Eksiksiz kullanıcı çeşitliliği ve bütün telemetri | Tekrarlanabilir human-in-loop benchmark örneği |
| BigMEC [20] | Edge hizmet göçü ham/işlenmiş ölçümleri | Genel offloading niyeti ve bütün görev türleri | Service migration alt problemi |
| SST-2/CIFAR-10/VQAv2/COCO [3,8] | Metin/görüntü/multimodal içerik ve sınıf/cevap etiketi | MEC ağı, CPU/enerji/deadline, QoS niyeti | Uygulama içeriği ve doğruluk-sıkıştırma eğrisi |

### 7.1 Neden tek dataset yetmiyor?

İhtiyaç aynı zaman çizelgesinde şu zinciri bağlamaktır:

```text
görev metni/niyeti
        ↓
insan etiketli gereksinimler
        ↓
görev boyutu ve gerçek hesaplama süresi
        ↓
o andaki kanal + mobilite + kuyruk + sunucu yükü
        ↓
hedef, gecikme, enerji, başarı ve ihlaller
```

Mevcut açık kaynakların her biri zincirin bir-iki parçasını verir. Farklı kaynaklar birleştirilebilir; fakat eşleştirme “doğal korelasyon” gibi sunulamaz. Her kaynak ayrı deney faktörü olmalı, köken manifesti tutulmalı ve sonuç **trace-driven hybrid benchmark** diye adlandırılmalıdır.

### 7.2 Üç katmanlı veri stratejisi

1. **Doğrulanmış sentetik çekirdek:** Denklemleri, queue’yu, action mask’i ve oracle’ı elle hesaplanabilir örneklerle doğrular.
2. **İz güdümlü hibrit benchmark:** BUPT’tan trafik/hareket, UCI’dan hizmet süresi, Alibaba’dan ayrı yük stresi alınabilir. Her alanın kökeni yazılır; korelasyon icat edilmez.
3. **Küçük ortak testbed:** Görev niyeti, sistem durumu ve gerçek sonuç birlikte ölçülür. Mevcut donanıma göre Docker/Kubernetes, laptop, Raspberry Pi veya Jetson kullanılabilir. Az örnekli fakat nedensel bağı güçlü bu set en değerli doğrulama olabilir.

## 8. Yeni semantik analiz tasarımı

### 8.1 LLM’nin tek görevi

LLM yalnız doğal dil görev tanımını ve kullanıcı politikasını katı JSON’a dönüştürmelidir:

```json
{
  "latency": {"class": "critical", "max_ms": 50, "explicit": true},
  "privacy": {"local_only": false, "edge_allowed": true, "cloud_allowed": false},
  "reliability": {"tier": "high", "min_probability": null},
  "accuracy": {"tier": "medium", "min_score": null},
  "energy_priority": "low",
  "task_divisible": false,
  "evidence_spans": ["bulut ortamına gönderilmemeli", "50 ms"],
  "confidence": 0.91,
  "unknown_fields": ["min_probability", "min_score"]
}
```

Model bilmediği sayıyı uydurmamalı; `null/unknown` döndürmelidir. Kanıt parçaları etiketin metnin neresinden çıktığını denetletir.

### 8.2 İnsan ground truth ve baselines

- ETSI MEC kullanım senaryoları [21] temel alınarak sağlık, fabrika, video analizi, AR, trafik, tarım gibi dengeli niyet evreni hazırlanır.
- Şablon dışı insan yazımı/parafraz, çelişki, eksik bilgi ve alan dışı örnek eklenir.
- En az iki etiketçi bağımsız etiketler; anlaşmazlık adjudication ile çözülür.
- Kategorik alanlarda Cohen’s kappa veya Krippendorff alpha raporlanır.
- Split’ler şablon, uygulama ve paraphrase ailesi sızmayacak biçimde yapılır.
- Sabit profil, kural tabanlı parser, TF-IDF+lojistik regresyon, küçük encoder, zero-shot LLM, few-shot LLM ve insan oracle karşılaştırılır.
- Exact match, macro/micro F1, alan bazlı precision/recall, Brier/ECE, şema geçerliliği, abstention, gecikme ve maliyet ölçülür.

Model/prompt/sıcaklık/parser/örnek seti hash’lenip dondurulmalı; RL sırasında canlı API yerine önceden üretilmiş cache kullanılmalıdır.

### 8.3 Nedensel izolasyon

Aynı fiziksel görev farklı niyetlerle eşleştirilir: “en hızlı”, “veri tesisten çıkmasın”, “pil öncelikli; iki saniye kabul”. Fizik sabitken yalnız semantik değişirse eylem farkı semantiğe atfedilebilir. Ters deneyde semantik sabit, kanal/yük değişir. Bu faktöriyel tasarım eski yapay korelasyon sorununu önler.

## 9. Yeni simülatör ve yazılım mimarisi

```text
src/task_offloading/
├── domain/       # Birimli Task, Device, Server, LinkMetrics tipleri
├── sim/          # Tek geçiş çekirdeği, kanal, enerji, kuyruk
├── env/          # Gymnasium adaptörü
├── data/         # Manifest, doğrulama, split, dönüşüm
├── semantics/    # Şema, annotasyon, parser, LLM cache/evaluation
├── policies/     # Heuristics, oracle, PPO/DQN, daha sonra GNN
├── evaluation/   # Metrik, paired seeds, istatistik
└── experiments/  # İnce orchestration katmanı

configs/          # Sürüm kontrollü deney tanımları
data/
├── raw/          # Değiştirilemez; çoğunlukla Git dışında
├── interim/
├── processed/
└── manifests/    # URL, lisans, checksum, şema, köken
tests/
├── unit/
├── integration/
└── scientific/   # invariant, oracle, parity, leakage
artifacts/        # Git dışında model ve sonuçlar
phase_reports/    # Her fazın kanıt raporu
docs/             # Protokol ve metodoloji
```

### 9.1 Tek geçiş çekirdeği

```text
transition(state, task, action, exogenous_event, config)
    -> next_state, outcome, reward_components
```

Gym, event adaptörü, baseline ve oracle aynı saf fonksiyonu çağırmalıdır. Rastgele olay dışarıda üretilip açıkça verilmelidir. Böylece aynı giriş aynı sonucu üretir ve tek formül kaynağı oluşur.

### 9.2 İlk eylem alanı

İlk sürüm `local / belirli edge_i / cloud` olmalıdır. Gizlilik ve kapasite sert kısıtları action mask ile uygulanır. İki genişleme ancak temel sonuçtan sonra yapılmalıdır:

- Bölünebilir görevler için oran ile bölme/birleştirme/aktarım ek yükü.
- CPU/bant genişliği için model tabanlı iç optimize edici veya hibrit eylem.

### 9.3 Bilimsel test kapıları

- Veri hızı mesafeyle azalır; bant genişliği/güçle beklenen yönde değişir.
- Sıfır veri/sıfır cycle sınırları sonlu ve açıklanmış sonuç verir.
- Aynı giriş ve seed birebir aynı trajectory üretir.
- Queue arttıkça bekleme azalmaz; enerji negatif olmaz.
- Gizli görev için cloud eylemi maskelidir.
- Switching maliyeti varsa önceki eylem gözlemdedir.
- Küçük problem elle hesap ve exhaustive oracle ile eşleşir.
- Gym ve event koşucusu aynı olay dizisinde aynı sonuçları verir.
- Normalizasyon test verisine fit olmaz; split’ler zaman/entity açısından kesişmez.

## 10. Deney protokolü

### 10.1 Önceden kaydedilecek hipotezler

- **H1:** İnsan etiketlerinde yeterli doğruluğa ulaşan dondurulmuş LLM çıkarıcı, semantiksiz PPO’ya göre kısıt ihlalini düşürür.
- **H2:** LLM, görülmemiş uygulama/paraphrase holdout’larında kural tabanlı ve küçük denetimli semantic baselines’ı geçer.
- **H3:** Kazanç ödül bonusu veya baskın eylemden değil; eşleştirilmiş fiziksel durumlarda semantik duyarlılıktan gelir.
- **H4:** Sentetik kazanımın yönü iz güdümlü benchmark ve küçük testbed üzerinde korunur.

H1 ters veya anlamsız çıkarsa sonuç saklanmamalıdır.

### 10.2 Baseline merdiveni

1. Always-local, always-edge, always-cloud.
2. Uniform random ve valid-action random.
3. Latency, energy ve deadline-aware greedy.
4. Küçük örneklerde exhaustive/MILP oracle.
5. PPO-MLP, semantiksiz.
6. PPO + kural tabanlı semantik.
7. PPO + küçük sınıflandırıcı semantiği.
8. PPO + LLM semantiği.
9. PPO + insan oracle semantiği.
10. İkincil DQN/A2C; gerekirse recurrent PPO/GNN-PPO.

İnsan oracle ile LLM farkı “semantik faydalı ama çıkarıcı yetersiz mi?” sorusunu ayırır. LLM ile semantiksiz fark uçtan uca faydayı gösterir.

### 10.3 Metrikler

- Başarı/kabul oranı; p50/p95/p99 uçtan uca gecikme; deadline miss ratio.
- Toplam, görev başına ve başarılı görev başına enerji.
- Gizlilik, güvenilirlik, doğruluk ve diğer kısıt ihlalleri.
- Throughput, queue bekleme ve drop.
- Kullanıcı başına başarı/slowdown ve sunucu yüküne dayalı fairness.
- Eylem dağılımı, entropi ve durum dilimi duyarlılığı.
- LLM, parser, state builder, policy inference ve toplam overhead.
- Semantik F1, exact match, kalibrasyon, abstention ve maliyet.

QoE varsa tek, boyutsuz, gerekçeli formül ve sensitivity analizi gerekir.

### 10.4 İstatistik

- En az 5, tercihen 10 bağımsız eğitim seed’i.
- Politikalar aynı değerlendirme episode seed’lerini görür.
- Seed düzeyinde ortalama ve bootstrap %95 güven aralığı.
- Planlı ikili karşılaştırmada paired Wilcoxon, etki büyüklüğü ve Holm düzeltmesi.
- Çok algoritmalı genel test gerekiyorsa Friedman ve uygun post-hoc.
- Binlerce step bağımsız binlerce deney sayılmaz; bağımsızlık birimi açık tanımlanır.
- Eylemlerin %90’dan fazlası tek hedefe yığılıyorsa collapse alarmı; ana deneye geçilmez.

## 11. Faz kapıları

### Faz 0 — Denetim ve protokol

Çıktı bu rapor ve yeni yol haritalarıdır. Eski sonuçlar kanıt diye değil, doğrulanacak hipotez ve regresyon vakası olarak taşınır.

### Faz 1 — Araştırma sözleşmesi ve semantik şema

Problem/MDP, tehdit modeli, JSON Schema, annotasyon kılavuzu ve ön-kayıtlı analiz planı hazırlanır. Her terimin operasyonel tanımı ve her iddianın ölçümü olmadan faz kapanmaz.

### Faz 2 — Deterministik simülatör

Domain tipleri, tek geçiş çekirdeği, Gym ve basit event adaptörü yazılır. Birim, invariant, oracle ve parity testleri kapıdır.

### Faz 3 — Veri kayıt sistemi

Kaynak manifesti, checksum/lisans, alan kökeni, dönüştürücü, train-only normalizer ve split’ler hazırlanır. “Downloaded” statüsü doğrulanmış sayılmaz; leakage testleri geçmelidir.

### Faz 4 — Semantik benchmark

İnsan etiketli corpus, etiketçi anlaşması, rule/ML/LLM baselines ve dondurulmuş cache hazırlanır. Önceden belirlenen şema, F1 ve kalibrasyon eşiğini geçmeyen model RL’ye girmez.

### Faz 5 — Baselines ve oracle

Sabit/sezgisel politikalar ve küçük oracle hazırlanır. Oracle anlamlı eylem çeşitliliği göstermeli; yapısal baskın eylem varsa veri/ödül düzeltilmelidir.

### Faz 6 — PPO temel deneyi

Semantiksiz PPO ve tanımlı semantik varyantlar eğitilir. Reward ayrıştırması, eylem collapse ve paired-seed kontrolleri geçmelidir.

### Faz 7 — İz güdümlü genelleme

Zaman/entity/application holdout, domain shift ve ablation yapılır. Hibrit veri “tam gerçek” diye adlandırılmaz.

### Faz 8 — Gerekçeli ileri mimari

Yalnız ihtiyaç varsa gerçek GNN-PPO, recurrent PPO veya kısmi offloading eklenir. MLP ile bilgi, parametre ve eğitim bütçesi adil olmalıdır.

### Faz 9 — Küçük testbed

Görev-niyet-sistem-sonuç ortak zamanlı ölçülür. Sim-to-real farkı ve başarısızlıklar raporlanır.

### Faz 10 — İstatistik, tez ve sunum

Tablo/figürler dondurulur; her iddia config, veri manifesti, kod sürümü, seed listesi ve raporla izlenir. GUI en son yapılır.

## 12. Yapılmaması gerekenler

- Eski kodu topluca yeni repoya kopyalamak.
- Büyük LLM’yi otomatik “semantik katkı” saymak.
- LLM’ye ağ/doğru eylemi gösterip semantik çıkardığını iddia etmek.
- Ayrı izleri satır sırasıyla birleştirip ortak gerçek gözlem demek.
- Deadline/önceliği hedef eylemden üretmek.
- Tek eyleme çöken politikayı küçük ortalama farkla başarılı göstermek.
- GNN’yi değişken topoloji gereği olmadan eklemek.
- Raw data/checkpoint/sonuçları kaynak kodla aynı Git geçmişine koymak.
- Test ve faz raporu bitmeden kutu işaretlemek.

## 13. Nihai değerlendirme

Eski çalışmanın ana sorunu yetersiz model karmaşıklığı değildir. Semantik doğruluk, fizik doğruluğu ve veri kökenini ayıran deneysel kontrol kurulmamıştır. Daha büyük LLM, daha fazla RL algoritması veya daha karmaşık GNN bu temeli tek başına düzeltmez.

Yeni sıranın **tek ve doğru fizik → izlenebilir veri → insan ground truth’lu semantik → güçlü baseline/oracle → DRL → ileri mimari** olması gerekir. Bu sıra tezi küçültmez; LLM’nin gerçekten işe yarayıp yaramadığını gösterebilen savunulabilir bir çalışma yapar.

## 14. Kaynaklar

1. Cang et al., “Resource Allocation for Semantic-Aware MEC Systems,” 2023. <https://arxiv.org/abs/2309.11736>
2. Ji & Qin, “Computational Offloading in Semantic-Aware Cloud-Edge-End Collaborative Networks,” IEEE JSTSP, 2024. <https://arxiv.org/html/2402.18183v2>
3. Chen et al., “Online Multi-Task Offloading for Semantic-Aware Edge Computing Systems,” 2025. <https://arxiv.org/html/2407.11018v2>
4. Song et al., “Task Offloading with Large Language Models in Mobile Edge Computing,” ICTC 2024. <https://ieeexplore.ieee.org/document/10827049>
5. Ren et al., “Retrieval-Augmented Generation Empowered Decision-Making for LLM-based Mobile Edge Computing,” 2024. <https://arxiv.org/html/2412.20820>
6. Zhu et al., “LLM-QTRAN for Collaborative Task Offloading in UAV-Assisted MEC,” Sensors, 2025. <https://pmc.ncbi.nlm.nih.gov/articles/PMC11723390/>
7. Zheng et al., “AgentVNE: LLM-Driven Graph Reinforcement Learning for Affinity-Aware Virtual Network Embedding,” 2026 preprint. <https://arxiv.org/html/2601.02021>
8. Guo et al., “LeDRL: LLM-Enhanced Deep Reinforcement Learning for Task Offloading,” 2026 preprint. <https://arxiv.org/html/2605.05727v2>
9. Yang et al., “COMLLM,” 2026 preprint. <https://arxiv.org/html/2604.07148>
10. “Dynamic task offloading in vehicular networks using LLM,” Scientific Reports, 2026. <https://pmc.ncbi.nlm.nih.gov/articles/PMC12996603/>
11. Lin et al., “Decentralized Task Offloading in Edge Computing: An Offline-to-Online DRL Approach,” 2024. <https://research.polyu.edu.hk/en/publications/decentralized-task-offloading-in-edge-computing-an-offline-to-onl/>
12. Chen et al., “Asynchronous Deep Progressive Reinforcement Learning for Task Offloading,” 2024. <https://ieeexplore.ieee.org/document/10480253/>
13. University of Glasgow, “MEC IoT Dataset.” <https://researchdata.gla.ac.uk/896/>
14. UCI, “Image Recognition Task Execution Times in Mobile Edge Computing.” <https://archive.ics.uci.edu/dataset/859/image+recognition+task+execution+times+in+mobile+edge+computing>
15. Alibaba, “Cluster Trace v2018.” <https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018>
16. Google, “Cluster Data.” <https://github.com/google/cluster-data>
17. BUPT, “Edge Computing Dataset.” <https://github.com/BuptMecMigration/Edge-Computing-Dataset>
18. Lai et al., “EUA Datasets.” <https://github.com/PhuLai/eua-dataset>
19. Olguín Muñoz et al., “EdgeDroid,” 2019. <https://www.cs.cmu.edu/~satya/docdir/olguin-hotmobile2019.pdf>
20. TUdata, “BigMEC Dataset.” <https://tudatalib.ulb.tu-darmstadt.de/items/f5396674-b717-4214-bb61-407f83df2122>
21. ETSI, “Multi-access Edge Computing.” <https://www.etsi.org/technologies/multi-access-edge-computing>
22. Kolosov et al., “Benchmarking in the Dark: On the Absence of Comprehensive Edge Datasets,” HotEdge 2020. <https://www.usenix.org/conference/hotedge20/presentation/kolosov>

## 15. Faz 0 doğrulama kaydı

- [x] Eski `task.md` ve `TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md` incelendi.
- [x] Sohbet kaydı talimat değil, geçmiş bağlam olarak ele alındı.
- [x] Eski repo, veri üretimi, deney sonuçları ve testler denetlendi.
- [x] Eski test paketi çalıştırıldı: 23/23 geçti; LLM/GUI fallback uyarıları kaydedildi.
- [x] AgentVNE ve yakın dönem birincil literatür karşılaştırıldı.
- [x] Datasetlerin gerçek/sentetik/hibrit sınırları raporlandı.
- [x] Yeni soru, veri stratejisi, semantik değerlendirme, mimari ve faz kapıları tanımlandı.
- [x] Eski depoda hiçbir kod/sonuç değiştirilmedi; otomatik commit atılmadı.

**Faz 0 kararı:** Denetim ve araştırma tamamlandı. Kodlama, Faz 1 araştırma sözleşmesi ve semantik şema kabul edildikten sonra başlamalıdır.

