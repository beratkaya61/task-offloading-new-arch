# Faz 1 Araştırma Sözleşmesi

**Sürüm:** 1.0.0

**Durum:** Faz 2 için donduruldu

**Makinece okunabilir karşılık:** `configs/research_contract.json`

Bu belge tezin hangi soruyu cevaplayacağını, simülatörün sınırını, LLM ile DRL'nin görev ayrımını ve hangi sonuçların bilimsel iddia sayılacağını önceden belirler. Ana deneyler görüldükten sonra hedefi değiştirmeyi engellemek için yapılacak her değişiklik gerekçeli bir protokol ekiyle kaydedilecektir.

## 1. Dondurulan başlık ve araştırma sorusu

**Türkçe başlık**

> İz Güdümlü MEC Sistemlerinde İnsan Doğrulamalı LLM Semantik Gereksinimleriyle Derin Pekiştirmeli Öğrenme Tabanlı Görev Aktarımı

**İngilizce başlık**

> Deep Reinforcement Learning-Based Task Offloading with Human-Validated LLM Semantic Requirements in Trace-Driven MEC Systems

**Tek ana araştırma sorusu**

> Ağ durumunu ve doğru eylemi görmeyen, bağımsız insan ground truth'una göre doğrulanıp dondurulan bir LLM semantik gereksinim çıkarıcısı; aynı MEC fiziği, görevler ve dışsal olaylar altında semantiksiz ve LLM'siz semantik taban çizgilerine göre kısıt ihlallerini azaltıp görülmemiş koşullara genellemeyi iyileştiriyor mu?

Bu soruda LLM'nin görevi bir offloading eylemi seçmek değildir. LLM yalnızca görev metnindeki açık gereksinimleri yapılandırılmış JSON'a dönüştürür. `local / edge_i / cloud` kararını DRL politikası verir.

## 2. Önceden kaydedilen hipotezler

- **H1 — Uçtan uca fayda:** LLM semantiği kullanan PPO, eşleştirilmiş olaylar altında semantiksiz PPO'ya göre toplam kısıt ihlali oranını düşürür.
- **H2 — Semantik genelleme:** Dondurulmuş LLM çıkarıcısı, görülmemiş senaryo ve paraphrase ailelerinde rule parser ve küçük denetimli semantik modellerden daha yüksek macro-F1 elde eder.
- **H3 — Nedensel kontrol:** Politika kazanımı LLM'ye uyum ödülünden kaynaklanmaz; aynı fiziksel görevde yalnızca semantik gereksinim değiştirildiğinde anlamlı eylem duyarlılığı görülür ve politika tek eyleme çökmez.
- **H4 — İz tutarlılığı:** Ana etkinin yönü doğrulanmış sentetik ve trace-driven-hybrid değerlendirmelerde tutarlıdır.

H1'in desteklenmemesi başarısız deney olarak gizlenmeyecektir. “Semantik doğru çıkarılsa bile offloading politikasını iyileştirmedi” sonucu da geçerli bir bilimsel bulgudur.

## 3. İzin verilen ve verilmeyen iddialar

### İzin verilen

- Doğrulanmış sentetik MEC simülasyonu.
- Gerçek izlerden beslenen trace-driven-hybrid değerlendirme.
- İki gerçek insan tarafından bağımsız etiketlenip uzlaştırılmış semantik benchmark.
- Dondurulmuş LLM semantik çıkarıcısının DRL kararlarına ölçülen katkısı.
- Kaynak katmanı, model ve reward ablation sonuçları.

### Kanıt olmadan izin verilmeyen

- “Tamamen gerçek MEC verisi” veya “gerçek MEC deployment doğrulaması”.
- LLM'nin doğrudan optimal offloading kararı verdiği iddiası.
- Partial offloading, resource allocation veya GNN-PPO'nun uygulanmadan başlıkta kullanılması.
- İlişkisiz iki veri kaynağının aynı gerçek olaymış gibi birleştirilmesi.
- Testbed olmadan sim-to-real başarısı iddiası.

## 4. Başlangıç sistem sınırı

İlk bilimsel çekirdek bilinçli olarak küçük ve denetlenebilir tutulur:

| Bileşen | Faz 2–7 ana kapsamı |
|---|---:|
| IoT/mobil cihaz | 20 |
| Edge sunucu | 3 |
| Cloud bölgesi | 1 |
| Karar | Her görev gelişinde bir kez |
| Episode uzunluğu | 512 görev geliş olayı |
| Politika | Merkezi, ayrık eylemli |
| Offloading | Görev bütünüyle tek hedefe gider |
| Topoloji | İlk deneyde sabit edge sayısı |

İlk eylem uzayı:

```text
local, edge_0, edge_1, edge_2, cloud
```

İlk kapsamda şunlar bulunmaz:

- görevin yüzdeyle parçalanması,
- CPU frekansı veya bant genişliğinin doğrudan RL eylemi olması,
- değişken sayıda edge düğümü,
- GNN veya çok ajanlı politika.

Bunlar ancak temel model güçlü baselineları geçer ve ihtiyaç deneyle gösterilirse Faz 8'de eklenebilir.

## 5. Zaman ve karar modeli

Model, **olay indeksli sonlu ufuklu MDP** olarak tanımlanır. Her yeni görev geldiğinde ajan bir eylem seçer. İki görev gelişi arasında tamamlanan upload, hesaplama ve download olayları kuyrukları günceller.

Bir geçişin kavramsal biçimi:

```text
transition(state_t, task_t, action_t, exogenous_event_t, config)
  -> next_state, outcome, reward_components
```

Rastgelelik geçiş fonksiyonunun içine gizlenmez. Geliş, kanal bozunması, arıza ve trace örneği gibi dışsal olaylar önceden üretilir. Aynı başlangıç durumu, eylem, dışsal olay ve config birebir aynı sonucu vermelidir.

### 5.1 Durum

DRL gözlemi aşağıdaki gruplardan oluşur:

1. **Görev fiziği:** input/output bitleri, gereken CPU cycle, varsa doğruluk–hesap profili.
2. **Dondurulmuş semantik:** latency, privacy, execution policy, reliability, accuracy ve enerji önceliği.
3. **Belirsizlik:** alan bazlı `unknown/ambiguous` maskeleri ve confidence değerleri.
4. **Cihaz:** işlem kapasitesi, pil/enerji durumu, mevcut bağlantı ve gerekliyse hareket bilgisi.
5. **Bağlantılar:** aday edge'lere veri hızı, SNR, mesafe ve erişilebilirlik; isimli ve birimli tiplerle.
6. **Edge:** kapasite, kalan iş/kuyruk, güncel yük ve arıza durumu.
7. **Cloud:** backhaul gecikmesi, kapasite/yük ve kullanılabilirlik.
8. **Zaman:** bir sonraki olaya kadar süre ve izdeki zaman bağlamı.
9. **Önceki eylem:** yalnızca switching maliyeti etkinse.

LLM bu durumun yalnızca görev metni ve kullanıcı-politika metni bölümünü görür. Ağ, kuyruk, sunucu yükü ve oracle kararı LLM'ye verilmez.

### 5.2 Eylem maskesi

Bir eylem yalnızca şu durumlarda sert biçimde maskelenebilir:

- metinde açıkça belirtilmiş execution-policy yasağı,
- hedefin arızalı/kapalı olması,
- fiziksel olarak mümkün olmayan admission durumu.

`unknown` semantik alanı bir eylemi yasaklamaz. Deadline tek başına eylem maskesi oluşturmaz; sonucu fizik ve kuyruk belirler.

### 5.3 Geçiş ve sonuç

Geçiş çekirdeği aşağıdakileri tek yerde hesaplar:

- upload/download süresi,
- local/edge/cloud işlem süresi,
- kuyruk bekleme süresi,
- uçtan uca gecikme,
- cihaz enerjisi,
- deadline ve semantik kısıt ihlalleri,
- kabul, drop ve başarısızlık,
- bir sonraki kuyruk ve cihaz durumu.

Gym ortamı, event runner, baseline ve oracle aynı çekirdeği çağıracaktır.

### 5.4 Ödül

Birincil scalar reward, normalize edilmiş maliyetin negatifidir:

```text
r = -(latency_norm
      + energy_norm
      + 5  * deadline_miss
      + 5  * soft_constraint_violation
      + 10 * failure_or_drop
      + 20 * hard_constraint_violation)
```

- Latency ve enerji referansları yalnızca training split'inden hesaplanır.
- Queue bekleme uçtan uca gecikmenin içindedir; ikinci kez cezalandırılmaz.
- LLM önerisine veya insan etiketine “uydu” diye ödül verilmez.
- Ana reward için 0.5× ve 2× ağırlık duyarlılık deneyleri yapılır.
- Primary tez metriği reward değil, **toplam kısıt ihlali oranıdır**.

## 6. Sembol ve birim sözleşmesi

| Sembol | Anlam | SI birimi |
|---|---|---|
| `B_in`, `B_out` | Girdi/çıktı veri büyüklüğü | bit |
| `C` | Görevin hesap ihtiyacı | cycle |
| `f_d`, `f_e`, `f_c` | Cihaz/edge/cloud işlem hızı | cycle/s |
| `R_ul`, `R_dl` | Uplink/downlink veri hızı | bit/s |
| `d` | Cihaz–edge mesafesi | m |
| `SNR` | Sinyal-gürültü oranı | dB |
| `P_tx`, `P_rx` | Gönderme/alma gücü | W |
| `T_queue` | Kuyruk beklemesi | s |
| `T_e2e` | Uçtan uca gecikme | s |
| `D` | Deadline | s |
| `E` | Cihaz tarafından harcanan enerji | J |
| `u` | Kaynak kullanımı | boyutsuz `[0,1]` |
| `p_success` | Başarı olasılığı | boyutsuz `[0,1]` |
| `Δt` | İki olay arasındaki süre | s |
| `H` | Episode karar sayısı | adet; ana değer 512 |

Dosyada MB/GB, MHz/GHz veya ms gibi kolay okunan değerler bulunabilir; domain katmanına girerken SI birimine açıkça dönüştürülecektir.

## 7. Semantik katman sözleşmesi

LLM çıktısı `schemas/semantic_requirements.schema.json` ile doğrulanır. Çıktı şu bilgi gruplarını kapsar:

- domain,
- latency sınıfı ve varsa açık deadline,
- privacy veri sınıfı,
- device/edge/cloud çalıştırma izinleri,
- reliability,
- accuracy,
- enerji önceliği,
- görev bölünebilirliği,
- kanıt metin parçaları,
- alan bazlı ve genel confidence,
- abstention alanları.

Metinde belirtilmeyen alan `null/not_stated` kalır. Çelişkili metin `ambiguous` olur. Model, literatürden veya sağduyudan sayısal deadline uyduramaz.

LLM çıktıları RL eğitimi başlamadan önce üretilir, hash'lenir ve cache'lenir. Canlı API/model çağrıları RL episode'larının içine konmaz.

## 8. İnsan ground truth

- 20 benzersiz pilot ve 240 benzersiz ana görev kullanılacaktır.
- İki gerçek insan ana görevlerin tamamını bağımsız ve kör biçimde etiketleyecektir.
- Ekranda LLM/model önerisi gösterilmeyecektir.
- Yaklaşık yüzde 5 gizli tekrar, annotator içi tutarlılığı ölçmek için sunulacaktır.
- Ham cevaplar kilitlendikten sonra anlaşmazlıklar uzlaştırılacaktır.
- Ham A/B cevapları değiştirilmeyecek, nihai adjudicated gold ayrı tutulacaktır.
- Alan bazında minimum Cohen's kappa 0.70, hedef 0.80'dir.
- Düşük uyum gizlenmez; yönerge revizyonu ve ilgili alanın yeniden etiketlenmesi raporlanır.

Ayrıntılı süreç `docs/ANNOTATION_PROTOCOL.md` içindedir.

## 9. Trace-driven-hybrid veri stratejisi

Hiçbir kaynak bütün gözlemleri birlikte sağlamaz. Bu nedenle her alanın kökeni şu etiketlerden biriyle tutulur:

```text
observed, measured, derived, matched,
generated, human_labeled, simulated
```

### Kaynak planı ve güncel statü

| Kaynak | Rol | 2026-09-18 statüsü |
|---|---|---|
| BUPT Edge Computing Dataset | Mobil oturum gelişleri, byte, RAT, hücre geçişi, hizmet metadata | `schema_validated`; research-only, raw paylaşılmaz |
| EdgeWorkloadsTraces NEP-large/full | Edge VM CPU/bant genişliği, site RTT ve kapasite | `checksum_verified`; full satır profili bekleniyor |
| UCI MEC Execution Times 859 | Dört cihazda ölçülmüş image-recognition turnaround | `schema_validated`; CC BY 4.0 |
| EUA | Kullanıcı ve edge coğrafi konum senaryosu | `schema_validated`; commit-sabit, MIT |
| T-Drive veya DiDi GAIA | Sürekli mobilite stres deneyi | Bir kaynak erişim/lisans denetiminden sonra seçilecek |
| Alibaba v2018 | Cloud/cluster yükü ve OOD stres deneyi | İkincil kaynak |

Kaynaklar satır numarasına göre birleştirilmeyecektir. Örneğin BUPT görevi ile NEP yük penceresi eşleştirilirse alan `matched` olarak işaretlenecek, “aynı gerçek olay” denmeyecektir.

Trace-driven ana değerlendirmede ayrıca kaynak katmanı ablation'ı yapılacaktır:

1. yalnız sentetik fizik/geliş,
2. gerçek görev-geliş/traffic replay,
3. gerçek edge-load replay,
4. ikisinin açıkça hibrit eşleştirilmesi.

Bu ayrım, sonucun hangi gerçek veri katmanına bağlı olduğunu gösterecektir.

## 10. Model seçimi

İlk açık model adayları:

- `Qwen/Qwen3-4B`,
- `Qwen/Qwen3-8B`.

Donanım modeli belirlemez; donanım yalnızca quantization ve çalışma ortamını belirler. Faz 4'te validation seti üzerinde şema geçerliliği, macro-F1, kritik privacy/latency recall, gecikme ve maliyet birlikte ölçülür. Bütün kalite kapılarını geçen en küçük tekrarlanabilir model seçilir.

Minimum kapılar:

| Ölçüm | Eşik |
|---|---:|
| JSON şema geçerliliği | `>= 0.99` |
| Macro-F1 | `>= 0.80` |
| Kritik privacy recall | `>= 0.95` |
| Kritik latency recall | `>= 0.90` |

Model kararı test setine bakılarak verilmez. Seçilen modelin revision, tokenizer, quantization, prompt, parser, decoding ayarları, ham cache ve checksum'ları dondurulur.

## 11. Baseline ve bilgi adaleti

Fiziksel karar baselineları:

- always-local, always-edge, always-cloud,
- uniform random, valid-action random,
- latency-aware, energy-aware, deadline-aware greedy,
- küçük exhaustive/MILP oracle,
- semantiksiz PPO-MLP.

Semantik baselineları:

- semantiksiz,
- sabit profil,
- rule/keyword parser,
- TF-IDF + logistic regression,
- küçük encoder,
- zero-shot ve few-shot LLM,
- insan-oracle etiketi.

Oracle dışında aynı karşılaştırmadaki politikalar aynı gözlenebilir fizik bilgisine erişir. LLM yalnızca metne erişir. İnsan-oracle, semantik bilginin teorik faydasını LLM çıkarım hatasından ayırır.

## 12. Ana ölçümler ve istatistik

**Primary sistem metriği:** toplam kısıt ihlali oranı.

İkincil sistem metrikleri:

- success/drop/throughput,
- deadline miss ratio,
- latency p50/p95/p99,
- görev ve başarılı görev başına enerji,
- queue waiting,
- privacy/reliability/accuracy ihlalleri,
- kullanıcı başarı/slowdown ve sunucu yükü fairness'i.

Politika kontrolleri:

- action histogram ve entropy,
- tek eylem oranı,
- reward decomposition,
- karşı-olgusal semantik çiftlerde eylem duyarlılığı,
- training stability ve generalization gap.

İstatistik protokolü:

- minimum 5, hedef 10 bağımsız eğitim seed'i,
- bütün politikalar için aynı evaluation episode seed'leri,
- seed düzeyinde bootstrap yüzde 95 güven aralığı,
- planlı ana ikili karşılaştırmada paired Wilcoxon,
- etki büyüklüğü,
- çoklu karşılaştırmada Holm düzeltmesi,
- eylemlerin yüzde 90'dan fazlası tek hedefteyse collapse alarmı.

Binlerce görev adımı binlerce bağımsız deney sayılmayacaktır; bağımsızlık birimi eğitim seed'i/evaluation episode'u olarak açıkça raporlanacaktır.

## 13. Testbed kararı

Kullanıcının fiziksel MEC/IoT test ortamı bulunmadığı için küçük testbed çekirdek tez iddiasının kabul koşulu değildir. Faz 9 isteğe bağlı uzantıdır.

Testbed olmadan kullanılacak doğru ifade:

> validated synthetic and trace-driven-hybrid simulation

Kullanılmayacak ifade:

> real MEC deployment validation

İleride donanım, üniversite laboratuvarı veya kiralık makineler erişilebilir olursa küçük emulation/testbed sonucu ayrı ve sınırlı bir dış doğrulama olarak eklenebilir.

## 14. Değişiklik kontrolü

Bu sözleşme Faz 2 başlamadan önce dondurulmuştur. Sonraki değişiklikler:

1. tarih ve gerekçe,
2. etkilenen hipotez/metrik,
3. değişikliğin test sonuçları görülmeden mi sonra mı yapıldığı,
4. yeniden çalıştırılması gereken deneyler

ile birlikte ayrı amendment kaydına yazılır. Test seti sonuçları yöntem veya eşik seçmek için kullanılmaz.
