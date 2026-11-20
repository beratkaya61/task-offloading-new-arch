# Exploratory Improvement Protocol v1

**Kayıt tarihi:** 2026-11-20

**Statü:** v3.1 semantic ve PPO-v1 sonuçları görüldükten sonra tanımlanmış
exploratory iyileştirme turu. `docs/PREREGISTRATION_V2.md` içindeki confirmatory
hipotezleri veya karar kurallarını değiştirmez.

## 1. Amaç ve değişmez kurallar

Bu turun amacı, beklenen üstünlüğü göstermeyen dört bileşenin nedenini development
verisi üzerinde teşhis etmek ve önceden sınırlandırılmış aday iyileştirmeleri
karşılaştırmaktır:

1. E5 fine-tuned extractor,
2. Qwen few-shot extractor,
3. confidence tabanlı conservative fallback,
4. PPO'nun greedy baseline karşısındaki davranışı.

Değerlendirme etiketleri model, prompt, threshold veya hyperparameter seçmek için
kullanılmaz. Mevcut 60 görevlik insan referansı ile compositional/independent test
havuzları yalnız son değerlendirmede kullanılır. Makine üretimi veya aktarılmış
etiketler insan etiketi olarak raporlanmaz. Bütün eski sonuçlar korunur; yeni
sonuçlar `exploratory` olarak ayrı gösterilir.

## 2. İnsan annotation bütçesi için aşamalı kapılar

Tam annotation paketine doğrudan geçilmez.

### Kapı A — referans kalitesi pilotu

- Mevcut 60 insan-referans görevinden model sonuçlarına bakılmadan, sabit tohumla
  seçilen 20 görev ikinci bağımsız annotator tarafından etiketlenir.
- Annotator model tahminlerini, metrikleri ve ilk annotator etiketini görmez.
- Alan bazlı agreement, execution-policy Jaccard, evidence completeness, görev
  başına süre ve protokol kaynaklı uyuşmazlıklar raporlanır.
- Pilot, referans kalitesini artırır; model iyileştirmesi olarak yorumlanmaz.
- Agreement/protokol sorunu çözülmeden kalan 40 görev ikinci tura açılmaz.

### Kapı B — insan etiketli development pilotu

- Test ailelerinden ayrık, diversity/uncertainty sampling ile seçilmiş 30 yeni
  development görevi insan tarafından etiketlenir.
- Öğrenme eğrisi 0/10/20/30 yeni etiketle, yalnız development içi grouped
  validation üzerinde ölçülür.
- Genişlemeye geçmek için aşağıdakilerden en az biri gerekir:
  - macro-F1 point estimate artışı en az 2 puan, veya
  - privacy/execution-policy kritik F1 artışı en az 5 puan,
  - ve diğer birincil alanlarda 2 puandan büyük macro-F1 kaybı olmaması.
- Ölçüt karşılanmazsa yüzlerce ek annotation yapılmaz; pilot negatif sonuç olarak
  raporlanır.

### Kapı C — genişletme ve yeni kilitli test

- Kapı B geçilirse insan-etiketli development havuzu 80–120 göreve çıkarılır.
- Nihai model seçildikten sonra, ayrı scenario family'lerden yeni bir çift-
  annotator test seti kilitlenir ve yalnız bir kez çalıştırılır.

## 3. E5 fine-tuning iyileştirmesi

Bilinen sonuç: augmented E5 fine-tuned macro-F1; human60-original üzerinde
%70,2, human60-paraphrase üzerinde %62,3 ve compositional unseen üzerinde
%73,0'dır. Özellikle domain genellemesi düşüktür; bütün backbone'un tek learning
rate ile sentetik-ağırlıklı veri üzerinde güncellenmesi catastrophic forgetting
riski taşır.

Development üzerinde karşılaştırılacak sınırlı adaylar:

1. mevcut v3.1 ayarı (kontrol),
2. head warm-up + yalnız üst encoder katmanlarını gradual unfreezing,
3. düşük backbone learning rate ve daha yüksek head learning rate,
4. original/insan-development örneklerine kaynak dengeli sampling,
5. grouped validation macro-F1 ile early stopping.

Seçim ölçütü önce overall macro-F1, eşitlikte privacy ve execution-policy F1'dır.
Human60 sonucu seçimde kullanılmaz.

## 4. Qwen iyileştirmesi

Bilinen sonuç: Qwen2.5-1.5B-Instruct human60-original macro-F1 %48,3;
human60-paraphrase %43,7. Paraphrase üzerinde kuraldan +3,4 puan fark anlamlı
değildir. Mevcut prompt küçük modelden uzun signature listelerini birebir JSON
olarak kopyalamasını ister.

Development üzerinde karşılaştırılacak adaylar:

1. mevcut 4-shot signature çıktısı (kontrol),
2. her alanın izinli signature'larını kısa sınıf kodlarına çeviren 4-shot prompt,
3. aynı kodlanmış prompt ile coverage-balanced 8-shot örnek seçimi.

Decoding greedy kalır. Ayrı modeller, test etiketleriyle prompt yazımı ve
self-consistency kullanılmaz. Birincil ölçüt macro-F1; ikincil ölçüt invalid field
oranı ve görev başına latency'dir.

## 5. Confidence calibration ve fallback

Önce v3.2'de önceden tanımlanmış per-field isotonic calibration çalıştırılır.
Calibration etiket tahminini değiştirmez; yalnız confidence, ECE/Brier ve fallback
davranışını değiştirir. E5 fallback için yeni threshold, evaluation setleri
üzerinden seçilmez. Gerekirse threshold yalnız ayrı calibration setinde;
violation riskini azaltırken validation başarısını en fazla 2 puan düşüren değer
olarak seçilir.

## 6. PPO ve greedy

Hatasız full-action preview koşulunda greedy çok güçlü bir baseline'dır; PPO'nun
onu geçmesi zorlanmayacaktır. Önce pre-registration v2'deki sigma=0,5 noisy-preview
koşulu aynı bilgiyle PPO ve greedy için çalıştırılır. PPO hyperparameter araması
gerekirse yalnız training/validation scenario'larında yapılır; test episode seed'i
kullanılmaz. Raporlama üç parçalıdır:

- exact-preview: greedy üst sınırına yakın baseline,
- noisy-preview: H5'in adil karşılaştırması,
- NEP: gerçek iz katmanı altında yön tutarlılığı.

## 7. Raporlama

Her aday için development seçimi, seçilmeyen adaylar ve compute bütçesi saklanır.
Son tabloda eski ve yeni yöntemler yan yana verilir; yalnız iyileşen koşullar seçilip
raporlanmaz. Confirmatory v2 ile exploratory improvement sonuçları açıkça ayrılır.
