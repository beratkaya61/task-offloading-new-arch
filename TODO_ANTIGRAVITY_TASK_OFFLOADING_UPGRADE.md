# Task Offloading New Architecture — Bilimsel ve Mimari Sözleşme

Bu belge projenin “kuzey yıldızı”dır. `task.md` ilerlemeyi, bu dosya ise bir işin kabul edilmesi için gereken kaliteyi tanımlar.

## 1. Ana araştırma hedefi

İlk hedef, LLM’nin doğrudan local/edge/cloud kararı vermesi değildir. Hedef; doğal dil görev niyetinden yapılandırılmış gereksinim çıkaran, insan ground truth’una göre doğrulanmış ve dondurulmuş bir LLM katmanının DRL offloading politikasına nedensel katkısını ölçmektir.

Önerilen soru:

> Bağımsız doğrulanmış LLM semantik gereksinimleri, aynı MEC fiziği ve aynı değerlendirme olayları altında, semantiksiz ve LLM’siz semantik baselines’a göre constraint satisfaction ve genellemeyi iyileştiriyor mu?

## 2. Değiştirilemez ilkeler

### 2.1 Tek fizik kaynağı

- Kanal, gecikme, enerji, queue ve başarısızlık hesabı tek geçiş çekirdeğinde yaşar.
- Gym, event simulation, baseline ve oracle aynı çekirdeği çağırır.
- Birimler tiplerde ve dokümantasyonda görünürdür.
- Tuple ile belirsiz fizik değeri taşınmaz; isimli veri tipleri kullanılır.

### 2.2 Semantik izolasyonu

- LLM yalnız görev metni/kullanıcı politikasını görür.
- Ağ durumu, doğru eylem, ödül ve oracle sonucu LLM girdisine verilmez.
- Çıktı eylem değil, gereksinim JSON’udur.
- Bilinmeyen alan uydurulmaz; `unknown/null` ve confidence kullanılır.
- LLM çıktıları RL’den önce cache’lenir ve sürümlenir.
- Ödülde “LLM önerisine uyum” bonusu bulunmaz.

### 2.3 Veri dürüstlüğü

- Her dataset için kaynak, sürüm, lisans ve checksum manifestte bulunur.
- `downloaded`, `validated` anlamına gelmez.
- Her kolon observed/measured/derived/matched/generated/labeled olarak işaretlenir.
- İlişkisiz kaynakların satırları doğal ortak gözlem gibi sunulmaz.
- Normalizer yalnız training split’ine fit edilir.
- Zaman ve entity sızıntısı otomatik test edilir.
- Hibrit benchmark açıkça hibrit diye adlandırılır.

### 2.4 Deneysel adalet

- Karşılaştırmalar aynı görevler, exogenous events ve episode seed’lerini kullanır.
- Baseline’ların bilgi erişimi eşittir; oracle ayrı etiketlenir.
- En az 5, tercihen 10 bağımsız eğitim seed’i kullanılır.
- Güven aralığı, eşleştirilmiş test ve etki büyüklüğü raporlanır.
- Tek eylem oranı, eylem entropisi ve durum duyarlılığı zorunlu metriktir.
- Ana baseline sezgisel/oracle’dan kötü ise neden çözülmeden karmaşıklık eklenmez.

### 2.5 Aşamalı karmaşıklık

- İlk eylem alanı local/edge_i/cloud’dur.
- Partial offloading yalnız görev bölünebilirliği ve ek yük modeli doğrulanınca eklenir.
- Resource allocation gerçekten eylem alanına girmeden başlıkta iddia edilmez.
- GNN yalnız değişken topoloji/ilişki yapısı gerektiriyorsa kullanılır.
- GNN-PPO iddiası için grafik politika PPO ile uçtan uca eğitilir.
- GUI ve demo bilimsel çekirdeği değiştirmez, son fazda yapılır.

## 3. Zorunlu baseline ve ablation ailesi

### Fiziksel karar baselines

- Always-local, always-edge, always-cloud.
- Uniform random ve valid-action random.
- Latency-aware, energy-aware, deadline-aware greedy.
- Küçük exhaustive/MILP oracle.
- PPO-MLP ana öğrenen politika; DQN/A2C ikincil.

### Semantik baselines

- Semantiksiz.
- Sabit profil.
- Rule/keyword parser.
- TF-IDF + lojistik regresyon veya benzer küçük ML.
- Küçük encoder.
- Zero-shot ve few-shot LLM.
- İnsan oracle etiketi.

### Ana ablation’lar

- Semantik özellikleri çıkar.
- Action mask’i çıkar.
- LLM yerine rule/ML/human oracle koy.
- Confidence/abstention kullanımını çıkar.
- Ödül bileşenlerini ayrı çıkar.
- Veri kaynak katmanlarını ayrı çıkar.
- Ön-eğitim varsa scratch karşılaştır.
- İleri mimaride GNN/recurrent/partial bileşenini ayrı çıkar.

## 4. Zorunlu metrikler

### Sistem

- Success/acceptance, throughput, drop.
- End-to-end latency p50/p95/p99 ve deadline miss ratio.
- Total/per-task/per-success energy.
- Queue waiting.
- Privacy/reliability/accuracy ve toplam constraint violations.
- User success/slowdown ve server load temelli fairness.

### Politika

- Action histogram, entropy, collapse ratio.
- Durum dilimleri ve karşı-olgusal semantik çiftlerde eylem duyarlılığı.
- Reward component decomposition.
- Training stability ve generalization gap.

### Semantik katman

- Schema validity ve exact match.
- Macro/micro F1 ve alan bazlı precision/recall.
- Calibration: Brier/ECE.
- Abstention/unknown doğruluğu.
- Model, prompt, parser ve toplam latency/token/cost.

## 5. Test matrisi

- Birim ve boyut/birim testleri.
- Fiziksel monotonicity/property testleri.
- Determinizm ve seed testi.
- Elle hesap/exhaustive oracle testi.
- Gym-event parity testi.
- Action-mask ve hard constraint testi.
- Split/duplicate/leakage testi.
- Semantic JSON ve parser robustness testi.
- LLM cache hash/reproducibility testi.
- Metric definition golden testleri.
- End-to-end küçük deney smoke testi.

## 6. Faz kabul kuralları

Bir faz yalnız şu koşullarda tamamlanır:

1. `task.md` içindeki çıktılar üretilmiştir.
2. O fazın testleri geçmektedir.
3. `phase_reports/` altında girişler, yöntem, sonuç, başarısızlık ve sınırlamalar yazılmıştır.
4. Veri/config/kod/seed bağı izlenebilir durumdadır.
5. Sonraki fazın riskleri ve açık işleri yazılmıştır.

Kutunun işaretlenmesi akademik iddia değildir; rapordaki kanıt akademik iddiadır.

## 7. Definition of Done

- [ ] Tez sorusu ve hipotezleri dondurulmuş.
- [ ] Semantik ground truth iki etiketçi ve anlaşma analiziyle mevcut.
- [ ] LLM semantik çıkarıcı bağımsız benchmark’ta ölçülmüş ve cache’lenmiş.
- [ ] Tek fizik çekirdeği oracle/parity/invariant testlerini geçiyor.
- [ ] Veri manifesti, lisans, checksum, alan kökeni ve sızıntısız split var.
- [ ] Güçlü sabit/sezgisel/oracle baselines tamam.
- [ ] Semantiksiz/rule/ML/LLM/human-oracle politika karşılaştırmaları tamam.
- [ ] En az 5 seed, %95 CI, planlı test ve effect size mevcut.
- [ ] p95/p99, DMR, enerji, constraint, fairness, overhead ve collapse raporlu.
- [ ] Sentetik, hibrit trace ve küçük testbed sonuçlarının sınırları ayrı yazılmış.
- [ ] GNN/partial/resource allocation iddiaları yalnız gerçekten uygulanmışsa başlıkta.
- [ ] Bütün deneyler tek komut/config ve sürümlü manifestlerle yeniden üretilebilir.
- [ ] Negative results ve threats to validity tezde açıkça bulunuyor.

## 8. Faz 0 kararı

Eski repo doğrudan taşınmayacaktır. Fizik hataları regresyon testine, eski veri üretim sorunları leakage/provenance testine, çökmüş politikalar collapse testine dönüştürülecektir. Yeni kod başlamadan önce Faz 1 araştırma sözleşmesi tamamlanmalıdır.

