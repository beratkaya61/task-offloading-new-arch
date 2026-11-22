# Faz 5 Kapanış Raporu — Baselines, Oracle ve Reward Denetimi

**Kapanış tarihi:** 21 Kasım 2026

**Kapsam:** NEP-v2 trace-driven-hybrid benchmark üzerinde fiziksel karar
baselines, küçük exhaustive oracle/parity kanıtı ve reward decomposition

**Karar:** Faz 5'in deneysel çıktıları ve repository-wide kalite kapısı
tamamlandı.

## 1. Amaç ve kabul ölçütü

Bu fazın amacı PPO'yu doğrudan öne çıkarmak değil, öğrenen politikanın
karşılaştırılacağı güçlü ve adil alt/üst sınırları oluşturmaktır. Bütün yöntemler
aynı `transition` çekirdeğini, aynı 60 episode × 64 task değerlendirme olaylarını
ve aynı exogenous seed'leri kullanır. Böylece yöntemler arasındaki fark fizik
motoru veya test olayından değil, yalnız karar kuralından kaynaklanır.

Kabul ölçütleri şunlardır:

1. Always-local/edge/cloud, random-valid ve greedy aileleri aynı çekirdekte
   ölçülmelidir.
2. Küçük exhaustive oracle, Gym–event parity ve action-mask kuralları otomatik
   testlerle korunmalıdır.
3. Success yanında latency tail, queue, throughput, energy, constraint ve policy
   collapse ölçülmelidir.
4. Reward bileşenleri ayrı gösterilmeli ve ağırlık duyarlılığı incelenmelidir.

## 2. Deney sözleşmesi

| Öğe | Dondurulan değer |
|---|---|
| Benchmark | `trace-driven-hybrid`, NEP full-trace aggregate quantile layer |
| Değerlendirme | 60 episode × 64 task = 3.840 karar/koşul |
| Fizik | Tek deterministik `transition` çekirdeği |
| Bilgi rejimi | Exact-preview; ek olarak Phase 6'da σ=0,5 noisy-preview |
| PPO eğitimi | 5 bağımsız seed × 150.000 step |
| Belirsizlik | Episode-paired bootstrap, 2.000 resample, %95 percentile CI |
| Çoklu test | Aynı metric family içinde Holm step-down düzeltmesi |

NEP katmanı ham kullanıcı satırı veya ham ağ trace'i dağıtmaz. CPU ve RTT
dağılımları, tam arşiv taramasından elde edilen sürümlü aggregate quantile
profili üzerinden replay edilir. Bu nedenle sonuç “gerçek testbed” değil,
trace-informed simulation sonucudur.

## 3. Baseline ailesi ve kullanım amacı

- **Always-local / always-edge / always-cloud:** sistem uçlarını ve fizik
  modelinin beklenen davranışını görünür kılan sabit politikalar.
- **Random-valid:** action mask'e uyan fakat state bilgisini kullanmayan alt sınır.
- **Greedy-no-semantic:** yalnız fiziksel preview ile karar veren güçlü
  öğrenmesiz baseline.
- **Greedy-semantic:** aynı fiziksel kuralı structured semantic requirements ile
  çalıştıran nedensel kontrol.
- **Exhaustive oracle:** küçük test durumlarında bütün valid action'ları tarayan
  doğrulama aracı; ana sonuç tablosunda erişilebilir politika gibi sunulmaz.

## 4. NEP-v2 exact-preview baseline sonuçları

| Politika | Success ↑ | Hard violation ↓ | Deadline miss ↓ | Drop ↓ |
|---|---:|---:|---:|---:|
| Always-local | %57,5 | %0,00 | %21,6 | %0,00 |
| Always-edge-0 | %76,1 | %0,00 | %9,4 | %0,55 |
| Always-cloud | %75,5 | %17,71 | %5,7 | %0,03 |
| Random-valid | %70,9 | %3,62 | %8,8 | %0,34 |
| Greedy no-semantic | %78,8 | %7,45 | %3,7 | %0,31 |
| Greedy single-human reference | **%91,7** | **%0,00** | %3,6 | %0,18 |
| PPO no-semantic | %76,2 | %0,01 | %9,4 | %0,55 |
| PPO single-human reference | %91,2 | **%0,00** | %3,6 | %0,12 |

Sabit politikaların hiçbiri bütün amaçları birlikte karşılamaz. Always-local
hard constraint ihlalini sıfırlarken queue kaynaklı latency ve deadline
maliyetini büyütür; always-cloud latency'yi düşürürken privacy/execution hard
constraint ihlali üretir. Bu çapraz davranış, tek bir latency veya energy
metriğinin yeterli olmadığını doğrular.

Greedy single-human reference, %91,7 success ve %0 hard violation ile bu deney
rejimindeki en güçlü erişilebilir baseline'dır. PPO-reference %91,2 success ile
ona yakın fakat üstün değildir. Bu nedenle ilerideki mimari katkı “PPO greedy'yi
geçti” şeklinde kurulamaz.

## 5. Resource, tail ve policy diagnostics

| Koşul | p50 latency | p95 latency | p99 latency | Queue wait | Successful task/s | Energy/success | Collapse |
|---|---:|---:|---:|---:|---:|---:|---:|
| Greedy no-semantic | 66,7 ms | 124,9 ms | 154,3 ms | 25,6 ms | 25,68 | 7,71 mJ | %61,2 |
| Greedy reference | 67,5 ms | 133,6 ms | 195,0 ms | 26,0 ms | 29,72 | 7,39 mJ | %58,6 |
| PPO no-semantic | 241,0 ms | 554,1 ms | 688,7 ms | 230,7 ms | 21,63 | 23,63 mJ | **%99,7** |
| PPO reference | 76,8 ms | 192,6 ms | 304,7 ms | 37,8 ms | 28,96 | 8,84 mJ | %72,3 |

Semantiksiz PPO'nun yaklaşık %99,7 max-action share değeri gerçek bir policy
collapse bulgusudur. Ortalama success tek başına verilse bu hata görünmezdi.
Semantic state, PPO collapse'ını önemli ölçüde azaltmış; ancak greedy-reference
hem daha dengeli action use hem de daha düşük decision overhead göstermiştir.

Jain load fairness bütün action target'ları üzerinde anlamlı bir dağılım
tanısıdır. Buna karşılık mevcut deney her episode'da tek source device kullandığı
için source-success Jain fairness mekanik olarak 1'dir; çok kullanıcı adaleti
kanıtı olarak yorumlanamaz.

## 6. Reward component decomposition

| Koşul | Latency | Energy | Deadline | Soft constraint | Failure/drop | Hard constraint | Weighted cost ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Greedy no-semantic | 0,351 | 0,059 | 0,037 | 0,114 | 0,003 | 0,074 | 2,689 |
| Greedy reference | 0,366 | 0,066 | 0,036 | 0,051 | 0,002 | 0,000 | **0,887** |
| PPO no-semantic | 1,343 | 0,172 | 0,094 | 0,174 | 0,005 | 0,000 | 2,903 |
| PPO reference | 0,459 | 0,079 | 0,036 | 0,044 | 0,001 | 0,000 | 1,018 |

Decomposition, başarı artışının reward hacking'den gelmediğini incelemek için
tutulmuştur. Reference semantics hard constraint maliyetini sıfırlarken özellikle
soft constraint ve queue/latency bileşenlerini de düşürmüştür.

## 7. Reward sensitivity sonucu

Base ağırlıklar ile her reward bileşeninin ayrı ayrı `0,5×` ve `2×` yapıldığı 13
profil incelenmiştir. Bütün profillerde en düşük scalar cost koşulu
`greedy_semantic[reference]` olarak kalmıştır. PPO-reference eksi
greedy-reference cost farkı base profilde `+0,131`, %95 CI `[+0,075; +0,205]`;
profiller arasında `[+0,085; +0,223]` aralığındadır.

Bu analiz aynı dondurulmuş test trajectories üzerindeki reward components'i
yeniden ağırlıklandırır. PPO her ağırlık için yeniden eğitilmediği için kanıt
**evaluation-cost ranking robustness** kanıtıdır; training sensitivity veya
optimal-policy invariance iddiası değildir.

## 8. Oracle, parity ve adalet denetimi

Küçük exhaustive oracle yalnız valid action'ları tarar ve doğrudan tek fizik
çekirdeğinin döndürdüğü cost'u kullanır. Gym adaptörü, event replay ve baseline
runner aynı `transition` fonksiyonuna bağlıdır. Oracle ayrı bir fizik veya daha
iyi gizli preview kullanmaz. Oracle'ın amacı gerçekçi deployed comparator olmak
değil, reward/action-mask/parity hatalarını yakalamaktır.

## 9. Üretilen kanıtlar

- `artifacts/offloading_experiment/results_v2_nep.json`
- `phase_reports/OFFLOADING_RESULTS_V2_NEP_EXACT.md`
- `phase_reports/REWARD_SENSITIVITY_V2_NEP_EXACT.md`
- `src/task_offloading/system_metrics.py`
- `tests/phase_5/test_system_metrics.py`
- simulation oracle/parity/invariant testleri

## 10. Test ve kabul denetimi

- NEP-v2 manifest doğrulaması: **0 issue**;
- 5/5 PPO seed checkpoint ve her koşulda 3.840 karar: **tam**;
- 60/60 episode metric record ve finite-value denetimi: **geçti**;
- system metric golden/unit testleri: **geçti**;
- oracle, parity, mask ve deterministic seed testleri: **geçti**;
- repository-wide `pytest`: **264 passed**;
- repository-wide Ruff: **geçti**;
- strict mypy `src tests scripts`: **111 source file, 0 issue**.

## 11. Sınırlamalar ve sonraki riskler

1. Aggregate NEP quantiles gerçek iz dağılımını taşır fakat ham zaman sırasını ve
   kullanıcı korelasyonlarını yeniden kurmaz.
2. Tek source-device episode tasarımı user fairness'i ölçemez.
3. Reward sensitivity post-hoc'tur; her profil için yeniden training yapılmamıştır.
4. Exhaustive oracle küçük doğrulama durumları içindir; ana 20-device problem için
   global optimality iddiası kurulmaz.
5. Gerçek donanım/testbed enerji ve wall-clock latency ölçümü Phase 9'da ancak
   ayrı bir emulation protokolü açılırsa yapılabilir.
