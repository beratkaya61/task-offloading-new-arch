# Faz 7 Genelleme ve Source-Layer Ablation Sonuçları

**Durum:** Faz 7'nin bu bilgisayarda yürütülebilen dondurulmuş deneyleri tamamlandı. Protokol: `configs/generalization_protocol.v1.json`. Ana iddia gerçek testbed veya deployed-application genellemesi değil; trace-driven-hybrid simülasyonda zaman/entity/scenario-family/domain sağlamlığıdır.

## 1. Araştırma soruları ve yöntem

1. Semantik gereksinimleri gözleme eklemek, görülmemiş physical entity ve domain koşullarında no-semantic PPO'ya göre başarı avantajını koruyor mu?
2. BUPT zaman ayrımı ve disjoint semantic family ayrımı altında sonuç yönü korunuyor mu?
3. NEP background, UCI compute heterogeneity, BUPT workload variability ve EUA topology variability katmanları frozen policy sonucunu ne ölçüde değiştiriyor?

| Boyut | Dondurulmuş uygulama | İddia sınırı |
|---|---|---|
| Temporal | BUPT day <24 train, day >=24 test | Kronolojik trace holdout |
| Entity | Seeded EUA row split: 80% train / 20% test; yeniden eğitim | Görülmemiş EUA entity/location row |
| Application proxy | Development/evaluation scenario ve paraphrase family'leri disjoint | Gerçek application-ID değil |
| Domain | `public_safety`, `transportation` yalnız testte; yeniden eğitim | Reference semantics koşullu policy-domain generalization |
| Physical source | Aynı `v2_nep` checkpoint; tek evaluation katmanı counterfactual olarak sabitlenir/çıkarılır | Retraining sensitivity değil |

Her retraining koşulu 5 seed × 150.000 PPO step; evaluation 60 episode × 64 task = 3.840 karar/seed'dir. Holdout farkı PPO-reference − PPO-no-semantic episode-paired bootstrap ortalaması ve %95 CI'dır (2.000 resample, seed `20261120`).

## 2. Neden bu metrikler?

- `success rate`, task'ın completion ile semantic deadline/reliability/accuracy koşullarını birlikte karşılayıp karşılamadığını ölçen birincil uçtan uca metriktir.
- `hard violation rate`, privacy/device-permission gibi kesin kısıtların başarı ortalaması içinde gizlenmesini önler.
- `PPO-reference − PPO-no-semantic` paired farkı, mimari katkıyı aynı evaluation episodes üzerinde izole eder.
- Greedy-reference sonucu, PPO'yu zayıf baseline ile karşılaştırma riskini önleyen güçlü heuristic kontrolüdür.
- Source counterfactual'ları, aggregate başarı değişiminin hangi physical katmanlara duyarlı olduğunu gösterir; nedensel kaynak katkısı kanıtlamaz.

## 3. Strict retraining holdout sonuçları

| Koşul | EUA train/test row | Held-out domain | PPO-ref success | PPO-ref hard | PPO-no-sem success | Success farkı [CI] |
|---|---:|---|---:|---:|---:|---:|
| Entity holdout | 3415/854 | — | 91.22 | 0.00 | 76.19 | +15.03 [+13.82; +16.25] |
| Domain holdout | — | public_safety, transportation | 92.93 | 0.00 | 79.53 | +13.41 [+12.37; +14.51] |

Entity holdout'ta semantic avantaj `+15,03` puan [%95 CI `+13,82; +16,25`], domain holdout'ta `+13,41` puandır [%95 CI `+12,37; +14,51`]. İki durumda da PPO-reference hard violation `%0,00` kalmıştır. Böylece semantic-aware kararın yönü entity ve domain shift altında korunmuştur.

Domain holdout'ta PPO-reference success `%92,93`, aynı koşuldaki güçlü greedy-reference `%93,45`'tir; PPO − greedy farkı `−0,52` puandır [%95 CI `−0,98; −0,08`, Holm `p=0,023`]. Bu nedenle PPO'nun greedy'yi geçtiği iddiası desteklenmez.

## 4. Fiziksel source-layer counterfactual'ları

Aynı `v2_nep` checkpoint'leri yüklenmiş, yalnız bir evaluation source layer'ı değiştirilmiştir. `Δ PPO success`, bütün katmanlı base'e göredir.

| Evaluation katmanı | Greedy-ref success | PPO-ref success | Δ PPO success | PPO-ref hard | PPO-no-sem success |
|---|---:|---:|---:|---:|---:|
| Bütün katmanlar | 91.74 | 91.21 | 0.00 | 0.00 | 76.19 |
| NEP background çıkarıldı | 91.35 | 90.73 | -0.48 | 0.00 | 75.87 |
| UCI compute heterogeneity çıkarıldı | 85.94 | 82.29 | -8.92 | 0.00 | 67.66 |
| BUPT workload variability çıkarıldı | 92.84 | 92.13 | 0.92 | 0.00 | 80.94 |
| EUA topology variability çıkarıldı | 92.01 | 91.52 | 0.31 | 0.00 | 76.38 |

En büyük mutlak değişim, UCI compute heterogeneity çıkarıldığında PPO-reference success'in `−8,92` puan düşmesidir. BUPT variability sabitlendiğinde `+0,92`, EUA variability sabitlendiğinde `+0,31`, NEP background çıkarıldığında `−0,48` puan değişim görülmüştür. Bunlar aynı frozen policy'nin evaluation-distribution duyarlılığıdır; katmanın nedensel faydası veya yeniden eğitim sonucu değildir.

## 5. Desteklenen iddialar ve negative results

- **Desteklendi:** Semantic-aware PPO avantajı temporal protocol, EUA entity holdout ve iki-domain holdout altında no-semantic PPO'ya karşı korundu; hard violation artmadı.
- **Desteklendi:** Synthetic/trace-driven-hybrid ana yön tutarlılığı korundu; bu gerçek testbed genellemesi değildir.
- **Desteklenmedi:** PPO'nun güçlü semantic greedy baseline'ı geçtiği iddiası domain holdout'ta da doğrulanmadı.
- **Sınırlandı:** Application sonucu disjoint scenario/paraphrase-family proxy'sidir; deployed application-ID holdout olarak sunulamaz.
- **Sınırlandı:** Frozen semantic extractor'lar domain dışarıda bırakılarak yeniden eğitilmedi; domain sonucu reference-semantics koşulludur.

## 6. Yorum ve geçerlilik sınırları

- Entity ve domain tabloları yeni 5-seed retraining kullanır.
- Domain holdout, reference semantics koşulunda policy generalization kanıtıdır; extractor'lar leave-domain-out yeniden eğitilmemiştir.
- Source-layer tablosu evaluation-distribution counterfactual'ıdır; her katman için policy yeniden eğitilmediğinden training sensitivity değildir.
- Aggregate NEP ve çok kaynaklı hybrid sonuçlar gerçek testbed sonucu olarak sunulamaz.
- Tek source-device nedeniyle source-success Jain fairness mekanik olarak 1'dir; user fairness kanıtı değildir.
- Holdout domain'leri sekiz domain'den dondurulmuş seed ile seçilen iki domain'dir; daha geniş domain çeşitliliği dış geçerliliği güçlendirebilir.

## 7. Kanıt, tekrar üretim ve kalite kapısı

- Protocol: `configs/generalization_protocol.v1.json`
- Entity artifact/report: `results_v2_nep_entity_holdout.json`, `OFFLOADING_RESULTS_V2_NEP_ENTITY_HOLDOUT.md`
- Domain artifact/report: `results_v2_nep_domain_holdout.json`, `OFFLOADING_RESULTS_V2_NEP_DOMAIN_HOLDOUT.md`
- Source artifacts: `results_v2_nep_source_without_{nep,uci,bupt,eua}.json`
- Runner: `scripts/run_offloading_experiment.py`; bu özet: `scripts/summarize_phase7_generalization.py`
- Otomatik kontroller: frozen protocol schema testi, deterministic/disjoint entity split testi, source-isolation testi ve paired CI testi.
- Artifact kabul kapısı bu rapor üretilirken fail-closed çalışır: her koşul 5 seed, her PPO ailesi 5 run ve her run 3.840 karar olmalıdır.
