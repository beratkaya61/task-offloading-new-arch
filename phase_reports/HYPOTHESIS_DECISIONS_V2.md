# Pre-registered System Hypotheses - v2 Decision Matrix

Kaynaklar: `results_v2_nep.json` (exact-preview) ve `results_v2_nep_noise.json` (sigma=0.5 noisy-preview). Farklar paired episode bootstrap, 2,000 resample ve %95 percentile CI ile hesaplanmistir. Yuzde-oran farklari percentage point cinsindedir.

| Hipotez | On-kayitli test | Sonuc | Karar |
|---|---|---|---|
| H1 | Reference semantic success > no-semantic, PPO ve greedy | PPO +15.03 [+13.85, +16.27]; greedy +12.94 [+11.77, +14.12] | Desteklendi |
| H2 | Rule hard violation > TF-IDF(+data), PPO ve greedy | PPO +10.10 [+9.24, +11.01]; greedy +5.65 [+4.71, +6.64] | Desteklendi |
| H3 | Calibrated fallback hard violation'i azaltir ve success CI alt siniri > -2 pp | Asagidaki model/policy tablosu | Desteklenmedi |
| H4 | Mask-PPO hard violation < penalty-PPO ve success alt sinir > -1 pp | hard +0.00 [+0.00, +0.00]; success -0.18 [-0.36, -0.02] | Desteklenmedi |
| H5 | Noisy-preview reference PPO - greedy success CI alt siniri > -1 pp | -0.98 [-1.44, -0.55] | Desteklenmedi |
| H6 | NEP katmaninda H1 ve H2 yonu korunur | H1/H2 NEP'de ayni yonde | Desteklendi |

## H3 fallback ayrintisi

Fallback-baseline farkidir; hard violation icin negatif, success icin CI alt sinirinin -2 pp'den buyuk olmasi beklenir.

| Policy/source | Success farki [CI] | Hard violation farki [CI] | Iki kosul birlikte gecti mi? |
|---|---:|---:|---|
| ppo_semantic/tfidf_aug_paraphrase | -0.52 [-0.92, -0.13] | -0.40 [-0.59, -0.20] | Evet |
| ppo_semantic/e5_frozen_aug_paraphrase | -1.99 [-2.41, -1.56] | +0.10 [+0.05, +0.17] | Hayir |
| greedy_semantic/tfidf_aug_paraphrase | -0.36 [-0.78, +0.08] | -0.05 [-0.29, +0.16] | Hayir |
| greedy_semantic/e5_frozen_aug_paraphrase | -1.61 [-2.14, -1.12] | +0.26 [+0.00, +0.52] | Hayir |

## Yorum sinirlari

- H4'te iki reference PPO kosulu da sifir hard violation verdi; esitlik, on-kayitli 'mask daha dusuk' yonunu desteklemez.
- H5 point estimate margin civarinda olsa da CI alt siniri -1 pp'nin altindadir; non-inferiority ilan edilmez.
- H3 fine-tuned E5 parcasi calibrated independent prediction/checkpoint olmadigi icin tamamlanamadi. Mevcut TF-IDF/E5-frozen kosullarinin da tamami kurali gecmediginden H3 genel karari degismez.
- Holm-adjusted p-values full publication tablolarinda raw p-values ile birlikte raporlanmistir.
