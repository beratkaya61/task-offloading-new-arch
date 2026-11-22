# Faz 10 — Reproducibility ve Nihai Kalite Kapanış Raporu

**Kapanış tarihi:** 22 Kasım 2026

**Kapsam:** Semantic benchmark v3.2, NEP-v2 Faz 5/6, Faz 7 generalization,
publication outputs ve danışman Markdown/DOCX paketi

**Karar:** Bu bilgisayarda yürütülen zorunlu publication core tamamlandı. Quadro
deneyleri ve fiziksel testbed ana sonucu bloke etmeyen opsiyonel genişletmelerdir.

## 1. Kapanış ölçütleri

Faz 10 aşağıdaki ölçütlerle kapatıldı:

1. Sonuç artifact'leri, tablolar, figürler ve raporlar sürümlü bir suite altında
   toplandı.
2. Config–code–data–seed ilişkisi statik SHA-256 kanıt manifesti ve fail-closed
   doğrulama ile bağlandı.
3. Uzun training aşamaları ile read-only evidence verification birbirinden
   ayrıldı; varsayılan komut training veya overwrite yapmıyor.
4. Advisor Markdown tek canonical metin olarak tutuldu; DOCX visible-text parity
   ve embedded figure sayısı otomatik denetlendi.
5. Repository-wide unit/integration tests, lint ve strict type checking geçti.
6. Negative results, threats to validity ve claim sınırları advisor belgesine,
   faz raporlarına ve reproducibility rehberine taşındı.

## 2. Dondurulan deney sözleşmesi

Ana sözleşme `configs/publication_suite.v1.json` içindedir:

- Python: `>=3.11,<3.12`;
- experiment seed: `20261120`;
- PPO: 5 independent seed × 150.000 training step;
- değerlendirme: 60 episode × 64 task = 3.840 decision/run;
- paired bootstrap: 2.000 resample;
- zorunlu current-PC stages: semantic v3.2, NEP-v2 exact/noisy/independent,
  entity/domain holdout, dört source-layer counterfactual, publication reports ve
  advisor document;
- `quadro_qwen3_scale_extension`: açıkça `quadro_optional`.

`configs/publication_evidence.v1.json`; runtime package sürümlerini ve dondurulmuş
config, source, manifest, result, report, figure ve advisor document SHA-256
değerlerini taşır. Manifestteki `git_base_head_before_user_commit`, çalışmaya
başlanılan commit'i gösterir; tamamlanan fakat henüz kullanıcı tarafından commit
edilmemiş working tree içeriği dosya hash'leriyle dondurulur.

## 3. Yeniden üretilebilirlik davranışı

Varsayılan doğrulama komutu read-only'dir:

```powershell
./.venv/Scripts/python.exe scripts/reproduce_publication.py
```

Bu kapı zorunlu output varlığını, metadata sözleşmesini, finite JSON değerlerini,
5 run/condition ve 3.840 decision/run koşulunu, runtime versions ve frozen evidence
hash'lerini denetler. Uzun işler yalnız `--execute` ile başlar. Mevcut frozen
sonucun bilinçsizce ezilmesini önlemek için execution modunda ayrıca
`--allow-frozen-overwrite` gerekir.

Fresh execution'da host wall-clock `decision overhead` byte-identical olmak
zorunda değildir. Bu nedenle fresh run structural/statistical gates ile;
publication snapshot ise SHA-256 ile doğrulanır. Bilimsel sonuç değişirse suite ve
evidence version artırılmalıdır.

## 4. Nihai kalite matrisi

| Kapı | Komut/kapsam | Sonuç |
|---|---|---:|
| Full test suite | `python -m pytest -q` | **264 passed (47.60 s)** |
| Lint | `python -m ruff check .` | **All checks passed** |
| Strict typing | `python -m mypy --strict src tests scripts` | **111 source file, 0 issue** |
| Publication structure/evidence | `scripts/reproduce_publication.py` | **Geçti; `issues=[]`** |
| Advisor parity | `scripts/build_advisor_report.py --verify-only` | **Geçti; 4 embedded figure** |

Test paketi; simulator invariants/oracle/parity, provenance/split leakage,
semantic parsing/calibration/cache, system metrics, statistical contrasts,
generalization/source protocol ve publication orchestrator fail-closed davranışını
kapsar. Generated external raw traces ve model weights kalite taramasına source
code olarak dahil edilmez; bunların source identity/checksum ve acceptance
contracts'ı sürümlenir.

## 5. Dondurulan yayın kanıtı

- Semantic benchmark: `phase_reports/PHASE_4_SEMANTIC_BENCHMARK.md` ve
  `phase_reports/SEMANTIC_RESULTS_V3_2_CACHE.md`;
- baseline/oracle/reward: `phase_reports/PHASE_5_BASELINES_ORACLE_NEP_V2.md`;
- PPO/semantic ablation: `phase_reports/PHASE_6_PPO_SEMANTIC_ABLATION_NEP_V2.md`;
- hypothesis decisions: `phase_reports/HYPOTHESIS_DECISIONS_V2.md`;
- generalization/source counterfactuals:
  `phase_reports/PHASE_7_GENERALIZATION_RESULTS.md`;
- advisor canonical source:
  `TEZ_DANISMANI_MEVCUT_DURUM_VE_LITERATUR_OZETI.md`;
- çalıştırma rehberi: `REPRODUCIBILITY.md`.

Ana sonuçların yorumu değişmedi: H1, H2 ve H6 desteklendi; H3, H4 ve H5
desteklenmedi. PPO'nun güçlü greedy baseline'ı geçtiği iddiası desteklenmedi.
Source-layer sonuçları retraining sensitivity veya causal source contribution
değil, frozen-policy evaluation-distribution counterfactual'larıdır.

## 6. Claim sınırları ve kalan riskler

- Çalışma gerçek testbed değil, provenance-korumalı multi-source
  **trace-driven-hybrid simulation**dır.
- Single-human-60 adjudicated two-expert gold değildir; pilot A/B/C agreement
  kanıtı ve independent paraphrase evaluation bu sınırlamayı azaltır fakat ortadan
  kaldırmaz.
- Application sonucu deployed application-ID holdout değil, disjoint
  scenario/paraphrase-family proxy'sidir.
- Domain holdout reference semantics koşullu policy generalization ölçer;
  extractor leave-domain-out kanıtı değildir.
- Source-device fairness tek source nedeniyle mekanik olarak 1'dir ve user fairness
  iddiasına dönüştürülemez.
- Qwen3-4B/8B model-scale analizi, 10-seed confirmatory extension ve physical
  testbed yapılırsa yeni sürüm/ek deney olarak raporlanmalıdır.

## 7. Sonraki işlem

Bu bilgisayarda zorunlu deneysel ve yazılımsal kapanış tamamlanmıştır. Kalan tek
depo işlemi kullanıcının farkları inceleyip **manuel commit** oluşturmasıdır; bu
rapor hazırlanırken otomatik commit atılmamıştır. Quadro RTX 4000 işleri daha sonra
opsiyonel extension olarak ele alınabilir ve mevcut danışman teslimini bloke etmez.
