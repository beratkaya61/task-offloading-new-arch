# Publication Reproducibility Guide

Bu rehber semantic benchmark v3.2, NEP-v2 offloading, Phase 7 generalization,
publication tables/figures ve advisor DOCX çıktısını temiz bir checkout'tan yeniden
üretmek veya mevcut dondurulmuş çıktıları doğrulamak içindir.

## 1. Bilimsel kapsam ve iddia sınırı

- Benchmark bir **multi-source trace-driven-hybrid simulation**dır; gerçek testbed
  değildir.
- BUPT, NEP, UCI ve EUA doğal olarak jointly observed tek event gibi sunulmaz.
- Ana offloading koşulları 5 PPO seed × 150.000 step; evaluation 60 episode × 64
  task = 3.840 decision/seed içerir.
- `configs/publication_suite.v1.json` current-PC zorunlu stages ile
  `quadro_optional` genişletmeyi ayırır.
- Qwen3-4B/8B, 10-seed extension, GNN/recurrent/partial offloading ve testbed ana
  sonucu yeniden üretmek için zorunlu değildir.

## 2. Platform ve software prerequisites

Doğrulanan yerel platform:

- Windows / PowerShell;
- Python `3.11`;
- paket aralıkları `requirements.txt` ve quality tools `requirements-dev.txt`;
- advisor DOCX üretimi için `pandoc` komutu;
- yaklaşık 40+ GB boş alan gerektirebilen external trace/model cache.

Temiz environment:

```powershell
py -3.11 -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

`requirements.txt`, kod tarafından doğrudan kullanılan bütün runtime/scientific
dependencies'i içerir. Gerçek kurulu sürümler `reproduce_publication.py` verify
çıktısında kaydedilir; publication evidence manifest bu environment ile üretilir.

## 3. External data ve model cache

Ham trace arşivleri, büyük model ağırlıkları, PPO checkpoints ve generated
`artifacts/` Git'e eklenmez. Kaynak URL/revision/license/status sözleşmesi
`configs/data_sources.v1.json`, schema `schemas/dataset_manifest.schema.json`,
governance kuralları `docs/DATA_GOVERNANCE.md` içindedir.

Dataset root'u açıkça ayarlayın:

```powershell
$env:TASK_OFFLOADING_DATASETS = "D:\task_offloading_datasets"
```

Runner bu root altında pinned UCI/BUPT/EUA archive layout'unu doğrular. NEP için
ham arşiv tekrar taranmak zorunda değildir; publication run sürümlü aggregate
profile `configs/data_profiles/nep_quantiles_v1.json` kullanır. Profile kaynak
hash/satır kabul bilgilerini taşır; ham trace sonucu gibi sunulmaz.

Transformer/Sentence-Transformers model cache'i ilgili pinned revision'ları daha
önce indirmiş olmalıdır. Offline makinede eksik model indirme denenmez.

## 4. Suite config'i ve outputs'u doğrulama

Yalnız config contract'ı:

```powershell
./.venv/Scripts/python.exe scripts/reproduce_publication.py --validate-config
```

Mevcut publication outputs için read-only full verify:

```powershell
./.venv/Scripts/python.exe scripts/reproduce_publication.py
```

Bu komut training başlatmaz. Şunları doğrular:

- bütün zorunlu outputs mevcut mu;
- result metadata config ile aynı mı;
- her PPO condition 5 run içeriyor mu;
- her run `60 × 64 = 3.840` decision taşıyor mu;
- JSON içinde NaN/Infinity var mı;
- output SHA-256 değerleri nedir;
- aktif Python/package versions nedir.

Belirli stages:

```powershell
./.venv/Scripts/python.exe scripts/reproduce_publication.py `
  --stage offloading_entity_holdout `
  --stage offloading_domain_holdout
```

Stage listesi:

```powershell
./.venv/Scripts/python.exe scripts/reproduce_publication.py --list
```

## 5. Deneyleri yeniden yürütme

Uzun işler yalnız explicit `--execute` ile başlar. Training stages `--resume`
kullanır; tamamlanmış checkpoint varsa yüklenir, yalnız eksik seed/condition yeniden
eğitilir.

Tek stage örneği:

```powershell
./.venv/Scripts/python.exe scripts/reproduce_publication.py `
  --stage offloading_domain_holdout --execute
```

Frozen result zaten mevcutsa orchestrator sessiz overwrite yapmaz. Bilinçli yeni
evidence run için ayrıca `--allow-frozen-overwrite` gerekir. Bu seçenek eski sonucu
korumak veya version bump yapmak planlanmadan kullanılmamalıdır.

Bütün current-PC suite:

```powershell
./.venv/Scripts/python.exe scripts/reproduce_publication.py --execute
```

Full suite CPU üzerinde çok uzun sürebilir. Exact/noisy/independent ve iki strict
holdout ayrı 5-seed training içerir. Süreci kesmek yerine stage bazında çalıştırmak
ve partial/checkpoint dosyalarını korumak önerilir.

### 5.1 Publication stage sırası

1. `semantic_score_v3_2`
2. `offloading_v2_nep_exact`
3. `offloading_v2_nep_noisy`
4. `offloading_v2_nep_independent`
5. `offloading_entity_holdout`
6. `offloading_domain_holdout`
7. `source_without_nep`, `source_without_uci`, `source_without_bupt`,
   `source_without_eua`
8. `publication_reports`
9. `advisor_document`

Source stages aynı frozen `v2_nep` checkpoints'i `eval-only` yükler. Bu nedenle
sonuçları training sensitivity veya causal source contribution diye yorumlamayın.

Fresh run'da host wall-clock `decision overhead` alanları byte-identical olmayabilir.
Bu nedenle `--execute` sonrası structural/statistical gates uygulanır ve yeni
SHA-256 değerleri yazdırılır; frozen evidence hash karşılaştırması yalnız varsayılan
read-only verify modunda zorunludur. Bilimsel sonuç değişirse yeni suite/evidence
version üretilmelidir.

## 6. Advisor Markdown/DOCX parity

Canonical source:

```text
TEZ_DANISMANI_MEVCUT_DURUM_VE_LITERATUR_OZETI.md
```

Figures ve DOCX:

```powershell
./.venv/Scripts/python.exe scripts/build_advisor_report_figures.py
./.venv/Scripts/python.exe scripts/build_advisor_report.py
./.venv/Scripts/python.exe scripts/build_advisor_report.py --verify-only
```

Build işlemi Word lock dosyası varsa durur, Markdown/DOCX visible-text parity'yi ve
embedded figure count'u doğrulamadan canonical DOCX'i değiştirmez.

## 7. Quality gate

```powershell
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/ruff.exe check .
./.venv/Scripts/mypy.exe --strict src tests scripts
```

Project `pyproject.toml` package-level strict mypy ayarını taşır. Publication
closure'da scripts/tests de açıkça strict mode ile kontrol edilir. Final sayılar ve
varsa intentionally excluded external/generated paths
`phase_reports/PHASE_10_REPRODUCIBILITY_AND_QUALITY.md` içinde kaydedilir.

## 8. Kanıt zinciri

- Suite/config: `configs/publication_suite.v1.json`
- Static evidence hashes/environment: `configs/publication_evidence.v1.json`
- Semantic: `phase_reports/PHASE_4_SEMANTIC_BENCHMARK.md`,
  `phase_reports/SEMANTIC_RESULTS_V3_2_CACHE.md`
- Baselines: `phase_reports/PHASE_5_BASELINES_ORACLE_NEP_V2.md`
- PPO/ablation: `phase_reports/PHASE_6_PPO_SEMANTIC_ABLATION_NEP_V2.md`
- Generalization: `phase_reports/PHASE_7_GENERALIZATION_RESULTS.md`
- Advisor: canonical Markdown + parity-checked DOCX
- Final software gate: `phase_reports/PHASE_10_REPRODUCIBILITY_AND_QUALITY.md`

Bir artifact değişirse read-only verify yeni SHA-256 değerini gösterecektir. Bilimsel
sonuç değişikliği yapılacaksa suite/evidence version artırılmalı; eski sonucu sessizce
overwrite etmek yerine yeni tag ve report üretilmelidir.
