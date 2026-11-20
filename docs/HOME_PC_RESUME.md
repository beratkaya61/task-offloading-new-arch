# Ev bilgisayarında hızlı devam notu

Bu dosya 20.11.2026 tarihli commit anındaki taşınabilir durumu kaydeder. Ayrıntılı
bilimsel devir notu `chat_histories/20_11_2026_chat.md`, iyileştirme kuralları
`docs/EXPLORATORY_IMPROVEMENT_PROTOCOL_V1.md` içindedir.

## Commit ile taşınan önemli durum

- Faz-5 semantic benchmark ve PPO-v1 kodu/raporları önceki `bcf1920` commit'indedir.
- NEP quantile profili üretildi ve commit'e alındı.
- Bağımsız paraphrase seti hazır: 200 adaydan 188 kabul, 12 ret. Metinler makine
  üretimidir; etiketler değişmezlik varsayımıyla aktarılmıştır, insan etiketi değildir.
- Qwen2.5-1.5B 4-shot sonuçları ve ham insan-60 tahminleri commit'e alındı.
- E5 gradual-unfreezing ve Qwen kısa label-code iyileştirme adayları hazırdır.
- Son doğrulama: `243 passed`.

## Git dışında kalanlar

Git'e model ağırlıkları, ham trace'ler, PPO checkpoint'leri, loglar ve ara/smoke
çıktıları girmez.

- Semantic/LLM deneyleri için model ilk çalıştırmada
  `models/huggingface/` altına indirilir. İnternet veya aynı cache'in kopyası gerekir.
- PPO/trace deneyleri için şu arşivler beklenir:
  - `D:\task_offloading_datasets\uci\TATDescriptionDataset.zip`
  - `D:\task_offloading_datasets\bupt\Edge-Computing-Dataset-04e664fab9cdb2a58d04ebc615cd74405e6062e2.zip`
  - `D:\task_offloading_datasets\eua\eua-dataset-61238e00a6f01019c8a6ba1619675337a879a3d4.zip`
- NEP'nin 55 GB açılmış CSV'leri yalnız quantile profilini yeniden üretmek için
  gerekir. Normal `--nep` PPO çalışması commit'teki aggregate profili kullanır.
- `artifacts/` Git dışında olduğundan ev bilgisayarında önce PPO training komutu,
  sonra `--eval-only-from` komutları çalıştırılmalıdır.

## Kurulum

```powershell
git pull origin main
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

## Sıradaki komutlar

Mevcut bilgisayardaki arka plan süreci Git ile taşınmaz. Sonucu ayrıca push
edilmemişse ev bilgisayarında buradan devam edilir:

```powershell
# 1) Kalan mevcut Qwen değerlendirmeleri
$env:LLM_SETS="independent_paraphrase"
.\.venv\Scripts\python.exe scripts\run_llm_baseline.py

$env:LLM_SUBSET="200"
$env:LLM_SETS="compositional600_paraphrase"
.\.venv\Scripts\python.exe scripts\run_llm_baseline.py
Remove-Item Env:LLM_SUBSET -ErrorAction SilentlyContinue
Remove-Item Env:LLM_SETS -ErrorAction SilentlyContinue

# 2) Ön kayıtlı v3.2 calibration ve semantic özet
.\.venv\Scripts\python.exe scripts\run_semantic_benchmark_v3.py
.\.venv\Scripts\python.exe scripts\summarize_semantic_v3.py

# 3) Yalnız development üzerinde exploratory iyileştirme seçimi
$env:IMPROVEMENT_MODE="e5"
.\.venv\Scripts\python.exe scripts\run_semantic_improvement_dev.py
$env:IMPROVEMENT_MODE="llm"
.\.venv\Scripts\python.exe scripts\run_semantic_improvement_dev.py
Remove-Item Env:IMPROVEMENT_MODE -ErrorAction SilentlyContinue

# 4) NEP PPO training ve kilitli değerlendirmeler
.\.venv\Scripts\python.exe scripts\run_offloading_experiment.py --nep --tag v2_nep --beliefs reference
.\.venv\Scripts\python.exe scripts\run_offloading_experiment.py --nep --tag v2_nep_eval --eval-only-from v2_nep
.\.venv\Scripts\python.exe scripts\run_offloading_experiment.py --nep --pool compositional_subset200 --tag v2_nep_sub200 --eval-only-from v2_nep
.\.venv\Scripts\python.exe scripts\run_offloading_experiment.py --nep --pool independent --tag v2_nep_indep --eval-only-from v2_nep
.\.venv\Scripts\python.exe scripts\run_offloading_experiment.py --nep --preview-noise 0.5 --conditions ppo_semantic,ppo_no_semantic --tag v2_nep_noise
.\.venv\Scripts\python.exe scripts\summarize_offloading_results.py artifacts\offloading_experiment\results_v2_nep_eval.json --output phase_reports\OFFLOADING_RESULTS_V2.md

# 5) Tez yöntem bölümünü senkron tut
.\.venv\Scripts\python.exe scripts\build_thesis_method_chapter.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Değişmez araştırma kuralları

- Makine/AI etiketi insan etiketi olarak sunulmaz.
- Test etiketleri training, prompt, threshold veya hyperparameter seçmek için kullanılmaz.
- Mevcut preregistration v2 sonuçları ile sonuçlardan sonra başlayan exploratory
  improvement sonuçları ayrı raporlanır.
- Smoke, log, model cache, ham veri ve eski/sızıntılı robustness dosyaları commit edilmez.
