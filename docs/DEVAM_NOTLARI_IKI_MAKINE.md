# Devam Notları — İki Makineli Çalışma (27.11.2026)

| Makine | Donanım | Ne için |
|---|---|---|
| **Quadro makinesi** | Quadro RTX 4000, 8 GB | Qwen3-8B inference ve QLoRA eğitimi, fine-tuned E5 ve GPU gerektiren her şey |
| **Ev makinesi** | GPU yok | Etiketleme koordinasyonu ve analizi, veri çeşitliliği (bağlam yazımı, v2 yeniden build), TF-IDF/frozen E5 benchmark'ları, rapor yazımı |

## Makineye geçince ilk adımlar

```powershell
git fetch origin
git checkout semantic-expansion-v2
git pull
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest -q
```

Kör etiketleme anahtarı (`blinding_key.coordinator_only.jsonl`) ve HTML formlar git'te tutulmaz. Paketler ve anahtar deterministik olarak yeniden üretilir:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts\prepare_expansion_v2_annotation.py
```

Üretilen anahtarın SHA-256 değeri `annotation_plan.v2.json` içindeki `blinding_key_sha256` ile aynı olmalıdır.

**Quadro makinesi uyarısı:** `pip install -r requirements*.txt` torch'u CPU sürümüne yükseltebilir, çünkü stable-baselines3 torch ≥ 2.8 istiyor. Bu olursa:

```powershell
.\.venv\Scripts\python.exe -m pip install "torch==2.7.1+cu118" --index-url https://download.pytorch.org/whl/cu118 --no-deps
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

## Durum

| İş | Durum | Makine |
|---|---|---|
| Expanded-v2 (sızıntısız, 4.560 satır) | Tamamlandı | — |
| Qwen3-8B 0/4/8-shot, eski setler | Tamamlandı | Quadro |
| Eşit supervision geçici tablo: rule, TF-IDF, frozen E5, fine-tuned E5 | Tamamlandı | Quadro |
| QLoRA'nın 8 GB'de çalışması (tepe 6,24 GiB) | Doğrulandı | Quadro |
| Annotation paketleri, kılavuz, analiz betiği | Hazır | — |
| Pilot ve ana etiketleme; analiz komutu | Sırada | Ev |
| Qwen3-8B ile doğal ifade varyantları | Planlandı | Quadro (üretim), ev (filtreleme ve inceleme) |
| QLoRA geçici eğitim ve değerlendirme; adaptör değerlendirme betiği yazılacak | Kullanıcı kararı bekleniyor | Quadro |
| Qwen3-8B 0/4/8-shot'ın v2 validation/test'te çalıştırılması | Sırada | Quadro |
| Final: insan etiketleriyle bütün yöntemlerin yeniden eğitimi | İnsan etiketleri sonrası | Quadro (Qwen ve fine-tuned E5), ev (diğerleri) |

## Sık kullanılan komutlar

| Amaç | Komut (`$env:PYTHONPATH='src'` ile) | GPU |
|---|---|---|
| v2 veri setini yeniden üret ve sızıntı denetimi | `python scripts\build_semantic_expansion_v2.py` | Hayır |
| Annotation paketleri ve formlar | `python scripts\prepare_expansion_v2_annotation.py` | Hayır |
| Teslim denetimi ve uyum raporu | `python scripts\analyze_expansion_v2_annotation.py` | Hayır |
| v2 karşılaştırması (fine-tune hariç) | `$env:SKIP_FINETUNE='1'; python scripts\run_semantic_benchmark_expanded_v2.py` | Hayır (CPU'da ~6 dk) |
| v2 karşılaştırması (fine-tuned E5 dahil) | `python scripts\run_semantic_benchmark_expanded_v2.py` | Evet (~20 dk) |
| Qwen prompt-only | `$env:LLM_MODEL='Qwen/Qwen3-8B'; $env:LLM_QUANTIZATION='4bit'; $env:LLM_SHOTS='4'; python scripts\run_llm_baseline.py` | Evet |
| QLoRA eğitimi | `python scripts\train_qwen_lora.py` (`QWEN_*` ortam değişkenleri) | Evet (~1 saat/epoch) |
| Denetim raporu ve danışman DOCX | `python scripts\build_semantic_method_audit.py`; `python scripts\build_advisor_report.py` (pandoc gerekir) | Hayır |

**Not:** Ev makinesinde E5 modeli ilk kullanımda Hugging Face'ten indirilir (yaklaşık 470 MB). Qwen3-8B ise yalnız Quadro makinesinde `models/huggingface` altında durur ve git'e girmez.
