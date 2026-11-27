# Semantik Veri, Yöntem ve Karşılaştırma Denetimi — 27.11.2026

## 1. Karar özeti

- Eski confirmatory v3.2 sonuçları korunur; 180 temel development görevi ve bunlardan türetilmiş 1.924 training satırı tarihsel baseline'dır.
- Cached sonuç dosyasındaki `1.980` değeri sabit yazılmış metadata hatasıdır. Güncel training dosyaları ve authoritative `results.v3.json` değeri `1.924`'tür.
- Qwen2.5-1.5B koşulu supervised training değildir: 180 development etiketi çıktı sözlüğünü tanımlar, yalnız dört metin–etiket çifti prompt demonstrasyonudur.
- Expanded-v1 iki sızıntı yolu nedeniyle geri çekildi: 600 ailenin 244'ü kilitli değerlendirme bağlamlarını kullanıyordu ve 120 cihazın 98'i split'ler arasında geçiyordu.
- Yeni ortak koşul Expanded-v2'dir: `3024` train, `768` validation, `768` test satırı; cihazlar ve işlemler split'ler arasında geçmez. Prompt-only Qwen 0/4/8-shot ayrı track'tir.
- Expanded-v2 henüz final veri seti değildir; insan annotation ve adjudication tamamlanana kadar yalnız candidate/engineering artifact statüsündedir.

## 2. Veri envanteri

| Artifact | Bağımsız birim | İlişkili satır | Label durumu | Rol |
|---|---:|---:|---|---|
| v3 development | 90 scenario family / 180 metin | 180 | Machine weak | Eski D180 training |
| v3 augmented | Aynı 180 temel metin | 1.924 | Weak + construction | Eski augmented training |
| Human-60 original | 30 scenario family / 60 metin | 60 | Single-human reference | Eski primary test |
| Human-60 rewrite | Human-60 ile aynı görevler | 60 | Label transfer | Wording robustness |
| Compositional | Evaluation domain cümleleri | 600 + 600 | Construction | Diagnostic test |
| Independent paraphrase | 188 kabul edilen metin | 188 | Label transfer | External wording robustness |
| Expanded-v2 candidate | 250 cihaz grubu / 570 senaryo ailesi | 4560 | Construction, human review pending | Yeni ortak benchmark |

Expanded-v2 split'i cihaz-grubu düzeyindedir: `3024` train, `768` validation, `768` test satırı. Validation/test yalnız yeni cihazlardan oluşur. Zorunlu sızıntı denetimleri: `duplicate_visible_texts=0`, `subject_group_cross_split=0`, `semantic_profile_cross_split=0`, `subject_cross_split=0`, `operation_cross_split=0`, `exact_text_overlap_with_locked_evaluation=0`, `heldout_exact_text_overlap_with_historical_training=0`, `rows_containing_banned_evaluation_context_terms=0`, `heldout_subject_or_operation_found_in_historical_texts=0`.

İnsan annotation (yalın tasarım, 2 etiketleyici): rol sayıları `hidden_repeat=24`, `pilot=40`, `test_reference=384`, `training_audit=192`, `validation_reference=384`; etiketleyici başına `1024` görev; B'ye ek `60` Human-60 görevi.

## 3. Yöntem × task-specific supervision matrisi

| Yöntem | Eski task-specific exposure | Parametre öğrenimi | Yeni ortak koşul |
|---|---|---|---|
| Constant profile | 180 label | Basit profil | Expanded train |
| Rule/keyword | Training yok; kurallar önceden yazılı | Yok | Aynı test; ayrı non-learning baseline |
| TF-IDF + LR | D180 ve D1924 | Evet | Expanded train/validation/test |
| Frozen multilingual-E5 + LR | D180 ve D1924 | Yalnız LR head | Expanded train/validation/test |
| Fine-tuned multilingual-E5 | D1924 | Backbone + heads | Expanded train/validation/test |
| Qwen2.5-1.5B 4-shot | 180 label vocabulary + 4 demo | Yok | Tarihsel prompt baseline |
| Qwen3-8B 0/4/8-shot | Label vocabulary + k demo | Yok | Ayrı in-context track |
| Qwen3-8B QLoRA | Henüz final eğitim yok | LoRA adapter (8 GB nedeniyle 4-bit taban) | Expanded-v2 train/validation/test |

## 4. Mevcut Macro-F1 sonuçlarının tamamı

| Yöntem | Human-60 original | Human-60 rewrite | Comp-600 canonical | Comp-600 unseen wording | Independent-188 |
|---|---:|---:|---:|---:|---:|
| Constant profile | %25,3 | %25,3 | %20,7 | %20,7 | %20,9 |
| Rule/keyword | %78,0 | %40,3 | %98,5 | %47,5 | %54,6 |
| TF-IDF + LR (180) | %76,6 | %48,1 | %68,4 | %36,7 | %44,0 |
| TF-IDF + LR (1.924) | %78,2 | %67,9 | %99,6 | %73,6 | %80,9 |
| Frozen E5 + LR (180) | %75,8 | %51,8 | %60,9 | %44,9 | %51,7 |
| Frozen E5 + LR (1.924) | %76,3 | %60,4 | %86,8 | %70,9 | %73,8 |
| Fine-tuned E5 (1.924) | %70,2 | %62,3 | %87,5 | %73,0 | — |
| Qwen2.5-1.5B 4-shot | %48,3 | %43,7 | — | — | %42,0 |

## 5. Human-60 original ayrıntılı metrikler

| Yöntem | Exact-record | Macro-F1 | Privacy F1 | Abstention F1 | Schema validity | Latency ms/task |
|---|---:|---:|---:|---:|---:|---:|
| Constant profile | %3,3 | %25,3 | %0,0 | %93,0 | %100,0 | 0,0 |
| Rule/keyword | %38,3 | %78,0 | %54,5 | %95,7 | %100,0 | 0,2 |
| TF-IDF + LR (180) | %30,0 | %76,6 | %54,5 | %95,7 | %100,0 | 1,6 |
| TF-IDF + LR (1.924) | %38,3 | %78,2 | %54,5 | %95,7 | %100,0 | 1,6 |
| Frozen E5 + LR (180) | %33,3 | %75,8 | %50,0 | %94,9 | %100,0 | 19,0 |
| Frozen E5 + LR (1.924) | %36,7 | %76,3 | %54,5 | %94,8 | %100,0 | 17,3 |
| Fine-tuned E5 (1.924) | %10,0 | %70,2 | %54,5 | %95,7 | %100,0 | 13,4 |
| Qwen2.5-1.5B 4-shot | %5,0 | %48,3 | %35,3 | %91,6 | %100,0 | 14.696,3 |

## 6. Veri artırmanın ölçülen etkisi

| Model ailesi | Set | D180 Macro-F1 | D1924 Macro-F1 | Fark |
|---|---|---:|---:|---:|
| TF-IDF + LR | Human-60 original | %76,6 | %78,2 | +1,6 puan |
| TF-IDF + LR | Human-60 rewrite | %48,1 | %67,9 | +19,8 puan |
| TF-IDF + LR | Comp-600 canonical | %68,4 | %99,6 | +31,2 puan |
| TF-IDF + LR | Comp-600 unseen wording | %36,7 | %73,6 | +36,9 puan |
| TF-IDF + LR | Independent-188 | %44,0 | %80,9 | +36,9 puan |
| Frozen E5 + LR | Human-60 original | %75,8 | %76,3 | +0,5 puan |
| Frozen E5 + LR | Human-60 rewrite | %51,8 | %60,4 | +8,6 puan |
| Frozen E5 + LR | Comp-600 canonical | %60,9 | %86,8 | +25,9 puan |
| Frozen E5 + LR | Comp-600 unseen wording | %44,9 | %70,9 | +26,0 puan |
| Frozen E5 + LR | Independent-188 | %51,7 | %73,8 | +22,1 puan |

## 7. Qwen3-8B prompt-only 0/4/8-shot

Aynı Qwen3-8B (4-bit NF4, 8 GB GPU), aynı label encoding ve greedy decoding; yalnız paired demonstrasyon sayısı değişir. Ağırlıklar güncellenmez.

| Koşul | Human-60 original | Human-60 rewrite | Independent-188 | Ortalama gecikme | OOV alan |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B 0-shot | %50,2 | %51,3 | %47,3 | 8,2–9,3 s | 123 |
| Qwen3-8B 4-shot | %55,4 | %54,4 | %55,3 | 9,3–10,9 s | 60 |
| Qwen3-8B 8-shot | %54,8 | %52,3 | %52,4 | 19,4–22,0 s | 41 |

## 8. Eşit supervision geçici karşılaştırması (Expanded-v2, construction label)

Bütün supervised yöntemler aynı Expanded-v2 train ile eğitildi; hiperparametreler v3'ten sabit; validation/test hiçbir seçimde kullanılmadı. v2 setlerinde CI cihaz-grubu cluster bootstrap'ıdır.

| Yöntem | v2_validation | v2_test | human60_original | human60_paraphrase | compositional600_canonical | compositional600_paraphrase | independent_paraphrase |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant_profile | %20,6 | %20,6 | %25,3 | %25,3 | %20,7 | %20,7 | %20,9 |
| rule_keyword | %79,8 | %79,7 | %78,0 | %40,3 | %98,5 | %47,5 | %54,6 |
| tfidf_lr_v2 | %86,9 | %87,6 | %78,2 | %61,0 | %99,4 | %72,4 | %81,7 |
| e5_frozen_lr_v2 | %84,2 | %84,5 | %70,6 | %55,3 | %93,0 | %74,8 | %78,6 |
| e5_finetuned_v2 | %88,9 | %89,3 | %77,3 | %65,8 | %98,6 | %80,5 | %85,5 |

- `tfidf_lr_v2` − rule (v2 test): +7,9 puan [95% CI +6,7; +9,4]
- `e5_frozen_lr_v2` − rule (v2 test): +4,8 puan [95% CI +3,0; +6,3]

## 9. Qwen3-8B task-specific eğitim teknik kapısı

- Tarihsel v1 smoke: `FP16 Qwen3-8B base weights plus training activations exceed the detected 8.0 GiB device; using QLoRA`; seçilen yükleme `4bit`; `1` update; peak CUDA allocation `9.15` GiB (8 GiB kartı aştı).
- Düzeltmeler: hafif k-bit hazırlık (embedding/lm_head FP32'ye çevrilmez), 8-bit paged AdamW, reentrant olmayan gradient checkpointing; max length 320 ve cevap kesilirse eğitim durur (256 token v2'de 157 satırın cevabını kesiyordu).
- Final eğitim insan etiketlemesi ve validation'da ayar dondurma sonrasıdır.

## 10. Final karşılaştırma kapıları

1. Candidate metinler naturalness ve label agreement açısından iki kör annotator tarafından etiketlenmeden benchmark dondurulmaz.
2. Aile-gruplu train/validation/test ayrımı değiştirilemez; test hiçbir prompt, hyperparameter, epoch veya threshold seçiminde kullanılmaz.
3. TF-IDF, frozen E5, fine-tuned E5 ve Qwen3-8B QLoRA aynı adjudicated training satırları ve aynı validation/test üzerinde çalıştırılır.
4. Qwen3-8B 0/4/8-shot ayrı track olarak verilir; LoRA sonucuyla birleştirilmez.
5. Macro-F1, exact-record, per-field F1, privacy/execution recall, schema validity, OOV/parse error, latency, VRAM ve training compute birlikte raporlanır.
