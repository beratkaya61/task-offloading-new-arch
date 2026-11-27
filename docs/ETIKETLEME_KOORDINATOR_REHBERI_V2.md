# Expanded-v2 İnsan Etiketleme — Koordinatör Rehberi

Bu belge koordinatör içindir. Etiketleyicilere yalnız **kılavuz kartı** (`docs/ETIKETLEYICI_KILAVUZ_KARTI_V2.md`) ve **kendi HTML formları** verilir.

## Kapsam (2 etiketleyici, yalın tasarım)

| Rol | Görev | Not |
|---|---:|---|
| Validation referansı | 384 | 96 ailenin tamamı, aile başına 2 profil × 2 ifade |
| Test referansı | 384 | 32 yeni cihazın tamamı |
| Train denetimi | 192 | 48 aile; inşa etiketlerinin hata oranını ölçer |
| Pilot | 40 | Yalnız train; A ve B'ye aynı görevler |
| Gizli tekrar | 24 | Oturum 1–2'deki görevler, oturum 6–7'de yeniden |
| **Etiketleyici başına toplam** | **1.024** | 8 oturum (pilot + 7) |
| B'ye ek: Human-60 ikinci etiketleme | 60 | Protokol 1.3; A'nın mevcut etiketleri B'ye **gösterilmez** |

## Dosyalar

| Dosya | Kime |
|---|---|
| `artifacts/semantic_annotation/expanded_v2/annotator_a/*.html` | Yalnız A |
| `artifacts/semantic_annotation/expanded_v2/annotator_b/*.html` (Human-60 formu dahil) | Yalnız B |
| `docs/ETIKETLEYICI_KILAVUZ_KARTI_V2.md` | A ve B |
| `data/semantic_benchmark/expanded_v2/annotation/blinding_key.coordinator_only.jsonl` | **Kimseyle paylaşılmaz** |
| `candidates.construction_labels.jsonl`, `split_manifest.candidate.json` | **Kimseyle paylaşılmaz** |

Formlar `artifacts/` altında olduğu için git'e girmez. Yeniden üretmek için:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts\prepare_expansion_v2_annotation.py
```

## Adımlar

1. **Hazırlık:** Gerekiyorsa etik/insan katılımcı onayını danışmanla teyit edin. Etiketleyicilerin arka planını yalnız toplu olarak not edin (eğitim düzeyi, teknik deneyim). İsimler veride tutulmaz.
2. **Eğitim (~1 saat):** Kılavuz kartını birlikte okuyun ve 2–3 örneği birlikte tartışın. Bu örnekler pilot veya ana paketlerden **seçilmez**.
3. **Pilot:** İki kişi de `pilot.html` formunu bağımsız doldurur. Gelen dosyaları şu adlarla `data/semantic_benchmark/expanded_v2/annotation/submissions/` klasörüne koyun: `annotator_a.pilot.jsonl`, `annotator_b.pilot.jsonl`. Ardından:
   ```powershell
   $env:PYTHONPATH='src'; .\.venv\Scripts\python.exe scripts\analyze_expansion_v2_annotation.py
   ```
   Herhangi bir alanda durum veya değer kappa'sı **< 0,70** ise yalnız pilot uyuşmazlıklarına bakarak kılavuz kartını netleştirin. Test/validation görevlerine bakılmaz. Kılavuzda yapılan değişikliği tarihle not edin.
4. **Ana oturumlar (1–7):** Günde en fazla bir oturum. Her teslimden sonra aynı analiz komutunu çalıştırın:
   - **Teslim hatası** çıkarsa (eksik alan, kanıt-sayı uyuşmazlığı) etiketleyiciden formu düzeltip yeniden indirmesini isteyin.
   - **Oturum bazlı kappa** belirgin düşerse yorgunluk veya kural kayması olabilir; kısa bir hatırlatma yapın. Ama tek tek görevler hakkında "doğru cevap" söylemeyin.
5. **Hakem kararı:** Tüm oturumlar bitince `reports/disagreements.<oturum>.jsonl` dosyalarındaki uyuşmazlıklar hakemle çözülür. Ham A/B dosyaları değiştirilmez. Uzlaşılamayan alan `ambiguous` kalır.
6. **Kapılar ve final:** Hakem çıktısından sonra:
   - altın etiket dosyası üretilir;
   - iki kişinin de "doğal değil" dediği metinler çıkarılır;
   - train denetim uyumu raporlanır;
   - bütün yöntemler tek komutla yeniden eğitilip puanlanır.

   Test etiketleri bu adıma kadar hiçbir ayar seçiminde kullanılmaz.

## Kapı ölçütleri (önceden sabit)

| Ölçüt | Eşik | Altında kalırsa |
|---|---|---|
| Alan başına A/B kappa (durum ve değer) | ≥ 0,70 (tercihen ≥ 0,80) | Alan raporda "düşük uyum" diye işaretlenir; hakem kararı zorunlu |
| Gizli tekrar öz-uyumu | ≥ 0,85 alan uyumu | O etiketleyicinin ilgili oturumları incelenir |
| Train denetiminde insan ile inşa etiketi uyumu | ≥ 0,90 alan uyumu | Train inşa etiketleri gözden geçirilir veya train'in bir kısmı etiketlenir |
| "İkisi de doğal değil" oranı | Raporlanır | Bu metinler bütün split'lerden çıkarılır |
