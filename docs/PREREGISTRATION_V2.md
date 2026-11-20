# Ön Kayıt (Pre-registration) v2 — Ek Deneyler

**Yazılma zamanı:** 2026-11-20, aşağıdaki deneylerin hiçbiri çalıştırılmadan önce.
**Kapsam:** Bu belgede listelenen deneyler ve karar kuralları, sonuçlar görülmeden sabitlenmiştir. Sonuçlar ne çıkarsa çıksın bu kurallarla raporlanacaktır; desteklenmeyen hipotezler de raporlanır.

## 0. Bu belgeden önce bilinenler (şeffaflık)

Aşağıdaki sonuçlar bu belgeden **önce** görülmüştür ve bu belgedeki hipotezleri doğal olarak etkilemiştir:

- Semantik benchmark v3.1 (kural, TF-IDF, E5 frozen, E5 fine-tuned; insan referanslı 60 ve kompozisyonel 600 set).
- PPO v1 değerlendirmesi (UCI/BUPT/EUA ile hibrit ortam, maske tabanlı semantik PPO ve semantiksiz PPO, greedy aileleri).
- Qwen2.5-1.5B-Instruct ile 3 görevlik duman testi (yalnız teknik doğrulama).

Bu nedenle S1, S2, H1 ve H2 **doğrulayıcı** (confirmatory) değil, **yeni koşullarda tekrar** (replication under new conditions) niteliğindedir. S3, S4, H3, H4, H5 ve H6 yeni koşullardır.

## 1. Yeni koşullar

1. **LLM çıkarıcı:** Qwen2.5-1.5B-Instruct, 4 örnekli few-shot, greedy decoding; örnekler yalnız 180 geliştirme görevinden. Hesaplama bütçesi nedeniyle kompozisyonel görülmemiş ifade setinde, sonuçlar görülmeden sabitlenen tohumla (seed 20261120) seçilmiş **200 görevlik alt küme** kullanılır. Bu alt kümede tüm çıkarıcılar karşılaştırılır.
2. **Bağımsız yeniden ifade seti:** Gereksinim cümleleri, eğitim havuzunu görmeyen bir LLM tarafından serbestçe yeniden yazılır. Sayısal değerleri değişen veya boş kalan yeniden yazımlar otomatik olarak elenir; elenen oran raporlanır. Bu set makine üretimidir ve etiketleri değişmezlik varsayımıyla aktarılır.
3. **Güven kalibrasyonu:** Her öğrenen çıkarıcı için alan başına isotonic regression; geliştirme bölümünün alan cümlelerinden yeni bir tohumla üretilmiş, hiçbir modelin eğitiminde kullanılmamış kompozisyonel **kalibrasyon seti** üzerinde öğrenilir. Test etiketleri kullanılmaz.
4. **Ceza tabanlı PPO (literatür tasarımı):** Semantik gözlem açık, semantik maske kapalı; yasak eylem yalnız ödülde 20 birimlik ceza ile caydırılır.
5. **Gürültülü önizleme (gerçekçi bilgi):** Hem PPO'nun gözlemindeki hem de greedy'nin kullandığı gecikme/enerji önizlemeleri, her karar ve eylem için bağımsız log-normal çarpanla bozulur (σ = 0,5, tohumlu). Fiziksel sonuç gürültüsüzdür; yalnız karar vericinin bilgisi gürültülüdür.
6. **NEP katmanı:** Her bölümde edge sunucu arka plan CPU yükü NEP `VM_CPU.cpu_rate` dağılımından, bulut tek yön gecikmesi NEP `SITE_RTT.rtt` dağılımından örneklenir. Edge etkin hızı `cpu_hz × (1 − yük)` olur. Bu katman sonuçlar görülmeden eklenir; PPO bu ortamda yeniden eğitilir.

## 2. Hipotezler ve karar kuralları

Karar kuralı: eşleştirilmiş bootstrap (görev veya bölüm düzeyinde, 2000 tekrar) %95 güven aralığı belirtilen yönde sıfırı dışarıda bırakıyorsa **desteklendi**. Aynı ailedeki karşılaştırmalar için Holm düzeltmeli p-değerleri raporlanır (α = 0,05).

### Semantik çıkarım

- **S1 (dayanıklılık, tekrar):** TF-IDF(+veri), bağımsız yeniden ifade setinde kuraldan daha yüksek macro-F1 elde eder.
- **S2 (eşdeğerlik, tekrar):** İnsan referanslı orijinal 60 görevde TF-IDF(+veri) − kural macro-F1 farkının %95 GA alt sınırı −5 puandan büyüktür (non-inferiority, marj 5 puan).
- **S3 (LLM):** LLM few-shot, (a) insan referanslı yeniden yazım setinde kuraldan daha yüksek macro-F1 elde eder; (b) görev başına çıkarım süresi kuralın en az 100 katıdır. LLM ile TF-IDF(+veri) arasında yön beklentisi **yoktur**; yalnız raporlanır.
- **S4 (kalibrasyon):** Kalibrasyon, kompozisyonel görülmemiş ifade setinde E5 fine-tuned ve E5 frozen(+veri) çıkarıcıların ECE değerini düşürür.

### Sistem (offloading)

Birincil metrikler: başarı oranı ve kesin kısıt ihlali oranı. Test: 60 bölüm × 64 görev, kompozisyonel görülmemiş ifade havuzu.

- **H1 (semantik katkı, tekrar):** Referans semantikle hem PPO hem greedy, semantiksiz karşılıklarından daha yüksek başarı elde eder.
- **H2 (hata yayılımı, tekrar):** Görülmemiş ifadede kural çıkarıcıyla kesin ihlal oranı, TF-IDF(+veri) çıkarıcıyla elde edilenden yüksektir (PPO ve greedy için ayrı ayrı).
- **H3 (kalibre fallback):** Kalibre güven skorlarıyla fallback, TF-IDF(+veri), E5 frozen(+veri) ve E5 fine-tuned çıkarıcıların her biri için kesin ihlali azaltır ve başarıyı 2 puandan fazla düşürmez (başarı farkı GA alt sınırı > −2 puan).
- **H4 (maske ve ceza):** Referans semantikle maske tabanlı PPO'nun kesin ihlal oranı, ceza tabanlı PPO'dan düşüktür; başarı farkının GA alt sınırı −1 puandan büyüktür.
- **H5 (gürültülü bilgi altında DRL):** Gürültülü önizleme koşulunda, referans semantikle PPO'nun başarısı greedy'den düşük değildir (fark GA alt sınırı > −1 puan). **Bu hipotezin desteklenmemesi olasıdır ve sonuç her durumda raporlanır.**
- **H6 (NEP ile tutarlılık):** NEP katmanı eklendiğinde H1 ve H2'nin yönü korunur.

## 3. Değiştirilmeyecekler

Ödül ağırlıkları, eylem uzayı, bölüm uzunluğu, PPO hiperparametreleri, kalibrasyon dışı çıkarıcı ayarları ve test bölümlerinin tohumu bu deneyler için değiştirilmez. Bir hata bulunursa düzeltme ve etkisi ayrıca raporlanır.
