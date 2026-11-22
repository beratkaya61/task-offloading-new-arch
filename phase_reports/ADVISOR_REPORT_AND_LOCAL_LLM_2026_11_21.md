# Danışman Raporu, Güncel Literatür ve Yerel LLM Uygulanabilirliği — 21.11.2026

## Amaç

Bu çalışma adımının amacı, 20 Kasım 2026 Claude/Codex oturumunda üretilen fakat
eski `task.md` durumuna henüz yansımayan kanıtları doğrulamak; danışmana verilecek
mevcut durum belgesini yöntem, metrik, sonuç, sınırlama ve güncel literatürle
yeniden kurmak; bu bilgisayara uygun küçük LLM seçeneğini sonuçtan önce dondurulan
bir protokolle teknik olarak sınamaktı.

Bu adımın ilk sürümü Faz 4/5/6'yı resmen kapatmıyordu. Son tutarlılık denetiminde
Faz 4 ana benchmark'ı ayrı kapanış raporuyla tamamlandı; bağımsız paraphrase Faz 7
sağlamlık genişletmesine taşındı. Faz 5/6 kapanışları için NEP-v2, calibration,
LLM-semantikli uçtan uca koşul, eksik sistem metrikleri, Holm correction ve ayrı
faz raporları gereklidir.

## Bağlam ve kanıt denetimi

- `TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md`, `task.md`,
  `chat_histories/20_11_2026_chat.md`, son commit'ler ve mevcut çalışma ağacı
  birlikte incelendi.
- Yol haritasının E5 ve PPO durumunun 14 Ekim düzeyinde kaldığı doğrulandı.
- Dondurulmuş ana kanıt olarak `phase_reports/SEMANTIC_RESULTS_V3.md`,
  `phase_reports/OFFLOADING_RESULTS_V1.md`, `results.v3.json`,
  `results_llm.json` ve `nep_quantiles_v1.json` kullanıldı.
- Eski/sızıntılı paraphrase v1/v2 ara çıktıları danışman raporuna kanıt olarak
  alınmadı.
- Sistem sonucu `v1 ön sonuç`, insan etiketi `single_human_reference`, fizik
  `trace-driven hybrid` olarak sınırlandırıldı; gold/saha testi iddiası yapılmadı.

## Güncel literatür denetimi

21 Kasım 2026'ya kadar birincil kaynaklar üzerinden STEPS, AgentVNE, LeDRL,
COMLLM, Jev, IoT-Brain, MAS-DecStream, privacy-aware MARL ve güçlü heuristic/LLM
kontrol karşılaştırması yeniden incelendi. Mimari/yöntem kararları ayrıca PPO,
invalid action masking, calibration ve E5 birincil çalışmalarına bağlandı.

Bu denetim sonucunda ana yenilik iddiası şu şekilde daraltıldı:

> İlk LLM-assisted edge scheduling iddiası değil; izole ve dondurulmuş doğal dil
> semantik sözleşmesinin, privacy/execution action mask ve belirsizlik fallback'i
> aracılığıyla aynı MEC olaylarındaki başarı/hard-ihlal etkisini extractor
> kalitesi boyunca ölçen yeniden üretilebilir deney tasarımı.

## Danışman belgesi

`TEZ_DANISMANI_MEVCUT_DURUM_VE_LITERATUR_OZETI.md` baştan yazıldı ve aynı adla
DOCX üretildi. Belge şunları içerir:

- öğrenci ağzından araştırma problemi ve tek cümlelik ana sonuç;
- tasarlanan semantik/fizik izolasyon mimarisi;
- 240 ana metin, 1.924 artırılmış eğitim kaydı, 600+600 kompozisyonel görev ve
  188 bağımsız paraphrase dahil veri kökeni;
- BUPT, NEP, UCI ve EUA'nın birbirine karıştırılmayan kullanım rolleri;
- her semantik ve offloading yönteminin kullanım gerekçesi ve sonuçtan önceki
  beklenen rolü;
- macro-F1, exact record, privacy P/R/F1, abstention, ECE, latency, başarı,
  hard ihlal, deadline, enerji, collapse ve istatistik metriklerinin gerekçesi;
- doğrulanmış semantik ve 3.840 karar/5 seed offloading v1 tabloları;
- hipotez desteklenme tablosu, negatif sonuçlar, literatür karşılaştırma matrisi,
  dürüst sınırlamalar ve makale yol haritası;
- metin içinde numaralı atıflar ve 26 kaynaklık birincil ağırlıklı kaynakça.

### Üretilen şekiller

`scripts/build_advisor_report_figures.py` ile üç yeniden üretilebilir PNG üretildi:

1. `docs/advisor_assets/research_architecture.png`
2. `docs/advisor_assets/semantic_macro_f1.png`
3. `docs/advisor_assets/offloading_success_and_violations.png`

Şekiller görsel olarak incelendi; etiketler, oranlar ve yöntem sırası phase
raporlarındaki dondurulmuş değerlerle karşılaştırıldı. DOCX paketinin üç PNG'yi de
`word/media` altında içerdiği doğrulandı.

## Yerel küçük LLM smoke deneyi

Sonuçlar görülmeden önce `docs/EXPLORATORY_IMPROVEMENT_PROTOCOL_V1.md` içine iki
yerel aday ve revision'ları donduruldu:

- `Qwen/Qwen3-0.6B`, revision
  `c1899de289a04d12100db370d81485cdf75e47ca`
- `utter-project/EuroLLM-1.7B-Instruct`, revision
  `a25c7fa65fc2a644e6270b8940dbe295b51da681`

Qwen3 için chat template'te `enable_thinking=False` desteği eklendi. Model,
revision, prompt/parser ve çıktı dosya adları çok-model karşılaştırmasında
çakışmayacak biçimde sürümlendi.

Qwen3-0.6B üç özgün örnek smoke sonucu:

| Ölçüm | Sonuç |
|---|---:|
| Örnek | 3 |
| Macro-F1 | %50,0 |
| Tam kayıt | %0,0 |
| Parse edilemeyen çıktı | 0 |
| Sözlük dışı alan | 0 |
| Ortalama inference | 289.060 ms/görev |
| Model + ortak prefix hazırlığı | yaklaşık 253 sn |

Üç örnek kalite karşılaştırması veya güven aralığı iddiası için yetersizdir.
Sonucun geçerli yorumu yalnız şudur: pinned Qwen3 non-thinking teknik yol çalışır,
fakat bu CPU'da tam 60/120 kayıtlı deney operasyonel olarak uygulanabilir değildir.
EuroLLM tam koşusu da süre tahmini kabul edilebilir olmadan başlatılmayacaktır.
Quadro Qwen3-4B/8B ana yolu bloke etmeyen opsiyonel genişletmedir.

## Kod ve bağımlılık değişiklikleri

- `src/task_offloading/semantic/llm.py`: revision pin'i ve Qwen3 non-thinking chat
  template seçeneği.
- `scripts/run_llm_baseline.py`: model/revision ortam değişkeni, model-adlı sonuç
  dosyaları ve görünür ilerleme çıktısı.
- `tests/phase_5/test_llm_coded_output.py`: Qwen3 ve diğer model template testleri.
- `requirements.txt` ve `pyproject.toml`: doğrudan `torch>=2.7,<3` bağımlılığı.
- `matplotlib` requirements ile kuruldu; danışman şekil üretim scripti eklendi.
- Kaynak strict mypy'deki iki artık `type: ignore` kaldırıldı.
- Beş eski test dosyasındaki 48 test-only annotation hatası gerçek veri tipleri,
  `ClassVar`, `cast`, `Scenario` ve numpy dönüşümleriyle temizlendi; test davranışı
  değiştirilmedi.

## Yol haritası eşitlemesi

- `task.md` içinde E5, Qwen2.5 ve offloading v1 tamamlandı olarak işaretlendi.
- Faz 4 ana benchmark'ı `PHASE_4_SEMANTIC_BENCHMARK.md` ile kapatıldı; bağımsız
  188 paraphrase Faz 7 sağlamlık genişletmesine taşındı. Faz 5/6 kapanışları açık.
- `TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md` içine tarihsel bölümleri
  değiştirmeden 20–21 Kasım kanıt eşitlemesi ve güncel kapanış hedefi eklendi.
- Sıradaki tek iş NEP-v2 + bağımsız sağlamlık + development-only kalibrasyon +
  Holm-düzeltilmiş ana kontrastlar olarak güncellendi.

## Doğrulama kapıları

21 Kasım 2026 son kontrolünde:

- `pytest -q`: **245 passed**
- `ruff check .`: **geçti**
- `mypy src tests`: **69 dosyada geçti**
- danışman DOCX yapısal metin parity: Markdown paragraf ve tablo hücreleriyle eşit
- DOCX medya denetimi: üç şekil gömülü

## Açık bilimsel işler

1. NEP quantile profilini nihai offloading v2'ye bağlamak.
2. Bağımsız 188 paraphrase değerlendirmesini tamamlamak.
3. Calibration/fallback seçimini yalnız development verisinde dondurmak.
4. Gürültülü önizleme veya non-stationarity altında PPO–greedy headroom'unu ölçmek.
5. Ön kayıtlı ana kontrastlara Holm düzeltmesi uygulamak.
6. Faz 5, 6 ve 7 için ayrı kapanış raporlarını ve nihai veri/model kartını üretmek.

Bu adımda commit atılmadı; commit kullanıcı tarafından manuel yapılacaktır.
