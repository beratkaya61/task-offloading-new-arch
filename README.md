# Task Offloading — New Architecture

Bu depo, MEC/IoT ortamında LLM ile çıkarılan semantik görev gereksinimlerinin DRL tabanlı task offloading üzerindeki katkısını sıfırdan ve bilimsel olarak doğrulanabilir biçimde incelemek için kurulmaktadır.

Başlangıç belgeleri:

- [Faz 0 eski depo denetimi ve literatür taraması](phase_reports/PHASE_0_LEGACY_AUDIT_AND_LITERATURE_REVIEW.md)
- [Faz 1 araştırma sözleşmesi](docs/RESEARCH_CONTRACT.md)
- [İki insanlı semantik annotasyon protokolü](docs/ANNOTATION_PROTOCOL.md)
- [Faz 1 kanıt raporu](phase_reports/PHASE_1_RESEARCH_CONTRACT.md)
- [Faz bazlı görev listesi](task.md)
- [Bilimsel ve mimari sözleşme](TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md)

Mevcut durum: Faz 0 ve Faz 1 tamamlandı; Faz 1'de 16/16 sözleşme/şema testi geçti. Sıradaki tek iş Faz 2 deterministik simülasyon çekirdeğidir. Henüz eski kaynak kod taşınmamıştır.

Faz 1 testleri:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
