# Task Offloading — New Architecture

Bu depo, MEC/IoT ortamında LLM ile çıkarılan semantik görev gereksinimlerinin DRL tabanlı task offloading üzerindeki katkısını sıfırdan ve bilimsel olarak doğrulanabilir biçimde incelemek için kurulmaktadır.

Başlangıç belgeleri:

- [Faz 0 eski depo denetimi ve literatür taraması](phase_reports/PHASE_0_LEGACY_AUDIT_AND_LITERATURE_REVIEW.md)
- [Faz 1 araştırma sözleşmesi](docs/RESEARCH_CONTRACT.md)
- [İki insanlı semantik annotasyon protokolü](docs/ANNOTATION_PROTOCOL.md)
- [Faz 1 kanıt raporu](phase_reports/PHASE_1_RESEARCH_CONTRACT.md)
- [Faz 2 simülasyon çekirdeği sözleşmesi](docs/SIMULATION_CORE.md)
- [Faz 2 kanıt raporu](phase_reports/PHASE_2_DETERMINISTIC_SIMULATION_CORE.md)
- [Hesaplama donanımı envanteri](docs/COMPUTE_INVENTORY.md)
- [Faz bazlı görev listesi](task.md)
- [Bilimsel ve mimari sözleşme](TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md)

Mevcut durum: Faz 0, Faz 1 ve Faz 2 tamamlandı. Son doğrulamada 50/50 test
(16 Faz 1 + 34 Faz 2), Ruff ve strict mypy geçti; branch coverage yüzde 88'dir.
Gymnasium ile event replay aynı fizik çekirdeğini kullanır ve parity testi
geçmiştir. Sıradaki tek iş Faz 3 veri kökeni ve split sistemidir. Eski kaynak kod
taşınmamıştır.

Kurulum ve bütün testler:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\mypy.exe
.\.venv\Scripts\python.exe -m pytest --cov=task_offloading --cov-branch --cov-report=term-missing -q
```
