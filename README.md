# Task Offloading — New Architecture

Bu depo, MEC/IoT ortamında LLM ile çıkarılan semantik görev gereksinimlerinin DRL tabanlı task offloading üzerindeki katkısını sıfırdan ve bilimsel olarak doğrulanabilir biçimde incelemek için kurulmaktadır.

Başlangıç belgeleri:

- [Faz 0 eski depo denetimi ve literatür taraması](phase_reports/PHASE_0_LEGACY_AUDIT_AND_LITERATURE_REVIEW.md)
- [Faz 1 araştırma sözleşmesi](docs/RESEARCH_CONTRACT.md)
- [İki insanlı semantik annotasyon protokolü](docs/ANNOTATION_PROTOCOL.md)
- [Faz 1 kanıt raporu](phase_reports/PHASE_1_RESEARCH_CONTRACT.md)
- [Faz 2 simülasyon çekirdeği sözleşmesi](docs/SIMULATION_CORE.md)
- [Faz 2 kanıt raporu](phase_reports/PHASE_2_DETERMINISTIC_SIMULATION_CORE.md)
- [Veri yönetişimi ve split sözleşmesi](docs/DATA_GOVERNANCE.md)
- [Veri erişim talebi takibi](docs/DATA_ACCESS_REQUESTS.md)
- [Faz 3 kanıt raporu](phase_reports/PHASE_3_DATA_PROVENANCE_AND_SPLITS.md)
- [Faz 3A BUPT edinim raporu](phase_reports/PHASE_3A_BUPT_ACQUISITION.md)
- [Hesaplama donanımı envanteri](docs/COMPUTE_INVENTORY.md)
- [Faz bazlı görev listesi](task.md)
- [Bilimsel ve mimari sözleşme](TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md)

Mevcut durum: Faz 0–3 tamamlandı. Son doğrulamada 163/163 test, Ruff ve strict
mypy geçti; toplam branch coverage yüzde 92'dir. Gymnasium ile event replay aynı
fizik çekirdeğini kullanır ve parity testi geçmiştir. Faz 3'te veri manifesti,
alan kökeni sözlüğü, checksum/statü kapıları, train-only normalizer ve sızıntı
denetimli split sistemi kuruldu. BUPT'nin commit ile sabitlenmiş 482.687 satırlık
artifact'ı için checksum, şema profili ve mahremiyet-minimum adaptör doğrulandı.
UCI MEC 859'un 4.000 ölçümü ile EUA'nın commit-sabit 100.310 seçili koordinat
kaydı da `schema_validated` durumundadır. Erişimle alınan 5,89 GB
`Full_trace.7z`, 2026-09-18'de NEP-large/full olarak tanımlandı; SHA-256 ve tam
üye CRC taraması geçti, tam satır profili bekleniyor. Bütün raw artifact'lar
repo dışında `D:\task_offloading_datasets` altında tutulur; Git'e yalnız küçük
şema/checksum/profil kanıtları girer.
Sıradaki ana iş Faz 4 semantik benchmark'tır. Eski kaynak kod taşınmamıştır.

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
