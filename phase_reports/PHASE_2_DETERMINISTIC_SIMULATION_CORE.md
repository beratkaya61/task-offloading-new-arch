# Faz 2 — Deterministik Simülasyon Çekirdeği Raporu

**Tarih:** 2026-09-14

**Faz durumu:** Tamamlandı; kullanıcı incelemesi ve manuel commit bekleniyor

**Sürüm kontrolü:** Bu fazda commit veya push yapılmadı

## 1. Amaç

Faz 2'nin amacı, ileride baseline, oracle ve DRL politikalarının tamamının aynı
MEC fiziğinde değerlendirilebilmesi için küçük, tipli, deterministik ve test
edilebilir tek bir geçiş çekirdeği kurmaktı. Eski repodaki kod taşınmadı. Eski
hatalar—özellikle belirsiz fizik tuple'ları, adaptörler arasında denklem
tekrarı, seed belirsizliği ve kanıtsız simülasyon çıktıları—yeni testlerin
hedefine dönüştürüldü.

## 2. Kontrol edilen girdiler

- `TODO_ANTIGRAVITY_TASK_OFFLOADING_UPGRADE.md`
- `task.md`
- `docs/RESEARCH_CONTRACT.md`
- `configs/research_contract.json`
- `phase_reports/PHASE_0_LEGACY_AUDIT_AND_LITERATURE_REVIEW.md`
- `phase_reports/PHASE_1_RESEARCH_CONTRACT.md`
- Eski repodaki `AgentVNE.pdf` ve Faz 0'da kaydedilmiş bulgular
- Kullanıcının yeni bilgisi: ikinci makinede NVIDIA Quadro RTX 4000 bulunuyor

Faz 1 kapsamı değişmedi: whole-task `local/edge_i/cloud`, olay indeksli karar,
LLM'nin fizik ve eylemden izolasyonu, trace-driven-hybrid terminolojisi ve
negatif normalize maliyet reward'u korundu.

## 3. Uygulanan mimari

### 3.1 Paket ve kalite araçları

- `src/` tabanlı kurulabilir Python paketi oluşturuldu.
- Runtime bağımlılıkları `requirements.txt`, geliştirme/test bağımlılıkları
  `requirements-dev.txt` içine ayrıldı.
- `pyproject.toml` içinde setuptools, pytest, branch coverage, Ruff ve strict
  mypy ayarlandı.
- Bilimsel çekirdeğin runtime bağımlılıkları yalnız NumPy ve Gymnasium'dur;
  Faz 1 şema testleri için jsonschema korunmuştur.

### 3.2 Birimli ve değişmez domain tipleri

`domain/models.py` içinde bütün fizik alanlarının birimi adına yazıldı:
`input_bits`, `compute_cycles`, `cpu_hz`, `battery_j`, `available_at_s`,
`uplink_rate_bps` gibi. Negatif, sonsuz, NaN ve geçersiz olasılıklar nesne
oluşturulurken reddedilir. Nesneler frozen dataclass olduğu için bir geçiş eski
durumu yerinde değiştiremez.

Ana tipler: `Task`, `TaskRequirements`, `DeviceSpec/State`,
`ServerSpec/State`, `LinkMetrics`, `SystemState`, `ExogenousEvent`,
`OffloadOutcome`, `RewardComponents` ve `TransitionResult`.

### 3.3 Tek fizik geçişi

`sim/core.py::transition` şu hesapların tek kaynağıdır:

- log-distance/Shannon tabanlı link snapshot'larının tüketimi,
- local/edge/cloud uçtan uca gecikmesi,
- mutlak sunucu boşalma zamanı ile FCFS queue,
- cihaz local CPU veya radio/idle enerjisi,
- batarya nedenselliği,
- bilinen hedef kapanması ile sürpriz exogenous failure ayrımı,
- beklenen reliability ve accuracy constraint'leri,
- deadline/soft/hard/failure reward ayrıştırması,
- bir sonraki geliş zamanına durum geçişi.

Detaylı denklem ve davranış sözleşmesi `docs/SIMULATION_CORE.md` içindedir.

### 3.4 İki ince adaptör

- `EventSimulationRunner`, sabit scenario ve eylem dizisini replay eder.
- `TaskOffloadingEnv`, aynı scenario'yu Gymnasium arayüzünde sunar.

İki adaptörde fizik denklemi yoktur; ikisi de doğrudan `transition` çağırır.
Gymnasium ortamı güncel beşli `step` dönüşünü, `super().reset(seed=seed)`
kalıbını, sonlu observation-space sınırlarını ve `info.action_mask` değerini
kullanır.

### 3.5 Determinizm

Rastgele gerçekleşmeler core içinde çekilmez; `ExogenousEvent` olarak dışarıdan
verilir. `RngFactory` kök seed + sabit namespace hash + NumPy `SeedSequence`
kullanır. Böylece arrival, failure veya mobility akışlarını çağırma sırası
mevcut akışları değiştirmez. Aynı task/state/action/event/config girdisi bit-bit
eşit domain sonucu üretir.

## 4. Test matrisi

Phase 2 için şu kanıt türleri eklendi:

| Sınıf | Doğrulama |
|---|---|
| Unit/domain | Negatif/sonsuz/geçersiz değer reddi, kimlik/link bütünlüğü, immutable state |
| Oracle | Local, edge ve cloud için elle hesaplanan gecikme, enerji ve queue değerleri |
| Property | Mesafe artınca rate düşer; güç/bant genişliği artınca kapasite artar |
| Invariant | Gecikme/enerji sonlu ve negatif değil; batarya negatife düşmez; FCFS gerilemez |
| Determinizm | Aynı girdiler aynı geçişi; aynı adlandırılmış seed aynı diziyi üretir |
| Isolation | Bilinmeyen semantik mask üretmez; sürpriz failure maskeye sızmaz |
| Ablation | Maskeyi kapatmak hard violation veya kapalı hedef fiziğini silmez |
| Parity | Gym ve event adaptörlerinin tüm geçiş/final state değerleri birebir eşit |
| API | Resmi Gymnasium `check_env` denetimi |
| Regression | Faz 1'in 16 sözleşme/şema testi korunur |

## 5. Çalıştırılan komutlar ve sonuç

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\mypy.exe
.\.venv\Scripts\python.exe -m pytest --cov=task_offloading --cov-branch --cov-report=term-missing -q
```

Son temiz doğrulama:

- **50/50 test geçti** (16 Faz 1 + 34 Faz 2), uyarı yok.
- **Ruff:** bütün kontroller geçti.
- **mypy strict:** 11 source dosyasında hata yok.
- **Branch coverage toplamı:** yüzde 88.
- **Gym/event parity:** tüm transition kayıtları ve final state eşit.

## 6. Başarısızlıklar ve düzeltmeler

1. İlk bağımlılık kurulumu 120 saniyede zaman aşımına uğradı; NumPy, Ruff ve
   bazı paketler kurulmuştu. Aynı requirements dosyası idempotent olarak yeniden
   çalıştırıldı ve eksik Gymnasium/pytest-cov tamamlandı.
2. İlk test turu 45/45 geçti fakat resmi Gym checker sonsuz Box sınırları için
   iki uyarı verdi. Observation-space alt/üst sınırları sonlu ve alan anlamına
   uygun hâle getirildi; uyarısız tekrar test edildi.
3. İlk lint turu biçim ve satır uzunluğu sorunları buldu. Mekanik format ve küçük
   kod sadeleştirmeleri sonrası Ruff temiz geçti.
4. Kod incelemesinde mask-ablation açıkken yasak/kapalı eylemin yanlışlıkla
   geçerli fizik gibi çalışabileceği görüldü. Maske görünürlüğü ile fiziksel/hard
   geçerlilik ayrıldı ve iki regresyon testi eklendi.
5. Cloud sunucu kuyruğuna varış zamanına tek yön propagasyon eklendi. Toplam
   gecikmede gidiş-dönüş propagation korunurken FCFS başlangıcı da fiziksel varış
   anını kullanır hâle geldi.

## 7. Bilimsel dayanak ve dış sözleşmeler

- Mao, Zhang ve Letaief'in MEC modeli local CPU için `kappa*W*f^2` enerji
  formunu ve gecikme/failure ortak maliyetini destekler:
  <https://arxiv.org/abs/1605.05488>
- Gymnasium `Env` API'si:
  <https://gymnasium.farama.org/api/env/>
- Gymnasium ortam denetleyicisi:
  <https://gymnasium.farama.org/api/utils/#gymnasium.utils.env_checker.check_env>
- NumPy `SeedSequence`:
  <https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html>
- Quadro RTX 4000 üretici veri sayfası 8 GB GDDR6 bilgisini doğrular:
  <https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/quadro-rtx-4000-datasheet-us-nvidia-1060942-r2-web.pdf>

## 8. Sınırlamalar ve yapılmayan işler

- Faz 2 test başarısı tez hipotezlerinin doğrulandığı anlamına gelmez; yalnız
  çekirdeğin tanımlanan örnek ve özelliklerde tutarlı olduğunu gösterir.
- Henüz gerçek/trace veri indirilmedi, eşlenmedi veya kalibre edilmedi.
- Kanal snapshot'ı interference ve çok kullanıcılı kaynak paylaşımını içermez.
- FCFS dışında scheduling, preemption veya batching yoktur.
- FCFS sırası offloading admission sırasıdır; data-ready-time overtaking ve
  gecikmiş reward modeli bu ilk çekirdekte yoktur.
- Cihaz uplink/radio erişimi için ayrı serialization kuyruğu yoktur.
- Sunucu/datacenter enerjisi henüz objective'e dahil değildir.
- Başarı olasılıklarının bağımsız yön çarpımı ilk model varsayımıdır.
- Reward normalizasyon referansları henüz training split'inden fit edilmemiştir.
- LLM, semantic corpus, iki-insan etiketi ve PPO bu fazın dışında tutuldu.
- Quadro RTX 4000 üzerinde model benchmark'ı çalıştırılmadı; donanım yalnız
  kullanıcı bildirimi + üretici spec'i olarak kaydedildi.

## 9. Faz kabul kontrolü

- [x] Paketleme ve minimal runtime/dev bağımlılık ayrımı yapıldı.
- [x] Lint, strict type-check, test ve branch coverage araçları kuruldu.
- [x] Birimli, adlandırılmış, immutable domain tipleri oluşturuldu.
- [x] Kanal, gecikme, FCFS kuyruk, cihaz enerjisi ve failure tek core'da yazıldı.
- [x] Gymnasium ve event adaptörleri aynı core'a bağlandı.
- [x] Seed/RNG ve exogenous-event sözleşmesi uygulandı.
- [x] Unit, property/invariant, oracle, determinism, mask ve parity testleri geçti.
- [x] Test sonucu, sorunlar, sınırlamalar ve sonraki riskler raporlandı.

## 10. Sonraki faz ve riskler

Sıradaki tek iş **Faz 3 — veri kökeni ve split sistemi**dir:

1. dataset manifest şeması ve checksum/lisans statüleri,
2. her kolon için observed/measured/derived/matched/generated/labeled/simulated
   provenance,
3. raw veriyi Git dışında tutan indir/doğrula/cache hattı,
4. training-only normalizer,
5. kronolojik ve entity/application holdout split'leri,
6. duplicate ve leakage testleri,
7. sentetik ile trace-driven-hybrid benchmark'ın açık ayrımı.

Ana riskler BUPT açık lisansının hâlâ doğrulanmamış olması, NEP-small erişiminin
başvuru gerektirmesi ve birbirinden bağımsız kaynakların sahte ortak satırlar
olarak birleştirilmesi tehlikesidir. Bu üç risk çözülmeden veri “gerçek ve
doğrulanmış” sayılmayacaktır.
