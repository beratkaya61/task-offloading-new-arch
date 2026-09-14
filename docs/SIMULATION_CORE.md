# Faz 2 Simülasyon Çekirdeği Sözleşmesi

## 1. Neden tek çekirdek var?

Bir görevin local, edge veya cloud üzerinde çalıştırılmasının fiziksel sonucu yalnız
`task_offloading.sim.core.transition` fonksiyonunda hesaplanır. Gymnasium ortamı
ve olay tabanlı replay çalıştırıcısı bu fonksiyonu çağırır. Böylece bir algoritma
için farklı, başka bir algoritma için farklı fizik hesabı oluşamaz.

Fonksiyonun girdileri şunlardır:

1. karar anındaki değişmez `SystemState`,
2. o anda gelen bölünmez `Task`,
3. seçilen `ActionTarget`,
4. politika tarafından görülmeyen, önceden örneklenmiş `ExogenousEvent`,
5. ortak `PhysicsConfig`.

Çıktı `TransitionResult` içinde yeni durum, ayrıntılı fiziksel sonuç, reward
bileşenleri ve toplam reward olarak döner. Girdi nesneleri değiştirilmez.

## 2. Zaman ve kuyruk modeli

Bir karar, her görev gelişinde alınır. Durumdaki `now_s` ile görevin
`arrival_time_s` değeri aynı olmak zorundadır. Sunucu FCFS kuyruğu mutlak
`available_at_s` değeriyle temsil edilir:

```text
compute_arrival = now + upload + one_way_backhaul
compute_start   = max(compute_arrival, server.available_at)
queue_wait      = compute_start - compute_arrival
new_available   = compute_start + compute_time
```

Sonraki karar zamanı, önceden örneklenmiş `next_interarrival_s` kadar ilerler.
Bir iş henüz bitmemiş olsa bile sonraki görev gelebilir; mutlak
`available_at_s` değeri bu örtüşmeyi kaybetmez.

Bu ilk modelde kuyruk disiplini **offloading admission sırasına göre FIFO**'dur:
kabul edilen remote görev, upload sürerken de sunucudaki yerini rezerve eder ve
sonradan gelen görev onu geçemez. Bu, karar anında outcome'u kesinleştirmeyi
sağlayan açık bir sadeleştirmedir; veri sunucuya ulaştığı sıraya göre overtaking
ve gecikmiş reward modeli ileride ayrı bir ablation gerektirir.

## 3. Gecikme modeli

- Local: `queue_wait + compute_cycles / device_cpu_hz`
- Edge: `upload + queue_wait + compute + download`
- Cloud: `upload + backhaul_upload + propagation_gidiş + queue_wait + compute +
  propagation_dönüş + backhaul_download + download`

Cloud kuyruğuna varış hesabı da tek yön propagasyonu içerir. Bu ayrıntı,
sunucunun dolu olduğu örneklerde kuyruk beklemesini doğru hesaplamak için
gereklidir.

## 4. Kanal modeli

`channel.py`, 1 metre referanslı log-distance path loss ve Shannon kapasite
formülünü kullanır:

```text
PL(d) = PL(d0) + 10*n*log10(d/d0) + shadowing
noise = -174 dBm/Hz + 10*log10(B) + noise_figure
rate  = B*log2(1 + SNR_linear)
```

Uplink ve downlink güçleri ayrı hesaplanır. Sonuçlar sırası karışabilecek bir
tuple yerine `LinkMetrics` içindeki `uplink_rate_bps`, `downlink_rate_bps`,
`snr_db`, `distance_m` ve `path_loss_db` alanlarıyla taşınır.

## 5. Enerji modeli

Local dinamik CPU enerjisi:

```text
E_local = kappa * compute_cycles * cpu_hz^2 + idle_power * queue_wait
```

Remote cihaz enerjisi:

```text
E_remote = tx_power * upload
         + idle_power * (backhaul + queue_wait + compute + propagation)
         + rx_power * download
```

Bu aşamada sunucu/datacenter enerjisi değil, kaynak IoT cihazının enerjisi
optimize edilir. Batarya yetersizse değer negatife düşürülmez; görev
`battery_depleted` ile tamamlanamaz.

## 6. Güvenilirlik, başarısızlık ve üç farklı sonuç

- `accepted`: hedef görevi kabul etti mi?
- `completed`: fiziksel işlem tamamlandı mı?
- `success`: tamamlandı ve deadline/reliability/accuracy/hard constraint
  koşullarının tamamı sağlandı mı?

Beklenen uçtan uca başarı olasılığında son kilometre uplink ve downlink için
iki kez, cloud backhaul da iki yön için iki kez kullanılır. Bu, yönlerin
bağımsız olduğu ilk-model varsayımıdır ve ileride trace kalibrasyonuyla
sınanacaktır. Gerçekleşen arıza ayrıca `ExogenousEvent` içindedir; çekirdek
kendi içinde rastgele sayı çekmez. Böylece iki politika aynı arıza dizisinde
karşılaştırılabilir.

Bilinen kapalı hedef action mask'e girer. Karar sonrasında gerçekleşen link veya
sunucu arızası maskeye sızmaz. Maskeyi ablation amacıyla kapatmak fiziksel olarak
kapalı hedefi çalışır hâle getirmez; eylem görünür olur ama başarısızlık kaydı
korunur.

## 7. Semantik sınır

Çekirdek LLM çağırmaz ve metin yorumlamaz. Yalnız daha önce üretilmiş tipli
`TaskRequirements` değerlerini kullanır. Metinde belirtilmeyen bir izin bütün
eylemleri açık bırakır; `unknown` hiçbir zaman kendiliğinden hard mask üretmez.
Açık execution policy yasağı ise maskelenir. Maske ablation'ında yasak eylem
çalıştırılabilir fakat hard-constraint cezası kaybolmaz.

## 8. Reward

Faz 1 sözleşmesi doğrudan uygulanır:

```text
reward = -(1*latency_norm
         + 1*energy_norm
         + 5*deadline_miss
         + 5*soft_constraint_violation
         + 10*failure_or_drop
         + 20*hard_constraint_violation)
```

Queue beklemesi toplam gecikmenin içindedir; ayrıca ikinci kez cezalandırılmaz.
Normalizasyon referansları şimdilik `PhysicsConfig` girdisidir. Faz 3'ten sonra
yalnız training split'ine fit edilip validation/test için dondurulacaktır.

## 9. RNG ve adaptör sözleşmesi

`RngFactory(root_seed).stream("arrivals")` gibi adlandırılmış NumPy akışları
kullanılır. İsim, SHA-256 ile sabit kelimelere çevrilip `SeedSequence` girdisine
katılır. Akış isteme sırası veya yeni bir akış eklemek mevcut dizileri değiştirmez.

`EventSimulationRunner` bütün episode'u doğrudan replay eder.
`TaskOffloadingEnv` aynı geçişi Gymnasium `reset -> (observation, info)` ve
`step -> (observation, reward, terminated, truncated, info)` arayüzüne sarar.
Parity testi iki yolun tüm `TransitionResult` ve final state değerlerinin birebir
eşit olduğunu doğrular.

## 10. Bilinen sınırlar

- Şu anda binary/whole-task offloading vardır; partial offloading yoktur.
- CPU frekansı, bant genişliği ve transmit power eylem değildir.
- FCFS tek kuyruk modeli kullanılır; preemption ve batching yoktur.
- Remote kuyruk admission-order FIFO'dur; data-ready-time sıralaması ve
  overtaking yoktur.
- Cihaz radio erişimi için ayrı bir serialization kuyruğu yoktur.
- Interference/SINR paylaşımı henüz yoktur; link snapshot'ları dışarıdan gelir.
- Path-loss ve başarı bağımsızlığı ilk sentetik modeldir; gerçek/trace
  kalibrasyonu Faz 3 ve Faz 7 işidir.
- Failure detection süresi basitleştirilmiş ortak parametredir.
- Gerçek veri, LLM, insan etiketi, PPO ve sonuç iddiası bu fazda yoktur.

## 11. Dayanaklar

- Y. Mao, J. Zhang, K. B. Letaief, “Dynamic Computation Offloading for
  Mobile-Edge Computing with Energy Harvesting Devices”, local CPU için
  `kappa*W*f^2` enerji modelini ve gecikme/enerji/failure ortak maliyetini
  kullanır: <https://arxiv.org/abs/1605.05488>
- Gymnasium güncel `Env` sözleşmesi:
  <https://gymnasium.farama.org/api/env/>
- Gymnasium resmi environment checker:
  <https://gymnasium.farama.org/api/utils/#gymnasium.utils.env_checker.check_env>
- NumPy `SeedSequence` tekrar üretilebilir bağımsız akış yaklaşımı:
  <https://numpy.org/doc/stable/reference/random/bit_generators/generated/numpy.random.SeedSequence.html>
