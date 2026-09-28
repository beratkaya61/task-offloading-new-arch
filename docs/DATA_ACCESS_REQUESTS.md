# Veri Erişim Talebi Takibi

Bu belge erişim gerektiren veri kaynaklarının unutulmaması için tutulur. Ham
veri, başvuru formu yanıtları, kişisel bilgiler veya erişim bağlantıları Git'e
eklenmez.

## NEP-large / Full Trace

- Talep tarihi: **2026-09-14**
- Talebi gönderen: proje sahibi
- Erişim/alım tarihi: **2026-09-18**
- Durum: **erişim, checksum, CRC ve tam satır profili doğrulandı**
  (`schema_validated`)
- Resmî başvuru: <https://forms.gle/j3QDp9qtCVyrcTwm9>
- Resmî iletişim: `mwx@bupt.edu.cn`
- Artifact: `Full_trace.7z`, **5.885.170.081 byte**
- SHA-256:
  `ff80a07e25f8055fed1a64439194a47721d9bc2330c79c9a8201649072852816`
- Kimlik: **NEP-large/full**; 7.410 VM ve 139 VM sitesi, ayrıca üç aylık
  bant genişliği dosyası.
- Kullanım kararı: CPU, bant genişliği, site RTT ve kapasite katmanlarında ana
  `trace-driven-hybrid` kaynağı olarak kullanılacak.
- Saklama kuralı: yalnız yerel araştırma kullanımı; ham dosya Git'e veya başka
  kişilere gönderilmeyecek.

Erişim izninin özel yazışması Git'e konmaz. Kullanıcının izin aldığı beyanı,
resmî research-only/offline paylaşım yasağı ve yerel artifact kanıtı ayrı
alanlarda tutulur. Tam CRC taraması ve 685.506.639 satırın profil taraması
geçmiştir. Kabul politikası v1.0.0, değerleri ve PM-VM-site soyu geçerli
592.724.139 satırı kabul eder; 92.782.500 satırı onarmadan reddeder.

## BUPT Edge Computing Dataset

- İndirme tarihi: **2026-09-14**
- Sabit sürüm: Git commit
  `04e664fab9cdb2a58d04ebc615cd74405e6062e2`
- Durum: **yerel akademik kullanıma açıldı** (`schema_validated`)
- Kullanım dayanağı: sabit README, verinin Edge Computing araştırmaları için
  herkese açık yayımlandığını bildiriyor ve ilgili makaleye atıf istiyor.
- Muhafazakâr sınır: yalnız yerel, ticari olmayan tez araştırması; ham veya satır
  seviyesinde türetilmiş veri yeniden dağıtılmayacak.
- Açık iş: yeniden dağıtım/türev paylaşımı konusunda yazılı açıklama almak hâlâ
  tercih edilir fakat yerel analiz için başlangıç engeli değildir.

Ham URL, IP, User-Agent, telefon/cihaz kimliği semantik corpus'a veya model
girdisine taşınmaz. İşlenmiş cihaz ve istasyon kimlikleri proje-yerel anahtarlı
HMAC-SHA256 ile üretilir.
