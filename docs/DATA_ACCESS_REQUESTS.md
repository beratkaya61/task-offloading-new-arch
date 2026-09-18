# Veri Erişim Talebi Takibi

Bu belge erişim gerektiren veri kaynaklarının unutulmaması için tutulur. Ham
veri, başvuru formu yanıtları, kişisel bilgiler veya erişim bağlantıları Git'e
eklenmez.

## NEP-large / Full Trace

- Talep tarihi: **2026-09-14**
- Talebi gönderen: proje sahibi
- Erişim/alım tarihi: **2026-09-18**
- Durum: **erişim alındı, SHA-256 doğrulandı** (`checksum_verified`)
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
alanlarda tutulur. Tam CRC taraması geçmiştir; satır düzeyi şema/range profili
bitmeden kaynak `schema_validated` sayılmaz.

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
