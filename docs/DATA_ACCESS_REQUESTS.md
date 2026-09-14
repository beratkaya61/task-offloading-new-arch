# Veri Erişim Talebi Takibi

Bu belge erişim gerektiren veri kaynaklarının unutulmaması için tutulur. Ham
veri, başvuru formu yanıtları, kişisel bilgiler veya erişim bağlantıları Git'e
eklenmez.

## NEP-small

- Talep tarihi: **2026-09-14**
- Talebi gönderen: proje sahibi
- Durum: **yanıt bekleniyor** (`access_pending`)
- Resmî başvuru: <https://forms.gle/j3QDp9qtCVyrcTwm9>
- Resmî iletişim: `mwx@bupt.edu.cn`
- İlk takip tarihi: **2026-09-21**
- İkinci takip tarihi: **2026-09-28**
- Kullanım kararı: erişim verilirse CPU, bant genişliği, site RTT ve kapasite
  katmanlarında ana `trace-driven-hybrid` kaynağı olarak kullanılacak.
- Saklama kuralı: yalnız yerel araştırma kullanımı; ham dosya Git'e veya başka
  kişilere gönderilmeyecek.

Yanıt gelirse tarih, erişim koşulu ve sağlanan artifact adı/checksum manifestte
kaydedilir. 2026-09-21'e kadar yanıt gelmezse resmî iletişim adresine kısa ve
nazik bir takip e-postası gönderilir.

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
