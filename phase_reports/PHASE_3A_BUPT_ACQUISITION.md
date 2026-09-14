# Faz 3A — BUPT Veri Edinimi ve Güvenli Alım Raporu

**Tarih:** 2026-09-14

**Durum:** BUPT artifact edinildi, bütünlük ve güvenli şema kapıları doğrulandı;
manuel commit bekleniyor

## Amaç

BUPT mobil ağ izini tekrar üretilebilir sabit bir sürümden edinmek, ham veriyi
Git dışında tutmak ve eğitimde kullanılabilecek alanları mahremiyet sınırıyla
ayırmaktır.

## Kaynak ve kullanım kararı

Resmî repo, ISP kaynaklı verinin Edge Computing araştırmaları için herkese açık
yayımlandığını söyler ve Liu, Li ve Wang'ın 2022 IEEE Internet of Things Journal
makalesine atıf ister. Proje bu beyanı yalnız **yerel, ticari olmayan akademik
araştırma** kapsamı olarak yorumlar. Genel bir yeniden dağıtım lisansı olmadığı
için ham veri ve satır seviyesinde türev veri yayımlanmaz. Bu bir hukuk görüşü
değildir; yazılı açıklama alınması hâlâ tercih edilir.

## Artifact kanıtı

- Commit: `04e664fab9cdb2a58d04ebc615cd74405e6062e2`
- Dosya: `Edge-Computing-Dataset-04e664f...zip`
- Sıkıştırılmış boyut: 28.964.429 byte (27,62 MiB)
- ZIP girdisi: 102
- Açılmış toplam boyut: 133.919.441 byte (127,72 MiB)
- SHA-256:
  `5d89bd853a5300207892dad38070957b935a95ea02bfe1ec063e0c4829e21822`
- Ham konum: `data/raw/...`; `.gitignore` ile izlenmez

## Veri profili

- 93 başlıksız CSV
- 482.687 toplam satır
- Beijing: 96.650; Guangzhou: 311.942; Shanghai: 74.095
- UTF-8 strict decode başarısız dosya: 0
- Kaynak içinde önceden bulunan replacement karakteri: 3.759
- Güvenli ilk 18 alanı geçen satır: 476.419 (%98,70)
- Reddedilen satır: 6.268 (%1,30)

Reddedilenlerin 6.258'inde başlangıç/bitiş zamanı bozuk; ayrıca 10 geçersiz
destination IP ve 1 geçersiz source IP görüldü. Aynı satır birden fazla nedenle
reddedilebildiği için neden sayıları reddedilen satır toplamına eklenmez.
Gerçek adaptör fail-fast taramasında ilk hata dağılımı 6.258 başlangıç zamanı,
9 destination IP ve 1 source IP olarak gerçekleşti; toplam yine 6.268'dir.

## Güvenli ayrıştırma kararı

Dosyalar header içermez. Serbest URL/User-Agent alanları tırnaksız virgül
içerebildiği için bütün satırı normal CSV kolon sayısına göre bölmek güvenli
değildir. `data/bupt.py` en fazla 18 bölme yapar, kararlı sayısal prefix'i okur ve
kalan serbest metni hiç döndürmez.

İşlenmiş kayıtta yalnız şunlar bulunur: kaynak satır kimliği, şehir,
pseudonymous cihaz/istasyon, geliş zamanı, süre, input/output bit, RAT,
destination port ve status code. Ham telefon/IMEI/IP/URL/User-Agent tutulmaz.

## Eğitimde kullanım sınırı

- BUPT sayısal izleri DRL görev gelişleri, trafik boyutları, cihaz/istasyon
  grupları ve privacy-safe uygulama türetimleri için kullanılabilir.
- Ham URL metni LLM eğitimi veya prompt girdisi değildir.
- 6.268 bozuk satır sessizce onarılmaz; deterministik olarak reddedilir.
- Train/validation/test split cihaz, istasyon, uygulama veya zaman grubuna göre
  veri sızıntısı denetiminden sonra yapılır.
- Normalizer yalnız train split'ine fit edilir.

## Açık işler

- Arşivden streaming işlenmiş tablo üretimi ve rejection log özeti.
- Uygulama kategorisi için yalnız privacy-safe metadata kullanan dondurulmuş
  eşleme.
- UCI ve EUA artifact edinimi.
- NEP-small erişim yanıtının takibi.
- BUPT yeniden dağıtım/türev paylaşımı için tercihen yazılı açıklama.

## Kabul kanıtı

- BUPT safe-prefix/privacy testleri: geçti.
- Faz 3 veri katmanı testleri: **74/74 geçti.**
- Tüm proje testleri: **124/124 geçti.**
- BUPT adaptörü branch coverage: **yüzde 100.**
- Toplam proje branch coverage: **yüzde 91.**
- Ruff: temiz.
- Strict mypy: 17 source dosyasında hata yok.
- `git diff --check`: temiz; yalnız Windows LF→CRLF bilgilendirme uyarıları var.
