# Pilot insan annotasyonları

Bu klasör henüz insan etiketi içermez. `annotator_a` ve `annotator_b` isimleri
iki **gerçek ve bağımsız** kişiyi temsil eder; yapay zekâ çıktısı veya iki yapay
persona bu dosyaların yerine kullanılamaz.

İş akışı:

1. Corpus metinleri araştırmacı tarafından gözden geçirilip dondurulur.
2. Her kişi yalnız kendisine ait `../packets/annotator_*.v1.jsonl` paketini ve
   `docs/ANNOTATION_PROTOCOL.md` kılavuzunu görür.
3. Tamamlanan kayıtlar JSON Schema ve metin-içi evidence denetiminden geçer.
4. Ham A/B dosyaları SHA-256 ile kilitlendikten sonra yalnız anlaşmazlıklar
   adjudication'a açılır.
5. Gold dosyası ham dosyaların üstüne yazılmaz; ayrı üretilir.

Bu klasörde `annotator_a.jsonl`, `annotator_b.jsonl` veya
`adjudicated_gold.jsonl` görülmesi, tek başına gerçek insan etiketlemesinin
tamamlandığını kanıtlamaz. Manifest checksum'ları, annotator onayı ve anlaşma
raporu birlikte bulunmalıdır.
