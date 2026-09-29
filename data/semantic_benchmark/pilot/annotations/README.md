# Pilot insan annotasyonları

Bu klasör, checksum ile dondurulmuş insan etiketi turlarını içerir.
`round_1/`, 20 görevlik ilk pilotta iki **gerçek ve bağımsız** birincil kişi olan
`annotator_a` ve `annotator_b` ile kör tanısal üçüncü kişi `annotator_c` tarafından
üretilen ham dosyaları saklar. C, birincil A/B anlaşmasını veya gold kararı
otomatik olarak değiştirmez.

İş akışı:

1. Corpus metinleri araştırmacı tarafından gözden geçirilip dondurulur.
2. Her kişi yalnız kendisine ait `../packets/annotator_*.v1.jsonl` paketini ve
   `docs/ANNOTATION_PROTOCOL.md` kılavuzunu görür.
3. Tamamlanan kayıtlar JSON Schema ve metin-içi evidence denetiminden geçer.
4. Ham A/B dosyaları SHA-256 ile kilitlendikten sonra yalnız anlaşmazlıklar
   adjudication'a açılır.
5. Gold dosyası ham dosyaların üstüne yazılmaz; ayrı üretilir.

Bu klasörde `annotator_a.jsonl`, `annotator_b.jsonl`, `annotator_c.jsonl` veya
`adjudicated_gold.jsonl` görülmesi, tek başına gerçek insan etiketlemesinin
tamamlandığını kanıtlamaz. Manifest checksum'ları, annotator onayı ve anlaşma
raporu birlikte bulunmalıdır.

`round_1/submission_manifest.v1.json` ham girdileri; `round_1/agreement.v1.json`
ise bu girdilerden deterministik üretilen A/B birincil ve C tanısal ölçümlerini
bağlar. Round 1 henüz adjudicate edilmemiştir ve gold değildir.
