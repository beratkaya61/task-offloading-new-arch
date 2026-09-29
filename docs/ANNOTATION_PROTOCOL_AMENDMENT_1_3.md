# Annotasyon Protokolü 1.3.0 Eki

**Yürürlük tarihi:** 2026-09-29  
**Önceki ek:** `docs/ANNOTATION_PROTOCOL_AMENDMENT_1_2.md`  
**Kapsam:** İnsan ana-corpus etiketlemesi başlamadan önce yapılan iş yükü düzeltmesi.

## Gerekçe

Annotator B'nin yeni ana-corpus görevi tamamlayacak zamanı yoktur. Gerçek bir ikinci
annotator yerine yapay zekâ personası kullanmak veya etiketleri iki insan üretmiş
gibi göstermek yasaktır. Bu nedenle insan emeği, veri toplanmadan önce azaltılır
ve veri kalitesi iddiaları aynı ölçüde daraltılır.

## Yürürlükteki tasarım

1. **Makine etiketli geliştirme — 180 görev:** Deterministik üretim kuralları zayıf
   etiket üretir. Bu kayıtlar insan doğrulamalı veya gold değildir. Eğitim, prompt
   geliştirme ve hata analizi için kullanılabilir.
2. **A değerlendirmesi — 60 görev:** Annotator A, makine önerisi görmeden 60 görevi
   bir kez etiketler. Bu kayıtlar `single_human_reference` olarak adlandırılır;
   `adjudicated_gold` değildir.
3. **Mevcut insan kanıtı — yeni görev yok:** Pilot Round 1'de A/B/C'nin, Round 2'de
   A/B'nin daha önce tamamladığı 20'şer görev korunur. C'nin Round 1 kayıtları
   pairwise sensitivity ve A/B uyuşmazlık teşhisinde kullanılır; otomatik çoğunluk
   oyu sayılmaz. B veya C'den yeni ana-corpus etiketi istenmez.

## İddia sınırları

- “İki annotator 60 görevi etiketledi” denemez.
- Ana 60 değerlendirmenin tamamı tek-insan referansıdır; gold değildir.
- 180 geliştirme kaydı insan etiketi değil, makine tarafından üretilmiş zayıf
  etikettir.
- Annotatorlar arası uyum yalnız mevcut pilot turlarında raporlanır; küçük örneklem,
  protokol sürümü ve sınıf prevalansı sınırlılık olarak yazılır.
- C'nin etiketi yeni ana görevlerin etiketi gibi yeniden kullanılamaz; çünkü C
  yalnız pilot metinlerini görmüştür.
- Gizli tekrar yapılmadığından annotator-içi tutarlılık ölçülemez.

## İş akışı

1. A, 60 görevlik kör formu tamamlar.
2. A dosyası şema, görev hash'i ve metin-içi kanıt kurallarıyla doğrulanıp kilitlenir.
3. Mevcut Round 1 A/B/C ve Round 2 A/B uyum kanıtları tezde ayrı raporlanır.
4. Pilot uyuşmazlıkları çözülecekse C yalnız tanısal üçüncü görüş olur; otomatik
   çoğunlukla insan gold üretilmez.

`1.3.0`, ana-corpus iş yükü ve gold kapsamı konusunda `1.2.0` hükümlerinin yerini
alır. Pilot Round 1, Round 2 ve hedefli kalibrasyonun tarihsel kayıtları değişmez.
