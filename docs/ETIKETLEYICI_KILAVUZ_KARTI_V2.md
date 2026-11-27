# Etiketleyici Kılavuz Kartı — Semantik Görev Etiketleme (Protokol 2.0)

Bu kart, size verilen çevrimdışı etiketleme formlarını doldururken kullanacağınız kısa rehberdir. Ayrıntılı kurallar için `docs/ANNOTATION_PROTOCOL.md` belgesine bakabilirsiniz; bu kartla çelişen bir durum görürseniz koordinatöre sorun.

## Ne yapıyoruz?

Her görevde bir IoT cihazının yapacağı işi anlatan kısa bir Türkçe metin göreceksiniz. Sizden, **yalnızca metinde yazanlara dayanarak** sekiz alanı etiketlemeniz ve metnin ne kadar doğal olduğunu değerlendirmeniz isteniyor.

Doğru ya da yanlış "beklenen" bir cevap yoktur; bizim için önemli olan sizin metni nasıl okuduğunuzdur.

## Üç altın kural

1. **Metinde yoksa uydurmayın.** "Sağlık cihazı" demek, verinin gizli olduğu ya da buluta gönderilemeyeceği anlamına gelmez. Yazmıyorsa **Belirtilmemiş** seçin. Belirtilmemiş, hata değil doğru bir cevaptır.
2. **Metin kendi içinde çelişiyorsa Belirsiz / çelişkili seçin.** Örneğin: "Analiz yalnız bulutta yapılmalıdır. Aynı kaydın buluta gönderilmesi kesinlikle yasaktır."
3. **Bağımsız çalışın.** Diğer etiketleyiciyle görevleri konuşmayın. ChatGPT, Claude, çeviri aracı gibi hiçbir yapay zekâ aracı kullanmayın. Emin olmadığınızda güven düzeyini **Emin değilim** seçin.

## Sekiz alan — hızlı başvuru

| Alan | Ne arıyorsunuz? | Örnek → etiket |
|---|---|---|
| **Domain (alan)** | Görevin ait olduğu sektör | "Diyaliz makinesi monitörü…" → Sağlık |
| **Latency (gecikme)** | Sonucun ne kadar sürede gerekli olduğu. Sayı varsa sınıfı sayı belirler: <50 ms sert gerçek zamanlı; 50–200 ms gerçek zamanlı; 200–1000 ms etkileşimli; >1000 ms rahat. | "Sonuç en geç 80 ms içinde hazır olmalıdır." → Gerçek zamanlı, 80 ms |
| **Privacy (veri gizliliği)** | Verinin hassasiyeti; yalnız yazıyorsa | "Kullanılan veri kamuya açıktır." → Kamuya açık |
| **Execution policy (çalışma yeri)** | Cihaz / yakın edge / bulut için ayrı ayrı: izinli, yasak, bilinmiyor | "Cihaz veya yakın edge kullanılabilir; genel bulut kullanılamaz." → cihaz izinli, edge izinli, bulut yasak (üçü de biliniyor) |
| **Reliability (güvenilirlik)** | İsteklerin başarıyla tamamlanma oranı | "İsteklerin en az %99,9'u başarıyla tamamlanmalıdır." → Görev-kritik, 0,999 |
| **Accuracy (doğruluk)** | Sonucun doğruluğu; skor ve metrik yazıyorsa onları da girin | "Sonucun F1 skoru en az 0,95 olmalıdır." → Yüksek, 0,95, F1 |
| **Energy priority (enerji önceliği)** | Pil/enerji tasarrufu tercihi | "Pil tüketimini azaltmak birinci önceliktir." → Yüksek öncelik |
| **Divisibility (bölünebilirlik)** | Görevin parçalara ayrılıp paralel işlenip işlenemeyeceği | "Veri parçaları bağımsız ve paralel işlenebilir." → Bölünebilir |

İpuçları:
- **"Başarısız istek"** güvenilirliktir; **"yanlış sonuç"** doğruluktur.
- Yalnız bulut hakkında bir cümle varsa cihaz ve edge **bilinmiyor** kalır; durum **En az bir hedef bilinmiyor** olur.
- Gizlilik ile çalışma yeri ayrı alanlardır: "Buluta gönderme" yasağı tek başına veriyi "hassas" yapmaz.
- Bir alanı işaretlerken kararınızı destekleyen kısa metin parçasını **kanıt** kutusuna yapıştırın. Form, kanıttaki sayıyla girdiğiniz sayının uyuşmasını kontrol eder.

## Metin doğallığı (yeni soru)

Etiketlerden bağımsız olarak, metnin gerçek bir kullanıcının ya da kurumun yazabileceği bir görev açıklaması olup olmadığını değerlendirin:

- **Doğal:** Böyle bir görev açıklaması gerçekte yazılabilir.
- **Kabul edilebilir:** Biraz yapay ya da kalıp gibi, ama anlaşılır ve mantıklı.
- **Doğal değil:** Anlamsız cihaz-görev eşleşmesi (örneğin "robot süpürge baş hareketi işler"), bozuk Türkçe veya anlaşılmaz cümle.

Bilerek yazılmış çelişkili kısıtlar doğallığı düşürmez. Onları ilgili alanda **Belirsiz / çelişkili** olarak işaretlemeniz yeterli.

## Formu kullanma

1. Koordinatörün verdiği `.html` dosyasını Chrome, Edge veya Firefox ile açın. İnternet gerekmez.
2. Her görevde sekiz alanı ve doğallık sorusunu doldurun. **Bu görevi kontrol et** düğmesi eksikleri gösterir. **Boş alanları "belirtilmemiş" yap** düğmesini yalnız metni gerçekten okuduktan sonra kullanın.
3. Çalışmanız tarayıcıya otomatik kaydedilir; formu kapatıp aynı tarayıcıda yeniden açabilirsiniz. **Taslağı sıfırla** düğmesine basmayın.
4. Oturumun tamamı geçerli olunca **Tamamını JSONL indir** düğmesiyle bir `.jsonl` dosyası indirilir. Bu dosyayı koordinatöre gönderin.
5. Ekranda görev numarası ve `v2x_0123` gibi anlamsız bir kimlik görürsünüz. Kimlikler bilinçli olarak anlamsızdır.

## Oturum düzeni

| Oturum | Görev sayısı | Yaklaşık süre |
|---|---:|---:|
| Pilot | 40 | 45–60 dk |
| Oturum 1–7 | 137–149 | her biri ~2 saat |

- Günde **en fazla bir oturum** önerilir; yorgunken hata oranı artar. Bir oturumu iki parçada bitirebilirsiniz.
- Pilot oturumdan sonra kısa bir geri bildirim toplantısı yapılacak ve kurallardaki belirsizlikler netleştirilecek.
- Bazı görevler size tekrar gösterilebilir. Bunu fark etseniz bile, metni yeniden okuyup yeni baştan karar verin.

## Sorular

Bir kural belirsizse kendi yorumunuzu yapın, güveni **Emin değilim** seçin ve görev numarasını not edip koordinatöre iletin. Diğer etiketleyiciye sormayın.
