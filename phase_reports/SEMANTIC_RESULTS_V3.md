# Semantik Çıkarım Benchmark'ı v3 — Yayın Tabloları

Değerler yüzde olarak verilmiştir; köşeli parantez içinde görev düzeyinde 1000 tekrarlı bootstrap ile %95 güven aralığı (GA) yer alır. *Fark* sütunu, aynı bootstrap örnekleri üzerinde kural tabanlı yönteme göre eşleştirilmiş farkı, %95 GA'sını ve iki yönlü bootstrap p-değerini gösterir. Değerlendirme etiketleri eğitim veya ayar için kullanılmamıştır.

Etiket kaynakları: insan referansı = protokol-normalize tek-insan referansı (yalnız orijinal 60 metin); yeniden ifade seti = makine üretimi metin + aktarılmış insan etiketi; kompozisyonel = makine üretimi, inşa yoluyla etiketli. Toplam gerçek insan etiketli metin: 60 ana + 20 pilot. *(+veri)* = 180 görev + eğitim havuzundan yeniden ifade + kompozisyonel eğitim görevleri (toplam 1924).

## İnsan referansı (n=60), orijinal ifade

| Yöntem | Eğitim n | Tam kayıt doğruluğu | Macro-F1 | Macro-F1 farkı (vs kural) | Kritik gizlilik P / R / F1 (n+) | Abstention F1 | ECE ↓ | ms/görev |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Sabit profil | 180 | 3.3 [0.0, 8.3] | **25.3 [25.1, 29.7]** | -52.8 [-57.2, -42.3]; p=<0.001 | 0.0 / 0.0 / 0.0 (7) | 93.0 | 0.037 | 0.01 |
| Kural/anahtar kelime | 180 | 38.3 [26.7, 51.7] | **78.0 [70.9, 83.8]** | — | 75.0 / 42.9 / 54.5 (7) | 95.7 | 0.227 | 0.16 |
| TF-IDF + LR | 180 | 30.0 [20.0, 41.7] | **76.6 [69.4, 82.0]** | -1.4 [-2.6, -0.5]; p=0.002 | 75.0 / 42.9 / 54.5 (7) | 95.7 | 0.167 | 1.58 |
| TF-IDF + LR (+veri) | 1924 | 38.3 [26.7, 51.7] | **78.2 [71.1, 83.9]** | +0.2 [+0.0, +0.5]; p=0.254 | 75.0 / 42.9 / 54.5 (7) | 95.7 | 0.132 | 1.58 |
| E5-small donuk + LR | 180 | 33.3 [21.7, 45.0] | **75.8 [68.3, 81.1]** | -2.3 [-4.4, -0.8]; p=<0.001 | 60.0 / 42.9 / 50.0 (7) | 94.9 | 0.408 | 18.97 |
| E5-small donuk + LR (+veri) | 1924 | 36.7 [25.0, 48.4] | **76.3 [69.0, 80.8]** | -1.8 [-5.5, -0.3]; p=0.002 | 75.0 / 42.9 / 54.5 (7) | 94.8 | 0.236 | 17.32 |
| E5-small ince ayar (+veri) | 1924 | 10.0 [3.3, 18.3] | **70.2 [63.2, 75.8]** | -7.8 [-8.8, -6.7]; p=<0.001 | 75.0 / 42.9 / 54.5 (7) | 95.7 | 0.300 | 13.45 |

Alan bazında macro-F1 (%):

| Yöntem | domain | latency | privacy | execution_policy | reliability | accuracy | energy_priority | divisibility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sabit profil | 3.0 | 28.9 | 22.5 | 15.5 | 31.3 | 23.2 | 30.6 | 47.4 |
| Kural/anahtar kelime | 73.2 | 100.0 | 77.9 | 60.4 | 87.9 | 57.3 | 87.2 | 80.4 |
| TF-IDF + LR | 61.8 | 100.0 | 77.9 | 60.4 | 87.9 | 57.3 | 87.2 | 80.4 |
| TF-IDF + LR (+veri) | 74.6 | 100.0 | 77.9 | 60.4 | 87.9 | 57.3 | 87.2 | 80.4 |
| E5-small donuk + LR | 69.1 | 100.0 | 76.3 | 60.4 | 75.4 | 57.3 | 87.2 | 80.4 |
| E5-small donuk + LR (+veri) | 71.3 | 100.0 | 77.9 | 60.4 | 80.6 | 56.8 | 87.2 | 76.0 |
| E5-small ince ayar (+veri) | 10.4 | 100.0 | 77.9 | 60.4 | 87.9 | 57.3 | 87.2 | 80.4 |

## Makine yeniden yazımı, insan etiketi aktarılmış (n=60) — metinler otomatik yeniden ifade edildi; insan bu metinleri görmedi, etiketler anlam korunduğu varsayımıyla orijinalden aktarıldı

| Yöntem | Eğitim n | Tam kayıt doğruluğu | Macro-F1 | Macro-F1 farkı (vs kural) | Kritik gizlilik P / R / F1 (n+) | Abstention F1 | ECE ↓ | ms/görev |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Sabit profil | 180 | 3.3 [0.0, 8.3] | **25.3 [25.1, 29.7]** | -15.0 [-17.8, -12.6]; p=<0.001 | 0.0 / 0.0 / 0.0 (7) | 93.0 | 0.037 | 0.01 |
| Kural/anahtar kelime | 180 | 23.3 [13.3, 35.0] | **40.3 [38.4, 45.9]** | — | 0.0 / 0.0 / 0.0 (7) | 94.3 | 0.150 | 0.09 |
| TF-IDF + LR | 180 | 25.0 [15.0, 36.7] | **48.1 [40.9, 54.4]** | +7.8 [-1.8, +13.5]; p=0.144 | 100.0 / 14.3 / 25.0 (7) | 93.6 | 0.222 | 1.50 |
| TF-IDF + LR (+veri) | 1924 | 23.3 [13.3, 33.4] | **67.9 [59.3, 72.8]** | +27.6 [+16.6, +32.0]; p=<0.001 | 60.0 / 42.9 / 50.0 (7) | 94.4 | 0.092 | 1.49 |
| E5-small donuk + LR | 180 | 20.0 [10.0, 30.0] | **51.8 [47.4, 58.3]** | +11.5 [+4.2, +16.5]; p=0.002 | 0.0 / 0.0 / 0.0 (7) | 94.0 | 0.390 | 17.43 |
| E5-small donuk + LR (+veri) | 1924 | 26.7 [16.7, 38.3] | **60.4 [55.5, 67.2]** | +20.1 [+12.3, +26.0]; p=<0.001 | 0.0 / 0.0 / 0.0 (7) | 94.3 | 0.231 | 18.84 |
| E5-small ince ayar (+veri) | 1924 | 8.3 [1.7, 15.0] | **62.3 [55.1, 68.6]** | +22.0 [+11.6, +28.3]; p=<0.001 | 60.0 / 42.9 / 50.0 (7) | 94.5 | 0.288 | 17.79 |

Alan bazında macro-F1 (%):

| Yöntem | domain | latency | privacy | execution_policy | reliability | accuracy | energy_priority | divisibility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sabit profil | 3.0 | 28.9 | 22.5 | 15.5 | 31.3 | 23.2 | 30.6 | 47.4 |
| Kural/anahtar kelime | 73.2 | 65.3 | 22.5 | 15.5 | 44.9 | 23.2 | 30.6 | 47.4 |
| TF-IDF + LR | 74.6 | 52.6 | 29.8 | 37.7 | 44.6 | 40.1 | 47.5 | 58.0 |
| TF-IDF + LR (+veri) | 74.6 | 87.2 | 48.1 | 39.9 | 76.5 | 70.1 | 66.3 | 80.4 |
| E5-small donuk + LR | 70.4 | 68.8 | 22.5 | 33.7 | 53.3 | 57.3 | 30.3 | 78.4 |
| E5-small donuk + LR (+veri) | 71.4 | 100.0 | 22.4 | 49.3 | 65.0 | 54.7 | 42.4 | 78.1 |
| E5-small ince ayar (+veri) | 19.1 | 94.5 | 62.3 | 60.4 | 81.3 | 57.3 | 43.1 | 80.4 |

## Kompozisyonel (n=600), şablon ifadesi — yalnız doğruluk kontrolü; test cümleleri kural ve eğitim verisiyle aynı şablondan, kanıt değildir

| Yöntem | Eğitim n | Tam kayıt doğruluğu | Macro-F1 | Macro-F1 farkı (vs kural) | Kritik gizlilik P / R / F1 (n+) | Abstention F1 | ECE ↓ | ms/görev |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Sabit profil | 180 | 0.8 [0.2, 1.7] | **20.7 [20.5, 21.0]** | -77.8 [-78.3, -77.2]; p=<0.001 | 0.0 / 0.0 / 0.0 (127) | 85.2 | 0.092 | 0.01 |
| Kural/anahtar kelime | 180 | 93.3 [91.2, 95.3] | **98.5 [97.9, 99.0]** | — | 100.0 / 100.0 / 100.0 (127) | 100.0 | 0.254 | 0.08 |
| TF-IDF + LR | 180 | 31.3 [27.7, 34.8] | **68.4 [66.8, 69.6]** | -30.1 [-31.8, -28.8]; p=<0.001 | 64.2 / 33.9 / 44.3 (127) | 91.0 | 0.197 | 1.87 |
| TF-IDF + LR (+veri) | 1924 | 97.8 [96.7, 99.0] | **99.6 [99.2, 99.9]** | +1.1 [+0.5, +1.7]; p=<0.001 | 100.0 / 100.0 / 100.0 (127) | 100.0 | 0.184 | 1.94 |
| E5-small donuk + LR | 180 | 26.7 [23.2, 30.0] | **60.9 [59.2, 62.4]** | -37.6 [-39.4, -36.0]; p=<0.001 | 66.2 / 41.7 / 51.2 (127) | 89.7 | 0.372 | 17.51 |
| E5-small donuk + LR (+veri) | 1924 | 67.0 [63.2, 70.8] | **86.8 [85.3, 88.2]** | -11.7 [-13.3, -10.3]; p=<0.001 | 98.3 / 89.8 / 93.8 (127) | 98.3 | 0.323 | 17.57 |
| E5-small ince ayar (+veri) | 1924 | 33.3 [29.7, 37.2] | **87.5 [86.6, 88.3]** | -11.0 [-12.1, -10.0]; p=<0.001 | 69.8 / 100.0 / 82.2 (127) | 96.9 | 0.371 | 15.18 |

Alan bazında macro-F1 (%):

| Yöntem | domain | latency | privacy | execution_policy | reliability | accuracy | energy_priority | divisibility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sabit profil | 2.3 | 14.0 | 21.0 | 16.8 | 21.0 | 21.2 | 27.6 | 42.1 |
| Kural/anahtar kelime | 96.8 | 100.0 | 100.0 | 100.0 | 91.2 | 100.0 | 100.0 | 100.0 |
| TF-IDF + LR | 95.5 | 36.0 | 45.8 | 87.9 | 55.7 | 58.1 | 84.4 | 83.7 |
| TF-IDF + LR (+veri) | 100.0 | 97.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| E5-small donuk + LR | 96.5 | 41.3 | 45.6 | 80.6 | 39.1 | 52.3 | 65.8 | 66.1 |
| E5-small donuk + LR (+veri) | 98.4 | 67.2 | 90.5 | 96.9 | 69.9 | 77.0 | 98.6 | 96.3 |
| E5-small ince ayar (+veri) | 42.9 | 86.2 | 86.4 | 93.9 | 99.4 | 91.8 | 99.5 | 99.8 |

## Kompozisyonel (n=600), görülmemiş ifade

| Yöntem | Eğitim n | Tam kayıt doğruluğu | Macro-F1 | Macro-F1 farkı (vs kural) | Kritik gizlilik P / R / F1 (n+) | Abstention F1 | ECE ↓ | ms/görev |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Sabit profil | 180 | 0.8 [0.2, 1.7] | **20.7 [20.5, 21.0]** | -26.8 [-27.7, -25.8]; p=<0.001 | 0.0 / 0.0 / 0.0 (127) | 85.2 | 0.092 | 0.17 |
| Kural/anahtar kelime | 180 | 16.5 [13.7, 19.5] | **47.5 [46.5, 48.5]** | — | 0.0 / 0.0 / 0.0 (127) | 87.8 | 0.079 | 0.09 |
| TF-IDF + LR | 180 | 15.0 [12.2, 17.8] | **36.7 [35.4, 37.8]** | -10.9 [-12.4, -9.5]; p=<0.001 | 100.0 / 3.1 / 6.1 (127) | 85.6 | 0.122 | 1.59 |
| TF-IDF + LR (+veri) | 1924 | 37.0 [33.3, 40.5] | **73.6 [71.5, 75.2]** | +26.0 [+23.9, +27.8]; p=<0.001 | 100.0 / 72.4 / 84.0 (127) | 92.8 | 0.130 | 1.61 |
| E5-small donuk + LR | 180 | 17.7 [14.5, 20.5] | **44.9 [43.2, 46.2]** | -2.7 [-4.5, -1.1]; p=0.004 | 57.1 / 6.3 / 11.3 (127) | 87.3 | 0.318 | 18.51 |
| E5-small donuk + LR (+veri) | 1924 | 33.0 [29.5, 36.8] | **70.9 [69.1, 72.4]** | +23.4 [+21.4, +25.2]; p=<0.001 | 95.4 / 48.8 / 64.6 (127) | 94.9 | 0.264 | 19.11 |
| E5-small ince ayar (+veri) | 1924 | 20.5 [17.7, 23.7] | **73.0 [71.4, 74.4]** | +25.5 [+23.6, +27.1]; p=<0.001 | 62.3 / 100.0 / 76.7 (127) | 93.5 | 0.296 | 16.37 |

Alan bazında macro-F1 (%):

| Yöntem | domain | latency | privacy | execution_policy | reliability | accuracy | energy_priority | divisibility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sabit profil | 2.3 | 14.0 | 21.0 | 16.8 | 21.0 | 21.2 | 27.6 | 42.1 |
| Kural/anahtar kelime | 96.8 | 82.8 | 21.0 | 16.8 | 65.7 | 27.5 | 27.6 | 42.1 |
| TF-IDF + LR | 99.1 | 15.2 | 24.3 | 29.3 | 21.6 | 24.6 | 32.4 | 47.1 |
| TF-IDF + LR (+veri) | 100.0 | 69.9 | 72.5 | 62.8 | 76.3 | 46.6 | 62.1 | 98.3 |
| E5-small donuk + LR | 95.6 | 31.5 | 26.5 | 46.0 | 31.6 | 45.4 | 27.6 | 54.7 |
| E5-small donuk + LR (+veri) | 98.7 | 74.0 | 61.2 | 71.6 | 67.4 | 59.4 | 45.8 | 88.9 |
| E5-small ince ayar (+veri) | 44.1 | 88.5 | 70.3 | 78.2 | 92.4 | 63.3 | 48.2 | 99.2 |

