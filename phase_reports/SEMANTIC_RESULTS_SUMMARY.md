# Semantik Çıkarım Sonuçları — Özet Tablo

Değerlendirme: 60 kilitli görev, protokol-normalize tek-insan referansı (`annotator_a.protocol_normalized.v1.3.1`). *Yeniden ifade* seti, aynı 60 görevin anlamı koruyan otomatik (makine üretimi) yeniden yazımıdır; eğitim çoğaltmasında kullanılan ifadelerden ayrık bir havuzdan gelir. Değerlendirme etiketleri eğitim veya ayar için kullanılmamıştır.

Çoğaltma: görev başına 4 yeniden ifade varyantı (yalnız eğitim).

## Orijinal ifade — genel metrikler

| Yöntem | Eğitim örneği | Tam kayıt | Alan doğruluğu | Macro-F1 | Kritik gizlilik P/R/F1 | Abstention F1 | ECE ↓ | Brier ↓ | Süre (ms/görev) |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| Sabit profil | 180 | %3.3 | %75.8 | **%25.3** | %0.0 / %0.0 / %0.0 | %93.0 | 0.037 | 0.127 | 0.03 |
| Kural/anahtar kelime | 180 | %38.3 | %90.8 | **%78.0** | %75.0 / %42.9 / %54.5 | %95.7 | 0.227 | 0.129 | 0.10 |
| TF-IDF + LR | 180 | %30.0 | %89.0 | **%76.6** | %75.0 / %42.9 / %54.5 | %95.7 | 0.167 | 0.129 | 1.57 |
| TF-IDF + LR + çoğaltma | 844 | %36.7 | %90.4 | **%77.8** | %75.0 / %42.9 / %54.5 | %95.7 | 0.080 | 0.111 | 1.48 |
| E5-small (donuk) + LR | 180 | %33.3 | %89.0 | **%75.8** | %60.0 / %42.9 / %50.0 | %94.9 | 0.408 | 0.269 | 15.39 |
| E5-small (donuk) + LR + çoğaltma | 844 | %33.3 | %89.2 | **%75.9** | %50.0 / %42.9 / %46.2 | %94.8 | 0.227 | 0.148 | 15.39 |
| E5-small ince ayar + çoğaltma | 844 | %11.7 | %82.9 | **%69.4** | %75.0 / %42.9 / %54.5 | %94.8 | 0.301 | 0.206 | 14.79 |

## Orijinal ifade — alan bazında macro-F1

| Yöntem | domain | latency | privacy | execution_policy | reliability | accuracy | energy_priority | divisibility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sabit profil | %3.0 | %28.9 | %22.5 | %15.5 | %31.3 | %23.2 | %30.6 | %47.4 |
| Kural/anahtar kelime | %73.2 | %100.0 | %77.9 | %60.4 | %87.9 | %57.3 | %87.2 | %80.4 |
| TF-IDF + LR | %61.8 | %100.0 | %77.9 | %60.4 | %87.9 | %57.3 | %87.2 | %80.4 |
| TF-IDF + LR + çoğaltma | %71.6 | %100.0 | %77.9 | %60.4 | %87.9 | %57.3 | %87.2 | %80.4 |
| E5-small (donuk) + LR | %69.1 | %100.0 | %76.3 | %60.4 | %75.4 | %57.3 | %87.2 | %80.4 |
| E5-small (donuk) + LR + çoğaltma | %71.3 | %100.0 | %74.9 | %60.4 | %75.4 | %57.3 | %87.2 | %80.4 |
| E5-small ince ayar + çoğaltma | %21.1 | %100.0 | %77.9 | %60.4 | %85.4 | %57.3 | %72.4 | %80.4 |

## Yeniden ifade — genel metrikler

| Yöntem | Eğitim örneği | Tam kayıt | Alan doğruluğu | Macro-F1 | Kritik gizlilik P/R/F1 | Abstention F1 | ECE ↓ | Brier ↓ | Süre (ms/görev) |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| Sabit profil | 180 | %3.3 | %75.8 | **%25.3** | %0.0 / %0.0 / %0.0 | %93.0 | 0.037 | 0.127 | 0.01 |
| Kural/anahtar kelime | 180 | %23.3 | %86.5 | **%40.3** | %0.0 / %0.0 / %0.0 | %94.3 | 0.150 | 0.142 | 0.07 |
| TF-IDF + LR | 180 | %25.0 | %85.6 | **%48.1** | %100.0 / %14.3 / %25.0 | %93.6 | 0.222 | 0.156 | 1.57 |
| TF-IDF + LR + çoğaltma | 844 | %30.0 | %87.7 | **%62.6** | %100.0 / %28.6 / %44.4 | %94.3 | 0.123 | 0.114 | 1.54 |
| E5-small (donuk) + LR | 180 | %20.0 | %84.6 | **%51.8** | %0.0 / %0.0 / %0.0 | %94.0 | 0.390 | 0.274 | 15.21 |
| E5-small (donuk) + LR + çoğaltma | 844 | %21.7 | %86.0 | **%61.7** | %33.3 / %14.3 / %20.0 | %94.4 | 0.231 | 0.160 | 15.62 |
| E5-small ince ayar + çoğaltma | 844 | %11.7 | %80.4 | **%64.2** | %44.4 / %57.1 / %50.0 | %93.6 | 0.298 | 0.222 | 14.92 |

## Yeniden ifade — alan bazında macro-F1

| Yöntem | domain | latency | privacy | execution_policy | reliability | accuracy | energy_priority | divisibility |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sabit profil | %3.0 | %28.9 | %22.5 | %15.5 | %31.3 | %23.2 | %30.6 | %47.4 |
| Kural/anahtar kelime | %73.2 | %65.3 | %22.5 | %15.5 | %44.9 | %23.2 | %30.6 | %47.4 |
| TF-IDF + LR | %74.6 | %52.6 | %29.8 | %37.7 | %44.6 | %40.1 | %47.5 | %58.0 |
| TF-IDF + LR + çoğaltma | %74.6 | %90.3 | %57.5 | %52.8 | %58.2 | %50.3 | %47.5 | %69.8 |
| E5-small (donuk) + LR | %70.4 | %68.8 | %22.5 | %33.7 | %53.3 | %57.3 | %30.3 | %78.4 |
| E5-small (donuk) + LR + çoğaltma | %71.4 | %68.8 | %44.8 | %63.2 | %75.0 | %57.3 | %30.3 | %82.9 |
| E5-small ince ayar + çoğaltma | %34.0 | %70.8 | %66.7 | %57.5 | %81.3 | %54.7 | %67.9 | %80.4 |

