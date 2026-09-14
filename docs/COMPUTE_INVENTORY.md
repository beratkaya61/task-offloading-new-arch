# Hesaplama Donanımı Envanteri

Bu belge yalnız planlama girdisidir; çalıştırılmış benchmark veya model seçimi
kanıtı değildir.

## Kullanılabilir ikinci makine

- Konum: kullanıcının iş yeri makinesi
- Kullanıcı tarafından bildirilen GPU: **NVIDIA Quadro RTX 4000**
- Üretici teknik özelliği: **8 GB GDDR6**, Turing, 2304 CUDA çekirdeği,
  288 Tensor çekirdeği
- Kaynak: NVIDIA ürün sayfası ve veri sayfası:
  <https://www.nvidia.com/en-eu/products/workstations/quadro/rtx-4000/>

## Faz 4'e etkisi

- Qwen3-4B için 4-bit yerel inference ilk adaydır.
- Qwen3-8B için 8 GB VRAM sınırı nedeniyle CPU offload, daha kısa context veya
  daha küçük batch gerekebilir; uygun olduğu henüz iddia edilemez.
- Nihai model GPU adına bakılarak seçilmeyecek. Aynı validation setinde schema
  validity, macro-F1, kritik alan recall, latency, peak VRAM ve yeniden üretim
  ölçülecektir.
- Faz 4 başında `nvidia-smi` çıktısı, işletim sistemi, driver/CUDA sürümü, RAM ve
  kullanılabilir disk kaydedilmeden donanım doğrulandı sayılmayacaktır.
- Model ağırlıkları Git'e eklenmeyecek; model/tokenizer revision, quantization,
  prompt/parser hash ve cache checksum kaydedilecektir.

