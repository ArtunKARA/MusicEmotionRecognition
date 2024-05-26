#%% kontrol
import torchaudio
info = torchaudio.info("C:/Users/Artun/Desktop/Müzik Veri Seti/wav/Agresif Müzikler/(s)AINT with lyrics (128kbit_AAC)_resampled_chunk0.wav")
print(info.sample_rate)

#%% modeli yeniden başlat hubert

import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2Processor


# Yeni bir işlemci ve model oluşturma
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

# Mevcut modeli ve işlemciyi kaldırma
del processor
del model
torch.cuda.empty_cache()  # GPU belleğini temizleme (isteğe bağlı)


#%% hubert model görüntğleme

import torch
from transformers import HubertModel, Wav2Vec2Processor

# HuBert modelini ve tokenizer'ı yükleyin
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

# Modelin ve işlemcinin yapılandırmasını görüntüleme
print("Processor Configuration:")
print(processor)

print("\nModel Configuration:")
print(model)

# Modelin katmanlarını detaylı görüntüleme
print("\nModel Architecture:")
print(model.config)

# Örnek bir giriş verisiyle modelin çıktısını inceleme
input_values = torch.randn(1, 16000)  # 1 saniyelik rastgele bir ses örneği
inputs = processor(input_values, return_tensors="pt", sampling_rate=16000)
outputs = model(**inputs)

print("\nModel Output Shape:")
print(outputs.last_hidden_state.shape)
#%% 5mb den az ola dosya silme
import os

# Silmek istediğiniz dizin yolunu belirtin
directory = "C:/Users/Artun/Desktop/Müzik Veri Seti/wav/Agresif"

# 5 MB'yi bayta çevirin
size_limit = 5 * 1024 * 1024  # 5 MB in bytes

# Klasördeki dosyaları kontrol edin
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    # Dosya olup olmadığını kontrol edin
    if os.path.isfile(file_path):
        # Dosya boyutunu alın
        file_size = os.path.getsize(file_path)
        # Eğer dosya boyutu 5 MB'den küçükse dosyayı silin
        if file_size < size_limit:
            os.remove(file_path)
            print(f"{filename} silindi. (Boyut: {file_size} bytes)")

print("İşlem tamamlandı.")
#%% rasgle dosya silen kod
import os
import random

def delete_random_files(directory, num_files_to_delete):
    # Klasördeki tüm dosyaları al
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Eğer silinmesi istenen dosya sayısı mevcut dosya sayısından fazla ise hata ver
    if num_files_to_delete > len(all_files):
        print(f"Silinecek dosya sayısı mevcut dosya sayısından fazla: {len(all_files)} adet dosya var.")
        return
    
    # Rasgele dosyaları seç
    files_to_delete = random.sample(all_files, num_files_to_delete)
    
    # Dosyaları sil
    for file_path in files_to_delete:
        os.remove(file_path)
        print(f"{os.path.basename(file_path)} silindi.")
    
    print(f"{num_files_to_delete} adet dosya silindi.")

# Kullanım
directory = "C:/Users/Artun/Desktop/Müzik Veri Seti/wav/Hüzünlü"  # Klasör yolunu belirtin
num_files_to_delete = 1037  # Silinecek dosya sayısını belirtin

delete_random_files(directory, num_files_to_delete)

