import os
import torchaudio
from tqdm import tqdm

# Klasör yolu ve hedef örnekleme hızı
base_folder = "C:/Users/Artun/Desktop/Müzik Veri Seti/wav"
target_sample_rate = 16000

# Alt klasörlerin adları
subfolders = ['Agresif Müzikler', 'Hüzünlü Müzikler', 'Neşeli Müzikler']

# Her alt klasördeki tüm wav dosyalarını dönüştür
for subfolder in subfolders:
    subfolder_path = os.path.join(base_folder, subfolder)
    for root, _, files in os.walk(subfolder_path):
        for file in tqdm(files, desc=f"Processing {subfolder}"):
            if file.endswith('.wav'):
                wav_file_path = os.path.join(root, file)
                # Ses dosyasını yükle
                waveform, sample_rate = torchaudio.load(wav_file_path)
                # Örnekleme hızını dönüştür
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
                waveform_resampled = resampler(waveform)
                # Dönüştürülmüş ses dosyasını kaydet
                output_file_path = os.path.join(root, f"{os.path.splitext(file)[0]}_resampled.wav")
                torchaudio.save(output_file_path, waveform_resampled, sample_rate=target_sample_rate)

print("Resampling completed!")
