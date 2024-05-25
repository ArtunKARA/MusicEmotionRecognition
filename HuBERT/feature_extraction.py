# feature_extraction.py

import torch
import torchaudio
from transformers import HubertModel, Wav2Vec2Processor
import os
import numpy as np

# HuBert modelini ve tokenizer'ı yükleyin
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

# Bir wav dosyasından özellik çıkarma
def extract_features(wav_file):
    waveform, sample_rate = torchaudio.load(wav_file)
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        features = model(**inputs).last_hidden_state
    return features.mean(dim=1).squeeze().numpy()  # Ortalamayı alarak tek boyutlu özellik vektörü elde edin

# Özellikleri ve etiketleri saklamak için listeler
features = []
labels = []

# Wav dosyalarının bulunduğu ana dizinler
directories = {
    'Agresif Müzikler': 0,
    'Hüzünlü Müzikler': 1,
    'Neşeli Müzikler': 2
}

# Wav dosyalarından özellikleri çıkarma ve veri setini hazırlama
def process_and_extract_features(directories):
    for label, class_idx in directories.items():
        folder_path = f'C:/Users/Artun/Desktop/Müzik Veri Seti/wav/{label}'  # Örneğin: 'path/to/your/Agresif Müzikler'
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.wav'):
                    wav_file_path = os.path.join(root, file)
                    feature = extract_features(wav_file_path)
                    features.append(feature)
                    labels.append(class_idx)

# Veri setinin hazırlanması ve özellik çıkarma işlemleri
process_and_extract_features(directories)

# Veri setinin boyutlarını kontrol etme
print("Features shape:", np.array(features).shape)
print("Labels shape:", np.array(labels).shape)

# Özellikleri ve etiketleri kaydetme
np.save('features.npy', features)
np.save('labels.npy', labels)
