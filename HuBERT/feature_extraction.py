import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Veri yollarını ve etiketleri belirleyin
data_dir = "C:/Users/Artun/Desktop/Müzik Veri Seti/wav"  # Bu kısmı kendi veri yolunuza göre düzenleyin
categories = ["Agresif", "Hüzünlü", "Neşeli"]

def extract_features(data_dir, categories, n_mfcc=13):
    features = []
    labels = []
    
    for category in categories:
        category_path = os.path.join(data_dir, category)
        print(f"Kategori: {category}")
        for file in os.listdir(category_path):
            if file.endswith('.wav'):
                file_path = os.path.join(category_path, file)
                print(f"  Dosya: {file_path}")
                y, sr = librosa.load(file_path, sr=None)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                mfccs = np.mean(mfccs.T, axis=0)
                features.append(mfccs)
                labels.append(category)
    
    return np.array(features), labels

features, labels = extract_features(data_dir, categories)

# Etiketleri sayısal değere dönüştürme
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

print("Etiketlerin sayısal karşılıkları:")
for category, encoded_label in zip(categories, label_encoder.transform(categories)):
    print(f"  {category}: {encoded_label}")

# Verileri kaydetme
np.save('features.npy', features)
np.save('labels.npy', labels)

print("Veriler ve etiketler kaydedildi:")
print("  features.npy dosyasına özellikler kaydedildi.")
print("  labels.npy dosyasına etiketler kaydedildi.")
