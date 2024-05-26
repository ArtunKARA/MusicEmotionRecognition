import os
import librosa
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import Wav2Vec2Processor, HubertForSequenceClassification
import matplotlib.pyplot as plt

# Dosya yollarını ve etiketleri belirleyin
data_dir = "C:/Users/Artun/Desktop/Müzik Veri Seti/wav"  # Bu kısmı kendi veri yolunuza göre düzenleyin
categories = ["Agresif", "Hüzünlü", "Neşeli"]

def load_audio_files(data_dir, categories, sr=16000):
    audio_files = []
    labels = []
    
    for category in categories:
        category_path = os.path.join(data_dir, category)
        print(f"Kategori: {category}")
        for file in os.listdir(category_path):
            if file.endswith('.wav'):
                file_path = os.path.join(category_path, file)
                print(f"  Dosya: {file_path}")
                y, _ = librosa.load(file_path, sr=sr)  # 16kHz örnekleme hızıyla yükleyin
                audio_files.append(y)
                labels.append(category)
    
    return audio_files, labels

print("Ses dosyaları yükleniyor...")
audio_files, labels = load_audio_files(data_dir, categories)
print("Ses dosyaları yüklendi.")

# Etiketleri sayısal değere dönüştürme
from sklearn.preprocessing import LabelEncoder

print("Etiketler sayısal değerlere dönüştürülüyor...")
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
print("Etiketler dönüştürüldü.")
print("Etiketlerin sayısal karşılıkları:")
for category, encoded_label in zip(categories, label_encoder.transform(categories)):
    print(f"  {category}: {encoded_label}")

# HuBERT model ve processor'ı yükleyin
print("HuBERT model ve processor yükleniyor...")
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForSequenceClassification.from_pretrained("facebook/hubert-large-ls960-ft", num_labels=3)
print("HuBERT model ve processor yüklendi.")

# Verilerinizi tokenize edin
print("Veriler tokenize ediliyor...")
max_length = 16000 * 30  # 10 saniyelik audio için
input_values = processor(audio_files, return_tensors="pt", padding=True, truncation=True, max_length=max_length, sampling_rate=16000).input_values
print("Veriler tokenize edildi.")

# Verileri eğitim ve test setlerine ayırın
print("Veriler eğitim ve test setlerine ayrılıyor...")
from sklearn.model_selection import train_test_split
train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_values, labels, test_size=0.2, random_state=42)
print("Veriler ayrıldı.")

# DataLoader oluşturun
print("DataLoader oluşturuluyor...")
train_data = TensorDataset(train_inputs, torch.tensor(train_labels))
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_data = TensorDataset(test_inputs, torch.tensor(test_labels))
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
print("DataLoader oluşturuldu.")

# Modeli eğitin
print("Model eğitiliyor...")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
epochs = 5

train_loss = []
model.train()

for epoch in range(epochs):
    total_loss = 0
    print(f"Epoch {epoch+1} başlıyor...")
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_loss.append(avg_loss)
    print(f"Epoch {epoch+1} tamamlandı, Loss: {avg_loss}")

# Eğitim süreci grafiği
print("Eğitim süreci grafiği oluşturuluyor...")
plt.plot(range(epochs), train_loss, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('training_loss.png')
plt.show()
print("Eğitim süreci grafiği oluşturuldu ve kaydedildi.")

# Modeli kaydetme
print("Model kaydediliyor...")
model.save_pretrained('./model')
print("Model kaydedildi.")
