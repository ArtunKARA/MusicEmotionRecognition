import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import Wav2Vec2Tokenizer, HubertForSequenceClassification
import matplotlib.pyplot as plt

# Dosya yollarını belirleyin
features_path = 'C:/Users/Artun/Desktop/MusicEmotionRecognition/HuBERT/features.npy'
labels_path = 'C:/Users/Artun/Desktop/MusicEmotionRecognition/HuBERT/labels.npy'

# Dosya yollarını kontrol edin
if not os.path.exists(features_path):
    raise FileNotFoundError(f"'{features_path}' dosyası bulunamadı.")
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"'{labels_path}' dosyası bulunamadı.")

# Verileri yükleyin
data = np.load(features_path, allow_pickle=True)
labels = np.load(labels_path)

# HuBERT model ve tokenizer'ı yükleyin
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForSequenceClassification.from_pretrained("facebook/hubert-large-ls960-ft", num_labels=3)

# Verilerinizi tokenize edin
input_values = tokenizer(data.tolist(), return_tensors="pt", padding=True, truncation=True).input_values

# Verileri eğitim ve test setlerine ayırın
from sklearn.model_selection import train_test_split
train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_values, labels, test_size=0.2, random_state=42)

# DataLoader oluşturun
train_data = TensorDataset(train_inputs, torch.tensor(train_labels))
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_data = TensorDataset(test_inputs, torch.tensor(test_labels))
test_loader = DataLoader(test_data, batch_size=8, shuffle=False)

# Modeli eğitin
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
epochs = 5

train_loss = []
model.train()

for epoch in range(epochs):
    total_loss = 0
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
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

# Eğitim süreci grafiği
plt.plot(range(epochs), train_loss, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('training_loss.png')
plt.show()

# Modeli kaydetme
model.save_pretrained('./model')
