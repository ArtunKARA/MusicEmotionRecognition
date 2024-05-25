#%% kontrol
import torchaudio
info = torchaudio.info("C:/Users/Artun/Desktop/Müzik Veri Seti/wav/Agresif Müzikler/2Pac - Hit 'Em Up (Dirty) (Music Video) HD (128kbit_AAC)_resampled.wav")
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


