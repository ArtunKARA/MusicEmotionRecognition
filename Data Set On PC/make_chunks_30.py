import os
from pydub import AudioSegment
from pydub.utils import make_chunks

# Giriş ve çıkış klasör yolları
input_base_folder = "C:/Users/Artun/Desktop/Müzik Veri Seti/bol/"
output_base_folder = "C:/Users/Artun/Desktop/Müzik Veri Seti/bol"

# Alt klasörlerin adları
subfolders = ['Agresif', 'Hüzünlü', 'Neşeli']

# Her alt klasördeki tüm wav dosyalarını işleyin
for subfolder in subfolders:
    input_subfolder_path = os.path.join(input_base_folder, subfolder)
    output_subfolder_path = os.path.join(output_base_folder, subfolder)
    os.makedirs(output_subfolder_path, exist_ok=True)  # Çıkış klasörünü oluştur veya varsa geç
    for root, _, files in os.walk(input_subfolder_path):
        for file in files:
            if file.endswith('.wav'):
                input_wav_file_path = os.path.join(root, file)
                audio = AudioSegment.from_wav(input_wav_file_path)
                
                # Ses dosyasını 30 saniyelik parçalara bölme
                chunk_length_ms = 30 * 1000  # 30 saniye
                chunks = make_chunks(audio, chunk_length_ms)
                
                # Her parçayı ayrı bir dosya olarak kaydetme
                for i, chunk in enumerate(chunks):
                    chunk_name = f"{os.path.splitext(file)[0]}_chunk{i}.wav"
                    chunk_output_path = os.path.join(output_subfolder_path, chunk_name)
                    chunk.export(chunk_output_path, format="wav")
                    print(f"Eklendi: {chunk_output_path}")
                
                # Orijinal dosyayı silme
                os.remove(input_wav_file_path)
                print(f"Silindi: {input_wav_file_path}")
                
print("Audio segmentation and deletion completed!")
