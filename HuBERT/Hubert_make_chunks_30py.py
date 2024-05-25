import os
from pydub import AudioSegment
from pydub.utils import make_chunks

# Giriş ve çıkış klasör yolları
input_base_folder = "path/to/your/input/base/folder"
output_base_folder = "path/to/your/output/base/folder"

# Alt klasörlerin adları
subfolders = ['Agresif Müzikler', 'Hüzünlü Müzikler', 'Neşeli Müzikler']

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
                
                # Orijinal dosyayı silme
                os.remove(input_wav_file_path)

print("Audio segmentation and deletion completed!")
