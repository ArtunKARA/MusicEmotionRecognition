import librosa
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

def extract_features(path):
    id = 1
    feature_set = pd.DataFrame()

    songname_vector = []
    label_vector = []  # Add a label vector
    tempo_vector = []
    total_beats = []
    average_beats = []
    chroma_stft_mean = []
    chroma_stft_std = []
    chroma_stft_var = []
    chroma_cq_mean = []
    chroma_cq_std = []
    chroma_cq_var = []
    chroma_cens_mean = []
    chroma_cens_std = []
    chroma_cens_var = []
    mel_mean = []
    mel_std = []
    mel_var = []
    mfcc_mean = []
    mfcc_std = []
    mfcc_var = []
    mfcc_delta_mean = []
    mfcc_delta_std = []
    mfcc_delta_var = []
    rmse_mean = []
    rmse_std = []
    rmse_var = []
    cent_mean = []
    cent_std = []
    cent_var = []
    spec_bw_mean = []
    spec_bw_std = []
    spec_bw_var = []
    contrast_mean = []
    contrast_std = []
    contrast_var = []
    rolloff_mean = []
    rolloff_std = []
    rolloff_var = []
    poly_mean = []
    poly_std = []
    poly_var = []
    tonnetz_mean = []
    tonnetz_std = []
    tonnetz_var = []
    zcr_mean = []
    zcr_std = []
    zcr_var = []
    harm_mean = []
    harm_std = []
    harm_var = []
    perc_mean = []
    perc_std = []
    perc_var = []
    frame_mean = []
    frame_std = []
    frame_var = []

    file_data = [f for f in listdir(path) if isfile(join(path, f))]
    for line in file_data:
        if line[-1:] == '\n':
            line = line[:-1]

        songname = join(path, line)
        try:
            y, sr = librosa.load(songname, duration=60)
        except Exception as e:
            print(f"Error loading {songname}: {e}")
            continue

        S = np.abs(librosa.stft(y))
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        mfcc_delta = librosa.feature.delta(mfcc)
        rmse = librosa.feature.rms(y=y)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        poly = librosa.feature.poly_features(S=S, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        harm = librosa.effects.harmonic(y)
        perc = librosa.effects.percussive(y)
        frames_to_time = librosa.frames_to_time(frames=beats, sr=sr)

        songname_vector.append(line)
        label_vector.append(1)  # Placeholder label, replace with actual labels as needed
        tempo_vector.append(tempo)
        total_beats.append(sum(beats))
        average_beats.append(np.average(beats))
        chroma_stft_mean.append(np.mean(chroma_stft))
        chroma_stft_std.append(np.std(chroma_stft))
        chroma_stft_var.append(np.var(chroma_stft))
        chroma_cq_mean.append(np.mean(chroma_cq))
        chroma_cq_std.append(np.std(chroma_cq))
        chroma_cq_var.append(np.var(chroma_cq))
        chroma_cens_mean.append(np.mean(chroma_cens))
        chroma_cens_std.append(np.std(chroma_cens))
        chroma_cens_var.append(np.var(chroma_cens))
        mel_mean.append(np.mean(mel))
        mel_std.append(np.std(mel))
        mel_var.append(np.var(mel))
        mfcc_mean.append(np.mean(mfcc))
        mfcc_std.append(np.std(mfcc))
        mfcc_var.append(np.var(mfcc))
        mfcc_delta_mean.append(np.mean(mfcc_delta))
        mfcc_delta_std.append(np.std(mfcc_delta))
        mfcc_delta_var.append(np.var(mfcc_delta))
        rmse_mean.append(np.mean(rmse))
        rmse_std.append(np.std(rmse))
        rmse_var.append(np.var(rmse))
        cent_mean.append(np.mean(cent))
        cent_std.append(np.std(cent))
        cent_var.append(np.var(cent))
        spec_bw_mean.append(np.mean(spec_bw))
        spec_bw_std.append(np.std(spec_bw))
        spec_bw_var.append(np.var(spec_bw))
        contrast_mean.append(np.mean(contrast))
        contrast_std.append(np.std(contrast))
        contrast_var.append(np.var(contrast))
        rolloff_mean.append(np.mean(rolloff))
        rolloff_std.append(np.std(rolloff))
        rolloff_var.append(np.var(rolloff))
        poly_mean.append(np.mean(poly))
        poly_std.append(np.std(poly))
        poly_var.append(np.var(poly))
        tonnetz_mean.append(np.mean(tonnetz))
        tonnetz_std.append(np.std(tonnetz))
        tonnetz_var.append(np.var(tonnetz))
        zcr_mean.append(np.mean(zcr))
        zcr_std.append(np.std(zcr))
        zcr_var.append(np.var(zcr))
        harm_mean.append(np.mean(harm))
        harm_std.append(np.std(harm))
        harm_var.append(np.var(harm))
        perc_mean.append(np.mean(perc))
        perc_std.append(np.std(perc))
        perc_var.append(np.var(perc))
        frame_mean.append(np.mean(frames_to_time))
        frame_std.append(np.std(frames_to_time))
        frame_var.append(np.var(frames_to_time))
        id += 1

    feature_set['song_name'] = songname_vector
    feature_set['label'] = label_vector  # Add labels to the dataset
    feature_set['tempo'] = tempo_vector
    feature_set['total_beats'] = total_beats
    feature_set['average_beats'] = average_beats
    feature_set['chroma_stft_mean'] = chroma_stft_mean
    feature_set['chroma_stft_std'] = chroma_stft_std
    feature_set['chroma_stft_var'] = chroma_stft_var
    feature_set['chroma_cq_mean'] = chroma_cq_mean
    feature_set['chroma_cq_std'] = chroma_cq_std
    feature_set['chroma_cq_var'] = chroma_cq_var
    feature_set['chroma_cens_mean'] = chroma_cens_mean
    feature_set['chroma_cens_std'] = chroma_cens_std
    feature_set['chroma_cens_var'] = chroma_cens_var
    feature_set['mel_mean'] = mel_mean
    feature_set['mel_std'] = mel_std
    feature_set['mel_var'] = mel_var
    feature_set['mfcc_mean'] = mfcc_mean
    feature_set['mfcc_std'] = mfcc_std
    feature_set['mfcc_var'] = mfcc_var
    feature_set['mfcc_delta_mean'] = mfcc_delta_mean
    feature_set['mfcc_delta_std'] = mfcc_delta_std
    feature_set['mfcc_delta_var'] = mfcc_delta_var
    feature_set['rmse_mean'] = rmse_mean
    feature_set['rmse_std'] = rmse_std
    feature_set['rmse_var'] = rmse_var
    feature_set['cent_mean'] = cent_mean
    feature_set['cent_std'] = cent_std
    feature_set['cent_var'] = cent_var
    feature_set['spec_bw_mean'] = spec_bw_mean
    feature_set['spec_bw_std'] = spec_bw_std
    feature_set['spec_bw_var'] = spec_bw_var
    feature_set['contrast_mean'] = contrast_mean
    feature_set['contrast_std'] = contrast_std
    feature_set['contrast_var'] = contrast_var
    feature_set['rolloff_mean'] = rolloff_mean
    feature_set['rolloff_std'] = rolloff_std
    feature_set['rolloff_var'] = rolloff_var
    feature_set['poly_mean'] = poly_mean
    feature_set['poly_std'] = poly_std
    feature_set['poly_var'] = poly_var
    feature_set['tonnetz_mean'] = tonnetz_mean
    feature_set['tonnetz_std'] = tonnetz_std
    feature_set['tonnetz_var'] = tonnetz_var
    feature_set['zcr_mean'] = zcr_mean
    feature_set['zcr_std'] = zcr_std
    feature_set['zcr_var'] = zcr_var
    feature_set['harm_mean'] = harm_mean
    feature_set['harm_std'] = harm_std
    feature_set['harm_var'] = harm_var
    feature_set['perc_mean'] = perc_mean
    feature_set['perc_std'] = perc_std
    feature_set['perc_var'] = perc_var
    feature_set['frame_mean'] = frame_mean
    feature_set['frame_std'] = frame_std
    feature_set['frame_var'] = frame_var

    # Save to CSV file
    feature_set.to_csv('dataset.csv', index=False)

    return feature_set

# Example usage
path = "C:\\Users\\Artun\\Desktop\\Müzik Veri Seti\\Agresif Müzikler"
features = extract_features(path)
print(features)