import os
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from transformers import Wav2Vec2Processor
import torch
import librosa
import numpy as np

label_dim = 174


def read_data(audio_path, rig_path):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    sampling_rate = 16000

    speech_array, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
    audio_data = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)

    # read rig data
    rig_data = np.loadtxt(rig_path, delimiter=",")

    return audio_data, rig_data


class AudioDataset(Dataset):
    def __init__(self, path):
        self._audios = []
        self._labels = []
        audiopath = path + "/WAV"
        for root, dirs, files in os.walk(audiopath):
            for file in files:
                if file.endswith(".wav"):
                    audio = os.path.join(root, file)
                    ctr = audio.replace("WAV", "VALID_CTR")
                    ctr = ctr.replace(".wav", ".txt")
                    if os.path.exists(ctr):
                        audio_data, rig_data = read_data(audio, ctr)
                        self._audios.append(audio_data)
                        self._labels.append(rig_data)
        self._file_cnt = len(self._audios)

    def __len__(self):
        return self._file_cnt

    def __getitem__(self, idx):
        return self._audios[idx], self._labels[idx]


def getAudioDataLoader(train_path, valid_path):
    train_dataset = AudioDataset(train_path)
    valid_dataset = AudioDataset(valid_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
    return train_loader, valid_loader
