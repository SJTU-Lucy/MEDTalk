import os
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor
from transformers import BertTokenizer


label_dim = 174
emotion_id = {"ang": 0, "dis": 1, "fea": 2, "hap": 3, "neu": 4, "sad": 5, "sur": 6}
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")


def load_audio(audio_path):
    sampling_rate = 16000
    speech_array, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
    audio_data = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)

    return audio_data


def read_data(audio_path, rig_path):
    sampling_rate = 16000

    speech_array, sampling_rate = librosa.load(audio_path, sr=sampling_rate)
    audio_data = np.squeeze(processor(speech_array, sampling_rate=sampling_rate).input_values)

    # read rig data
    rig_data = np.loadtxt(rig_path, delimiter=",")

    return audio_data, rig_data


class TextDataset(Dataset):
    def __init__(self, path, texts):
        self._audios = []
        self._labels = []
        self._emotions = []
        self._texts = texts
        audiopath = path + "/WAV"
        for root, dirs, files in os.walk(audiopath):
            for file in files:
                if file.endswith(".wav"):
                    audio = os.path.join(root, file)
                    ctr = audio.replace("WAV", "VALID_CTR")
                    ctr = ctr.replace(".wav", ".txt")
                    if os.path.exists(ctr):
                        audio_data, rig_data = read_data(audio, ctr)
                        # audio and rig
                        self._audios.append(audio_data)
                        self._labels.append(rig_data)
                        # emotion label
                        emotion = emotion_id[file[0:3]]
                        self._emotions.append(emotion)

        self._file_cnt = len(self._audios)

    def __len__(self):
        return self._file_cnt

    def __getitem__(self, idx):
        return self._audios[idx], self._labels[idx], self._emotions[idx], self._texts[idx]


def getTextDataLoader(train_path, valid_path, train_text, test_text):
    train_corpus = pd.read_csv(train_text, encoding="gbk")
    test_corpus = pd.read_csv(test_text, encoding="gbk")
    train_text = train_corpus['text'].values.tolist()
    test_text = test_corpus['text'].values.tolist()
    train_dataset = TextDataset(train_path, train_text)
    valid_dataset = TextDataset(valid_path, test_text)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
    return train_loader, valid_loader

