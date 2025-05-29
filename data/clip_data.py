import os
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor
from transformers import BertTokenizer

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


class CLIPDataset(Dataset):
    def __init__(self, path):
        self._audios = []
        self._rigs = []
        self._images = []
        self._texts = []
        audio_path = os.path.join(path, "audio")
        rig_path = os.path.join(path, "rig")
        image_path = os.path.join(path, "image")
        text_path = os.path.join(path, "text")
        for file in os.listdir(audio_path):
            audio_file = os.path.join(audio_path, file[:-4] + ".wav")
            rig_file = os.path.join(rig_path, file[:-4] + ".txt")
            image_file = os.path.join(image_path, file[:-4] + ".png")
            text_file = os.path.join(text_path, file[:-4] + ".txt")
            # audio and rig
            audio_data, rig_data = read_data(audio_file, rig_file)
            self._audios.append(audio_data)
            self._rigs.append(rig_data)
            # image and text
            self._images.append(image_file)
            self._texts.append(text_file)

        self._file_cnt = len(self._audios)

    def __len__(self):
        return self._file_cnt

    def __getitem__(self, idx):
        return self._audios[idx], self._rigs[idx], self._images[idx], self._texts[idx]


def getCLIPDataLoader(data_path):
    dataset = CLIPDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    return dataloader

