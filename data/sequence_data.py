import os
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
import torch
import numpy as np

fps = 60
label_dim = 174
emotion_id = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "sad": 5, "surprise": 6}


def process_rig(rig_path, max_duration=30):
    rig_data = np.loadtxt(rig_path, delimiter=",")
    rig_shape = rig_data.shape
    if rig_data.shape[0] % max_duration != 0:
        tail_rig = np.zeros((max_duration - rig_data.shape[0] % max_duration, rig_shape[1]))
        rig_data = np.concatenate((rig_data, tail_rig), axis=0)
    rig_indices = list(range(0, len(rig_data), max_duration))[1:]
    rig_chunks = np.split(rig_data, rig_indices)

    return rig_chunks, rig_shape[0]


class SeqDataset(Dataset):
    def __init__(self, path):
        self._emotion_num = 7
        self._rig = [[] for i in range(self._emotion_num)]
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".txt"):
                    ctr = os.path.join(root, file)
                    for key in emotion_id.keys():
                        if key in root:
                            index = emotion_id[key]
                            break
                    rig_data = np.loadtxt(ctr, delimiter=",")
                    self._rig[index].append(rig_data)
        self._seq_len = len(self._rig[0])
        print(self._seq_len)

    def __getitem__(self, index):
        # choose random index
        eidx = int(index / self._seq_len)
        cidx = index % self._seq_len
        eidx1, eidx2 = np.random.choice(self._emotion_num, size=2, replace=False)
        cidx1, cidx2 = np.random.choice(self._seq_len, size=2, replace=False)

        # Stage 1: self reconstruction
        rig = self._rig[eidx][cidx]

        # Stage 2.1 Emotion Reconstruction
        rig01 = self._rig[eidx][cidx1]
        rig02 = self._rig[eidx][cidx2]
        emo_seq_len = min(rig01.shape[0], rig02.shape[0])
        rig01 = rig01[:emo_seq_len]
        rig02 = rig02[:emo_seq_len]

        # Stage 2.2 Content Reconstruction
        rig10 = self._rig[eidx1][cidx]
        rig20 = self._rig[eidx2][cidx]

        # Stage 3: Cross Recontruction
        rig11 = self._rig[eidx1][cidx1]
        rig22 = self._rig[eidx2][cidx2]
        cross_seq_len = min(rig11.shape[0], rig22.shape[0])
        rig11 = rig11[:cross_seq_len]
        rig22 = rig22[:cross_seq_len]

        return {"self": rig, "emotion": [rig01, rig02], "content": [rig10, rig20], "cross": [rig11, rig22]}

    def __len__(self):
        return self._seq_len * self._emotion_num


def getSeqDataLoader(data_path):
    dataset = SeqDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    return dataloader

