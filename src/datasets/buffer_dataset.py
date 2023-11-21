import torch
from g2p_en import G2p
from torch.utils.data import Dataset, DataLoader
import time

import os
import numpy as np


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


class BufferDataset(Dataset):
    def __init__(self, data_path, **kwargs):
        self.g2p = G2p()
        self.symbols = {" ": 0, "": 0, "spn": 1} | {
            k: v for v, k in enumerate(self.g2p.phonemes)
        }
        buffer = self.get_data_to_buffer(data_path)
        self.buffer = buffer
        self.length_dataset = len(self.buffer)

    def get_data_to_buffer(self, data_path):
        text = process_text(os.path.join(data_path, "train.txt"))
        text = text
        start = time.perf_counter()
        buffer = [
            {
                "text": torch.from_numpy(
                    np.array([self.symbols[x] for x in t.replace("\n", "").split(" ")])
                ),
                "duration": torch.from_numpy(
                    np.load(
                        os.path.join(data_path, f"alignments/alignment-{i + 1:05d}.npy")
                    )
                ),
                "mel_target": torch.from_numpy(
                    np.load(os.path.join(data_path, f"mels/mel-{i + 1:05d}.npy"))
                ),
                "pitch_target": torch.from_numpy(
                    np.load(os.path.join(data_path, f"pitch/pitch-{i + 1:05d}.npy"))
                ),
                "energy_target": torch.from_numpy(
                    np.load(os.path.join(data_path, f"energy/energy-{i + 1:05d}.npy"))
                ),
            }
            for i, t in enumerate(text)
        ]

        end = time.perf_counter()
        print(f"cost {end - start:.2f}s to load all data into buffer.")
        return buffer

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]
