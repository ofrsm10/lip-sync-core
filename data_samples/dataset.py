import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.numpy_utils import pad_sequence


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.labels = []

        for word in os.listdir(root_dir):
            path = os.path.join(root_dir, word)
            for subdir, dirs, files in os.walk(path):
                for file in files:
                    try:
                        file_path = os.path.join(subdir, file)
                        if file == "features_norm.csv":
                            data1 = pd.read_csv(file_path, header=None, skiprows=1).values[:, 1:]
                            padded = pad_sequence(data1, 60)
                            if padded is not None:
                                self.data.append(padded)
                                self.labels.append(word)
                    except Exception as err:
                        print(file_path)
                        print(err)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        label = self.labels[idx]
        return data, label
