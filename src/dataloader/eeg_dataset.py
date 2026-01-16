import numpy as np
import torch
from torch.utils.data import Dataset

class EEGWindowDataset(Dataset):
    """
    Dataset mínim que rep X i Y ja carregats a RAM.
    """
    def __init__(self, X, Y):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)  # [21,128]
        y = torch.tensor(self.Y[idx], dtype=torch.long)
        return x, y
