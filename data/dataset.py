import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PoseDataset(Dataset):
    def __init__(self, data_path, train=True):
        data = np.load(data_path, allow_pickle=True)
        print(data.files)  # print all keys so we can see what's inside

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass