import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PoseDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        keypoints2d = data['keypoints2d'][:, :, :2]
        keypoints3d = data['keypoints3d'][:, :, :3]
        root_2d = keypoints2d[:, 0:1, :]
        root_3d = keypoints3d[:, 0:1, :]
        keypoints2d = keypoints2d - root_2d
        keypoints3d = keypoints3d - root_3d
        self.inputs = torch.tensor(keypoints2d.reshape(len(keypoints2d), -1), dtype=torch.float32)
        self.targets = torch.tensor(keypoints3d, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]