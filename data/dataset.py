import torch
from torch.utils.data import Dataset
import numpy as np

class PoseDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        kps2d = data['keypoints2d']  # (N, 17, 2) already in COCO format
        kps3d = data['keypoints3d']  # (N, 17, 3)

        # subtract root (left hip = index 11)
        root_2d = kps2d[:, 11:12, :]
        root_3d = kps3d[:, 11:12, :]
        kps2d = kps2d - root_2d
        kps3d = kps3d - root_3d

        # normalise by torso size
        torso_2d = (
            np.linalg.norm(kps2d[:, 5, :] - kps2d[:, 11, :], axis=1) +
            np.linalg.norm(kps2d[:, 6, :] - kps2d[:, 12, :], axis=1)
        ) / 2
        torso_2d = np.maximum(torso_2d, 1e-6)
        kps2d = kps2d / torso_2d[:, None, None]

        torso_3d = (
            np.linalg.norm(kps3d[:, 5, :] - kps3d[:, 11, :], axis=1) +
            np.linalg.norm(kps3d[:, 6, :] - kps3d[:, 12, :], axis=1)
        ) / 2
        torso_3d = np.maximum(torso_3d, 1e-6)
        kps3d = kps3d / torso_3d[:, None, None]

        self.inputs = torch.tensor(kps2d.reshape(len(kps2d), -1), dtype=torch.float32)
        self.targets = torch.tensor(kps3d, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]