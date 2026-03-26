import torch
from torch.utils.data import Dataset
import numpy as np

MPI_TO_COCO = [6, 6, 6, 6, 6, 9, 14, 10, 15, 11, 16, 18, 23, 19, 24, 20, 25]

class PoseDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        keypoints2d = data['keypoints2d'][:, MPI_TO_COCO, :2]
        keypoints3d = data['keypoints3d'][:, MPI_TO_COCO, :3]
        
        root_2d = keypoints2d[:, 11:12, :]  # left hip as root
        root_3d = keypoints3d[:, 11:12, :]
        keypoints2d = keypoints2d - root_2d
        keypoints3d = keypoints3d - root_3d

        self.inputs = torch.tensor(keypoints2d.reshape(len(keypoints2d), -1), dtype=torch.float32)
        self.targets = torch.tensor(keypoints3d, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]