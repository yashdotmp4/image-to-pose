"""
lighting_dataset.py
DPR dataset loader for lighting direction regression.

Each sample: (image, light_xyz) where light_xyz is extracted from
9 SH coefficients — first-order terms give dominant light direction.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def sh_to_light_direction(sh_coeffs):
    """
    Convert 9 second-order SH coefficients to a dominant light direction (x, y, z).
    First-order SH bands (indices 1, 2, 3) encode directional lighting.
    Convention matches DPR / shtools coordinate system.
    """
    sh = np.array(sh_coeffs, dtype=np.float32)
    # L1 band: sh[1]=Y, sh[2]=Z, sh[3]=X (in shtools convention)
    x = -sh[3]
    y = -sh[1]
    z =  sh[2]
    vec = np.array([x, y, z], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return vec / norm


def load_sh(txt_path):
    with open(txt_path, 'r') as f:
        vals = [float(line.strip()) for line in f if line.strip()]
    return vals


class DPRDataset(Dataset):
    """
    Loads the DPR 256x256 dataset.
    Root structure:
        root/
            imgHQ00000/
                imgHQ00000_00.jpg
                imgHQ00000_light_00.txt
                ...
            imgHQ00001/
                ...
    """

    def __init__(self, root, split='train', val_fraction=0.05, transform=None, seed=42):
        self.root = Path(root)
        self.transform = transform

        # Collect all (image_path, light_path) pairs
        all_pairs = []
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue
            name = folder.name
            for i in range(5):
                img_path   = folder / f"{name}_{i:02d}.jpg"
                light_path = folder / f"{name}_light_{i:02d}.txt"
                if img_path.exists() and light_path.exists():
                    all_pairs.append((img_path, light_path))

        # Reproducible train/val split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(all_pairs))
        n_val = int(len(all_pairs) * val_fraction)
        if split == 'val':
            self.pairs = [all_pairs[i] for i in indices[:n_val]]
        else:
            self.pairs = [all_pairs[i] for i in indices[n_val:]]

        print(f"DPRDataset [{split}]: {len(self.pairs)} samples")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, light_path = self.pairs[idx]

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        sh = load_sh(light_path)
        light = sh_to_light_direction(sh)
        light = torch.tensor(light, dtype=torch.float32)

        return img, light


def get_transforms(split='train'):
    if split == 'train':
        return T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
