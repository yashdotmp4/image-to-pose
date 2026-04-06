import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

FLIP_PAIRS = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16]]

def generate_heatmaps(keypoints, visibility, heatmap_size=(96, 72), sigma=3):
    num_joints = keypoints.shape[0]
    heatmaps = np.zeros((num_joints, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
    for i in range(num_joints):
        if visibility[i] == 0:
            continue
        x = keypoints[i, 0] * heatmap_size[1] / 288
        y = keypoints[i, 1] * heatmap_size[0] / 384
        if x < 0 or y < 0 or x >= heatmap_size[1] or y >= heatmap_size[0]:
            continue
        size = 6 * sigma + 1
        xx, yy = np.meshgrid(np.arange(size), np.arange(size))
        gaussian = np.exp(-((xx - 3*sigma)**2 + (yy - 3*sigma)**2) / (2 * sigma**2))
        x, y = int(x), int(y)
        x1, y1 = max(0, x - 3*sigma), max(0, y - 3*sigma)
        x2, y2 = min(heatmap_size[1], x + 3*sigma + 1), min(heatmap_size[0], y + 3*sigma + 1)
        gx1 = max(0, 3*sigma - x)
        gy1 = max(0, 3*sigma - y)
        gx2 = gx1 + (x2 - x1)
        gy2 = gy1 + (y2 - y1)
        heatmaps[i, y1:y2, x1:x2] = gaussian[gy1:gy2, gx1:gx2]
    return heatmaps

class COCOKeypointDataset(Dataset):
    def __init__(self, img_dir, ann_file, input_size=(384, 288)):
        self.img_dir = img_dir
        self.input_size = input_size

        with open(ann_file) as f:
            data = json.load(f)

        self.id_to_file = {img['id']: img['file_name'] for img in data['images']}
        self.annotations = [a for a in data['annotations']
                           if a['num_keypoints'] > 0 and a['iscrowd'] == 0
                           and os.path.exists(os.path.join(img_dir, self.id_to_file[a['image_id']]))]

        self.base_transform = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_file = self.id_to_file[ann['image_id']]
        img = Image.open(os.path.join(self.img_dir, img_file)).convert('RGB')

        x, y, w, h = ann['bbox']
        img = img.crop((x, y, x+w, y+h))
        orig_w, orig_h = img.size

        img = img.resize((self.input_size[1], self.input_size[0]))

        kps = np.array(ann['keypoints']).reshape(17, 3)
        kps[:, 0] = (kps[:, 0] - x) * self.input_size[1] / orig_w if orig_w > 0 else kps[:, 0]
        kps[:, 1] = (kps[:, 1] - y) * self.input_size[0] / orig_h if orig_h > 0 else kps[:, 1]

        # random horizontal flip with correct keypoint swap
        if random.random() > 0.5:
            img = TF.hflip(img)
            kps[:, 0] = self.input_size[1] - kps[:, 0]
            for left, right in FLIP_PAIRS:
                kps[left], kps[right] = kps[right].copy(), kps[left].copy()

        # random rotation
        angle = random.uniform(-30, 30)
        img = TF.rotate(img, angle)

        img = self.base_transform(img)

        visibility = torch.tensor(kps[:, 2], dtype=torch.float32)
        heatmaps = generate_heatmaps(kps[:, :2], kps[:, 2])
        heatmaps = torch.tensor(heatmaps, dtype=torch.float32)

        return img, heatmaps, visibility