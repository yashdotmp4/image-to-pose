import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from PIL import Image
import torchvision.transforms as T

def generate_heatmaps(keypoints, visibility, heatmap_size=(64, 48), sigma=2):
    # heatmap_size is (height, width)
    num_joints = keypoints.shape[0]
    heatmaps = np.zeros((num_joints, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
    
    for i in range(num_joints):
        if visibility[i] == 0:
            continue
            
        # scale keypoint to heatmap size
        x = keypoints[i, 0] * heatmap_size[1] / 192
        y = keypoints[i, 1] * heatmap_size[0] / 256
        
        if x < 0 or y < 0 or x >= heatmap_size[1] or y >= heatmap_size[0]:
            continue
        
        # draw gaussian centered on keypoint
        size = 6 * sigma + 1
        xx, yy = np.meshgrid(np.arange(size), np.arange(size))
        gaussian = np.exp(-((xx - 3*sigma)**2 + (yy - 3*sigma)**2) / (2 * sigma**2))
        
        # paste gaussian onto heatmap
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
    def __init__(self, img_dir, ann_file, input_size=(256, 192)):
        self.img_dir = img_dir
        self.input_size = input_size
        
        with open(ann_file) as f:
            data = json.load(f)
        
        # build image id to filename map
        self.id_to_file = {img['id']: img['file_name'] for img in data['images']}
        
        # filter annotations - only keep those with at least one keypoint
        self.annotations = [a for a in data['annotations'] 
                           if a['num_keypoints'] > 0 and a['iscrowd'] == 0]
        
        self.transform = T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # load and crop image to bounding box
        img_file = self.id_to_file[ann['image_id']]
        img = Image.open(os.path.join(self.img_dir, img_file)).convert('RGB')
        
        # crop to bounding box
        x, y, w, h = ann['bbox']
        img = img.crop((x, y, x+w, y+h))
        
        # get original crop size for keypoint scaling
        orig_w, orig_h = img.size
        
        # apply transforms
        img = self.transform(img)
        
        # parse keypoints - flat list of x,y,v triplets
        kps = np.array(ann['keypoints']).reshape(17, 3)
        
        # make keypoints relative to bounding box
        kps[:, 0] = kps[:, 0] - x
        kps[:, 1] = kps[:, 1] - y
        
        # scale keypoints to match resized image
        scale_x = self.input_size[1] / orig_w if orig_w > 0 else 1
        scale_y = self.input_size[0] / orig_h if orig_h > 0 else 1
        kps[:, 0] = kps[:, 0] * scale_x
        kps[:, 1] = kps[:, 1] * scale_y
        
        # visibility mask
        visibility = torch.tensor(kps[:, 2], dtype=torch.float32)
        
        heatmaps = generate_heatmaps(kps[:, :2], kps[:, 2])
        heatmaps = torch.tensor(heatmaps, dtype=torch.float32)
        
        return img, heatmaps, visibility