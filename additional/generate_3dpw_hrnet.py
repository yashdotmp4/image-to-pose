"""
generate_3dpw_hrnet.py

Runs HRNet on 3DPW images to generate matched 2D-3D pairs for MartinezNet training.
Samples every Nth frame from each sequence.

Usage:
    python3 generate_3dpw_hrnet.py \
        --image_dir /scratch/fyp-stuff/imageFiles \
        --seq_dir /scratch/fyp-stuff/sequenceFiles/sequenceFiles \
        --checkpoint checkpoints/hrnet_best.pth \
        --output /scratch/fyp-stuff/3dpw_hrnet.npz \
        --sample_every 3
"""

import torch
import numpy as np
import pickle
import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.hrnet import HRNet

# SMPL 24 joints -> COCO 17
SMPL_TO_COCO = [
    15, 15, 15, 15, 15,  # 0-4: face -> head
    16,  # 5: left_shoulder
    17,  # 6: right_shoulder
    18,  # 7: left_elbow
    19,  # 8: right_elbow
    20,  # 9: left_wrist
    21,  # 10: right_wrist
    1,   # 11: left_hip
    2,   # 12: right_hip
    4,   # 13: left_knee
    5,   # 14: right_knee
    7,   # 15: left_ankle
    8,   # 16: right_ankle
]

transform = T.Compose([
    T.Resize((384, 288)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_keypoints_hrnet(model, img_path, device):
    """Run HRNet on an image and return 17 keypoints in pixel coords."""
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        heatmaps = model(inp)[0].cpu().numpy()  # (17, 96, 72)
    keypoints = []
    for i in range(17):
        hm = heatmaps[i]
        idx = np.unravel_index(np.argmax(hm), hm.shape)
        y, x = idx
        x = x * orig_w / 72
        y = y * orig_h / 96
        keypoints.append([x, y])
    return np.array(keypoints, dtype=np.float32)  # (17, 2)

def generate(image_dir, seq_dir, checkpoint, output_path, sample_every=3, splits=['train', 'validation']):
    image_dir = Path(image_dir)
    seq_dir = Path(seq_dir)

    # load HRNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = HRNet(num_keypoints=17).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    model.eval()
    print('HRNet loaded')

    all_kps2d = []
    all_kps3d = []

    for split in splits:
        split_dir = seq_dir / split
        if not split_dir.exists():
            print(f"Skipping {split} — not found")
            continue

        pkl_files = list(split_dir.glob('*.pkl'))
        print(f"\nProcessing {split}: {len(pkl_files)} sequences")

        for pkl_path in pkl_files:
            seq_name = pkl_path.stem
            img_folder = image_dir / seq_name

            if not img_folder.exists():
                print(f"  Skipping {seq_name} — image folder not found")
                continue

            # load sequence
            with open(pkl_path, 'rb') as f:
                seq = pickle.load(f, encoding='latin1')

            # get sorted image files
            img_files = sorted(img_folder.glob('*.jpg'))
            if not img_files:
                img_files = sorted(img_folder.glob('*.png'))
            if not img_files:
                continue

            n_frames = len(img_files)
            num_actors = len(seq['jointPositions'])

            # sample frames
            frame_indices = list(range(0, n_frames, sample_every))
            print(f"  {seq_name}: {n_frames} frames, sampling {len(frame_indices)}")

            for frame_idx in frame_indices:
                if frame_idx >= len(img_files):
                    continue

                img_path = img_files[frame_idx]

                # get 2D keypoints from HRNet
                kps2d = extract_keypoints_hrnet(model, img_path, device)  # (17, 2)

                # get 3D ground truth for each actor
                for actor_idx in range(num_actors):
                    joint_positions = seq['jointPositions'][actor_idx]
                    if frame_idx >= len(joint_positions):
                        continue

                    kps3d_full = joint_positions[frame_idx].reshape(24, 3)
                    kps3d = kps3d_full[SMPL_TO_COCO, :]  # (17, 3)

                    all_kps2d.append(kps2d)
                    all_kps3d.append(kps3d)

    if not all_kps2d:
        print("No data collected!")
        return

    all_kps2d = np.array(all_kps2d, dtype=np.float32)
    all_kps3d = np.array(all_kps3d, dtype=np.float32)

    print(f"\nTotal samples: {len(all_kps2d)}")
    print(f"2D shape: {all_kps2d.shape}")
    print(f"3D shape: {all_kps3d.shape}")

    np.savez(output_path, keypoints2d=all_kps2d, keypoints3d=all_kps3d)
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--seq_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--sample_every', type=int, default=3)
    args = parser.parse_args()

    generate(
        args.image_dir,
        args.seq_dir,
        args.checkpoint,
        args.output,
        args.sample_every
    )