"""
eval_hrnet.py

Evaluates HRNet 2D pose estimation on COCO val2017 using PCK metric.

PCK@0.2: keypoint is correct if distance < 0.2 * torso_size

Usage:
    PYTHONPATH=. python3 evaluation/eval_hrnet.py \
        --images /scratch/fyp-stuff/val2017 \
        --annotations /scratch/fyp-stuff/annotations_trainval2017/annotations/person_keypoints_val2017.json \
        --checkpoint checkpoints/hrnet_best.pth
"""

import torch
import numpy as np
import json
import os
import sys
import argparse
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.hrnet import HRNet

# COCO keypoint indices
NUM_KEYPOINTS = 17

transform = T.Compose([
    T.Resize((384, 288)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_keypoints(heatmaps, orig_w, orig_h):
    """Extract keypoint coordinates from heatmaps."""
    keypoints = []
    for i in range(NUM_KEYPOINTS):
        hm = heatmaps[i]
        idx = np.unravel_index(np.argmax(hm), hm.shape)
        y, x = idx
        x = x * orig_w / 72
        y = y * orig_h / 96
        keypoints.append([x, y])
    return np.array(keypoints)

def compute_pck(pred, gt, gt_vis, threshold=0.2):
    """
    Compute PCK for a single person.
    threshold: fraction of torso size
    Returns (correct_per_joint, total_per_joint)
    """
    # torso size = average of left/right shoulder-to-hip distances
    # left shoulder=5, left hip=11, right shoulder=6, right hip=12
    visible_joints = gt_vis > 0

    ls, lh = gt[5], gt[11]
    rs, rh = gt[6], gt[12]

    torso_size = (np.linalg.norm(ls - lh) + np.linalg.norm(rs - rh)) / 2
    if torso_size < 1e-6:
        return np.zeros(NUM_KEYPOINTS), np.zeros(NUM_KEYPOINTS)

    dist_threshold = threshold * torso_size

    correct = np.zeros(NUM_KEYPOINTS)
    total = np.zeros(NUM_KEYPOINTS)

    for j in range(NUM_KEYPOINTS):
        if not visible_joints[j]:
            continue
        dist = np.linalg.norm(pred[j] - gt[j])
        total[j] = 1
        if dist < dist_threshold:
            correct[j] = 1

    return correct, total

def evaluate(args):
    # ── Load model ─────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = HRNet(num_keypoints=17).to(device)
    model.load_state_dict(torch.load(
        args.checkpoint, map_location=device, weights_only=False))
    model.eval()
    print('HRNet loaded')

    # ── Load annotations ────────────────────────────────────────
    print(f'Loading annotations from {args.annotations}...')
    with open(args.annotations) as f:
        coco = json.load(f)

    # build image id -> file name map
    id_to_file = {img['id']: img['file_name'] for img in coco['images']}

    # group annotations by image
    img_to_anns = {}
    for ann in coco['annotations']:
        if ann['num_keypoints'] == 0:
            continue
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    print(f'Found {len(img_to_anns)} images with keypoint annotations')

    # ── Evaluate ────────────────────────────────────────────────
    total_correct = np.zeros(NUM_KEYPOINTS)
    total_count   = np.zeros(NUM_KEYPOINTS)

    COCO_KEYPOINTS = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]

    skipped = 0
    processed = 0

    img_ids = list(img_to_anns.keys())
    if args.max_images:
        img_ids = img_ids[:args.max_images]

    for img_id in tqdm(img_ids, desc='Evaluating'):
        file_name = id_to_file[img_id]
        img_path = os.path.join(args.images, file_name)

        if not os.path.exists(img_path):
            skipped += 1
            continue

        try:
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
        except Exception:
            skipped += 1
            continue

        inp = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            heatmaps = model(inp)[0].cpu().numpy()

        pred_kps = extract_keypoints(heatmaps, orig_w, orig_h)

        for ann in img_to_anns[img_id]:
            kps = np.array(ann['keypoints']).reshape(NUM_KEYPOINTS, 3)
            gt_kps = kps[:, :2].astype(np.float32)
            gt_vis = kps[:, 2]  # 0=invisible, 1=occluded, 2=visible

            correct, total = compute_pck(
                pred_kps, gt_kps, gt_vis, threshold=args.threshold)
            total_correct += correct
            total_count   += total

        processed += 1

    # ── Results ─────────────────────────────────────────────────
    print(f'\n{"="*50}')
    print(f'RESULTS — PCK@{args.threshold}')
    print(f'{"="*50}')
    print(f'Images processed: {processed} | Skipped: {skipped}')
    print()

    per_joint_pck = np.zeros(NUM_KEYPOINTS)
    for j in range(NUM_KEYPOINTS):
        if total_count[j] > 0:
            per_joint_pck[j] = total_correct[j] / total_count[j] * 100
        print(f'  {COCO_KEYPOINTS[j]:20s}: {per_joint_pck[j]:.1f}%  ({int(total_correct[j])}/{int(total_count[j])})')

    mean_pck = per_joint_pck[total_count > 0].mean()
    print()
    print(f'  Mean PCK@{args.threshold}: {mean_pck:.1f}%')
    print(f'{"="*50}')

    # save results
    results = {
        'threshold': args.threshold,
        'mean_pck': float(mean_pck),
        'per_joint_pck': {COCO_KEYPOINTS[j]: float(per_joint_pck[j]) for j in range(NUM_KEYPOINTS)},
        'processed': processed,
        'skipped': skipped
    }

    out_path = os.path.join(BASE_DIR, 'evaluation', 'hrnet_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images',      type=str, required=True, help='Path to val2017 images')
    parser.add_argument('--annotations', type=str, required=True, help='Path to person_keypoints_val2017.json')
    parser.add_argument('--checkpoint',  type=str, required=True, help='Path to hrnet_best.pth')
    parser.add_argument('--threshold',   type=float, default=0.2,  help='PCK threshold (default 0.2)')
    parser.add_argument('--max-images',  type=int,   default=None,  help='Limit number of images (for testing)')
    args = parser.parse_args()

    evaluate(args)