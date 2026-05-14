"""
eval_hrnet.py

Evaluates all HRNet models in a folder on COCO val2017 using PCK metric.
Produces a comparison table and bar charts.

Usage:
    PYTHONPATH=. python3 evaluation/eval_hrnet.py \
        --images /scratch/fypstuffs/val2017 \
        --annotations /scratch/fypstuffs/annotations_trainval2017/annotations/person_keypoints_val2017.json \
        --checkpoint-dir checkpoints/hrnet-models-to-test \
        --output evaluation/hrnet_results \
        --max-images 100
"""

import torch
import numpy as np
import json
import os
import sys
import argparse
import glob
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.hrnet import HRNet

NUM_KEYPOINTS = 17
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

transform = T.Compose([
    T.Resize((384, 288)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_keypoints(heatmaps, orig_w, orig_h):
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
    visible = gt_vis > 0
    ls, lh = gt[5], gt[11]
    rs, rh = gt[6], gt[12]
    torso_size = (np.linalg.norm(ls - lh) + np.linalg.norm(rs - rh)) / 2
    if torso_size < 1e-6:
        return np.zeros(NUM_KEYPOINTS), np.zeros(NUM_KEYPOINTS)
    dist_threshold = threshold * torso_size
    correct = np.zeros(NUM_KEYPOINTS)
    total   = np.zeros(NUM_KEYPOINTS)
    for j in range(NUM_KEYPOINTS):
        if not visible[j]:
            continue
        total[j] = 1
        if np.linalg.norm(pred[j] - gt[j]) < dist_threshold:
            correct[j] = 1
    return correct, total

def evaluate_model(checkpoint, img_ids, id_to_file, img_to_anns, images_dir, threshold, device, max_images=None):
    state_dict = torch.load(checkpoint, map_location=device, weights_only=False)
    width = state_dict['final_layer.weight'].shape[1]
    model = HRNet(num_keypoints=17, width=width).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    ids = img_ids[:max_images] if max_images else img_ids
    total_correct = np.zeros(NUM_KEYPOINTS)
    total_count   = np.zeros(NUM_KEYPOINTS)
    skipped = 0

    for img_id in tqdm(ids, desc=f'  {os.path.basename(checkpoint)}', leave=False):
        file_name = id_to_file[img_id]
        img_path  = os.path.join(images_dir, file_name)
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
            kps    = np.array(ann['keypoints']).reshape(NUM_KEYPOINTS, 3)
            gt_kps = kps[:, :2].astype(np.float32)
            gt_vis = kps[:, 2]
            c, t   = compute_pck(pred_kps, gt_kps, gt_vis, threshold)
            total_correct += c
            total_count   += t

    per_joint = np.zeros(NUM_KEYPOINTS)
    for j in range(NUM_KEYPOINTS):
        if total_count[j] > 0:
            per_joint[j] = total_correct[j] / total_count[j] * 100
    mean_pck = per_joint[total_count > 0].mean() if total_count.sum() > 0 else 0.0
    return per_joint, float(mean_pck), skipped

def print_comparison_table(names, all_per_joint, all_mean_pck, threshold):
    col_w = max(12, max(len(n) for n in names) + 2)
    kp_w  = 22

    print(f'\n{"="*70}')
    print(f'COMPARISON TABLE — PCK@{threshold}')
    print(f'{"="*70}')

    header = f'{"Keypoint":<{kp_w}}'
    for name in names:
        header += f'{name:>{col_w}}'
    print(header)
    print('-' * (kp_w + col_w * len(names)))

    for j in range(NUM_KEYPOINTS):
        row = f'{COCO_KEYPOINTS[j]:<{kp_w}}'
        for pj in all_per_joint:
            row += f'{pj[j]:>{col_w}.1f}%'
        print(row)

    print('-' * (kp_w + col_w * len(names)))
    mean_row = f'{"MEAN PCK":<{kp_w}}'
    for m in all_mean_pck:
        mean_row += f'{m:>{col_w}.1f}%'
    print(mean_row)
    print(f'{"="*70}')

    best_idx = int(np.argmax(all_mean_pck))
    print(f'\nBest model: {names[best_idx]} ({all_mean_pck[best_idx]:.1f}%)')

def plot_results(names, all_per_joint, all_mean_pck, output_dir, threshold):
    os.makedirs(output_dir, exist_ok=True)
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2'][:len(names)]

    # Mean PCK bar chart
    fig, ax = plt.subplots(figsize=(max(6, len(names)*2), 5))
    bars = ax.bar(names, all_mean_pck, color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    ax.set_ylabel(f'Mean PCK@{threshold} (%)', fontsize=11)
    ax.set_title(f'HRNet Model Comparison — Mean PCK@{threshold}', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, all_mean_pck):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = os.path.join(output_dir, 'hrnet_mean_pck.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

    # Per-joint grouped bar chart
    x     = np.arange(NUM_KEYPOINTS)
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(18, 6))
    for i, (name, pj) in enumerate(zip(names, all_per_joint)):
        offset = (i - len(names)/2 + 0.5) * width
        ax.bar(x + offset, pj, width, label=name,
               color=colors[i % len(colors)], edgecolor='white', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(COCO_KEYPOINTS, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(f'PCK@{threshold} (%)', fontsize=11)
    ax.set_title(f'HRNet Per-Joint PCK@{threshold} Comparison', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = os.path.join(output_dir, 'hrnet_per_joint_pck.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images',         type=str,   required=True)
    parser.add_argument('--annotations',    type=str,   required=True)
    parser.add_argument('--checkpoint-dir', type=str,   required=True)
    parser.add_argument('--output',         type=str,   default='evaluation/hrnet_results')
    parser.add_argument('--threshold',      type=float, default=0.2)
    parser.add_argument('--max-images',     type=int,   default=None)
    args = parser.parse_args()

    # find all .pth files
    checkpoints = sorted(glob.glob(os.path.join(args.checkpoint_dir, '*.pth')))
    if not checkpoints:
        print(f'No .pth files found in {args.checkpoint_dir}')
        sys.exit(1)

    names = [os.path.splitext(os.path.basename(c))[0] for c in checkpoints]
    print(f'Found {len(checkpoints)} models:')
    for name, ckpt in zip(names, checkpoints):
        print(f'  {name}: {ckpt}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # load annotations once
    print(f'Loading annotations...')
    with open(args.annotations) as f:
        coco = json.load(f)

    id_to_file  = {img['id']: img['file_name'] for img in coco['images']}
    img_to_anns = {}
    for ann in coco['annotations']:
        if ann['num_keypoints'] == 0:
            continue
        img_to_anns.setdefault(ann['image_id'], []).append(ann)

    img_ids = list(img_to_anns.keys())
    print(f'Images with annotations: {len(img_ids)}')
    if args.max_images:
        print(f'Limiting to {args.max_images} images')

    # evaluate
    all_per_joint = []
    all_mean_pck  = []

    for i, (checkpoint, name) in enumerate(zip(checkpoints, names)):
        print(f'\n[{i+1}/{len(checkpoints)}] {name}')
        pj, mean, skipped = evaluate_model(
            checkpoint, img_ids, id_to_file, img_to_anns,
            args.images, args.threshold, device, args.max_images
        )
        all_per_joint.append(pj)
        all_mean_pck.append(mean)
        print(f'  Mean PCK@{args.threshold}: {mean:.1f}% | Skipped: {skipped}')

    # print table
    print_comparison_table(names, all_per_joint, all_mean_pck, args.threshold)

    # save charts
    plot_results(names, all_per_joint, all_mean_pck, args.output, args.threshold)

    # save JSON
    results = {
        'threshold': args.threshold,
        'models': [
            {
                'name': name,
                'checkpoint': ckpt,
                'mean_pck': mean,
                'per_joint_pck': {COCO_KEYPOINTS[j]: float(pj[j]) for j in range(NUM_KEYPOINTS)}
            }
            for name, ckpt, mean, pj in zip(names, checkpoints, all_mean_pck, all_per_joint)
        ]
    }
    json_path = os.path.join(args.output, 'hrnet_comparison.json')
    os.makedirs(args.output, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {json_path}')

if __name__ == '__main__':
    main()