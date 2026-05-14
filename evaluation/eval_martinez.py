"""
eval_martinez.py

Evaluates all MartinezNet models in a folder using MPJPE metric.
Produces a comparison table and bar chart.

Usage:
    PYTHONPATH=. python3 evaluation/eval_martinez.py \
        --data /user/HS402/yv00051/Downloads/3dpw_hrnet_full.npz \
        --checkpoint-dir checkpoints/martinez-to-test \
        --output evaluation/martinez_results
"""

import torch
import numpy as np
import os
import sys
import argparse
import json
import glob
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from data.dataset import PoseDataset
from models.lifting_network import MartinezNet

COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

def mpjpe_per_joint(pred, target):
    return torch.norm(pred - target, dim=-1)  # (B, 17)

def evaluate_model(checkpoint, val_loader, device):
    model = MartinezNet(num_joints_in=17, num_joints_out=17).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    model.eval()

    all_per_joint = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            per_joint = mpjpe_per_joint(outputs, targets)
            all_per_joint.append(per_joint.cpu().numpy())

    all_per_joint  = np.concatenate(all_per_joint, axis=0)  # (N, 17)
    mean_per_joint = all_per_joint.mean(axis=0)              # (17,)
    mean_mpjpe     = mean_per_joint.mean()
    return mean_per_joint, float(mean_mpjpe)

def print_comparison_table(names, all_per_joint, all_mean_mpjpe):
    col_w = max(12, max(len(n) for n in names) + 2)
    kp_w  = 22

    print(f'\n{"="*70}')
    print(f'COMPARISON TABLE — MPJPE (normalised units)')
    print(f'{"="*70}')

    header = f'{"Keypoint":<{kp_w}}'
    for name in names:
        header += f'{name:>{col_w}}'
    print(header)
    print('-' * (kp_w + col_w * len(names)))

    for j in range(17):
        row = f'{COCO_KEYPOINTS[j]:<{kp_w}}'
        for pj in all_per_joint:
            row += f'{pj[j]:>{col_w}.4f}'
        print(row)

    print('-' * (kp_w + col_w * len(names)))
    mean_row = f'{"MEAN MPJPE":<{kp_w}}'
    for m in all_mean_mpjpe:
        mean_row += f'{m:>{col_w}.4f}'
    print(mean_row)
    print(f'{"="*70}')

    best_idx = int(np.argmin(all_mean_mpjpe))
    print(f'\nBest model: {names[best_idx]} (MPJPE={all_mean_mpjpe[best_idx]:.4f})')

def plot_results(names, all_per_joint, all_mean_mpjpe, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # ── Bar chart: mean MPJPE per model ─────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, len(names)*2), 5))
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52'][:len(names)]
    bars = ax.bar(names, all_mean_mpjpe, color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Mean MPJPE (normalised)', fontsize=11)
    ax.set_title('MartinezNet Model Comparison — Mean MPJPE', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(all_mean_mpjpe) * 1.25)
    for bar, val in zip(bars, all_mean_mpjpe):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = os.path.join(output_dir, 'martinez_mean_mpjpe.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

    # ── Per-joint grouped bar chart ──────────────────────────────
    x = np.arange(17)
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(18, 6))
    for i, (name, pj) in enumerate(zip(names, all_per_joint)):
        offset = (i - len(names)/2 + 0.5) * width
        ax.bar(x + offset, pj, width, label=name,
               color=colors[i % len(colors)], edgecolor='white', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(COCO_KEYPOINTS, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('MPJPE (normalised)', fontsize=11)
    ax.set_title('MartinezNet Per-Joint MPJPE Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = os.path.join(output_dir, 'martinez_per_joint_mpjpe.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',           type=str, required=True,
                        help='Path to 3dpw_hrnet_full.npz')
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='Folder containing .pth model files')
    parser.add_argument('--output',         type=str, default='evaluation/martinez_results',
                        help='Output folder for charts and JSON')
    parser.add_argument('--test-split',     type=float, default=0.1)
    parser.add_argument('--batch-size',     type=int,   default=64)
    args = parser.parse_args()

    # find all .pth files in checkpoint dir
    checkpoints = sorted(glob.glob(os.path.join(args.checkpoint_dir, '*.pth')))
    if not checkpoints:
        print(f'No .pth files found in {args.checkpoint_dir}')
        sys.exit(1)

    names = [os.path.splitext(os.path.basename(c))[0] for c in checkpoints]
    print(f'Found {len(checkpoints)} models:')
    for name, ckpt in zip(names, checkpoints):
        print(f'  {name}: {ckpt}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')

    # load dataset
    print(f'Loading dataset: {args.data}')
    dataset   = PoseDataset(args.data)
    test_size = int(len(dataset) * args.test_split)
    train_size = len(dataset) - test_size
    generator  = torch.Generator().manual_seed(42)
    _, test_set = random_split(dataset, [train_size, test_size], generator=generator)
    val_loader  = DataLoader(test_set, batch_size=args.batch_size, drop_last=False)
    print(f'Test samples: {len(test_set)}')

    # evaluate
    all_per_joint  = []
    all_mean_mpjpe = []

    for i, (checkpoint, name) in enumerate(zip(checkpoints, names)):
        print(f'\n[{i+1}/{len(checkpoints)}] {name}')
        pj, mean = evaluate_model(checkpoint, val_loader, device)
        all_per_joint.append(pj)
        all_mean_mpjpe.append(mean)
        print(f'  Mean MPJPE: {mean:.4f}')

    # print table
    print_comparison_table(names, all_per_joint, all_mean_mpjpe)

    # save charts
    plot_results(names, all_per_joint, all_mean_mpjpe, args.output)

    # save JSON
    results = {
        'models': [
            {
                'name': name,
                'checkpoint': ckpt,
                'mean_mpjpe': mean,
                'per_joint_mpjpe': {COCO_KEYPOINTS[j]: float(pj[j]) for j in range(17)}
            }
            for name, ckpt, mean, pj in zip(names, checkpoints, all_mean_mpjpe, all_per_joint)
        ]
    }
    json_path = os.path.join(args.output, 'martinez_comparison.json')
    os.makedirs(args.output, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {json_path}')

if __name__ == '__main__':
    main()