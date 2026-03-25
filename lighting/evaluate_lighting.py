"""
evaluate_lighting.py

Evaluates a trained lighting estimator on the val split.
Reports mean angular error in degrees and plots a histogram.

Usage:
    python evaluate_lighting.py --data /path/to/data256 --checkpoint checkpoints/lighting_best.pth
"""

import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from lighting_dataset import DPRDataset, get_transforms
from lighting_model import LightingEstimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch',      type=int, default=64)
    parser.add_argument('--workers',    type=int, default=4)
    args = parser.parse_args()

    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps')  if torch.backends.mps.is_available() else
        torch.device('cpu')
    )

    # Load model
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = LightingEstimator(pretrained=False).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val err: {ckpt['val_err_deg']:.2f}°)")

    # Dataset
    val_ds     = DPRDataset(args.data, split='val', transform=get_transforms('val'))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    all_errors = []
    with torch.no_grad():
        for imgs, lights in val_loader:
            imgs, lights = imgs.to(device), lights.to(device)
            pred    = model(imgs)
            cos_sim = (pred * lights).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
            angles  = torch.acos(cos_sim) * (180.0 / math.pi)
            all_errors.extend(angles.cpu().numpy().tolist())

    errors = np.array(all_errors)
    print(f"\n── Evaluation Results ──────────────────────")
    print(f"  Samples:         {len(errors)}")
    print(f"  Mean error:      {errors.mean():.2f}°")
    print(f"  Median error:    {np.median(errors):.2f}°")
    print(f"  Std:             {errors.std():.2f}°")
    print(f"  % under 15°:     {(errors < 15).mean()*100:.1f}%")
    print(f"  % under 30°:     {(errors < 30).mean()*100:.1f}%")
    print(f"  % under 45°:     {(errors < 45).mean()*100:.1f}%")

    # Save errors for dissertation
    np.save('lighting_eval_errors.npy', errors)
    print(f"\nErrors saved to lighting_eval_errors.npy")


if __name__ == '__main__':
    main()
