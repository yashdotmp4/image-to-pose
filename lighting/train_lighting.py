"""
train_lighting.py

Trains a ResNet-18 lighting estimator on the DPR dataset.

Usage:
    python train_lighting.py --data /path/to/data256 --epochs 30 --batch 64

Outputs:
    checkpoints/lighting_best.pth   — best val checkpoint
    checkpoints/lighting_last.pth   — last epoch checkpoint
"""

import argparse
import os
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lighting_dataset import DPRDataset, get_transforms
from lighting_model import LightingEstimator


# ── Loss ───────────────────────────────────────────────────────────────────────

def angular_loss(pred, target):
    """
    Mean angular error in degrees between predicted and target unit vectors.
    Used as the primary loss — minimising angle directly.
    """
    # Clamp dot product to [-1, 1] to avoid NaN in acos
    cos_sim = (pred * target).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
    angle   = torch.acos(cos_sim)          # radians
    return angle.mean()


def angular_error_degrees(pred, target):
    """Mean angular error in degrees — for logging."""
    cos_sim = (pred * target).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
    angle   = torch.acos(cos_sim) * (180.0 / math.pi)
    return angle.mean().item()


# ── Train / Val loops ──────────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, device):
    model.train()
    total_loss = 0.0
    total_err  = 0.0
    for imgs, lights in loader:
        imgs, lights = imgs.to(device), lights.to(device)
        optimiser.zero_grad()
        pred = model(imgs)
        loss = angular_loss(pred, lights)
        loss.backward()
        optimiser.step()
        total_loss += loss.item()
        total_err  += angular_error_degrees(pred.detach(), lights)
    n = len(loader)
    return total_loss / n, total_err / n


def val_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_err  = 0.0
    with torch.no_grad():
        for imgs, lights in loader:
            imgs, lights = imgs.to(device), lights.to(device)
            pred = model(imgs)
            total_loss += angular_loss(pred, lights).item()
            total_err  += angular_error_degrees(pred, lights)
    n = len(loader)
    return total_loss / n, total_err / n


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     type=str,   required=True,  help='Path to data256 folder')
    parser.add_argument('--epochs',   type=int,   default=30)
    parser.add_argument('--batch',    type=int,   default=64)
    parser.add_argument('--lr',       type=float, default=1e-3)
    parser.add_argument('--workers',  type=int,   default=4)
    parser.add_argument('--out',      type=str,   default='checkpoints')
    parser.add_argument('--no_pretrain', action='store_true')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device('cuda')  if torch.cuda.is_available()  else
        torch.device('mps')   if torch.backends.mps.is_available() else
        torch.device('cpu')
    )
    print(f"Using device: {device}")

    # Datasets
    train_ds = DPRDataset(args.data, split='train', transform=get_transforms('train'))
    val_ds   = DPRDataset(args.data, split='val',   transform=get_transforms('val'))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # Model
    model = LightingEstimator(pretrained=not args.no_pretrain).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimiser + scheduler
    optimiser = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=1e-5)

    best_val_err = float('inf')

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Err°':>12} {'Val Loss':>10} {'Val Err°':>10} {'LR':>10} {'Time':>8}")    
    print('-' * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_err = train_epoch(model, train_loader, optimiser, device)
        val_loss,   val_err   = val_epoch(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        lr = scheduler.get_last_lr()[0]
        print(f"{epoch:>6} {train_loss:>12.4f} {train_err:>11.2f}° {val_loss:>10.4f} {val_err:>9.2f}° {lr:>10.6f} {elapsed:>7.1f}s")

        # Save best
        if val_err < best_val_err:
            best_val_err = val_err
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_err_deg': val_err,
            }, out_dir / 'lighting_best.pth')
            print(f"  ↑ New best: {val_err:.2f}°")

    # Save last
    torch.save({
        'epoch': args.epochs,
        'model_state': model.state_dict(),
        'val_err_deg': val_err,
    }, out_dir / 'lighting_last.pth')

    print(f"\nDone. Best val error: {best_val_err:.2f}°")
    print(f"Checkpoint saved to {out_dir / 'lighting_best.pth'}")


if __name__ == '__main__':
    main()
