"""
predict_lighting.py

Run lighting estimation on a single image and visualise the predicted
light direction as an arrow overlaid on the image.

Usage:
    py -3.11 lighting/predict_lighting.py --image path/to/image.jpg --checkpoint checkpoints/lighting_best.pth
"""

import argparse
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

import torch
import torchvision.transforms as T

from lighting_model import LightingEstimator


def load_image(path):
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0), img


def draw_light_arrow(img, light_dir, size=300):
    """
    Draw the predicted light direction as an arrow on a resized copy of the image.
    The arrow shows where the light is coming FROM.
    """
    img = img.resize((size, size))
    draw = ImageDraw.Draw(img)

    cx, cy = size // 2, size // 2
    scale  = size * 0.35

    # light_dir is where light points TO — arrow shows where it comes FROM
    lx = -float(light_dir[0])
    ly = -float(light_dir[1])   # flip Y for image coords

    # Arrow tip (where light comes from)
    tx = int(cx + lx * scale)
    ty = int(cy + ly * scale)

    # Arrow base (centre of image)
    bx, by = cx, cy

    # Draw line
    draw.line([(tx, ty), (bx, by)], fill=(255, 220, 0), width=4)

    # Draw arrowhead
    angle = math.atan2(by - ty, bx - tx)
    ah = 15
    for da in [0.4, -0.4]:
        ax = int(tx + ah * math.cos(angle + da + math.pi))
        ay = int(ty + ah * math.sin(angle + da + math.pi))
        draw.line([(tx, ty), (ax, ay)], fill=(255, 220, 0), width=4)

    # Draw sun circle at arrow tip
    r = 10
    draw.ellipse([(tx-r, ty-r), (tx+r, ty+r)], fill=(255, 220, 0))

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',      type=str, required=True,  help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True,  help='Path to model checkpoint')
    parser.add_argument('--out',        type=str, default=None,   help='Output image path (optional)')
    args = parser.parse_args()

    device = (
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('mps')  if torch.backends.mps.is_available() else
        torch.device('cpu')
    )

    # Load model
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = LightingEstimator(pretrained=False).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Load and run image
    tensor, orig_img = load_image(args.image)
    tensor = tensor.to(device)

    with torch.no_grad():
        light = model(tensor)[0].cpu().numpy()

    print(f"\nPredicted light direction:")
    print(f"  x: {light[0]:+.4f}  ({'right' if light[0] > 0 else 'left'})")
    print(f"  y: {light[1]:+.4f}  ({'up' if light[1] > 0 else 'down'})")
    print(f"  z: {light[2]:+.4f}  ({'toward camera' if light[2] < 0 else 'away from camera'})")

    # Elevation and azimuth in degrees
    azimuth   = math.degrees(math.atan2(light[0], light[2]))
    elevation = math.degrees(math.asin(np.clip(light[1], -1, 1)))
    print(f"\n  Azimuth:   {azimuth:.1f}°")
    print(f"  Elevation: {elevation:.1f}°")

    # Save annotated image
    out_path = args.out or str(Path(args.image).stem) + '_light.jpg'
    annotated = draw_light_arrow(orig_img, light)
    annotated.save(out_path)
    print(f"\nAnnotated image saved to: {out_path}")


if __name__ == '__main__':
    main()